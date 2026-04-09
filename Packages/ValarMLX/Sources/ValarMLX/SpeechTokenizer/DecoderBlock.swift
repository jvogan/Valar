import Foundation
@preconcurrency import MLX
import MLXNN

// MARK: - Configuration

/// Configuration for the convolutional decoder blocks in the speech tokenizer.
///
/// Default values match the Qwen3-TTS speech tokenizer decoder:
/// `decoder_dim=1536`, `upsample_rates=[8, 5, 4, 3]`, producing 4 decoder
/// blocks that progressively halve channels and upsample time.
///
/// Reference: `config.py` class `Qwen3TTSTokenizerDecoderConfig`.
struct DecoderBlockConfig: Sendable {
    let decoderDim: Int
    let upsampleRates: [Int]

    static let `default` = DecoderBlockConfig(
        decoderDim: 1536,
        upsampleRates: [8, 5, 4, 3]
    )
}

enum DecoderBlockConfigError: Error, LocalizedError, Equatable, Sendable {
    case invalidLayerIndex(layerIndex: Int, validIndices: Range<Int>)

    var errorDescription: String? {
        switch self {
        case .invalidLayerIndex(let layerIndex, let validIndices):
            return "DecoderBlock layerIndex \(layerIndex) is out of bounds for upsampleRates indices \(validIndices)."
        }
    }
}

// MARK: - DecoderResidualUnit

/// Residual unit for the decoder with SnakeBeta activations.
///
/// Architecture: SnakeBeta → CausalConv1d(k=7, dilation=d) →
/// SnakeBeta → CausalConv1d(k=1) → residual add.
///
/// The dilated convolution captures temporal context at different scales.
/// Standard dilations in each DecoderBlock are 1, 3, and 9.
///
/// Weight keys match the Python structure:
/// `act1.{alpha,beta}`, `conv1.conv.{weight,bias}`,
/// `act2.{alpha,beta}`, `conv2.conv.{weight,bias}`.
///
/// Reference: `speech_tokenizer.py` class `DecoderResidualUnit`.
final class DecoderResidualUnit: Module {
    @ModuleInfo var act1: SnakeBeta
    @ModuleInfo var conv1: CausalConv1d
    @ModuleInfo var act2: SnakeBeta
    @ModuleInfo var conv2: CausalConv1d

    init(dim: Int, dilation: Int = 1) {
        _act1.wrappedValue = SnakeBeta(channels: dim)
        _conv1.wrappedValue = CausalConv1d(
            inChannels: dim, outChannels: dim,
            kernelSize: 7, dilation: dilation
        )
        _act2.wrappedValue = SnakeBeta(channels: dim)
        _conv2.wrappedValue = CausalConv1d(
            inChannels: dim, outChannels: dim,
            kernelSize: 1
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = x
        var h = act1(x)
        h = conv1(h)
        h = act2(h)
        h = conv2(h)
        return h + residual
    }
}

// MARK: - DecoderBlockUpsample

/// Causal transposed convolution wrapper for upsampling in a decoder block.
///
/// Uses `kernel_size = 2 * upsample_rate` and trims `kernel_size - stride`
/// samples from the right for causal behavior.
///
/// Weight key: `conv.{weight,bias}` — matches the Python `DecoderBlockUpsample`
/// which wraps a single `nn.ConvTranspose1d`.
///
/// Reference: `speech_tokenizer.py` class `DecoderBlockUpsample`.
final class DecoderBlockUpsample: Module {
    @ModuleInfo var conv: TransposeConv1dWeight
    let trimRight: Int

    init(inDim: Int, outDim: Int, upsampleRate: Int) {
        let kernelSize = 2 * upsampleRate
        self.trimRight = kernelSize - upsampleRate
        _conv.wrappedValue = TransposeConv1dWeight(
            inChannels: inDim,
            outChannels: outDim,
            kernelSize: kernelSize,
            stride: upsampleRate
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: [batch, channels, time] (NCL)
        var h = x.transposed(0, 2, 1)   // NCL -> NLC
        h = conv(h)                       // NLC (upsampled)
        h = h.transposed(0, 2, 1)        // NLC -> NCL
        if trimRight > 0 {
            let t = h.dim(2)
            h = h[0..., 0..., 0 ..< (t - trimRight)]
        }
        return h
    }
}

// MARK: - DecoderBlock

/// Single decoder block: SnakeBeta activation → causal upsample → 3 dilated
/// residual units (dilation 1, 3, 9).
///
/// Each block halves the channel dimension and upsamples the temporal
/// dimension by the configured rate. The `@ModuleInfo` keys use `"block"`
/// with integer indices to match the Python `self.block` `ModuleList`:
///
/// - `block.0`: SnakeBeta (alpha, beta)
/// - `block.1`: DecoderBlockUpsample (conv.weight, conv.bias)
/// - `block.2`: DecoderResidualUnit (act1, conv1, act2, conv2) — dilation 1
/// - `block.3`: DecoderResidualUnit — dilation 3
/// - `block.4`: DecoderResidualUnit — dilation 9
///
/// Reference: `speech_tokenizer.py` class `DecoderBlock`.
final class DecoderBlock: Module {
    @ModuleInfo var block: [Module]

    init(inDim: Int, outDim: Int, upsampleRate: Int) {
        _block.wrappedValue = [
            SnakeBeta(channels: inDim),
            DecoderBlockUpsample(inDim: inDim, outDim: outDim, upsampleRate: upsampleRate),
            DecoderResidualUnit(dim: outDim, dilation: 1),
            DecoderResidualUnit(dim: outDim, dilation: 3),
            DecoderResidualUnit(dim: outDim, dilation: 9),
        ]
    }

    /// Convenience initializer from config and layer index.
    ///
    /// Computes `inDim`, `outDim`, and `upsampleRate` from the config's
    /// `decoderDim` and `upsampleRates` array, matching the Python
    /// `DecoderBlock.__init__(config, layer_idx)`.
    convenience init(config: DecoderBlockConfig, layerIndex: Int) throws {
        guard config.upsampleRates.indices.contains(layerIndex) else {
            throw DecoderBlockConfigError.invalidLayerIndex(
                layerIndex: layerIndex,
                validIndices: config.upsampleRates.indices
            )
        }

        let inDim = config.decoderDim / (1 << layerIndex)
        let outDim = config.decoderDim / (1 << (layerIndex + 1))
        let upsampleRate = config.upsampleRates[layerIndex]
        self.init(inDim: inDim, outDim: outDim, upsampleRate: upsampleRate)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for layer in block {
            if let snakeBeta = layer as? SnakeBeta {
                h = snakeBeta(h)
            } else if let upsample = layer as? DecoderBlockUpsample {
                h = upsample(h)
            } else if let residual = layer as? DecoderResidualUnit {
                h = residual(h)
            }
        }
        return h
    }
}
