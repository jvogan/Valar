import Foundation
@preconcurrency import MLX
import MLXNN

// MARK: - Configuration

/// Configuration for the causal convolutional upsampler.
///
/// Default values match the Qwen3-TTS speech tokenizer decoder:
/// 512-dim input (codebook_dim), 1024-dim latent space, 2×2 temporal
/// upsampling via CausalTransposeConv1d + ConvNeXt stages.
///
/// Reference: `config.py` class `Qwen3TTSTokenizerDecoderConfig`.
struct CausalConvUpsamplerConfig: Sendable {
    let inputDim: Int
    let latentDim: Int
    let upsamplingRatios: [Int]
    let convNeXtKernelSize: Int

    static let `default` = CausalConvUpsamplerConfig(
        inputDim: 512,
        latentDim: 1024,
        upsamplingRatios: [2, 2],
        convNeXtKernelSize: 7
    )
}

// MARK: - CausalConv1d

/// Causal 1D convolution with left-only padding.
///
/// Pads `(kernelSize - 1) * dilation` samples on the left so that output
/// at time t depends only on inputs at times <= t. Wraps `MLXNN.Conv1d`
/// (NLC format) with NCL input/output.
///
/// Supports grouped convolutions for depthwise use in ConvNeXt blocks.
///
/// Reference: `speech_tokenizer.py` class `CausalConv1d`.
final class CausalConv1d: Module {
    @ModuleInfo var conv: MLXNN.Conv1d
    let causalPadding: Int

    init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        dilation: Int = 1,
        groups: Int = 1
    ) {
        let effectiveKernel = (kernelSize - 1) * dilation + 1
        self.causalPadding = effectiveKernel - stride
        _conv.wrappedValue = MLXNN.Conv1d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: kernelSize,
            stride: stride,
            padding: 0,
            dilation: dilation,
            groups: groups
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: [batch, channels, time] (NCL)
        var h = x
        if causalPadding > 0 {
            h = padded(h, widths: [IntOrPair(0), IntOrPair(0), IntOrPair((causalPadding, 0))])
        }
        h = h.transposed(0, 2, 1)   // NCL -> NLC
        h = conv(h)
        return h.transposed(0, 2, 1) // NLC -> NCL
    }
}

// MARK: - TransposeConv1dWeight

/// Weight container for transposed 1D convolution.
///
/// Stores weight and bias with key paths matching Python's
/// `nn.ConvTranspose1d` (`weight`, `bias`). Calls the MLX
/// `convTransposed1d` free function internally (NLC format).
final class TransposeConv1dWeight: Module {
    var weight: MLXArray
    var bias: MLXArray
    let stride: Int

    init(inChannels: Int, outChannels: Int, kernelSize: Int, stride: Int) {
        // MLX layout: [outChannels, kernelSize, inChannels]
        self.weight = MLXArray.zeros([outChannels, kernelSize, inChannels])
        self.bias = MLXArray.zeros([outChannels])
        self.stride = stride
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: [batch, time, channels] (NLC)
        var y = convTransposed1d(x, weight, stride: stride, padding: 0)
        y = y + bias
        return y
    }
}

// MARK: - CausalTransposeConv1d

/// Causal transposed 1D convolution for temporal upsampling.
///
/// Applies `convTransposed1d` then trims `kernelSize - stride` samples
/// from the right to ensure causal behavior: output at time t depends
/// only on inputs at times <= t.
///
/// With the default Qwen3-TTS config (`kernelSize == stride == 2`),
/// trim is zero and the layer is a clean 2× temporal upsampler.
///
/// Reference: `speech_tokenizer.py` class `CausalTransposeConv1d`.
final class CausalTransposeConv1d: Module {
    @ModuleInfo var conv: TransposeConv1dWeight
    let trimRight: Int

    init(inChannels: Int, outChannels: Int, kernelSize: Int, stride: Int) {
        self.trimRight = kernelSize - stride
        _conv.wrappedValue = TransposeConv1dWeight(
            inChannels: inChannels,
            outChannels: outChannels,
            kernelSize: kernelSize,
            stride: stride
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

// MARK: - ConvNeXtBlock

/// ConvNeXt block with causal depthwise convolution.
///
/// Architecture: depthwise causal conv → LayerNorm → pointwise expand →
/// GELU → pointwise contract → layer scale (gamma) → residual add.
///
/// Operates in NCL format. The depthwise conv uses causal left-padding
/// so the block preserves the causal property.
///
/// Reference: `speech_tokenizer.py` class `ConvNeXtBlock`.
final class ConvNeXtBlock: Module {
    @ModuleInfo var dwconv: CausalConv1d
    @ModuleInfo var norm: LayerNorm
    @ModuleInfo var pwconv1: Linear
    @ModuleInfo var pwconv2: Linear
    var gamma: MLXArray

    init(dim: Int, kernelSize: Int = 7) {
        _dwconv.wrappedValue = CausalConv1d(
            inChannels: dim, outChannels: dim,
            kernelSize: kernelSize, groups: dim
        )
        _norm.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6)
        _pwconv1.wrappedValue = Linear(dim, 4 * dim)
        _pwconv2.wrappedValue = Linear(4 * dim, dim)
        self.gamma = MLXArray.ones([dim]) * Float(1e-6)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: [batch, channels, time] (NCL)
        let residual = x
        var h = dwconv(x)                // Causal depthwise conv (NCL)
        h = h.transposed(0, 2, 1)        // NCL -> NLC
        h = norm(h)
        h = pwconv1(h)
        h = gelu(h)
        h = pwconv2(h)
        h = gamma * h                     // Layer scale
        h = h.transposed(0, 2, 1)        // NLC -> NCL
        return residual + h
    }
}

// MARK: - UpsampleStage

/// Single temporal upsampling stage: CausalTransposeConv1d followed by
/// a ConvNeXt block.
///
/// The `@ModuleInfo` keys `"0"` and `"1"` match the Python module list
/// structure `self.upsample[i] = [CausalTransposeConv1d, ConvNeXtBlock]`.
final class UpsampleStage: Module {
    @ModuleInfo(key: "0") var transposeConv: CausalTransposeConv1d
    @ModuleInfo(key: "1") var convNext: ConvNeXtBlock

    init(dim: Int, upsampleFactor: Int, convNextKernelSize: Int = 7) {
        _transposeConv.wrappedValue = CausalTransposeConv1d(
            inChannels: dim, outChannels: dim,
            kernelSize: upsampleFactor, stride: upsampleFactor
        )
        _convNext.wrappedValue = ConvNeXtBlock(dim: dim, kernelSize: convNextKernelSize)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        convNext(transposeConv(x))
    }
}

// MARK: - CausalConvUpsampler

/// Initial upsampler for the speech tokenizer decoder.
///
/// Projects input from `inputDim` to `latentDim` channels via a causal
/// convolution, then applies N stages of CausalTransposeConv1d + ConvNeXt
/// for temporal upsampling. All operations are causal: output at time t
/// depends only on inputs at times <= t.
///
/// Default config: `[B, 512, T]` → `[B, 1024, T]` → `[B, 1024, 2T]` →
/// `[B, 1024, 4T]`.
///
/// Reference: `speech_tokenizer.py` class `Qwen3TTSSpeechTokenizerDecoder`
/// (`pre_conv` + `upsample` fields).
final class CausalConvUpsampler: Module {
    @ModuleInfo(key: "pre_conv") var preConv: CausalConv1d
    @ModuleInfo var upsample: [UpsampleStage]
    let config: CausalConvUpsamplerConfig

    init(config: CausalConvUpsamplerConfig = .default) {
        self.config = config
        _preConv.wrappedValue = CausalConv1d(
            inChannels: config.inputDim,
            outChannels: config.latentDim,
            kernelSize: 3
        )
        _upsample.wrappedValue = config.upsamplingRatios.map { factor in
            UpsampleStage(
                dim: config.latentDim,
                upsampleFactor: factor,
                convNextKernelSize: config.convNeXtKernelSize
            )
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: [batch, inputDim, time] (NCL)
        var h = preConv(x)
        for stage in upsample {
            h = stage(h)
        }
        return h
    }
}
