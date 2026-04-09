import Foundation
@preconcurrency import MLX
import MLXNN

// MARK: - Configuration

/// Unified configuration for the speech tokenizer decoder.
///
/// Combines quantizer, convolutional, transformer, and decoder block parameters
/// into a single config struct matching the Python `Qwen3TTSTokenizerDecoderConfig`.
///
/// Reference: `config.py` class `Qwen3TTSTokenizerDecoderConfig`.
struct SpeechTokenizerDecoderConfig: Sendable {
    let codebookDim: Int
    let codebookSize: Int
    let numQuantizers: Int
    let numSemanticQuantizers: Int
    let latentDim: Int
    let decoderDim: Int
    let upsampleRates: [Int]
    let upsamplingRatios: [Int]
    let hiddenSize: Int
    let intermediateSize: Int
    let numHiddenLayers: Int
    let numAttentionHeads: Int
    let numKeyValueHeads: Int
    let headDim: Int
    let rmsNormEps: Float
    let ropeTheta: Float
    let slidingWindow: Int
    let layerScaleInitialScale: Float
    let attentionBias: Bool

    static let `default` = SpeechTokenizerDecoderConfig(
        codebookDim: 512,
        codebookSize: 2048,
        numQuantizers: 16,
        numSemanticQuantizers: 1,
        latentDim: 1024,
        decoderDim: 1536,
        upsampleRates: [8, 5, 4, 3],
        upsamplingRatios: [2, 2],
        hiddenSize: 512,
        intermediateSize: 1024,
        numHiddenLayers: 8,
        numAttentionHeads: 16,
        numKeyValueHeads: 16,
        headDim: 64,
        rmsNormEps: 1e-5,
        ropeTheta: 10000.0,
        slidingWindow: 72,
        layerScaleInitialScale: 0.01,
        attentionBias: false
    )

    /// Total temporal upsampling factor across all stages.
    var totalUpsample: Int {
        upsampleRates.reduce(1, *) * upsamplingRatios.reduce(1, *)
    }

    /// Output channel dimension after all decoder blocks halve channels.
    var outputDim: Int {
        decoderDim / (1 << upsampleRates.count)
    }

    /// Derived transformer configuration.
    var transformerConfig: DecoderTransformerConfig {
        DecoderTransformerConfig(
            hiddenSize: hiddenSize,
            intermediateSize: intermediateSize,
            numHiddenLayers: numHiddenLayers,
            numAttentionHeads: numAttentionHeads,
            numKeyValueHeads: numKeyValueHeads,
            headDim: headDim,
            rmsNormEps: rmsNormEps,
            ropeTheta: ropeTheta,
            slidingWindow: slidingWindow,
            layerScaleInitialScale: layerScaleInitialScale,
            latentDim: latentDim,
            attentionBias: attentionBias
        )
    }

    /// Derived decoder block configuration.
    var decoderBlockConfig: DecoderBlockConfig {
        DecoderBlockConfig(
            decoderDim: decoderDim,
            upsampleRates: upsampleRates
        )
    }
}

// MARK: - SpeechTokenizerDecoder

/// Full decoder for the speech tokenizer.
///
/// Pipeline: codes → SplitResidualVQ.decode → pre_conv → DecoderTransformer
/// → upsample stages → decoder blocks (initial conv → N DecoderBlocks →
/// SnakeBeta → output conv) → clip(-1, 1).
///
/// The decoder array uses positional indices matching the Python
/// `self.decoder` ModuleList:
/// - `decoder.0`: CausalConv1d (latentDim → decoderDim, k=7) — initial conv
/// - `decoder.1–N`: DecoderBlock (SnakeBeta + upsample + 3 residual units)
/// - `decoder.N+1`: SnakeBeta (output activation)
/// - `decoder.N+2`: CausalConv1d (outputDim → 1, k=7) — output conv
///
/// With default config (4 upsample rates): `decoder.0` through `decoder.6`.
///
/// Reference: `speech_tokenizer.py` class `Qwen3TTSSpeechTokenizerDecoder`.
final class SpeechTokenizerDecoder: Module {
    @ModuleInfo var quantizer: SplitResidualVectorQuantizer
    @ModuleInfo var preConv: CausalConv1d
    @ModuleInfo var preTransformer: DecoderTransformer
    @ModuleInfo var upsample: [UpsampleStage]
    @ModuleInfo var decoder: [Module]

    let config: SpeechTokenizerDecoderConfig

    init(config: SpeechTokenizerDecoderConfig = .default) {
        self.config = config

        _quantizer.wrappedValue = SplitResidualVectorQuantizer(
            nQ: config.numQuantizers,
            nQSemantic: config.numSemanticQuantizers,
            dimension: config.codebookDim / 2,
            inputDimension: config.codebookDim,
            outputDimension: config.codebookDim,
            bins: config.codebookSize
        )

        _preConv.wrappedValue = CausalConv1d(
            inChannels: config.codebookDim,
            outChannels: config.latentDim,
            kernelSize: 3
        )

        _preTransformer.wrappedValue = DecoderTransformer(
            config: config.transformerConfig
        )

        _upsample.wrappedValue = config.upsamplingRatios.map { factor in
            UpsampleStage(dim: config.latentDim, upsampleFactor: factor)
        }

        let blockConfig = config.decoderBlockConfig
        let outputDim = config.outputDim
        var decoderModules: [Module] = [
            CausalConv1d(
                inChannels: config.latentDim,
                outChannels: config.decoderDim,
                kernelSize: 7
            )
        ]
        for i in config.upsampleRates.indices {
            // Safe: iterating the array's own valid indices guarantees DecoderBlock validation passes.
            decoderModules.append(
                try! DecoderBlock(config: blockConfig, layerIndex: i)
            )
        }
        decoderModules.append(SnakeBeta(channels: outputDim))
        decoderModules.append(
            CausalConv1d(
                inChannels: outputDim,
                outChannels: 1,
                kernelSize: 7
            )
        )
        _decoder.wrappedValue = decoderModules
    }

    /// Decode speech tokens to audio waveform.
    ///
    /// - Parameter codes: Integer codes with shape `[batch, numQuantizers, time]`.
    /// - Returns: Audio waveform with shape `[batch, 1, samples]`, clipped to [-1, 1].
    func callAsFunction(_ codes: MLXArray) -> MLXArray {
        // Dequantize
        var hidden = quantizer.decode(codes) // [batch, codebookDim, time]

        // Pre-conv
        hidden = preConv(hidden) // [batch, latentDim, time]

        // Transpose for transformer (NCL → NLC)
        hidden = hidden.transposed(0, 2, 1) // [batch, time, latentDim]

        // Transformer
        hidden = preTransformer(hidden)

        // Back to conv format (NLC → NCL)
        hidden = hidden.transposed(0, 2, 1) // [batch, latentDim, time]

        // Upsampling
        for stage in upsample {
            hidden = stage(hidden)
        }

        // Decoder: initial conv → blocks → output snake → output conv
        var wav = hidden
        for layer in decoder {
            if let conv = layer as? CausalConv1d {
                wav = conv(wav)
            } else if let block = layer as? DecoderBlock {
                wav = block(wav)
            } else if let snake = layer as? SnakeBeta {
                wav = snake(wav)
            }
        }

        return clip(wav, min: -1.0, max: 1.0)
    }

    /// Decode in chunks to handle long sequences.
    ///
    /// Processes `chunkSize` code frames at a time with `leftContextSize`
    /// overlap for continuity. Left-context audio is discarded to avoid
    /// boundary artifacts.
    ///
    /// - Parameters:
    ///   - codes: Integer codes with shape `[batch, numQuantizers, time]`.
    ///   - chunkSize: Number of code frames per chunk (default 300).
    ///   - leftContextSize: Number of overlap frames from the left (default 25).
    /// - Returns: Audio waveform with shape `[batch, 1, samples]`.
    func chunkedDecode(
        _ codes: MLXArray,
        chunkSize: Int = 300,
        leftContextSize: Int = 25
    ) -> MLXArray {
        var wavChunks: [MLXArray] = []
        var startIndex = 0
        let totalFrames = codes.dim(2)
        let upsample = config.totalUpsample

        while startIndex < totalFrames {
            let endIndex = min(startIndex + chunkSize, totalFrames)
            let contextSize = startIndex > leftContextSize
                ? leftContextSize : startIndex
            let chunk = codes[0..., 0..., (startIndex - contextSize) ..< endIndex]
            let wavChunk = self(chunk)
            wavChunks.append(wavChunk[0..., 0..., (contextSize * upsample)...])
            startIndex = endIndex
        }

        return concatenated(wavChunks, axis: 2)
    }

    // MARK: - Weight Loading

    /// Load real weights from a `speech_tokenizer/` directory into this module.
    ///
    /// Uses ``SpeechTokenizerWeightLoader`` to discover, load, sanitize, and
    /// filter decoder weights, then applies them via `Module.update(parameters:verify:)`.
    ///
    /// - Parameter directory: URL to the `speech_tokenizer/` subdirectory
    ///   containing one or more `.safetensors` files.
    /// - Returns: Diagnostic result with any missing or unexpected keys.
    @discardableResult
    func loadWeights(from directory: URL) throws -> SpeechTokenizerWeightLoadResult {
        let result = try SpeechTokenizerWeightLoader.load(from: directory)
        let decoderOnly = SpeechTokenizerWeightLoader.decoderWeights(from: result.weights)
        let pairs = decoderOnly.map { ($0.key, $0.value) }
        try update(parameters: ModuleParameters.unflattened(pairs), verify: .noUnusedKeys)
        MLX.eval(parameters())
        return result
    }
}
