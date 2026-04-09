import Foundation
import HuggingFace
@preconcurrency import MLX
import MLXAudioCodecs
import MLXAudioCore
@preconcurrency import MLXFast
@preconcurrency import MLXLMCommon
import MLXNN
import Tokenizers

public struct TADAConfig: Codable, Sendable {
    public struct RopeScaling: Codable, Sendable {
        public let factor: Float
        public let highFreqFactor: Float
        public let lowFreqFactor: Float
        public let originalMaxPositionEmbeddings: Float
        public let ropeType: String?

        enum CodingKeys: String, CodingKey {
            case factor
            case highFreqFactor = "high_freq_factor"
            case lowFreqFactor = "low_freq_factor"
            case originalMaxPositionEmbeddings = "original_max_position_embeddings"
            case ropeType = "rope_type"
        }
    }

    public let vocabSize: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let headDim: Int?
    public let rmsNormEps: Float
    public let ropeTheta: Float
    public let ropeScaling: RopeScaling?
    public let tieWordEmbeddings: Bool
    public let maxPositionEmbeddings: Int
    public let acousticDim: Int
    public let numTimeClasses: Int
    public let shiftAcoustic: Int
    public let headLayers: Int
    public let headFfnRatio: Float
    public let bottleneckDim: Int?
    public let acousticMean: Float
    public let acousticStd: Float
    public let bosTokenId: Int
    public let eosTokenId: [Int]
    public let eotId: Int

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case tieWordEmbeddings = "tie_word_embeddings"
        case maxPositionEmbeddings = "max_position_embeddings"
        case acousticDim = "acoustic_dim"
        case numTimeClasses = "num_time_classes"
        case shiftAcoustic = "shift_acoustic"
        case headLayers = "head_layers"
        case headFfnRatio = "head_ffn_ratio"
        case bottleneckDim = "bottleneck_dim"
        case acousticMean = "acoustic_mean"
        case acousticStd = "acoustic_std"
        case bosTokenId = "bos_token_id"
        case eosTokenId = "eos_token_id"
        case eotId = "eot_id"
    }

    public init(
        vocabSize: Int = 128_256,
        hiddenSize: Int = 3_072,
        intermediateSize: Int = 8_192,
        numHiddenLayers: Int = 28,
        numAttentionHeads: Int = 24,
        numKeyValueHeads: Int = 8,
        headDim: Int? = nil,
        rmsNormEps: Float = 1e-5,
        ropeTheta: Float = 500_000,
        ropeScaling: RopeScaling? = nil,
        tieWordEmbeddings: Bool = true,
        maxPositionEmbeddings: Int = 131_072,
        acousticDim: Int = 512,
        numTimeClasses: Int = 256,
        shiftAcoustic: Int = 5,
        headLayers: Int = 6,
        headFfnRatio: Float = 4.0,
        bottleneckDim: Int? = nil,
        acousticMean: Float = 0.0,
        acousticStd: Float = 1.5,
        bosTokenId: Int = 128_000,
        eosTokenId: [Int] = [128_001],
        eotId: Int = 128_009
    ) {
        self.vocabSize = vocabSize
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.headDim = headDim
        self.rmsNormEps = rmsNormEps
        self.ropeTheta = ropeTheta
        self.ropeScaling = ropeScaling
        self.tieWordEmbeddings = tieWordEmbeddings
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.acousticDim = acousticDim
        self.numTimeClasses = numTimeClasses
        self.shiftAcoustic = shiftAcoustic
        self.headLayers = headLayers
        self.headFfnRatio = headFfnRatio
        self.bottleneckDim = bottleneckDim
        self.acousticMean = acousticMean
        self.acousticStd = acousticStd
        self.bosTokenId = bosTokenId
        self.eosTokenId = eosTokenId
        self.eotId = eotId
    }

    public init(from decoder: Swift.Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        vocabSize = try container.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 128_256
        hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 3_072
        intermediateSize = try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 8_192
        numHiddenLayers = try container.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 28
        numAttentionHeads = try container.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 24
        numKeyValueHeads = try container.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 8
        headDim = try container.decodeIfPresent(Int.self, forKey: .headDim)
        rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 500_000
        ropeScaling = try container.decodeIfPresent(RopeScaling.self, forKey: .ropeScaling)
        tieWordEmbeddings = try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
        maxPositionEmbeddings = try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 131_072
        acousticDim = try container.decodeIfPresent(Int.self, forKey: .acousticDim) ?? 512
        numTimeClasses = try container.decodeIfPresent(Int.self, forKey: .numTimeClasses) ?? 256
        shiftAcoustic = try container.decodeIfPresent(Int.self, forKey: .shiftAcoustic) ?? 5
        headLayers = try container.decodeIfPresent(Int.self, forKey: .headLayers) ?? 6
        headFfnRatio = try container.decodeIfPresent(Float.self, forKey: .headFfnRatio) ?? 4.0
        bottleneckDim = try container.decodeIfPresent(Int.self, forKey: .bottleneckDim)
        acousticMean = try container.decodeIfPresent(Float.self, forKey: .acousticMean) ?? 0.0
        acousticStd = try container.decodeIfPresent(Float.self, forKey: .acousticStd) ?? 1.5
        bosTokenId = try container.decodeIfPresent(Int.self, forKey: .bosTokenId) ?? 128_000

        if let tokenArray = try? container.decode([Int].self, forKey: .eosTokenId) {
            eosTokenId = tokenArray
        } else if let token = try? container.decode(Int.self, forKey: .eosTokenId) {
            eosTokenId = [token]
        } else {
            eosTokenId = [128_001]
        }

        eotId = try container.decodeIfPresent(Int.self, forKey: .eotId)
            ?? eosTokenId.last
            ?? 128_009
    }

    public var resolvedHeadDim: Int {
        headDim ?? (hiddenSize / max(numAttentionHeads, 1))
    }

    public var numTimeBits: Int {
        max(1, Int(ceil(log2(Double(max(numTimeClasses, 2))))))
    }

    public var latentSize: Int {
        acousticDim + (2 * numTimeBits)
    }
}

/// Pre-computed reference conditioning produced by the TADA encoder/aligner.
/// Returned by `TADATTSModel.extractReferenceConditioning` so callers can cache
/// and reuse the result across multiple synthesis calls without re-running the
/// encoder on the reference audio.
public struct TADAReferenceConditioningData: Sendable {
    /// Acoustic token embeddings, shape [1, tokenCount, acousticDim] in float32.
    public let tokenValues: MLXArray
    /// Aligned frame position for each token (length = tokenCount).
    public let tokenPositions: [Int]
    /// Per-frame alignment mask (length = frameCount).
    public let tokenMask: [UInt8]
    /// Reference text token IDs (length = tokenCount).
    public let textTokens: [Int32]
    /// Total frame count of the reference audio.
    public let frameCount: Int
    /// Transcript of the reference audio.
    public let transcript: String
}

private struct TADAReferenceConditioning {
    let tokenValues: MLXArray
    let tokenPositions: [Int]
    let tokenMask: [UInt8]
    let textTokens: [Int32]
    let frameCount: Int
    let transcript: String
}

private final class TADATimestepIdentity: Module {
    func callAsFunction(_ x: MLXArray) -> MLXArray { x }
}

private func tadaSinusoidalEmbedding(_ timesteps: MLXArray, dim: Int) -> MLXArray {
    var t = timesteps
    if t.ndim == 0 {
        t = t.expandedDimensions(axis: 0)
    }
    let halfDim = max(1, dim / 2)
    // Match Python: freqs[i] = exp(-log(10000) * i / half), denominator is halfDim (not halfDim-1)
    let freqs = exp(
        MLXArray(0 ..< halfDim).asType(.float32)
            * MLXArray(Float(-log(10_000.0) / Float(halfDim)))
    )
    let args = t.asType(.float32).expandedDimensions(axis: 1) * freqs.expandedDimensions(axis: 0)
    // Match Python: concatenate [cos, sin] (NOT [sin, cos])
    var embedding = MLX.concatenated([MLX.cos(args), MLX.sin(args)], axis: -1)
    if embedding.dim(-1) < dim {
        let pad = MLXArray.zeros([embedding.dim(0), dim - embedding.dim(-1)], dtype: embedding.dtype)
        embedding = MLX.concatenated([embedding, pad], axis: -1)
    }
    return embedding
}

private func tadaNormalizeLanguage(_ language: String?) -> String? {
    guard let language else { return nil }
    let normalized = language.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    guard !normalized.isEmpty else { return nil }
    if normalized == "zh" {
        return "ch"
    }
    return normalized
}

private func tadaLogSNRSchedule(_ stepCount: Int) -> [Float] {
    guard stepCount > 0 else { return [0.0, 1.0] }
    let denom = Float(stepCount)
    return (0...stepCount).map { index in
        let linear = 5.0 - 10.0 * (Float(index) / denom)
        return 1.0 / (1.0 + exp(linear / 2.0))
    }
}

private func tadaReadConfig(from modelDir: URL) throws -> TADAConfig {
    let configURL = modelDir.appendingPathComponent("model/config.json")
    let data = try Data(contentsOf: configURL)
    return try JSONDecoder().decode(TADAConfig.self, from: data)
}

private func tadaLoadWeights(from modelDir: URL) throws -> [String: MLXArray] {
    let fileManager = FileManager.default
    guard let enumerator = fileManager.enumerator(at: modelDir, includingPropertiesForKeys: nil) else {
        return [:]
    }

    var weights: [String: MLXArray] = [:]
    for case let fileURL as URL in enumerator where fileURL.pathExtension == "safetensors" {
        let shard = try MLX.loadArrays(url: fileURL)
        weights.merge(shard) { _, new in new }
    }
    return weights
}

private final class TADAKVCache {
    var keys: MLXArray?
    var values: MLXArray?

    var offset: Int { keys.map { $0.dim(2) } ?? 0 }

    @discardableResult
    func update(keys newK: MLXArray, values newV: MLXArray) -> (keys: MLXArray, values: MLXArray) {
        if let k = keys, let v = values {
            keys = MLX.concatenated([k, newK], axis: 2)
            values = MLX.concatenated([v, newV], axis: 2)
        } else {
            keys = newK
            values = newV
        }
        return (keys!, values!)
    }
}

private final class TADALlamaRoPE: Module {
    let dimensions: Int
    let traditional: Bool
    let freqs: MLXArray

    init(dimensions: Int, config: TADAConfig) {
        self.dimensions = dimensions
        self.traditional = false

        let scaling = config.ropeScaling
        let base = config.ropeTheta
        let factor = scaling?.factor ?? 32.0
        let lowFreqFactor = scaling?.lowFreqFactor ?? 1.0
        let highFreqFactor = scaling?.highFreqFactor ?? 4.0
        let oldContext = scaling?.originalMaxPositionEmbeddings ?? 8192.0

        let indices = MLXArray(stride(from: 0, to: dimensions, by: 2)).asType(.float32)
        var freqs = MLX.pow(MLXArray(base), indices / MLXArray(Float(dimensions)))
        let wavelengths = MLXArray(2.0 * Float.pi) * freqs
        let lowFreqWavelength = oldContext / lowFreqFactor
        let highFreqWavelength = oldContext / highFreqFactor

        freqs = MLX.where(wavelengths .> MLXArray(lowFreqWavelength), freqs * factor, freqs)

        let isMedium = logicalAnd(
            wavelengths .> MLXArray(highFreqWavelength),
            wavelengths .< MLXArray(lowFreqWavelength)
        )
        let smoothFactors = (MLXArray(oldContext) / wavelengths - MLXArray(lowFreqFactor))
            / MLXArray(highFreqFactor - lowFreqFactor)
        let smoothDenominator = (MLXArray(1.0) - smoothFactors) / MLXArray(factor) + smoothFactors
        let smoothFreqs = freqs / smoothDenominator
        self.freqs = MLX.where(isMedium, smoothFreqs, freqs)
    }

    func callAsFunction(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        MLXFast.RoPE(
            x,
            dimensions: dimensions,
            traditional: traditional,
            base: nil,
            scale: 1.0,
            offset: offset,
            freqs: freqs
        )
    }
}

private final class TADALlamaAttention: Module {
    let numHeads: Int
    let numKeyValueHeads: Int
    let headDim: Int
    let scale: Float
    let causal: Bool

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    let rope: TADALlamaRoPE

    init(hiddenSize: Int, numHeads: Int, numKeyValueHeads: Int, headDim: Int, causal: Bool, config: TADAConfig) {
        self.numHeads = numHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.headDim = headDim
        self.scale = 1.0 / sqrt(Float(max(headDim, 1)))
        self.causal = causal

        _qProj.wrappedValue = Linear(hiddenSize, numHeads * headDim, bias: false)
        _kProj.wrappedValue = Linear(hiddenSize, numKeyValueHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(hiddenSize, numKeyValueHeads * headDim, bias: false)
        _oProj.wrappedValue = Linear(numHeads * headDim, hiddenSize, bias: false)
        rope = TADALlamaRoPE(dimensions: headDim, config: config)
    }

    func callAsFunction(_ x: MLXArray, cache: TADAKVCache? = nil) -> MLXArray {
        let batch = x.dim(0)
        let length = x.dim(1)
        let cacheOffset = cache?.offset ?? 0

        var q = qProj(x).reshaped(batch, length, numHeads, headDim).transposed(0, 2, 1, 3)
        var k = kProj(x).reshaped(batch, length, numKeyValueHeads, headDim).transposed(0, 2, 1, 3)
        var v = vProj(x).reshaped(batch, length, numKeyValueHeads, headDim).transposed(0, 2, 1, 3)

        q = rope(q, offset: cacheOffset)
        k = rope(k, offset: cacheOffset)

        if let cache {
            (k, v) = cache.update(keys: k, values: v)
        }

        if numKeyValueHeads != numHeads {
            let repeatFactor = max(1, numHeads / max(numKeyValueHeads, 1))
            k = MLX.repeated(k, count: repeatFactor, axis: 1)
            v = MLX.repeated(v, count: repeatFactor, axis: 1)
        }

        var scores = matmul(q, k.transposed(0, 1, 3, 2)) * MLXArray(scale)
        // Apply causal mask: during prefill (length > 1) or when no cache.
        // During single-token decode the query can attend to all past keys by construction.
        if causal && length > 1 {
            let keyLen = k.dim(2)
            let qPos = MLXArray(Array((cacheOffset ..< cacheOffset + length).map { Int32($0) }))
                           .expandedDimensions(axis: 1)  // [Q, 1]
            let kPos = MLXArray(Array((0 ..< keyLen).map { Int32($0) }))
                           .expandedDimensions(axis: 0)  // [1, K]
            let mask = (kPos .<= qPos).expandedDimensions(axes: [0, 1])
            scores = MLX.where(mask, scores, MLXArray(Float(-1e9)))
        }

        let attn = softmax(scores, axis: -1)
        let output = matmul(attn, v).transposed(0, 2, 1, 3).reshaped(batch, length, numHeads * headDim)
        return oProj(output)
    }
}

private final class TADASwiGLUFFN: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(hiddenSize: Int, intermediateSize: Int) {
        _gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

private final class TADALlamaBlock: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: TADALlamaAttention
    @ModuleInfo(key: "mlp") var mlp: TADASwiGLUFFN
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(config: TADAConfig) {
        _selfAttention.wrappedValue = TADALlamaAttention(
            hiddenSize: config.hiddenSize,
            numHeads: config.numAttentionHeads,
            numKeyValueHeads: config.numKeyValueHeads,
            headDim: config.resolvedHeadDim,
            causal: true,
            config: config
        )
        _mlp.wrappedValue = TADASwiGLUFFN(hiddenSize: config.hiddenSize, intermediateSize: config.intermediateSize)
        _inputLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ x: MLXArray, cache: TADAKVCache? = nil) -> MLXArray {
        let h = x + selfAttention(inputLayerNorm(x), cache: cache)
        return h + mlp(postAttentionLayerNorm(h))
    }
}

private final class TADALlamaBackbone: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [TADALlamaBlock]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(config: TADAConfig) {
        _embedTokens.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        _layers.wrappedValue = (0..<config.numHiddenLayers).map { _ in TADALlamaBlock(config: config) }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(inputIds: MLXArray, inputsEmbeds: MLXArray? = nil, caches: [TADAKVCache]? = nil) -> MLXArray {
        var hidden = inputsEmbeds ?? embedTokens(inputIds)
        for (i, layer) in layers.enumerated() {
            hidden = layer(hidden, cache: caches?[i])
        }
        return norm(hidden)
    }

    /// Returns (hidden [B,L,H], logits [B,L,V]) using tied embedding weights.
    func forwardWithLogits(
        inputIds: MLXArray,
        inputsEmbeds: MLXArray? = nil,
        caches: [TADAKVCache]? = nil
    ) -> (MLXArray, MLXArray) {
        let hidden = callAsFunction(inputIds: inputIds, inputsEmbeds: inputsEmbeds, caches: caches)
        let logits = matmul(hidden, embedTokens.weight.T)
        return (hidden, logits)
    }
}

private final class TADAWav2Vec2FeatureExtractorLayer: Module {
    @ModuleInfo(key: "conv") var conv: MLXNN.Conv1d
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
    private let useLayerNorm: Bool

    init(inputChannels: Int, outputChannels: Int, kernelSize: Int, stride: Int, useLayerNorm: Bool) {
        self.useLayerNorm = useLayerNorm
        _conv.wrappedValue = MLXNN.Conv1d(
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: kernelSize,
            stride: stride,
            bias: true
        )
        _layerNorm.wrappedValue = LayerNorm(dimensions: outputChannels)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        useLayerNorm ? gelu(layerNorm(conv(x))) : gelu(conv(x))
    }
}

private final class TADAWav2Vec2FeatureExtractor: Module {
    @ModuleInfo(key: "conv_layers") var convLayers: [TADAWav2Vec2FeatureExtractorLayer]

    override init() {
        let dims = Array(repeating: 512, count: 7)
        let kernels = [10, 3, 3, 3, 3, 2, 2]
        let strides = [5, 2, 2, 2, 2, 2, 2]
        var inputChannels = 1
        _convLayers.wrappedValue = zip(zip(dims, kernels), strides).enumerated().map { index, entry in
            let layer = TADAWav2Vec2FeatureExtractorLayer(
                inputChannels: inputChannels,
                outputChannels: entry.0.0,
                kernelSize: entry.0.1,
                stride: entry.1,
                useLayerNorm: index == 0
            )
            inputChannels = entry.0.0
            return layer
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        convLayers.reduce(x) { partial, layer in layer(partial) }
    }
}

private final class TADAWav2Vec2PositionalConvEmbedding: Module {
    @ModuleInfo(key: "conv") var conv: MLXNN.Conv1d

    init(hiddenSize: Int) {
        _conv.wrappedValue = MLXNN.Conv1d(
            inputChannels: hiddenSize,
            outputChannels: hiddenSize,
            kernelSize: 128,
            padding: 64,
            groups: 16,
            bias: true
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = gelu(conv(x))
        return y[0..., ..<x.dim(1), 0...]
    }
}

private final class TADAWav2Vec2Attention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    let numHeads: Int
    let headDim: Int
    let scale: Float

    init(hiddenSize: Int, numHeads: Int) {
        self.numHeads = numHeads
        headDim = hiddenSize / max(numHeads, 1)
        scale = 1.0 / sqrt(Float(max(headDim, 1)))
        _qProj.wrappedValue = Linear(hiddenSize, hiddenSize)
        _kProj.wrappedValue = Linear(hiddenSize, hiddenSize)
        _vProj.wrappedValue = Linear(hiddenSize, hiddenSize)
        _outProj.wrappedValue = Linear(hiddenSize, hiddenSize)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let batch = x.dim(0)
        let length = x.dim(1)
        let q = qProj(x).reshaped(batch, length, numHeads, headDim).transposed(0, 2, 1, 3)
        let k = kProj(x).reshaped(batch, length, numHeads, headDim).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped(batch, length, numHeads, headDim).transposed(0, 2, 1, 3)
        let weights = softmax(matmul(q, k.transposed(0, 1, 3, 2)) * MLXArray(scale), axis: -1)
        let out = matmul(weights, v).transposed(0, 2, 1, 3).reshaped(batch, length, numHeads * headDim)
        return outProj(out)
    }
}

private final class TADAWav2Vec2FeedForward: Module {
    @ModuleInfo(key: "intermediate_dense") var intermediateDense: Linear
    @ModuleInfo(key: "output_dense") var outputDense: Linear

    init(hiddenSize: Int, intermediateSize: Int) {
        _intermediateDense.wrappedValue = Linear(hiddenSize, intermediateSize)
        _outputDense.wrappedValue = Linear(intermediateSize, hiddenSize)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        outputDense(gelu(intermediateDense(x)))
    }
}

private final class TADAWav2Vec2EncoderLayer: Module {
    @ModuleInfo(key: "attention") var attention: TADAWav2Vec2Attention
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
    @ModuleInfo(key: "feed_forward") var feedForward: TADAWav2Vec2FeedForward
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: LayerNorm

    init(hiddenSize: Int, numHeads: Int, intermediateSize: Int) {
        _attention.wrappedValue = TADAWav2Vec2Attention(hiddenSize: hiddenSize, numHeads: numHeads)
        _layerNorm.wrappedValue = LayerNorm(dimensions: hiddenSize)
        _feedForward.wrappedValue = TADAWav2Vec2FeedForward(hiddenSize: hiddenSize, intermediateSize: intermediateSize)
        _finalLayerNorm.wrappedValue = LayerNorm(dimensions: hiddenSize)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = x + attention(layerNorm(x))
        return h + feedForward(finalLayerNorm(h))
    }
}

private final class TADAWav2Vec2Encoder: Module {
    @ModuleInfo(key: "pos_conv_embed") var posConvEmbed: TADAWav2Vec2PositionalConvEmbedding
    @ModuleInfo(key: "layers") var layers: [TADAWav2Vec2EncoderLayer]
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm

    override init() {
        _posConvEmbed.wrappedValue = TADAWav2Vec2PositionalConvEmbedding(hiddenSize: 1024)
        _layers.wrappedValue = (0..<24).map { _ in
            TADAWav2Vec2EncoderLayer(hiddenSize: 1024, numHeads: 16, intermediateSize: 4096)
        }
        _layerNorm.wrappedValue = LayerNorm(dimensions: 1024)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var hidden = x
        hidden = hidden + posConvEmbed(hidden)
        for layer in layers {
            hidden = layer(hidden)
        }
        return layerNorm(hidden)
    }
}

private final class TADAAligner: Module {
    @ModuleInfo(key: "feature_extractor") var featureExtractor: TADAWav2Vec2FeatureExtractor
    @ModuleInfo(key: "encoder") var encoder: TADAWav2Vec2Encoder
    @ModuleInfo(key: "lm_head") var lmHead: Linear
    @ModuleInfo(key: "input_projection") var inputProjection: Linear

    init(vocabSize: Int) {
        _featureExtractor.wrappedValue = TADAWav2Vec2FeatureExtractor()
        _encoder.wrappedValue = TADAWav2Vec2Encoder()
        _inputProjection.wrappedValue = Linear(512, 1024)
        _lmHead.wrappedValue = Linear(1024, vocabSize)
    }

    func callAsFunction(_ audio16k: MLXArray) -> MLXArray {
        let meanValue = mean(audio16k, axis: -1, keepDims: true)
        let variance = mean((audio16k - meanValue) * (audio16k - meanValue), axis: -1, keepDims: true)
        let normalized = (audio16k - meanValue) / sqrt(variance + 1e-7)
        let input = normalized.ndim == 1 ? normalized.reshaped(1, -1, 1) : normalized.reshaped(1, -1, 1)
        let features = featureExtractor(input)
        let projected = inputProjection(features)
        let hidden = encoder(projected)
        return lmHead(hidden)
    }

    func align(audio16k: MLXArray, tokenIds: [Int32]) -> (positions: [Int], mask: [UInt8], frameCount: Int) {
        guard !tokenIds.isEmpty else { return ([], [], 0) }

        let logits = callAsFunction(audio16k)
        let tokenIdArray = MLXArray(tokenIds).asType(.int32)
        let tokenScores = take(logits, tokenIdArray, axis: 2)
        eval(tokenScores)

        let frameCount = tokenScores.dim(1)
        if frameCount == 0 {
            return (Array(repeating: 1, count: tokenIds.count), [], 0)
        }

        let flattened = tokenScores.squeezed(axis: 0).asArray(Float.self)
        let tokenCount = tokenIds.count
        var positions: [Int] = []
        positions.reserveCapacity(tokenCount)

        var startFrame = 0
        for tokenIndex in 0..<tokenCount {
            let remainingTokens = tokenCount - tokenIndex
            let endFrameExclusive = max(startFrame + 1, frameCount - (remainingTokens - 1))
            var bestFrame = startFrame
            var bestScore = -Float.greatestFiniteMagnitude
            for frame in startFrame..<endFrameExclusive {
                let score = flattened[(frame * tokenCount) + tokenIndex]
                if score > bestScore {
                    bestScore = score
                    bestFrame = frame
                }
            }
            positions.append(bestFrame + 1)
            startFrame = min(frameCount - 1, bestFrame + 1)
        }

        var mask = Array(repeating: UInt8(0), count: frameCount)
        for position in positions {
            let index = max(0, min(frameCount - 1, position - 1))
            mask[index] = 1
        }
        return (positions, mask, frameCount)
    }
}

// Python-faithful ResidualUnit: named keys snake1/conv1/snake2/conv2 with plain Conv1d
private final class TADAResidualUnit: Module, UnaryLayer {
    @ModuleInfo(key: "snake1") var snake1: DescriptSnake1d
    @ModuleInfo(key: "conv1") var conv1: MLXNN.Conv1d
    @ModuleInfo(key: "snake2") var snake2: DescriptSnake1d
    @ModuleInfo(key: "conv2") var conv2: MLXNN.Conv1d

    init(channels: Int, dilation: Int) {
        let padding = ((7 - 1) * dilation) / 2
        _snake1.wrappedValue = DescriptSnake1d(channels: channels)
        _conv1.wrappedValue = MLXNN.Conv1d(
            inputChannels: channels,
            outputChannels: channels,
            kernelSize: 7,
            padding: padding,
            dilation: dilation
        )
        _snake2.wrappedValue = DescriptSnake1d(channels: channels)
        _conv2.wrappedValue = MLXNN.Conv1d(inputChannels: channels, outputChannels: channels, kernelSize: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = snake1(x)
        y = conv1(y)
        y = snake2(y)
        y = conv2(y)
        let diff = x.dim(1) - y.dim(1)
        let residual = diff > 0 ? x[0..., (diff / 2)..<(diff / 2 + y.dim(1)), 0...] : x
        return residual + y
    }
}

// Python-faithful EncoderBlock: named keys res1/res2/res3/snake/conv
// dim = output channels (input = dim/2); mirrors Python EncoderBlock(dim, stride)
private final class TADAEncoderBlock: Module, UnaryLayer {
    @ModuleInfo(key: "res1") var res1: TADAResidualUnit
    @ModuleInfo(key: "res2") var res2: TADAResidualUnit
    @ModuleInfo(key: "res3") var res3: TADAResidualUnit
    @ModuleInfo(key: "snake") var snake: DescriptSnake1d
    @ModuleInfo(key: "conv") var conv: MLXNN.Conv1d

    init(dim: Int, stride: Int) {
        let inputDim = dim / 2
        _res1.wrappedValue = TADAResidualUnit(channels: inputDim, dilation: 1)
        _res2.wrappedValue = TADAResidualUnit(channels: inputDim, dilation: 3)
        _res3.wrappedValue = TADAResidualUnit(channels: inputDim, dilation: 9)
        _snake.wrappedValue = DescriptSnake1d(channels: inputDim)
        _conv.wrappedValue = MLXNN.Conv1d(
            inputChannels: inputDim,
            outputChannels: dim,
            kernelSize: 2 * stride,
            stride: stride,
            padding: Int(ceil(Double(stride) / 2.0))
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = res1(x)
        y = res2(y)
        y = res3(y)
        y = snake(y)
        y = conv(y)
        return y
    }
}

// Python-faithful WavEncoder (NLC): initial_conv/blocks/final_snake/final_conv
// Mirrors Python WavEncoder with STRIDES=(6,5,4,4) and d_model doubling from 64
// Named TADANLCWavEncoder to avoid conflict with public TADAWavEncoder in TADATTSEncoder.swift
private final class TADANLCWavEncoder: Module {
    @ModuleInfo(key: "initial_conv") var initialConv: MLXNN.Conv1d
    @ModuleInfo(key: "blocks") var blocks: [TADAEncoderBlock]
    @ModuleInfo(key: "final_snake") var finalSnake: DescriptSnake1d
    @ModuleInfo(key: "final_conv") var finalConv: MLXNN.Conv1d

    override init() {
        _initialConv.wrappedValue = MLXNN.Conv1d(inputChannels: 1, outputChannels: 64, kernelSize: 7, padding: 3)
        var dModel = 64
        var encoderBlocks: [TADAEncoderBlock] = []
        for stride in [6, 5, 4, 4] {
            dModel *= 2
            encoderBlocks.append(TADAEncoderBlock(dim: dModel, stride: stride))
        }
        _blocks.wrappedValue = encoderBlocks
        _finalSnake.wrappedValue = DescriptSnake1d(channels: dModel)
        _finalConv.wrappedValue = MLXNN.Conv1d(inputChannels: dModel, outputChannels: 1024, kernelSize: 3, padding: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = initialConv(x)
        for block in blocks {
            y = block(y)
        }
        y = finalSnake(y)
        y = finalConv(y)
        return y
    }
}

private final class TADAMonoLocalAttention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    let numHeads: Int
    let headDim: Int
    let scale: Float
    let rope: TADALlamaRoPE

    init(hiddenSize: Int, numHeads: Int, config: TADAConfig) {
        self.numHeads = numHeads
        headDim = hiddenSize / max(numHeads, 1)
        scale = 1.0 / sqrt(Float(max(headDim, 1)))
        _qProj.wrappedValue = Linear(hiddenSize, hiddenSize)
        _kProj.wrappedValue = Linear(hiddenSize, hiddenSize)
        _vProj.wrappedValue = Linear(hiddenSize, hiddenSize)
        _outProj.wrappedValue = Linear(hiddenSize, hiddenSize)
        rope = TADALlamaRoPE(dimensions: headDim, config: config)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let batch = x.dim(0)
        let length = x.dim(1)
        var q = qProj(x).reshaped(batch, length, numHeads, headDim).transposed(0, 2, 1, 3)
        var k = kProj(x).reshaped(batch, length, numHeads, headDim).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped(batch, length, numHeads, headDim).transposed(0, 2, 1, 3)
        q = rope(q)
        k = rope(k)
        let weights = softmax(matmul(q, k.transposed(0, 1, 3, 2)) * MLXArray(scale), axis: -1)
        let out = matmul(weights, v).transposed(0, 2, 1, 3).reshaped(batch, length, numHeads * headDim)
        return outProj(out)
    }
}

private final class TADALocalTransformerLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: TADAMonoLocalAttention
    @ModuleInfo(key: "norm1") var norm1: RMSNorm
    @ModuleInfo(key: "ffn") var ffn: TADASwiGLUFFN
    @ModuleInfo(key: "norm2") var norm2: RMSNorm

    init(hiddenSize: Int, numHeads: Int, intermediateSize: Int, config: TADAConfig) {
        _selfAttention.wrappedValue = TADAMonoLocalAttention(hiddenSize: hiddenSize, numHeads: numHeads, config: config)
        _norm1.wrappedValue = RMSNorm(dimensions: hiddenSize)
        _ffn.wrappedValue = TADASwiGLUFFN(hiddenSize: hiddenSize, intermediateSize: intermediateSize)
        _norm2.wrappedValue = RMSNorm(dimensions: hiddenSize)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = x + selfAttention(norm1(x))
        return h + ffn(norm2(h))
    }
}

private final class TADALocalTransformer: Module {
    @ModuleInfo(key: "layers") var layers: [TADALocalTransformerLayer]

    init(hiddenSize: Int, numHeads: Int, intermediateSize: Int, layerCount: Int, config: TADAConfig) {
        _layers.wrappedValue = (0..<layerCount).map { _ in
            TADALocalTransformerLayer(
                hiddenSize: hiddenSize,
                numHeads: numHeads,
                intermediateSize: intermediateSize,
                config: config
            )
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        layers.reduce(x) { hidden, layer in layer(hidden) }
    }
}

private final class TADAReferenceEncoder: Module {
    @ModuleInfo(key: "wav_encoder") var wavEncoder: TADANLCWavEncoder
    // Key matches Python encoder/weights.safetensors → local_attention_encoder.*
    // Architecture matches Python LocalAttentionEncoder (post-norm, GELU FFN, combined QKV)
    @ModuleInfo(key: "local_attention_encoder") var encoder: TADAv2AttentionStack
    @ModuleInfo(key: "hidden_linear") var hiddenLinear: Linear
    @ModuleInfo(key: "pos_emb") var posEmb: Embedding

    let config: TADAConfig

    init(config: TADAConfig) {
        self.config = config
        _wavEncoder.wrappedValue = TADANLCWavEncoder()
        _encoder.wrappedValue = TADAv2AttentionStack(
            hiddenSize: 1024,
            numLayers: 6,
            numHeads: 8,
            feedForwardSize: 4096
        )
        // Python hidden_linear uses default bias=True
        _hiddenLinear.wrappedValue = Linear(1024, config.acousticDim, bias: true)
        _posEmb.wrappedValue = Embedding(embeddingCount: 2, dimensions: 1024)
    }

    func encode(audio24k: MLXArray, tokenPositions: [Int], tokenMask: [UInt8]) -> MLXArray {
        guard !tokenPositions.isEmpty else {
            return MLXArray.zeros([1, 0, config.acousticDim], dtype: .float32)
        }

        let input1D = audio24k.ndim == 1 ? audio24k : audio24k.reshaped(-1)
        let padded = MLX.padded(
            input1D.reshaped(1, -1, 1),
            widths: [IntOrPair(0), IntOrPair((960, 960)), IntOrPair(0)]
        )
        var hidden = wavEncoder(padded)

        let frameCount = hidden.dim(1)
        var boundary = Array(repeating: Int32(0), count: frameCount)
        for (index, value) in tokenMask.enumerated() where index < frameCount {
            boundary[index] = Int32(value)
        }
        let boundaryTensor = MLXArray(boundary).reshaped(1, -1).asType(.int32)
        hidden = hidden + posEmb(boundaryTensor)
        // Use segment attention mask matching Python encoder.create_segment_attention_mask
        let attnMask = tadaEncoderSegmentMask(tokenMask: boundaryTensor)
        hidden = encoder(hidden, mask: attnMask)

        let projected = hiddenLinear(hidden)
        let gatherIndices = tokenPositions.map { Int32(max(0, min(frameCount - 1, $0 - 1))) }
        let gathered = take(projected, MLXArray(gatherIndices).asType(.int32), axis: 1)
        return (gathered - MLXArray(config.acousticMean)) / MLXArray(config.acousticStd)
    }
}

// Python-faithful DecoderBlock: named keys snake/conv_transpose/res1/res2/res3
private final class TADADecoderBlock: Module, UnaryLayer {
    @ModuleInfo(key: "snake") var snake: DescriptSnake1d
    @ModuleInfo(key: "conv_transpose") var convTranspose: ConvTransposed1d
    @ModuleInfo(key: "res1") var res1: TADAResidualUnit
    @ModuleInfo(key: "res2") var res2: TADAResidualUnit
    @ModuleInfo(key: "res3") var res3: TADAResidualUnit

    init(inputDim: Int, outputDim: Int, stride: Int) {
        _snake.wrappedValue = DescriptSnake1d(channels: inputDim)
        _convTranspose.wrappedValue = ConvTransposed1d(
            inputChannels: inputDim,
            outputChannels: outputDim,
            kernelSize: 2 * stride,
            stride: stride,
            padding: Int(ceil(Double(stride) / 2.0))
        )
        _res1.wrappedValue = TADAResidualUnit(channels: outputDim, dilation: 1)
        _res2.wrappedValue = TADAResidualUnit(channels: outputDim, dilation: 3)
        _res3.wrappedValue = TADAResidualUnit(channels: outputDim, dilation: 9)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = snake(x)
        y = convTranspose(y)
        y = res1(y)
        y = res2(y)
        y = res3(y)
        return y
    }
}

// Python-faithful DACDecoder: initial_conv/blocks/final_snake/final_conv with plain Conv1d
// Mirrors Python DACDecoder with input_channel=1024, channels=1536, rates=(4,4,5,6)
private final class TADADacDecoder: Module {
    @ModuleInfo(key: "initial_conv") var initialConv: MLXNN.Conv1d
    @ModuleInfo(key: "blocks") var blocks: [TADADecoderBlock]
    @ModuleInfo(key: "final_snake") var finalSnake: DescriptSnake1d
    @ModuleInfo(key: "final_conv") var finalConv: MLXNN.Conv1d

    override init() {
        _initialConv.wrappedValue = MLXNN.Conv1d(inputChannels: 1024, outputChannels: 1536, kernelSize: 7, padding: 3)
        let strides = [4, 4, 5, 6]
        var decoderBlocks: [TADADecoderBlock] = []
        for (index, stride) in strides.enumerated() {
            let inputDim = 1536 / Int(pow(2.0, Double(index)))
            let outputDim = 1536 / Int(pow(2.0, Double(index + 1)))
            decoderBlocks.append(TADADecoderBlock(inputDim: inputDim, outputDim: outputDim, stride: stride))
        }
        _blocks.wrappedValue = decoderBlocks
        let finalChannels = 1536 / Int(pow(2.0, Double(strides.count)))
        _finalSnake.wrappedValue = DescriptSnake1d(channels: finalChannels)
        _finalConv.wrappedValue = MLXNN.Conv1d(inputChannels: finalChannels, outputChannels: 1, kernelSize: 7, padding: 3)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = initialConv(x)
        for block in blocks {
            y = block(y)
        }
        y = finalSnake(y)
        y = finalConv(y)
        return tanh(y)
    }
}

private final class TADAWaveDecoder: Module {
    @ModuleInfo(key: "decoder_proj") var decoderProj: Linear
    // Key matches Python decoder/weights.safetensors → local_attention_decoder.*
    // Architecture matches Python LocalAttentionEncoder (post-norm, GELU FFN, combined QKV, no posEmb)
    @ModuleInfo(key: "local_attention_decoder") var decoder: TADAv2AttentionStack
    @ModuleInfo(key: "wav_decoder") var wavDecoder: TADADacDecoder

    let config: TADAConfig

    init(config: TADAConfig) {
        self.config = config
        // Python decoder_proj has bias; match it
        _decoderProj.wrappedValue = Linear(config.acousticDim, 1024, bias: true)
        _decoder.wrappedValue = TADAv2AttentionStack(
            hiddenSize: 1024,
            numLayers: 6,
            numHeads: 8,
            feedForwardSize: 4096
        )
        _wavDecoder.wrappedValue = TADADacDecoder()
    }

    func callAsFunction(acoustic: MLXArray, timeBefore: [Int], timeAfter: [Int]) -> MLXArray {
        let expanded = expand(acoustic: acoustic, timeBefore: timeBefore, timeAfter: timeAfter)
        var hidden = decoderProj(expanded.frames)
        // Use segment attention mask matching Python decoder.create_segment_attention_mask
        let attnMask = tadaDecoderSegmentMask(tokenMask: expanded.tokenMasks)
        hidden = decoder(hidden, mask: attnMask)
        let waveform = wavDecoder(hidden).squeezed()
        return waveform
    }

    private func expand(acoustic: MLXArray, timeBefore: [Int], timeAfter: [Int]) -> (frames: MLXArray, tokenMasks: MLXArray) {
        let tokenCount = acoustic.dim(1)
        guard tokenCount > 0 else {
            return (
                MLXArray.zeros([1, 1, config.acousticDim], dtype: .float32),
                MLXArray.zeros([1, 1], type: Int32.self)
            )
        }

        var frames: [MLXArray] = []
        var tokenMaskVals: [Int32] = []
        frames.reserveCapacity(tokenCount * 2)
        tokenMaskVals.reserveCapacity(tokenCount * 2)

        for index in 0..<tokenCount {
            let before = index < timeBefore.count ? max(0, timeBefore[index] - 1) : 0
            let after = index < timeAfter.count ? max(0, timeAfter[index]) : 0

            for _ in 0..<before {
                frames.append(MLXArray.zeros([config.acousticDim], dtype: acoustic.dtype))
                tokenMaskVals.append(0)
            }
            frames.append(acoustic[0, index])
            tokenMaskVals.append(1)
            for _ in 0..<after {
                frames.append(MLXArray.zeros([config.acousticDim], dtype: acoustic.dtype))
                tokenMaskVals.append(0)
            }
        }

        let stacked = MLX.stacked(frames, axis: 0).expandedDimensions(axis: 0)
        let tokenMaskTensor = MLXArray(tokenMaskVals).reshaped(1, -1).asType(.int32)
        return (stacked, tokenMaskTensor)
    }
}

private final class TADAMonoTimestepEmbedder: Module {
    @ModuleInfo(key: "mlp") var mlp: [Module]
    let hiddenSize: Int

    init(hiddenSize: Int) {
        self.hiddenSize = hiddenSize
        _mlp.wrappedValue = [
            Linear(256, hiddenSize, bias: false),
            TADATimestepIdentity(),
            Linear(hiddenSize, hiddenSize, bias: false),
        ]
    }

    func callAsFunction(_ timestep: MLXArray) -> MLXArray {
        var hidden = tadaSinusoidalEmbedding(timestep, dim: 256)
        hidden = (mlp[0] as! Linear)(hidden)
        hidden = silu(hidden)
        hidden = (mlp[2] as! Linear)(hidden)
        return hidden
    }
}

private final class TADAMonoVibeVoiceHeadLayer: Module {
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "adaLN_modulation") var adaLNModulation: Linear
    @ModuleInfo(key: "ffn") var ffn: TADASwiGLUFFN

    let hiddenSize: Int

    init(hiddenSize: Int, ffnRatio: Float) {
        self.hiddenSize = hiddenSize
        _norm.wrappedValue = RMSNorm(dimensions: hiddenSize)
        _adaLNModulation.wrappedValue = Linear(hiddenSize, hiddenSize * 3, bias: false)
        _ffn.wrappedValue = TADASwiGLUFFN(hiddenSize: hiddenSize, intermediateSize: Int(Float(hiddenSize) * ffnRatio))
    }

    func callAsFunction(_ x: MLXArray, conditioning: MLXArray) -> MLXArray {
        let modulation = adaLNModulation(MLXNN.silu(conditioning))  // [B, 3H]
        let shift = modulation[0..., ..<hiddenSize]
        let scale = modulation[0..., hiddenSize..<(2 * hiddenSize)]
        let gate = modulation[0..., (2 * hiddenSize)...]
        let normalized = norm(x) * (MLXArray(Float(1)) + scale) + shift
        return x + gate * ffn(normalized)
    }
}

private final class TADAMonoVibeVoiceFinalLayer: Module {
    @ModuleInfo(key: "norm_final") var normFinal: RMSNorm
    @ModuleInfo(key: "adaLN_modulation") var adaLNModulation: Linear
    @ModuleInfo(key: "linear") var linear: Linear

    let hiddenSize: Int

    init(hiddenSize: Int, latentSize: Int) {
        self.hiddenSize = hiddenSize
        _normFinal.wrappedValue = RMSNorm(dimensions: hiddenSize)
        _adaLNModulation.wrappedValue = Linear(hiddenSize, hiddenSize * 2, bias: false)
        _linear.wrappedValue = Linear(hiddenSize, latentSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray, conditioning: MLXArray) -> MLXArray {
        let modulation = adaLNModulation(MLXNN.silu(conditioning))  // [B, 2H]
        let shift = modulation[0..., ..<hiddenSize]
        let scale = modulation[0..., hiddenSize...]
        let normalized = normFinal(x) * (MLXArray(Float(1)) + scale) + shift
        return linear(normalized)
    }
}

private final class TADAVibeVoiceHead: Module {
    @ModuleInfo(key: "noisy_images_proj") var noisyImagesProj: Linear
    @ModuleInfo(key: "cond_proj") var condProj: Linear
    @ModuleInfo(key: "t_embedder") var tEmbedder: TADAMonoTimestepEmbedder
    @ModuleInfo(key: "layers") var layers: [TADAMonoVibeVoiceHeadLayer]
    @ModuleInfo(key: "final_layer") var finalLayer: TADAMonoVibeVoiceFinalLayer

    let config: TADAConfig

    init(config: TADAConfig) {
        self.config = config
        _noisyImagesProj.wrappedValue = Linear(config.latentSize, config.hiddenSize, bias: false)
        _condProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: false)
        _tEmbedder.wrappedValue = TADAMonoTimestepEmbedder(hiddenSize: config.hiddenSize)
        _layers.wrappedValue = (0..<config.headLayers).map { _ in
            TADAMonoVibeVoiceHeadLayer(hiddenSize: config.hiddenSize, ffnRatio: config.headFfnRatio)
        }
        _finalLayer.wrappedValue = TADAMonoVibeVoiceFinalLayer(hiddenSize: config.hiddenSize, latentSize: config.latentSize)
    }

    func callAsFunction(noise: MLXArray, timestep: MLXArray, cond: MLXArray) -> MLXArray {
        let t = tEmbedder(timestep)  // [batch, hidden]
        // cond arrives as [batch, 1, hidden] from backbone — squeeze to 2D
        let condSqueezed = cond.ndim == 3 ? cond.reshaped(cond.dim(0), cond.dim(2)) : cond
        let conditioning = condProj(condSqueezed) + t  // [batch, hidden]
        var hidden = noisyImagesProj(noise)  // [batch, hidden], noise=[batch, latentSize]
        for layer in layers {
            hidden = layer(hidden, conditioning: conditioning)
        }
        return finalLayer(hidden, conditioning: conditioning)
    }

    func solve(
        cond: MLXArray,
        negativeCond: MLXArray,
        steps: Int,
        acousticCFGScale: Float,
        durationCFGScale: Float,
        noiseTemperature: Float
    ) -> MLXArray {
        let batch = cond.dim(0)
        var speech = MLXRandom.normal([batch, config.latentSize]) * MLXArray(noiseTemperature)
        let schedule = tadaLogSNRSchedule(max(steps, 2))

        for index in 0..<(schedule.count - 1) {
            let tPrev = schedule[index]
            let tCurr = schedule[index + 1]
            let dt = tCurr - tPrev
            let timestep = MLXArray(Array(repeating: tPrev, count: batch))

            let velocityPos = callAsFunction(noise: speech, timestep: timestep, cond: cond)
            let velocityNeg = callAsFunction(noise: speech, timestep: timestep, cond: negativeCond)

            // Cosine CFG schedule: full guidance at t≈0, decays to 1.0 at t≈1
            let acousticCFGActual = 1.0 + (acousticCFGScale - 1.0) * 0.5 * (1.0 + Foundation.cos(Float.pi * tPrev))
            let durationCFGActual = 1.0 + (durationCFGScale - 1.0) * 0.5 * (1.0 + Foundation.cos(Float.pi * tPrev))

            let acousticPos = velocityPos[0..., ..<config.acousticDim]
            let acousticNeg = velocityNeg[0..., ..<config.acousticDim]
            let timePos = velocityPos[0..., config.acousticDim...]
            let timeNeg = velocityNeg[0..., config.acousticDim...]

            let guidedAcoustic = acousticNeg + MLXArray(acousticCFGActual) * (acousticPos - acousticNeg)
            let guidedTime = timeNeg + MLXArray(durationCFGActual) * (timePos - timeNeg)
            let velocity = MLX.concatenated([guidedAcoustic, guidedTime], axis: -1)

            speech = speech + MLXArray(dt) * velocity
        }

        return speech
    }
}

public final class TADATTSModel: Module, SpeechGenerationModel, ConditionedSpeechGenerationModel, @unchecked Sendable {
    public let config: TADAConfig

    @ModuleInfo(key: "model") fileprivate var model: TADALlamaBackbone
    @ModuleInfo(key: "acoustic_proj") var acousticProj: Linear
    @ModuleInfo(key: "time_start_embed") var timeStartEmbed: Embedding
    @ModuleInfo(key: "time_end_embed") var timeEndEmbed: Embedding
    @ModuleInfo(key: "acoustic_mask_emb") var acousticMaskEmb: Embedding
    @ModuleInfo(key: "encoder") fileprivate var encoder: TADAReferenceEncoder
    @ModuleInfo(key: "decoder") fileprivate var decoder: TADAWaveDecoder
    @ModuleInfo(key: "aligner") fileprivate var aligner: TADAAligner
    @ModuleInfo(key: "prediction_head") fileprivate var predictionHead: TADAVibeVoiceHead

    public var tokenizer: Tokenizer?

    public var sampleRate: Int { 24_000 }

    public var defaultGenerationParameters: GenerateParameters {
        GenerateParameters(
            maxTokens: 1024,
            temperature: 0.6,
            topP: 0.9,
            repetitionPenalty: 1.1
        )
    }

    public init(config: TADAConfig) {
        self.config = config
        _model.wrappedValue = TADALlamaBackbone(config: config)
        _acousticProj.wrappedValue = Linear(config.acousticDim, config.hiddenSize, bias: false)
        _timeStartEmbed.wrappedValue = Embedding(embeddingCount: config.numTimeClasses, dimensions: config.hiddenSize)
        _timeEndEmbed.wrappedValue = Embedding(embeddingCount: config.numTimeClasses, dimensions: config.hiddenSize)
        _acousticMaskEmb.wrappedValue = Embedding(embeddingCount: 2, dimensions: config.hiddenSize)
        _encoder.wrappedValue = TADAReferenceEncoder(config: config)
        _decoder.wrappedValue = TADAWaveDecoder(config: config)
        _aligner.wrappedValue = TADAAligner(vocabSize: config.vocabSize)
        _predictionHead.wrappedValue = TADAVibeVoiceHead(config: config)
    }

    public func generate(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) async throws -> MLXArray {
        _ = voice
        guard let tokenizer else {
            throw AudioGenerationError.modelNotInitialized("Tokenizer not loaded")
        }
        guard let refAudio else {
            throw AudioGenerationError.invalidInput("TADA requires reference audio for cloning.")
        }

        let resolvedLanguage = tadaNormalizeLanguage(language)
        let reference = try buildReferenceConditioning(
            tokenizer: tokenizer,
            text: text,
            refAudio: refAudio,
            refText: refText,
            language: resolvedLanguage
        )
        return try await generateFromReference(
            text: text,
            reference: reference,
            tokenizer: tokenizer,
            generationParameters: generationParameters
        )
    }

    public func generateStream(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        let (stream, continuation) = AsyncThrowingStream<AudioGeneration, Error>.makeStream()
        Task { @Sendable [weak self] in
            guard let self else {
                continuation.finish(throwing: AudioGenerationError.modelNotInitialized("Model deallocated"))
                return
            }
            do {
                let audio = try await self.generate(
                    text: text,
                    voice: voice,
                    refAudio: refAudio,
                    refText: refText,
                    language: language,
                    generationParameters: generationParameters
                )
                continuation.yield(.audio(audio))
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }
        return stream
    }

    // MARK: - ConditionedSpeechGenerationModel

    /// Synthesise with pre-computed reference conditioning (skips encoder/aligner).
    /// When `conditioning` is nil or its `namedAssets` are empty the call falls
    /// through to the standard `generate(refAudio:)` path.
    public func generate(
        text: String,
        voice: String?,
        conditioning: SpeechConditioning?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) async throws -> MLXArray {
        if let conditioning,
           conditioning.format == "tada.reference/v1",
           !conditioning.namedAssets.isEmpty {
            let reference = try deserializeConditioning(conditioning)
            return try await synthesizeWithPrecomputedConditioning(
                text: text,
                reference: reference,
                language: language,
                generationParameters: generationParameters
            )
        }
        return try await generate(
            text: text,
            voice: voice,
            refAudio: refAudio,
            refText: refText,
            language: language,
            generationParameters: generationParameters
        )
    }

    public func generateStream(
        text: String,
        voice: String?,
        conditioning: SpeechConditioning?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        let (stream, continuation) = AsyncThrowingStream<AudioGeneration, Error>.makeStream()
        Task { @Sendable [weak self] in
            guard let self else {
                continuation.finish(throwing: AudioGenerationError.modelNotInitialized("Model deallocated"))
                return
            }
            do {
                let audio = try await self.generate(
                    text: text,
                    voice: voice,
                    conditioning: conditioning,
                    refAudio: refAudio,
                    refText: refText,
                    language: language,
                    generationParameters: generationParameters
                )
                continuation.yield(.audio(audio))
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }
        return stream
    }

    public func generateStream(
        text: String,
        voice: String?,
        conditioning: SpeechConditioning?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters,
        streamingInterval: Double
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        _ = streamingInterval
        return generateStream(
            text: text,
            voice: voice,
            conditioning: conditioning,
            refAudio: refAudio,
            refText: refText,
            language: language,
            generationParameters: generationParameters
        )
    }

    public func generateCodeStream(
        text: String,
        voice: String?,
        conditioning: SpeechConditioning?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) -> AsyncThrowingStream<MLXArray, Error> {
        AsyncThrowingStream { $0.finish() }
    }

    // MARK: - Reference conditioning extraction

    /// Runs the TADA encoder and aligner on reference audio and returns the
    /// pre-computed conditioning so it can be cached and reused.
    /// - Parameters:
    ///   - audio: Mono reference audio at the model's native sample rate (24 kHz).
    ///   - referenceTranscript: Transcript of the reference audio.
    ///   - language: Optional language tag (e.g. "en"). Defaults to English.
    public func extractReferenceConditioning(
        audio: MLXArray,
        referenceTranscript: String,
        language: String?
    ) throws -> TADAReferenceConditioningData {
        guard let tokenizer else {
            throw AudioGenerationError.modelNotInitialized("Tokenizer not loaded")
        }
        let flat = audio.ndim == 1 ? audio : audio.reshaped(-1)
        let resolvedLanguage = tadaNormalizeLanguage(language)
        let ref = try buildReferenceConditioning(
            tokenizer: tokenizer,
            text: referenceTranscript,
            refAudio: flat,
            refText: referenceTranscript,
            language: resolvedLanguage
        )
        return TADAReferenceConditioningData(
            tokenValues: ref.tokenValues,
            tokenPositions: ref.tokenPositions,
            tokenMask: ref.tokenMask,
            textTokens: ref.textTokens,
            frameCount: ref.frameCount,
            transcript: ref.transcript
        )
    }

    public static func fromPretrained(
        _ modelRepo: String,
        cache: HubCache = .default
    ) async throws -> TADATTSModel {
        let fileManager = FileManager.default
        if fileManager.fileExists(atPath: modelRepo) {
            return try await fromDirectory(URL(fileURLWithPath: modelRepo))
        }

        guard let repoID = Repo.ID(rawValue: modelRepo) else {
            throw AudioGenerationError.invalidInput("Invalid model repository ID: \(modelRepo)")
        }

        let modelDir = try await ModelUtils.resolveOrDownloadModel(
            repoID: repoID,
            requiredExtension: "safetensors",
            additionalMatchingPatterns: [
                "model/config.json",
                "model/*.safetensors",
                "encoder/*.safetensors",
                "decoder/*.safetensors",
                "aligner/*.safetensors",
                "tokenizer.json",
            ],
            cache: cache
        )
        return try await fromDirectory(modelDir)
    }

    public static func fromDirectory(_ modelDir: URL) async throws -> TADATTSModel {
        let tokenizerURL = modelDir.appendingPathComponent("tokenizer.json")
        guard FileManager.default.fileExists(atPath: tokenizerURL.path) else {
            throw AudioGenerationError.modelNotInitialized(
                "Missing tokenizer.json at \(tokenizerURL.path). Reinstall the HumeAI/mlx-tada-* snapshot or materialize tokenizer.json into the local pack."
            )
        }

        let config = try tadaReadConfig(from: modelDir)
        let model = TADATTSModel(config: config)
        model.tokenizer = try await AutoTokenizer.from(modelFolder: modelDir)

        let weights = try tadaLoadWeights(from: modelDir)
        let sanitized = sanitize(weights: weights)

        if sanitized.keys.contains(where: { $0.hasSuffix(".scales") }) {
            quantize(
                model: model,
                filter: { path, _ in
                    guard path.hasPrefix("model.") || path.hasPrefix("prediction_head.") else { return nil }
                    return sanitized["\(path).scales"] != nil ? (64, 4, .affine) : nil
                }
            )
        }

        try model.update(parameters: ModuleParameters.unflattened(sanitized), verify: [])
        eval(model.parameters())
        return model
    }

    public static func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]

        for (rawKey, value) in weights {
            let key = rawKey.hasPrefix("module.") ? String(rawKey.dropFirst("module.".count)) : rawKey
            if key == "lm_head.weight" || key == "model.lm_head.weight" {
                continue
            }
            if key == "acoustic_proj.bias" {
                // Public TADA checkpoints may include this bias tensor even though the Swift port
                // models acoustic_proj as a bias-free projection, matching the port spec.
                continue
            }

            // ── Encoder submodule weights (encoder/weights.safetensors) ──────────────
            // Python stores these without any "encoder." prefix; add it so the Swift
            // TADAReferenceEncoder (@ModuleInfo key:"encoder") can find them.
            if key.hasPrefix("wav_encoder.")
                || key.hasPrefix("local_attention_encoder.")
                || key.hasPrefix("hidden_linear.")
                || key == "pos_emb.weight"
            {
                sanitized["encoder.\(key)"] = value
                continue
            }

            // ── Decoder submodule weights (decoder/weights.safetensors) ──────────────
            // Python keys: decoder_proj.*, local_attention_decoder.*, wav_decoder.*
            if key.hasPrefix("decoder_proj.")
                || key.hasPrefix("local_attention_decoder.")
                || key.hasPrefix("wav_decoder.")
            {
                sanitized["decoder.\(key)"] = value
                continue
            }

            // ── Aligner submodule weights (aligner/weights.safetensors) ──────────────
            // Python Aligner wraps Wav2Vec2ForCTC under self.wav2vec2.
            // Maps wav2vec2.feature_extractor.* → aligner.feature_extractor.*
            //      wav2vec2.feature_projection.projection.* → aligner.input_projection.*
            //      wav2vec2.encoder.* → aligner.encoder.*
            //      wav2vec2.lm_head.* → aligner.lm_head.*
            if key.hasPrefix("wav2vec2.feature_extractor.") {
                let suffix = key.dropFirst("wav2vec2.feature_extractor.".count)
                sanitized["aligner.feature_extractor.\(suffix)"] = value
                continue
            }
            if key.hasPrefix("wav2vec2.feature_projection.projection.") {
                let suffix = key.dropFirst("wav2vec2.feature_projection.projection.".count)
                sanitized["aligner.input_projection.\(suffix)"] = value
                continue
            }
            if key.hasPrefix("wav2vec2.encoder.") {
                let suffix = key.dropFirst("wav2vec2.encoder.".count)
                sanitized["aligner.encoder.\(suffix)"] = value
                continue
            }
            if key.hasPrefix("wav2vec2.lm_head.") {
                let suffix = key.dropFirst("wav2vec2.lm_head.".count)
                sanitized["aligner.lm_head.\(suffix)"] = value
                continue
            }

            // ── Main model / prediction-head weights ─────────────────────────────────
            if key.hasPrefix("model.")
                || key.hasPrefix("encoder.")
                || key.hasPrefix("decoder.")
                || key.hasPrefix("aligner.")
                || key.hasPrefix("prediction_head.")
                || key.hasPrefix("acoustic_proj.")
                || key.hasPrefix("time_start_embed.")
                || key.hasPrefix("time_end_embed.")
                || key.hasPrefix("acoustic_mask_emb.")
            {
                // Fix Python mlp_0/mlp_2 (direct attributes) → Swift mlp.0/mlp.2 (array indices)
                var renamedKey = key
                    .replacingOccurrences(of: "t_embedder.mlp_0.", with: "t_embedder.mlp.0.")
                    .replacingOccurrences(of: "t_embedder.mlp_2.", with: "t_embedder.mlp.2.")
                // Fix Python adaLN_modulation_linear → Swift adaLN_modulation (no _linear suffix)
                renamedKey = renamedKey.replacingOccurrences(
                    of: "adaLN_modulation_linear.", with: "adaLN_modulation.")
                sanitized[renamedKey] = value
                continue
            }

            if key.hasPrefix("t_embedder.mlp.0.") {
                sanitized["prediction_head.t_embedder.mlp.0.\(key.split(separator: ".").last!)"] = value
                continue
            }
            if key.hasPrefix("t_embedder.mlp.2.") {
                sanitized["prediction_head.t_embedder.mlp.2.\(key.split(separator: ".").last!)"] = value
            }
        }

        return sanitized
    }

    // MARK: - Autoregressive generation

    private struct TADABuildInputsResult {
        var inputIDs: [Int32]           // full token sequence (mutated during AR loop)
        let paf: MLXArray               // [1, P, acousticDim] — prompt acoustic features
        let pam: MLXArray               // [1, P] int32 — acoustic mask (forward-shifted)
        let tlb: MLXArray               // [1, P] int32 — time-before per prompt token
        let tla: MLXArray               // [1, P] int32 — time-after per prompt token
        let prefillLen: Int             // tokens to batch-prefill (0 = all AR)
        let promptLength: Int           // P = paf.dim(1)
        let numTransitionSteps: Int
    }

    /// Builds the full token sequence and per-position acoustic/timing tensors (Python §1).
    private func buildInputs(
        tokenizer: Tokenizer,
        reference: TADAReferenceConditioning,
        targetText: String,
        numTransitionSteps: Int = 5
    ) -> TADABuildInputsResult {
        let shift = config.shiftAcoustic
        let bosID = Int32(config.bosTokenId)
        let eotID = Int32(config.eotId)
        let padID = Int32(128_004)
        let startHeaderID = Int32(128_006)
        let endHeaderID = Int32(128_007)

        // §1a: Tokenize (target gets leading space per Python convention)
        let refTokenIDs = reference.textTokens
        let targetTokenIDs = tokenizer.encode(text: " " + targetText, addSpecialTokens: false).map { Int32($0) }

        // §1b: Time gap computation from reference token_positions (1-indexed 50Hz frames)
        let tokenPositions = reference.tokenPositions
        let tRef = tokenPositions.count
        // timeGaps[0] = 0 sentinel, timeGaps[i+1] = gap at position i
        var timeGaps = [Int32](repeating: 0, count: tRef + 1)
        for i in 0..<tRef {
            let prev = i == 0 ? 1 : tokenPositions[i - 1]
            timeGaps[i + 1] = Int32(min(max(tokenPositions[i] - prev, 0), 255))
        }
        let tlbRaw = Array(timeGaps[0..<tRef])
        let tlaRaw = Array(timeGaps[1...tRef])

        // §1c: Reference acoustic arrays
        var pafArray = reference.tokenValues  // [1, T_ref, acousticDim]
        var pamArray = MLXArray(Array(repeating: Int32(1), count: tRef)).reshaped(1, tRef)
        var tlbArray = MLXArray(tlbRaw).reshaped(1, tRef)
        var tlaArray = MLXArray(tlaRaw).reshaped(1, tRef)

        // §1d: Prefix tokens (system/assistant header)
        let prefixText = "<|start_header_id|>system<|end_header_id|><|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        let prefixIDs = tokenizer.encode(text: prefixText, addSpecialTokens: false).map { Int32($0) }
        let prefixLen = prefixIDs.count

        // §1e: Pad acoustic arrays with zeros for prefix tokens
        if prefixLen > 0 {
            let zeroPadAc = MLXArray.zeros([1, prefixLen, config.acousticDim], dtype: pafArray.dtype)
            pafArray = MLX.concatenated([zeroPadAc, pafArray], axis: 1)
            let zeroPad = MLXArray.zeros([1, prefixLen], dtype: .int32)
            pamArray = MLX.concatenated([zeroPad, pamArray], axis: 1)
            tlbArray = MLX.concatenated([zeroPad, tlbArray], axis: 1)
            tlaArray = MLX.concatenated([zeroPad, tlaArray], axis: 1)
        }

        // §1f: Truncate last numTransitionSteps frames
        let rawLen = pafArray.dim(1)
        let keepLen = max(0, rawLen - numTransitionSteps)
        if numTransitionSteps > 0 && keepLen < rawLen && keepLen > 0 {
            pafArray = pafArray[0..., ..<keepLen, 0...]
            pamArray = pamArray[0..., ..<keepLen]
            tlbArray = tlbArray[0..., ..<keepLen]
            tlaArray = tlaArray[0..., ..<keepLen]
        } else if keepLen == 0 {
            pafArray = MLXArray.zeros([1, 0, config.acousticDim], dtype: .float32)
            pamArray = MLXArray.zeros([1, 0], dtype: .int32)
            tlbArray = MLXArray.zeros([1, 0], dtype: .int32)
            tlaArray = MLXArray.zeros([1, 0], dtype: .int32)
        }
        let P = pafArray.dim(1)

        // §1g: Acoustic mask forward shift: pam = [pam[:, 1:], ones[:, :1]]
        if P > 0 {
            pamArray = MLX.concatenated([pamArray[0..., 1...], MLXArray([Int32(1)]).reshaped(1, 1)], axis: 1)
        }

        // §1h: Full input_ids: [BOS, prefix..., ref_tokens..., target_tokens..., eot*shift]
        var fullInputIDs = [bosID] + prefixIDs + refTokenIDs + targetTokenIDs + Array(repeating: eotID, count: shift)

        // §1i: Mask non-structural prompt tokens with padID
        var cumStart = 0
        var cumEnd = 0
        for i in 0..<min(P, fullInputIDs.count) {
            let id = fullInputIDs[i]
            if id == startHeaderID { cumStart += 1 }
            if id == endHeaderID { cumEnd += 1 }
            let depth = cumStart - cumEnd
            let inHeader = depth > 0 || id == startHeaderID || id == endHeaderID
            let isStructural = inHeader || id == eotID || id == bosID || id == Int32(128_001)
            if !isStructural { fullInputIDs[i] = padID }
        }

        // §1j: prefill_len computation
        let prefillTokensLen = 1 + prefixLen + tRef
        let nAc = min(prefillTokensLen - shift - 1, P)
        let nT = min(prefillTokensLen - shift - 1, max(0, P - 1))
        let nFramesCap = max(0, P - 2)
        let nPrefill = (nAc > 0 && nT > 0) ? min(nAc, min(nT, nFramesCap)) : 0
        let prefillLen = nPrefill > 0 ? min(prefillTokensLen, shift + nPrefill + 1) : 0

        return TADABuildInputsResult(
            inputIDs: fullInputIDs, paf: pafArray, pam: pamArray, tlb: tlbArray, tla: tlaArray,
            prefillLen: prefillLen, promptLength: P, numTransitionSteps: numTransitionSteps)
    }

    /// Batch-prefill the KV cache with both positive and negative paths (Python §2).
    private func prefill(inputs: TADABuildInputsResult) -> [TADAKVCache] {
        let caches = (0..<config.numHiddenLayers).map { _ in TADAKVCache() }
        let pfLen = inputs.prefillLen
        guard pfLen > 0 else { return caches }

        let shift = config.shiftAcoustic
        let P = inputs.promptLength

        // Acoustic arrays: positions 0..<pfLen, with shift+1 offset for acoustic frames
        let nAc = min(pfLen - shift - 1, P)
        let acousticFull: MLXArray
        let masksFull: MLXArray
        if nAc > 0 {
            let leftAc = MLXArray.zeros([1, shift + 1, config.acousticDim], dtype: .float32)
            let midAc = inputs.paf[0..., 0..<nAc, 0...]
            let right = pfLen - (shift + 1) - nAc
            acousticFull = right > 0
                ? MLX.concatenated([leftAc, midAc, MLXArray.zeros([1, right, config.acousticDim], dtype: .float32)], axis: 1)
                : MLX.concatenated([leftAc, midAc], axis: 1)
            let leftM = MLXArray.zeros([1, shift + 1], dtype: .int32)
            let midM = inputs.pam[0..., 0..<nAc]
            masksFull = right > 0
                ? MLX.concatenated([leftM, midM, MLXArray.zeros([1, right], dtype: .int32)], axis: 1)
                : MLX.concatenated([leftM, midM], axis: 1)
        } else {
            acousticFull = MLXArray.zeros([1, pfLen, config.acousticDim], dtype: .float32)
            masksFull = MLXArray.zeros([1, pfLen], dtype: .int32)
        }

        let nT = min(pfLen - shift - 1, max(0, P - 1))
        let timeBefore: MLXArray
        let timeAfter: MLXArray
        if nT > 0 {
            let leftT = MLXArray.zeros([1, shift + 1], dtype: .int32)
            let midTlb = inputs.tlb[0..., 1..<(1 + nT)]
            let midTla = inputs.tla[0..., 1..<(1 + nT)]
            let right = pfLen - (shift + 1) - nT
            timeBefore = right > 0
                ? MLX.concatenated([leftT, midTlb, MLXArray.zeros([1, right], dtype: .int32)], axis: 1)
                : MLX.concatenated([leftT, midTlb], axis: 1)
            timeAfter = right > 0
                ? MLX.concatenated([leftT, midTla, MLXArray.zeros([1, right], dtype: .int32)], axis: 1)
                : MLX.concatenated([leftT, midTla], axis: 1)
        } else {
            timeBefore = MLXArray.zeros([1, pfLen], dtype: .int32)
            timeAfter = MLXArray.zeros([1, pfLen], dtype: .int32)
        }

        let tokenSlice = MLXArray(Array(inputs.inputIDs[0..<pfLen])).reshaped(1, pfLen).asType(.int32)
        let embeds = model.embedTokens(tokenSlice)
            + acousticProj(acousticFull)
            + acousticMaskEmb(masksFull)
            + timeStartEmbed(timeBefore)
            + timeEndEmbed(timeAfter)
        // Duplicate for batch=2 (both paths share identical prompt embeddings)
        let batchEmbeds = MLX.concatenated([embeds, embeds], axis: 0)  // [2, pfLen, H]
        let dummyIDs = MLXArray.zeros([2, pfLen], dtype: .int32)
        let _ = model.callAsFunction(inputIds: dummyIDs, inputsEmbeds: batchEmbeds, caches: caches)
        return caches
    }

    /// Per-token autoregressive generation loop (Python §3).
    private func autoregressiveLoop(
        inputs: TADABuildInputsResult,
        caches: [TADAKVCache],
        generationParameters: GenerateParameters
    ) -> (acoustic: [MLXArray], timeBefore: [MLXArray]) {
        let shift = config.shiftAcoustic
        let padID = Int32(128_004)
        let startHeaderID = Int32(128_006)
        let endHeaderID = Int32(128_007)
        // Python treats 128001 (end_of_text), 128008, and 128009 (eot_id) all as EOS.
        // The config.json only has 128001; add eotId explicitly so the loop stops at natural EOT.
        let eosSet = Set(config.eosTokenId.map { Int32($0) } + [Int32(config.eotId), Int32(128_008)])
        let P = inputs.promptLength
        let nPf = inputs.prefillLen > 0 ? inputs.prefillLen - shift : 0

        // Initial acoustic feedback state
        var acousticFeatVal: MLXArray
        var acousticMaskVal: MLXArray
        var timeBeforeVal: MLXArray
        var timeAfterVal: MLXArray

        if nPf > 0 {
            acousticFeatVal = inputs.paf[0..., (nPf - 1)..<nPf, 0...]
            acousticMaskVal = inputs.pam[0..., (nPf - 1)..<nPf]
            if nPf < inputs.tlb.dim(1) {
                timeBeforeVal = inputs.tlb[0..., nPf..<(nPf + 1)]
                timeAfterVal = inputs.tla[0..., nPf..<(nPf + 1)]
            } else {
                timeBeforeVal = MLXArray.zeros([1, 1], dtype: .int32)
                timeAfterVal = MLXArray.zeros([1, 1], dtype: .int32)
            }
        } else {
            acousticFeatVal = MLXArray.zeros([1, 1, config.acousticDim], dtype: .float32)
            acousticMaskVal = MLXArray.zeros([1, 1], dtype: .int32)
            timeBeforeVal = MLXArray.zeros([1, 1], dtype: .int32)
            timeAfterVal = MLXArray.zeros([1, 1], dtype: .int32)
        }

        // Pre-fill output with reference frames already processed in prefill
        var allAcoustic: [MLXArray] = []
        var allTimeBefore: [MLXArray] = []
        for i in 0..<nPf {
            allAcoustic.append(inputs.paf[0..., i..<(i + 1), 0...])
            let tbIdx = i + 1
            allTimeBefore.append(tbIdx < inputs.tlb.dim(1)
                ? inputs.tlb[0..., tbIdx..<(tbIdx + 1)]
                : MLXArray.zeros([1, 1], dtype: .int32))
        }

        var inputIDs = inputs.inputIDs
        var lastTimeBefore: MLXArray? = nil
        let temperature = generationParameters.temperature
        let topP = generationParameters.topP
        let repPenalty = generationParameters.repetitionPenalty ?? 1.1

        for step in inputs.prefillLen..<inputIDs.count {
            // Build positive and negative token slices
            let stepID = inputIDs[step]
            let isStructural = stepID == startHeaderID || stepID == endHeaderID
                || stepID == Int32(config.eotId)
            let negID = isStructural ? stepID : padID
            let posSlice = MLXArray([stepID]).reshaped(1, 1).asType(.int32)
            let negSlice = MLXArray([negID]).reshaped(1, 1).asType(.int32)

            // Batch-2 embeddings (pos + neg)
            let c2Slice = MLX.concatenated([posSlice, negSlice], axis: 0)
            let c2Ac = MLX.concatenated([acousticFeatVal, acousticFeatVal], axis: 0)
            let c2M = MLX.concatenated([acousticMaskVal, acousticMaskVal], axis: 0)
            let c2Tb = MLX.concatenated([timeBeforeVal, timeBeforeVal], axis: 0)
            let c2Ta = MLX.concatenated([timeAfterVal, timeAfterVal], axis: 0)

            let stepEmbeds = model.embedTokens(c2Slice)
                + acousticProj(c2Ac) + acousticMaskEmb(c2M)
                + timeStartEmbed(c2Tb) + timeEndEmbed(c2Ta)

            let (hidden, logits) = model.forwardWithLogits(
                inputIds: c2Slice, inputsEmbeds: stepEmbeds, caches: caches)
            let posHidden = hidden[0..<1, 0..., 0...]
            let negHidden = hidden[1..<2, 0..., 0...]
            let stepLogits = logits[0..<1, 0..., 0...]  // [1, 1, V]

            // VibeVoice: [1, latentSize]
            let speech = predictionHead.solve(
                cond: posHidden, negativeCond: negHidden,
                steps: 20, acousticCFGScale: 1.6, durationCFGScale: 1.0, noiseTemperature: 0.9)

            // Decode Gray-code time bits (speech is [batch, latentSize] — 2D)
            let beforeBits = speech[0..., config.acousticDim..<(config.acousticDim + config.numTimeBits)]
            let afterBits = speech[0..., (config.acousticDim + config.numTimeBits)...]
            let predTb = TADAGrayCodeDurationCodec.decode(beforeBits.reshaped(1, config.numTimeBits))
                             .reshaped(1, 1).asType(.int32)
            let predTa = TADAGrayCodeDurationCodec.decode(afterBits.reshaped(1, config.numTimeBits))
                             .reshaped(1, 1).asType(.int32)
            // DEBUG: log time prediction and speech stats
            let predTbVal = predTb.asArray(Int32.self).first ?? 0
            let speechAcousticMax = speech[0..., 0..<config.acousticDim].max().item(Float.self)
            FileHandle.standardError.write(Data("[TADA-DEBUG] step=\(step) predTb=\(predTbVal) speechAcMax=\(speechAcousticMax)\n".utf8))

            // Sample next token when beyond known sequence
            if step >= inputIDs.count - 1 {
                let nextToken = sampleToken(
                    logits: stepLogits.reshaped(config.vocabSize),
                    pastIDs: inputIDs, temperature: temperature, topP: topP, repPenalty: repPenalty)
                inputIDs.append(nextToken)
                if eosSet.contains(nextToken) { break }
            }

            // Acoustic feedback for next step
            if step >= shift {
                let refIdx = step - shift
                if refIdx < P {
                    acousticFeatVal = inputs.paf[0..., refIdx..<(refIdx + 1), 0...]
                    acousticMaskVal = inputs.pam[0..., refIdx..<(refIdx + 1)]
                } else {
                    acousticFeatVal = speech[0..., 0..<config.acousticDim].reshaped(1, 1, config.acousticDim)
                    acousticMaskVal = MLXArray([Int32(1)]).reshaped(1, 1)
                }
                allAcoustic.append(acousticFeatVal)

                if refIdx < inputs.tlb.dim(1) - 1 {
                    timeBeforeVal = inputs.tlb[0..., (refIdx + 1)..<(refIdx + 2)]
                    timeAfterVal = inputs.tla[0..., (refIdx + 1)..<(refIdx + 2)]
                } else {
                    timeBeforeVal = predTb
                    timeAfterVal = predTa
                }
                allTimeBefore.append(timeBeforeVal)
                lastTimeBefore = timeBeforeVal
            }
        }

        // Trailing time-before sentinel (decode_frames needs N+1 values for N frames)
        if let last = lastTimeBefore {
            allTimeBefore.append(last)
        } else if let last = allTimeBefore.last {
            allTimeBefore.append(last)
        }
        return (allAcoustic, allTimeBefore)
    }

    /// Decode accumulated acoustic frames to audio (Python §4).
    private func decodeOutput(
        acoustic: [MLXArray],
        timeBefore: [MLXArray],
        numPromptTokens: Int,
        numTransitionSteps: Int
    ) -> MLXArray {
        guard !acoustic.isEmpty else { return MLXArray([Float]()) }
        let allAcoustic = MLX.concatenated(acoustic, axis: 1)
        let allTimeBefore = MLX.concatenated(timeBefore, axis: 1)  // [1, N+1]

        let acousticFeatures = allAcoustic * MLXArray(config.acousticStd) + MLXArray(config.acousticMean)
        let skip = numPromptTokens + numTransitionSteps - 1
        let N = acousticFeatures.dim(1)
        guard skip < N else {
            return MLXArray([Float]())
        }

        let encoded = acousticFeatures[0..., skip..., 0...]
        let tbOut = allTimeBefore[0..., skip...]
        let tCount = encoded.dim(1)
        let tbArr = tbOut.reshaped(-1).asArray(Int32.self)
        let decoderTb = Array(tbArr[0..<tCount]).map { Int($0) }
        let trailing = tbArr.count > tCount ? Int(tbArr[tCount]) : 0
        let decoderTa = Array(repeating: 0, count: max(0, tCount - 1)) + [trailing]

        let waveform = decoder(acoustic: encoded, timeBefore: decoderTb, timeAfter: decoderTa)

        // Trim leading silence (first frame's time_before gap)
        let leadFrames = decoderTb.first ?? 0
        let leadSamples = Int(Float(sampleRate) * Float(leadFrames) / 50.0)
        if leadSamples > 0 && leadSamples < waveform.shape[0] {
            return waveform[leadSamples...]
        }
        return waveform
    }

    /// Top-p sampling with repetition penalty (Python §3d).
    private func sampleToken(
        logits: MLXArray,
        pastIDs: [Int32],
        temperature: Float,
        topP: Float,
        repPenalty: Float
    ) -> Int32 {
        let padID = 128_004
        var logitsArr = logits.asArray(Float.self)
        guard !logitsArr.isEmpty else { return Int32(config.eotId) }

        // Suppress pad token
        if padID < logitsArr.count { logitsArr[padID] = -1e9 }

        // Repetition penalty
        if repPenalty != 1.0 {
            for id in Set(pastIDs) {
                let idx = Int(id)
                guard idx >= 0 && idx < logitsArr.count else { continue }
                logitsArr[idx] = logitsArr[idx] < 0 ? logitsArr[idx] * repPenalty : logitsArr[idx] / repPenalty
            }
        }

        // Temperature
        if temperature != 1.0 { for i in logitsArr.indices { logitsArr[i] /= temperature } }

        // Top-p nucleus filter
        let sortedIdx = logitsArr.indices.sorted { logitsArr[$0] > logitsArr[$1] }
        let maxL = logitsArr[sortedIdx[0]]
        let expVals = sortedIdx.map { Foundation.exp(logitsArr[$0] - maxL) }
        let expSum = expVals.reduce(Float(0), +)
        var cumProb: Float = 0
        for (rank, origIdx) in sortedIdx.enumerated() {
            let prob = expVals[rank] / expSum
            cumProb += prob
            if cumProb - prob >= topP { logitsArr[origIdx] = -1e9 }
        }

        // Categorical sample
        let probs = softmax(MLXArray(logitsArr), axis: 0)
        let sample = MLX.categorical(log(MLX.maximum(probs, MLXArray(Float(1e-12)))))
        return Int32(sample.item(Int.self))
    }

    /// Common AR generation path used by both generate() variants.
    private func generateFromReference(
        text: String,
        reference: TADAReferenceConditioning,
        tokenizer: Tokenizer,
        generationParameters: GenerateParameters
    ) async throws -> MLXArray {
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw AudioGenerationError.invalidInput("Input text is empty.")
        }
        let inputs = buildInputs(tokenizer: tokenizer, reference: reference, targetText: text)
        let caches = prefill(inputs: inputs)
        let (acoustic, tbFrames) = autoregressiveLoop(inputs: inputs, caches: caches, generationParameters: generationParameters)
        return decodeOutput(acoustic: acoustic, timeBefore: tbFrames,
                            numPromptTokens: inputs.promptLength, numTransitionSteps: inputs.numTransitionSteps)
    }

    private func buildReferenceConditioning(
        tokenizer: Tokenizer,
        text: String,
        refAudio: MLXArray,
        refText: String?,
        language: String?
    ) throws -> TADAReferenceConditioning {
        let transcript: String
        if let refText, !refText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            transcript = refText
        } else if language == nil || language == "en" {
            transcript = text
        } else {
            throw AudioGenerationError.invalidInput(
                "Transcript required for non-English TADA cloning. Provide --reference-transcript or install a speech recognition model."
            )
        }

        let cleanedTranscript = transcript.trimmingCharacters(in: .whitespacesAndNewlines)
        let tokenIDs = tokenizer.encode(text: cleanedTranscript, addSpecialTokens: false).map { Int32($0) }
        guard !tokenIDs.isEmpty else {
            throw AudioGenerationError.invalidInput("Reference transcript is empty after tokenization.")
        }

        let audio24k = refAudio.ndim == 1 ? refAudio : refAudio.reshaped(-1)
        let audio16k = try resampleAudio(audio24k, from: sampleRate, to: 16_000)
        let alignment = aligner.align(audio16k: audio16k, tokenIds: tokenIDs)
        let tokenValues = encoder.encode(audio24k: audio24k, tokenPositions: alignment.positions, tokenMask: alignment.mask)

        return TADAReferenceConditioning(
            tokenValues: tokenValues,
            tokenPositions: alignment.positions,
            tokenMask: alignment.mask,
            textTokens: tokenIDs,
            frameCount: alignment.frameCount,
            transcript: cleanedTranscript
        )
    }

    // MARK: - Private helpers for conditioned synthesis

    /// Minimal decodable shim for the JSON metadata payload in a TADA SpeechConditioning.
    private struct TADAConditioningManifestShim: Decodable {
        let token_count: Int
        let acoustic_dim: Int
        let frame_count: Int?
        let transcript: String
    }

    /// Deserialise a `SpeechConditioning` (format "tada.reference/v1") back into
    /// a `TADAReferenceConditioning` so synthesis can skip the encoder/aligner.
    private func deserializeConditioning(_ conditioning: SpeechConditioning) throws -> TADAReferenceConditioning {
        let manifest = try JSONDecoder().decode(TADAConditioningManifestShim.self, from: conditioning.payload)
        let tokenCount = manifest.token_count
        let acousticDim = manifest.acoustic_dim
        let frameCount = manifest.frame_count ?? 0

        guard let valuesData = conditioning.namedAssets["token_values.f16"] else {
            throw AudioGenerationError.invalidInput("Missing token_values.f16 in TADA conditioning payload.")
        }
        guard let positionsData = conditioning.namedAssets["token_positions.i32"] else {
            throw AudioGenerationError.invalidInput("Missing token_positions.i32 in TADA conditioning payload.")
        }
        guard let textTokensData = conditioning.namedAssets["text_tokens.i32"] else {
            throw AudioGenerationError.invalidInput("Missing text_tokens.i32 in TADA conditioning payload.")
        }

        // token_values: float16 binary → [1, tokenCount, acousticDim] float32
        let valuesCount = valuesData.count / 2
        guard valuesCount == tokenCount * acousticDim else {
            throw AudioGenerationError.invalidInput(
                "token_values size mismatch: expected \(tokenCount * acousticDim), got \(valuesCount)"
            )
        }
        let f16Values: [Float16] = valuesData.withUnsafeBytes { raw in
            (0..<valuesCount).map { i in raw.load(fromByteOffset: i * 2, as: Float16.self) }
        }
        let tokenValues = MLXArray(f16Values).asType(.float32).reshaped(1, tokenCount, acousticDim)

        // token_positions: int32 binary → [Int]
        let posCount = positionsData.count / 4
        let tokenPositions: [Int] = positionsData.withUnsafeBytes { raw in
            (0..<posCount).map { i in Int(raw.load(fromByteOffset: i * 4, as: Int32.self)) }
        }

        // text_tokens: int32 binary → [Int32]
        let ttCount = textTokensData.count / 4
        let textTokens: [Int32] = textTokensData.withUnsafeBytes { raw in
            (0..<ttCount).map { i in raw.load(fromByteOffset: i * 4, as: Int32.self) }
        }

        // token_masks: uint8 binary (optional)
        let tokenMask: [UInt8] = conditioning.namedAssets["token_masks.u8"].map { Array($0) } ?? []

        return TADAReferenceConditioning(
            tokenValues: tokenValues,
            tokenPositions: tokenPositions,
            tokenMask: tokenMask,
            textTokens: textTokens,
            frameCount: frameCount,
            transcript: manifest.transcript
        )
    }

    /// Run synthesis from a pre-computed reference, bypassing the encoder/aligner.
    private func synthesizeWithPrecomputedConditioning(
        text: String,
        reference: TADAReferenceConditioning,
        language: String?,
        generationParameters: GenerateParameters
    ) async throws -> MLXArray {
        guard let tokenizer else {
            throw AudioGenerationError.modelNotInitialized("Tokenizer not loaded")
        }
        return try await generateFromReference(
            text: text, reference: reference, tokenizer: tokenizer,
            generationParameters: generationParameters)
    }
}
