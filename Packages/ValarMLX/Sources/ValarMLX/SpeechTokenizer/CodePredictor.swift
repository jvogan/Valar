import Foundation
@preconcurrency import MLX
import MLXNN

// MARK: - Configuration

/// Configuration for the code predictor transformer used in the Qwen3-TTS talker.
///
/// Default values match the Qwen3-TTS code predictor config:
/// 5 layers, 16 query heads with 8 KV heads (GQA), 1024 hidden size,
/// SwiGLU MLP with 3072 intermediate size, RoPE with theta=1_000_000,
/// QK norms, and no sliding window.
///
/// Reference: `config.py` class `Qwen3TTSTalkerCodePredictorConfig`.
struct CodePredictorConfig: Sendable {
    let vocabSize: Int
    let hiddenSize: Int
    let intermediateSize: Int
    let numHiddenLayers: Int
    let numAttentionHeads: Int
    let numKeyValueHeads: Int
    let headDim: Int
    let rmsNormEps: Float
    let ropeTheta: Float
    let attentionBias: Bool
    let numCodeGroups: Int

    static let `default` = CodePredictorConfig(
        vocabSize: 2048,
        hiddenSize: 1024,
        intermediateSize: 3072,
        numHiddenLayers: 5,
        numAttentionHeads: 16,
        numKeyValueHeads: 8,
        headDim: 128,
        rmsNormEps: 1e-6,
        ropeTheta: 1_000_000.0,
        attentionBias: false,
        numCodeGroups: 16
    )

    /// Number of codebook groups the predictor handles (groups 1 through numCodeGroups-1).
    var numPredictedGroups: Int { numCodeGroups - 1 }
}

enum CodePredictorError: Error, LocalizedError, Sendable, Equatable {
    case invalidGenerationStep(requested: Int, validRange: Range<Int>)

    var errorDescription: String? {
        switch self {
        case .invalidGenerationStep(let requested, let validRange):
            return "Invalid code predictor generation step \(requested). Expected a value in \(validRange.lowerBound)..<\(validRange.upperBound)."
        }
    }
}

// MARK: - CodePredictorKVCache

/// Stateful key-value cache for autoregressive decoding.
///
/// Stores accumulated key and value tensors across decode steps so that
/// previously computed KV pairs are not recomputed. Each layer gets its
/// own cache instance.
final class CodePredictorKVCache: @unchecked Sendable {
    var keys: MLXArray?
    var values: MLXArray?
    private(set) var offset: Int = 0

    /// Append new K/V slices and return the full accumulated K/V tensors.
    ///
    /// - Parameters:
    ///   - newKeys: Keys for the current step, shape `[batch, heads, seqLen, headDim]`.
    ///   - newValues: Values for the current step, shape `[batch, heads, seqLen, headDim]`.
    /// - Returns: Tuple of accumulated `(keys, values)`.
    func updateAndFetch(
        keys newKeys: MLXArray,
        values newValues: MLXArray
    ) -> (MLXArray, MLXArray) {
        if let existingKeys = keys, let existingValues = values {
            keys = concatenated([existingKeys, newKeys], axis: 2)
            values = concatenated([existingValues, newValues], axis: 2)
        } else {
            keys = newKeys
            values = newValues
        }
        offset += newKeys.dim(2)
        return (keys!, values!)
    }
}

// MARK: - CodePredictorRMSNorm

/// RMS normalization for the code predictor transformer.
///
/// Uses the MLX fast RMS norm kernel internally. The learnable `weight`
/// parameter scales the normalized output per-channel.
///
/// Reference: `talker.py` RMSNorm usage in CodePredictorAttention / CodePredictorDecoderLayer.
final class CodePredictorRMSNorm: Module {
    var weight: MLXArray
    let eps: Float

    init(dimensions: Int, eps: Float = 1e-6) {
        self.weight = MLXArray.ones([dimensions])
        self.eps = eps
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: weight, eps: eps)
    }
}

// MARK: - CodePredictorMLP

/// SwiGLU MLP for the code predictor transformer.
///
/// Computes `down_proj(silu(gate_proj(x)) * up_proj(x))`. All three
/// linear layers are bias-free.
///
/// Reference: `talker.py` class `CodePredictorMLP`.
final class CodePredictorMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(config: CodePredictorConfig) {
        _gateProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

// MARK: - CodePredictorAttention

/// Multi-head attention for the code predictor with GQA and QK norms.
///
/// Supports grouped-query attention (16 query heads, 8 KV heads by default).
/// Applies per-head RMS normalization to queries and keys before RoPE.
/// Uses `MLXFast.scaledDotProductAttention` with built-in GQA support.
///
/// Reference: `talker.py` class `CodePredictorAttention`.
final class CodePredictorAttention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: CodePredictorRMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: CodePredictorRMSNorm

    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float
    let ropeTheta: Float

    init(config: CodePredictorConfig) {
        self.numHeads = config.numAttentionHeads
        self.numKVHeads = config.numKeyValueHeads
        self.headDim = config.headDim
        self.scale = pow(Float(config.headDim), -0.5)
        self.ropeTheta = config.ropeTheta

        let qDim = numHeads * headDim
        let kvDim = numKVHeads * headDim
        _qProj.wrappedValue = Linear(config.hiddenSize, qDim, bias: config.attentionBias)
        _kProj.wrappedValue = Linear(config.hiddenSize, kvDim, bias: config.attentionBias)
        _vProj.wrappedValue = Linear(config.hiddenSize, kvDim, bias: config.attentionBias)
        _oProj.wrappedValue = Linear(qDim, config.hiddenSize, bias: config.attentionBias)
        _qNorm.wrappedValue = CodePredictorRMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _kNorm.wrappedValue = CodePredictorRMSNorm(dimensions: headDim, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXArray?,
        cache: CodePredictorKVCache?,
        offset: Int
    ) -> MLXArray {
        let batch = x.dim(0)
        let seqLen = x.dim(1)

        // Project to Q/K/V, reshape to [batch, seq, heads, headDim]
        var q = qProj(x).reshaped(batch, seqLen, numHeads, headDim)
        var k = kProj(x).reshaped(batch, seqLen, numKVHeads, headDim)
        let v = vProj(x)
            .reshaped(batch, seqLen, numKVHeads, headDim)
            .transposed(0, 2, 1, 3)

        // Apply per-head QK norms
        q = qNorm(q)
        k = kNorm(k)

        // Transpose to [batch, heads, seq, headDim]
        q = q.transposed(0, 2, 1, 3)
        k = k.transposed(0, 2, 1, 3)

        // Apply RoPE
        q = applyRoPE(q, offset: offset)
        k = applyRoPE(k, offset: offset)

        // KV cache update
        var effectiveK = k
        var effectiveV = v
        if let cache {
            (effectiveK, effectiveV) = cache.updateAndFetch(keys: k, values: effectiveV)
        }

        // Scaled dot-product attention (handles GQA natively)
        let output = MLXFast.scaledDotProductAttention(
            queries: q, keys: effectiveK, values: effectiveV,
            scale: scale, mask: mask
        )

        return oProj(
            output.transposed(0, 2, 1, 3).reshaped(batch, seqLen, -1)
        )
    }

    /// Apply rotary position embeddings using the MLX optimized kernel.
    private func applyRoPE(_ x: MLXArray, offset: Int) -> MLXArray {
        let shape = x.shape
        var y = x.reshaped(-1, x.dim(-2), x.dim(-1))
        y = MLXFast.RoPE(
            y,
            dimensions: headDim,
            traditional: false,
            base: ropeTheta,
            scale: 1.0,
            offset: offset
        )
        return y.reshaped(shape)
    }
}

// MARK: - CodePredictorDecoderLayer

/// Single transformer layer: pre-norm attention + pre-norm MLP with
/// residual connections. No layer scale (unlike the speech tokenizer decoder).
///
/// Reference: `talker.py` class `CodePredictorDecoderLayer`.
final class CodePredictorDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: CodePredictorAttention
    @ModuleInfo var mlp: CodePredictorMLP
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: CodePredictorRMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: CodePredictorRMSNorm

    init(config: CodePredictorConfig) {
        _selfAttn.wrappedValue = CodePredictorAttention(config: config)
        _mlp.wrappedValue = CodePredictorMLP(config: config)
        _inputLayernorm.wrappedValue = CodePredictorRMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps
        )
        _postAttentionLayernorm.wrappedValue = CodePredictorRMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps
        )
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXArray?,
        cache: CodePredictorKVCache?,
        offset: Int
    ) -> MLXArray {
        // Self-attention with pre-norm residual
        var result = x
        let attnOut = selfAttn(inputLayernorm(result), mask: mask, cache: cache, offset: offset)
        result = result + attnOut

        // MLP with pre-norm residual
        let mlpOut = mlp(postAttentionLayernorm(result))
        result = result + mlpOut

        return result
    }
}

// MARK: - CodePredictorModel

/// Inner transformer model for the code predictor.
///
/// Contains the transformer layer stack, final norm, and codec embeddings.
/// Matches the weight key structure `code_predictor.model.*` from PyTorch.
///
/// Reference: `talker.py` class `CodePredictorModel`.
final class CodePredictorModel: Module {
    @ModuleInfo var layers: [CodePredictorDecoderLayer]
    @ModuleInfo var norm: CodePredictorRMSNorm
    @ModuleInfo(key: "codec_embedding") var codecEmbedding: [Embedding]

    let config: CodePredictorConfig

    init(config: CodePredictorConfig, talkerHiddenSize: Int) {
        self.config = config
        _layers.wrappedValue = (0 ..< config.numHiddenLayers).map { _ in
            CodePredictorDecoderLayer(config: config)
        }
        _norm.wrappedValue = CodePredictorRMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps
        )
        _codecEmbedding.wrappedValue = (0 ..< config.numPredictedGroups).map { _ in
            Embedding(embeddingCount: config.vocabSize, dimensions: talkerHiddenSize)
        }
    }

    /// Forward pass through the code predictor transformer.
    ///
    /// - Parameters:
    ///   - inputsEmbeds: Input embeddings with shape `[batch, seq, hiddenSize]`.
    ///   - mask: Optional additive attention mask. When `nil` and `seq > 1`,
    ///     a causal mask is created automatically.
    ///   - cache: Optional per-layer KV caches for autoregressive decoding.
    /// - Returns: Hidden states with shape `[batch, seq, hiddenSize]`.
    func callAsFunction(
        _ inputsEmbeds: MLXArray,
        mask: MLXArray? = nil,
        cache: [CodePredictorKVCache]? = nil
    ) -> MLXArray {
        let seqLen = inputsEmbeds.dim(1)

        // Compute offset from cache
        let offset: Int
        if let cache, let first = cache.first {
            offset = first.offset
        } else {
            offset = 0
        }

        // Auto-generate causal mask if not provided
        var effectiveMask = mask
        if effectiveMask == nil, seqLen > 1 {
            effectiveMask = MultiHeadAttention.createAdditiveCausalMask(seqLen)
                .asType(inputsEmbeds.dtype)
        }

        var x = inputsEmbeds

        for (i, layer) in layers.enumerated() {
            let layerCache = cache?[i]
            x = layer(x, mask: effectiveMask, cache: layerCache, offset: offset)
        }

        return norm(x)
    }

    /// Create a fresh set of KV caches, one per layer.
    func makeCache() -> [CodePredictorKVCache] {
        (0 ..< config.numHiddenLayers).map { _ in CodePredictorKVCache() }
    }
}

// MARK: - CodePredictor

/// Code predictor sub-model for multi-codebook token prediction.
///
/// Given the hidden state from codebook 0 prediction, iteratively predicts
/// codebooks 1 through 15. Each codebook group has its own embedding table
/// and linear LM head. The 5-layer transformer with GQA refines the
/// representation at each prediction step.
///
/// ## Architecture
///
/// - Optional linear projection from talker hidden size to code predictor hidden size
/// - 5-layer causal decoder transformer with GQA (16 Q heads, 8 KV heads)
/// - QK normalization per attention head
/// - SwiGLU MLP
/// - RoPE with theta = 1,000,000
/// - 15 per-codebook embedding tables and LM heads
///
/// ## Weight Loading
///
/// Weight keys follow the PyTorch structure:
/// ```
/// code_predictor.small_to_mtp_projection.{weight,bias}  (optional)
/// code_predictor.model.layers.N.self_attn.{q,k,v,o}_proj.weight
/// code_predictor.model.layers.N.self_attn.{q,k}_norm.weight
/// code_predictor.model.layers.N.{input_layernorm,post_attention_layernorm}.weight
/// code_predictor.model.layers.N.mlp.{gate,up,down}_proj.weight
/// code_predictor.model.norm.weight
/// code_predictor.model.codec_embedding.N.weight
/// code_predictor.lm_head.N.weight
/// ```
///
/// Reference: `talker.py` class `Qwen3TTSTalkerCodePredictor`.
final class CodePredictor: Module {
    @ModuleInfo var model: CodePredictorModel
    @ModuleInfo(key: "lm_head") var lmHead: [Linear]
    @ModuleInfo(key: "small_to_mtp_projection") var projection: Linear?

    let config: CodePredictorConfig
    let talkerHiddenSize: Int

    init(config: CodePredictorConfig = .default, talkerHiddenSize: Int = 1024) {
        self.config = config
        self.talkerHiddenSize = talkerHiddenSize

        _model.wrappedValue = CodePredictorModel(
            config: config, talkerHiddenSize: talkerHiddenSize
        )
        _lmHead.wrappedValue = (0 ..< config.numPredictedGroups).map { _ in
            Linear(config.hiddenSize, config.vocabSize, bias: false)
        }

        if config.hiddenSize != talkerHiddenSize {
            _projection.wrappedValue = Linear(talkerHiddenSize, config.hiddenSize, bias: true)
        } else {
            _projection.wrappedValue = nil
        }
    }

    /// Access codec embeddings from the inner model.
    var codecEmbedding: [Embedding] { model.codecEmbedding }

    static func validateGenerationStep(
        _ generationStep: Int,
        validRange: Range<Int>
    ) throws -> Int {
        guard validRange.contains(generationStep) else {
            throw CodePredictorError.invalidGenerationStep(
                requested: generationStep,
                validRange: validRange
            )
        }
        return generationStep
    }

    /// Forward pass: predict logits for a specific codebook group.
    ///
    /// - Parameters:
    ///   - inputsEmbeds: Input embeddings `[batch, seq, talkerHiddenSize]`.
    ///   - mask: Optional additive attention mask.
    ///   - cache: Optional per-layer KV caches.
    ///   - generationStep: Index into `lmHead` (0 = codebook 1, 14 = codebook 15).
    /// - Throws: ``CodePredictorError.invalidGenerationStep`` when `generationStep`
    ///   falls outside the available LM-head range.
    /// - Returns: Tuple of `(logits, cache, nextStep)` where logits has shape
    ///   `[batch, seq, vocabSize]`.
    func callAsFunction(
        inputsEmbeds: MLXArray,
        mask: MLXArray? = nil,
        cache: [CodePredictorKVCache]? = nil,
        generationStep: Int = 0
    ) throws -> (logits: MLXArray, cache: [CodePredictorKVCache]?, nextStep: Int) {
        let generationStep = try Self.validateGenerationStep(
            generationStep,
            validRange: lmHead.indices
        )

        var embeds = inputsEmbeds

        // Project if talker and predictor hidden sizes differ
        if let projection {
            embeds = projection(embeds)
        }

        // Forward through transformer
        let hidden = model(embeds, mask: mask, cache: cache)

        // Select the appropriate LM head for this codebook group
        let logits = lmHead[generationStep](hidden)

        return (logits, cache, generationStep + 1)
    }

    /// Create a fresh set of KV caches for autoregressive generation.
    func makeCache() -> [CodePredictorKVCache] {
        model.makeCache()
    }
}
