// Copyright (c) 2025, Prince Canuma and contributors
// Swift port for ValarTTS

import Foundation
@preconcurrency import MLX
import MLXNN

// MARK: - RMSNorm

/// Root Mean Square Layer Normalization for VibeVoice backbone.
final class VibeVoiceRMSNorm: Module {
    let weight: MLXArray
    let eps: Float

    init(dims: Int, eps: Float = 1e-6) {
        self.weight = MLXArray.ones([dims])
        self.eps = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: weight, eps: eps)
    }
}

// MARK: - Rotary Embedding

/// Rotary Position Embedding (RoPE) for VibeVoice's Qwen2 attention.
final class VibeVoiceRotaryEmbedding: Module {
    let dim: Int
    let maxPositionEmbeddings: Int
    let base: Float

    init(dim: Int, maxPositionEmbeddings: Int = 8192, base: Float = 1_000_000.0) {
        self.dim = dim
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.base = base
        super.init()
    }

    /// Compute cos and sin for rotary embeddings at given positions.
    func callAsFunction(_ positionIds: MLXArray) -> (cos: MLXArray, sin: MLXArray) {
        let t = positionIds.asType(.float32)
        let invFreq = 1.0 / MLX.pow(
            MLXArray(base),
            MLX.arange(0, dim, step: 2).asType(.float32) / Float(dim)
        )
        // freqs: (L, dim/2)
        let freqs = MLX.outer(t, invFreq)
        // Concatenate to get full dimension: (L, dim)
        let emb = MLX.concatenated([freqs, freqs], axis: -1)
        return (cos: MLX.cos(emb), sin: MLX.sin(emb))
    }
}

// MARK: - RoPE Application

private func rotateHalf(_ x: MLXArray) -> MLXArray {
    let half = x.dim(-1) / 2
    let x1 = x[.ellipsis, ..<half]
    let x2 = x[.ellipsis, half...]
    return MLX.concatenated([-x2, x1], axis: -1)
}

/// Apply rotary position embeddings to query and key tensors.
/// - q: (B, L, H, D), k: (B, L, H_kv, D)
/// - cos, sin: (L, D) — will be broadcast to (1, L, 1, D)
private func applyRotaryPosEmb(
    q: MLXArray, k: MLXArray,
    cos: MLXArray, sin: MLXArray
) -> (MLXArray, MLXArray) {
    // Expand: (L, D) -> (1, L, 1, D)
    let cosExp = cos.expandedDimensions(axes: [0, 2])
    let sinExp = sin.expandedDimensions(axes: [0, 2])
    let qEmbed = (q * cosExp) + (rotateHalf(q) * sinExp)
    let kEmbed = (k * cosExp) + (rotateHalf(k) * sinExp)
    return (qEmbed, kEmbed)
}

// MARK: - Attention

/// Multi-head attention with grouped query attention (GQA) for VibeVoice.
///
/// Q/K/V projections have bias=True, O projection has bias=False (Qwen2 convention).
final class VibeVoiceAttention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo var qProj: Linear
    @ModuleInfo var kProj: Linear
    @ModuleInfo var vProj: Linear
    @ModuleInfo var oProj: Linear

    init(config: VibeVoiceQwen2DecoderConfig) {
        let headDim = config.effectiveHeadDim
        self.numHeads = config.numAttentionHeads
        self.numKVHeads = config.numKeyValueHeads
        self.headDim = headDim
        self.scale = 1.0 / sqrtf(Float(headDim))

        _qProj.wrappedValue = Linear(config.hiddenSize, numHeads * headDim, bias: true)
        _kProj.wrappedValue = Linear(config.hiddenSize, numKVHeads * headDim, bias: true)
        _vProj.wrappedValue = Linear(config.hiddenSize, numKVHeads * headDim, bias: true)
        _oProj.wrappedValue = Linear(numHeads * headDim, config.hiddenSize, bias: false)

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        cos: MLXArray,
        sin: MLXArray,
        mask: MLXArray? = nil,
        cache: (key: MLXArray, value: MLXArray)? = nil
    ) -> (MLXArray, (key: MLXArray, value: MLXArray)) {
        let B = x.dim(0)
        let L = x.dim(1)

        var q = qProj(x).reshaped(B, L, numHeads, headDim)
        var k = kProj(x).reshaped(B, L, numKVHeads, headDim)
        let v = vProj(x).reshaped(B, L, numKVHeads, headDim)

        // Apply rotary embeddings
        (q, k) = applyRotaryPosEmb(q: q, k: k, cos: cos, sin: sin)

        // KV cache concatenation
        var kFull = k
        var vFull = v
        if let cache {
            kFull = MLX.concatenated([cache.key, k], axis: 1)
            vFull = MLX.concatenated([cache.value, v], axis: 1)
        }
        let newCache = (key: kFull, value: vFull)

        // Transpose for attention: (B, L, H, D) -> (B, H, L, D)
        let qT = q.transposed(0, 2, 1, 3)
        let kT = kFull.transposed(0, 2, 1, 3)
        let vT = vFull.transposed(0, 2, 1, 3)

        let out = MLXFast.scaledDotProductAttention(
            queries: qT, keys: kT, values: vT, scale: scale, mask: mask
        )

        // Reshape back: (B, H, L, D) -> (B, L, H*D)
        let reshaped = out.transposed(0, 2, 1, 3).reshaped(B, L, -1)
        return (oProj(reshaped), newCache)
    }
}

// MARK: - MLP (SwiGLU)

/// Feed-forward network with SwiGLU activation for VibeVoice Qwen2 backbone.
final class VibeVoiceMLP: Module {
    @ModuleInfo var gateProj: Linear
    @ModuleInfo var upProj: Linear
    @ModuleInfo var downProj: Linear

    init(config: VibeVoiceQwen2DecoderConfig) {
        _gateProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

// MARK: - Decoder Layer

/// A single transformer decoder layer for VibeVoice.
final class VibeVoiceDecoderLayer: Module {
    @ModuleInfo var selfAttn: VibeVoiceAttention
    @ModuleInfo var mlp: VibeVoiceMLP
    @ModuleInfo var inputLayernorm: VibeVoiceRMSNorm
    @ModuleInfo var postAttentionLayernorm: VibeVoiceRMSNorm

    init(config: VibeVoiceQwen2DecoderConfig) {
        _selfAttn.wrappedValue = VibeVoiceAttention(config: config)
        _mlp.wrappedValue = VibeVoiceMLP(config: config)
        _inputLayernorm.wrappedValue = VibeVoiceRMSNorm(dims: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayernorm.wrappedValue = VibeVoiceRMSNorm(dims: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        cos: MLXArray,
        sin: MLXArray,
        mask: MLXArray? = nil,
        cache: (key: MLXArray, value: MLXArray)? = nil
    ) -> (MLXArray, (key: MLXArray, value: MLXArray)) {
        // Self attention with residual
        var residual = x
        var h: MLXArray
        var newCache: (key: MLXArray, value: MLXArray)
        (h, newCache) = selfAttn(inputLayernorm(x), cos: cos, sin: sin, mask: mask, cache: cache)
        var out = residual + h

        // MLP with residual
        residual = out
        h = mlp(postAttentionLayernorm(out))
        out = residual + h

        return (out, newCache)
    }
}

// MARK: - SpeechConnector

/// Projects speech latents from acoustic VAE dimension to LM hidden size.
final class VibeVoiceSpeechConnector: Module {
    @ModuleInfo var fc1: Linear
    @ModuleInfo var norm: VibeVoiceRMSNorm
    @ModuleInfo var fc2: Linear

    init(inputDim: Int, outputDim: Int, eps: Float = 1e-6) {
        _fc1.wrappedValue = Linear(inputDim, outputDim)
        _norm.wrappedValue = VibeVoiceRMSNorm(dims: outputDim, eps: eps)
        _fc2.wrappedValue = Linear(outputDim, outputDim)
        super.init()
    }

    func callAsFunction(_ features: MLXArray) -> MLXArray {
        fc2(norm(fc1(features)))
    }
}

// MARK: - Binary Classifier

/// Binary classifier for TTS end-of-speech detection (sigmoid > 0.5 = EOS).
final class VibeVoiceBinaryClassifier: Module {
    @ModuleInfo var fc1: Linear
    @ModuleInfo var fc2: Linear

    init(hiddenSize: Int) {
        _fc1.wrappedValue = Linear(hiddenSize, hiddenSize)
        _fc2.wrappedValue = Linear(hiddenSize, 1)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(relu(fc1(x)))
    }
}

// MARK: - Qwen2 Model

/// Qwen2 transformer model used by VibeVoice for both `language_model` (lower layers)
/// and `tts_language_model` (upper layers).
///
/// - `useNorm`: If true, applies a final RMSNorm after all layers (TTS LM uses this).
///   The base LM does not apply final norm since it feeds directly into TTS LM.
final class VibeVoiceQwen2Model: Module {
    let config: VibeVoiceQwen2DecoderConfig
    let useNorm: Bool

    @ModuleInfo var embedTokens: Embedding?
    @ModuleInfo(key: "layers") var layers: [VibeVoiceDecoderLayer]
    @ModuleInfo var norm: VibeVoiceRMSNorm?
    let rotaryEmb: VibeVoiceRotaryEmbedding

    init(config: VibeVoiceQwen2DecoderConfig, useNorm: Bool = true) {
        self.config = config
        self.useNorm = useNorm

        if config.vocabSize > 0 {
            _embedTokens.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        } else {
            _embedTokens.wrappedValue = nil
        }

        _layers.wrappedValue = (0..<config.numHiddenLayers).map { _ in
            VibeVoiceDecoderLayer(config: config)
        }

        if useNorm {
            _norm.wrappedValue = VibeVoiceRMSNorm(dims: config.hiddenSize, eps: config.rmsNormEps)
        } else {
            _norm.wrappedValue = nil
        }

        let headDim = config.effectiveHeadDim
        self.rotaryEmb = VibeVoiceRotaryEmbedding(
            dim: headDim,
            maxPositionEmbeddings: config.maxPositionEmbeddings,
            base: config.ropeTheta
        )

        super.init()
    }

    /// Forward pass through the Qwen2 model.
    ///
    /// - Parameters:
    ///   - inputsEmbeds: Pre-computed embeddings, shape (B, L, D). Mutually exclusive with inputIds.
    ///   - inputIds: Token IDs, shape (B, L). Used if inputsEmbeds is nil.
    ///   - cache: Per-layer KV cache from previous steps.
    /// - Returns: (hiddenStates, newCaches) — hidden states of shape (B, L, D) and per-layer KV caches.
    func callAsFunction(
        inputsEmbeds: MLXArray? = nil,
        inputIds: MLXArray? = nil,
        cache: [(key: MLXArray, value: MLXArray)?]? = nil
    ) -> (MLXArray, [(key: MLXArray, value: MLXArray)]) {
        let embeds: MLXArray
        if let inputsEmbeds {
            embeds = inputsEmbeds
        } else if let inputIds, let embed = embedTokens {
            embeds = embed(inputIds)
        } else {
            fatalError("VibeVoiceQwen2Model: must provide either inputsEmbeds or inputIds")
        }

        let L = embeds.dim(1)

        // Compute position offset from cache
        var offset = 0
        if let cache, let first = cache.first, let kv = first {
            offset = kv.key.dim(1)
        }

        // Position IDs and rotary embeddings
        let positionIds = MLX.arange(offset, offset + L).asType(.int32)
        let (cos, sin) = rotaryEmb(positionIds)

        // Create causal mask when sequence length > 1
        var mask: MLXArray? = nil
        if L > 1 {
            let kLen = offset + L
            // qPos: (L, 1), kPos: (1, K)
            let qPos = MLX.arange(offset, offset + L).asType(.int32).expandedDimensions(axis: 1)
            let kPos = MLX.arange(0, kLen).asType(.int32).expandedDimensions(axis: 0)
            let allow = qPos .>= kPos
            let negInf = MLXArray(Float(-1e9))
            let zero = MLXArray(Float(0))
            let causalMask = MLX.which(allow, zero, negInf)
            mask = causalMask.expandedDimensions(axes: [0, 1])  // (1, 1, L, K)
        }

        var h = embeds
        var newCaches: [(key: MLXArray, value: MLXArray)] = []
        newCaches.reserveCapacity(layers.count)

        for (i, layer) in layers.enumerated() {
            let layerCache = cache?[i]
            let (out, c) = layer(h, cos: cos, sin: sin, mask: mask, cache: layerCache)
            h = out
            newCaches.append(c)
        }

        if let norm {
            h = norm(h)
        }
        return (h, newCaches)
    }
}
