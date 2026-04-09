import Foundation
@preconcurrency import MLX
import MLXNN

// MARK: - Configuration

/// Configuration for the decoder transformer used in the speech tokenizer.
///
/// Default values match the Qwen3-TTS speech tokenizer decoder config:
/// 8 layers, 16 attention heads, 512 hidden size, SwiGLU MLP with 1024
/// intermediate size, RoPE with theta=10000, and a 72-token sliding window.
///
/// Reference: `config.py` class `Qwen3TTSTokenizerDecoderConfig`.
struct DecoderTransformerConfig: Sendable {
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
    let latentDim: Int
    let attentionBias: Bool

    static let `default` = DecoderTransformerConfig(
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
        latentDim: 1024,
        attentionBias: false
    )
}

// MARK: - DecoderRMSNorm

/// RMS normalization for the decoder transformer.
///
/// Uses the MLX fast RMS norm kernel internally. The learnable `weight`
/// parameter scales the normalized output per-channel.
///
/// Reference: `speech_tokenizer.py` class `DecoderRMSNorm`.
final class DecoderRMSNorm: Module {
    var weight: MLXArray
    let eps: Float

    init(hiddenSize: Int, eps: Float = 1e-5) {
        self.weight = MLXArray.ones([hiddenSize])
        self.eps = eps
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: weight, eps: eps)
    }
}

// MARK: - LayerScale

/// Per-channel scaling for residual connections.
///
/// Initialized to a small value (default 0.01) so that early in training
/// the residual path contributes minimally, improving stability.
///
/// Reference: `speech_tokenizer.py` class `LayerScale`.
final class LayerScale: Module {
    var scale: MLXArray

    init(channels: Int, initialScale: Float = 0.01) {
        self.scale = MLXArray.ones([channels]) * initialScale
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        scale * x
    }
}

// MARK: - DecoderMLP

/// SwiGLU MLP for the decoder transformer.
///
/// Computes `down_proj(silu(gate_proj(x)) * up_proj(x))`. All three
/// linear layers are bias-free.
///
/// Reference: `speech_tokenizer.py` class `DecoderMLP`.
final class DecoderMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(config: DecoderTransformerConfig) {
        _gateProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

// MARK: - DecoderAttention

/// Multi-head attention for the decoder transformer with RoPE.
///
/// Uses separate Q/K/V/O projections (no GQA in the default config since
/// `numKeyValueHeads == numAttentionHeads`). RoPE is applied via the
/// optimized `MLXFast.RoPE` kernel rather than manual cos/sin computation.
///
/// Reference: `speech_tokenizer.py` class `DecoderAttention`.
final class DecoderAttention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float
    let ropeTheta: Float

    init(config: DecoderTransformerConfig) {
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
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?, offset: Int) -> MLXArray {
        let batch = x.dim(0)
        let seqLen = x.dim(1)

        var q = qProj(x)
            .reshaped(batch, seqLen, numHeads, headDim)
            .transposed(0, 2, 1, 3)
        var k = kProj(x)
            .reshaped(batch, seqLen, numKVHeads, headDim)
            .transposed(0, 2, 1, 3)
        let v = vProj(x)
            .reshaped(batch, seqLen, numKVHeads, headDim)
            .transposed(0, 2, 1, 3)

        // Apply RoPE to queries and keys
        q = applyRoPE(q, offset: offset)
        k = applyRoPE(k, offset: offset)

        let output = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: mask
        )

        return oProj(
            output.transposed(0, 2, 1, 3).reshaped(batch, seqLen, -1)
        )
    }

    /// Apply rotary position embeddings using the MLX optimized kernel.
    ///
    /// Reshapes from `[batch, heads, seq, headDim]` to `[batch*heads, seq, headDim]`,
    /// applies RoPE, and reshapes back.
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

// MARK: - DecoderTransformerLayer

/// Single transformer layer: pre-norm attention + pre-norm MLP, both with
/// layer-scaled residual connections.
///
/// Reference: `speech_tokenizer.py` class `DecoderTransformerLayer`.
final class DecoderTransformerLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: DecoderAttention
    @ModuleInfo var mlp: DecoderMLP
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: DecoderRMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: DecoderRMSNorm
    @ModuleInfo(key: "self_attn_layer_scale") var selfAttnLayerScale: LayerScale
    @ModuleInfo(key: "mlp_layer_scale") var mlpLayerScale: LayerScale

    init(config: DecoderTransformerConfig) {
        _selfAttn.wrappedValue = DecoderAttention(config: config)
        _mlp.wrappedValue = DecoderMLP(config: config)
        _inputLayernorm.wrappedValue = DecoderRMSNorm(
            hiddenSize: config.hiddenSize, eps: config.rmsNormEps
        )
        _postAttentionLayernorm.wrappedValue = DecoderRMSNorm(
            hiddenSize: config.hiddenSize, eps: config.rmsNormEps
        )
        _selfAttnLayerScale.wrappedValue = LayerScale(
            channels: config.hiddenSize, initialScale: config.layerScaleInitialScale
        )
        _mlpLayerScale.wrappedValue = LayerScale(
            channels: config.hiddenSize, initialScale: config.layerScaleInitialScale
        )
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?, offset: Int) -> MLXArray {
        // Attention block with pre-norm and layer-scaled residual
        var result = x
        let attnOut = selfAttn(inputLayernorm(result), mask: mask, offset: offset)
        result = result + selfAttnLayerScale(attnOut)

        // MLP block with pre-norm and layer-scaled residual
        let mlpOut = mlp(postAttentionLayernorm(result))
        result = result + mlpLayerScale(mlpOut)

        return result
    }
}

// MARK: - DecoderTransformer

/// 8-layer causal transformer with sliding window attention for the speech
/// tokenizer decoder.
///
/// Projects input embeddings from `latentDim` to `hiddenSize`, applies
/// the transformer stack with RoPE and sliding-window causal masking,
/// normalizes, and projects back to `latentDim`.
///
/// The sliding window limits each position to attend only to the most
/// recent `slidingWindow` positions (default 72), reducing memory cost
/// for long sequences while preserving local context.
///
/// Reference: `speech_tokenizer.py` class `DecoderTransformer`.
final class DecoderTransformer: Module {
    @ModuleInfo var layers: [DecoderTransformerLayer]
    @ModuleInfo var norm: DecoderRMSNorm
    @ModuleInfo var inputProj: Linear
    @ModuleInfo var outputProj: Linear

    let config: DecoderTransformerConfig

    init(config: DecoderTransformerConfig = .default) {
        self.config = config
        _layers.wrappedValue = (0 ..< config.numHiddenLayers).map { _ in
            DecoderTransformerLayer(config: config)
        }
        _norm.wrappedValue = DecoderRMSNorm(
            hiddenSize: config.hiddenSize, eps: config.rmsNormEps
        )
        _inputProj.wrappedValue = Linear(config.latentDim, config.hiddenSize)
        _outputProj.wrappedValue = Linear(config.hiddenSize, config.latentDim)
    }

    /// Forward pass through the decoder transformer.
    ///
    /// - Parameters:
    ///   - inputsEmbeds: Input embeddings with shape `[batch, seq, latentDim]`.
    ///   - mask: Optional additive attention mask. When `nil` and `seq > 1`,
    ///     a sliding-window causal mask is created automatically.
    /// - Returns: Output embeddings with shape `[batch, seq, latentDim]`.
    func callAsFunction(_ inputsEmbeds: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let seqLen = inputsEmbeds.dim(1)

        var x = inputProj(inputsEmbeds)

        // Build sliding-window causal mask if none provided
        var effectiveMask = mask
        if effectiveMask == nil, seqLen > 1 {
            effectiveMask = Self.createSlidingWindowCausalMask(
                seqLen: seqLen,
                windowSize: config.slidingWindow,
                dtype: x.dtype
            )
        }

        for layer in layers {
            x = layer(x, mask: effectiveMask, offset: 0)
        }

        x = norm(x)
        return outputProj(x)
    }

    /// Create an additive causal mask with a sliding window constraint.
    ///
    /// Each query position can attend only to key positions satisfying:
    /// `0 <= queryPos - keyPos < windowSize`. Positions outside this range
    /// receive a large negative value (`-1e9`) so they are effectively
    /// zeroed out by softmax.
    ///
    /// - Parameters:
    ///   - seqLen: Sequence length.
    ///   - windowSize: Sliding window size (number of positions to attend to).
    ///   - dtype: Data type of the mask array.
    /// - Returns: Mask array with shape `[seqLen, seqLen]`.
    static func createSlidingWindowCausalMask(
        seqLen: Int,
        windowSize: Int,
        dtype: DType = .float32
    ) -> MLXArray {
        let indices = MLXArray(Int32(0) ..< Int32(seqLen))
        let queryIdx = expandedDimensions(indices, axis: 1) // [seq, 1]
        let keyIdx = expandedDimensions(indices, axis: 0)   // [1, seq]

        // Causal constraint: mask positions where key comes after query
        let causalMask = (queryIdx .< keyIdx).asType(dtype) * Float(-1e9)

        // Sliding window constraint: mask positions too far in the past
        let windowMask = ((queryIdx - keyIdx) .>= Int32(windowSize)).asType(dtype) * Float(-1e9)

        return causalMask + windowMask
    }
}
