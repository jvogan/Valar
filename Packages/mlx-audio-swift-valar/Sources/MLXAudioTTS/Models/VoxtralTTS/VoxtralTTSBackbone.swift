import Foundation
import MLX
import MLXNN

struct VoxtralTTSBackboneKVCache {
    var keys: MLXArray   // [kvLen, nKvHeads * headDim]
    var values: MLXArray // [kvLen, nKvHeads * headDim]
    var positionOffset: Int
}

private func voxtralTTSComputeRopeFrequencies(
    positions: MLXArray,
    headDim: Int,
    theta: Float
) -> (cos: MLXArray, sin: MLXArray) {
    let idx = MLXArray(stride(from: 0, to: headDim, by: 2)).asType(.float32)
    let invFreq = MLX.exp((-log(theta)) * (idx / Float(headDim)))
    let angles = positions.asType(.float32).expandedDimensions(axis: 1) * invFreq.expandedDimensions(axis: 0)
    return (MLX.cos(angles), MLX.sin(angles))
}

private func voxtralTTSApplyInterleavedRoPE(
    _ x: MLXArray,
    cos: MLXArray,
    sin: MLXArray,
    nHeads: Int,
    headDim: Int
) -> MLXArray {
    let seqLen = x.shape[0]
    let halfDim = headDim / 2

    let reshaped = x.reshaped(seqLen, nHeads, halfDim, 2)
    let x1 = reshaped[0..., 0..., 0..., 0]
    let x2 = reshaped[0..., 0..., 0..., 1]

    let cosE = cos.expandedDimensions(axis: 1)
    let sinE = sin.expandedDimensions(axis: 1)

    let o1 = x1 * cosE - x2 * sinE
    let o2 = x2 * cosE + x1 * sinE

    let out = concatenated(
        [o1.expandedDimensions(axis: -1), o2.expandedDimensions(axis: -1)],
        axis: -1
    )
    return out.reshaped(seqLen, nHeads * headDim)
}

private let voxtralTTSCompiledSwiGLU: @Sendable (MLXArray, MLXArray) -> MLXArray = {
    compile(shapeless: true) { gate, up in
        silu(gate) * up
    }
}()

final class VoxtralTTSBackboneFeedForward: Module {
    @ModuleInfo(key: "w1") var w1: Linear
    @ModuleInfo(key: "w2") var w2: Linear
    @ModuleInfo(key: "w3") var w3: Linear

    init(_ config: VoxtralTTSBackboneConfig) {
        _w1.wrappedValue = Linear(config.dim, config.hiddenDim, bias: false)
        _w2.wrappedValue = Linear(config.hiddenDim, config.dim, bias: false)
        _w3.wrappedValue = Linear(config.dim, config.hiddenDim, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        w2(voxtralTTSCompiledSwiGLU(w1(x), w3(x)))
    }
}

final class VoxtralTTSBackboneAttention: Module {
    let nHeads: Int
    let nKvHeads: Int
    let headDim: Int
    let ropeTheta: Float
    let scale: Float

    @ModuleInfo(key: "wq") var wq: Linear
    @ModuleInfo(key: "wk") var wk: Linear
    @ModuleInfo(key: "wv") var wv: Linear
    @ModuleInfo(key: "wo") var wo: Linear

    init(_ config: VoxtralTTSBackboneConfig) {
        nHeads = config.nHeads
        nKvHeads = config.nKvHeads
        headDim = config.headDim
        ropeTheta = config.ropeTheta
        scale = pow(Float(config.headDim), -0.5)

        let qDim = config.nHeads * config.headDim
        let kvDim = config.nKvHeads * config.headDim

        _wq.wrappedValue = Linear(config.dim, qDim, bias: false)
        _wk.wrappedValue = Linear(config.dim, kvDim, bias: false)
        _wv.wrappedValue = Linear(config.dim, kvDim, bias: false)
        _wo.wrappedValue = Linear(qDim, config.dim, bias: false)
    }

    func callAsFunction(
        _ x: MLXArray,
        positions: MLXArray,
        cache: VoxtralTTSBackboneKVCache?
    ) -> (MLXArray, VoxtralTTSBackboneKVCache) {
        let seqLen = x.shape[0]

        var q = wq(x)
        var k = wk(x)
        var v = wv(x)

        let (cos, sin) = voxtralTTSComputeRopeFrequencies(
            positions: positions,
            headDim: headDim,
            theta: ropeTheta
        )
        q = voxtralTTSApplyInterleavedRoPE(q, cos: cos, sin: sin, nHeads: nHeads, headDim: headDim)
        k = voxtralTTSApplyInterleavedRoPE(k, cos: cos, sin: sin, nHeads: nKvHeads, headDim: headDim)

        let positionOffset = cache?.positionOffset ?? 0
        if let cache {
            // TODO: O(N²) per AR step — each step copies the entire accumulated KV cache.
            // Fix: pre-allocate keys/values to [maxSteps, kvDim] once (not maxSeqLen which
            // defaults to 65536 and would exhaust RAM), then scatter-write new tokens at
            // writeOffset via cache.keys.at[offset..<(offset+seqLen)].add(k) and pass only
            // the slice [0..<writeOffset] to attention — making per-step work O(seqLen).
            k = concatenated([cache.keys, k], axis: 0)
            v = concatenated([cache.values, v], axis: 0)
        }

        let kvLen = k.shape[0]
        let newCache = VoxtralTTSBackboneKVCache(keys: k, values: v, positionOffset: positionOffset)

        let q4 = q.reshaped(1, seqLen, nHeads, headDim).transposed(0, 2, 1, 3)
        let k4 = k.reshaped(1, kvLen, nKvHeads, headDim).transposed(0, 2, 1, 3)
        let v4 = v.reshaped(1, kvLen, nKvHeads, headDim).transposed(0, 2, 1, 3)

        let maskMode: MLXFast.ScaledDotProductAttentionMaskMode
        if seqLen == 1 && cache == nil {
            maskMode = .none
        } else if cache == nil {
            maskMode = .causal
        } else {
            let qPos = positions.expandedDimensions(axis: 1)
            let kPos = MLXArray(positionOffset..<(positionOffset + kvLen))
                .asType(.int32)
                .expandedDimensions(axis: 0)
            let allowed = kPos .<= qPos
            let mask = MLX.where(allowed, MLXArray(0.0), MLXArray(-1e9))
            maskMode = .array(mask)
        }

        let attn = MLXFast.scaledDotProductAttention(
            queries: q4,
            keys: k4,
            values: v4,
            scale: scale,
            mask: maskMode
        )
        let out = attn.transposed(0, 2, 1, 3).reshaped(seqLen, nHeads * headDim)
        return (wo(out), newCache)
    }
}

final class VoxtralTTSBackboneLayer: Module {
    @ModuleInfo(key: "attention_norm") var attentionNorm: RMSNorm
    @ModuleInfo(key: "attention") var attention: VoxtralTTSBackboneAttention
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm
    @ModuleInfo(key: "feed_forward") var feedForward: VoxtralTTSBackboneFeedForward

    init(_ config: VoxtralTTSBackboneConfig) {
        _attentionNorm.wrappedValue = RMSNorm(dimensions: config.dim, eps: config.normEps)
        _attention.wrappedValue = VoxtralTTSBackboneAttention(config)
        _ffnNorm.wrappedValue = RMSNorm(dimensions: config.dim, eps: config.normEps)
        _feedForward.wrappedValue = VoxtralTTSBackboneFeedForward(config)
    }

    func callAsFunction(
        _ x: MLXArray,
        positions: MLXArray,
        cache: VoxtralTTSBackboneKVCache?
    ) -> (MLXArray, VoxtralTTSBackboneKVCache) {
        let attn = attention(attentionNorm(x), positions: positions, cache: cache)
        let h = x + attn.0
        return (h + feedForward(ffnNorm(h)), attn.1)
    }
}

public final class VoxtralTTSBackbone: Module {
    public let config: VoxtralTTSBackboneConfig

    @ModuleInfo(key: "tok_embeddings") var tokEmbeddings: Embedding
    @ModuleInfo(key: "layers") var layers: [VoxtralTTSBackboneLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    public init(_ config: VoxtralTTSBackboneConfig) {
        self.config = config
        _tokEmbeddings.wrappedValue = Embedding(
            embeddingCount: config.vocabSize,
            dimensions: config.dim
        )
        _layers.wrappedValue = (0..<config.nLayers).map { _ in
            VoxtralTTSBackboneLayer(config)
        }
        _norm.wrappedValue = RMSNorm(dimensions: config.dim, eps: config.normEps)
    }

    func embedToken(_ tokenId: Int) -> MLXArray {
        tokEmbeddings.weight[tokenId]
    }

    func embedTokens(_ tokenIds: MLXArray) -> MLXArray {
        tokEmbeddings(tokenIds)
    }

    func callAsFunction(
        _ embeds: MLXArray,
        startPos: Int,
        cache: [VoxtralTTSBackboneKVCache?]? = nil
    ) -> (MLXArray, [VoxtralTTSBackboneKVCache?]) {
        var h = embeds
        let seqLen = h.shape[0]
        let positions = MLXArray(startPos..<(startPos + seqLen)).asType(.int32)

        var newCache: [VoxtralTTSBackboneKVCache?] = []
        newCache.reserveCapacity(layers.count)

        for layerIndex in layers.indices {
            let layerCache = cache?[layerIndex]
            let next = layers[layerIndex](h, positions: positions, cache: layerCache)
            h = next.0
            newCache.append(next.1)
        }

        return (norm(h), newCache)
    }

    func logits(_ hiddenStates: MLXArray) -> MLXArray {
        matmul(hiddenStates, tokEmbeddings.weight.transposed(1, 0))
    }
}
