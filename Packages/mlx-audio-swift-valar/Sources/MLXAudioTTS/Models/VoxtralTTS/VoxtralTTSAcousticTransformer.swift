import Foundation
@preconcurrency import MLX
import MLXNN

private let voxtralTTSAcousticSwiGLU: @Sendable (MLXArray, MLXArray) -> MLXArray = {
    compile(shapeless: true) { gate, up in
        silu(gate) * up
    }
}()

private func voxtralTTSSinusoidalTimeEmbedding(
    tValue: Float,
    dim: Int,
    theta: Float = 10_000
) -> MLXArray {
    let halfDim = dim / 2
    let invFreq = MLX.exp(
        -log(theta) * MLXArray(0..<halfDim).asType(.float32) / Float(halfDim)
    )
    let emb = tValue * invFreq
    return concatenated([MLX.cos(emb), MLX.sin(emb)], axis: 0).expandedDimensions(axis: 0)
}

final class VoxtralTTSAcousticFeedForward: Module {
    @ModuleInfo(key: "w1") var w1: Linear
    @ModuleInfo(key: "w2") var w2: Linear
    @ModuleInfo(key: "w3") var w3: Linear

    init(_ config: VoxtralTTSAcousticTransformerConfig) {
        _w1.wrappedValue = Linear(config.dim, config.hiddenDim, bias: false)
        _w2.wrappedValue = Linear(config.hiddenDim, config.dim, bias: false)
        _w3.wrappedValue = Linear(config.dim, config.hiddenDim, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        w2(voxtralTTSAcousticSwiGLU(w1(x), w3(x)))
    }
}

final class VoxtralTTSAcousticAttention: Module {
    let nHeads: Int
    let nKvHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "wq") var wq: Linear
    @ModuleInfo(key: "wk") var wk: Linear
    @ModuleInfo(key: "wv") var wv: Linear
    @ModuleInfo(key: "wo") var wo: Linear

    init(_ config: VoxtralTTSAcousticTransformerConfig) {
        nHeads = config.nHeads
        nKvHeads = config.nKvHeads
        headDim = config.headDim
        scale = pow(Float(config.headDim), -0.5)

        _wq.wrappedValue = Linear(config.dim, config.nHeads * config.headDim, bias: false)
        _wk.wrappedValue = Linear(config.dim, config.nKvHeads * config.headDim, bias: false)
        _wv.wrappedValue = Linear(config.dim, config.nKvHeads * config.headDim, bias: false)
        _wo.wrappedValue = Linear(config.nHeads * config.headDim, config.dim, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let batch = x.dim(0)
        let seqLen = x.dim(1)

        let q = wq(x).reshaped(batch, seqLen, nHeads, headDim).transposed(0, 2, 1, 3)
        let k = wk(x).reshaped(batch, seqLen, nKvHeads, headDim).transposed(0, 2, 1, 3)
        let v = wv(x).reshaped(batch, seqLen, nKvHeads, headDim).transposed(0, 2, 1, 3)

        let output = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: k,
            values: v,
            scale: scale,
            mask: .none
        )

        return wo(output.transposed(0, 2, 1, 3).reshaped(batch, seqLen, nHeads * headDim))
    }
}

final class VoxtralTTSAcousticTransformerLayer: Module {
    @ModuleInfo(key: "attention_norm") var attentionNorm: RMSNorm
    @ModuleInfo(key: "attention") var attention: VoxtralTTSAcousticAttention
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm
    @ModuleInfo(key: "feed_forward") var feedForward: VoxtralTTSAcousticFeedForward

    init(_ config: VoxtralTTSAcousticTransformerConfig) {
        _attentionNorm.wrappedValue = RMSNorm(dimensions: config.dim, eps: 1e-5)
        _attention.wrappedValue = VoxtralTTSAcousticAttention(config)
        _ffnNorm.wrappedValue = RMSNorm(dimensions: config.dim, eps: 1e-5)
        _feedForward.wrappedValue = VoxtralTTSAcousticFeedForward(config)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = x + attention(attentionNorm(x))
        return h + feedForward(ffnNorm(h))
    }
}

public final class VoxtralTTSAcousticTransformer: Module {
    public static let eulerSteps = 7
    public static let cfgAlpha: Float = 1.2
    public static let noiseScale: Float = 1.0

    public let config: VoxtralTTSAcousticTransformerConfig
    public let semanticCodebookSize: Int
    public let acousticCodebookSize: Int
    public let nAcousticCodebook: Int

    @ModuleInfo(key: "semantic_codebook_output") var semanticCodebookOutput: Linear
    @ModuleInfo(key: "acoustic_codebook_output") var acousticCodebookOutput: Linear
    @ModuleInfo(key: "time_projection") var timeProjection: Linear
    @ModuleInfo(key: "input_projection") var inputProjection: Linear
    @ModuleInfo(key: "llm_projection") var llmProjection: Linear
    @ModuleInfo(key: "layers") var layers: [VoxtralTTSAcousticTransformerLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    public init(
        config: VoxtralTTSAcousticTransformerConfig,
        semanticCodebookSize: Int,
        acousticCodebookSize: Int,
        nAcousticCodebook: Int
    ) {
        self.config = config
        self.semanticCodebookSize = semanticCodebookSize
        self.acousticCodebookSize = acousticCodebookSize
        self.nAcousticCodebook = nAcousticCodebook

        _semanticCodebookOutput.wrappedValue = Linear(config.dim, (semanticCodebookSize / 128 + 1) * 128, bias: false)
        _acousticCodebookOutput.wrappedValue = Linear(config.dim, nAcousticCodebook, bias: false)
        _timeProjection.wrappedValue = Linear(config.dim, config.dim, bias: false)
        _inputProjection.wrappedValue = Linear(nAcousticCodebook, config.dim, bias: false)
        _llmProjection.wrappedValue = Linear(config.inputDim, config.dim, bias: false)
        _layers.wrappedValue = (0..<config.nLayers).map { _ in
            VoxtralTTSAcousticTransformerLayer(config)
        }
        _norm.wrappedValue = RMSNorm(dimensions: config.dim, eps: 1e-5)
    }

    func predictFrame(from llmHidden: MLXArray) -> (codes: MLXArray, semanticCode: Int, isEOS: Bool) {
        let llmProjected = llmProjection(llmHidden)
        var semanticLogits = semanticCodebookOutput(llmHidden)

        let vocabRange = MLXArray(0..<semanticLogits.dim(-1)).asType(.int32)
        let allowedMask = (vocabRange .> MLXArray(Int32(0)))
            .&& (vocabRange .< MLXArray(Int32(semanticCodebookSize + 2)))
        let additiveMask = MLX.where(allowedMask, MLXArray(0.0), MLXArray(-1e9))
            .expandedDimensions(axis: 0)
        semanticLogits = semanticLogits + additiveMask

        let semanticCodeArray = argMax(semanticLogits, axis: -1)
        let semanticCode = semanticCodeArray.item(Int.self)
        let isEOS = semanticCode == 1

        let acousticCodes = decodeOneFrame(llmCondition: llmProjected)
        let frameCodes = concatenated([semanticCodeArray.expandedDimensions(axis: 1), acousticCodes], axis: 1)
        return (frameCodes, semanticCode, isEOS)
    }

    func decodeOneFrame(llmCondition: MLXArray) -> MLXArray {
        let batch = llmCondition.dim(0)
        let zerosCondition = MLXArray.zeros([batch, config.dim], dtype: llmCondition.dtype)
        let dt = Float(1.0 / Float(Self.eulerSteps))
        var x = Self.noiseScale * MLXRandom.normal([batch, nAcousticCodebook])

        for step in 0..<Self.eulerSteps {
            let tValue = Float(step) * dt
            let baseTimeEmbedding = voxtralTTSSinusoidalTimeEmbedding(
                tValue: tValue,
                dim: config.dim,
                theta: 10_000
            )
            let timeEmbedding = timeProjection(
                broadcast(baseTimeEmbedding, to: [batch, config.dim])
            )

            let xBatch = concatenated([x, x], axis: 0)
            let llmBatch = concatenated([llmCondition, zerosCondition], axis: 0)
            let tBatch = concatenated([timeEmbedding, timeEmbedding], axis: 0)

            let allVelocity = predictVelocity(
                xT: xBatch,
                llmOutput: llmBatch,
                timeEmbedding: tBatch
            )
            let vCond = allVelocity[..<batch]
            let vUncond = allVelocity[batch...]
            let guided = Self.cfgAlpha * vCond + (1.0 - Self.cfgAlpha) * vUncond
            x = x + guided * dt
        }

        let xClamped = clip(x, min: -1.0, max: 1.0)
        let scaled = (xClamped + 1.0) / 2.0 * Float(acousticCodebookSize - 1)
        return round(scaled).asType(.int32) + MLXArray(Int32(2))
    }

    func predictVelocity(
        xT: MLXArray,
        llmOutput: MLXArray,
        timeEmbedding: MLXArray
    ) -> MLXArray {
        let xProjected = inputProjection(xT)
        var tokens = concatenated(
            [
                xProjected.expandedDimensions(axis: 1),
                timeEmbedding.expandedDimensions(axis: 1),
                llmOutput.expandedDimensions(axis: 1),
            ],
            axis: 1
        )

        for layer in layers {
            tokens = layer(tokens)
        }

        tokens = norm(tokens)
        return acousticCodebookOutput(tokens[0..., 0, 0...])
    }
}
