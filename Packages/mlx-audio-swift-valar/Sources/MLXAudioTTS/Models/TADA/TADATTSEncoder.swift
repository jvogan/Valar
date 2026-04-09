import Foundation
@preconcurrency import MLX
import MLXFast
import MLXNN
import MLXAudioCodecs

internal let tadaSwiGLU: @Sendable (MLXArray, MLXArray) -> MLXArray = {
    compile(shapeless: true) { gate, up in
        silu(gate) * up
    }
}()

@inline(__always)
internal func tadaWeightNormMagnitude(_ weight: MLXArray, exceptDim: Int = 0) -> MLXArray {
    let axes = (0..<weight.ndim).filter { $0 != exceptDim }
    return MLX.sqrt(MLX.sum(weight * weight, axes: axes, keepDims: true))
}

internal func tadaSanitizeModuleWeights(
    _ weights: [String: MLXArray],
    prefix: String
) -> [String: MLXArray] {
    let dottedPrefix = "\(prefix)."
    let wnPrefixes = Set(weights.keys.compactMap { key -> String? in
        let stripped = key.hasPrefix(dottedPrefix) ? String(key.dropFirst(dottedPrefix.count)) : key
        let marker = ".parametrizations.weight.original0"
        guard let range = stripped.range(of: marker) else { return nil }
        return String(stripped[..<range.lowerBound])
    })

    var sanitized: [String: MLXArray] = [:]
    sanitized.reserveCapacity(weights.count)

    for (rawKey, value) in weights {
        var key = rawKey
        if key.hasPrefix(dottedPrefix) {
            key = String(key.dropFirst(dottedPrefix.count))
        } else if rawKey.contains(".") {
            continue
        }

        if key.contains(".parametrizations.weight.original0") {
            key = key.replacingOccurrences(
                of: ".parametrizations.weight.original0",
                with: ".weight_g"
            )
        } else if key.contains(".parametrizations.weight.original1") {
            key = key.replacingOccurrences(
                of: ".parametrizations.weight.original1",
                with: ".weight_v"
            )
        } else if key.hasSuffix(".conv.bias") {
            let prefix = String(key.dropLast(".conv.bias".count))
            if wnPrefixes.contains(prefix) {
                key = prefix + ".bias"
            }
        }

        sanitized[key] = value
    }

    return sanitized
}

internal func tadaGatherFrames(_ values: MLXArray, positions: MLXArray) -> MLXArray {
    let clipped = clip(
        positions - MLXArray(Int32(1)),
        min: 0,
        max: values.dim(1) - 1
    ).asType(.int32)
    let gatherIndex = broadcast(
        clipped.expandedDimensions(axis: -1),
        to: [values.dim(0), clipped.dim(1), values.dim(2)]
    )
    return MLX.takeAlong(values, gatherIndex, axis: 1)
}

internal func tadaSegmentAttentionAdditiveMask(
    boundaries: MLXArray?,
    attentionMask: MLXArray?,
    batchSize: Int,
    sequenceLength: Int
) -> MLXArray? {
    guard boundaries != nil || attentionMask != nil else {
        return nil
    }

    let validMask = attentionMask?.asType(.bool)
        ?? MLXArray.ones([batchSize, sequenceLength], dtype: .bool)
    let boundaryMask = boundaries?.asType(.int32)
        ?? MLXArray.zeros([batchSize, sequenceLength], dtype: .int32)
    let segmentIDs = MLX.cumsum(clip(boundaryMask, min: 0, max: 1), axis: 1)

    let querySegments = segmentIDs.expandedDimensions(axes: [1, 3])
    let keySegments = segmentIDs.expandedDimensions(axes: [1, 2])
    let sameSegment = querySegments .== keySegments
    let previousSegment = (querySegments - MLXArray(Int32(1))) .== keySegments
    let segmentAllowed = (sameSegment.asType(.int32) + previousSegment.asType(.int32)) .> 0

    let validQuery = validMask.expandedDimensions(axes: [1, 3])
    let validKey = validMask.expandedDimensions(axes: [1, 2])
    let allowed = (
        segmentAllowed.asType(.int32)
        * validQuery.asType(.int32)
        * validKey.asType(.int32)
    ) .> 0

    return MLX.where(
        allowed,
        MLXArray(Float(0)),
        MLXArray(Float(-1e9))
    )
}

public final class TADALocalAttentionFeedForward: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(hiddenSize: Int, intermediateSize: Int) {
        _gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(tadaSwiGLU(gateProj(x), upProj(x)))
    }
}

internal final class TADALocalAttention: Module {
    let hiddenSize: Int
    let numHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    let rope: RoPE

    init(hiddenSize: Int, numHeads: Int, ropeTheta: Float) {
        self.hiddenSize = hiddenSize
        self.numHeads = numHeads
        self.headDim = hiddenSize / numHeads
        self.scale = 1.0 / Foundation.sqrt(Float(headDim))

        _qProj.wrappedValue = Linear(hiddenSize, hiddenSize, bias: false)
        _kProj.wrappedValue = Linear(hiddenSize, hiddenSize, bias: false)
        _vProj.wrappedValue = Linear(hiddenSize, hiddenSize, bias: false)
        _oProj.wrappedValue = Linear(hiddenSize, hiddenSize, bias: false)

        self.rope = RoPE(dimensions: headDim, traditional: false, base: ropeTheta)
    }

    func callAsFunction(
        _ x: MLXArray,
        boundaries: MLXArray?,
        attentionMask: MLXArray?
    ) -> MLXArray {
        let batch = x.dim(0)
        let sequence = x.dim(1)

        var q = qProj(x).reshaped(batch, sequence, numHeads, headDim).transposed(0, 2, 1, 3)
        var k = kProj(x).reshaped(batch, sequence, numHeads, headDim).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped(batch, sequence, numHeads, headDim).transposed(0, 2, 1, 3)

        q = rope(q)
        k = rope(k)

        let additiveMask = tadaSegmentAttentionAdditiveMask(
            boundaries: boundaries,
            attentionMask: attentionMask,
            batchSize: batch,
            sequenceLength: sequence
        )

        let attended = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: k,
            values: v,
            scale: scale,
            mask: additiveMask
        )

        return oProj(attended.transposed(0, 2, 1, 3).reshaped(batch, sequence, hiddenSize))
    }
}

internal final class TADALocalAttentionLayer: Module {
    @ModuleInfo(key: "attention_norm") var attentionNorm: RMSNorm
    @ModuleInfo(key: "attention") var attention: TADALocalAttention
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm
    @ModuleInfo(key: "ffn") var ffn: TADALocalAttentionFeedForward

    init(
        hiddenSize: Int,
        numHeads: Int,
        feedForwardSize: Int,
        ropeTheta: Float
    ) {
        _attentionNorm.wrappedValue = RMSNorm(dimensions: hiddenSize, eps: 1e-5)
        _attention.wrappedValue = TADALocalAttention(
            hiddenSize: hiddenSize,
            numHeads: numHeads,
            ropeTheta: ropeTheta
        )
        _ffnNorm.wrappedValue = RMSNorm(dimensions: hiddenSize, eps: 1e-5)
        _ffn.wrappedValue = TADALocalAttentionFeedForward(
            hiddenSize: hiddenSize,
            intermediateSize: feedForwardSize
        )
    }

    func callAsFunction(
        _ x: MLXArray,
        boundaries: MLXArray?,
        attentionMask: MLXArray?
    ) -> MLXArray {
        let attended = x + attention(attentionNorm(x), boundaries: boundaries, attentionMask: attentionMask)
        return attended + ffn(ffnNorm(attended))
    }
}

public final class TADALocalAttentionStack: Module {
    @ModuleInfo(key: "layers") var layers: [TADALocalAttentionLayer]

    init(
        hiddenSize: Int = 1024,
        numLayers: Int = 6,
        numHeads: Int = 8,
        feedForwardSize: Int = 4096,
        ropeTheta: Float = 10_000
    ) {
        _layers.wrappedValue = (0..<numLayers).map { _ in
            TADALocalAttentionLayer(
                hiddenSize: hiddenSize,
                numHeads: numHeads,
                feedForwardSize: feedForwardSize,
                ropeTheta: ropeTheta
            )
        }
    }

    func callAsFunction(
        _ x: MLXArray,
        boundaries: MLXArray?,
        attentionMask: MLXArray?
    ) -> MLXArray {
        var hidden = x
        for layer in layers {
            hidden = layer(hidden, boundaries: boundaries, attentionMask: attentionMask)
        }
        return hidden
    }
}

// MARK: - TADA v2: Python-faithful attention (matches mlx_tada/encoder.py LocalSelfAttention)
// Merged QKV, post-attention LayerNorm (not pre-norm RMSNorm), traditional interleaved-pair RoPE,
// GELU FFN with linear1/linear2 (not SwiGLU). Required to load HumeAI safetensor weights.

// Encoder segment attention mask — matches Python encoder.create_segment_attention_mask.
// Returns additive float mask [B, 1, S, S]: 0.0 where can attend, -1e9 where blocked.
internal func tadaEncoderSegmentMask(tokenMask: MLXArray) -> MLXArray {
    let maskInt = clip(tokenMask, min: 0, max: 1).asType(.int32)
    let blockIDs = MLX.cumsum(maskInt, axis: 1)            // [B, S]
    let blockI = blockIDs.expandedDimensions(axis: 2)     // [B, S, 1]
    let blockJ = blockIDs.expandedDimensions(axis: 1)     // [B, 1, S]
    let sameBlock = (blockI .== blockJ).asType(.int32)    // [B, S, S]
    let isMarkedI = maskInt.expandedDimensions(axis: 2)   // [B, S, 1]
    let isMarkedJ = maskInt.expandedDimensions(axis: 1)   // [B, 1, S]
    let notMarkedJ = MLXArray(Int32(1)) - isMarkedJ
    // sameValid = same_block & (~is_marked_j | (is_marked_i & same_block))
    let innerOr = (notMarkedJ + isMarkedI * sameBlock) .> 0
    let sameValid = (sameBlock * innerOr.asType(.int32)) .> 0
    // prevValid = prev_block & ~is_marked_j
    let prevBlock = (blockJ .== (blockI - MLXArray(Int32(1)))).asType(.int32)
    let prevValid = (prevBlock * notMarkedJ) .> 0
    // canAttend = sameValid | (is_marked_i & prev_valid)
    let canAttend = (sameValid.asType(.int32) + isMarkedI * prevValid.asType(.int32)) .> 0
    return MLX.where(canAttend, MLXArray(Float(0)), MLXArray(Float(-1e9)))
        .expandedDimensions(axis: 1)
}

// Decoder segment attention mask — matches Python decoder.create_segment_attention_mask.
// Returns additive float mask [B, 1, S, S]: 0.0 where can attend, -1e9 where blocked.
internal func tadaDecoderSegmentMask(tokenMask: MLXArray) -> MLXArray {
    let maskInt = clip(tokenMask, min: 0, max: 1).asType(.int32)
    let blockIDs = MLX.cumsum(maskInt, axis: 1) - maskInt  // [B, S] shifted
    let blockI = blockIDs.expandedDimensions(axis: 2)
    let blockJ = blockIDs.expandedDimensions(axis: 1)
    let sameBlock = (blockI .== blockJ).asType(.int32)
    let prevBlock = (blockJ .== (blockI - MLXArray(Int32(1)))).asType(.int32)
    let canAttend = (sameBlock + prevBlock) .> 0
    return MLX.where(canAttend, MLXArray(Float(0)), MLXArray(Float(-1e9)))
        .expandedDimensions(axis: 1)
}

internal final class TADAv2LocalAttention: Module {
    let numHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "qkv") var qkv: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm

    // Traditional (interleaved-pair) RoPE matching Python's apply_rope
    let rope: RoPE

    init(hiddenSize: Int, numHeads: Int) {
        self.numHeads = numHeads
        self.headDim = hiddenSize / numHeads
        self.scale = 1.0 / Foundation.sqrt(Float(headDim))
        _qkv.wrappedValue = Linear(hiddenSize, 3 * hiddenSize)
        _outProj.wrappedValue = Linear(hiddenSize, hiddenSize)
        _layerNorm.wrappedValue = LayerNorm(dimensions: hiddenSize, eps: 1e-5)
        self.rope = RoPE(dimensions: headDim, traditional: true, base: 10_000)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        let batch = x.dim(0)
        let sequence = x.dim(1)
        let hiddenSize = x.dim(2)

        // Merged QKV → split into q, k, v per head
        let qkvOut = qkv(x).reshaped(batch, sequence, 3, numHeads, headDim)
        let qkvT = qkvOut.transposed(2, 0, 3, 1, 4)  // [3, B, H, S, D]
        var q = qkvT[0]  // [B, H, S, D]
        var k = qkvT[1]
        let v = qkvT[2]

        q = rope(q)
        k = rope(k)

        let attended = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v,
            scale: scale, mask: mask
        )
        let out = outProj(attended.transposed(0, 2, 1, 3).reshaped(batch, sequence, hiddenSize))
        // Post-attention residual + LayerNorm (inside attention, matching Python)
        return layerNorm(x + out)
    }
}

internal final class TADAv2EncoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: TADAv2LocalAttention
    @ModuleInfo(key: "linear1") var linear1: Linear
    @ModuleInfo(key: "linear2") var linear2: Linear
    @ModuleInfo(key: "norm") var norm: LayerNorm

    init(hiddenSize: Int, numHeads: Int, feedForwardSize: Int) {
        _selfAttn.wrappedValue = TADAv2LocalAttention(hiddenSize: hiddenSize, numHeads: numHeads)
        _linear1.wrappedValue = Linear(hiddenSize, feedForwardSize)
        _linear2.wrappedValue = Linear(feedForwardSize, hiddenSize)
        _norm.wrappedValue = LayerNorm(dimensions: hiddenSize, eps: 1e-5)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        // self_attn includes its own post-attn residual + LayerNorm
        let attended = selfAttn(x, mask: mask)
        let ffnOut = linear2(gelu(linear1(attended)))
        return norm(attended + ffnOut)
    }
}

// Stack with final LayerNorm — matches Python LocalAttentionEncoder.
// Module key must be "local_attention_encoder" (encoder) or "local_attention_decoder" (decoder).
public final class TADAv2AttentionStack: Module {
    @ModuleInfo(key: "layers") var layers: [TADAv2EncoderLayer]
    @ModuleInfo(key: "final_norm") var finalNorm: LayerNorm

    public init(
        hiddenSize: Int = 1024,
        numLayers: Int = 6,
        numHeads: Int = 8,
        feedForwardSize: Int = 4096
    ) {
        _layers.wrappedValue = (0..<numLayers).map { _ in
            TADAv2EncoderLayer(hiddenSize: hiddenSize, numHeads: numHeads, feedForwardSize: feedForwardSize)
        }
        _finalNorm.wrappedValue = LayerNorm(dimensions: hiddenSize, eps: 1e-5)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        var hidden = x
        for layer in layers {
            hidden = layer(hidden, mask: mask)
        }
        return finalNorm(hidden)
    }
}

public final class TADAWavEncoder: Module {
    @ModuleInfo(key: "conv") public var conv: DescriptWNConv1d
    @ModuleInfo(key: "blocks") public var blocks: [DescriptEncoderBlock]
    @ModuleInfo(key: "final_snake") public var finalSnake: DescriptSnake1d
    @ModuleInfo(key: "final_conv") public var finalConv: DescriptWNConv1d

    public let hiddenDim: Int
    public let strides: [Int]

    public init(hiddenDim: Int = 1024, strides: [Int] = [6, 5, 4, 4]) {
        self.hiddenDim = hiddenDim
        self.strides = strides

        _conv.wrappedValue = DescriptWNConv1d(
            inChannels: 1,
            outChannels: 64,
            kernelSize: 7,
            padding: 3
        )

        var nextBlocks: [DescriptEncoderBlock] = []
        var currentDim = 64
        for stride in strides {
            currentDim *= 2
            nextBlocks.append(DescriptEncoderBlock(dim: currentDim, stride: stride))
        }
        _blocks.wrappedValue = nextBlocks
        _finalSnake.wrappedValue = DescriptSnake1d(channels: currentDim)
        _finalConv.wrappedValue = DescriptWNConv1d(
            inChannels: currentDim,
            outChannels: hiddenDim,
            kernelSize: 3,
            padding: 1
        )
    }

    public func callAsFunction(_ audio: MLXArray) -> MLXArray {
        var hidden = audio.transposed(0, 2, 1)
        hidden = conv(hidden)
        for block in blocks {
            hidden = block(hidden)
        }
        hidden = finalSnake(hidden)
        hidden = finalConv(hidden)
        return hidden.transposed(0, 2, 1)
    }
}

public final class TADATTSEncoder: Module {
    public static let defaultAudioPadding = 960

    public let acousticMean: Float
    public let acousticStd: Float

    @ModuleInfo(key: "wav_encoder") public var wavEncoder: TADAWavEncoder
    // key matches Python: encoder/weights.safetensors → local_attention_encoder.*
    @ModuleInfo(key: "local_attention_encoder") public var localAttentionEncoder: TADAv2AttentionStack
    @ModuleInfo(key: "hidden_linear") public var hiddenLinear: Linear
    @ModuleInfo(key: "pos_emb") public var posEmb: Embedding

    public init(
        hiddenDim: Int = 1024,
        embedDim: Int = 512,
        strides: [Int] = [6, 5, 4, 4],
        numLayers: Int = 6,
        numHeads: Int = 8,
        feedForwardSize: Int = 4096,
        acousticMean: Float = 0,
        acousticStd: Float = 1.5
    ) {
        self.acousticMean = acousticMean
        self.acousticStd = acousticStd

        _wavEncoder.wrappedValue = TADAWavEncoder(hiddenDim: hiddenDim, strides: strides)
        _localAttentionEncoder.wrappedValue = TADAv2AttentionStack(
            hiddenSize: hiddenDim,
            numLayers: numLayers,
            numHeads: numHeads,
            feedForwardSize: feedForwardSize
        )
        _hiddenLinear.wrappedValue = Linear(hiddenDim, embedDim, bias: false)
        _posEmb.wrappedValue = Embedding(embeddingCount: 2, dimensions: hiddenDim)
    }

    public func frameEncode(
        _ audio: MLXArray,
        tokenMask: MLXArray? = nil
    ) -> MLXArray {
        let padded = padAudio(audio)
        var hidden = wavEncoder(padded)
        let seqLen = hidden.dim(1)

        let attnMask: MLXArray?
        if let tokenMask {
            // Adjust token mask length to match encoder output (as in Python get_encoder_outputs)
            let maskLen = tokenMask.dim(1)
            let paddedMask: MLXArray
            if seqLen > maskLen {
                paddedMask = MLX.padded(
                    tokenMask.asType(.int32),
                    widths: [IntOrPair(0), IntOrPair((0, seqLen - maskLen))]
                )
            } else {
                paddedMask = tokenMask.asType(.int32)[0..., ..<seqLen]
            }
            let boundaryIDs = clip(paddedMask, min: 0, max: 1).asType(.int32)
            hidden = hidden + posEmb(boundaryIDs)
            attnMask = tadaEncoderSegmentMask(tokenMask: paddedMask)
        } else {
            attnMask = nil
        }

        return localAttentionEncoder(hidden, mask: attnMask)
    }

    public func callAsFunction(
        _ audio: MLXArray,
        tokenPositions: MLXArray,
        tokenMask: MLXArray,
        attentionMask: MLXArray? = nil
    ) -> MLXArray {
        let hidden = frameEncode(audio, tokenMask: tokenMask)
        let projected = hiddenLinear(hidden)
        let tokenValues = tadaGatherFrames(projected, positions: tokenPositions)
        return (tokenValues - MLXArray(acousticMean)) / MLXArray(acousticStd)
    }

    public func padAudio(_ audio: MLXArray) -> MLXArray {
        let batched: MLXArray
        if audio.ndim == 2 {
            batched = audio.expandedDimensions(axis: -1)
        } else {
            batched = audio
        }
        return MLX.padded(
            batched,
            widths: [
                IntOrPair(0),
                IntOrPair((0, Self.defaultAudioPadding)),
                IntOrPair(0)
            ]
        )
    }

    public static func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        tadaSanitizeModuleWeights(weights, prefix: "encoder")
    }
}
