import Foundation
@preconcurrency import MLX
import MLXNN

private func voxtralTTSRMSNormalize(_ x: MLXArray, eps: Float) -> MLXArray {
    let xFloat = x.asType(.float32)
    let normalized = xFloat * rsqrt(mean(xFloat * xFloat, axis: -1, keepDims: true) + MLXArray(eps))
    return normalized.asType(x.dtype)
}

public enum VoxtralTTSCodecVariant: Sendable {
    case legacy
    case community
}

final class VoxtralTTSCausalConv1d: Module {
    let kernelSize: Int
    let stride: Int
    let leftPadding: Int

    @ModuleInfo(key: "conv") var conv: Conv1d

    init(inChannels: Int, outChannels: Int, kernelSize: Int, stride: Int = 1) {
        self.kernelSize = kernelSize
        self.stride = stride
        self.leftPadding = kernelSize - 1
        _conv.wrappedValue = Conv1d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: kernelSize,
            stride: stride,
            padding: 0,
            bias: false
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        if leftPadding > 0 {
            h = padded(h, widths: [.init(0), .init(0), .init((leftPadding, 0))])
        }
        return conv(h.transposed(0, 2, 1)).transposed(0, 2, 1)
    }
}

final class VoxtralTTSCausalTransposeConv1d: Module {
    let trimRight: Int

    @ModuleInfo(key: "conv") var conv: ConvTransposed1d

    init(inChannels: Int, outChannels: Int, kernelSize: Int, stride: Int) {
        trimRight = max(0, kernelSize - stride)
        _conv.wrappedValue = ConvTransposed1d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: kernelSize,
            stride: stride,
            padding: 0,
            bias: false
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = conv(x.transposed(0, 2, 1)).transposed(0, 2, 1)
        if trimRight > 0 {
            h = h[0..., 0..., ..<(-trimRight)]
        }
        return h
    }
}

final class VoxtralTTSLayerScale: Module {
    var scale: MLXArray

    init(dim: Int, initialValue: Float) {
        scale = MLXArray.full([dim], values: MLXArray(initialValue))
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // scale [dim] broadcasts over trailing dims; works for [B, L, C] residual format
        x * scale
    }
}

final class VoxtralTTSCodecFeedForward: Module {
    @ModuleInfo(key: "conv1") var conv1: VoxtralTTSCausalConv1d
    @ModuleInfo(key: "conv2") var conv2: VoxtralTTSCausalConv1d

    init(dim: Int, hiddenDim: Int) {
        _conv1.wrappedValue = VoxtralTTSCausalConv1d(inChannels: dim, outChannels: hiddenDim, kernelSize: 1)
        _conv2.wrappedValue = VoxtralTTSCausalConv1d(inChannels: hiddenDim, outChannels: dim, kernelSize: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        conv2(gelu(conv1(x)))
    }
}

final class VoxtralTTSCodecAttention: Module {
    let nHeads: Int
    let nKvHeads: Int
    let headDim: Int
    let qkNorm: Bool
    let qkNormEps: Float
    let slidingWindow: Int
    let scale: Float
    let alibiSlopes: MLXArray

    private var cachedMask: MLXArray? = nil
    private var cachedMaskSeqLen: Int = 0

    @ModuleInfo(key: "wq") var wq: Linear
    @ModuleInfo(key: "wk") var wk: Linear
    @ModuleInfo(key: "wv") var wv: Linear
    @ModuleInfo(key: "wo") var wo: Linear

    init(config: VoxtralTTSCodecConfig, slidingWindow: Int) {
        self.nHeads = config.nHeads
        self.nKvHeads = config.nKvHeads
        self.headDim = config.headDim
        self.qkNorm = config.qkNorm
        self.qkNormEps = config.qkNormEps
        self.slidingWindow = slidingWindow
        self.scale = pow(Float(config.headDim), -0.5)

        _wq.wrappedValue = Linear(config.dim, config.nHeads * config.headDim, bias: false)
        _wk.wrappedValue = Linear(config.dim, config.nKvHeads * config.headDim, bias: false)
        _wv.wrappedValue = Linear(config.dim, config.nKvHeads * config.headDim, bias: false)
        _wo.wrappedValue = Linear(config.nHeads * config.headDim, config.dim, bias: false)

        let ratio = pow(Float(2.0), Float(-8.0 / Float(config.nHeads)))
        let powers = MLXArray(1...(config.nHeads)).asType(.float32)
        alibiSlopes = pow(MLXArray(ratio), powers)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let batch = x.dim(0)
        let seqLen = x.dim(1)

        var q = wq(x).reshaped(batch, seqLen, nHeads, headDim).transposed(0, 2, 1, 3)
        var k = wk(x).reshaped(batch, seqLen, nKvHeads, headDim).transposed(0, 2, 1, 3)
        let v = wv(x).reshaped(batch, seqLen, nKvHeads, headDim).transposed(0, 2, 1, 3)

        if qkNorm {
            q = voxtralTTSRMSNormalize(q, eps: qkNormEps)
            k = voxtralTTSRMSNormalize(k, eps: qkNormEps)
        }

        let mask: MLXArray
        if seqLen == cachedMaskSeqLen, let cached = cachedMask {
            mask = cached
        } else {
            let positions = MLXArray(0..<seqLen).asType(.float32)
            let qPos = positions.expandedDimensions(axis: 1)
            let kPos = positions.expandedDimensions(axis: 0)
            let relative = kPos - qPos
            let causal = kPos .<= qPos
            let windowMask = (qPos - kPos) .<= MLXArray(Float(slidingWindow))
            let allowed = causal .&& windowMask
            let additiveMask = MLX.where(allowed, MLXArray(0.0), MLXArray(-1e9))
            let alibi = alibiSlopes.reshaped(1, nHeads, 1, 1) * relative.reshaped(1, 1, seqLen, seqLen)
            let built = alibi + additiveMask.reshaped(1, 1, seqLen, seqLen)
            cachedMask = built
            cachedMaskSeqLen = seqLen
            mask = built
        }

        let output = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: k,
            values: v,
            scale: scale,
            mask: .array(mask)
        )

        return wo(output.transposed(0, 2, 1, 3).reshaped(batch, seqLen, nHeads * headDim))
    }
}

final class VoxtralTTSCodecTransformerLayer: Module {
    @ModuleInfo(key: "sa_norm") var saNorm: RMSNorm
    @ModuleInfo(key: "attention") var attention: VoxtralTTSCodecAttention
    @ModuleInfo(key: "ff_norm") var ffNorm: RMSNorm
    @ModuleInfo(key: "ff") var ff: VoxtralTTSCodecFeedForward
    @ModuleInfo(key: "layer_scale") var layerScale: VoxtralTTSLayerScale
    @ModuleInfo(key: "layer_scale_1") var ffLayerScale: VoxtralTTSLayerScale

    init(config: VoxtralTTSCodecConfig, slidingWindow: Int) {
        _saNorm.wrappedValue = RMSNorm(dimensions: config.dim, eps: config.normEps)
        _attention.wrappedValue = VoxtralTTSCodecAttention(config: config, slidingWindow: slidingWindow)
        _ffNorm.wrappedValue = RMSNorm(dimensions: config.dim, eps: config.normEps)
        _ff.wrappedValue = VoxtralTTSCodecFeedForward(dim: config.dim, hiddenDim: config.hiddenDim)
        _layerScale.wrappedValue = VoxtralTTSLayerScale(dim: config.dim, initialValue: config.layerScaleInit)
        _ffLayerScale.wrappedValue = VoxtralTTSLayerScale(dim: config.dim, initialValue: config.layerScaleInit)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x is [B, L, C] — residual stays in this format; no transposes for norm or attention
        let attnOut = attention(saNorm(x))
        let h = x + layerScale(attnOut)
        // ff uses causal conv which expects [B,C,L]; transpose only at this conv boundary
        let ffOut = ff(ffNorm(h).transposed(0, 2, 1)).transposed(0, 2, 1)
        return h + ffLayerScale(ffOut)
    }
}

final class VoxtralTTSCodecDecoderBlock: Module {
    @ModuleInfo(key: "convtr") var convtr: VoxtralTTSCausalTransposeConv1d
    @ModuleInfo(key: "layers") var layers: [VoxtralTTSCodecTransformerLayer]

    init(
        config: VoxtralTTSCodecConfig,
        kernelSize: Int,
        stride: Int,
        numLayers: Int,
        slidingWindow: Int
    ) {
        _convtr.wrappedValue = VoxtralTTSCausalTransposeConv1d(
            inChannels: config.dim,
            outChannels: config.dim,
            kernelSize: kernelSize,
            stride: stride
        )
        _layers.wrappedValue = (0..<numLayers).map { _ in
            VoxtralTTSCodecTransformerLayer(config: config, slidingWindow: slidingWindow)
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // convtr outputs [B,C,L]; transpose once to [B,L,C] for transformer layers
        var h = convtr(x).transposed(0, 2, 1)
        for layer in layers {
            h = layer(h)
        }
        return h.transposed(0, 2, 1)  // back to [B,C,L] for the next conv block
    }
}

final class VoxtralTTSCodecDecoderCore: Module {
    @ModuleInfo(key: "patch_proj") var patchProj: VoxtralTTSCausalConv1d
    @ModuleInfo(key: "blocks") var blocks: [VoxtralTTSCodecDecoderBlock]
    @ModuleInfo(key: "output_proj") var outputProj: VoxtralTTSCausalConv1d

    init(config: VoxtralTTSCodecConfig) {
        _patchProj.wrappedValue = VoxtralTTSCausalConv1d(
            inChannels: config.semanticDim + config.acousticDim,
            outChannels: config.dim,
            kernelSize: config.patchProjKernelSize
        )

        let lengths = config.decoderTransformerLengths
        let kernels = config.decoderConvsKernels
        let strides = config.decoderConvsStrides

        var window = config.attnSlidingWindowSize
        var builtBlocks: [VoxtralTTSCodecDecoderBlock] = []
        for index in 0..<min(lengths.count, min(kernels.count, strides.count)) {
            if config.halfAttnWindowUponDownsampling && strides[index] > 1 {
                window = max(1, window / 2)
            }
            builtBlocks.append(
                VoxtralTTSCodecDecoderBlock(
                    config: config,
                    kernelSize: kernels[index],
                    stride: strides[index],
                    numLayers: lengths[index],
                    slidingWindow: window
                )
            )
        }
        _blocks.wrappedValue = builtBlocks

        _outputProj.wrappedValue = VoxtralTTSCausalConv1d(
            inChannels: config.dim,
            outChannels: config.pretransformPatchSize,
            kernelSize: 1
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = patchProj(x)
        for block in blocks {
            h = block(h)
        }
        return outputProj(h)
    }
}

final class VoxtralTTSSemanticQuantizer: Module {
    @ModuleInfo(key: "codebook") var codebook: Embedding

    init(embeddingCount: Int, dimensions: Int) {
        _codebook.wrappedValue = Embedding(embeddingCount: embeddingCount, dimensions: dimensions)
    }
}

final class VoxtralTTSQuantizer: Module {
    @ModuleInfo(key: "semantic_quantizer") var semanticQuantizer: VoxtralTTSSemanticQuantizer

    init(config: VoxtralTTSCodecConfig) {
        _semanticQuantizer.wrappedValue = VoxtralTTSSemanticQuantizer(
            embeddingCount: config.semanticCodebookSize,
            dimensions: config.semanticDim
        )
    }
}

final class VoxtralTTSCommunityFeedForward: Module {
    @ModuleInfo(key: "w1") var w1: Linear
    @ModuleInfo(key: "w2") var w2: Linear
    @ModuleInfo(key: "w3") var w3: Linear

    init(dim: Int, hiddenDim: Int) {
        _w1.wrappedValue = Linear(dim, hiddenDim, bias: false)
        _w2.wrappedValue = Linear(hiddenDim, dim, bias: false)
        _w3.wrappedValue = Linear(dim, hiddenDim, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        w2(silu(w1(x)) * w3(x))
    }
}

final class VoxtralTTSCommunityAttention: Module {
    let nHeads: Int
    let nKvHeads: Int
    let headDim: Int
    let qkNorm: Bool
    let qkNormEps: Float
    let slidingWindow: Int
    let scale: Float
    let alibiSlopes: MLXArray

    private var cachedMask: MLXArray? = nil
    private var cachedMaskSeqLen: Int = 0

    @ModuleInfo(key: "wq") var wq: Linear
    @ModuleInfo(key: "wk") var wk: Linear
    @ModuleInfo(key: "wv") var wv: Linear
    @ModuleInfo(key: "wo") var wo: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm?
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm?

    init(config: VoxtralTTSCodecConfig, slidingWindow: Int) {
        self.nHeads = config.nHeads
        self.nKvHeads = config.nKvHeads
        self.headDim = config.headDim
        self.qkNorm = config.qkNorm
        self.qkNormEps = config.qkNormEps
        self.slidingWindow = slidingWindow
        self.scale = pow(Float(config.headDim), -0.5)

        _wq.wrappedValue = Linear(config.dim, config.nHeads * config.headDim, bias: false)
        _wk.wrappedValue = Linear(config.dim, config.nKvHeads * config.headDim, bias: false)
        _wv.wrappedValue = Linear(config.dim, config.nKvHeads * config.headDim, bias: false)
        _wo.wrappedValue = Linear(config.nHeads * config.headDim, config.dim, bias: false)

        if config.qkNorm {
            _qNorm.wrappedValue = RMSNorm(dimensions: config.dim, eps: config.qkNormEps)
            _kNorm.wrappedValue = RMSNorm(dimensions: config.dim, eps: config.qkNormEps)
        }

        let ratio = pow(Float(2.0), Float(-8.0 / Float(config.nHeads)))
        let powers = MLXArray(1...(config.nHeads)).asType(.float32)
        alibiSlopes = pow(MLXArray(ratio), powers)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let batch = x.dim(0)
        let seqLen = x.dim(1)

        var q = wq(x)
        var k = wk(x)
        let vLinear = wv(x)

        if qkNorm {
            if let qNorm, let kNorm {
                q = qNorm(q)
                k = kNorm(k)
            } else {
                q = voxtralTTSRMSNormalize(q, eps: qkNormEps)
                k = voxtralTTSRMSNormalize(k, eps: qkNormEps)
            }
        }

        q = q.reshaped(batch, seqLen, nHeads, headDim).transposed(0, 2, 1, 3)
        k = k.reshaped(batch, seqLen, nKvHeads, headDim).transposed(0, 2, 1, 3)
        let v = vLinear.reshaped(batch, seqLen, nKvHeads, headDim).transposed(0, 2, 1, 3)

        let mask: MLXArray
        if seqLen == cachedMaskSeqLen, let cached = cachedMask {
            mask = cached
        } else {
            let positions = MLXArray(0..<seqLen).asType(.float32)
            let qPos = positions.expandedDimensions(axis: 1)
            let kPos = positions.expandedDimensions(axis: 0)
            let relative = kPos - qPos
            let causal = kPos .<= qPos
            let windowMask = (qPos - kPos) .<= MLXArray(Float(slidingWindow))
            let allowed = causal .&& windowMask
            let additiveMask = MLX.where(allowed, MLXArray(0.0), MLXArray(-1e9))
            let alibi = alibiSlopes.reshaped(1, nHeads, 1, 1) * relative.reshaped(1, 1, seqLen, seqLen)
            let built = alibi + additiveMask.reshaped(1, 1, seqLen, seqLen)
            cachedMask = built
            cachedMaskSeqLen = seqLen
            mask = built
        }

        let output = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: k,
            values: v,
            scale: scale,
            mask: .array(mask)
        )

        return wo(output.transposed(0, 2, 1, 3).reshaped(batch, seqLen, nHeads * headDim))
    }
}

final class VoxtralTTSCommunityTransformerLayer: Module {
    @ModuleInfo(key: "attention_norm") var attentionNorm: RMSNorm
    @ModuleInfo(key: "attention") var attention: VoxtralTTSCommunityAttention
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm
    @ModuleInfo(key: "feed_forward") var feedForward: VoxtralTTSCommunityFeedForward
    @ModuleInfo(key: "attention_scale") var attentionScale: MLXArray
    @ModuleInfo(key: "ffn_scale") var ffnScale: MLXArray

    init(config: VoxtralTTSCodecConfig, slidingWindow: Int) {
        _attentionNorm.wrappedValue = RMSNorm(dimensions: config.dim, eps: config.normEps)
        _attention.wrappedValue = VoxtralTTSCommunityAttention(config: config, slidingWindow: slidingWindow)
        _ffnNorm.wrappedValue = RMSNorm(dimensions: config.dim, eps: config.normEps)
        _feedForward.wrappedValue = VoxtralTTSCommunityFeedForward(dim: config.dim, hiddenDim: config.hiddenDim)
        _attentionScale.wrappedValue = MLXArray.full([config.dim], values: MLXArray(config.layerScaleInit))
        _ffnScale.wrappedValue = MLXArray.full([config.dim], values: MLXArray(config.layerScaleInit))
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let attnOut = attention(attentionNorm(x))
        let h = x + (attnOut * attentionScale)
        return h + (feedForward(ffnNorm(h)) * ffnScale)
    }
}

final class VoxtralTTSCommunityTransformerGroup: Module {
    @ModuleInfo(key: "layers") var layers: [VoxtralTTSCommunityTransformerLayer]

    init(config: VoxtralTTSCodecConfig, numLayers: Int, slidingWindow: Int) {
        _layers.wrappedValue = (0..<numLayers).map { _ in
            VoxtralTTSCommunityTransformerLayer(config: config, slidingWindow: slidingWindow)
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for layer in layers {
            h = layer(h)
        }
        return h
    }
}

final class VoxtralTTSCommunityDecoderCore: Module {
    @ModuleInfo(key: "inputConv") var inputConv: VoxtralTTSCausalConv1d
    @ModuleInfo(key: "stage0") var stage0: VoxtralTTSCommunityTransformerGroup
    @ModuleInfo(key: "upsample1") var upsample1: VoxtralTTSCausalTransposeConv1d
    @ModuleInfo(key: "stage1") var stage1: VoxtralTTSCommunityTransformerGroup
    @ModuleInfo(key: "upsample2") var upsample2: VoxtralTTSCausalTransposeConv1d
    @ModuleInfo(key: "stage2") var stage2: VoxtralTTSCommunityTransformerGroup
    @ModuleInfo(key: "upsample3") var upsample3: VoxtralTTSCausalTransposeConv1d
    @ModuleInfo(key: "stage3") var stage3: VoxtralTTSCommunityTransformerGroup
    @ModuleInfo(key: "outputProj") var outputProj: VoxtralTTSCausalConv1d

    init(config: VoxtralTTSCodecConfig) {
        let kernels = config.decoderConvsKernels
        let strides = config.decoderConvsStrides
        let lengths = config.decoderTransformerLengths

        func value(_ values: [Int], _ index: Int, default fallback: Int) -> Int {
            index < values.count ? values[index] : fallback
        }

        _inputConv.wrappedValue = VoxtralTTSCausalConv1d(
            inChannels: config.semanticDim + config.acousticDim,
            outChannels: config.dim,
            kernelSize: value(kernels, 0, default: 3)
        )

        // In the decoder, upsampling increases temporal resolution so attention windows
        // should grow (not shrink). Start with the smallest window and double at each
        // upsample step — the inverse of encoder/downsampling behaviour.
        // e.g. attnSlidingWindowSize=16, strides=[1,2,2,2] → windows [2,4,8,16].
        let upsampleStrides = [
            value(strides, 1, default: 2),
            value(strides, 2, default: 2),
            value(strides, 3, default: 2),
        ]
        let totalUpsample = config.halfAttnWindowUponDownsampling
            ? upsampleStrides.reduce(1) { $1 > 1 ? $0 * $1 : $0 }
            : 1
        var window = config.attnSlidingWindowSize / max(1, totalUpsample)
        _stage0.wrappedValue = VoxtralTTSCommunityTransformerGroup(
            config: config,
            numLayers: value(lengths, 0, default: 2),
            slidingWindow: window
        )

        let stride1 = upsampleStrides[0]
        _upsample1.wrappedValue = VoxtralTTSCausalTransposeConv1d(
            inChannels: config.dim,
            outChannels: config.dim,
            kernelSize: value(kernels, 1, default: 4),
            stride: stride1
        )
        if config.halfAttnWindowUponDownsampling && stride1 > 1 {
            window *= stride1
        }
        _stage1.wrappedValue = VoxtralTTSCommunityTransformerGroup(
            config: config,
            numLayers: value(lengths, 1, default: 2),
            slidingWindow: window
        )

        let stride2 = upsampleStrides[1]
        _upsample2.wrappedValue = VoxtralTTSCausalTransposeConv1d(
            inChannels: config.dim,
            outChannels: config.dim,
            kernelSize: value(kernels, 2, default: 4),
            stride: stride2
        )
        if config.halfAttnWindowUponDownsampling && stride2 > 1 {
            window *= stride2
        }
        _stage2.wrappedValue = VoxtralTTSCommunityTransformerGroup(
            config: config,
            numLayers: value(lengths, 2, default: 2),
            slidingWindow: window
        )

        let stride3 = upsampleStrides[2]
        _upsample3.wrappedValue = VoxtralTTSCausalTransposeConv1d(
            inChannels: config.dim,
            outChannels: config.dim,
            kernelSize: value(kernels, 3, default: 4),
            stride: stride3
        )
        if config.halfAttnWindowUponDownsampling && stride3 > 1 {
            window *= stride3
        }
        _stage3.wrappedValue = VoxtralTTSCommunityTransformerGroup(
            config: config,
            numLayers: value(lengths, 3, default: 2),
            slidingWindow: window
        )

        _outputProj.wrappedValue = VoxtralTTSCausalConv1d(
            inChannels: config.dim,
            outChannels: config.pretransformPatchSize,
            kernelSize: config.patchProjKernelSize
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = inputConv(x).transposed(0, 2, 1)
        h = stage0(h)
        h = upsample1(h.transposed(0, 2, 1)).transposed(0, 2, 1)
        h = stage1(h)
        h = upsample2(h.transposed(0, 2, 1)).transposed(0, 2, 1)
        h = stage2(h)
        h = upsample3(h.transposed(0, 2, 1)).transposed(0, 2, 1)
        h = stage3(h)
        return outputProj(h.transposed(0, 2, 1))
    }
}

final class VoxtralTTSCommunityQuantizer: Module {
    @ModuleInfo(key: "semanticCodebook") var semanticCodebook: Embedding

    init(config: VoxtralTTSCodecConfig) {
        _semanticCodebook.wrappedValue = Embedding(
            embeddingCount: config.semanticCodebookSize,
            dimensions: config.semanticDim
        )
    }
}

public final class VoxtralTTSCodecDecoder: Module {
    public let config: VoxtralTTSCodecConfig
    public let variant: VoxtralTTSCodecVariant

    @ModuleInfo(key: "decoder") var decoder: VoxtralTTSCodecDecoderCore?
    @ModuleInfo(key: "quantizer") var quantizer: VoxtralTTSQuantizer?
    @ModuleInfo(key: "communityDecoder") var communityDecoder: VoxtralTTSCommunityDecoderCore?
    @ModuleInfo(key: "communityQuantizer") var communityQuantizer: VoxtralTTSCommunityQuantizer?

    public init(config: VoxtralTTSCodecConfig, variant: VoxtralTTSCodecVariant = .legacy) {
        self.config = config
        self.variant = variant

        switch variant {
        case .legacy:
            _decoder.wrappedValue = VoxtralTTSCodecDecoderCore(config: config)
            _quantizer.wrappedValue = VoxtralTTSQuantizer(config: config)
        case .community:
            _communityDecoder.wrappedValue = VoxtralTTSCommunityDecoderCore(config: config)
            _communityQuantizer.wrappedValue = VoxtralTTSCommunityQuantizer(config: config)
        }
    }

    func dequantize(_ codes: MLXArray) -> MLXArray {
        let semanticCodes = clip(codes[0..., 0, 0...] - MLXArray(Int32(2)), min: 0, max: config.semanticCodebookSize - 1)
            .asType(.int32)
        let semanticEmbedding: MLXArray
        if let quantizer {
            semanticEmbedding = quantizer.semanticQuantizer.codebook(semanticCodes).transposed(0, 2, 1)
        } else if let communityQuantizer {
            semanticEmbedding = communityQuantizer.semanticCodebook(semanticCodes).transposed(0, 2, 1)
        } else {
            fatalError("Voxtral codec quantizer was not initialized")
        }

        let acousticCodes = clip(codes[0..., 1..., 0...] - MLXArray(Int32(2)), min: 0, max: config.acousticCodebookSize - 1)
            .asType(.float32)
        let acousticEmbedding = acousticCodes / Float(config.acousticCodebookSize - 1) * 2.0 - 1.0

        return concatenated([semanticEmbedding, acousticEmbedding], axis: 1)
    }

    func decode(_ codes: MLXArray) -> MLXArray {
        let embedding = dequantize(codes)
        let decoded: MLXArray
        if let decoder {
            decoded = decoder(embedding)
        } else if let communityDecoder {
            decoded = communityDecoder(embedding)
        } else {
            fatalError("Voxtral codec decoder was not initialized")
        }
        // decoded is [B, patch_size=240, T] (CausalConv1d output); transpose to [B, T, 240]
        // so that the reshape produces time-major audio (matching Python x.reshape(B, -1))
        return decoded.transposed(0, 2, 1).reshaped(decoded.dim(0), 1, -1)
    }
}
