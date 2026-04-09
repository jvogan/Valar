import Foundation
import MLX
import MLXLMCommon
import MLXNN

public enum TADATTSAlignerDefaults {
    static let hiddenSize = 1_024
    static let numAttentionHeads = 16
    static let intermediateSize = 4_096
    static let numHiddenLayers = 24
    static let convDim = [512, 512, 512, 512, 512, 512, 512]
    static let convKernel = [10, 3, 3, 3, 3, 2, 2]
    static let convStride = [5, 2, 2, 2, 2, 2, 2]
    static let vocabSize = 128_256
    static let numConvPosEmbeddings = 128
    static let numConvPosEmbeddingGroups = 16
    static let layerNormEps: Float = 1e-5
    public static let blankTokenID = 0
}

public struct TADATTSAlignmentResult {
    public let tokenPositions: MLXArray
    public let tokenMasks: MLXArray

    public init(tokenPositions: MLXArray, tokenMasks: MLXArray) {
        self.tokenPositions = tokenPositions
        self.tokenMasks = tokenMasks
    }
}

public final class TADATTSBackbone: Module {
    public let config: TADATTSConfig

    @ModuleInfo(key: "model") var model: LlamaTTSModelInner
    @ModuleInfo(key: "acoustic_proj") var acousticProj: Linear
    @ModuleInfo(key: "time_start_embed") var timeStartEmbed: Embedding
    @ModuleInfo(key: "time_end_embed") var timeEndEmbed: Embedding
    @ModuleInfo(key: "acoustic_mask_emb") var acousticMaskEmbedding: Embedding

    public init(config: TADATTSConfig) {
        self.config = config
        _model.wrappedValue = LlamaTTSModelInner(config.llamaConfiguration)
        _acousticProj.wrappedValue = Linear(config.acousticDim, config.hiddenSize, bias: false)
        _timeStartEmbed.wrappedValue = Embedding(
            embeddingCount: config.numTimeClasses,
            dimensions: config.hiddenSize
        )
        _timeEndEmbed.wrappedValue = Embedding(
            embeddingCount: config.numTimeClasses,
            dimensions: config.hiddenSize
        )
        _acousticMaskEmbedding.wrappedValue = Embedding(
            embeddingCount: 2,
            dimensions: config.hiddenSize
        )
    }

    public func callAsFunction(
        inputIDs: MLXArray,
        acousticFeatures: MLXArray,
        acousticMask: MLXArray,
        timeBefore: MLXArray,
        timeAfter: MLXArray,
        cache: [KVCache]? = nil
    ) -> MLXArray {
        var embeddings = model.embedTokens(inputIDs.asType(.int32))
        embeddings = embeddings + acousticProj(acousticFeatures)
        embeddings = embeddings + acousticMaskEmbedding(acousticMask.asType(.int32))
        embeddings = embeddings + timeStartEmbed(timeBefore.asType(.int32))
        embeddings = embeddings + timeEndEmbed(timeAfter.asType(.int32))
        return model(embeddings: embeddings, cache: cache)
    }

    public func makeCache() -> [KVCache] {
        (0..<config.numHiddenLayers).map { _ in KVCacheSimple() }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        for (key, value) in weights where !key.contains("self_attn.rotary_emb.inv_freq") {
            sanitized[key] = value
        }
        return sanitized
    }
}

private final class TADATTSAlignerFeatureExtractorLayer: Module {
    @ModuleInfo(key: "conv") var conv: Conv1d
    @ModuleInfo(key: "layer_norm") var layerNorm: GroupNorm?

    init(
        inputChannels: Int,
        outputChannels: Int,
        kernelSize: Int,
        stride: Int,
        useLayerNorm: Bool
    ) {
        _conv.wrappedValue = Conv1d(
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: kernelSize,
            stride: stride,
            bias: false
        )
        if useLayerNorm {
            _layerNorm.wrappedValue = GroupNorm(
                groupCount: outputChannels,
                dimensions: outputChannels,
                pytorchCompatible: true
            )
        } else {
            _layerNorm.wrappedValue = nil
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var hiddenStates = conv(x)
        if let layerNorm {
            hiddenStates = hiddenStates.transposed(0, 2, 1)
            hiddenStates = layerNorm(hiddenStates)
            hiddenStates = hiddenStates.transposed(0, 2, 1)
        }
        return gelu(hiddenStates)
    }
}

private final class TADATTSAlignerFeatureExtractor: Module {
    @ModuleInfo(key: "conv_layers") var convLayers: [TADATTSAlignerFeatureExtractorLayer]

    override init() {
        let inputChannels = [1] + Array(TADATTSAlignerDefaults.convDim.dropLast())
        _convLayers.wrappedValue = zip(
            zip(inputChannels, TADATTSAlignerDefaults.convDim),
            zip(TADATTSAlignerDefaults.convKernel, TADATTSAlignerDefaults.convStride)
        ).enumerated().map { index, item in
            TADATTSAlignerFeatureExtractorLayer(
                inputChannels: item.0.0,
                outputChannels: item.0.1,
                kernelSize: item.1.0,
                stride: item.1.1,
                useLayerNorm: index == 0
            )
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        convLayers.reduce(x) { partialResult, layer in
            layer(partialResult)
        }
    }
}

private final class TADATTSAlignerFeatureProjection: Module {
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
    @ModuleInfo var projection: Linear

    override init() {
        _layerNorm.wrappedValue = LayerNorm(
            dimensions: TADATTSAlignerDefaults.convDim.last ?? 512,
            eps: TADATTSAlignerDefaults.layerNormEps
        )
        _projection.wrappedValue = Linear(
            TADATTSAlignerDefaults.convDim.last ?? 512,
            TADATTSAlignerDefaults.hiddenSize
        )
    }

    func callAsFunction(_ x: MLXArray) -> (hiddenStates: MLXArray, extractFeatures: MLXArray) {
        let normalized = layerNorm(x)
        return (projection(normalized), normalized)
    }
}

private final class TADATTSAlignerPositionalConvEmbedding: Module {
    @ModuleInfo var conv: Conv1d

    override init() {
        _conv.wrappedValue = Conv1d(
            inputChannels: TADATTSAlignerDefaults.hiddenSize,
            outputChannels: TADATTSAlignerDefaults.hiddenSize,
            kernelSize: TADATTSAlignerDefaults.numConvPosEmbeddings,
            padding: TADATTSAlignerDefaults.numConvPosEmbeddings / 2,
            groups: TADATTSAlignerDefaults.numConvPosEmbeddingGroups,
            bias: true
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var positional = conv(x)
        if positional.dim(1) > x.dim(1) {
            positional = positional[0..., ..<x.dim(1), 0...]
        }
        return gelu(positional)
    }
}

private final class TADATTSAlignerAttention: Module {
    let numHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    override init() {
        numHeads = TADATTSAlignerDefaults.numAttentionHeads
        headDim = TADATTSAlignerDefaults.hiddenSize / TADATTSAlignerDefaults.numAttentionHeads
        scale = pow(Float(headDim), -0.5)
        _qProj.wrappedValue = Linear(TADATTSAlignerDefaults.hiddenSize, TADATTSAlignerDefaults.hiddenSize)
        _kProj.wrappedValue = Linear(TADATTSAlignerDefaults.hiddenSize, TADATTSAlignerDefaults.hiddenSize)
        _vProj.wrappedValue = Linear(TADATTSAlignerDefaults.hiddenSize, TADATTSAlignerDefaults.hiddenSize)
        _outProj.wrappedValue = Linear(TADATTSAlignerDefaults.hiddenSize, TADATTSAlignerDefaults.hiddenSize)
    }

    func callAsFunction(_ x: MLXArray, attentionMask: MLXArray?) -> MLXArray {
        let batchSize = x.dim(0)
        let sequenceLength = x.dim(1)

        let q = qProj(x).reshaped(batchSize, sequenceLength, numHeads, headDim).transposed(0, 2, 1, 3)
        let k = kProj(x).reshaped(batchSize, sequenceLength, numHeads, headDim).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped(batchSize, sequenceLength, numHeads, headDim).transposed(0, 2, 1, 3)

        var scores = matmul(q, k.transposed(0, 1, 3, 2)) * MLXArray(scale)
        if let attentionMask {
            let mask = attentionMask.asType(.float32)
                .expandedDimensions(axis: 1)
                .expandedDimensions(axis: 1)
            let additive = MLX.where(mask .> MLXArray(0.5), MLXArray(0.0), MLXArray(-1e9))
            scores = scores + additive
        }

        let weights = softmax(scores, axis: -1)
        let output = matmul(weights, v).transposed(0, 2, 1, 3).reshaped(
            batchSize,
            sequenceLength,
            TADATTSAlignerDefaults.hiddenSize
        )
        return outProj(output)
    }
}

private final class TADATTSAlignerFeedForward: Module {
    @ModuleInfo(key: "intermediate_dense") var intermediateDense: Linear
    @ModuleInfo(key: "output_dense") var outputDense: Linear

    override init() {
        _intermediateDense.wrappedValue = Linear(
            TADATTSAlignerDefaults.hiddenSize,
            TADATTSAlignerDefaults.intermediateSize
        )
        _outputDense.wrappedValue = Linear(
            TADATTSAlignerDefaults.intermediateSize,
            TADATTSAlignerDefaults.hiddenSize
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        outputDense(gelu(intermediateDense(x)))
    }
}

private final class TADATTSAlignerEncoderLayer: Module {
    @ModuleInfo var attention: TADATTSAlignerAttention
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
    @ModuleInfo(key: "feed_forward") var feedForward: TADATTSAlignerFeedForward
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: LayerNorm

    override init() {
        _attention.wrappedValue = TADATTSAlignerAttention()
        _layerNorm.wrappedValue = LayerNorm(
            dimensions: TADATTSAlignerDefaults.hiddenSize,
            eps: TADATTSAlignerDefaults.layerNormEps
        )
        _feedForward.wrappedValue = TADATTSAlignerFeedForward()
        _finalLayerNorm.wrappedValue = LayerNorm(
            dimensions: TADATTSAlignerDefaults.hiddenSize,
            eps: TADATTSAlignerDefaults.layerNormEps
        )
    }

    func callAsFunction(_ x: MLXArray, attentionMask: MLXArray?) -> MLXArray {
        var hiddenStates = x + attention(layerNorm(x), attentionMask: attentionMask)
        hiddenStates = hiddenStates + feedForward(finalLayerNorm(hiddenStates))
        return hiddenStates
    }
}

private final class TADATTSAlignerEncoder: Module {
    @ModuleInfo(key: "pos_conv_embed") var posConvEmbed: TADATTSAlignerPositionalConvEmbedding
    @ModuleInfo var layers: [TADATTSAlignerEncoderLayer]
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm

    override init() {
        _posConvEmbed.wrappedValue = TADATTSAlignerPositionalConvEmbedding()
        _layers.wrappedValue = (0..<TADATTSAlignerDefaults.numHiddenLayers).map { _ in
            TADATTSAlignerEncoderLayer()
        }
        _layerNorm.wrappedValue = LayerNorm(
            dimensions: TADATTSAlignerDefaults.hiddenSize,
            eps: TADATTSAlignerDefaults.layerNormEps
        )
    }

    func callAsFunction(_ x: MLXArray, attentionMask: MLXArray?) -> MLXArray {
        var hiddenStates = x
        hiddenStates = hiddenStates + posConvEmbed(hiddenStates)
        for layer in layers {
            hiddenStates = layer(hiddenStates, attentionMask: attentionMask)
        }
        return layerNorm(hiddenStates)
    }
}

public final class TADATTSAligner: Module {
    @ModuleInfo(key: "feature_extractor") fileprivate var featureExtractor: TADATTSAlignerFeatureExtractor
    @ModuleInfo(key: "feature_projection") fileprivate var featureProjection: TADATTSAlignerFeatureProjection
    @ModuleInfo fileprivate var encoder: TADATTSAlignerEncoder
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public override init() {
        _featureExtractor.wrappedValue = TADATTSAlignerFeatureExtractor()
        _featureProjection.wrappedValue = TADATTSAlignerFeatureProjection()
        _encoder.wrappedValue = TADATTSAlignerEncoder()
        _lmHead.wrappedValue = Linear(
            TADATTSAlignerDefaults.hiddenSize,
            TADATTSAlignerDefaults.vocabSize
        )
    }

    public func callAsFunction(
        _ waveform: MLXArray,
        attentionMask: MLXArray? = nil
    ) -> MLXArray {
        var normalized = waveform
        let meanValue = mean(normalized, axis: -1, keepDims: true)
        let variance = mean((normalized - meanValue) * (normalized - meanValue), axis: -1, keepDims: true)
        normalized = (normalized - meanValue) / sqrt(variance + MLXArray(Float(1e-7)))

        var hiddenStates = normalized.expandedDimensions(axis: -1)
        hiddenStates = featureExtractor(hiddenStates)

        let reducedAttentionMask = attentionMask.map { Self.reduceAttentionMask($0) }
        let projected = featureProjection(hiddenStates)
        hiddenStates = encoder(projected.hiddenStates, attentionMask: reducedAttentionMask)
        return logSoftmax(lmHead(hiddenStates), axis: -1)
    }

    public func align(
        waveform: MLXArray,
        tokenIDs: MLXArray,
        attentionMask: MLXArray? = nil,
        blankTokenID: Int = TADATTSAlignerDefaults.blankTokenID
    ) -> TADATTSAlignmentResult {
        let batchedTokenIDs: MLXArray
        if tokenIDs.ndim == 1 {
            batchedTokenIDs = tokenIDs.expandedDimensions(axis: 0)
        } else {
            batchedTokenIDs = tokenIDs
        }

        let logProbs = callAsFunction(waveform, attentionMask: attentionMask)
        let batchSize = batchedTokenIDs.dim(0)
        let frameCount = logProbs.dim(1)

        var allPositions: [[Int32]] = []
        var allMasks: [[Int32]] = []
        allPositions.reserveCapacity(batchSize)
        allMasks.reserveCapacity(batchSize)

        for batchIndex in 0..<batchSize {
            let tokens = batchedTokenIDs[batchIndex].asArray(Int.self).filter { $0 >= 0 }
            // Use Python-faithful forward DP (align_text_tokens) instead of CTC Viterbi.
            // Positions returned are 0-indexed; add 1 to match encoder's 1-indexed convention.
            let (positions0, mask) = Self.forwardAlignTokens(
                logProbs: logProbs[batchIndex],
                tokenIDs: tokens
            )
            allPositions.append(positions0.map { Int32($0 + 1) })
            allMasks.append(mask.map(Int32.init))
        }

        let maxTokenCount = allPositions.map(\.count).max() ?? 0
        let paddedPositions = allPositions.flatMap { row in
            row + Array(repeating: Int32(0), count: max(0, maxTokenCount - row.count))
        }
        let paddedMasks = allMasks.flatMap { row in
            row + Array(repeating: Int32(0), count: max(0, frameCount - row.count))
        }

        return TADATTSAlignmentResult(
            tokenPositions: MLXArray(paddedPositions, [batchSize, maxTokenCount]).asType(.int32),
            tokenMasks: MLXArray(paddedMasks, [batchSize, frameCount]).asType(.int32)
        )
    }

    /// Forward DP alignment matching Python `align_text_tokens` in mlx_tada/aligner.py.
    /// Finds the best monotonically-increasing frame assignment for each text token
    /// using a max-sum DP (no blank tokens — direct token-to-frame alignment).
    ///
    /// Returns 0-indexed positions and a per-frame binary mask.
    static func forwardAlignTokens(
        logProbs: MLXArray,
        tokenIDs: [Int]
    ) -> (positions: [Int], mask: [Int]) {
        let frameCount = logProbs.dim(0)
        guard !tokenIDs.isEmpty else {
            return ([], Array(repeating: 0, count: frameCount))
        }

        let T = tokenIDs.count
        let negInf = -Float.greatestFiniteMagnitude

        // Extract per-frame scores for each text token: tokenScores[i][j] = logProbs[frame=i, tokenID=tokenIDs[j]]
        let tokenScores: [[Float]] = (0..<frameCount).map { frame in
            tokenIDs.map { tokenID in logProbs[frame, tokenID].item(Float.self) }
        }

        // dp[i][j]: best cumulative score placing tokens 0..j into frames 0..i
        var dp = Array(repeating: Array(repeating: negInf, count: T), count: frameCount)
        // bp[i][j]: frame where token j was placed (== i), or -1 meaning "token j placed before frame i"
        //           For column j=0: stores the cum-max frame index for the first token.
        var bp = Array(repeating: Array(repeating: 0, count: T), count: frameCount)

        // First column (j=0): cumulative-max over frames, tracking which frame had the best score
        var cumMaxVal = tokenScores[0][0]
        var cumMaxIdx = 0
        dp[0][0] = cumMaxVal
        bp[0][0] = 0

        for i in 1..<frameCount {
            if tokenScores[i][0] > cumMaxVal {
                cumMaxVal = tokenScores[i][0]
                cumMaxIdx = i
            }
            dp[i][0] = cumMaxVal
            bp[i][0] = cumMaxIdx
        }

        // Diagonal initialization when T <= frameCount (ensures valid initial path)
        if T <= frameCount {
            var cumsum = Float(0)
            for k in 0..<T {
                cumsum += tokenScores[k][k]
                dp[k][k] = cumsum
                bp[k][k] = k
            }
        }

        // Main DP: for each frame i, for each token j in 1..<min(i, T)
        for i in 1..<frameCount {
            let maxJ = min(i, T)
            guard maxJ > 1 else { continue }
            for j in 1..<maxJ {
                let skipScore = dp[i - 1][j]              // token j already placed, skip frame i
                let useScore = dp[i - 1][j - 1] + tokenScores[i][j]  // place token j at frame i
                if useScore >= skipScore {
                    dp[i][j] = useScore
                    bp[i][j] = i     // token j was placed at frame i
                } else {
                    dp[i][j] = skipScore
                    bp[i][j] = -1   // token j placed before frame i (skip sentinel)
                }
            }
        }

        // Backtrack from (frameCount-1, T-1) to assign each token its frame
        var positions = Array(repeating: 0, count: T)
        var i = frameCount - 1
        var j = T - 1
        var posIdx = T - 1

        while j >= 0 {
            if j == 0 {
                positions[posIdx] = bp[i][0]  // cumulative-max frame for first token
                break
            } else if bp[i][j] == -1 {
                i -= 1  // skip: move to previous frame, same token
            } else {
                positions[posIdx] = bp[i][j]  // token j placed at this frame
                posIdx -= 1
                i -= 1
                j -= 1
            }
        }

        // Build per-frame binary mask marking where tokens were placed
        var frameMask = Array(repeating: 0, count: frameCount)
        for pos in positions where pos >= 0 && pos < frameCount {
            frameMask[pos] = 1
        }

        return (positions, frameMask)
    }

    public static func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        var weightG: [String: MLXArray] = [:]
        var weightV: [String: MLXArray] = [:]

        for (key, var value) in weights {
            var mappedKey = key

            if mappedKey.hasPrefix("aligner.") {
                mappedKey = String(mappedKey.dropFirst("aligner.".count))
            } else if mappedKey.hasPrefix("wav2vec2.") {
                mappedKey = String(mappedKey.dropFirst("wav2vec2.".count))
            }

            if mappedKey.contains("masked_spec_embed") || mappedKey.hasPrefix("projector.") || mappedKey.hasPrefix("classifier.") {
                continue
            }

            if mappedKey.hasSuffix(".parametrizations.weight.original0") {
                mappedKey = mappedKey.replacingOccurrences(
                    of: ".parametrizations.weight.original0",
                    with: ".weight_g"
                )
            } else if mappedKey.hasSuffix(".parametrizations.weight.original1") {
                mappedKey = mappedKey.replacingOccurrences(
                    of: ".parametrizations.weight.original1",
                    with: ".weight_v"
                )
            }

            if mappedKey.hasSuffix(".weight_g") {
                weightG[String(mappedKey.dropLast(".weight_g".count))] = value
                continue
            }

            if mappedKey.hasSuffix(".weight_v") {
                weightV[String(mappedKey.dropLast(".weight_v".count))] = value
                continue
            }

            if mappedKey.hasSuffix(".conv.weight") && value.ndim == 3 {
                value = value.transposed(0, 2, 1)
            }

            sanitized[mappedKey] = value
        }

        for baseKey in Set(weightG.keys).union(weightV.keys) {
            guard let g = weightG[baseKey], let v = weightV[baseKey] else { continue }
            let norm = sqrt(sum(v * v, axes: [0, 1], keepDims: true) + MLXArray(Float(1e-12)))
            var fused = g * (v / norm)
            if fused.ndim == 3 {
                fused = fused.transposed(0, 2, 1)
            }
            sanitized["\(baseKey).weight"] = fused
        }

        return sanitized
    }

    private static func reduceAttentionMask(_ mask: MLXArray) -> MLXArray {
        let batchSize = mask.dim(0)
        let lengths = sum(mask.asType(.int32), axis: -1).asArray(Int.self)
        let reducedLengths = lengths.map { rawLength in
            TADATTSAlignerDefaults.convKernel.enumerated().reduce(rawLength) { currentLength, pair in
                let nextLength = max(0, currentLength - pair.element)
                return nextLength / TADATTSAlignerDefaults.convStride[pair.offset] + 1
            }
        }
        let maxLength = reducedLengths.max() ?? 0
        var flattened: [Int32] = []
        flattened.reserveCapacity(batchSize * maxLength)

        for length in reducedLengths {
            flattened.append(contentsOf: (0..<maxLength).map { $0 < length ? 1 : 0 })
        }

        return MLXArray(flattened, [batchSize, maxLength]).asType(.int32)
    }

    static func viterbiAlignment(
        logProbs: MLXArray,
        tokenIDs: [Int],
        blankTokenID: Int
    ) -> (positions: [Int], mask: [Int]) {
        let frameCount = logProbs.dim(0)
        guard !tokenIDs.isEmpty else {
            return ([], Array(repeating: 0, count: frameCount))
        }

        var states: [Int] = [blankTokenID]
        for tokenID in tokenIDs {
            states.append(tokenID)
            states.append(blankTokenID)
        }

        let stateCount = states.count
        let negativeInfinity = -Float.greatestFiniteMagnitude
        var previous = Array(repeating: negativeInfinity, count: stateCount)
        var backpointers = Array(
            repeating: Array(repeating: 0, count: stateCount),
            count: frameCount
        )

        previous[0] = logProbs[0, blankTokenID].item(Float.self)
        if stateCount > 1 {
            previous[1] = logProbs[0, states[1]].item(Float.self)
            backpointers[0][1] = 1
        }

        if frameCount > 1 {
            for frame in 1..<frameCount {
                var current = Array(repeating: negativeInfinity, count: stateCount)

                for state in 0..<stateCount {
                    let tokenID = states[state]
                    let emission = logProbs[frame, tokenID].item(Float.self)

                    var bestState = state
                    var bestScore = previous[state]

                    if state > 0, previous[state - 1] > bestScore {
                        bestScore = previous[state - 1]
                        bestState = state - 1
                    }

                    if state > 1 && tokenID != blankTokenID && tokenID != states[state - 2] && previous[state - 2] > bestScore {
                        bestScore = previous[state - 2]
                        bestState = state - 2
                    }

                    current[state] = bestScore + emission
                    backpointers[frame][state] = bestState
                }

                previous = current
            }
        }

        let finalState: Int
        if stateCount > 1 && previous[stateCount - 2] > previous[stateCount - 1] {
            finalState = stateCount - 2
        } else {
            finalState = stateCount - 1
        }

        var path = Array(repeating: 0, count: frameCount)
        var state = finalState
        for frame in stride(from: frameCount - 1, through: 0, by: -1) {
            path[frame] = state
            state = backpointers[frame][state]
        }

        var positions = Array(repeating: 1, count: tokenIDs.count)
        var mask = Array(repeating: 0, count: frameCount)

        for tokenIndex in tokenIDs.indices {
            let tokenState = tokenIndex * 2 + 1
            let frames = path.enumerated().compactMap { frame, state in
                state == tokenState ? frame : nil
            }
            let chosenFrame = frames.last ?? (tokenIndex > 0 ? positions[tokenIndex - 1] - 1 : 0)
            positions[tokenIndex] = chosenFrame + 1
            if chosenFrame < mask.count {
                mask[chosenFrame] = 1
            }
        }

        return (positions, mask)
    }
}
