import Foundation
import MLXLMCommon

public struct TADATTSRoPEScalingConfig: Codable, Sendable, Equatable {
    public var factor: Float
    public var highFreqFactor: Float
    public var lowFreqFactor: Float
    public var originalMaxPositionEmbeddings: Int
    public var ropeType: String

    enum CodingKeys: String, CodingKey {
        case factor
        case highFreqFactor = "high_freq_factor"
        case lowFreqFactor = "low_freq_factor"
        case originalMaxPositionEmbeddings = "original_max_position_embeddings"
        case ropeType = "rope_type"
    }

    public init(
        factor: Float = 32.0,
        highFreqFactor: Float = 4.0,
        lowFreqFactor: Float = 1.0,
        originalMaxPositionEmbeddings: Int = 8_192,
        ropeType: String = "llama3"
    ) {
        self.factor = factor
        self.highFreqFactor = highFreqFactor
        self.lowFreqFactor = lowFreqFactor
        self.originalMaxPositionEmbeddings = originalMaxPositionEmbeddings
        self.ropeType = ropeType
    }

    var asLlamaScaling: [String: StringOrNumber] {
        [
            "factor": .float(Float(factor)),
            "high_freq_factor": .float(Float(highFreqFactor)),
            "low_freq_factor": .float(Float(lowFreqFactor)),
            "original_max_position_embeddings": .int(originalMaxPositionEmbeddings),
            "rope_type": .string(ropeType)
        ]
    }
}

public struct TADATTSConfig: Codable, Sendable, Equatable {
    public var vocabSize: Int
    public var hiddenSize: Int
    public var intermediateSize: Int
    public var numHiddenLayers: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var headDim: Int
    public var rmsNormEps: Float
    public var ropeTheta: Float
    public var ropeScaling: TADATTSRoPEScalingConfig?
    public var tieWordEmbeddings: Bool
    public var maxPositionEmbeddings: Int
    public var acousticDim: Int
    public var numTimeClasses: Int
    public var shiftAcoustic: Int
    public var headLayers: Int
    public var headFfnRatio: Float
    public var bottleneckDim: Int?
    public var acousticMean: Float
    public var acousticStd: Float
    public var bosTokenID: Int
    public var eosTokenIDs: [Int]
    public var eotID: Int

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
        case bosTokenID = "bos_token_id"
        case eosTokenIDs = "eos_token_id"
        case eotID = "eot_id"
    }

    public init(
        vocabSize: Int = 128_256,
        hiddenSize: Int = 3_072,
        intermediateSize: Int = 8_192,
        numHiddenLayers: Int = 28,
        numAttentionHeads: Int = 24,
        numKeyValueHeads: Int = 8,
        headDim: Int = 128,
        rmsNormEps: Float = 1e-5,
        ropeTheta: Float = 500_000,
        ropeScaling: TADATTSRoPEScalingConfig? = TADATTSRoPEScalingConfig(),
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
        bosTokenID: Int = 128_000,
        eosTokenIDs: [Int] = [128_001, 128_008, 128_009],
        eotID: Int = 128_009
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
        self.bosTokenID = bosTokenID
        self.eosTokenIDs = eosTokenIDs
        self.eotID = eotID
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        vocabSize = try container.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 128_256
        hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 3_072
        intermediateSize = try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 8_192
        numHiddenLayers = try container.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 28
        numAttentionHeads = try container.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 24
        numKeyValueHeads = try container.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 8
        headDim = try container.decodeIfPresent(Int.self, forKey: .headDim) ?? 128
        rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 500_000
        ropeScaling = try container.decodeIfPresent(TADATTSRoPEScalingConfig.self, forKey: .ropeScaling)
            ?? TADATTSRoPEScalingConfig()
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
        bosTokenID = try container.decodeIfPresent(Int.self, forKey: .bosTokenID) ?? 128_000

        if let array = try? container.decode([Int].self, forKey: .eosTokenIDs) {
            eosTokenIDs = array
        } else if let single = try? container.decode(Int.self, forKey: .eosTokenIDs) {
            eosTokenIDs = [single]
        } else {
            eosTokenIDs = [128_001, 128_008, 128_009]
        }

        eotID = try container.decodeIfPresent(Int.self, forKey: .eotID) ?? 128_009
    }

    public var llamaConfiguration: LlamaTTSConfiguration {
        LlamaTTSConfiguration(
            hiddenSize: hiddenSize,
            hiddenLayers: numHiddenLayers,
            intermediateSize: intermediateSize,
            attentionHeads: numAttentionHeads,
            headDimensions: headDim,
            rmsNormEps: rmsNormEps,
            vocabularySize: vocabSize,
            kvHeads: numKeyValueHeads,
            maxPositionEmbeddings: maxPositionEmbeddings,
            ropeTheta: ropeTheta,
            ropeTraditional: false,
            ropeScaling: ropeScaling?.asLlamaScaling,
            tieWordEmbeddings: tieWordEmbeddings,
            attentionBias: false,
            mlpBias: false,
            sampleRate: 24_000,
            tokenizerName: nil
        )
    }
}
