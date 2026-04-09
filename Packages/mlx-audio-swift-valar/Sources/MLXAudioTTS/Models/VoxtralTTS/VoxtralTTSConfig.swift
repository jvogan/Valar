import Foundation

private struct VoxtralTTSMultimodalConfig: Decodable {
    let audioModelArgs: VoxtralTTSAudioModelConfig?
    let audioTokenizerArgs: VoxtralTTSCodecConfig?
    let audioTokenId: Int?
    let beginAudioTokenId: Int?

    enum CodingKeys: String, CodingKey {
        case audioModelArgs = "audio_model_args"
        case audioTokenizerArgs = "audio_tokenizer_args"
        case audioTokenId = "audio_token_id"
        case beginAudioTokenId = "begin_audio_token_id"
    }
}

public struct VoxtralTTSBackboneConfig: Codable, Sendable {
    public var dim: Int
    public var nLayers: Int
    public var headDim: Int
    public var hiddenDim: Int
    public var nHeads: Int
    public var nKvHeads: Int
    public var vocabSize: Int
    public var ropeTheta: Float
    public var normEps: Float
    public var tiedEmbeddings: Bool
    public var causal: Bool
    public var maxSeqLen: Int
    public var modelType: String

    enum CodingKeys: String, CodingKey {
        case dim
        case nLayers = "n_layers"
        case headDim = "head_dim"
        case hiddenDim = "hidden_dim"
        case nHeads = "n_heads"
        case nKvHeads = "n_kv_heads"
        case vocabSize = "vocab_size"
        case ropeTheta = "rope_theta"
        case normEps = "norm_eps"
        case tiedEmbeddings = "tied_embeddings"
        case causal
        case maxSeqLen = "max_seq_len"
        case modelType = "model_type"
    }

    public init(
        dim: Int = 3072,
        nLayers: Int = 26,
        headDim: Int = 128,
        hiddenDim: Int = 9216,
        nHeads: Int = 32,
        nKvHeads: Int = 8,
        vocabSize: Int = 131072,
        ropeTheta: Float = 1_000_000,
        normEps: Float = 1e-5,
        tiedEmbeddings: Bool = true,
        causal: Bool = true,
        maxSeqLen: Int = 65_536,
        modelType: String = "voxtral_tts"
    ) {
        self.dim = dim
        self.nLayers = nLayers
        self.headDim = headDim
        self.hiddenDim = hiddenDim
        self.nHeads = nHeads
        self.nKvHeads = nKvHeads
        self.vocabSize = vocabSize
        self.ropeTheta = ropeTheta
        self.normEps = normEps
        self.tiedEmbeddings = tiedEmbeddings
        self.causal = causal
        self.maxSeqLen = maxSeqLen
        self.modelType = modelType
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        dim = try c.decodeIfPresent(Int.self, forKey: .dim) ?? 3072
        nLayers = try c.decodeIfPresent(Int.self, forKey: .nLayers) ?? 26
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 128
        hiddenDim = try c.decodeIfPresent(Int.self, forKey: .hiddenDim) ?? 9216
        nHeads = try c.decodeIfPresent(Int.self, forKey: .nHeads) ?? 32
        nKvHeads = try c.decodeIfPresent(Int.self, forKey: .nKvHeads) ?? 8
        vocabSize = try c.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 131072
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000
        normEps = try c.decodeIfPresent(Float.self, forKey: .normEps) ?? 1e-5
        tiedEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tiedEmbeddings) ?? true
        causal = try c.decodeIfPresent(Bool.self, forKey: .causal) ?? true
        maxSeqLen = try c.decodeIfPresent(Int.self, forKey: .maxSeqLen) ?? 65_536
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "voxtral_tts"
    }
}

public struct VoxtralTTSAudioEncodingConfig: Codable, Sendable {
    public var codebookPattern: String
    public var numCodebooks: Int
    public var samplingRate: Int
    public var frameRate: Float

    enum CodingKeys: String, CodingKey {
        case codebookPattern = "codebook_pattern"
        case numCodebooks = "num_codebooks"
        case samplingRate = "sampling_rate"
        case frameRate = "frame_rate"
    }

    public init(
        codebookPattern: String = "parallel",
        numCodebooks: Int = 37,
        samplingRate: Int = 24_000,
        frameRate: Float = 12.5
    ) {
        self.codebookPattern = codebookPattern
        self.numCodebooks = numCodebooks
        self.samplingRate = samplingRate
        self.frameRate = frameRate
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        codebookPattern = try c.decodeIfPresent(String.self, forKey: .codebookPattern) ?? "parallel"
        numCodebooks = try c.decodeIfPresent(Int.self, forKey: .numCodebooks) ?? 37
        samplingRate = try c.decodeIfPresent(Int.self, forKey: .samplingRate) ?? 24_000
        frameRate = try c.decodeIfPresent(Float.self, forKey: .frameRate) ?? 12.5
    }
}

public struct VoxtralTTSAcousticTransformerConfig: Codable, Sendable {
    public var inputDim: Int
    public var dim: Int
    public var nLayers: Int
    public var headDim: Int
    public var hiddenDim: Int
    public var nHeads: Int
    public var nKvHeads: Int
    public var useBiases: Bool
    public var ropeTheta: Float
    public var sigma: Float
    public var sigmaMax: Float

    enum CodingKeys: String, CodingKey {
        case inputDim = "input_dim"
        case dim
        case nLayers = "n_layers"
        case headDim = "head_dim"
        case hiddenDim = "hidden_dim"
        case nHeads = "n_heads"
        case nKvHeads = "n_kv_heads"
        case useBiases = "use_biases"
        case ropeTheta = "rope_theta"
        case sigma
        case sigmaMax = "sigma_max"
    }

    public init(
        inputDim: Int = 3072,
        dim: Int = 768,
        nLayers: Int = 3,
        headDim: Int = 128,
        hiddenDim: Int = 2048,
        nHeads: Int = 6,
        nKvHeads: Int = 2,
        useBiases: Bool = false,
        ropeTheta: Float = 10_000,
        sigma: Float = 1e-5,
        sigmaMax: Float = 1.0
    ) {
        self.inputDim = inputDim
        self.dim = dim
        self.nLayers = nLayers
        self.headDim = headDim
        self.hiddenDim = hiddenDim
        self.nHeads = nHeads
        self.nKvHeads = nKvHeads
        self.useBiases = useBiases
        self.ropeTheta = ropeTheta
        self.sigma = sigma
        self.sigmaMax = sigmaMax
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        inputDim = try c.decodeIfPresent(Int.self, forKey: .inputDim) ?? 3072
        dim = try c.decodeIfPresent(Int.self, forKey: .dim) ?? 768
        nLayers = try c.decodeIfPresent(Int.self, forKey: .nLayers) ?? 3
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 128
        hiddenDim = try c.decodeIfPresent(Int.self, forKey: .hiddenDim) ?? 2048
        nHeads = try c.decodeIfPresent(Int.self, forKey: .nHeads) ?? 6
        nKvHeads = try c.decodeIfPresent(Int.self, forKey: .nKvHeads) ?? 2
        useBiases = try c.decodeIfPresent(Bool.self, forKey: .useBiases) ?? false
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000
        sigma = try c.decodeIfPresent(Float.self, forKey: .sigma) ?? 1e-5
        sigmaMax = try c.decodeIfPresent(Float.self, forKey: .sigmaMax) ?? 1.0
    }
}

public struct VoxtralTTSAudioModelConfig: Codable, Sendable {
    public var semanticCodebookSize: Int
    public var acousticCodebookSize: Int
    public var nAcousticCodebook: Int
    public var audioEncodingArgs: VoxtralTTSAudioEncodingConfig
    public var acousticTransformerArgs: VoxtralTTSAcousticTransformerConfig

    enum CodingKeys: String, CodingKey {
        case semanticCodebookSize = "semantic_codebook_size"
        case acousticCodebookSize = "acoustic_codebook_size"
        case nAcousticCodebook = "n_acoustic_codebook"
        case audioEncodingArgs = "audio_encoding_args"
        case acousticTransformerArgs = "acoustic_transformer_args"
    }

    public init(
        semanticCodebookSize: Int = 8192,
        acousticCodebookSize: Int = 21,
        nAcousticCodebook: Int = 36,
        audioEncodingArgs: VoxtralTTSAudioEncodingConfig = VoxtralTTSAudioEncodingConfig(),
        acousticTransformerArgs: VoxtralTTSAcousticTransformerConfig = VoxtralTTSAcousticTransformerConfig()
    ) {
        self.semanticCodebookSize = semanticCodebookSize
        self.acousticCodebookSize = acousticCodebookSize
        self.nAcousticCodebook = nAcousticCodebook
        self.audioEncodingArgs = audioEncodingArgs
        self.acousticTransformerArgs = acousticTransformerArgs
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        semanticCodebookSize = try c.decodeIfPresent(Int.self, forKey: .semanticCodebookSize) ?? 8192
        acousticCodebookSize = try c.decodeIfPresent(Int.self, forKey: .acousticCodebookSize) ?? 21
        nAcousticCodebook = try c.decodeIfPresent(Int.self, forKey: .nAcousticCodebook) ?? 36
        audioEncodingArgs = try c.decodeIfPresent(VoxtralTTSAudioEncodingConfig.self, forKey: .audioEncodingArgs)
            ?? VoxtralTTSAudioEncodingConfig()
        acousticTransformerArgs = try c.decodeIfPresent(VoxtralTTSAcousticTransformerConfig.self, forKey: .acousticTransformerArgs)
            ?? VoxtralTTSAcousticTransformerConfig()
    }
}

public struct VoxtralTTSCodecConfig: Codable, Sendable {
    public var channels: Int
    public var samplingRate: Int
    public var pretransformPatchSize: Int
    public var patchProjKernelSize: Int
    public var semanticCodebookSize: Int
    public var semanticDim: Int
    public var acousticCodebookSize: Int
    public var acousticDim: Int
    public var convWeightNorm: Bool
    public var causal: Bool
    public var attnSlidingWindowSize: Int
    public var halfAttnWindowUponDownsampling: Bool
    public var dim: Int
    public var hiddenDim: Int
    public var headDim: Int
    public var nHeads: Int
    public var nKvHeads: Int
    public var qkNormEps: Float
    public var qkNorm: Bool
    public var useBiases: Bool
    public var normEps: Float
    public var layerScale: Bool
    public var layerScaleInit: Float
    public var decoderTransformerLengthsString: String
    public var decoderConvsKernelsString: String
    public var decoderConvsStridesString: String

    enum CodingKeys: String, CodingKey {
        case channels
        case samplingRate = "sampling_rate"
        case pretransformPatchSize = "pretransform_patch_size"
        case patchProjKernelSize = "patch_proj_kernel_size"
        case semanticCodebookSize = "semantic_codebook_size"
        case semanticDim = "semantic_dim"
        case acousticCodebookSize = "acoustic_codebook_size"
        case acousticDim = "acoustic_dim"
        case convWeightNorm = "conv_weight_norm"
        case causal
        case attnSlidingWindowSize = "attn_sliding_window_size"
        case halfAttnWindowUponDownsampling = "half_attn_window_upon_downsampling"
        case dim
        case hiddenDim = "hidden_dim"
        case headDim = "head_dim"
        case nHeads = "n_heads"
        case nKvHeads = "n_kv_heads"
        case qkNormEps = "qk_norm_eps"
        case qkNorm = "qk_norm"
        case useBiases = "use_biases"
        case normEps = "norm_eps"
        case layerScale = "layer_scale"
        case layerScaleInit = "layer_scale_init"
        case decoderTransformerLengthsString = "decoder_transformer_lengths_str"
        case decoderConvsKernelsString = "decoder_convs_kernels_str"
        case decoderConvsStridesString = "decoder_convs_strides_str"
    }

    public init(
        channels: Int = 1,
        samplingRate: Int = 24_000,
        pretransformPatchSize: Int = 240,
        patchProjKernelSize: Int = 7,
        semanticCodebookSize: Int = 8192,
        semanticDim: Int = 256,
        acousticCodebookSize: Int = 21,
        acousticDim: Int = 36,
        convWeightNorm: Bool = true,
        causal: Bool = true,
        attnSlidingWindowSize: Int = 16,
        halfAttnWindowUponDownsampling: Bool = true,
        dim: Int = 1024,
        hiddenDim: Int = 4096,
        headDim: Int = 128,
        nHeads: Int = 8,
        nKvHeads: Int = 8,
        qkNormEps: Float = 1e-6,
        qkNorm: Bool = true,
        useBiases: Bool = false,
        normEps: Float = 0.01,
        layerScale: Bool = true,
        layerScaleInit: Float = 0.01,
        decoderTransformerLengthsString: String = "2,2,2,2",
        decoderConvsKernelsString: String = "3,4,4,4",
        decoderConvsStridesString: String = "1,2,2,2"
    ) {
        self.channels = channels
        self.samplingRate = samplingRate
        self.pretransformPatchSize = pretransformPatchSize
        self.patchProjKernelSize = patchProjKernelSize
        self.semanticCodebookSize = semanticCodebookSize
        self.semanticDim = semanticDim
        self.acousticCodebookSize = acousticCodebookSize
        self.acousticDim = acousticDim
        self.convWeightNorm = convWeightNorm
        self.causal = causal
        self.attnSlidingWindowSize = attnSlidingWindowSize
        self.halfAttnWindowUponDownsampling = halfAttnWindowUponDownsampling
        self.dim = dim
        self.hiddenDim = hiddenDim
        self.headDim = headDim
        self.nHeads = nHeads
        self.nKvHeads = nKvHeads
        self.qkNormEps = qkNormEps
        self.qkNorm = qkNorm
        self.useBiases = useBiases
        self.normEps = normEps
        self.layerScale = layerScale
        self.layerScaleInit = layerScaleInit
        self.decoderTransformerLengthsString = decoderTransformerLengthsString
        self.decoderConvsKernelsString = decoderConvsKernelsString
        self.decoderConvsStridesString = decoderConvsStridesString
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        channels = try c.decodeIfPresent(Int.self, forKey: .channels) ?? 1
        samplingRate = try c.decodeIfPresent(Int.self, forKey: .samplingRate) ?? 24_000
        pretransformPatchSize = try c.decodeIfPresent(Int.self, forKey: .pretransformPatchSize) ?? 240
        patchProjKernelSize = try c.decodeIfPresent(Int.self, forKey: .patchProjKernelSize) ?? 7
        semanticCodebookSize = try c.decodeIfPresent(Int.self, forKey: .semanticCodebookSize) ?? 8192
        semanticDim = try c.decodeIfPresent(Int.self, forKey: .semanticDim) ?? 256
        acousticCodebookSize = try c.decodeIfPresent(Int.self, forKey: .acousticCodebookSize) ?? 21
        acousticDim = try c.decodeIfPresent(Int.self, forKey: .acousticDim) ?? 36
        convWeightNorm = try c.decodeIfPresent(Bool.self, forKey: .convWeightNorm) ?? true
        causal = try c.decodeIfPresent(Bool.self, forKey: .causal) ?? true
        attnSlidingWindowSize = try c.decodeIfPresent(Int.self, forKey: .attnSlidingWindowSize) ?? 16
        halfAttnWindowUponDownsampling = try c.decodeIfPresent(Bool.self, forKey: .halfAttnWindowUponDownsampling) ?? true
        dim = try c.decodeIfPresent(Int.self, forKey: .dim) ?? 1024
        hiddenDim = try c.decodeIfPresent(Int.self, forKey: .hiddenDim) ?? 4096
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 128
        nHeads = try c.decodeIfPresent(Int.self, forKey: .nHeads) ?? 8
        nKvHeads = try c.decodeIfPresent(Int.self, forKey: .nKvHeads) ?? 8
        qkNormEps = try c.decodeIfPresent(Float.self, forKey: .qkNormEps) ?? 1e-6
        qkNorm = try c.decodeIfPresent(Bool.self, forKey: .qkNorm) ?? true
        useBiases = try c.decodeIfPresent(Bool.self, forKey: .useBiases) ?? false
        normEps = try c.decodeIfPresent(Float.self, forKey: .normEps) ?? 0.01
        layerScale = try c.decodeIfPresent(Bool.self, forKey: .layerScale) ?? true
        layerScaleInit = try c.decodeIfPresent(Float.self, forKey: .layerScaleInit) ?? 0.01
        decoderTransformerLengthsString = try c.decodeIfPresent(String.self, forKey: .decoderTransformerLengthsString) ?? "2,2,2,2"
        decoderConvsKernelsString = try c.decodeIfPresent(String.self, forKey: .decoderConvsKernelsString) ?? "3,4,4,4"
        decoderConvsStridesString = try c.decodeIfPresent(String.self, forKey: .decoderConvsStridesString) ?? "1,2,2,2"
    }

    public var decoderTransformerLengths: [Int] {
        Self.parseIntegerList(decoderTransformerLengthsString, fallback: [2, 2, 2, 2])
    }

    public var decoderConvsKernels: [Int] {
        Self.parseIntegerList(decoderConvsKernelsString, fallback: [3, 4, 4, 4])
    }

    public var decoderConvsStrides: [Int] {
        Self.parseIntegerList(decoderConvsStridesString, fallback: [1, 2, 2, 2])
    }

    public var samplesPerFrame: Int {
        pretransformPatchSize * decoderConvsStrides.reduce(1, *)
    }

    private static func parseIntegerList(_ value: String, fallback: [Int]) -> [Int] {
        let parsed = value.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
        return parsed.isEmpty ? fallback : parsed
    }
}

public struct VoxtralTTSConfig: Codable, Sendable {
    public var backbone: VoxtralTTSBackboneConfig
    public var audioModelArgs: VoxtralTTSAudioModelConfig
    public var audioTokenizerArgs: VoxtralTTSCodecConfig
    public var audioTokenId: Int
    public var beginAudioTokenId: Int

    enum CodingKeys: String, CodingKey {
        case backbone
        case audioModelArgs = "audio_model_args"
        case audioTokenizerArgs = "audio_tokenizer_args"
        case audioTokenId = "audio_token_id"
        case beginAudioTokenId = "begin_audio_token_id"
    }

    private enum DecodeOnlyCodingKeys: String, CodingKey {
        case multimodal
    }

    public init(
        backbone: VoxtralTTSBackboneConfig = VoxtralTTSBackboneConfig(),
        audioModelArgs: VoxtralTTSAudioModelConfig = VoxtralTTSAudioModelConfig(),
        audioTokenizerArgs: VoxtralTTSCodecConfig = VoxtralTTSCodecConfig(),
        audioTokenId: Int = 24,
        beginAudioTokenId: Int = 25
    ) {
        self.backbone = backbone
        self.audioModelArgs = audioModelArgs
        self.audioTokenizerArgs = audioTokenizerArgs
        self.audioTokenId = audioTokenId
        self.beginAudioTokenId = beginAudioTokenId
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let decodeOnly = try decoder.container(keyedBy: DecodeOnlyCodingKeys.self)
        let multimodal = try decodeOnly.decodeIfPresent(VoxtralTTSMultimodalConfig.self, forKey: .multimodal)
        backbone = try VoxtralTTSBackboneConfig(from: decoder)
        if let directAudioModelArgs = try c.decodeIfPresent(VoxtralTTSAudioModelConfig.self, forKey: .audioModelArgs) {
            audioModelArgs = directAudioModelArgs
        } else if let nestedAudioModelArgs = multimodal?.audioModelArgs {
            audioModelArgs = nestedAudioModelArgs
        } else {
            audioModelArgs = try VoxtralTTSAudioModelConfig(from: decoder)
        }

        if let directAudioTokenizerArgs = try c.decodeIfPresent(VoxtralTTSCodecConfig.self, forKey: .audioTokenizerArgs) {
            audioTokenizerArgs = directAudioTokenizerArgs
        } else if let nestedAudioTokenizerArgs = multimodal?.audioTokenizerArgs {
            audioTokenizerArgs = nestedAudioTokenizerArgs
        } else {
            audioTokenizerArgs = try VoxtralTTSCodecConfig(from: decoder)
        }

        audioTokenId = try c.decodeIfPresent(Int.self, forKey: .audioTokenId)
            ?? multimodal?.audioTokenId
            ?? 24
        beginAudioTokenId = try c.decodeIfPresent(Int.self, forKey: .beginAudioTokenId)
            ?? multimodal?.beginAudioTokenId
            ?? 25
    }

    public var modelType: String { backbone.modelType }
    public var sampleRate: Int { audioTokenizerArgs.samplingRate }
    public var frameRate: Float { audioModelArgs.audioEncodingArgs.frameRate }
    public var totalCodebooks: Int { audioModelArgs.audioEncodingArgs.numCodebooks }
    public var samplesPerFrame: Int { audioTokenizerArgs.samplesPerFrame }
}

public enum VoxtralTTSPresetVoice {
    public static let idsByName: [String: Int] = [
        "casual_female": 0,
        "casual_male": 1,
        "cheerful_female": 2,
        "neutral_female": 3,
        "neutral_male": 4,
        "pt_male": 5,
        "pt_female": 6,
        "nl_male": 7,
        "nl_female": 8,
        "it_male": 9,
        "it_female": 10,
        "fr_male": 11,
        "fr_female": 12,
        "es_male": 13,
        "es_female": 14,
        "de_male": 15,
        "de_female": 16,
        "ar_male": 17,
        "hi_male": 18,
        "hi_female": 19,
    ]
}
