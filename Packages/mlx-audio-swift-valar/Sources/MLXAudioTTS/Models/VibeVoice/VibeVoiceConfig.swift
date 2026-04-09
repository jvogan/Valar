import Foundation
import MLXLMCommon

// MARK: - Qwen2 Decoder Config

/// Configuration for the Qwen2-0.5B decoder backbone shared by `language_model` (4 lower layers)
/// and `tts_language_model` (20 upper layers).
public struct VibeVoiceQwen2DecoderConfig: Codable, Sendable, Equatable {
    public var modelType: String
    public var hiddenSize: Int
    public var intermediateSize: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var numHiddenLayers: Int
    public var rmsNormEps: Float
    public var vocabSize: Int
    public var maxPositionEmbeddings: Int
    public var ropeTheta: Float
    public var tieWordEmbeddings: Bool
    public var headDim: Int?
    public var attentionDropout: Float
    public var hiddenAct: String
    public var maxWindowLayers: Int
    public var ropeScaling: [String: StringOrNumber]?
    public var slidingWindow: Int?
    public var useCache: Bool
    public var useSlidingWindow: Bool

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case numHiddenLayers = "num_hidden_layers"
        case rmsNormEps = "rms_norm_eps"
        case vocabSize = "vocab_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeTheta = "rope_theta"
        case tieWordEmbeddings = "tie_word_embeddings"
        case headDim = "head_dim"
        case attentionDropout = "attention_dropout"
        case hiddenAct = "hidden_act"
        case maxWindowLayers = "max_window_layers"
        case ropeScaling = "rope_scaling"
        case slidingWindow = "sliding_window"
        case useCache = "use_cache"
        case useSlidingWindow = "use_sliding_window"
    }

    public init(
        modelType: String = "qwen2",
        hiddenSize: Int = 896,
        intermediateSize: Int = 4_864,
        numAttentionHeads: Int = 14,
        numKeyValueHeads: Int = 2,
        numHiddenLayers: Int = 24,
        rmsNormEps: Float = 1e-6,
        vocabSize: Int = 151_936,
        maxPositionEmbeddings: Int = 8_192,
        ropeTheta: Float = 1_000_000.0,
        tieWordEmbeddings: Bool = false,
        headDim: Int? = nil,
        attentionDropout: Float = 0.0,
        hiddenAct: String = "silu",
        maxWindowLayers: Int = 24,
        ropeScaling: [String: StringOrNumber]? = nil,
        slidingWindow: Int? = nil,
        useCache: Bool = true,
        useSlidingWindow: Bool = false
    ) {
        self.modelType = modelType
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.numHiddenLayers = numHiddenLayers
        self.rmsNormEps = rmsNormEps
        self.vocabSize = vocabSize
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.ropeTheta = ropeTheta
        self.tieWordEmbeddings = tieWordEmbeddings
        self.headDim = headDim
        self.attentionDropout = attentionDropout
        self.hiddenAct = hiddenAct
        self.maxWindowLayers = maxWindowLayers
        self.ropeScaling = ropeScaling
        self.slidingWindow = slidingWindow
        self.useCache = useCache
        self.useSlidingWindow = useSlidingWindow
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "qwen2"
        hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 896
        intermediateSize = try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 4_864
        numAttentionHeads = try container.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 14
        numKeyValueHeads = try container.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 2
        numHiddenLayers = try container.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 24
        rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        vocabSize = try container.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 151_936
        maxPositionEmbeddings = try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 8_192
        ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000.0
        tieWordEmbeddings = try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        headDim = try container.decodeIfPresent(Int.self, forKey: .headDim)
        attentionDropout = try container.decodeIfPresent(Float.self, forKey: .attentionDropout) ?? 0.0
        hiddenAct = try container.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "silu"
        maxWindowLayers = try container.decodeIfPresent(Int.self, forKey: .maxWindowLayers) ?? 24
        ropeScaling = try container.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling)
        slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow)
        useCache = try container.decodeIfPresent(Bool.self, forKey: .useCache) ?? true
        useSlidingWindow = try container.decodeIfPresent(Bool.self, forKey: .useSlidingWindow) ?? false
    }

    /// Effective head dimension: explicit `head_dim` from config, or `hidden_size / num_attention_heads`.
    public var effectiveHeadDim: Int {
        headDim ?? (hiddenSize / numAttentionHeads)
    }

    /// Number of layers assigned to `language_model` (lower stack, no final norm).
    public func languageModelLayerCount(ttsBackboneNumHiddenLayers: Int) -> Int {
        numHiddenLayers - ttsBackboneNumHiddenLayers
    }
}

// MARK: - Diffusion Head Config

/// Configuration for the DDPM-style diffusion prediction head with AdaLN conditioning.
public struct VibeVoiceDiffusionHeadConfig: Codable, Sendable, Equatable {
    public var modelType: String
    public var hiddenSize: Int
    public var headLayers: Int
    public var headFfnRatio: Float
    public var rmsNormEps: Float
    public var latentSize: Int
    public var speechVaeDim: Int?
    public var predictionType: String
    public var diffusionType: String
    public var ddpmNumSteps: Int
    public var ddpmNumInferenceSteps: Int
    public var ddpmBetaSchedule: String
    public var ddpmBatchMul: Int

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case headLayers = "head_layers"
        case headFfnRatio = "head_ffn_ratio"
        case rmsNormEps = "rms_norm_eps"
        case latentSize = "latent_size"
        case speechVaeDim = "speech_vae_dim"
        case predictionType = "prediction_type"
        case diffusionType = "diffusion_type"
        case ddpmNumSteps = "ddpm_num_steps"
        case ddpmNumInferenceSteps = "ddpm_num_inference_steps"
        case ddpmBetaSchedule = "ddpm_beta_schedule"
        case ddpmBatchMul = "ddpm_batch_mul"
    }

    public init(
        modelType: String = "vibevoice_diffusion_head",
        hiddenSize: Int = 896,
        headLayers: Int = 4,
        headFfnRatio: Float = 3.0,
        rmsNormEps: Float = 1e-5,
        latentSize: Int = 64,
        speechVaeDim: Int? = 64,
        predictionType: String = "v_prediction",
        diffusionType: String = "ddpm",
        ddpmNumSteps: Int = 1_000,
        ddpmNumInferenceSteps: Int = 20,
        ddpmBetaSchedule: String = "cosine",
        ddpmBatchMul: Int = 4
    ) {
        self.modelType = modelType
        self.hiddenSize = hiddenSize
        self.headLayers = headLayers
        self.headFfnRatio = headFfnRatio
        self.rmsNormEps = rmsNormEps
        self.latentSize = latentSize
        self.speechVaeDim = speechVaeDim
        self.predictionType = predictionType
        self.diffusionType = diffusionType
        self.ddpmNumSteps = ddpmNumSteps
        self.ddpmNumInferenceSteps = ddpmNumInferenceSteps
        self.ddpmBetaSchedule = ddpmBetaSchedule
        self.ddpmBatchMul = ddpmBatchMul
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "vibevoice_diffusion_head"
        hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 896
        headLayers = try container.decodeIfPresent(Int.self, forKey: .headLayers) ?? 4
        headFfnRatio = try container.decodeIfPresent(Float.self, forKey: .headFfnRatio) ?? 3.0
        rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        latentSize = try container.decodeIfPresent(Int.self, forKey: .latentSize) ?? 64
        speechVaeDim = try container.decodeIfPresent(Int.self, forKey: .speechVaeDim)
        predictionType = try container.decodeIfPresent(String.self, forKey: .predictionType) ?? "v_prediction"
        diffusionType = try container.decodeIfPresent(String.self, forKey: .diffusionType) ?? "ddpm"
        ddpmNumSteps = try container.decodeIfPresent(Int.self, forKey: .ddpmNumSteps) ?? 1_000
        ddpmNumInferenceSteps = try container.decodeIfPresent(Int.self, forKey: .ddpmNumInferenceSteps) ?? 20
        ddpmBetaSchedule = try container.decodeIfPresent(String.self, forKey: .ddpmBetaSchedule) ?? "cosine"
        ddpmBatchMul = try container.decodeIfPresent(Int.self, forKey: .ddpmBatchMul) ?? 4
    }

    /// FFN hidden dimension: `int(hidden_size * head_ffn_ratio)`.
    public var ffnDim: Int {
        Int(Float(hiddenSize) * headFfnRatio)
    }
}

// MARK: - Acoustic Tokenizer Config

/// Configuration for the causal VAE decoder that converts speech latents to audio waveforms.
public struct VibeVoiceAcousticTokenizerConfig: Codable, Sendable, Equatable {
    public var modelType: String
    public var channels: Int
    public var vaeDim: Int
    public var causal: Bool
    public var fixStd: Float
    public var encoderNFilters: Int
    public var encoderRatios: [Int]
    public var encoderDepths: String
    public var mixerLayer: String
    public var convBias: Bool
    public var layerScaleInitValue: Float
    public var layernorm: String
    public var layernormEps: Float
    public var layernormElementwiseAffine: Bool
    public var corpusNormalize: Float
    public var stdDistType: String
    public var convNorm: String
    public var padMode: String
    public var disableLastNorm: Bool
    public var weightInitValue: Float
    public var decoderNFilters: Int
    public var decoderRatios: [Int]?
    public var decoderDepths: String?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case channels
        case vaeDim = "vae_dim"
        case causal
        case fixStd = "fix_std"
        case encoderNFilters = "encoder_n_filters"
        case encoderRatios = "encoder_ratios"
        case encoderDepths = "encoder_depths"
        case mixerLayer = "mixer_layer"
        case convBias = "conv_bias"
        case layerScaleInitValue = "layer_scale_init_value"
        case layernorm
        case layernormEps = "layernorm_eps"
        case layernormElementwiseAffine = "layernorm_elementwise_affine"
        case corpusNormalize = "corpus_normalize"
        case stdDistType = "std_dist_type"
        case convNorm = "conv_norm"
        case padMode = "pad_mode"
        case disableLastNorm = "disable_last_norm"
        case weightInitValue = "weight_init_value"
        case decoderNFilters = "decoder_n_filters"
        case decoderRatios = "decoder_ratios"
        case decoderDepths = "decoder_depths"
    }

    public init(
        modelType: String = "vibevoice_acoustic_tokenizer",
        channels: Int = 1,
        vaeDim: Int = 64,
        causal: Bool = true,
        fixStd: Float = 0.5,
        encoderNFilters: Int = 32,
        encoderRatios: [Int] = [8, 5, 5, 4, 2, 2],
        encoderDepths: String = "3-3-3-3-3-3-8",
        mixerLayer: String = "depthwise_conv",
        convBias: Bool = true,
        layerScaleInitValue: Float = 1e-6,
        layernorm: String = "RMSNorm",
        layernormEps: Float = 1e-5,
        layernormElementwiseAffine: Bool = true,
        corpusNormalize: Float = 0.0,
        stdDistType: String = "gaussian",
        convNorm: String = "none",
        padMode: String = "constant",
        disableLastNorm: Bool = true,
        weightInitValue: Float = 0.01,
        decoderNFilters: Int = 32,
        decoderRatios: [Int]? = nil,
        decoderDepths: String? = nil
    ) {
        self.modelType = modelType
        self.channels = channels
        self.vaeDim = vaeDim
        self.causal = causal
        self.fixStd = fixStd
        self.encoderNFilters = encoderNFilters
        self.encoderRatios = encoderRatios
        self.encoderDepths = encoderDepths
        self.mixerLayer = mixerLayer
        self.convBias = convBias
        self.layerScaleInitValue = layerScaleInitValue
        self.layernorm = layernorm
        self.layernormEps = layernormEps
        self.layernormElementwiseAffine = layernormElementwiseAffine
        self.corpusNormalize = corpusNormalize
        self.stdDistType = stdDistType
        self.convNorm = convNorm
        self.padMode = padMode
        self.disableLastNorm = disableLastNorm
        self.weightInitValue = weightInitValue
        self.decoderNFilters = decoderNFilters
        self.decoderRatios = decoderRatios
        self.decoderDepths = decoderDepths
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "vibevoice_acoustic_tokenizer"
        channels = try container.decodeIfPresent(Int.self, forKey: .channels) ?? 1
        vaeDim = try container.decodeIfPresent(Int.self, forKey: .vaeDim) ?? 64
        causal = try container.decodeIfPresent(Bool.self, forKey: .causal) ?? true
        fixStd = try container.decodeIfPresent(Float.self, forKey: .fixStd) ?? 0.5
        encoderNFilters = try container.decodeIfPresent(Int.self, forKey: .encoderNFilters) ?? 32
        encoderRatios = try container.decodeIfPresent([Int].self, forKey: .encoderRatios) ?? [8, 5, 5, 4, 2, 2]
        encoderDepths = try container.decodeIfPresent(String.self, forKey: .encoderDepths) ?? "3-3-3-3-3-3-8"
        mixerLayer = try container.decodeIfPresent(String.self, forKey: .mixerLayer) ?? "depthwise_conv"
        convBias = try container.decodeIfPresent(Bool.self, forKey: .convBias) ?? true
        layerScaleInitValue = try container.decodeIfPresent(Float.self, forKey: .layerScaleInitValue) ?? 1e-6
        layernorm = try container.decodeIfPresent(String.self, forKey: .layernorm) ?? "RMSNorm"
        layernormEps = try container.decodeIfPresent(Float.self, forKey: .layernormEps) ?? 1e-5
        layernormElementwiseAffine = try container.decodeIfPresent(Bool.self, forKey: .layernormElementwiseAffine) ?? true
        corpusNormalize = try container.decodeIfPresent(Float.self, forKey: .corpusNormalize) ?? 0.0
        stdDistType = try container.decodeIfPresent(String.self, forKey: .stdDistType) ?? "gaussian"
        convNorm = try container.decodeIfPresent(String.self, forKey: .convNorm) ?? "none"
        padMode = try container.decodeIfPresent(String.self, forKey: .padMode) ?? "constant"
        disableLastNorm = try container.decodeIfPresent(Bool.self, forKey: .disableLastNorm) ?? true
        weightInitValue = try container.decodeIfPresent(Float.self, forKey: .weightInitValue) ?? 0.01
        decoderNFilters = try container.decodeIfPresent(Int.self, forKey: .decoderNFilters) ?? 32
        decoderRatios = try container.decodeIfPresent([Int].self, forKey: .decoderRatios)
        decoderDepths = try container.decodeIfPresent(String.self, forKey: .decoderDepths)
    }

    /// Total upsampling factor: product of all encoder ratios (mirrored for decoder).
    /// At 24kHz sample rate this gives the hop length per latent frame.
    public var hopLength: Int {
        encoderRatios.reduce(1, *)
    }

    /// Parsed encoder depth values from the hyphen-separated depth string.
    public var parsedEncoderDepths: [Int] {
        encoderDepths.split(separator: "-").compactMap { Int($0) }
    }

    /// Parsed decoder depth values (reversed encoder depths for decoder).
    public var parsedDecoderDepths: [Int] {
        if let decoderDepths {
            return decoderDepths.split(separator: "-").compactMap { Int($0) }
        }
        return parsedEncoderDepths.reversed()
    }

    /// Effective decoder ratios (reversed encoder ratios if not explicitly set).
    public var effectiveDecoderRatios: [Int] {
        decoderRatios ?? encoderRatios.reversed()
    }
}

// MARK: - Top-level Model Config

/// Top-level VibeVoice model configuration, containing three sub-configs and model-wide parameters.
///
/// Matches the `config.json` layout at:
/// `mlx-community/VibeVoice-Realtime-0.5B-4bit/config.json`
public struct VibeVoiceModelConfig: Codable, Sendable, Equatable {
    public var modelType: String
    public var sampleRate: Int
    public var acousticVaeDim: Int
    public var ttsBackboneNumHiddenLayers: Int

    public var decoderConfig: VibeVoiceQwen2DecoderConfig
    public var acousticTokenizerConfig: VibeVoiceAcousticTokenizerConfig
    public var diffusionHeadConfig: VibeVoiceDiffusionHeadConfig

    public var quantization: BaseConfiguration.Quantization?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case sampleRate = "sample_rate"
        case acousticVaeDim = "acoustic_vae_dim"
        case ttsBackboneNumHiddenLayers = "tts_backbone_num_hidden_layers"
        case decoderConfig = "decoder_config"
        case acousticTokenizerConfig = "acoustic_tokenizer_config"
        case diffusionHeadConfig = "diffusion_head_config"
        case quantization
        case quantizationConfig = "quantization_config"
    }

    public init(
        modelType: String = "vibevoice_streaming",
        sampleRate: Int = 24_000,
        acousticVaeDim: Int = 64,
        ttsBackboneNumHiddenLayers: Int = 20,
        decoderConfig: VibeVoiceQwen2DecoderConfig = VibeVoiceQwen2DecoderConfig(),
        acousticTokenizerConfig: VibeVoiceAcousticTokenizerConfig = VibeVoiceAcousticTokenizerConfig(),
        diffusionHeadConfig: VibeVoiceDiffusionHeadConfig = VibeVoiceDiffusionHeadConfig(),
        quantization: BaseConfiguration.Quantization? = nil
    ) {
        self.modelType = modelType
        self.sampleRate = sampleRate
        self.acousticVaeDim = acousticVaeDim
        self.ttsBackboneNumHiddenLayers = ttsBackboneNumHiddenLayers
        self.decoderConfig = decoderConfig
        self.acousticTokenizerConfig = acousticTokenizerConfig
        self.diffusionHeadConfig = diffusionHeadConfig
        self.quantization = quantization
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "vibevoice_streaming"
        sampleRate = try container.decodeIfPresent(Int.self, forKey: .sampleRate) ?? 24_000
        acousticVaeDim = try container.decodeIfPresent(Int.self, forKey: .acousticVaeDim) ?? 64
        ttsBackboneNumHiddenLayers = try container.decodeIfPresent(Int.self, forKey: .ttsBackboneNumHiddenLayers) ?? 20
        decoderConfig = try container.decodeIfPresent(VibeVoiceQwen2DecoderConfig.self, forKey: .decoderConfig)
            ?? VibeVoiceQwen2DecoderConfig()
        acousticTokenizerConfig = try container.decodeIfPresent(VibeVoiceAcousticTokenizerConfig.self, forKey: .acousticTokenizerConfig)
            ?? VibeVoiceAcousticTokenizerConfig()
        diffusionHeadConfig = try container.decodeIfPresent(VibeVoiceDiffusionHeadConfig.self, forKey: .diffusionHeadConfig)
            ?? VibeVoiceDiffusionHeadConfig()

        // Accept quantization from either "quantization" or "quantization_config" key
        quantization = try container.decodeIfPresent(BaseConfiguration.Quantization.self, forKey: .quantization)
            ?? container.decodeIfPresent(BaseConfiguration.Quantization.self, forKey: .quantizationConfig)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)

        try container.encode(modelType, forKey: .modelType)
        try container.encode(sampleRate, forKey: .sampleRate)
        try container.encode(acousticVaeDim, forKey: .acousticVaeDim)
        try container.encode(ttsBackboneNumHiddenLayers, forKey: .ttsBackboneNumHiddenLayers)
        try container.encode(decoderConfig, forKey: .decoderConfig)
        try container.encode(acousticTokenizerConfig, forKey: .acousticTokenizerConfig)
        try container.encode(diffusionHeadConfig, forKey: .diffusionHeadConfig)
        try container.encodeIfPresent(quantization, forKey: .quantization)
    }

    // MARK: - Derived Constants

    /// Number of layers in `language_model` (lower text stack, no final norm).
    public var languageModelLayers: Int {
        decoderConfig.numHiddenLayers - ttsBackboneNumHiddenLayers
    }

    /// Number of layers in `tts_language_model` (upper TTS stack, with final norm).
    public var ttsLanguageModelLayers: Int {
        ttsBackboneNumHiddenLayers
    }

    /// Effective head dimension from the decoder config.
    public var headDim: Int {
        decoderConfig.effectiveHeadDim
    }

    /// Total upsampling factor from the acoustic tokenizer (hop length).
    public var hopLength: Int {
        acousticTokenizerConfig.hopLength
    }

    // MARK: - Special Tokens

    /// Special token constants for VibeVoice generation.
    ///
    /// The Qwen2.5-0.5B tokenizer is used with `add_special_tokens=False`.
    /// Text is encoded as `text.strip() + "\n"` — no BOS/EOS wrapping.
    ///
    /// Input type embeddings:
    /// - `0` = speech token (latent fed back via `acoustic_connector`)
    /// - `1` = text token (from `language_model.embed_tokens`)
    public enum SpecialTokens {
        /// Input type ID for speech tokens (latents fed back into TTS LM).
        public static let speechTypeID: Int = 0
        /// Input type ID for text tokens (from Qwen2 embedding table).
        public static let textTypeID: Int = 1
    }

    // MARK: - Generation Constants

    /// Number of text tokens consumed per text window step.
    public static let ttsTextWindowSize: Int = 5
    /// Number of speech latent steps generated per text window.
    public static let ttsSpeechWindowSize: Int = 6
    /// Number of speech latent steps emitted per streaming chunk.
    ///
    /// Keeping this below `ttsSpeechWindowSize` reduces warm first-chunk latency without
    /// changing the non-streaming decode path.
    public static let streamingSpeechWindowSize: Int = 4
    /// Approximate speech-latent emission rate used to convert requested stream
    /// intervals into per-chunk latent counts.
    public static let streamingSpeechFrameRateHz: Double = 12.5
    /// Default classifier-free guidance scale.
    public static let defaultCFGScale: Float = 1.5
    /// Default end-of-speech threshold used by the EOS classifier.
    ///
    /// The upstream VibeVoice realtime loop terminates once the sigmoid score clears 0.5.
    /// Keeping the Swift port aligned avoids long babbling tails on weaker multilingual presets.
    public static let defaultEOSThreshold: Float = 0.5
    /// Default maximum speech latent steps.
    public static let defaultMaxTokens: Int = 512
}
