import Foundation

public struct TokenizationRequest: Codable, Sendable, Hashable {
    public let model: ModelIdentifier
    public let text: String

    public init(model: ModelIdentifier, text: String) {
        self.model = model
        self.text = text
    }
}

public struct TokenizationResult: Codable, Sendable, Equatable, Hashable {
    public let model: ModelIdentifier
    public let tokenCount: Int
    public let chunkCount: Int

    public init(model: ModelIdentifier, tokenCount: Int, chunkCount: Int) {
        self.model = model
        self.tokenCount = tokenCount
        self.chunkCount = chunkCount
    }
}

public enum SpeechSynthesisVoiceBehavior: String, Codable, Sendable, Hashable, CaseIterable {
    case auto
    case expressive
    case stableNarrator
}

public enum VoiceKind: String, Codable, Sendable, Hashable, CaseIterable {
    case preset
    case namedSpeaker
    case legacyPrompt
    case clonePrompt
    case embeddingOnly
    case tadaReference
}

public struct VoiceProfile: Codable, Sendable, Hashable {
    public let id: UUID
    public let label: String
    public let backendVoiceID: String?
    public let sourceModel: ModelIdentifier
    public let localeIdentifier: String?
    public let runtimeModel: ModelIdentifier?
    public let referenceAudioAssetName: String?
    public let referenceTranscript: String?
    public let speakerEmbedding: Data?
    public let conditioningFormat: String?
    /// Pre-loaded asset files for asset-backed conditioning formats (e.g. TADA).
    /// `nil` means the bundle has not been loaded yet; an empty array means it was
    /// loaded but contained no binary payloads (treat as unavailable).
    public let conditioningAssets: [VoiceConditioningAssetFile]?
    /// Metadata decoded from a loaded conditioning bundle (token counts, dims, etc.).
    public let conditioningMetadata: VoiceConditioningMetadata?
    public let voiceKind: VoiceKind?
    public let isLegacyExpressive: Bool

    public static let qwenSpeakerEmbeddingConditioningFormat = "qwen.speaker_embedding/v1"
    public static let qwenClonePromptConditioningFormat = "qwen.clone_prompt/v1"
    public static let tadaReferenceConditioningFormat = "tada.reference/v1"

    public init(
        id: UUID = UUID(),
        label: String,
        backendVoiceID: String? = nil,
        sourceModel: ModelIdentifier,
        localeIdentifier: String? = nil,
        runtimeModel: ModelIdentifier? = nil,
        referenceAudioAssetName: String? = nil,
        referenceTranscript: String? = nil,
        speakerEmbedding: Data? = nil,
        conditioningFormat: String? = nil,
        conditioningAssets: [VoiceConditioningAssetFile]? = nil,
        conditioningMetadata: VoiceConditioningMetadata? = nil,
        voiceKind: VoiceKind? = nil,
        isLegacyExpressive: Bool = false
    ) {
        self.id = id
        self.label = label
        self.backendVoiceID = backendVoiceID
        self.sourceModel = sourceModel
        self.localeIdentifier = localeIdentifier
        self.runtimeModel = runtimeModel
        self.referenceAudioAssetName = referenceAudioAssetName
        self.referenceTranscript = referenceTranscript
        self.speakerEmbedding = speakerEmbedding
        self.conditioningFormat = conditioningFormat
        self.conditioningAssets = conditioningAssets
        self.conditioningMetadata = conditioningMetadata
        self.voiceKind = voiceKind
        self.isLegacyExpressive = isLegacyExpressive
    }

    public var voiceSelector: String {
        backendVoiceID ?? label
    }

    /// Derives a `VoiceConditioning` value from persisted fields.
    ///
    /// - For Qwen voices: reconstructs from `speakerEmbedding` when available.
    /// - For TADA voices: returns a reference conditioning whose `assetFiles` are
    ///   populated when `conditioningAssets` has been pre-loaded by the caller,
    ///   or empty when the bundle has not yet been loaded (WAV fallback will apply).
    public var conditioning: VoiceConditioning? {
        if let conditioningFormat {
            switch conditioningFormat {
            case Self.qwenSpeakerEmbeddingConditioningFormat:
                if let speakerEmbedding {
                    return .qwenSpeakerEmbedding(speakerEmbedding)
                }
                return nil
            case Self.qwenClonePromptConditioningFormat:
                return .qwenClonePrompt(speakerEmbedding)
            case Self.tadaReferenceConditioningFormat:
                return .tadaReference(
                    assetFiles: conditioningAssets ?? [],
                    assetName: "\(id.uuidString)-tada",
                    sourceModel: sourceModel,
                    metadata: conditioningMetadata
                )
            default:
                if let speakerEmbedding {
                    return VoiceConditioning(format: conditioningFormat, payload: speakerEmbedding)
                }
            }
        }
        if let speakerEmbedding {
            return .qwenSpeakerEmbedding(speakerEmbedding)
        }
        return nil
    }

    public var isModelDeclaredPreset: Bool {
        guard let backendVoiceID else { return false }
        return !backendVoiceID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    public func validateCompatibility(with model: ModelIdentifier, familyID: ModelFamilyID) throws {
        let voiceFamily = sourceModel.inferredFamilyHint
        let conditioningFamily = inferredConditioningFamily()

        if isModelDeclaredPreset, voiceFamily == .voxtralTTS, familyID != .voxtralTTS {
            throw VoiceProfileCompatibilityError.voxtralPresetRequiresVoxtralModel(
                voiceLabel: label,
                backendVoiceID: voiceSelector,
                requestedModel: model,
                requestedFamily: familyID
            )
        }

        if familyID == .voxtralTTS,
           (voiceFamily != .voxtralTTS || isModelDeclaredPreset == false) {
            throw VoiceProfileCompatibilityError.voxtralModelRequiresPresetVoice(
                voiceLabel: label,
                sourceModel: sourceModel,
                requestedModel: model
            )
        }

        guard let conditioningFamily else {
            return
        }

        guard conditioningFamily != familyID else {
            return
        }

        switch (conditioningFamily, familyID) {
        case (.qwen3TTS, .tadaTTS):
            throw VoiceProfileCompatibilityError.qwenConditioningRequiresQwenModel(
                voiceLabel: label,
                sourceModel: sourceModel,
                conditioningFormat: conditioningFormat ?? Self.qwenSpeakerEmbeddingConditioningFormat,
                requestedModel: model
            )
        case (.tadaTTS, .qwen3TTS):
            throw VoiceProfileCompatibilityError.tadaConditioningRequiresTadaModel(
                voiceLabel: label,
                sourceModel: sourceModel,
                conditioningFormat: conditioningFormat ?? Self.tadaReferenceConditioningFormat,
                requestedModel: model
            )
        default:
            throw VoiceProfileCompatibilityError.conditioningRequiresMatchingFamily(
                voiceLabel: label,
                sourceModel: sourceModel,
                conditioningFormat: conditioningFormat,
                conditioningFamily: conditioningFamily,
                requestedModel: model,
                requestedFamily: familyID
            )
        }
    }

    private func inferredConditioningFamily() -> ModelFamilyID? {
        if isModelDeclaredPreset {
            return nil
        }

        if let conditioningFormat {
            switch conditioningFormat {
            case Self.qwenSpeakerEmbeddingConditioningFormat:
                return .qwen3TTS
            case Self.qwenClonePromptConditioningFormat:
                return .qwen3TTS
            case Self.tadaReferenceConditioningFormat:
                return .tadaTTS
            default:
                break
            }
        }

        guard hasConditioningPayload else {
            return nil
        }

        let inferred = runtimeModel?.inferredFamilyHint ?? sourceModel.inferredFamilyHint
        return inferred == .unknown ? nil : inferred
    }

    private var hasConditioningPayload: Bool {
        if speakerEmbedding != nil { return true }
        if conditioningFormat != nil { return true }
        if let referenceAudioAssetName,
           !referenceAudioAssetName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return true
        }
        if let referenceTranscript,
           !referenceTranscript.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return true
        }
        return false
    }
}

public enum VoiceProfileCompatibilityError: LocalizedError, Equatable, Sendable {
    case voxtralPresetRequiresVoxtralModel(
        voiceLabel: String,
        backendVoiceID: String,
        requestedModel: ModelIdentifier,
        requestedFamily: ModelFamilyID
    )
    case voxtralModelRequiresPresetVoice(
        voiceLabel: String,
        sourceModel: ModelIdentifier,
        requestedModel: ModelIdentifier
    )
    case qwenConditioningRequiresQwenModel(
        voiceLabel: String,
        sourceModel: ModelIdentifier,
        conditioningFormat: String,
        requestedModel: ModelIdentifier
    )
    case tadaConditioningRequiresTadaModel(
        voiceLabel: String,
        sourceModel: ModelIdentifier,
        conditioningFormat: String,
        requestedModel: ModelIdentifier
    )
    case conditioningRequiresMatchingFamily(
        voiceLabel: String,
        sourceModel: ModelIdentifier,
        conditioningFormat: String?,
        conditioningFamily: ModelFamilyID,
        requestedModel: ModelIdentifier,
        requestedFamily: ModelFamilyID
    )

    public var errorDescription: String? {
        switch self {
        case .voxtralPresetRequiresVoxtralModel(
            let voiceLabel,
            let backendVoiceID,
            let requestedModel,
            let requestedFamily
        ):
            return "Voice '\(voiceLabel)' uses Voxtral preset '\(backendVoiceID)' and can only be used with Voxtral TTS models. Requested model '\(requestedModel.rawValue)' belongs to '\(requestedFamily.rawValue)'."
        case .voxtralModelRequiresPresetVoice(let voiceLabel, let sourceModel, let requestedModel):
            return "Voice '\(voiceLabel)' comes from '\(sourceModel.rawValue)' and cannot be used with Voxtral model '\(requestedModel.rawValue)'. Voxtral only supports model-declared preset voices."
        case .qwenConditioningRequiresQwenModel(let voiceLabel, _, let conditioningFormat, let requestedModel):
            return "Voice '\(voiceLabel)' has Qwen conditioning (\(conditioningFormat)) that is incompatible with model '\(requestedModel.rawValue)'. Use a Qwen TTS model."
        case .tadaConditioningRequiresTadaModel(let voiceLabel, _, let conditioningFormat, let requestedModel):
            return "Voice '\(voiceLabel)' has TADA conditioning (\(conditioningFormat)) that is incompatible with model '\(requestedModel.rawValue)'. Use a TADA TTS model."
        case .conditioningRequiresMatchingFamily(let voiceLabel, _, let conditioningFormat, let conditioningFamily, let requestedModel, let requestedFamily):
            return "Voice '\(voiceLabel)' has conditioning (\(conditioningFormat ?? "inferred")) from family '\(conditioningFamily.rawValue)' that is incompatible with model '\(requestedModel.rawValue)' (family '\(requestedFamily.rawValue)')."
        }
    }
}

// MARK: - Voice Conditioning

public struct VoiceConditioning: Codable, Sendable, Equatable, Hashable {
    public static let qwenSpeakerEmbeddingV1 = "qwen.speaker_embedding/v1"
    public static let qwenClonePromptV1 = "qwen.clone_prompt/v1"
    public static let tadaReferenceV1 = "tada.reference/v1"

    public let format: String
    public let payload: Data?
    public let assetFiles: [VoiceConditioningAssetFile]
    public let assetName: String?
    public let sourceModel: ModelIdentifier?
    public let metadata: VoiceConditioningMetadata?

    public init(
        format: String,
        payload: Data? = nil,
        assetFiles: [VoiceConditioningAssetFile] = [],
        assetName: String? = nil,
        sourceModel: ModelIdentifier? = nil,
        metadata: VoiceConditioningMetadata? = nil
    ) {
        self.format = format
        self.payload = payload
        self.assetFiles = assetFiles
        self.assetName = assetName
        self.sourceModel = sourceModel
        self.metadata = metadata
    }

    public static func qwenSpeakerEmbedding(_ data: Data) -> VoiceConditioning {
        VoiceConditioning(format: qwenSpeakerEmbeddingV1, payload: data)
    }

    public static func qwenClonePrompt(_ cachedSpeakerEmbedding: Data? = nil) -> VoiceConditioning {
        VoiceConditioning(format: qwenClonePromptV1, payload: cachedSpeakerEmbedding)
    }

    public static func tadaReference(
        assetFiles: [VoiceConditioningAssetFile] = [],
        assetName: String? = nil,
        sourceModel: ModelIdentifier? = nil,
        metadata: VoiceConditioningMetadata? = nil
    ) -> VoiceConditioning {
        VoiceConditioning(
            format: tadaReferenceV1,
            payload: nil,
            assetFiles: assetFiles,
            assetName: assetName,
            sourceModel: sourceModel,
            metadata: metadata
        )
    }
}

public struct VoiceConditioningMetadata: Codable, Sendable, Equatable, Hashable {
    public let modelID: String?
    public let transcript: String?
    public let language: String?
    public let sampleRate: Double?
    public let tokenCount: Int?
    public let acousticDimension: Int?
    public let frameCount: Int?

    public init(
        modelID: String? = nil,
        transcript: String? = nil,
        language: String? = nil,
        sampleRate: Double? = nil,
        tokenCount: Int? = nil,
        acousticDimension: Int? = nil,
        frameCount: Int? = nil
    ) {
        self.modelID = modelID
        self.transcript = transcript
        self.language = language
        self.sampleRate = sampleRate
        self.tokenCount = tokenCount
        self.acousticDimension = acousticDimension
        self.frameCount = frameCount
    }
}

public struct VoiceConditioningAssetFile: Codable, Sendable, Equatable, Hashable {
    public let filename: String
    public let data: Data

    public init(filename: String, data: Data) {
        self.filename = filename
        self.data = data
    }
}

public struct SpeechSynthesisRequest: Codable, Sendable, Hashable {
    public let model: ModelIdentifier
    public let text: String
    public let voice: VoiceProfile?
    public let language: String?
    public let referenceAudioAssetName: String?
    public let referenceAudioPCMFloat32LE: Data?
    /// Pre-decoded mono samples — preferred over referenceAudioPCMFloat32LE when available,
    /// avoiding a [Float]→Data→[Float] round-trip on the cache-hit path.
    public let referenceAudioSamples: [Float]?
    public let referenceAudioSampleRate: Double?
    public let referenceTranscript: String?
    public let instruct: String?
    public let exaggeration: Float?
    public let cfgWeight: Float?
    public let sampleRate: Double
    public let responseFormat: String
    public let temperature: Float?
    public let topP: Float?
    public let repetitionPenalty: Float?
    public let repetitionContextSize: Int?
    public let maxTokens: Int?
    public let voiceBehavior: SpeechSynthesisVoiceBehavior

    public init(
        model: ModelIdentifier,
        text: String,
        voice: VoiceProfile? = nil,
        language: String? = nil,
        referenceAudioAssetName: String? = nil,
        referenceAudioPCMFloat32LE: Data? = nil,
        referenceAudioSamples: [Float]? = nil,
        referenceAudioSampleRate: Double? = nil,
        referenceTranscript: String? = nil,
        instruct: String? = nil,
        exaggeration: Float? = nil,
        cfgWeight: Float? = nil,
        sampleRate: Double = 24_000,
        responseFormat: String = "wav",
        temperature: Float? = nil,
        topP: Float? = nil,
        repetitionPenalty: Float? = nil,
        repetitionContextSize: Int? = nil,
        maxTokens: Int? = nil,
        voiceBehavior: SpeechSynthesisVoiceBehavior = .auto
    ) {
        self.model = model
        self.text = text
        self.voice = voice
        self.language = language
        self.referenceAudioAssetName = referenceAudioAssetName
        self.referenceAudioPCMFloat32LE = referenceAudioPCMFloat32LE
        self.referenceAudioSamples = referenceAudioSamples
        self.referenceAudioSampleRate = referenceAudioSampleRate
        self.referenceTranscript = referenceTranscript
        self.instruct = instruct
        self.exaggeration = exaggeration
        self.cfgWeight = cfgWeight
        self.sampleRate = sampleRate
        self.responseFormat = responseFormat
        self.temperature = temperature
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
        self.repetitionContextSize = repetitionContextSize
        self.maxTokens = maxTokens
        self.voiceBehavior = voiceBehavior
    }
}

public struct SpeechRecognitionRequest: Codable, Sendable, Hashable {
    public let model: ModelIdentifier
    public let audioAssetName: String
    public let audioChunk: AudioChunk?
    public let languageHint: String?
    public let sampleRate: Double?

    public init(
        model: ModelIdentifier,
        audioAssetName: String,
        languageHint: String? = nil,
        sampleRate: Double? = nil
    ) {
        self.model = model
        self.audioAssetName = audioAssetName
        self.audioChunk = nil
        self.languageHint = languageHint
        self.sampleRate = sampleRate
    }

    public init(
        model: ModelIdentifier,
        audio: AudioChunk,
        languageHint: String? = nil
    ) {
        self.model = model
        self.audioAssetName = "__inline_audio_chunk__"
        self.audioChunk = audio
        self.languageHint = languageHint
        self.sampleRate = audio.sampleRate
    }
}

public struct TranslationRequest: Codable, Sendable, Hashable {
    public let sourceLanguage: String
    public let targetLanguage: String
    public let text: String

    public init(sourceLanguage: String, targetLanguage: String, text: String) {
        self.sourceLanguage = sourceLanguage
        self.targetLanguage = targetLanguage
        self.text = text
    }
}

public struct SpeechToSpeechRequest: Codable, Sendable, Hashable {
    public let sourceLanguage: String
    public let targetLanguage: String
    /// Optional text prompt for the target speech. Nil when doing pure voice conversion.
    public let targetText: String?
    public let voice: VoiceProfile?
    /// Source audio as PCM Float32 LE samples. Provide this for audio-in STS.
    public let sourceAudioPCMFloat32LE: Data?
    /// Sample rate of the source audio.
    public let sourceAudioSampleRate: Double?
    /// Optional transcript of the source audio (helps alignment-based STS).
    public let sourceTranscript: String?
    public let referenceAudioAssetName: String?
    public let sampleRate: Double

    public init(
        sourceLanguage: String,
        targetLanguage: String,
        targetText: String? = nil,
        voice: VoiceProfile? = nil,
        sourceAudioPCMFloat32LE: Data? = nil,
        sourceAudioSampleRate: Double? = nil,
        sourceTranscript: String? = nil,
        referenceAudioAssetName: String? = nil,
        sampleRate: Double = 24_000
    ) {
        self.sourceLanguage = sourceLanguage
        self.targetLanguage = targetLanguage
        self.targetText = targetText
        self.voice = voice
        self.sourceAudioPCMFloat32LE = sourceAudioPCMFloat32LE
        self.sourceAudioSampleRate = sourceAudioSampleRate
        self.sourceTranscript = sourceTranscript
        self.referenceAudioAssetName = referenceAudioAssetName
        self.sampleRate = sampleRate
    }
}

public struct ForcedAlignmentRequest: Codable, Sendable, Hashable {
    public let model: ModelIdentifier
    public let audioAssetName: String
    public let audioChunk: AudioChunk?
    public let transcript: String
    public let languageHint: String?
    public let sampleRate: Double?

    public init(
        model: ModelIdentifier,
        audioAssetName: String,
        transcript: String,
        languageHint: String? = nil,
        sampleRate: Double? = nil
    ) {
        self.model = model
        self.audioAssetName = audioAssetName
        self.audioChunk = nil
        self.transcript = transcript
        self.languageHint = languageHint
        self.sampleRate = sampleRate
    }

    public init(
        model: ModelIdentifier,
        audio: AudioChunk,
        transcript: String,
        languageHint: String? = nil
    ) {
        self.model = model
        self.audioAssetName = "__inline_audio_chunk__"
        self.audioChunk = audio
        self.transcript = transcript
        self.languageHint = languageHint
        self.sampleRate = audio.sampleRate
    }
}

public struct AlignmentToken: Codable, Sendable, Equatable, Hashable {
    public let text: String
    public let startTime: TimeInterval
    public let endTime: TimeInterval
    public let confidence: Double?

    public init(text: String, startTime: TimeInterval, endTime: TimeInterval, confidence: Double? = nil) {
        self.text = text
        self.startTime = startTime
        self.endTime = endTime
        self.confidence = confidence
    }
}

public struct ForcedAlignmentResult: Codable, Sendable, Equatable, Hashable {
    public let transcript: String
    public let tokens: [AlignmentToken]

    public init(transcript: String, tokens: [AlignmentToken]) {
        self.transcript = transcript
        self.tokens = tokens
    }

    public init(transcript: String, segments: [AlignmentToken]) {
        self.init(transcript: transcript, tokens: segments)
    }

    public var segments: [AlignmentToken] { tokens }
}

public typealias ForcedAlignmentResponse = ForcedAlignmentResult

/// A protocol for in-process audio sample buffers that carry typed Float samples,
/// eliminating the need to box PCM data as raw `Data` bytes.
public protocol AudioSampleBuffer: Sendable {
    var samples: [Float] { get }
    var sampleRate: Double { get }
}

public struct AudioChunk: AudioSampleBuffer, Codable, Sendable, Equatable, Hashable {
    public let samples: [Float]
    public let sampleRate: Double

    public init(samples: [Float], sampleRate: Double) {
        self.samples = samples
        self.sampleRate = sampleRate
    }
}

public struct TranscriptionResult: Codable, Sendable, Equatable, Hashable {
    public let text: String
    public let segments: [String]

    public init(text: String, segments: [String] = []) {
        self.text = text
        self.segments = segments
    }

    /// Convenience initializer that flattens a `RichTranscriptionResult` into the legacy flat format.
    public init(_ rich: RichTranscriptionResult) {
        self.init(text: rich.text, segments: rich.segments.map(\.text))
    }
}

public typealias SpeechToTextRequest = SpeechRecognitionRequest
public typealias SpeechToTextResponse = RichTranscriptionResult

// MARK: - Rich transcription types

/// A single timed segment (or word) within a transcription, optionally carrying confidence and finality metadata.
public struct TranscriptionSegment: Codable, Sendable, Equatable, Hashable {
    public let text: String
    public let startTime: Double?
    public let endTime: Double?
    public let confidence: Float?
    public let isFinal: Bool
    public let chunkIndex: Int?

    public init(
        text: String,
        startTime: Double? = nil,
        endTime: Double? = nil,
        confidence: Float? = nil,
        isFinal: Bool = true,
        chunkIndex: Int? = nil
    ) {
        self.text = text
        self.startTime = startTime
        self.endTime = endTime
        self.confidence = confidence
        self.isFinal = isFinal
        self.chunkIndex = chunkIndex
    }
}

/// Backend provenance metadata attached to inference results.
public struct BackendMetadata: Codable, Sendable, Equatable, Hashable {
    public let modelId: String
    public let backendKind: BackendKind
    public let inferenceTimeSeconds: Double?

    public init(
        modelId: String,
        backendKind: BackendKind,
        inferenceTimeSeconds: Double? = nil
    ) {
        self.modelId = modelId
        self.backendKind = backendKind
        self.inferenceTimeSeconds = inferenceTimeSeconds
    }
}

/// Streaming recognition performance counters emitted via `.metrics` events.
public struct RecognitionMetrics: Codable, Sendable, Equatable, Hashable {
    /// Seconds of audio that have been processed so far.
    public let audioDurationSeconds: Double?
    /// Wall-clock seconds spent in inference for those audio seconds.
    public let inferenceTimeSeconds: Double?
    /// Real-time factor: inferenceTimeSeconds / audioDurationSeconds (lower is faster).
    public let rtf: Double?
    /// Number of segments (chunks) processed so far.
    public let segmentsProcessed: Int

    public init(
        audioDurationSeconds: Double? = nil,
        inferenceTimeSeconds: Double? = nil,
        rtf: Double? = nil,
        segmentsProcessed: Int = 0
    ) {
        self.audioDurationSeconds = audioDurationSeconds
        self.inferenceTimeSeconds = inferenceTimeSeconds
        self.rtf = rtf
        self.segmentsProcessed = segmentsProcessed
    }
}

/// A rich transcription result that carries timing, word-level detail, and backend provenance.
public struct RichTranscriptionResult: Codable, Sendable, Equatable, Hashable {
    public let text: String
    public let language: String?
    public let durationSeconds: Double?
    public let segments: [TranscriptionSegment]
    public let words: [TranscriptionSegment]?
    public let alignmentReference: ForcedAlignmentResponse?
    public let backendMetadata: BackendMetadata

    public init(
        text: String,
        language: String? = nil,
        durationSeconds: Double? = nil,
        segments: [TranscriptionSegment] = [],
        words: [TranscriptionSegment]? = nil,
        alignmentReference: ForcedAlignmentResponse? = nil,
        backendMetadata: BackendMetadata
    ) {
        self.text = text
        self.language = language
        self.durationSeconds = durationSeconds
        self.segments = segments
        self.words = words
        self.alignmentReference = alignmentReference
        self.backendMetadata = backendMetadata
    }
}

/// An event emitted during streaming speech recognition.
public enum SpeechRecognitionEvent: Sendable {
    /// A partial (non-final) transcript segment, updated as audio arrives.
    case partial(TranscriptionSegment)
    /// A confirmed final segment for a completed audio chunk.
    case finalSegment(TranscriptionSegment)
    /// The full recognition result once the entire utterance has been processed.
    case completed(RichTranscriptionResult)
    /// Periodic performance metrics from the recognition engine.
    case metrics(RecognitionMetrics)
    /// A non-fatal advisory message from the backend (e.g. language fallback).
    case warning(String)
}

extension SpeechRecognitionEvent: Codable {
    private enum CodingKeys: String, CodingKey {
        case type
        case payload
    }

    private enum TypeKey: String, Codable {
        case partial
        case finalSegment
        case completed
        case metrics
        case warning
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let type = try container.decode(TypeKey.self, forKey: .type)
        switch type {
        case .partial:
            self = .partial(try container.decode(TranscriptionSegment.self, forKey: .payload))
        case .finalSegment:
            self = .finalSegment(try container.decode(TranscriptionSegment.self, forKey: .payload))
        case .completed:
            self = .completed(try container.decode(RichTranscriptionResult.self, forKey: .payload))
        case .metrics:
            self = .metrics(try container.decode(RecognitionMetrics.self, forKey: .payload))
        case .warning:
            self = .warning(try container.decode(String.self, forKey: .payload))
        }
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .partial(let segment):
            try container.encode(TypeKey.partial, forKey: .type)
            try container.encode(segment, forKey: .payload)
        case .finalSegment(let segment):
            try container.encode(TypeKey.finalSegment, forKey: .type)
            try container.encode(segment, forKey: .payload)
        case .completed(let result):
            try container.encode(TypeKey.completed, forKey: .type)
            try container.encode(result, forKey: .payload)
        case .metrics(let m):
            try container.encode(TypeKey.metrics, forKey: .type)
            try container.encode(m, forKey: .payload)
        case .warning(let message):
            try container.encode(TypeKey.warning, forKey: .type)
            try container.encode(message, forKey: .payload)
        }
    }
}

public struct SpeechToSpeechResult: Codable, Sendable, Equatable, Hashable {
    public let translation: String
    public let audio: AudioChunk
    public let transcription: TranscriptionResult?

    public init(
        translation: String,
        audio: AudioChunk,
        transcription: TranscriptionResult? = nil
    ) {
        self.translation = translation
        self.audio = audio
        self.transcription = transcription
    }
}

public protocol ValarModel: Sendable {
    var descriptor: ModelDescriptor { get }
    var backendKind: BackendKind { get }
}

public protocol TextToSpeechWorkflow: ValarModel {
    func synthesize(
        request: SpeechSynthesisRequest,
        in session: ModelRuntimeSession
    ) async throws -> AudioChunk
    func synthesizeStream(request: SpeechSynthesisRequest) async throws -> AsyncThrowingStream<AudioChunk, Error>
}

public protocol SpeechToTextWorkflow: ValarModel {
    func transcribe(
        request: SpeechRecognitionRequest,
        in session: ModelRuntimeSession
    ) async throws -> RichTranscriptionResult
}

public protocol TranslationWorkflow: ValarModel {
    func translate(
        request: TranslationRequest,
        in session: ModelRuntimeSession
    ) async throws -> String
}

public protocol SpeechToSpeechWorkflow: ValarModel {
    func synthesize(
        request: SpeechToSpeechRequest,
        in session: ModelRuntimeSession
    ) async throws -> SpeechToSpeechResult
}

public protocol ForcedAlignmentWorkflow: ValarModel {
    func align(
        request: ForcedAlignmentRequest,
        in session: ModelRuntimeSession
    ) async throws -> ForcedAlignmentResponse
}

public protocol TokenizerWorkflow: ValarModel {
    func tokenize(request: TokenizationRequest) async throws -> TokenizationResult
}

// MARK: - Streaming Speech Recognition

/// Controls how incoming audio is windowed into chunks for the recognition engine.
public struct ChunkPolicy: Codable, Sendable, Hashable {
    /// Target duration of each audio chunk sent to the model, in seconds. Defaults to 30 s.
    public let targetChunkDuration: Double
    /// Overlap between consecutive chunks in seconds, for continuity. Defaults to 1 s.
    public let overlapDuration: Double
    /// Minimum duration of speech that triggers processing, in seconds. Defaults to 0.25 s.
    public let minSpeechDuration: Double
    /// Amplitude threshold below which audio is considered silence (0–1 range). Defaults to 0.01.
    public let silenceThreshold: Double

    public init(
        targetChunkDuration: Double = 30.0,
        overlapDuration: Double = 1.0,
        minSpeechDuration: Double = 0.25,
        silenceThreshold: Double = 0.01
    ) {
        self.targetChunkDuration = targetChunkDuration
        self.overlapDuration = overlapDuration
        self.minSpeechDuration = minSpeechDuration
        self.silenceThreshold = silenceThreshold
    }
}

/// Selects the voice activity detection model used by `VADPolicy`.
public enum VADModelKind: String, Codable, Sendable, Hashable, CaseIterable {
    /// Silero VAD v5 CoreML model — preferred for accuracy on Apple Silicon.
    case sileroV5
    /// Simple energy-based VAD — always available without an external model.
    case energyBased
}

/// Controls voice activity detection behaviour during streaming recognition.
///
/// When `enabled` is `true`, only audio segments classified as speech are forwarded
/// to the recognition model. Backends that do not support VAD treat this as a no-op.
/// The `energyBased` model is always available; `sileroV5` requires the Silero CoreML
/// model to be present — if it is missing, the backend falls back to `energyBased`.
public struct VADPolicy: Codable, Sendable, Hashable {
    /// Whether VAD is active. Defaults to `false`.
    public let enabled: Bool
    /// VAD model to use. Defaults to `.energyBased` (no external model required).
    public let model: VADModelKind
    /// Probability threshold above which a frame is considered speech onset (0–1). Defaults to 0.5.
    public let onsetThreshold: Float
    /// Probability threshold below which a frame is considered speech offset (0–1). Defaults to 0.35.
    public let offsetThreshold: Float
    /// Minimum continuous speech duration in milliseconds before a segment is confirmed. Defaults to 250.
    public let minSpeechMs: Int
    /// Minimum silence duration in milliseconds before a speech segment is closed. Defaults to 300.
    public let minSilenceMs: Int

    public init(
        enabled: Bool = false,
        model: VADModelKind = .energyBased,
        onsetThreshold: Float = 0.5,
        offsetThreshold: Float = 0.35,
        minSpeechMs: Int = 250,
        minSilenceMs: Int = 300
    ) {
        self.enabled = enabled
        self.model = model
        self.onsetThreshold = onsetThreshold
        self.offsetThreshold = offsetThreshold
        self.minSpeechMs = minSpeechMs
        self.minSilenceMs = minSilenceMs
    }
}

/// Controls how decoded context tokens are carried across chunk boundaries.
///
/// When `enabled`, the backend passes up to `maxTokens` tokens of prior transcript
/// as a prefix when decoding the next chunk. This reduces hallucinations at boundaries
/// and improves consistency for long continuous streams.
public struct ContextCarryOverPolicy: Codable, Sendable, Hashable {
    /// Whether cross-chunk context carry-over is active. Defaults to `true`.
    public let enabled: Bool
    /// Maximum number of tokens to carry over. Nil means the backend uses its own default. Defaults to `nil`.
    public let maxTokens: Int?

    public init(
        enabled: Bool = true,
        maxTokens: Int? = nil
    ) {
        self.enabled = enabled
        self.maxTokens = maxTokens
    }
}

/// A request for live, streaming speech recognition where audio arrives incrementally.
///
/// Unlike `SpeechRecognitionRequest`, which operates on a complete audio file or inline chunk,
/// `StreamingSpeechRecognitionRequest` is designed for real-time scenarios — microphone input,
/// live broadcast feeds, or any source where audio arrives in a continuous stream.
public struct StreamingSpeechRecognitionRequest: Codable, Sendable, Hashable {
    /// The model to use for recognition.
    public let model: ModelIdentifier
    /// Unique session identifier, used to correlate events across a single streaming session.
    public let sessionId: UUID
    /// Optional BCP-47 language hint. Nil lets the model auto-detect.
    public let languageHint: String?
    /// Controls how audio is windowed into chunks.
    public let chunkPolicy: ChunkPolicy
    /// Controls voice activity detection.
    public let vadPolicy: VADPolicy
    /// Controls cross-chunk context carry-over.
    public let contextCarryOver: ContextCarryOverPolicy

    public init(
        model: ModelIdentifier,
        sessionId: UUID = UUID(),
        languageHint: String? = nil,
        chunkPolicy: ChunkPolicy = ChunkPolicy(),
        vadPolicy: VADPolicy = VADPolicy(),
        contextCarryOver: ContextCarryOverPolicy = ContextCarryOverPolicy()
    ) {
        self.model = model
        self.sessionId = sessionId
        self.languageHint = languageHint
        self.chunkPolicy = chunkPolicy
        self.vadPolicy = vadPolicy
        self.contextCarryOver = contextCarryOver
    }
}

/// A workflow protocol for models that support live, incremental speech recognition.
///
/// Conforming types receive a continuous stream of `AudioChunk` values and emit
/// `SpeechRecognitionEvent` values — including `.partial`, `.finalSegment`,
/// `.completed`, `.metrics`, and `.warning` events — via `AsyncThrowingStream`.
///
/// Compared to `SpeechToTextWorkflow` (which transcribes a complete audio asset),
/// `StreamingSpeechRecognitionWorkflow` is designed for real-time use cases such
/// as microphone input, live broadcast feeds, and voice command interfaces.
public protocol StreamingSpeechRecognitionWorkflow: ValarModel {
    /// Begin streaming recognition.
    ///
    /// - Parameters:
    ///   - request: Recognition parameters including model, session ID, chunking, VAD, and context policies.
    ///   - audioStream: An async stream of `AudioChunk` values representing incoming audio.
    /// - Returns: An `AsyncThrowingStream` of `SpeechRecognitionEvent` values. The stream
    ///   ends with a `.completed` event when `audioStream` is exhausted, or throws on a
    ///   non-recoverable error.
    func transcribeStream(
        request: StreamingSpeechRecognitionRequest,
        audioStream: AsyncStream<AudioChunk>
    ) async throws -> AsyncThrowingStream<SpeechRecognitionEvent, Error>
}
