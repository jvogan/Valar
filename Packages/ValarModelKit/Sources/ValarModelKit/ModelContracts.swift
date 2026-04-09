import Foundation

public enum ModelDomain: String, CaseIterable, Codable, Sendable {
    case tts
    case stt
    case sts
    case codec
    case utility
}

public struct CapabilityID: RawRepresentable, Hashable, Codable, Sendable, ExpressibleByStringLiteral, CustomStringConvertible {
    public let rawValue: String

    public init(rawValue: String) {
        self.rawValue = rawValue
    }

    public init(_ rawValue: String) {
        self.init(rawValue: rawValue)
    }

    public init(stringLiteral value: StringLiteralType) {
        self.init(value)
    }

    public var description: String { rawValue }

    public static let speechSynthesis: Self = "speech.synthesis"
    public static let speechRecognition: Self = "speech.recognition"
    public static let speechEnhancement: Self = "speech.enhancement"
    public static let translation: Self = "translation.text"
    public static let tokenization: Self = "text.tokenization"
    public static let voiceCloning: Self = "voice.cloning"
    public static let voiceDesign: Self = "voice.design"
    public static let multilingual: Self = "language.multilingual"
    public static let presetVoices: Self = "voice.preset_voices"
    public static let streaming: Self = "speech.streaming"
    public static let longFormRendering: Self = "render.long_form"
    public static let audioConditioning: Self = "audio.conditioning"
    public static let forcedAlignment: Self = "speech.forced_alignment"
    /// Experimental capability identifier for speech-to-speech model spikes, including LFM2.5 Audio evaluation.
    public static let speechToSpeech: Self = "speech.to_speech"
}

public typealias ModelCapability = CapabilityID

public struct ModelFamilyID: RawRepresentable, Hashable, Codable, Sendable, ExpressibleByStringLiteral, CustomStringConvertible {
    public let rawValue: String

    public init(rawValue: String) {
        self.rawValue = rawValue
    }

    public init(_ rawValue: String) {
        self.init(rawValue: rawValue)
    }

    public init(stringLiteral value: StringLiteralType) {
        self.init(value)
    }

    public var description: String { rawValue }

    public static let qwen3TTS: Self = "qwen3_tts"
    public static let qwen3ASR: Self = "qwen3_asr"
    public static let qwen3ForcedAligner: Self = "qwen3_forced_aligner"
    public static let soprano: Self = "soprano"
    public static let whisper: Self = "whisper"
    /// Experimental model family identifier for LFM2.5 Audio speech-to-speech spikes.
    public static let lfmAudio: Self = "lfm_audio"
    public static let unknown: Self = "unknown"
    public static let orpheus: Self = "orpheus_tts"
    public static let marvis: Self = "marvis_tts"
    public static let chatterbox: Self = "chatterbox_tts"
    public static let pocketTTS: Self = "pocket_tts"
    public static let voxtralTTS: Self = "voxtral_tts"
    public static let tadaTTS: Self = "tada_tts"
    public static let vibevoiceRealtimeTTS: Self = "vibevoice_realtime_tts"
}

public enum BackendKind: String, CaseIterable, Codable, Sendable, Hashable {
    case mlx
    case coreml
    case metal
    case cpu
    case mock
}

public enum ModelResidencyState: String, CaseIterable, Codable, Sendable {
    case unloaded
    case warming
    case resident
    case cooling
}

public enum ResidencyPolicy: String, CaseIterable, Codable, Sendable {
    case automatic
    case pinned
    case eager
    case onDemand
}

public enum ModelSupportTier: String, CaseIterable, Codable, Sendable {
    case supported
    case preview
    case experimental

    public var displayName: String {
        switch self {
        case .supported:
            return "Supported"
        case .preview:
            return "Preview"
        case .experimental:
            return "Experimental"
        }
    }
}

public enum ModelDistributionTier: String, CaseIterable, Codable, Sendable {
    case bundledFirstRun
    case optionalInstall
    case compatibilityPreview

    public var displayName: String {
        switch self {
        case .bundledFirstRun:
            return "Bundled First Run"
        case .optionalInstall:
            return "Optional Install"
        case .compatibilityPreview:
            return "Compatibility Preview"
        }
    }
}

public enum ModelLanguageSupportTier: String, CaseIterable, Codable, Sendable {
    case supported
    case preview
    case advisory
    case experimental

    public var displayName: String {
        switch self {
        case .supported:
            return "Supported"
        case .preview:
            return "Preview"
        case .advisory:
            return "Advisory"
        case .experimental:
            return "Experimental"
        }
    }
}

public struct ModelIdentifier: Hashable, Codable, Sendable, ExpressibleByStringLiteral, CustomStringConvertible {
    public let rawValue: String

    public init(_ rawValue: String) {
        self.rawValue = rawValue
    }

    public init(stringLiteral value: StringLiteralType) {
        self.init(value)
    }

    public var description: String { rawValue }

    public var canonicalValue: String {
        rawValue.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    @available(*, deprecated, message: "Use ModelDescriptor.familyID or ModelPackManifest.familyID instead.")
    public var family: String {
        inferredFamilyHint.rawValue
    }

    public var inferredFamilyHint: ModelFamilyID {
        let parts = canonicalValue.split(separator: "/")
        guard let tail = parts.last else { return .unknown }
        let normalized = tail.replacingOccurrences(of: "-", with: "_").lowercased()

        if normalized.contains("qwen3_tts") || normalized.contains("qwen3tts") {
            return .qwen3TTS
        }
        if normalized.contains("qwen3_forced_aligner") {
            return .qwen3ForcedAligner
        }
        if normalized.contains("qwen3_asr") {
            return .qwen3ASR
        }
        if normalized.contains("soprano") {
            return .soprano
        }
        if normalized.contains("whisper") {
            return .whisper
        }
        if normalized.contains("orpheus") { return .orpheus }
        if normalized.contains("marvis") { return .marvis }
        if normalized.contains("chatterbox") { return .chatterbox }
        if normalized.contains("pocket_tts") || normalized.contains("pockettts") { return .pocketTTS }
        if normalized.contains("voxtral_tts") || (normalized.contains("voxtral") && normalized.contains("tts")) {
            return .voxtralTTS
        }
        if normalized.contains("mlx_tada") || normalized.hasPrefix("tada_") || normalized.contains("_tada_") {
            return .tadaTTS
        }
        if normalized.contains("vibevoice") {
            return .vibevoiceRealtimeTTS
        }

        return .unknown
    }

    public var isCanonical: Bool {
        canonicalValue == rawValue
    }
}

public struct BackendRequirement: Hashable, Codable, Sendable {
    public let backendKind: BackendKind
    public let minimumMemoryBytes: Int?
    public let preferredQuantization: String?
    public let requiresLocalExecution: Bool
    public let minimumRuntimeVersion: String?

    public init(
        backendKind: BackendKind,
        minimumMemoryBytes: Int? = nil,
        preferredQuantization: String? = nil,
        requiresLocalExecution: Bool = true,
        minimumRuntimeVersion: String? = nil
    ) {
        self.backendKind = backendKind
        self.minimumMemoryBytes = minimumMemoryBytes
        self.preferredQuantization = preferredQuantization
        self.requiresLocalExecution = requiresLocalExecution
        self.minimumRuntimeVersion = minimumRuntimeVersion
    }
}

public enum ArtifactRole: String, CaseIterable, Codable, Sendable {
    case weights
    case config
    case tokenizer
    case vocabulary
    case promptTemplate
    case conditioning
    case voiceAsset
    case checksum
    case license
    case auxiliary
}

public struct ArtifactSpec: Hashable, Codable, Sendable, Identifiable {
    public let id: String
    public let role: ArtifactRole
    public let relativePath: String
    public let sha256: String?
    public let sizeBytes: Int?
    public let required: Bool

    public init(
        id: String,
        role: ArtifactRole,
        relativePath: String,
        sha256: String? = nil,
        sizeBytes: Int? = nil,
        required: Bool = true
    ) {
        self.id = id
        self.role = role
        self.relativePath = relativePath
        self.sha256 = sha256
        self.sizeBytes = sizeBytes
        self.required = required
    }
}

public struct TokenizerSpec: Hashable, Codable, Sendable {
    public let kind: String
    public let configPath: String?
    public let vocabularyPath: String?
    public let mergesPath: String?
    public let specialTokens: [String: String]

    public init(
        kind: String,
        configPath: String? = nil,
        vocabularyPath: String? = nil,
        mergesPath: String? = nil,
        specialTokens: [String: String] = [:]
    ) {
        self.kind = kind
        self.configPath = configPath
        self.vocabularyPath = vocabularyPath
        self.mergesPath = mergesPath
        self.specialTokens = specialTokens
    }
}

public struct PresetVoiceSpec: Hashable, Codable, Sendable {
    public let name: String
    public let displayName: String?
    public let languageAffinity: [String]
    public let aliases: [String]

    public init(
        name: String,
        displayName: String? = nil,
        languageAffinity: [String] = [],
        aliases: [String] = []
    ) {
        self.name = name
        self.displayName = displayName
        self.languageAffinity = languageAffinity
        self.aliases = aliases
    }
}

public struct LicenseSpec: Hashable, Codable, Sendable {
    public let name: String
    public let spdxIdentifier: String?
    public let relativePath: String?
    public let sourceURL: URL?
    public let requiresAttribution: Bool
    public let isNonCommercial: Bool

    private enum CodingKeys: String, CodingKey {
        case name
        case spdxIdentifier
        case relativePath
        case sourceURL
        case requiresAttribution
        case isNonCommercial
    }

    public init(
        name: String,
        spdxIdentifier: String? = nil,
        relativePath: String? = nil,
        sourceURL: URL? = nil,
        requiresAttribution: Bool = true,
        isNonCommercial: Bool = false
    ) {
        self.name = name
        self.spdxIdentifier = spdxIdentifier
        self.relativePath = relativePath
        self.sourceURL = sourceURL
        self.requiresAttribution = requiresAttribution
        self.isNonCommercial = isNonCommercial
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        name = try container.decode(String.self, forKey: .name)
        spdxIdentifier = try container.decodeIfPresent(String.self, forKey: .spdxIdentifier)
        relativePath = try container.decodeIfPresent(String.self, forKey: .relativePath)
        sourceURL = try container.decodeIfPresent(URL.self, forKey: .sourceURL)
        requiresAttribution = try container.decodeIfPresent(Bool.self, forKey: .requiresAttribution) ?? true
        isNonCommercial = try container.decodeIfPresent(Bool.self, forKey: .isNonCommercial) ?? false
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(name, forKey: .name)
        try container.encodeIfPresent(spdxIdentifier, forKey: .spdxIdentifier)
        try container.encodeIfPresent(relativePath, forKey: .relativePath)
        try container.encodeIfPresent(sourceURL, forKey: .sourceURL)
        try container.encode(requiresAttribution, forKey: .requiresAttribution)
        try container.encode(isNonCommercial, forKey: .isNonCommercial)
    }
}

public struct AudioConstraint: Hashable, Codable, Sendable {
    public let defaultSampleRate: Double?
    public let minimumSampleRate: Double?
    public let maximumSampleRate: Double?
    public let supportsReferenceAudio: Bool

    public init(
        defaultSampleRate: Double? = nil,
        minimumSampleRate: Double? = nil,
        maximumSampleRate: Double? = nil,
        supportsReferenceAudio: Bool = false
    ) {
        self.defaultSampleRate = defaultSampleRate
        self.minimumSampleRate = minimumSampleRate
        self.maximumSampleRate = maximumSampleRate
        self.supportsReferenceAudio = supportsReferenceAudio
    }
}

public enum ModelVoiceFeature: String, CaseIterable, Codable, Hashable, Sendable {
    case presetVoices
    case namedSpeakers
    case voiceDesign
    case referenceAudio
    case clonePrompt
    case stableNarrator

    public var displayName: String {
        switch self {
        case .presetVoices:
            return "Preset Voices"
        case .namedSpeakers:
            return "Named Speakers"
        case .voiceDesign:
            return "Voice Design"
        case .referenceAudio:
            return "Reference Audio"
        case .clonePrompt:
            return "Clone Prompt"
        case .stableNarrator:
            return "Stable Narrator"
        }
    }
}

public struct ModelVoiceSupport: Hashable, Codable, Sendable {
    public let features: [ModelVoiceFeature]

    public init(features: [ModelVoiceFeature]) {
        let canonicalOrder: [ModelVoiceFeature] = [
            .presetVoices,
            .namedSpeakers,
            .voiceDesign,
            .referenceAudio,
            .clonePrompt,
            .stableNarrator,
        ]
        let unique = Set(features)
        self.features = canonicalOrder.filter { unique.contains($0) }
    }

    public var supportsReferenceAudio: Bool {
        features.contains(.referenceAudio)
            || features.contains(.clonePrompt)
            || features.contains(.stableNarrator)
    }

    public static func resolve(for descriptor: ModelDescriptor) -> Self {
        guard descriptor.domain == .tts else {
            return ModelVoiceSupport(features: [])
        }

        switch descriptor.familyID {
        case .voxtralTTS:
            // Local Voxtral in Valar remains preset-only even though the hosted Mistral API
            // exposes saved voices and one-off reference audio.
            return ModelVoiceSupport(features: [.presetVoices])

        case .qwen3TTS:
            switch QwenCatalog.surface(for: descriptor.id) {
            case .qwen3TTSBase:
                return ModelVoiceSupport(features: [.referenceAudio, .clonePrompt, .stableNarrator])
            case .qwen3TTSCustomVoice:
                return ModelVoiceSupport(features: [.namedSpeakers])
            case .qwen3TTSVoiceDesign:
                return ModelVoiceSupport(features: [.voiceDesign])
            default:
                break
            }

        case .tadaTTS:
            return ModelVoiceSupport(features: [.referenceAudio])

        case .vibevoiceRealtimeTTS:
            return ModelVoiceSupport(features: [.presetVoices])

        default:
            break
        }

        var inferred: [ModelVoiceFeature] = []
        if descriptor.capabilities.contains(.presetVoices) {
            inferred.append(.presetVoices)
        }
        if descriptor.capabilities.contains(.voiceDesign) {
            inferred.append(.voiceDesign)
        }
        if descriptor.capabilities.contains(.voiceCloning) || descriptor.capabilities.contains(.audioConditioning) {
            inferred.append(.referenceAudio)
        }
        return ModelVoiceSupport(features: inferred)
    }
}

public struct PromptSchema: Hashable, Codable, Sendable {
    public let kind: String
    public let requiredFields: [String]
    public let optionalFields: [String]

    public init(kind: String, requiredFields: [String] = [], optionalFields: [String] = []) {
        self.kind = kind
        self.requiredFields = requiredFields
        self.optionalFields = optionalFields
    }
}

public struct ModelPackManifest: Hashable, Codable, Sendable, Identifiable {
    public let id: ModelIdentifier
    public let schemaVersion: Int
    public let familyID: ModelFamilyID
    public let displayName: String
    public let domain: ModelDomain
    public let capabilities: Set<CapabilityID>
    public let supportedBackends: [BackendRequirement]
    public let artifacts: [ArtifactSpec]
    public let tokenizer: TokenizerSpec?
    public let audio: AudioConstraint?
    public let promptSchema: PromptSchema?
    public let supportedLanguages: [String]?
    public let presetVoices: [PresetVoiceSpec]?
    public let supportTier: ModelSupportTier
    public let releaseEligible: Bool
    public let qualityTierByLanguage: [String: ModelLanguageSupportTier]
    public let licenses: [LicenseSpec]
    public let minimumAppVersion: String?
    public let parityProfileVersion: String?
    public let notes: String?

    public init(
        id: ModelIdentifier,
        schemaVersion: Int = 1,
        familyID: ModelFamilyID,
        displayName: String,
        domain: ModelDomain,
        capabilities: Set<CapabilityID>,
        supportedBackends: [BackendRequirement],
        artifacts: [ArtifactSpec],
        tokenizer: TokenizerSpec? = nil,
        audio: AudioConstraint? = nil,
        promptSchema: PromptSchema? = nil,
        supportedLanguages: [String]? = nil,
        presetVoices: [PresetVoiceSpec]? = nil,
        supportTier: ModelSupportTier = .supported,
        releaseEligible: Bool = true,
        qualityTierByLanguage: [String: ModelLanguageSupportTier] = [:],
        licenses: [LicenseSpec] = [],
        minimumAppVersion: String? = nil,
        parityProfileVersion: String? = nil,
        notes: String? = nil
    ) {
        self.id = id
        self.schemaVersion = schemaVersion
        self.familyID = familyID
        self.displayName = displayName
        self.domain = domain
        self.capabilities = capabilities
        self.supportedBackends = supportedBackends
        self.artifacts = artifacts
        self.tokenizer = tokenizer
        self.audio = audio
        self.promptSchema = promptSchema
        self.supportedLanguages = supportedLanguages
        self.presetVoices = presetVoices
        self.supportTier = supportTier
        self.releaseEligible = releaseEligible
        self.qualityTierByLanguage = qualityTierByLanguage
        self.licenses = licenses
        self.minimumAppVersion = minimumAppVersion
        self.parityProfileVersion = parityProfileVersion
        self.notes = notes
    }
}

public struct ModelDescriptor: Hashable, Codable, Sendable, Identifiable, CustomStringConvertible {
    public let id: ModelIdentifier
    public let familyID: ModelFamilyID
    public let displayName: String
    public let domain: ModelDomain
    public let capabilities: Set<CapabilityID>
    public let supportedBackends: [BackendRequirement]
    public let defaultSampleRate: Double?
    public let supportedLanguages: [String]?
    public let presetVoices: [PresetVoiceSpec]?
    public let supportTier: ModelSupportTier
    public let releaseEligible: Bool
    public let qualityTierByLanguage: [String: ModelLanguageSupportTier]
    public let notes: String?

    private enum CodingKeys: String, CodingKey {
        case id
        case familyID
        case displayName
        case domain
        case capabilities
        case supportedBackends
        case defaultSampleRate
        case supportedLanguages
        case presetVoices
        case supportTier
        case releaseEligible
        case qualityTierByLanguage
        case notes
    }

    public init(
        id: ModelIdentifier,
        familyID: ModelFamilyID? = nil,
        displayName: String,
        domain: ModelDomain,
        capabilities: Set<CapabilityID>,
        supportedBackends: [BackendRequirement] = [BackendRequirement(backendKind: .mlx)],
        defaultSampleRate: Double? = nil,
        supportedLanguages: [String]? = nil,
        presetVoices: [PresetVoiceSpec]? = nil,
        supportTier: ModelSupportTier = .supported,
        releaseEligible: Bool = true,
        qualityTierByLanguage: [String: ModelLanguageSupportTier] = [:],
        notes: String? = nil
    ) {
        self.id = id
        self.familyID = familyID ?? id.inferredFamilyHint
        self.displayName = displayName
        self.domain = domain
        self.capabilities = capabilities
        self.supportedBackends = supportedBackends
        self.defaultSampleRate = defaultSampleRate
        self.supportedLanguages = supportedLanguages
        self.presetVoices = presetVoices
        self.supportTier = supportTier
        self.releaseEligible = releaseEligible
        self.qualityTierByLanguage = qualityTierByLanguage
        self.notes = notes
    }

    public init(manifest: ModelPackManifest) {
        self.init(
            id: manifest.id,
            familyID: manifest.familyID,
            displayName: manifest.displayName,
            domain: manifest.domain,
            capabilities: manifest.capabilities,
            supportedBackends: manifest.supportedBackends,
            defaultSampleRate: manifest.audio?.defaultSampleRate,
            supportedLanguages: manifest.supportedLanguages,
            presetVoices: manifest.presetVoices,
            supportTier: manifest.supportTier,
            releaseEligible: manifest.releaseEligible,
            qualityTierByLanguage: manifest.qualityTierByLanguage,
            notes: manifest.notes
        )
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(ModelIdentifier.self, forKey: .id)
        familyID = try container.decode(ModelFamilyID.self, forKey: .familyID)
        displayName = try container.decode(String.self, forKey: .displayName)
        domain = try container.decode(ModelDomain.self, forKey: .domain)
        capabilities = try container.decode(Set<CapabilityID>.self, forKey: .capabilities)
        supportedBackends = try container.decodeIfPresent([BackendRequirement].self, forKey: .supportedBackends)
            ?? [BackendRequirement(backendKind: .mlx)]
        defaultSampleRate = try container.decodeIfPresent(Double.self, forKey: .defaultSampleRate)
        supportedLanguages = try container.decodeIfPresent([String].self, forKey: .supportedLanguages)
        presetVoices = try container.decodeIfPresent([PresetVoiceSpec].self, forKey: .presetVoices)
        supportTier = try container.decodeIfPresent(ModelSupportTier.self, forKey: .supportTier) ?? .supported
        releaseEligible = try container.decodeIfPresent(Bool.self, forKey: .releaseEligible) ?? true
        qualityTierByLanguage = try container.decodeIfPresent(
            [String: ModelLanguageSupportTier].self,
            forKey: .qualityTierByLanguage
        ) ?? [:]
        notes = try container.decodeIfPresent(String.self, forKey: .notes)
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(id, forKey: .id)
        try container.encode(familyID, forKey: .familyID)
        try container.encode(displayName, forKey: .displayName)
        try container.encode(domain, forKey: .domain)
        try container.encode(capabilities, forKey: .capabilities)
        try container.encode(supportedBackends, forKey: .supportedBackends)
        try container.encodeIfPresent(defaultSampleRate, forKey: .defaultSampleRate)
        try container.encodeIfPresent(supportedLanguages, forKey: .supportedLanguages)
        try container.encodeIfPresent(presetVoices, forKey: .presetVoices)
        try container.encode(supportTier, forKey: .supportTier)
        try container.encode(releaseEligible, forKey: .releaseEligible)
        try container.encode(qualityTierByLanguage, forKey: .qualityTierByLanguage)
        try container.encodeIfPresent(notes, forKey: .notes)
    }

    public var description: String {
        "\(displayName) [\(id.rawValue)]"
    }

    public var voiceSupport: ModelVoiceSupport {
        ModelVoiceSupport.resolve(for: self)
    }
}

public struct ModelRuntimeConfiguration: Hashable, Codable, Sendable {
    public let backendKind: BackendKind
    public let residencyPolicy: ResidencyPolicy
    public let preferredSampleRate: Double?
    public let memoryBudgetBytes: Int?
    public let allowQuantizedWeights: Bool
    public let allowWarmStart: Bool

    public init(
        backendKind: BackendKind,
        residencyPolicy: ResidencyPolicy = .automatic,
        preferredSampleRate: Double? = nil,
        memoryBudgetBytes: Int? = nil,
        allowQuantizedWeights: Bool = true,
        allowWarmStart: Bool = true
    ) {
        self.backendKind = backendKind
        self.residencyPolicy = residencyPolicy
        self.preferredSampleRate = preferredSampleRate
        self.memoryBudgetBytes = memoryBudgetBytes
        self.allowQuantizedWeights = allowQuantizedWeights
        self.allowWarmStart = allowWarmStart
    }
}

public struct ModelRuntimeSession: Hashable, Codable, Sendable, Identifiable {
    public let id: UUID
    public let descriptor: ModelDescriptor
    public let backendKind: BackendKind
    public let configuration: ModelRuntimeConfiguration
    public var state: ModelResidencyState
    public let startedAt: Date
    public var lastAccessedAt: Date

    public init(
        id: UUID = UUID(),
        descriptor: ModelDescriptor,
        backendKind: BackendKind,
        configuration: ModelRuntimeConfiguration,
        state: ModelResidencyState = .warming,
        startedAt: Date = .now,
        lastAccessedAt: Date = .now
    ) {
        self.id = id
        self.descriptor = descriptor
        self.backendKind = backendKind
        self.configuration = configuration
        self.state = state
        self.startedAt = startedAt
        self.lastAccessedAt = lastAccessedAt
    }

    public mutating func markResident() {
        state = .resident
        lastAccessedAt = .now
    }

    public mutating func markCooling() {
        state = .cooling
        lastAccessedAt = .now
    }
}

public struct ModelRegistryEntry: Hashable, Codable, Sendable, Identifiable {
    public let id: ModelIdentifier
    public let session: ModelRuntimeSession
    public let loadCount: Int
    public let lastUsedAt: Date

    public init(
        session: ModelRuntimeSession,
        loadCount: Int = 1,
        lastUsedAt: Date = .now
    ) {
        self.id = session.descriptor.id
        self.session = session
        self.loadCount = loadCount
        self.lastUsedAt = lastUsedAt
    }
}

public struct BackendFeatureID: RawRepresentable, Hashable, Codable, Sendable, ExpressibleByStringLiteral {
    public let rawValue: String

    public init(rawValue: String) {
        self.rawValue = rawValue
    }

    public init(_ rawValue: String) {
        self.init(rawValue: rawValue)
    }

    public init(stringLiteral value: StringLiteralType) {
        self.init(value)
    }

    public static let warmStart: Self = "backend.warm_start"
    public static let quantizedWeights: Self = "backend.quantized_weights"
    public static let streamingSynthesis: Self = "backend.streaming_synthesis"
    public static let streamingRecognition: Self = "backend.streaming_recognition"
    public static let forcedAlignment: Self = "backend.forced_alignment"
}

public struct BackendCapabilities: Hashable, Codable, Sendable {
    public let features: Set<BackendFeatureID>
    public let supportedFamilies: Set<ModelFamilyID>
    public let maximumConcurrentSessions: Int?

    public init(
        features: Set<BackendFeatureID> = [],
        supportedFamilies: Set<ModelFamilyID> = [],
        maximumConcurrentSessions: Int? = nil
    ) {
        self.features = features
        self.supportedFamilies = supportedFamilies
        self.maximumConcurrentSessions = maximumConcurrentSessions
    }
}

public protocol ModelAdapter: Sendable {
    var familyID: ModelFamilyID { get }
    var supportedCapabilities: Set<CapabilityID> { get }
    var supportedBackends: [BackendKind] { get }
    func validate(manifest: ModelPackManifest) throws
    func makeDescriptor(from manifest: ModelPackManifest) throws -> ModelDescriptor
}

public protocol InferenceBackend: Sendable {
    var backendKind: BackendKind { get }
    var runtimeCapabilities: BackendCapabilities { get }
    func validate(requirement: BackendRequirement) async throws
    /// Begins loading a model in the background without blocking the caller.
    ///
    /// If the model is already loaded or a load is already in flight, this is a no-op.
    /// A subsequent call to `loadModel` for the same descriptor will await the in-flight
    /// load task rather than starting a duplicate, so the model arrives with no redundant work.
    func prewarm(descriptor: ModelDescriptor, configuration: ModelRuntimeConfiguration) async
    func loadModel(descriptor: ModelDescriptor, configuration: ModelRuntimeConfiguration) async throws -> any ValarModel
    func unloadModel(_ model: any ValarModel) async throws
}

extension InferenceBackend {
    /// Default no-op — backends that do not support prewarming ignore this call.
    public func prewarm(descriptor: ModelDescriptor, configuration: ModelRuntimeConfiguration) async {}
}
