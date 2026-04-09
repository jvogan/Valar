import Foundation

public enum QwenSurface: String, CaseIterable, Codable, Sendable {
    case qwen3TTSBase
    case qwen3TTSCustomVoice
    case qwen3TTSVoiceDesign
    case qwen3ASR
    case qwen3ForcedAligner

    public var familyID: ModelFamilyID {
        switch self {
        case .qwen3TTSBase, .qwen3TTSCustomVoice, .qwen3TTSVoiceDesign:
            return .qwen3TTS
        case .qwen3ASR:
            return .qwen3ASR
        case .qwen3ForcedAligner:
            return .qwen3ForcedAligner
        }
    }

    public var domain: ModelDomain {
        switch self {
        case .qwen3TTSBase, .qwen3TTSCustomVoice, .qwen3TTSVoiceDesign:
            return .tts
        case .qwen3ASR, .qwen3ForcedAligner:
            return .stt
        }
    }

    public var capabilities: Set<CapabilityID> {
        switch self {
        case .qwen3TTSBase:
            return [.speechSynthesis, .tokenization, .voiceCloning, .audioConditioning, .longFormRendering]
        case .qwen3TTSCustomVoice:
            return [.speechSynthesis, .tokenization, .voiceCloning, .audioConditioning]
        case .qwen3TTSVoiceDesign:
            return [.speechSynthesis, .tokenization, .voiceDesign, .audioConditioning]
        case .qwen3ASR:
            return [.speechRecognition, .tokenization, .translation]
        case .qwen3ForcedAligner:
            return [.speechRecognition, .forcedAlignment, .tokenization]
        }
    }

    public var defaultSampleRate: Double {
        switch self {
        case .qwen3TTSBase, .qwen3TTSCustomVoice, .qwen3TTSVoiceDesign:
            return 24_000
        case .qwen3ASR, .qwen3ForcedAligner:
            return 16_000
        }
    }

    public var promptSchema: PromptSchema? {
        switch self {
        case .qwen3TTSBase:
            return PromptSchema(
                kind: "qwen3.tts.base",
                requiredFields: ["text"],
                optionalFields: ["referenceAudio", "referenceText"]
            )
        case .qwen3TTSCustomVoice:
            return PromptSchema(
                kind: "qwen3.tts.custom_voice",
                requiredFields: ["text"],
                optionalFields: ["referenceAudio"]
            )
        case .qwen3TTSVoiceDesign:
            return PromptSchema(
                kind: "qwen3.tts.voice_design",
                requiredFields: ["text"],
                optionalFields: ["voiceBrief"]
            )
        case .qwen3ASR:
            return PromptSchema(kind: "qwen3.asr", requiredFields: ["audio"])
        case .qwen3ForcedAligner:
            return PromptSchema(kind: "qwen3.forced_aligner", requiredFields: ["audio", "transcript"])
        }
    }
}

public struct SupportedModelCatalogEntry: Codable, Sendable, Equatable, Identifiable {
    public let id: ModelIdentifier
    public let manifest: ModelPackManifest
    public let remoteURL: URL?
    public let requiresManualDownload: Bool
    public let isRecommended: Bool
    public let distributionTier: ModelDistributionTier
    public let tags: [String]

    public init(
        manifest: ModelPackManifest,
        remoteURL: URL? = nil,
        requiresManualDownload: Bool = false,
        isRecommended: Bool = false,
        distributionTier: ModelDistributionTier = .optionalInstall,
        tags: [String] = []
    ) {
        self.id = manifest.id
        self.manifest = manifest
        self.remoteURL = remoteURL
        self.requiresManualDownload = requiresManualDownload
        self.isRecommended = isRecommended
        self.distributionTier = distributionTier
        self.tags = tags
    }
}

public enum QwenCatalog {
    public static func surface(for modelID: ModelIdentifier) -> QwenSurface? {
        let normalized = modelID.rawValue
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
            .replacingOccurrences(of: "_", with: "-")

        if normalized.contains("forcedaligner") {
            return .qwen3ForcedAligner
        }
        if normalized.contains("qwen3-asr") {
            return .qwen3ASR
        }
        if normalized.contains("voicedesign") {
            return .qwen3TTSVoiceDesign
        }
        if normalized.contains("customvoice") {
            return .qwen3TTSCustomVoice
        }
        if normalized.contains("qwen3-tts") && normalized.contains("base") {
            return .qwen3TTSBase
        }

        return nil
    }

    public static func acceptsNamedSpeaker(_ modelID: ModelIdentifier) -> Bool {
        surface(for: modelID) == .qwen3TTSCustomVoice
    }

    public static func makeManifest(
        surface: QwenSurface,
        modelID: ModelIdentifier,
        displayName: String,
        artifacts: [ArtifactSpec],
        isRecommended: Bool = true,
        remoteURL: URL? = nil
    ) -> SupportedModelCatalogEntry {
        let manifest = ModelPackManifest(
            id: modelID,
            familyID: surface.familyID,
            displayName: displayName,
            domain: surface.domain,
            capabilities: surface.capabilities,
            supportedBackends: [
                BackendRequirement(
                    backendKind: .mlx,
                    preferredQuantization: {
                        let lower = modelID.rawValue.lowercased()
                        return lower.contains("4bit") ? "4bit" : lower.contains("8bit") ? "8bit" : "bf16"
                    }(),
                    requiresLocalExecution: true
                ),
            ],
            artifacts: artifacts,
            tokenizer: TokenizerSpec(kind: "huggingface", configPath: "tokenizer.json"),
            audio: AudioConstraint(
                defaultSampleRate: surface.defaultSampleRate,
                minimumSampleRate: surface.defaultSampleRate,
                maximumSampleRate: surface.defaultSampleRate,
                supportsReferenceAudio: surface == .qwen3TTSBase
                    || surface == .qwen3TTSCustomVoice
                    || surface == .qwen3TTSVoiceDesign
            ),
            promptSchema: surface.promptSchema,
            licenses: [
                LicenseSpec(name: "Model license", sourceURL: remoteURL, requiresAttribution: true),
            ],
            notes: nil
        )

        return SupportedModelCatalogEntry(
            manifest: manifest,
            remoteURL: remoteURL,
            requiresManualDownload: remoteURL != nil,
            isRecommended: isRecommended,
            tags: ["qwen", surface.familyID.rawValue]
        )
    }

    public static var supportedEntries: [SupportedModelCatalogEntry] {
        [
            makeManifest(
                surface: .qwen3TTSBase,
                modelID: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
                displayName: "Qwen3-TTS 1.7B Base",
                artifacts: ttsArtifacts(
                    weightsSHA256: "81fb76175ff74e69be25fef2cc3e54f016df3034f1514c8e1c89da06a3510cff",
                    weightsBytes: 3_857_414_009
                ),
                remoteURL: URL(string: "https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16")
            ),
            makeManifest(
                surface: .qwen3TTSCustomVoice,
                modelID: "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16",
                displayName: "Qwen3-TTS 1.7B CustomVoice",
                artifacts: ttsArtifacts(
                    weightsSHA256: "3a791fb8250fc32ab0259b679d834159d3c8516af62f033ff2b9f42913e3fab6",
                    weightsBytes: 3_833_402_589
                ),
                remoteURL: URL(string: "https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16")
            ),
            makeManifest(
                surface: .qwen3TTSVoiceDesign,
                modelID: "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16",
                displayName: "Qwen3-TTS 1.7B VoiceDesign",
                artifacts: ttsArtifacts(
                    weightsSHA256: "96ae28bec2205ec0b5e0c750bea2b8a5deac4f14d33a8a25a5f753299486b70e",
                    weightsBytes: 3_833_402_589
                ),
                remoteURL: URL(string: "https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16")
            ),
            makeManifest(
                surface: .qwen3ASR,
                modelID: "mlx-community/Qwen3-ASR-0.6B-8bit",
                displayName: "Qwen3-ASR 0.6B 8bit",
                artifacts: asrArtifacts(
                    weightsSHA256: "b5bfe4abc1b4c6e58b633096682ec2b6297298add1527119936107d211adf0e8",
                    weightsBytes: 1_006_229_426
                ),
                remoteURL: URL(string: "https://huggingface.co/mlx-community/Qwen3-ASR-0.6B-8bit")
            ),
            makeManifest(
                surface: .qwen3ForcedAligner,
                modelID: "mlx-community/Qwen3-ForcedAligner-0.6B-8bit",
                displayName: "Qwen3-ForcedAligner 0.6B 8bit",
                artifacts: asrArtifacts(
                    weightsSHA256: "be19ef8ac4326d032e7673342930b14c2df30bd68c1632493b0f563e30829f91",
                    weightsBytes: 1_271_924_386
                ),
                remoteURL: URL(string: "https://huggingface.co/mlx-community/Qwen3-ForcedAligner-0.6B-8bit")
            ),
            makeManifest(
                surface: .qwen3TTSBase,
                modelID: "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
                displayName: "Qwen3-TTS 0.6B Base",
                artifacts: ttsArtifacts(weightsSHA256: nil, weightsBytes: nil),
                isRecommended: false,
                remoteURL: URL(string: "https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16")
            ),
            makeManifest(
                surface: .qwen3TTSBase,
                modelID: "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit",
                displayName: "Qwen3-TTS 0.6B Base (8-bit)",
                artifacts: ttsArtifacts(weightsSHA256: nil, weightsBytes: nil),
                isRecommended: false,
                remoteURL: URL(string: "https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit")
            ),
            makeManifest(
                surface: .qwen3TTSBase,
                modelID: "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit",
                displayName: "Qwen3-TTS 0.6B Base (4-bit)",
                artifacts: ttsArtifacts(weightsSHA256: nil, weightsBytes: nil),
                isRecommended: false,
                remoteURL: URL(string: "https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit")
            ),
            makeManifest(
                surface: .qwen3TTSCustomVoice,
                modelID: "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16",
                displayName: "Qwen3-TTS 0.6B CustomVoice",
                artifacts: ttsArtifacts(weightsSHA256: nil, weightsBytes: nil),
                isRecommended: false,
                remoteURL: URL(string: "https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16")
            ),
            makeManifest(
                surface: .qwen3TTSCustomVoice,
                modelID: "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit",
                displayName: "Qwen3-TTS 0.6B CustomVoice (8-bit)",
                artifacts: ttsArtifacts(weightsSHA256: nil, weightsBytes: nil),
                isRecommended: false,
                remoteURL: URL(string: "https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit")
            ),
            makeManifest(
                surface: .qwen3TTSCustomVoice,
                modelID: "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit",
                displayName: "Qwen3-TTS 0.6B CustomVoice (4-bit)",
                artifacts: ttsArtifacts(weightsSHA256: nil, weightsBytes: nil),
                isRecommended: false,
                remoteURL: URL(string: "https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit")
            ),
            makeManifest(
                surface: .qwen3TTSBase,
                modelID: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit",
                displayName: "Qwen3-TTS 1.7B Base (8-bit)",
                artifacts: ttsArtifacts(weightsSHA256: nil, weightsBytes: nil),
                isRecommended: false,
                remoteURL: URL(string: "https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit")
            ),
            makeManifest(
                surface: .qwen3TTSBase,
                modelID: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-4bit",
                displayName: "Qwen3-TTS 1.7B Base (4-bit)",
                artifacts: ttsArtifacts(weightsSHA256: nil, weightsBytes: nil),
                isRecommended: false,
                remoteURL: URL(string: "https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-Base-4bit")
            ),
            makeManifest(
                surface: .qwen3TTSVoiceDesign,
                modelID: "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-4bit",
                displayName: "Qwen3-TTS 1.7B VoiceDesign (4-bit)",
                artifacts: ttsArtifacts(weightsSHA256: nil, weightsBytes: nil),
                isRecommended: false,
                remoteURL: URL(string: "https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-4bit")
            ),
        ]
    }

    private static func ttsArtifacts(weightsSHA256: String?, weightsBytes: Int?) -> [ArtifactSpec] {
        [
            ArtifactSpec(id: "model-config", role: .config, relativePath: "config.json"),
            ArtifactSpec(id: "generation-config", role: .config, relativePath: "generation_config.json"),
            ArtifactSpec(id: "preprocessor-config", role: .config, relativePath: "preprocessor_config.json"),
            ArtifactSpec(id: "model-weights", role: .weights, relativePath: "model.safetensors", sha256: weightsSHA256, sizeBytes: weightsBytes),
            ArtifactSpec(id: "model-index", role: .weights, relativePath: "model.safetensors.index.json"),
            ArtifactSpec(id: "tokenizer", role: .tokenizer, relativePath: "tokenizer.json", required: false),
            ArtifactSpec(id: "tokenizer-config", role: .tokenizer, relativePath: "tokenizer_config.json"),
            ArtifactSpec(id: "vocab", role: .tokenizer, relativePath: "vocab.json"),
            ArtifactSpec(id: "merges", role: .tokenizer, relativePath: "merges.txt"),
            ArtifactSpec(id: "speech-tokenizer-config", role: .config, relativePath: "speech_tokenizer/config.json"),
            ArtifactSpec(id: "speech-tokenizer-configuration", role: .config, relativePath: "speech_tokenizer/configuration.json"),
            ArtifactSpec(id: "speech-tokenizer-preprocessor", role: .config, relativePath: "speech_tokenizer/preprocessor_config.json"),
            ArtifactSpec(id: "speech-tokenizer-weights", role: .weights, relativePath: "speech_tokenizer/model.safetensors"),
        ]
    }

    private static func asrArtifacts(weightsSHA256: String, weightsBytes: Int) -> [ArtifactSpec] {
        [
            ArtifactSpec(id: "model-config", role: .config, relativePath: "config.json"),
            ArtifactSpec(id: "generation-config", role: .config, relativePath: "generation_config.json"),
            ArtifactSpec(id: "preprocessor-config", role: .config, relativePath: "preprocessor_config.json"),
            ArtifactSpec(id: "model-weights", role: .weights, relativePath: "model.safetensors", sha256: weightsSHA256, sizeBytes: weightsBytes),
            ArtifactSpec(id: "model-index", role: .weights, relativePath: "model.safetensors.index.json"),
            ArtifactSpec(id: "tokenizer", role: .tokenizer, relativePath: "tokenizer.json"),
            ArtifactSpec(id: "tokenizer-config", role: .tokenizer, relativePath: "tokenizer_config.json"),
            ArtifactSpec(id: "vocab", role: .tokenizer, relativePath: "vocab.json"),
            ArtifactSpec(id: "merges", role: .tokenizer, relativePath: "merges.txt"),
        ]
    }
}
