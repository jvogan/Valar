import Foundation

public enum VoxtralSurface: String, CaseIterable, Codable, Sendable {
    case voxtral4BTTS2603
    case voxtral4BTTS2603MLX4Bit
    case voxtral4BTTS2603MLX6Bit

    public var familyID: ModelFamilyID { .voxtralTTS }
    public var domain: ModelDomain { .tts }

    public var capabilities: Set<CapabilityID> {
        [.speechSynthesis, .multilingual, .presetVoices, .streaming]
    }

    public var defaultSampleRate: Double { 24_000 }

    public var promptSchema: PromptSchema? {
        PromptSchema(
            kind: "voxtral.tts.preset_voice",
            requiredFields: ["text"],
            optionalFields: ["voice", "language"]
        )
    }
}

public enum VoxtralCatalog {
    public static let modelIdentifier: ModelIdentifier = "mistralai/Voxtral-4B-TTS-2603"
    public static let mlx4BitModelIdentifier: ModelIdentifier = "mlx-community/Voxtral-4B-TTS-2603-mlx-4bit"
    public static let mlx6BitModelIdentifier: ModelIdentifier = "mlx-community/Voxtral-4B-TTS-2603-mlx-6bit"
    public static let licenseName = "CC BY-NC 4.0"
    public static let supportedLanguages = ["EN", "FR", "DE", "ES", "NL", "PT", "IT", "HI", "AR"]

    static let rawVoiceEmbeddingArtifactIDPrefix = "voice-embedding-raw-"

    public static let presetVoices: [PresetVoiceSpec] = [
        PresetVoiceSpec(name: "casual_female", displayName: "Casual Female", languageAffinity: supportedLanguages, aliases: ["chloe", "relaxed_f"]),
        PresetVoiceSpec(name: "casual_male", displayName: "Casual Male", languageAffinity: supportedLanguages, aliases: ["jake", "relaxed_m"]),
        PresetVoiceSpec(name: "cheerful_female", displayName: "Cheerful Female", languageAffinity: supportedLanguages, aliases: ["lily", "happy_f"]),
        PresetVoiceSpec(name: "neutral_female", displayName: "Neutral Female", languageAffinity: supportedLanguages, aliases: ["emma", "default_f"]),
        PresetVoiceSpec(name: "neutral_male", displayName: "Neutral Male", languageAffinity: supportedLanguages, aliases: ["alex", "default_m"]),
        PresetVoiceSpec(name: "pt_male", displayName: "Portuguese Male", languageAffinity: ["PT"], aliases: ["pedro"]),
        PresetVoiceSpec(name: "pt_female", displayName: "Portuguese Female", languageAffinity: ["PT"], aliases: ["sofia"]),
        PresetVoiceSpec(name: "nl_male", displayName: "Dutch Male", languageAffinity: ["NL"], aliases: ["jan"]),
        PresetVoiceSpec(name: "nl_female", displayName: "Dutch Female", languageAffinity: ["NL"], aliases: ["anna"]),
        PresetVoiceSpec(name: "it_male", displayName: "Italian Male", languageAffinity: ["IT"], aliases: ["marco"]),
        PresetVoiceSpec(name: "it_female", displayName: "Italian Female", languageAffinity: ["IT"], aliases: ["giulia"]),
        PresetVoiceSpec(name: "fr_male", displayName: "French Male", languageAffinity: ["FR"], aliases: ["pierre"]),
        PresetVoiceSpec(name: "fr_female", displayName: "French Female", languageAffinity: ["FR"], aliases: ["claire"]),
        PresetVoiceSpec(name: "es_male", displayName: "Spanish Male", languageAffinity: ["ES"], aliases: ["carlos"]),
        PresetVoiceSpec(name: "es_female", displayName: "Spanish Female", languageAffinity: ["ES"], aliases: ["lucia"]),
        PresetVoiceSpec(name: "de_male", displayName: "German Male", languageAffinity: ["DE"], aliases: ["hans"]),
        PresetVoiceSpec(name: "de_female", displayName: "German Female", languageAffinity: ["DE"], aliases: ["lena"]),
        PresetVoiceSpec(name: "ar_male", displayName: "Arabic Male", languageAffinity: ["AR"], aliases: ["omar"]),
        PresetVoiceSpec(name: "hi_male", displayName: "Hindi Male", languageAffinity: ["HI"], aliases: ["arjun"]),
        PresetVoiceSpec(name: "hi_female", displayName: "Hindi Female", languageAffinity: ["HI"], aliases: ["priya"]),
    ]

    /// Presets excluded from random selection due to known quality issues.
    /// neutral_male: upstream startup garble artifact.
    /// ar_male: quality issues with Arabic synthesis.
    private static let randomExcluded: Set<String> = ["neutral_male", "ar_male"]

    /// Resolve a voice name or alias to the canonical preset name.
    /// Returns nil if not found. Accepts "random" to pick a random preset.
    public static func resolvePresetName(_ input: String) -> String? {
        let lowered = input.lowercased()
        if lowered == "random" {
            let eligible = presetVoices.filter { !randomExcluded.contains($0.name) }
            return eligible.randomElement()?.name
        }
        if let exact = presetVoices.first(where: { $0.name == lowered }) {
            return exact.name
        }
        if let aliased = presetVoices.first(where: { $0.aliases.contains(lowered) }) {
            return aliased.name
        }
        return nil
    }

    static var rawVoiceEmbeddingArtifacts: [ArtifactSpec] {
        presetVoices.map { preset in
            ArtifactSpec(
                id: "\(rawVoiceEmbeddingArtifactIDPrefix)\(preset.name)",
                role: .voiceAsset,
                relativePath: "voice_embedding/\(preset.name).pt"
            )
        }
    }

    public static func makeManifest(
        surface: VoxtralSurface,
        modelID: ModelIdentifier,
        displayName: String,
        artifacts: [ArtifactSpec],
        preferredQuantization: String,
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
                    preferredQuantization: preferredQuantization,
                    requiresLocalExecution: true
                ),
            ],
            artifacts: artifacts,
            tokenizer: TokenizerSpec(kind: "tekken", configPath: "tekken.json"),
            audio: AudioConstraint(
                defaultSampleRate: surface.defaultSampleRate,
                minimumSampleRate: surface.defaultSampleRate,
                maximumSampleRate: surface.defaultSampleRate,
                supportsReferenceAudio: false
            ),
            promptSchema: surface.promptSchema,
            supportedLanguages: supportedLanguages,
            presetVoices: presetVoices,
            supportTier: .preview,
            releaseEligible: false,
            licenses: [
                LicenseSpec(
                    name: "CC BY-NC 4.0",
                    spdxIdentifier: "CC-BY-NC-4.0",
                    relativePath: "LICENSE",
                    sourceURL: remoteURL,
                    requiresAttribution: true,
                    isNonCommercial: true
                ),
            ],
            notes: "CC BY-NC 4.0 license. Non-commercial use only. Attribution required. Includes 20 preset voices."
        )

        return SupportedModelCatalogEntry(
            manifest: manifest,
            remoteURL: remoteURL,
            requiresManualDownload: remoteURL != nil,
            tags: ["voxtral", "preset-voices", "multilingual", surface.familyID.rawValue]
        )
    }

    public static var supportedEntries: [SupportedModelCatalogEntry] {
        [
            makeManifest(
                surface: .voxtral4BTTS2603,
                modelID: modelIdentifier,
                displayName: "Voxtral 4B TTS 2603",
                artifacts: [
                    ArtifactSpec(id: "model-config", role: .config, relativePath: "params.json"),
                    ArtifactSpec(id: "model-weights", role: .weights, relativePath: "consolidated.safetensors"),
                    ArtifactSpec(id: "tokenizer", role: .tokenizer, relativePath: "tekken.json"),
                ] + rawVoiceEmbeddingArtifacts + [
                    ArtifactSpec(id: "voice-embeddings-safe", role: .voiceAsset, relativePath: "voice_embedding_safe/"),
                ],
                preferredQuantization: "bf16",
                remoteURL: URL(string: "https://huggingface.co/mistralai/Voxtral-4B-TTS-2603")
            ),
            makeManifest(
                surface: .voxtral4BTTS2603MLX4Bit,
                modelID: mlx4BitModelIdentifier,
                displayName: "Voxtral 4B TTS 2603 MLX (4-bit)",
                artifacts: [
                    ArtifactSpec(id: "model-config", role: .config, relativePath: "params.json"),
                    ArtifactSpec(id: "model-weights", role: .weights, relativePath: "model.safetensors"),
                    ArtifactSpec(id: "model-index", role: .weights, relativePath: "model.safetensors.index.json"),
                    ArtifactSpec(id: "tokenizer", role: .tokenizer, relativePath: "tekken.json"),
                ] + presetVoices.map { preset in
                    ArtifactSpec(
                        id: "voice-embedding-safetensors-\(preset.name)",
                        role: .voiceAsset,
                        relativePath: "voice_embedding/\(preset.name).safetensors"
                    )
                },
                preferredQuantization: "4bit",
                remoteURL: URL(string: "https://huggingface.co/mlx-community/Voxtral-4B-TTS-2603-mlx-4bit")
            ),
            makeManifest(
                surface: .voxtral4BTTS2603MLX6Bit,
                modelID: mlx6BitModelIdentifier,
                displayName: "Voxtral 4B TTS 2603 MLX (6-bit)",
                artifacts: [
                    ArtifactSpec(id: "model-config", role: .config, relativePath: "params.json"),
                    ArtifactSpec(id: "model-weights", role: .weights, relativePath: "model.safetensors"),
                    ArtifactSpec(id: "model-index", role: .weights, relativePath: "model.safetensors.index.json"),
                    ArtifactSpec(id: "tokenizer", role: .tokenizer, relativePath: "tekken.json"),
                ] + presetVoices.map { preset in
                    ArtifactSpec(
                        id: "voice-embedding-safetensors-\(preset.name)",
                        role: .voiceAsset,
                        relativePath: "voice_embedding/\(preset.name).safetensors"
                    )
                },
                preferredQuantization: "6bit",
                remoteURL: URL(string: "https://huggingface.co/mlx-community/Voxtral-4B-TTS-2603-mlx-6bit")
            ),
        ]
    }
}
