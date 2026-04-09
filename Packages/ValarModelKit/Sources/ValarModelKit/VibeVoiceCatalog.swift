import Foundation

public enum VibeVoiceSurface: String, CaseIterable, Codable, Sendable {
    case vibeVoiceRealtime05B4Bit

    public var familyID: ModelFamilyID { .vibevoiceRealtimeTTS }
    public var domain: ModelDomain { .tts }

    public var capabilities: Set<CapabilityID> {
        [.speechSynthesis, .multilingual, .presetVoices, .streaming]
    }

    public var defaultSampleRate: Double { 24_000 }

    /// Languages covered by the upstream Realtime model's quality-gated surface.
    ///
    /// The bundled pack still includes `in-Samuel_man`, but Hindi remains exploratory and is
    /// excluded from the guaranteed supported-language list until its quality clears the live
    /// acceptance suite consistently.
    public var supportedLanguages: [String] { ["en", "de", "fr", "it", "ja", "ko", "nl", "pl", "pt", "es"] }

    public var promptSchema: PromptSchema? {
        PromptSchema(
            kind: "vibevoice.tts.preset_voice",
            requiredFields: ["text"],
            optionalFields: ["voice", "cfgScale", "ddpmSteps"]
        )
    }
}

public enum VibeVoiceCatalog {
    public static let mlx4BitModelIdentifier: ModelIdentifier = "mlx-community/VibeVoice-Realtime-0.5B-4bit"
    public static let tokenizerSourceModelIdentifier: ModelIdentifier = "Qwen/Qwen2.5-0.5B"
    public static let qualityTierByLanguage: [String: ModelLanguageSupportTier] = [
        "en": .supported,
        "de": .preview,
        "es": .preview,
        "fr": .preview,
        "it": .preview,
        "ja": .preview,
        "ko": .preview,
        "nl": .preview,
        "pl": .preview,
        "pt": .preview,
        "hi": .experimental,
    ]
    public static let exploratoryLanguageCodes = ["hi"]
    public static let releaseVisibleLanguageCodes = qualityTierByLanguage.compactMap { entry -> String? in
        let (language, tier) = entry
        switch tier {
        case .experimental:
            return nil
        case .supported, .preview, .advisory:
            return language
        }
    }.sorted()
    private static let defaultPresetByLanguage: [String: String] = [
        "de": "de-Spk1_woman",
        "en": "en-Emma_woman",
        "es": "sp-Spk0_woman",
        "fr": "fr-Spk1_woman",
        "hi": "in-Samuel_man",
        "it": "it-Spk0_woman",
        "ja": "jp-Spk1_woman",
        "ko": "kr-Spk0_woman",
        "nl": "nl-Spk1_woman",
        "pl": "pl-Spk1_woman",
        "pt": "pt-Spk0_woman",
    ]

    // MARK: - Preset Voices

    /// 25 bundled preset voices shipped as KV-cache snapshots in `voices/<name>.safetensors`.
    /// Language affinity uses ISO 639-1 lowercase codes (e.g. `"en"`).
    public static let presetVoices: [PresetVoiceSpec] = [
        // English (6 named voices)
        PresetVoiceSpec(name: "en-Carter_man", displayName: "Carter", languageAffinity: ["en"]),
        PresetVoiceSpec(name: "en-Davis_man", displayName: "Davis", languageAffinity: ["en"]),
        PresetVoiceSpec(name: "en-Emma_woman", displayName: "Emma", languageAffinity: ["en"]),
        PresetVoiceSpec(name: "en-Frank_man", displayName: "Frank", languageAffinity: ["en"]),
        PresetVoiceSpec(name: "en-Grace_woman", displayName: "Grace", languageAffinity: ["en"]),
        PresetVoiceSpec(name: "en-Mike_man", displayName: "Mike", languageAffinity: ["en"]),
        // German
        PresetVoiceSpec(name: "de-Spk0_man", displayName: "German Male", languageAffinity: ["de"]),
        PresetVoiceSpec(name: "de-Spk1_woman", displayName: "German Female", languageAffinity: ["de"]),
        // French
        PresetVoiceSpec(name: "fr-Spk0_man", displayName: "French Male", languageAffinity: ["fr"]),
        PresetVoiceSpec(name: "fr-Spk1_woman", displayName: "French Female", languageAffinity: ["fr"]),
        // Hindi
        PresetVoiceSpec(name: "in-Samuel_man", displayName: "Samuel", languageAffinity: ["hi"]),
        // Italian
        PresetVoiceSpec(name: "it-Spk0_woman", displayName: "Italian Female", languageAffinity: ["it"]),
        PresetVoiceSpec(name: "it-Spk1_man", displayName: "Italian Male", languageAffinity: ["it"]),
        // Japanese
        PresetVoiceSpec(name: "jp-Spk0_man", displayName: "Japanese Male", languageAffinity: ["ja"]),
        PresetVoiceSpec(name: "jp-Spk1_woman", displayName: "Japanese Female", languageAffinity: ["ja"]),
        // Korean
        PresetVoiceSpec(name: "kr-Spk0_woman", displayName: "Korean Female", languageAffinity: ["ko"]),
        PresetVoiceSpec(name: "kr-Spk1_man", displayName: "Korean Male", languageAffinity: ["ko"]),
        // Dutch
        PresetVoiceSpec(name: "nl-Spk0_man", displayName: "Dutch Male", languageAffinity: ["nl"]),
        PresetVoiceSpec(name: "nl-Spk1_woman", displayName: "Dutch Female", languageAffinity: ["nl"]),
        // Polish
        PresetVoiceSpec(name: "pl-Spk0_man", displayName: "Polish Male", languageAffinity: ["pl"]),
        PresetVoiceSpec(name: "pl-Spk1_woman", displayName: "Polish Female", languageAffinity: ["pl"]),
        // Portuguese
        PresetVoiceSpec(name: "pt-Spk0_woman", displayName: "Portuguese Female", languageAffinity: ["pt"]),
        PresetVoiceSpec(name: "pt-Spk1_man", displayName: "Portuguese Male", languageAffinity: ["pt"]),
        // Spanish
        PresetVoiceSpec(name: "sp-Spk0_woman", displayName: "Spanish Female", languageAffinity: ["es"]),
        PresetVoiceSpec(name: "sp-Spk1_man", displayName: "Spanish Male", languageAffinity: ["es"]),
    ]

    public static var supportedLanguageCodes: [String] {
        VibeVoiceSurface.vibeVoiceRealtime05B4Bit.supportedLanguages
    }

    public static func primaryLanguage(for preset: PresetVoiceSpec) -> String? {
        preset.languageAffinity.first?.lowercased()
    }

    public static func languageSupportTier(for language: String) -> ModelLanguageSupportTier? {
        let normalizedLanguage = normalizedLanguageCode(language) ?? language.lowercased()
        return qualityTierByLanguage[normalizedLanguage]
    }

    public static func presetSupportTier(for presetIdentifier: String) -> ModelLanguageSupportTier? {
        guard let preset = preset(matching: presetIdentifier),
              let language = primaryLanguage(for: preset) else {
            return nil
        }
        return languageSupportTier(for: language)
    }

    public static func isReleaseVisiblePreset(_ presetIdentifier: String) -> Bool {
        guard let tier = presetSupportTier(for: presetIdentifier) else {
            return true
        }
        return tier != .experimental
    }

    public static func presets(forLanguage language: String) -> [PresetVoiceSpec] {
        let normalizedLanguage = normalizedLanguageCode(language) ?? language.lowercased()
        return presetVoices.filter { preset in
            preset.languageAffinity.map { $0.lowercased() }.contains(normalizedLanguage)
        }
    }

    public static func defaultPreset(forLanguage language: String) -> PresetVoiceSpec? {
        let normalizedLanguage = normalizedLanguageCode(language) ?? language.lowercased()
        guard let presetName = defaultPresetByLanguage[normalizedLanguage] else {
            return nil
        }
        return presetVoices.first { $0.name == presetName }
    }

    public static func preset(matching input: String) -> PresetVoiceSpec? {
        let lowered = input.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard lowered.isEmpty == false else {
            return nil
        }
        if let exact = presetVoices.first(where: { $0.name.lowercased() == lowered }) {
            return exact
        }
        if let displayName = presetVoices.first(where: { $0.displayName?.lowercased() == lowered }) {
            return displayName
        }
        if let aliased = presetVoices.first(where: { $0.aliases.contains(lowered) }) {
            return aliased
        }
        return nil
    }

    public static func acceptsPresetIdentifier(_ input: String?) -> Bool {
        guard let lowered = input?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased(),
            lowered.isEmpty == false else {
            return false
        }
        return lowered == "random" || preset(matching: lowered) != nil
    }

    public static func normalizedLanguageCode(_ input: String?) -> String? {
        guard let trimmed = input?.trimmingCharacters(in: .whitespacesAndNewlines),
              trimmed.isEmpty == false else {
            return nil
        }

        let lowered = trimmed.lowercased().replacingOccurrences(of: "_", with: "-")
        let prefix = String(lowered.split(separator: "-").first ?? "")

        switch prefix {
        case "en", "english":
            return "en"
        case "de", "german", "deutsch":
            return "de"
        case "fr", "french":
            return "fr"
        case "es", "sp", "spanish":
            return "es"
        case "it", "italian":
            return "it"
        case "nl", "dutch":
            return "nl"
        case "pt", "portuguese":
            return "pt"
        case "pl", "polish":
            return "pl"
        case "hi", "hindi", "in":
            return "hi"
        case "ja", "japanese", "jp":
            return "ja"
        case "ko", "korean", "kr":
            return "ko"
        default:
            return nil
        }
    }

    /// Resolve a voice name to the canonical preset name. Accepts "random" for random selection.
    public static func resolvePresetName(_ input: String) -> String? {
        let lowered = input.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if lowered == "random" {
            return presetVoices.randomElement()?.name
        }
        return preset(matching: lowered)?.name
    }

    // MARK: - Voice Artifacts

    static var voiceCacheArtifacts: [ArtifactSpec] {
        presetVoices.map { preset in
            ArtifactSpec(
                id: "voice-cache-\(preset.name)",
                role: .voiceAsset,
                relativePath: "voices/\(preset.name).safetensors"
            )
        }
    }

    // MARK: - Manifest

    public static func makeManifest(
        surface: VibeVoiceSurface,
        modelID: ModelIdentifier,
        displayName: String,
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
                    preferredQuantization: "4bit",
                    requiresLocalExecution: true
                ),
            ],
            artifacts: [
                ArtifactSpec(id: "model-config", role: .config, relativePath: "config.json"),
                ArtifactSpec(id: "model-weights", role: .weights, relativePath: "model.safetensors"),
                ArtifactSpec(id: "tokenizer", role: .tokenizer, relativePath: "tokenizer.json"),
                ArtifactSpec(id: "tokenizer-config", role: .tokenizer, relativePath: "tokenizer_config.json"),
                ArtifactSpec(id: "special-tokens-map", role: .auxiliary, relativePath: "special_tokens_map.json", required: false),
                ArtifactSpec(id: "added-tokens", role: .auxiliary, relativePath: "added_tokens.json", required: false),
                ArtifactSpec(id: "preprocessor-config", role: .config, relativePath: "preprocessor_config.json"),
            ] + voiceCacheArtifacts,
            tokenizer: TokenizerSpec(
                kind: "huggingface",
                configPath: "tokenizer.json"
            ),
            audio: AudioConstraint(
                defaultSampleRate: surface.defaultSampleRate,
                minimumSampleRate: surface.defaultSampleRate,
                maximumSampleRate: surface.defaultSampleRate,
                supportsReferenceAudio: false
            ),
            promptSchema: surface.promptSchema,
            supportedLanguages: surface.supportedLanguages,
            presetVoices: presetVoices,
            supportTier: .preview,
            releaseEligible: true,
            qualityTierByLanguage: qualityTierByLanguage,
            licenses: [
                LicenseSpec(
                    name: "MIT License",
                    spdxIdentifier: "MIT",
                    relativePath: "LICENSE",
                    sourceURL: remoteURL,
                    requiresAttribution: false,
                    isNonCommercial: false
                ),
            ],
            notes: "Fast preset-voice realtime TTS. English is release-supported. German, Spanish, French, Italian, Japanese, Korean, Dutch, Polish, and Portuguese remain preview/advisory. Hindi stays exploratory and is hidden from release-facing UI by default. The MLX pack reuses the Qwen/Qwen2.5-0.5B tokenizer and materializes it automatically during install when needed."
        )

        return SupportedModelCatalogEntry(
            manifest: manifest,
            remoteURL: remoteURL,
            isRecommended: false,
            distributionTier: .compatibilityPreview,
            tags: ["vibevoice", "preset-voices", "streaming", "fast", surface.familyID.rawValue]
        )
    }

    // MARK: - Supported Entries

    public static var supportedEntries: [SupportedModelCatalogEntry] {
        [
            makeManifest(
                surface: .vibeVoiceRealtime05B4Bit,
                modelID: mlx4BitModelIdentifier,
                displayName: "VibeVoice Realtime 0.5B (4-bit)",
                remoteURL: URL(string: "https://huggingface.co/mlx-community/VibeVoice-Realtime-0.5B-4bit")
            ),
        ]
    }
}
