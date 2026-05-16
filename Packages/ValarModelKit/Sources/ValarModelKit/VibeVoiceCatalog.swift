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

    private struct ArtifactIntegrity {
        let sha256: String
        let sizeBytes: Int
    }

    private static let artifactIntegrity: [String: ArtifactIntegrity] = [
        "config.json": .init(sha256: "ef672aca9e7deb835925970492a10e606d1b2c7fc741dd30cf36e3efe2886717", sizeBytes: 2655),
        "model.safetensors": .init(sha256: "d4e33a2daca2dd866b472e42210701fe9e28dc6fcec649b2a1fd05e5885b30bd", sizeBytes: 632_644_595),
        "tokenizer.json": .init(sha256: "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539", sizeBytes: 7_031_645),
        "tokenizer_config.json": .init(sha256: "c91efca15ceff6e9ee9424db58a6f59cd41294e550a86cbd07e3c1fb500b34f9", sizeBytes: 7_228),
        "preprocessor_config.json": .init(sha256: "ebf514b5d30a012e5ae00d9a19d01e735e35b27768c3926d980815db8fa742e5", sizeBytes: 360),
        "voices/de-Spk0_man.safetensors": .init(sha256: "2e34ddef90b8585c6298c2545841ef73c67e6adf8f728376107e8088244c1463", sizeBytes: 7_017_168),
        "voices/de-Spk1_woman.safetensors": .init(sha256: "7a6c9efd03b06a2a6c2cb4fe88a43ec5b4ea5f3fe46d597ed36056f324ab415c", sizeBytes: 5_268_176),
        "voices/en-Carter_man.safetensors": .init(sha256: "1b3efb89bc26bc14d86095da9b26b0aaf5989e8ed75e39efa958088bf301160c", sizeBytes: 4_241_352),
        "voices/en-Davis_man.safetensors": .init(sha256: "6d689ac3f6f630fd1617814a15ab165544772a208ce305d3779feb08f033f1e0", sizeBytes: 2_456_752),
        "voices/en-Emma_woman.safetensors": .init(sha256: "8572620ccf3384529c8fce7b211871482cfa1fc8e3068f80576b7ea15257e819", sizeBytes: 3_328_696),
        "voices/en-Frank_man.safetensors": .init(sha256: "869af2fd5e83b3f70cdc23b28fdc1ef82c11e7122c387adb8bc56902b323efb2", sizeBytes: 3_345_080),
        "voices/en-Grace_woman.safetensors": .init(sha256: "7b0cb4438eb8a2cc0d45eb8c4d724d27fb459f999a8ea0bf9ab221e67cb92ba6", sizeBytes: 2_758_064),
        "voices/en-Mike_man.safetensors": .init(sha256: "10524823aa1f90cd4cec05828f16d20d090d83ac3eb682646aa43e45c1f9dc0a", sizeBytes: 1_993_376),
        "voices/fr-Spk0_man.safetensors": .init(sha256: "8d6d1df2b70d05680bf2da34bb606b991eefa35068f091e2f17c603af8c8726e", sizeBytes: 4_363_984),
        "voices/fr-Spk1_woman.safetensors": .init(sha256: "b9ce6e695df9bef90ce926b6c570ce211c456d969164208c3445a4ac257bd1e4", sizeBytes: 4_249_296),
        "voices/in-Samuel_man.safetensors": .init(sha256: "3a22a118b02f1d2dbafe02284152bd83413dad344c7c65a66dc83ae2b528bc64", sizeBytes: 3_768_000),
        "voices/it-Spk0_woman.safetensors": .init(sha256: "87ba3fcecc31c1d639a86bf5631096e06297bcdff2b455756b014bd4ccad9672", sizeBytes: 2_529_456),
        "voices/it-Spk1_man.safetensors": .init(sha256: "b5c7cb194cca41d43e15e6cadbd7a7add10bc3d5fe53ef287979129f8fa90c21", sizeBytes: 2_832_056),
        "voices/jp-Spk0_man.safetensors": .init(sha256: "ab53e74e00e87577506db3817985bd3584fab079acfc80615dccd160435b3476", sizeBytes: 4_645_840),
        "voices/jp-Spk1_woman.safetensors": .init(sha256: "b331ec494ce24227ee4bd8a834aa407f3fb6d0fde9fd2ad6ac739f5df7b96a44", sizeBytes: 4_615_120),
        "voices/kr-Spk0_woman.safetensors": .init(sha256: "2ac9224493d510782c7ca2294c4309edbd74e5871a76fa6b8fe0c408c2ddba01", sizeBytes: 4_131_024),
        "voices/kr-Spk1_man.safetensors": .init(sha256: "67ccf3f76b5be71609d4c26ea64c6ce850ff8cfcf03ee623568a6bca916e40e9", sizeBytes: 5_842_640),
        "voices/nl-Spk0_man.safetensors": .init(sha256: "8645c3a0fd62e94609527c066fece280998824ebd6a4dae169d024ad3de085fc", sizeBytes: 3_681_992),
        "voices/nl-Spk1_woman.safetensors": .init(sha256: "cc8ac9607e1c61e3347ef3a25a0599886bf7c06dc69ab9a7633facfc80e6cbb4", sizeBytes: 5_073_104),
        "voices/pl-Spk0_man.safetensors": .init(sha256: "b8f61efaf59ea95f520b4231646f4d04dea367fdf39633e80106d20abc3873b8", sizeBytes: 3_728_328),
        "voices/pl-Spk1_woman.safetensors": .init(sha256: "7c3220f7ef26e8a06bfbb9f8acd0201139a9ee7be8a7b83df46bf28324e9677c", sizeBytes: 4_955_856),
        "voices/pt-Spk0_woman.safetensors": .init(sha256: "0828216576aae51b2ada2b542a393edf1177a4e2cd5338b230187162ae4fdcc3", sizeBytes: 2_245_544),
        "voices/pt-Spk1_man.safetensors": .init(sha256: "370345ebe6209bfea8084290104060297414e0d1015a9c7e5e9e3ce532bb17e2", sizeBytes: 3_532_488),
        "voices/sp-Spk0_woman.safetensors": .init(sha256: "7c05318ca1f3c94ba533d4f7c2fb4694332a7de67de68051018fbd97f158070e", sizeBytes: 4_221_128),
        "voices/sp-Spk1_man.safetensors": .init(sha256: "90c928352e59070b7a41c6d7ad76943f18e0f9d8dfc58d430d25e005c2287d79", sizeBytes: 5_107_920),
    ]

    private static func artifactSpec(
        id: String,
        role: ArtifactRole,
        relativePath: String,
        required: Bool = true
    ) -> ArtifactSpec {
        let integrity = artifactIntegrity[relativePath]
        return ArtifactSpec(
            id: id,
            role: role,
            relativePath: relativePath,
            sha256: integrity?.sha256,
            sizeBytes: integrity?.sizeBytes,
            required: required
        )
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
            artifactSpec(
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
                artifactSpec(id: "model-config", role: .config, relativePath: "config.json"),
                artifactSpec(id: "model-weights", role: .weights, relativePath: "model.safetensors"),
                artifactSpec(id: "tokenizer", role: .tokenizer, relativePath: "tokenizer.json"),
                artifactSpec(id: "tokenizer-config", role: .tokenizer, relativePath: "tokenizer_config.json"),
                artifactSpec(id: "special-tokens-map", role: .auxiliary, relativePath: "special_tokens_map.json", required: false),
                artifactSpec(id: "added-tokens", role: .auxiliary, relativePath: "added_tokens.json", required: false),
                artifactSpec(id: "preprocessor-config", role: .config, relativePath: "preprocessor_config.json"),
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
