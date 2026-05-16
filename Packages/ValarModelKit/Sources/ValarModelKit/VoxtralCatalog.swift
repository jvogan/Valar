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

    private struct ArtifactIntegrity {
        let sha256: String
        let sizeBytes: Int
    }

    private static let rawArtifactIntegrity: [String: ArtifactIntegrity] = [
        "params.json": .init(sha256: "f6408ee76dea8da16ce40ac66729d59406019ea71cdb9d656709a38d2e58691e", sizeBytes: 3_482),
        "consolidated.safetensors": .init(sha256: "66c4fd998db10e1a6d9cc5baa10e6264bf10701ec22ccdc0822c7dcc45dbe55b", sizeBytes: 8_004_752_248),
        "tekken.json": .init(sha256: "587989c9f56676b35e7d16d6fc61461301e402d908392a8ce16f0349f61b56d7", sizeBytes: 14_894_731),
        "voice_embedding/ar_male.pt": .init(sha256: "f44603f6433cbb4b2abc7f496a382632171118557a175cb385df168a0dc20464", sizeBytes: 413_253),
        "voice_embedding/casual_female.pt": .init(sha256: "780637984644064ee22e60b3152e0cd43fa64b2dcd39d9cab6cd2c62f2ce0342", sizeBytes: 1_316_421),
        "voice_embedding/casual_male.pt": .init(sha256: "7a056c9156ad0058e9d1368363bf3a25a9fcd8fe53e211ffac97de0bbffb3504", sizeBytes: 904_773),
        "voice_embedding/cheerful_female.pt": .init(sha256: "75fe69c8fcb5a0883a3d0bc1215b28f28cc0586aff5732eeebd2b254e8288253", sizeBytes: 812_613),
        "voice_embedding/de_female.pt": .init(sha256: "282fc191fda496de2ebf2c809acb44056dde6fbe2f1cb99e85e67985bc6f6619", sizeBytes: 904_773),
        "voice_embedding/de_male.pt": .init(sha256: "bd75d9fd3ffb9df0481668ce8781287a58f552e2388c5bbc0efdd4ebff0421bf", sizeBytes: 1_003_077),
        "voice_embedding/es_female.pt": .init(sha256: "90e01ad34f231cc881987c3b1c0728853fd9b904e52c296a07c71a132949d8a6", sizeBytes: 849_477),
        "voice_embedding/es_male.pt": .init(sha256: "ec116d8f4a102291bae3d9156d7c3222d9e1056020bf5894a7504bfc09640fdf", sizeBytes: 1_279_557),
        "voice_embedding/fr_female.pt": .init(sha256: "82628d963670f919aa302f9c8a7336c745418a145934edb211810b07d9c8b852", sizeBytes: 597_573),
        "voice_embedding/fr_male.pt": .init(sha256: "73395073472be3fb586b487705ac4ebf35f99db664f56400137e8bfcfe4cd8a8", sizeBytes: 597_573),
        "voice_embedding/hi_female.pt": .init(sha256: "aa7718cdd6f65735226bcc701379fdec64f36d0207ca79fc4c61b445ca7bde82", sizeBytes: 529_989),
        "voice_embedding/hi_male.pt": .init(sha256: "c3cde36ab9a336f67fd33b46435cdf645cff9e10117f13bcbcb67b44b80a11b0", sizeBytes: 579_141),
        "voice_embedding/it_female.pt": .init(sha256: "29e1714bdb3ce0726e590ce1862fbe953c168ba51a05bc7daa8cb35cddc312b4", sizeBytes: 1_058_373),
        "voice_embedding/it_male.pt": .init(sha256: "b98ba2253e2a0b872e20d33d29cab32263cc81062c01e3f5a8696de89e6f47b1", sizeBytes: 1_033_797),
        "voice_embedding/neutral_female.pt": .init(sha256: "2a03f4008614da7b1505a360a6b0d58d94dd72b0b0f49bf216e39de5eb733c61", sizeBytes: 1_340_997),
        "voice_embedding/neutral_male.pt": .init(sha256: "439df812990e6e4bcc6010ca12f12df90916e862bc1e1b56036d6433b892834e", sizeBytes: 1_039_941),
        "voice_embedding/nl_female.pt": .init(sha256: "b1bad34c22e0563f05c1f13c1db96680778c297aea6a5c0bb202950648b796b6", sizeBytes: 898_629),
        "voice_embedding/nl_male.pt": .init(sha256: "43fd2de89dc08503f37ae3107273eeb3f2a6195d705ff58d2228b3b5642ff7de", sizeBytes: 849_477),
        "voice_embedding/pt_female.pt": .init(sha256: "82f1006b2cd69118cba67085daa1795d9dab90b9bc70e1392e77f82cb616c9ce", sizeBytes: 1_076_805),
        "voice_embedding/pt_male.pt": .init(sha256: "7b30dca6c5d16c7b10a1c09c53e971c1bb1fab65692d7244876fbdc4ad52ba18", sizeBytes: 886_341),
    ]

    private static let sharedMLXArtifactIntegrity: [String: ArtifactIntegrity] = [
        "params.json": .init(sha256: "f6408ee76dea8da16ce40ac66729d59406019ea71cdb9d656709a38d2e58691e", sizeBytes: 3_482),
        "model.safetensors.index.json": .init(sha256: "6f550dfaf7569a5369fd22b361e5a5838bea06c624ded0c6dab60f1dcf1032d9", sizeBytes: 73_050),
        "tekken.json": .init(sha256: "587989c9f56676b35e7d16d6fc61461301e402d908392a8ce16f0349f61b56d7", sizeBytes: 14_894_731),
        "voice_embedding/ar_male.safetensors": .init(sha256: "f4c480657b730c169614c66dcb26684bc234ee2829cb2a9e490c300f60b1782a", sizeBytes: 411_736),
        "voice_embedding/casual_female.safetensors": .init(sha256: "9a2027d9265fd7ef4a55294b10ae8b2095dcad7547d50762a5012a3955ca0860", sizeBytes: 1_314_904),
        "voice_embedding/casual_male.safetensors": .init(sha256: "2056ade898f6f1b04c1af764f54d705038ac046f0b4967457a36989b69730fa8", sizeBytes: 903_256),
        "voice_embedding/cheerful_female.safetensors": .init(sha256: "35441e3030ba0076356ef1ad54fbfd5adfbde9e34bbf8a1a6535d6efabe63af1", sizeBytes: 811_096),
        "voice_embedding/de_female.safetensors": .init(sha256: "a89197fe5e77a2dd3cfc4ac0ed1cb5248e4e1ac26fdb13500e9782e43a90a69e", sizeBytes: 903_256),
        "voice_embedding/de_male.safetensors": .init(sha256: "cabfeeb98db0b713a80f234e11d90c3105fe83e64de75876509c3cda43946656", sizeBytes: 1_001_560),
        "voice_embedding/es_female.safetensors": .init(sha256: "52bca0b4e770afaccc3f2899d7396009422a0b3fdb5a6a96c523bcdc2f35c165", sizeBytes: 847_960),
        "voice_embedding/es_male.safetensors": .init(sha256: "b63fd5dff4dbc5070d470a364b8691f63fd6c10412a2692101bf6a90549c36e1", sizeBytes: 1_278_040),
        "voice_embedding/fr_female.safetensors": .init(sha256: "e8be0f673696dc1bf668fa6cc9999ab09ab43b29a6d6e85c5ac3f9f8ed43c449", sizeBytes: 596_056),
        "voice_embedding/fr_male.safetensors": .init(sha256: "d34fde0b57acaea4c3f78e841c93dd290f104a167dd0bea9e0673d9cdfedadd5", sizeBytes: 596_056),
        "voice_embedding/hi_female.safetensors": .init(sha256: "01e987092046cf5dc1041f00d10b0763f9ab4080c3bf124d97a01ccb04428e26", sizeBytes: 528_472),
        "voice_embedding/hi_male.safetensors": .init(sha256: "342d40fcda1e2a6ded081f0d2d8eefadcc7b8102a20da6e7e803ef4b4a21785c", sizeBytes: 577_624),
        "voice_embedding/it_female.safetensors": .init(sha256: "378b9bab596a68780ca07f4bf2032a9c70847825772aed9ee148ba8173b477d3", sizeBytes: 1_056_856),
        "voice_embedding/it_male.safetensors": .init(sha256: "05e313648254a57e4ca7e503b416590a4c832a106f7201efd4dea02ecce79035", sizeBytes: 1_032_280),
        "voice_embedding/neutral_female.safetensors": .init(sha256: "e229d5646ab8c2cad1b4e24cd63b88192f34da61de8f0595291799232460752f", sizeBytes: 1_339_480),
        "voice_embedding/neutral_male.safetensors": .init(sha256: "b114132e7301b2d2308ff7b0c1843574bab9ffeac5fc2dd48ca27db70a24ea0c", sizeBytes: 1_038_424),
        "voice_embedding/nl_female.safetensors": .init(sha256: "0124a62762a89b9a4ef0ff0c016db045ccfe30ed9fa3d886db0e78e444673ea3", sizeBytes: 897_112),
        "voice_embedding/nl_male.safetensors": .init(sha256: "6f931e4c34cf496cce4ed7ce5d8af286140c4969860f9410b12237ff28de2d33", sizeBytes: 847_960),
        "voice_embedding/pt_female.safetensors": .init(sha256: "ce23b97302fe6ca58507b1ae2c2fe1d0a472268b1faf597901cbf75e326a08f9", sizeBytes: 1_075_288),
        "voice_embedding/pt_male.safetensors": .init(sha256: "4ccd63a39d35f483c89485112d35a8a2121cbb0e6d5d124e403e40d03e0c0e82", sizeBytes: 884_824),
    ]

    private static let mlx4BitOnlyArtifactIntegrity: [String: ArtifactIntegrity] = [
        "model.safetensors": .init(sha256: "a62a28f02ce54f9157877df44ce2da92bed97159ab19c2878445d3ec4d357786", sizeBytes: 2_509_879_373),
    ]

    private static let mlx6BitOnlyArtifactIntegrity: [String: ArtifactIntegrity] = [
        "model.safetensors": .init(sha256: "faea8347d8d27f7f0d1a338c6cccd887d7b28df8978d447e5f1c20d414354af1", sizeBytes: 3_465_520_393),
    ]

    private static let mlx4BitArtifactIntegrity = sharedMLXArtifactIntegrity.merging(
        mlx4BitOnlyArtifactIntegrity,
        uniquingKeysWith: { _, new in new }
    )

    private static let mlx6BitArtifactIntegrity = sharedMLXArtifactIntegrity.merging(
        mlx6BitOnlyArtifactIntegrity,
        uniquingKeysWith: { _, new in new }
    )

    private static func artifactSpec(
        id: String,
        role: ArtifactRole,
        relativePath: String,
        integrity: [String: ArtifactIntegrity]
    ) -> ArtifactSpec {
        let resolvedIntegrity = integrity[relativePath]
        return ArtifactSpec(
            id: id,
            role: role,
            relativePath: relativePath,
            sha256: resolvedIntegrity?.sha256,
            sizeBytes: resolvedIntegrity?.sizeBytes
        )
    }

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
            artifactSpec(
                id: "\(rawVoiceEmbeddingArtifactIDPrefix)\(preset.name)",
                role: .voiceAsset,
                relativePath: "voice_embedding/\(preset.name).pt",
                integrity: rawArtifactIntegrity
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
                    artifactSpec(id: "model-config", role: .config, relativePath: "params.json", integrity: rawArtifactIntegrity),
                    artifactSpec(id: "model-weights", role: .weights, relativePath: "consolidated.safetensors", integrity: rawArtifactIntegrity),
                    artifactSpec(id: "tokenizer", role: .tokenizer, relativePath: "tekken.json", integrity: rawArtifactIntegrity),
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
                    artifactSpec(id: "model-config", role: .config, relativePath: "params.json", integrity: mlx4BitArtifactIntegrity),
                    artifactSpec(id: "model-weights", role: .weights, relativePath: "model.safetensors", integrity: mlx4BitArtifactIntegrity),
                    artifactSpec(id: "model-index", role: .weights, relativePath: "model.safetensors.index.json", integrity: mlx4BitArtifactIntegrity),
                    artifactSpec(id: "tokenizer", role: .tokenizer, relativePath: "tekken.json", integrity: mlx4BitArtifactIntegrity),
                ] + presetVoices.map { preset in
                    artifactSpec(
                        id: "voice-embedding-safetensors-\(preset.name)",
                        role: .voiceAsset,
                        relativePath: "voice_embedding/\(preset.name).safetensors",
                        integrity: mlx4BitArtifactIntegrity
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
                    artifactSpec(id: "model-config", role: .config, relativePath: "params.json", integrity: mlx6BitArtifactIntegrity),
                    artifactSpec(id: "model-weights", role: .weights, relativePath: "model.safetensors", integrity: mlx6BitArtifactIntegrity),
                    artifactSpec(id: "model-index", role: .weights, relativePath: "model.safetensors.index.json", integrity: mlx6BitArtifactIntegrity),
                    artifactSpec(id: "tokenizer", role: .tokenizer, relativePath: "tekken.json", integrity: mlx6BitArtifactIntegrity),
                ] + presetVoices.map { preset in
                    artifactSpec(
                        id: "voice-embedding-safetensors-\(preset.name)",
                        role: .voiceAsset,
                        relativePath: "voice_embedding/\(preset.name).safetensors",
                        integrity: mlx6BitArtifactIntegrity
                    )
                },
                preferredQuantization: "6bit",
                remoteURL: URL(string: "https://huggingface.co/mlx-community/Voxtral-4B-TTS-2603-mlx-6bit")
            ),
        ]
    }
}
