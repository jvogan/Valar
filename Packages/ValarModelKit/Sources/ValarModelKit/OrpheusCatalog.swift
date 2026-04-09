import Foundation

public enum OrpheusSurface: String, CaseIterable, Codable, Sendable {
    case orpheus3B_bf16

    public var familyID: ModelFamilyID { .orpheus }
    public var domain: ModelDomain { .tts }

    public var capabilities: Set<CapabilityID> {
        [.speechSynthesis, .tokenization, .presetVoices]
    }

    public var defaultSampleRate: Double { 24_000 }

    public var promptSchema: PromptSchema? {
        PromptSchema(kind: "orpheus.tts.base", requiredFields: ["text"])
    }
}

public enum OrpheusCatalog {
    public static let presetVoices: [PresetVoiceSpec] = [
        PresetVoiceSpec(name: "tara", displayName: "Tara"),
        PresetVoiceSpec(name: "leah", displayName: "Leah"),
        PresetVoiceSpec(name: "jess", displayName: "Jess"),
        PresetVoiceSpec(name: "leo", displayName: "Leo"),
        PresetVoiceSpec(name: "dan", displayName: "Dan"),
        PresetVoiceSpec(name: "mia", displayName: "Mia"),
        PresetVoiceSpec(name: "zac", displayName: "Zac"),
        PresetVoiceSpec(name: "zoe", displayName: "Zoe"),
    ]

    public static func makeManifest(
        surface: OrpheusSurface,
        modelID: ModelIdentifier,
        displayName: String,
        artifacts: [ArtifactSpec],
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
                    preferredQuantization: "bf16",
                    requiresLocalExecution: true
                ),
            ],
            artifacts: artifacts,
            tokenizer: TokenizerSpec(kind: "huggingface", configPath: "tokenizer.json"),
            audio: AudioConstraint(
                defaultSampleRate: surface.defaultSampleRate,
                minimumSampleRate: surface.defaultSampleRate,
                maximumSampleRate: surface.defaultSampleRate
            ),
            promptSchema: surface.promptSchema,
            presetVoices: presetVoices,
            licenses: [
                LicenseSpec(name: "Model license", sourceURL: remoteURL, requiresAttribution: true),
            ],
            notes: "LLM-based TTS with 8 preset voices (tara, leah, jess, leo, dan, mia, zac, zoe). Supports inline emotion tags such as <laugh>, <chuckle>, <sigh>, <cough>, <sniffle>, <groan>, <yawn>, and <gasp>. Requires snac_24khz codec (auto-fetched)."
        )

        return SupportedModelCatalogEntry(
            manifest: manifest,
            remoteURL: remoteURL,
            requiresManualDownload: remoteURL != nil,
            tags: ["orpheus", surface.familyID.rawValue]
        )
    }

    public static var supportedEntries: [SupportedModelCatalogEntry] {
        [
            makeManifest(
                surface: .orpheus3B_bf16,
                modelID: "mlx-community/orpheus-3b-0.1-ft-bf16",
                displayName: "Orpheus 3B",
                artifacts: [
                    ArtifactSpec(id: "model-config", role: .config, relativePath: "config.json"),
                    ArtifactSpec(id: "model-weights", role: .weights, relativePath: "model.safetensors"),
                    ArtifactSpec(id: "tokenizer", role: .tokenizer, relativePath: "tokenizer.json"),
                    ArtifactSpec(id: "tokenizer-config", role: .config, relativePath: "tokenizer_config.json"),
                ],
                remoteURL: URL(string: "https://huggingface.co/mlx-community/orpheus-3b-0.1-ft-bf16")
            ),
        ]
    }
}
