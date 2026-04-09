import Foundation

public enum MarvisSurface: String, CaseIterable, Codable, Sendable {
    case marvis100M_6bit

    public var familyID: ModelFamilyID { .marvis }
    public var domain: ModelDomain { .tts }

    public var capabilities: Set<CapabilityID> {
        [.speechSynthesis, .tokenization]
    }

    public var defaultSampleRate: Double { 24_000 }

    public var promptSchema: PromptSchema? {
        PromptSchema(kind: "marvis.tts.base", requiredFields: ["text"])
    }
}

public enum MarvisCatalog {
    public static func makeManifest(
        surface: MarvisSurface,
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
                    preferredQuantization: "6bit",
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
            licenses: [
                LicenseSpec(name: "Model license", sourceURL: remoteURL, requiresAttribution: true),
            ]
        )

        return SupportedModelCatalogEntry(
            manifest: manifest,
            remoteURL: remoteURL,
            requiresManualDownload: remoteURL != nil,
            tags: ["marvis", surface.familyID.rawValue]
        )
    }

    public static var supportedEntries: [SupportedModelCatalogEntry] {
        [
            makeManifest(
                surface: .marvis100M_6bit,
                modelID: "Marvis-AI/marvis-tts-100m-v0.2-MLX-6bit",
                displayName: "Marvis TTS 100M (6-bit)",
                artifacts: [
                    ArtifactSpec(id: "model-config", role: .config, relativePath: "config.json"),
                    ArtifactSpec(
                        id: "model-weights",
                        role: .weights,
                        relativePath: "model.safetensors",
                        sha256: "42e4baf974701d7c4d46c2dc1a68ce8caded1ce626bdd6160e9c709264399058",
                        sizeBytes: 376_602_670
                    ),
                    ArtifactSpec(id: "tokenizer", role: .tokenizer, relativePath: "tokenizer.json"),
                    ArtifactSpec(id: "tokenizer-config", role: .config, relativePath: "tokenizer_config.json"),
                    ArtifactSpec(id: "voice-prompts", role: .voiceAsset, relativePath: "prompts/", required: false),
                ],
                remoteURL: URL(string: "https://huggingface.co/Marvis-AI/marvis-tts-100m-v0.2-MLX-6bit")
            ),
        ]
    }
}
