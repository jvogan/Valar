import Foundation

public enum PocketTTSSurface: String, CaseIterable, Codable, Sendable {
    case pocketTTS_bf16

    public var familyID: ModelFamilyID { .pocketTTS }
    public var domain: ModelDomain { .tts }

    public var capabilities: Set<CapabilityID> {
        [.speechSynthesis, .tokenization]
    }

    public var defaultSampleRate: Double { 24_000 }

    public var promptSchema: PromptSchema? {
        PromptSchema(kind: "pocket_tts.tts.base", requiredFields: ["text"])
    }
}

public enum PocketTTSCatalog {
    public static func makeManifest(
        surface: PocketTTSSurface,
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
            supportTier: .preview,
            releaseEligible: false,
            licenses: [
                LicenseSpec(name: "Model license", sourceURL: remoteURL, requiresAttribution: true),
            ]
        )

        return SupportedModelCatalogEntry(
            manifest: manifest,
            remoteURL: remoteURL,
            requiresManualDownload: remoteURL != nil,
            tags: ["pocket_tts", surface.familyID.rawValue]
        )
    }

    public static var supportedEntries: [SupportedModelCatalogEntry] {
        [
            makeManifest(
                surface: .pocketTTS_bf16,
                modelID: "mlx-community/pocket-tts-0.1-bf16",
                displayName: "Pocket TTS",
                artifacts: [
                    ArtifactSpec(id: "model-config", role: .config, relativePath: "config.json"),
                    ArtifactSpec(id: "model-weights", role: .weights, relativePath: "model.safetensors"),
                    ArtifactSpec(id: "embeddings", role: .voiceAsset, relativePath: "embeddings/", required: false),
                ],
                remoteURL: URL(string: "https://huggingface.co/mlx-community/pocket-tts-0.1-bf16")
            ),
        ]
    }
}
