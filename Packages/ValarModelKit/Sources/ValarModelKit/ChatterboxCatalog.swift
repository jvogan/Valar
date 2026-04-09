import Foundation

public enum ChatterboxSurface: String, CaseIterable, Codable, Sendable {
    case chatterboxRegular_fp16
    case chatterboxTurbo_fp16

    public var familyID: ModelFamilyID { .chatterbox }
    public var domain: ModelDomain { .tts }

    public var capabilities: Set<CapabilityID> {
        [.speechSynthesis, .tokenization, .voiceCloning, .audioConditioning]
    }

    public var defaultSampleRate: Double { 24_000 }

    public var promptSchema: PromptSchema? {
        PromptSchema(kind: "chatterbox.tts.base", requiredFields: ["text"])
    }
}

public enum ChatterboxCatalog {
    public static func makeManifest(
        surface: ChatterboxSurface,
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
                    preferredQuantization: "fp16",
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
            tags: ["chatterbox", surface.familyID.rawValue]
        )
    }

    public static var supportedEntries: [SupportedModelCatalogEntry] {
        [
            makeManifest(
                surface: .chatterboxRegular_fp16,
                modelID: "mlx-community/Chatterbox-TTS-fp16",
                displayName: "Chatterbox TTS",
                artifacts: [
                    ArtifactSpec(id: "model-config", role: .config, relativePath: "config.json"),
                    ArtifactSpec(id: "model-weights", role: .weights, relativePath: "model.safetensors"),
                    ArtifactSpec(id: "conditioning", role: .conditioning, relativePath: "conds.safetensors", required: false),
                    ArtifactSpec(id: "tokenizer", role: .tokenizer, relativePath: "tokenizer.json"),
                    ArtifactSpec(id: "tokenizer-config", role: .config, relativePath: "tokenizer_config.json"),
                ],
                remoteURL: URL(string: "https://huggingface.co/mlx-community/Chatterbox-TTS-fp16")
            ),
            makeManifest(
                surface: .chatterboxTurbo_fp16,
                modelID: "mlx-community/chatterbox-turbo-fp16",
                displayName: "Chatterbox Turbo",
                artifacts: [
                    ArtifactSpec(id: "model-config", role: .config, relativePath: "config.json"),
                    ArtifactSpec(id: "model-weights", role: .weights, relativePath: "model.safetensors"),
                    ArtifactSpec(id: "conditioning", role: .conditioning, relativePath: "conds.safetensors", required: false),
                    ArtifactSpec(id: "vocabulary", role: .vocabulary, relativePath: "vocab.json"),
                    ArtifactSpec(id: "merges", role: .vocabulary, relativePath: "merges.txt"),
                ],
                remoteURL: URL(string: "https://huggingface.co/mlx-community/chatterbox-turbo-fp16")
            ),
        ]
    }
}
