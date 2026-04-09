import Foundation

public enum SopranoSurface: String, CaseIterable, Codable, Sendable {
    case soprano11_80MBF16

    public var familyID: ModelFamilyID { .soprano }
    public var domain: ModelDomain { .tts }

    public var capabilities: Set<CapabilityID> {
        [.speechSynthesis, .tokenization, .longFormRendering]
    }

    public var defaultSampleRate: Double { 24_000 }

    public var promptSchema: PromptSchema? {
        PromptSchema(kind: "soprano.tts.base", requiredFields: ["text"])
    }
}

public enum SopranoCatalog {
    public static func makeManifest(
        surface: SopranoSurface,
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
            licenses: [
                LicenseSpec(name: "Model license", sourceURL: remoteURL, requiresAttribution: true),
            ],
            notes: "Fastest first-run local TTS option in Valar. Best for simple single-speaker synthesis and quick local setup."
        )

        return SupportedModelCatalogEntry(
            manifest: manifest,
            remoteURL: remoteURL,
            requiresManualDownload: remoteURL != nil,
            distributionTier: .bundledFirstRun,
            tags: ["soprano", surface.familyID.rawValue]
        )
    }

    public static var supportedEntries: [SupportedModelCatalogEntry] {
        [
            makeManifest(
                surface: .soprano11_80MBF16,
                modelID: "mlx-community/Soprano-1.1-80M-bf16",
                displayName: "Soprano 1.1 80M",
                artifacts: [
                    ArtifactSpec(
                        id: "model-config",
                        role: .config,
                        relativePath: "config.json",
                        sha256: "6cc152a1b30adc090a0096d9d1b83dc2bcc01a0a8b2a3daf8c3e02fc9e0a703c",
                        sizeBytes: 1_249
                    ),
                    ArtifactSpec(
                        id: "model-weights",
                        role: .weights,
                        relativePath: "model.safetensors",
                        sha256: "1680a01ef354f883699c86cf434fc67b087a8c58cec288c9084d5828a3074575",
                        sizeBytes: 280_868_459
                    ),
                    ArtifactSpec(
                        id: "tokenizer",
                        role: .tokenizer,
                        relativePath: "tokenizer.json",
                        sha256: "eb27ca6c9b55d2f880659a9619d6b8f9b9e6d8553d3ea424f5427788688d7286",
                        sizeBytes: 1_630_675
                    ),
                    ArtifactSpec(
                        id: "tokenizer-config",
                        role: .config,
                        relativePath: "tokenizer_config.json",
                        sha256: "06e83fdb0bdf94eef83ebb08f30620c8ab640344babb4d405ff27befd1a4c270",
                        sizeBytes: 1_366_802
                    ),
                ],
                remoteURL: URL(string: "https://huggingface.co/mlx-community/Soprano-1.1-80M-bf16")
            ),
        ]
    }
}
