import Foundation

public enum LFMAudioCatalog {
    public static func makeManifest(
        modelID: ModelIdentifier,
        displayName: String,
        artifacts: [ArtifactSpec],
        remoteURL: URL? = nil
    ) -> SupportedModelCatalogEntry {
        let manifest = ModelPackManifest(
            id: modelID,
            familyID: .lfmAudio,
            displayName: displayName,
            domain: .sts,
            capabilities: [.speechToSpeech, .speechSynthesis],
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
                defaultSampleRate: 24_000,
                minimumSampleRate: 16_000,
                maximumSampleRate: 48_000
            ),
            promptSchema: nil,
            supportTier: .experimental,
            releaseEligible: false,
            licenses: [
                LicenseSpec(name: "Model license", sourceURL: remoteURL, requiresAttribution: true),
            ],
            notes: "Experimental STS family — feature-flagged, not beta-blocking"
        )

        return SupportedModelCatalogEntry(
            manifest: manifest,
            remoteURL: remoteURL,
            requiresManualDownload: true,
            tags: ["experimental", "sts", ModelFamilyID.lfmAudio.rawValue]
        )
    }

    public static var supportedEntries: [SupportedModelCatalogEntry] {
        [
            makeManifest(
                modelID: "mlx-community/LFM2.5-Audio-1.5B-fp16",
                displayName: "LFM 2.5 Audio 1.5B (Experimental)",
                artifacts: [
                    ArtifactSpec(id: "model-config", role: .config, relativePath: "config.json"),
                    ArtifactSpec(id: "model-weights", role: .weights, relativePath: "model.safetensors"),
                    ArtifactSpec(id: "tokenizer", role: .tokenizer, relativePath: "tokenizer.json"),
                    ArtifactSpec(id: "tokenizer-config", role: .config, relativePath: "tokenizer_config.json"),
                ],
                remoteURL: URL(string: "https://huggingface.co/mlx-community/LFM2.5-Audio-1.5B-fp16")
            ),
        ]
    }
}
