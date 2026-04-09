import Foundation

public enum TadaSurface: String, CaseIterable, Codable, Sendable {
    case tada1B
    case tada3B

    public var familyID: ModelFamilyID { .tadaTTS }
    public var domain: ModelDomain { .tts }

    public var capabilities: Set<CapabilityID> {
        switch self {
        case .tada1B:
            return [.speechSynthesis, .voiceCloning, .audioConditioning]
        case .tada3B:
            return [.speechSynthesis, .voiceCloning, .audioConditioning, .multilingual]
        }
    }

    public var defaultSampleRate: Double { 24_000 }

    public var supportedLanguages: [String] {
        switch self {
        case .tada1B:
            return ["EN"]
        case .tada3B:
            return ["EN", "FR", "DE", "ES", "PT", "IT", "PL", "JA", "ZH", "AR"]
        }
    }

    public var promptSchema: PromptSchema? {
        PromptSchema(
            kind: "tada.tts.clone",
            requiredFields: ["text", "referenceAudio"],
            optionalFields: ["referenceTranscript", "language"]
        )
    }
}

public enum TadaCatalog {
    public static let tada1BModelIdentifier: ModelIdentifier = "HumeAI/mlx-tada-1b"
    public static let tada3BModelIdentifier: ModelIdentifier = "HumeAI/mlx-tada-3b"

    public static func makeManifest(
        surface: TadaSurface,
        modelID: ModelIdentifier,
        displayName: String,
        remoteURL: URL? = nil
    ) -> SupportedModelCatalogEntry {
        // Current public HumeAI/mlx-tada-* snapshots use a compact layout:
        // only model/config.json is shipped as config, and all module weights
        // are single safetensors files under model/, encoder/, decoder/, aligner/.
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
                ArtifactSpec(id: "model-config", role: .config, relativePath: "model/config.json"),
                ArtifactSpec(id: "model-weights", role: .weights, relativePath: "model/weights.safetensors"),
                ArtifactSpec(id: "encoder-weights", role: .weights, relativePath: "encoder/weights.safetensors"),
                ArtifactSpec(id: "decoder-weights", role: .weights, relativePath: "decoder/weights.safetensors"),
                ArtifactSpec(id: "aligner-weights", role: .weights, relativePath: "aligner/weights.safetensors"),
                // TADA snapshots ship the tokenizer sidecars at repo root.
                ArtifactSpec(id: "tokenizer", role: .tokenizer, relativePath: "tokenizer.json"),
                ArtifactSpec(id: "tokenizer-config", role: .tokenizer, relativePath: "tokenizer_config.json"),
                ArtifactSpec(id: "special-tokens-map", role: .tokenizer, relativePath: "special_tokens_map.json"),
            ],
            tokenizer: TokenizerSpec(kind: "huggingface", configPath: "tokenizer.json"),
            audio: AudioConstraint(
                defaultSampleRate: surface.defaultSampleRate,
                minimumSampleRate: 16_000,
                maximumSampleRate: surface.defaultSampleRate,
                supportsReferenceAudio: true
            ),
            promptSchema: surface.promptSchema,
            supportedLanguages: surface.supportedLanguages,
            licenses: [
                LicenseSpec(
                    name: "Llama 3.2 Community License",
                    relativePath: "LICENSE",
                    sourceURL: remoteURL,
                    requiresAttribution: true,
                    isNonCommercial: false
                ),
            ],
            notes: "Voice cloning family. Create saved voices with 'voices create --reference-audio' or use inline reference audio at generation time."
        )

        return SupportedModelCatalogEntry(
            manifest: manifest,
            remoteURL: remoteURL,
            requiresManualDownload: remoteURL != nil,
            isRecommended: false,
            tags: ["tada", "cloning", "experimental", surface.familyID.rawValue]
        )
    }

    public static var supportedEntries: [SupportedModelCatalogEntry] {
        [
            makeManifest(
                surface: .tada1B,
                modelID: tada1BModelIdentifier,
                displayName: "TADA 1B",
                remoteURL: URL(string: "https://huggingface.co/HumeAI/mlx-tada-1b")
            ),
            makeManifest(
                surface: .tada3B,
                modelID: tada3BModelIdentifier,
                displayName: "TADA 3B",
                remoteURL: URL(string: "https://huggingface.co/HumeAI/mlx-tada-3b")
            ),
        ]
    }
}
