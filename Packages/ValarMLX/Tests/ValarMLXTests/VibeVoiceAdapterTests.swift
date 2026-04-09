import Testing
import ValarModelKit
@testable import ValarMLX

@Suite("VibeVoice Adapter Validation")
struct VibeVoiceAdapterTests {
    @Test("Catalog VibeVoice manifest passes adapter validation")
    func catalogManifestPassesValidation() throws {
        let entry = try #require(SupportedModelCatalog.entry(for: VibeVoiceCatalog.mlx4BitModelIdentifier))
        let adapter = VibeVoiceTTSAdapter()

        try adapter.validate(manifest: entry.manifest)
    }

    @Test("Adapter rejects manifests missing tokenizer artifact")
    func adapterRejectsMissingTokenizerArtifact() throws {
        let entry = try #require(SupportedModelCatalog.entry(for: VibeVoiceCatalog.mlx4BitModelIdentifier))
        let manifest = ModelPackManifest(
            id: entry.manifest.id,
            familyID: entry.manifest.familyID,
            displayName: entry.manifest.displayName,
            domain: entry.manifest.domain,
            capabilities: entry.manifest.capabilities,
            supportedBackends: entry.manifest.supportedBackends,
            artifacts: entry.manifest.artifacts.filter { $0.relativePath != "tokenizer.json" },
            tokenizer: entry.manifest.tokenizer,
            audio: entry.manifest.audio,
            promptSchema: entry.manifest.promptSchema,
            supportedLanguages: entry.manifest.supportedLanguages,
            presetVoices: entry.manifest.presetVoices,
            licenses: entry.manifest.licenses,
            notes: entry.manifest.notes
        )

        let adapter = VibeVoiceTTSAdapter()
        #expect(throws: AdapterError.self) {
            try adapter.validate(manifest: manifest)
        }
    }

    @Test("Adapter rejects manifests missing preprocessor config")
    func adapterRejectsMissingPreprocessorConfig() throws {
        let entry = try #require(SupportedModelCatalog.entry(for: VibeVoiceCatalog.mlx4BitModelIdentifier))
        let manifest = ModelPackManifest(
            id: entry.manifest.id,
            familyID: entry.manifest.familyID,
            displayName: entry.manifest.displayName,
            domain: entry.manifest.domain,
            capabilities: entry.manifest.capabilities,
            supportedBackends: entry.manifest.supportedBackends,
            artifacts: entry.manifest.artifacts.filter { $0.relativePath != "preprocessor_config.json" },
            tokenizer: entry.manifest.tokenizer,
            audio: entry.manifest.audio,
            promptSchema: entry.manifest.promptSchema,
            supportedLanguages: entry.manifest.supportedLanguages,
            presetVoices: entry.manifest.presetVoices,
            licenses: entry.manifest.licenses,
            notes: entry.manifest.notes
        )

        let adapter = VibeVoiceTTSAdapter()
        #expect(throws: AdapterError.self) {
            try adapter.validate(manifest: manifest)
        }
    }
}
