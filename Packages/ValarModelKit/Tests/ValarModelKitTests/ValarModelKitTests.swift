import XCTest
@testable import ValarModelKit

final class ValarModelKitTests: XCTestCase {
    func testModelIdentifierKeepsCanonicalValueAndInfersQwenHint() {
        let identifier = ModelIdentifier("mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16")
        XCTAssertEqual(identifier.rawValue, "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16")
        XCTAssertEqual(identifier.canonicalValue, "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16")
        XCTAssertEqual(identifier.inferredFamilyHint, .qwen3TTS)
        XCTAssertTrue(identifier.isCanonical)
    }

    func testModelIdentifierInfersSopranoHint() {
        let identifier = ModelIdentifier("mlx-community/Soprano-1.1-80M-bf16")
        XCTAssertEqual(identifier.inferredFamilyHint, .soprano)
    }

    func testModelIdentifierInfersVoxtralHint() {
        let identifier = ModelIdentifier("mistralai/Voxtral-4B-TTS-2603")
        XCTAssertEqual(identifier.inferredFamilyHint, .voxtralTTS)
    }

    func testCapabilityIDIsOpenAndSupportsStaticConstants() {
        let builtIn: CapabilityID = .speechSynthesis
        let custom = CapabilityID("voice.emotion_control")

        XCTAssertEqual(builtIn.rawValue, "speech.synthesis")
        XCTAssertEqual(custom.rawValue, "voice.emotion_control")
        XCTAssertNotEqual(builtIn, custom)
    }

    func testRuntimeSessionDefaultsToWarmingState() {
        let descriptor = ModelDescriptor(
            id: "mlx-community/Qwen3-ASR-0.6B-8bit",
            familyID: .qwen3ASR,
            displayName: "Qwen3 ASR 0.6B",
            domain: .stt,
            capabilities: [.speechRecognition, .translation],
            defaultSampleRate: 16_000
        )

        let configuration = ModelRuntimeConfiguration(backendKind: .mlx, residencyPolicy: .automatic)
        let session = ModelRuntimeSession(descriptor: descriptor, backendKind: .mlx, configuration: configuration)

        XCTAssertEqual(session.descriptor, descriptor)
        XCTAssertEqual(session.backendKind, .mlx)
        XCTAssertEqual(session.state, .warming)
    }

    func testDescriptorCanEncodeAndDecodeWithExplicitFamily() throws {
        let descriptor = ModelDescriptor(
            id: "mlx-community/Qwen3-ForcedAligner-0.6B-8bit",
            familyID: .qwen3ForcedAligner,
            displayName: "Qwen3 ForcedAligner",
            domain: .stt,
            capabilities: [.speechRecognition, .forcedAlignment],
            supportedBackends: [BackendRequirement(backendKind: .cpu)],
            defaultSampleRate: 16_000
        )

        let data = try JSONEncoder().encode(descriptor)
        let decoded = try JSONDecoder().decode(ModelDescriptor.self, from: data)
        XCTAssertEqual(decoded, descriptor)
        XCTAssertEqual(decoded.familyID, .qwen3ForcedAligner)
        XCTAssertEqual(decoded.supportedBackends.map(\.backendKind), [.cpu])
    }

    func testManifestRoundTripsAndPreservesBackendAndLicenseMetadata() throws {
        let manifest = ModelPackManifest(
            id: "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16",
            familyID: .qwen3TTS,
            displayName: "Qwen3-TTS CustomVoice",
            domain: .tts,
            capabilities: [.speechSynthesis, .voiceCloning, .audioConditioning],
            supportedBackends: [
                BackendRequirement(backendKind: .mlx, minimumMemoryBytes: 4_000_000_000, preferredQuantization: "bf16"),
            ],
            artifacts: [
                ArtifactSpec(id: "weights", role: .weights, relativePath: "weights/model.safetensors"),
            ],
            tokenizer: TokenizerSpec(kind: "huggingface", configPath: "tokenizers/tokenizer.json"),
            audio: AudioConstraint(defaultSampleRate: 24_000, supportsReferenceAudio: true),
            licenses: [
                LicenseSpec(name: "Qwen model license", sourceURL: URL(string: "https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16")),
            ]
        )

        let data = try JSONEncoder().encode(manifest)
        let decoded = try JSONDecoder().decode(ModelPackManifest.self, from: data)

        XCTAssertEqual(decoded.familyID, .qwen3TTS)
        XCTAssertEqual(decoded.supportedBackends.first?.backendKind, .mlx)
        XCTAssertEqual(decoded.licenses.first?.name, "Qwen model license")
        XCTAssertEqual(decoded.audio?.defaultSampleRate, 24_000)
    }

    func testQwenCatalogCoversAllCurrentQwenSurfaces() {
        let ids = Set(QwenCatalog.supportedEntries.map(\.id.rawValue))
        XCTAssertTrue(ids.contains("mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"))
        XCTAssertTrue(ids.contains("mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16"))
        XCTAssertTrue(ids.contains("mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16"))
        XCTAssertTrue(ids.contains("mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16"))
        XCTAssertTrue(ids.contains("mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit"))
        XCTAssertTrue(ids.contains("mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit"))
        XCTAssertTrue(ids.contains("mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16"))
        XCTAssertTrue(ids.contains("mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit"))
        XCTAssertTrue(ids.contains("mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit"))
        XCTAssertTrue(ids.contains("mlx-community/Qwen3-ASR-0.6B-8bit"))
        XCTAssertTrue(ids.contains("mlx-community/Qwen3-ForcedAligner-0.6B-8bit"))
    }

    func testQwenVoiceDesignManifestAdvertisesVoiceDesignCapability() {
        let entry = QwenCatalog.makeManifest(
            surface: .qwen3TTSVoiceDesign,
            modelID: "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16",
            displayName: "Qwen3-TTS VoiceDesign",
            artifacts: [ArtifactSpec(id: "weights", role: .weights, relativePath: "weights/model.safetensors")]
        )

        XCTAssertEqual(entry.manifest.familyID, .qwen3TTS)
        XCTAssertTrue(entry.manifest.capabilities.contains(.voiceDesign))
        XCTAssertEqual(entry.manifest.audio?.supportsReferenceAudio, true)
    }

    func testQwenBaseManifestAdvertisesClonePromptSupport() {
        let entry = QwenCatalog.makeManifest(
            surface: .qwen3TTSBase,
            modelID: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
            displayName: "Qwen3-TTS Base",
            artifacts: [ArtifactSpec(id: "weights", role: .weights, relativePath: "weights/model.safetensors")]
        )

        XCTAssertTrue(entry.manifest.capabilities.contains(.voiceCloning))
        XCTAssertTrue(entry.manifest.capabilities.contains(.audioConditioning))
        XCTAssertEqual(entry.manifest.audio?.supportsReferenceAudio, true)
        XCTAssertEqual(entry.manifest.promptSchema?.optionalFields, ["referenceAudio", "referenceText"])
    }

    func testQwenCatalogCanInferSurfaceFromModelIdentifier() {
        XCTAssertEqual(
            QwenCatalog.surface(for: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"),
            .qwen3TTSBase
        )
        XCTAssertEqual(
            QwenCatalog.surface(for: "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16"),
            .qwen3TTSCustomVoice
        )
        XCTAssertEqual(
            QwenCatalog.surface(for: "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16"),
            .qwen3TTSVoiceDesign
        )
        XCTAssertTrue(QwenCatalog.acceptsNamedSpeaker("mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16"))
        XCTAssertFalse(QwenCatalog.acceptsNamedSpeaker("mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"))
    }

    func testQwenBaseVoiceSupportPrefersClonePromptAndStableNarrator() {
        let descriptor = ModelDescriptor(
            id: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
            familyID: .qwen3TTS,
            displayName: "Qwen3-TTS Base",
            domain: .tts,
            capabilities: [.speechSynthesis, .voiceCloning, .audioConditioning, .longFormRendering]
        )

        XCTAssertEqual(
            descriptor.voiceSupport.features,
            [.referenceAudio, .clonePrompt, .stableNarrator]
        )
        XCTAssertTrue(descriptor.voiceSupport.supportsReferenceAudio)
    }

    func testQwenCustomVoiceSupportIsNamedSpeakerOnly() {
        let descriptor = ModelDescriptor(
            id: "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16",
            familyID: .qwen3TTS,
            displayName: "Qwen3-TTS CustomVoice",
            domain: .tts,
            capabilities: [.speechSynthesis, .voiceCloning, .audioConditioning]
        )

        XCTAssertEqual(descriptor.voiceSupport.features, [.namedSpeakers])
        XCTAssertFalse(descriptor.voiceSupport.supportsReferenceAudio)
    }

    func testVoxtralVoiceSupportStaysPresetOnly() {
        let descriptor = ModelDescriptor(
            id: VoxtralCatalog.mlx4BitModelIdentifier,
            familyID: .voxtralTTS,
            displayName: "Voxtral 4B TTS 2603 MLX (4-bit)",
            domain: .tts,
            capabilities: [.speechSynthesis, .presetVoices, .streaming]
        )

        XCTAssertEqual(descriptor.voiceSupport.features, [.presetVoices])
        XCTAssertFalse(descriptor.voiceSupport.supportsReferenceAudio)
    }

    func testSupportedModelCatalogIncludesSecondTTsFamily() {
        let sopranoEntry = SupportedModelCatalog.entry(for: "mlx-community/Soprano-1.1-80M-bf16")

        XCTAssertEqual(sopranoEntry?.manifest.familyID, .soprano)
        XCTAssertEqual(sopranoEntry?.manifest.domain, .tts)
        XCTAssertTrue(sopranoEntry?.manifest.capabilities.contains(.speechSynthesis) == true)
        XCTAssertEqual(sopranoEntry?.manifest.supportedBackends.map(\.backendKind), [.mlx])
    }

    func testSupportedModelCatalogIncludesVoxtralWithExplicitLicense() {
        let entry = SupportedModelCatalog.entry(for: VoxtralCatalog.modelIdentifier)

        XCTAssertEqual(entry?.manifest.familyID, .voxtralTTS)
        XCTAssertEqual(entry?.manifest.licenses.first?.name, VoxtralCatalog.licenseName)
        XCTAssertEqual(entry?.manifest.licenses.first?.spdxIdentifier, "CC-BY-NC-4.0")
        XCTAssertEqual(entry?.manifest.notes, "CC BY-NC 4.0 license. Non-commercial use only. Attribution required. Includes 20 preset voices.")
    }

    func testVoxtralCatalogIncludesRawPresetVoiceArtifacts() throws {
        let entry = try XCTUnwrap(SupportedModelCatalog.entry(for: VoxtralCatalog.modelIdentifier))
        let relativePaths = Set(entry.manifest.artifacts.map { $0.relativePath })

        for preset in VoxtralCatalog.presetVoices {
            XCTAssertTrue(relativePaths.contains("voice_embedding/\(preset.name).pt"))
        }
        XCTAssertTrue(relativePaths.contains("voice_embedding_safe/"))
        XCTAssertFalse(relativePaths.contains("voice_embedding/"))
    }

    func testVoxtralQuantizedCatalogEntriesArePresent() throws {
        let fourBit = try XCTUnwrap(SupportedModelCatalog.entry(for: VoxtralCatalog.mlx4BitModelIdentifier))
        let sixBit = try XCTUnwrap(SupportedModelCatalog.entry(for: VoxtralCatalog.mlx6BitModelIdentifier))

        XCTAssertEqual(fourBit.manifest.supportedBackends.first?.preferredQuantization, "4bit")
        XCTAssertEqual(sixBit.manifest.supportedBackends.first?.preferredQuantization, "6bit")
        XCTAssertTrue(fourBit.manifest.artifacts.contains { $0.relativePath == "voice_embedding/neutral_female.safetensors" })
        XCTAssertTrue(sixBit.manifest.artifacts.contains { $0.relativePath == "voice_embedding/neutral_female.safetensors" })
    }

    func testVoxtralCatalogResolvesAliasesToCanonicalPresetNames() {
        XCTAssertEqual(VoxtralCatalog.resolvePresetName("Emma"), "neutral_female")
        XCTAssertEqual(VoxtralCatalog.resolvePresetName("claire"), "fr_female")
        XCTAssertEqual(VoxtralCatalog.resolvePresetName("PEDRO"), "pt_male")
        XCTAssertEqual(VoxtralCatalog.resolvePresetName("neutral_female"), "neutral_female")
        XCTAssertNil(VoxtralCatalog.resolvePresetName("unknown-voice"))
    }

    func testVoxtralRandomPresetExcludesKnownBadVoices() {
        let excluded = Set(["neutral_male", "ar_male"])
        let draws = (0..<200).compactMap { _ in VoxtralCatalog.resolvePresetName("random") }

        XCTAssertFalse(draws.isEmpty)
        XCTAssertTrue(draws.allSatisfy { !excluded.contains($0) })
    }

    func testVibeVoiceCatalogResolvesCanonicalNamesCaseInsensitively() {
        XCTAssertEqual(VibeVoiceCatalog.resolvePresetName("en-Carter_man"), "en-Carter_man")
        XCTAssertEqual(VibeVoiceCatalog.resolvePresetName("EN-CARTER_MAN"), "en-Carter_man")
        XCTAssertEqual(VibeVoiceCatalog.resolvePresetName("en-emma_woman"), "en-Emma_woman")
        XCTAssertNil(VibeVoiceCatalog.resolvePresetName("unknown-vibevoice"))
    }

    func testVibeVoiceCatalogResolvesDisplayNames() {
        XCTAssertEqual(VibeVoiceCatalog.resolvePresetName("Emma"), "en-Emma_woman")
        XCTAssertEqual(VibeVoiceCatalog.resolvePresetName("German Female"), "de-Spk1_woman")
    }

    func testVibeVoiceRequestResolverCanonicalizesDisplayNames() throws {
        let resolved = try VibeVoiceRequestResolver.resolve(
            voice: "Emma",
            language: nil,
            randomPicker: { _ in nil }
        )

        XCTAssertEqual(resolved.effectiveVoice, "en-Emma_woman")
        XCTAssertEqual(resolved.effectiveLanguage, "en")
        XCTAssertEqual(resolved.selectionMode, .explicit)
    }

    func testVibeVoiceRequestResolverDefaultsMissingVoiceFromLanguage() throws {
        let resolved = try VibeVoiceRequestResolver.resolve(
            voice: nil,
            language: "es",
            randomPicker: { _ in nil }
        )

        XCTAssertEqual(resolved.effectiveVoice, "sp-Spk0_woman")
        XCTAssertEqual(resolved.effectiveLanguage, "es")
        XCTAssertEqual(resolved.selectionMode, .autoDefault)
    }

    func testVibeVoiceRequestResolverRejectsExplicitLanguageMismatch() throws {
        XCTAssertThrowsError(
            try VibeVoiceRequestResolver.resolve(
                voice: "en-Emma_woman",
                language: "es",
                randomPicker: { _ in nil }
            )
        ) { error in
            XCTAssertEqual(
                error as? VibeVoiceRequestResolutionError,
                .explicitVoiceLanguageMismatch(
                    voice: "en-Emma_woman",
                    requestedLanguage: "es",
                    presetLanguage: "en",
                    suggestedVoice: "sp-Spk0_woman"
                )
            )
        }
    }

    func testVibeVoiceRequestResolverRandomHonorsExplicitLanguagePool() throws {
        let resolved = try VibeVoiceRequestResolver.resolve(
            voice: "random",
            language: "ja",
            randomPicker: { presets in
                presets.first { $0.name == "jp-Spk1_woman" }
            }
        )

        XCTAssertEqual(resolved.effectiveVoice, "jp-Spk1_woman")
        XCTAssertEqual(resolved.effectiveLanguage, "ja")
        XCTAssertEqual(resolved.selectionMode, .random)
    }

    func testVibeVoiceCatalogManifestAdvertisesTokenizerArtifacts() throws {
        let entry = try XCTUnwrap(SupportedModelCatalog.entry(for: VibeVoiceCatalog.mlx4BitModelIdentifier))

        XCTAssertEqual(entry.manifest.tokenizer?.kind, "huggingface")
        XCTAssertEqual(entry.manifest.tokenizer?.configPath, "tokenizer.json")
        XCTAssertTrue(entry.manifest.capabilities.contains(.multilingual))
        XCTAssertEqual(
            Set(entry.manifest.supportedLanguages ?? []),
            Set(["en", "de", "fr", "it", "ja", "ko", "nl", "pl", "pt", "es"])
        )
        XCTAssertEqual(entry.manifest.supportTier, .preview)
        XCTAssertTrue(entry.manifest.releaseEligible)
        XCTAssertEqual(entry.manifest.qualityTierByLanguage["en"], .supported)
        XCTAssertEqual(entry.manifest.qualityTierByLanguage["hi"], .experimental)
        XCTAssertTrue(entry.manifest.artifacts.contains { $0.role == .tokenizer && $0.relativePath == "tokenizer.json" })
        XCTAssertTrue(entry.manifest.artifacts.contains { $0.role == .tokenizer && $0.relativePath == "tokenizer_config.json" })
        XCTAssertTrue(entry.manifest.artifacts.contains { $0.relativePath == "preprocessor_config.json" })
        XCTAssertEqual(VibeVoiceCatalog.tokenizerSourceModelIdentifier.rawValue, "Qwen/Qwen2.5-0.5B")
        XCTAssertEqual(
            entry.manifest.artifacts.first(where: { $0.relativePath == "special_tokens_map.json" })?.required,
            false
        )
        XCTAssertEqual(
            entry.manifest.artifacts.first(where: { $0.relativePath == "added_tokens.json" })?.required,
            false
        )
        XCTAssertTrue(entry.manifest.notes?.contains("Qwen/Qwen2.5-0.5B tokenizer") == true)
        XCTAssertTrue(entry.manifest.notes?.contains("Hindi stays exploratory") == true)
    }

    func testCuratedEntriesHideNonCommercialModelsByDefault() {
        let curated = SupportedModelCatalog.curatedEntries(includeNonCommercial: false)

        XCTAssertFalse(curated.contains(where: { $0.id == VoxtralCatalog.modelIdentifier }))
        XCTAssertFalse(curated.contains(where: { $0.id == VoxtralCatalog.mlx4BitModelIdentifier }))
        XCTAssertFalse(curated.contains(where: { $0.id == VoxtralCatalog.mlx6BitModelIdentifier }))
        XCTAssertTrue(curated.contains(where: { $0.id == "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16" }))
    }

    func testCuratedEntriesIncludeNonCommercialModelsWhenEnabled() {
        let curated = SupportedModelCatalog.curatedEntries(includeNonCommercial: true)

        XCTAssertTrue(curated.contains(where: { $0.id == VoxtralCatalog.modelIdentifier }))
        XCTAssertTrue(curated.contains(where: { $0.id == VoxtralCatalog.mlx4BitModelIdentifier }))
        XCTAssertTrue(curated.contains(where: { $0.id == VoxtralCatalog.mlx6BitModelIdentifier }))
    }

    func testTadaCatalogUsesDocumentedCheckpointLayout() throws {
        let oneB = try XCTUnwrap(SupportedModelCatalog.entry(for: TadaCatalog.tada1BModelIdentifier))
        let threeB = try XCTUnwrap(SupportedModelCatalog.entry(for: TadaCatalog.tada3BModelIdentifier))

        XCTAssertTrue(oneB.manifest.artifacts.contains { $0.relativePath == "model/config.json" })
        XCTAssertTrue(oneB.manifest.artifacts.contains { $0.relativePath == "model/weights.safetensors" })
        XCTAssertTrue(oneB.manifest.artifacts.contains { $0.relativePath == "encoder/weights.safetensors" })
        XCTAssertTrue(oneB.manifest.artifacts.contains { $0.relativePath == "decoder/weights.safetensors" })
        XCTAssertTrue(oneB.manifest.artifacts.contains { $0.relativePath == "aligner/weights.safetensors" })
        XCTAssertTrue(oneB.manifest.artifacts.contains { $0.relativePath == "tokenizer_config.json" })
        XCTAssertTrue(oneB.manifest.artifacts.contains { $0.relativePath == "special_tokens_map.json" })
        XCTAssertFalse(oneB.manifest.artifacts.contains { $0.relativePath == "encoder/config.json" })

        XCTAssertTrue(threeB.manifest.artifacts.contains { $0.relativePath == "model/config.json" })
        XCTAssertTrue(threeB.manifest.artifacts.contains { $0.relativePath == "model/weights.safetensors" })
        XCTAssertTrue(threeB.manifest.artifacts.contains { $0.relativePath == "encoder/weights.safetensors" })
        XCTAssertTrue(threeB.manifest.artifacts.contains { $0.relativePath == "decoder/weights.safetensors" })
        XCTAssertTrue(threeB.manifest.artifacts.contains { $0.relativePath == "aligner/weights.safetensors" })
        XCTAssertTrue(threeB.manifest.artifacts.contains { $0.relativePath == "tokenizer_config.json" })
        XCTAssertTrue(threeB.manifest.artifacts.contains { $0.relativePath == "special_tokens_map.json" })
        XCTAssertFalse(threeB.manifest.artifacts.contains { $0.relativePath == "model/model-00001-of-00002.safetensors" })
    }

    func testOrpheusCatalogAdvertisesPresetVoicesAndEmotionNotes() throws {
        let entry = try XCTUnwrap(SupportedModelCatalog.entry(for: "mlx-community/orpheus-3b-0.1-ft-bf16"))

        XCTAssertTrue(entry.manifest.capabilities.contains(.presetVoices))
        XCTAssertEqual(entry.manifest.presetVoices?.map(\.name), ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"])
        XCTAssertTrue(entry.manifest.notes?.contains("<laugh>") == true)
        XCTAssertTrue(entry.manifest.notes?.contains("<gasp>") == true)
    }

    func testChatterboxCatalogAdvertisesConditioningCapabilities() throws {
        let entry = try XCTUnwrap(SupportedModelCatalog.entry(for: "mlx-community/Chatterbox-TTS-fp16"))

        XCTAssertTrue(entry.manifest.capabilities.contains(.voiceCloning))
        XCTAssertTrue(entry.manifest.capabilities.contains(.audioConditioning))
    }

    func testIncludesNonCommercialModelsParsesTruthyFlags() {
        XCTAssertTrue(SupportedModelCatalog.includesNonCommercialModels(environment: ["VALARTTS_ENABLE_NONCOMMERCIAL_MODELS": "1"]))
        XCTAssertTrue(SupportedModelCatalog.includesNonCommercialModels(environment: ["VALARTTS_ENABLE_NONCOMMERCIAL_MODELS": " true "]))
        XCTAssertFalse(SupportedModelCatalog.includesNonCommercialModels(environment: [:]))
        XCTAssertFalse(SupportedModelCatalog.includesNonCommercialModels(environment: ["VALARTTS_ENABLE_NONCOMMERCIAL_MODELS": "0"]))
    }

    func testDescriptorBuiltFromManifestPreservesSupportedBackends() {
        let manifest = ModelPackManifest(
            id: "test/backend-order",
            familyID: .qwen3TTS,
            displayName: "Backend Order",
            domain: .tts,
            capabilities: [.speechSynthesis],
            supportedBackends: [
                BackendRequirement(backendKind: .metal),
                BackendRequirement(backendKind: .cpu),
            ],
            artifacts: []
        )

        let descriptor = ModelDescriptor(manifest: manifest)

        XCTAssertEqual(descriptor.supportedBackends.map(\.backendKind), [.metal, .cpu])
    }

    func testBackendSelectionPolicyPicksFirstCompatibleBackend() throws {
        let descriptor = ModelDescriptor(
            id: "test/backend-selection",
            familyID: .qwen3TTS,
            displayName: "Selection Test",
            domain: .tts,
            capabilities: [.speechSynthesis],
            supportedBackends: [
                BackendRequirement(backendKind: .mlx, minimumMemoryBytes: 8_000),
                BackendRequirement(backendKind: .cpu),
            ]
        )
        let runtime = BackendSelectionPolicy.Runtime(
            availableBackends: [.mlx, .cpu],
            availableMemoryBytes: 4_000,
            supportsLocalExecution: true,
            runtimeVersion: "14.0.0"
        )

        XCTAssertEqual(try BackendSelectionPolicy().backend(for: descriptor, runtime: runtime), .cpu)
    }

    func testBackendSelectionPolicyThrowsWhenNoCompatibleBackendExists() {
        let descriptor = ModelDescriptor(
            id: "test/no-compatible-backend",
            familyID: .qwen3TTS,
            displayName: "No Compatible Backend",
            domain: .tts,
            capabilities: [.speechSynthesis],
            supportedBackends: [BackendRequirement(backendKind: .cpu)]
        )
        let runtime = BackendSelectionPolicy.Runtime(
            availableBackends: [.mlx],
            availableMemoryBytes: 4_000,
            supportsLocalExecution: true,
            runtimeVersion: "14.0.0"
        )

        XCTAssertThrowsError(try BackendSelectionPolicy().backend(for: descriptor, runtime: runtime)) { error in
            XCTAssertEqual(
                error as? BackendSelectionPolicy.SelectionError,
                .noCompatibleBackend(descriptor.id)
            )
        }
    }

    func testForcedAlignmentRequestCarriesTranscriptAndAudio() {
        let request = ForcedAlignmentRequest(
            model: "mlx-community/Qwen3-ForcedAligner-0.6B-8bit",
            audioAssetName: "clip.wav",
            transcript: "Hello world",
            languageHint: "en"
        )

        XCTAssertEqual(request.model.rawValue, "mlx-community/Qwen3-ForcedAligner-0.6B-8bit")
        XCTAssertEqual(request.transcript, "Hello world")
        XCTAssertEqual(request.audioAssetName, "clip.wav")
        XCTAssertEqual(request.languageHint, "en")
    }
}
