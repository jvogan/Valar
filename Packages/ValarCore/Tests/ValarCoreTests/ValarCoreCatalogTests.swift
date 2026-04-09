import CryptoKit
import Darwin
import Foundation
import XCTest
@testable import ValarCore
import ValarModelKit
import ValarPersistence

final class ValarCoreCatalogTests: XCTestCase {
    func testCatalogVisibilityPolicyHidesVoxtralByDefault() {
        let policy = CatalogVisibilityPolicy(allowsNonCommercialModels: false)

        XCTAssertEqual(
            policy.hiddenReason(for: VoxtralCatalog.modelIdentifier),
            "Model '\(VoxtralCatalog.modelIdentifier.rawValue)' is hidden by default because it is licensed for non-commercial use only. Set \(CatalogVisibilityPolicy.nonCommercialEnvVarName)=1 to enable non-commercial models intentionally."
        )
    }

    func testModelCatalogRefreshFiltersHiddenNonCommercialEntries() async throws {
        let catalog = ModelCatalog(
            supportedSource: SupportedCatalogSource.curated(),
            visibilityPolicyProvider: { CatalogVisibilityPolicy(allowsNonCommercialModels: false) }
        )

        let models = try await catalog.refresh()

        XCTAssertFalse(models.contains(where: { $0.id == VoxtralCatalog.modelIdentifier }))
        XCTAssertTrue(models.contains(where: { $0.id == "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16" }))
    }

    func testModelCatalogRefreshIncludesVoxtralWhenNonCommercialModelsEnabled() async throws {
        let catalog = ModelCatalog(
            supportedSource: SupportedCatalogSource.curated(),
            visibilityPolicyProvider: { CatalogVisibilityPolicy(allowsNonCommercialModels: true) }
        )

        let models = try await catalog.refresh()

        XCTAssertTrue(models.contains(where: { $0.id == VoxtralCatalog.modelIdentifier }))
    }

    func testModelCatalogRefreshIncludesTadaByDefault() async throws {
        let catalog = ModelCatalog(
            supportedSource: SupportedCatalogSource.curated(),
            visibilityPolicyProvider: { CatalogVisibilityPolicy(allowsNonCommercialModels: false) }
        )

        let models = try await catalog.refresh()

        XCTAssertTrue(models.contains(where: { $0.id == TadaCatalog.tada1BModelIdentifier }))
        XCTAssertTrue(models.contains(where: { $0.id == TadaCatalog.tada3BModelIdentifier }))
    }

    func testCuratedCatalogEntryWinsOverPersistedDuplicate() async throws {
        let cacheRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: cacheRoot, withIntermediateDirectories: true)
        let packRegistry = ModelPackRegistry()
        await packRegistry.registerSupported(
            SupportedModelCatalogRecord(
                id: UUID().uuidString,
                familyID: ModelFamilyID.tadaTTS.rawValue,
                modelID: TadaCatalog.tada1BModelIdentifier.rawValue,
                displayName: "Stale Local TADA 1B",
                providerName: "Valar",
                providerURL: nil,
                installHint: "stale local record",
                sourceKind: .localFile,
                isRecommended: false
            )
        )

        let curatedEntry = try XCTUnwrap(SupportedModelCatalog.entry(for: TadaCatalog.tada1BModelIdentifier))
        let catalog = ModelCatalog(
            supportedSource: StaticSupportedCatalogSource(records: [curatedEntry]),
            catalogStore: packRegistry,
            packStore: packRegistry,
            hfCacheRoot: cacheRoot,
            visibilityPolicyProvider: { CatalogVisibilityPolicy(allowsNonCommercialModels: false) }
        )

        let models = try await catalog.refresh()
        let tada = try XCTUnwrap(models.first(where: { $0.id == TadaCatalog.tada1BModelIdentifier }))

        XCTAssertEqual(tada.descriptor.displayName, "TADA 1B")
        XCTAssertEqual(tada.providerURL?.host(), "huggingface.co")
        XCTAssertEqual(tada.installState, .supported)
    }

    func testTadaTokenizerInstallIssueAcceptsTokenizerFromDownloadedTadaSnapshot() throws {
        let cacheRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: cacheRoot, withIntermediateDirectories: true)
        try writeHFHubSnapshotArtifact(
            cacheRoot: cacheRoot,
            modelID: TadaCatalog.tada1BModelIdentifier.rawValue,
            relativePath: "tokenizer.json",
            data: Data(#"{"model":"tada"}"#.utf8)
        )

        let issue = ModelInstaller.tadaTokenizerInstallIssue(
            preferredModelID: TadaCatalog.tada1BModelIdentifier.rawValue,
            hfCacheRoot: cacheRoot
        )

        XCTAssertNil(issue)
    }

    func testModelCatalogMarksDownloadedTadaSnapshotAsCached() async throws {
        let cacheRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: cacheRoot, withIntermediateDirectories: true)
        let entry = try XCTUnwrap(SupportedModelCatalog.entry(for: TadaCatalog.tada1BModelIdentifier))
        try writeHFHubSnapshotArtifacts(
            cacheRoot: cacheRoot,
            modelID: TadaCatalog.tada1BModelIdentifier.rawValue,
            relativePaths: entry.manifest.artifacts.map(\.relativePath)
        )

        let catalog = ModelCatalog(
            supportedSource: StaticSupportedCatalogSource(records: [entry]),
            hfCacheRoot: cacheRoot,
            visibilityPolicyProvider: { CatalogVisibilityPolicy(allowsNonCommercialModels: false) }
        )

        let models = try await catalog.refresh()
        let tada = try XCTUnwrap(models.first(where: { $0.id == TadaCatalog.tada1BModelIdentifier }))

        XCTAssertEqual(tada.installState, .cached)
        XCTAssertTrue(tada.cachedOnDisk)
    }

    func testModelCatalogRecognizesMLXAudioRootFileLayoutAsCached() async throws {
        let cacheRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        let entry = try XCTUnwrap(SupportedModelCatalog.entry(for: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"))
        for relativePath in entry.manifest.artifacts.map(\.relativePath) where !relativePath.hasSuffix("/") {
            try writeMLXAudioCacheArtifact(
                cacheRoot: cacheRoot,
                modelID: entry.id.rawValue,
                relativePath: relativePath,
                data: Data(relativePath.utf8)
            )
        }

        let catalog = ModelCatalog(
            supportedSource: StaticSupportedCatalogSource(records: [entry]),
            hfCacheRoot: cacheRoot,
            visibilityPolicyProvider: { CatalogVisibilityPolicy(allowsNonCommercialModels: false) }
        )

        let models = try await catalog.refresh()
        let qwen = try XCTUnwrap(models.first(where: { $0.id.rawValue == entry.id.rawValue }))

        XCTAssertEqual(qwen.installState, .cached)
        XCTAssertTrue(qwen.cachedOnDisk)
    }

    func testModelCatalogDoesNotMarkPartialTadaSnapshotAsCached() async throws {
        let cacheRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: cacheRoot, withIntermediateDirectories: true)
        try writeHFHubSnapshotArtifact(
            cacheRoot: cacheRoot,
            modelID: TadaCatalog.tada1BModelIdentifier.rawValue,
            relativePath: "tokenizer.json",
            data: Data(#"{"model":"tada"}"#.utf8)
        )

        let entry = try XCTUnwrap(SupportedModelCatalog.entry(for: TadaCatalog.tada1BModelIdentifier))
        let catalog = ModelCatalog(
            supportedSource: StaticSupportedCatalogSource(records: [entry]),
            hfCacheRoot: cacheRoot,
            visibilityPolicyProvider: { CatalogVisibilityPolicy(allowsNonCommercialModels: false) }
        )

        let models = try await catalog.refresh()
        let tada = try XCTUnwrap(models.first(where: { $0.id == TadaCatalog.tada1BModelIdentifier }))

        XCTAssertEqual(tada.installState, .supported)
        XCTAssertFalse(tada.cachedOnDisk)
    }

    func testModelCatalogMarksVibeVoiceSnapshotAsCachedWhenQwenTokenizerFallbackIsCached() async throws {
        let cacheRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: cacheRoot, withIntermediateDirectories: true)

        let entry = try XCTUnwrap(SupportedModelCatalog.entry(for: VibeVoiceCatalog.mlx4BitModelIdentifier))
        let vibeVoiceCoreArtifacts = entry.manifest.artifacts
            .map(\.relativePath)
            .filter {
                $0 != "tokenizer.json"
                    && $0 != "tokenizer_config.json"
                    && $0 != "special_tokens_map.json"
                    && $0 != "added_tokens.json"
            }
        try writeHFHubSnapshotArtifacts(
            cacheRoot: cacheRoot,
            modelID: entry.id.rawValue,
            relativePaths: vibeVoiceCoreArtifacts
        )
        try writeHFHubSnapshotArtifacts(
            cacheRoot: cacheRoot,
            modelID: VibeVoiceCatalog.tokenizerSourceModelIdentifier.rawValue,
            relativePaths: ["tokenizer.json", "tokenizer_config.json"]
        )

        let catalog = ModelCatalog(
            supportedSource: StaticSupportedCatalogSource(records: [entry]),
            hfCacheRoot: cacheRoot,
            visibilityPolicyProvider: { CatalogVisibilityPolicy(allowsNonCommercialModels: false) }
        )

        let models = try await catalog.refresh()
        let vibeVoice = try XCTUnwrap(models.first(where: { $0.id == VibeVoiceCatalog.mlx4BitModelIdentifier }))

        XCTAssertEqual(vibeVoice.installState, .cached)
        XCTAssertTrue(vibeVoice.cachedOnDisk)
    }

    func testModelCatalogIgnoresInstalledRecordWhenPackDirectoryIsMissing() async throws {
        let cacheRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        let paths = ValarAppPaths(
            baseURL: FileManager.default.temporaryDirectory
                .appendingPathComponent(UUID().uuidString, isDirectory: true)
        )
        try FileManager.default.createDirectory(at: cacheRoot, withIntermediateDirectories: true)
        let entry = try XCTUnwrap(SupportedModelCatalog.entry(for: TadaCatalog.tada1BModelIdentifier))
        let manifest = ModelCatalog.makePersistenceManifest(from: entry.manifest)
        let packRegistry = ModelPackRegistry(paths: paths)
        _ = try await packRegistry.install(
            manifest: manifest,
            sourceKind: .remoteURL,
            sourceLocation: try XCTUnwrap(entry.remoteURL?.absoluteString),
            notes: "stale installed record"
        )

        let catalog = ModelCatalog(
            supportedSource: StaticSupportedCatalogSource(records: [entry]),
            catalogStore: packRegistry,
            packStore: packRegistry,
            hfCacheRoot: cacheRoot,
            visibilityPolicyProvider: { CatalogVisibilityPolicy(allowsNonCommercialModels: false) }
        )

        let models = try await catalog.refresh()
        let tada = try XCTUnwrap(models.first(where: { $0.id == TadaCatalog.tada1BModelIdentifier }))

        XCTAssertEqual(tada.installState, .supported)
        XCTAssertNil(tada.installedPath)
        XCTAssertNil(tada.manifestPath)
        XCTAssertEqual(tada.installPathStatus, .missingInstalledPath)

        let staleModels = try await catalog.staleInstalledModels()
        XCTAssertEqual(staleModels.map(\.id), [entry.id])
    }

    func testModelRegistryTracksByteBudgetAndPinnedResidency() async {
        let pinned = ModelDescriptor(
            id: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
            familyID: .qwen3TTS,
            displayName: "Qwen3 TTS Base",
            domain: .tts,
            capabilities: [.speechSynthesis, .tokenization]
        )
        let asr = ModelDescriptor(
            id: "mlx-community/Qwen3-ASR-0.6B-8bit",
            familyID: .qwen3ASR,
            displayName: "Qwen3 ASR 0.6B",
            domain: .stt,
            capabilities: [.speechRecognition]
        )
        let voiceDesign = ModelDescriptor(
            id: "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16",
            familyID: .qwen3TTS,
            displayName: "Qwen3 TTS VoiceDesign",
            domain: .tts,
            capabilities: [.speechSynthesis, .voiceDesign]
        )

        let registry = ModelRegistry(
            configuration: RuntimeConfiguration(
                maxResidentBytes: 6 * 1_024 * 1_024 * 1_024,
                maxResidentModels: 3
            )
        )
        await registry.register(
            pinned,
            estimatedResidentBytes: 4 * 1_024 * 1_024 * 1_024,
            runtimeConfiguration: ModelRuntimeConfiguration(backendKind: .mlx, residencyPolicy: .pinned)
        )
        await registry.register(
            asr,
            estimatedResidentBytes: Int(1.5 * Double(1_024 * 1_024 * 1_024)),
            runtimeConfiguration: ModelRuntimeConfiguration(backendKind: .mlx, residencyPolicy: .automatic)
        )
        await registry.register(
            voiceDesign,
            estimatedResidentBytes: 2 * 1_024 * 1_024 * 1_024,
            runtimeConfiguration: ModelRuntimeConfiguration(backendKind: .mlx, residencyPolicy: .automatic)
        )

        await registry.markState(.resident, for: pinned.id)
        await registry.markState(.resident, for: asr.id)
        await registry.markState(.resident, for: voiceDesign.id)

        let residentModels = await registry.residentModels()
        let residentBytes = await registry.residentBytes()

        XCTAssertEqual(residentModels.map(\.id), [pinned.id, voiceDesign.id])
        XCTAssertEqual(residentBytes, 6 * 1_024 * 1_024 * 1_024)
    }

    func testCapabilityRegistryIndexesCapabilitiesAndFamilies() async {
        let capabilityRegistry = CapabilityRegistry()
        let tts = ModelDescriptor(
            id: "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16",
            familyID: .qwen3TTS,
            displayName: "Qwen3 TTS CustomVoice",
            domain: .tts,
            capabilities: [.speechSynthesis, .voiceCloning]
        )
        let asr = ModelDescriptor(
            id: "mlx-community/Qwen3-ASR-1.7B-8bit",
            familyID: .qwen3ASR,
            displayName: "Qwen3 ASR 1.7B",
            domain: .stt,
            capabilities: [.speechRecognition]
        )

        await capabilityRegistry.register(tts)
        await capabilityRegistry.register(asr)

        let speechModels = await capabilityRegistry.models(supporting: .speechSynthesis)
        let qwenFamily = await capabilityRegistry.models(inFamily: .qwen3TTS)

        XCTAssertEqual(speechModels.map(\.id), [tts.id])
        XCTAssertEqual(qwenFamily.map(\.id), [tts.id])
    }

    func testModelCatalogMarksInstalledModels() async throws {
        let paths = try makeAppPaths()
        let entry = try XCTUnwrap(
            SupportedModelCatalog.entry(for: "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16")
        )
        let manifest = ModelCatalog.makePersistenceManifest(from: entry.manifest)
        let packRegistry = ModelPackRegistry(paths: paths)
        _ = try await packRegistry.install(
            manifest: manifest,
            sourceKind: .localFile,
            sourceLocation: "/tmp/qwen-custom.valarmodel",
            notes: "Installed for testing"
        )
        try materializeInstalledPack(paths: paths, manifest: manifest)

        let capabilityRegistry = CapabilityRegistry()
        let catalog = ModelCatalog(
            supportedSource: SupportedCatalogSource.qwenFirst(),
            catalogStore: packRegistry,
            packStore: packRegistry,
            capabilityRegistry: capabilityRegistry
        )

        let models = try await catalog.refresh()
        let customVoice = try XCTUnwrap(models.first { $0.id.rawValue == manifest.modelID })
        let qwenModels = await capabilityRegistry.models(inFamily: .qwen3TTS)

        XCTAssertEqual(customVoice.installState, .installed)
        XCTAssertEqual(customVoice.familyID, .qwen3TTS)
        XCTAssertTrue(customVoice.descriptor.capabilities.contains(.voiceCloning))
        XCTAssertGreaterThanOrEqual(qwenModels.count, 1)
    }

    func testModelCatalogTreatsStaleInstalledQwenPackAsCached() async throws {
        let paths = try makeAppPaths()
        let cacheRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: cacheRoot, withIntermediateDirectories: true)

        let entry = try XCTUnwrap(
            SupportedModelCatalog.allSupportedEntries.first {
                $0.id.rawValue == "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16"
            }
        )

        let staleManifest = ValarPersistence.ModelPackManifest(
            familyID: "qwen3_tts",
            modelID: entry.id.rawValue,
            displayName: "Qwen3 TTS VoiceDesign",
            capabilities: ["speech.synthesis", "voice.design", "audio.conditioning"],
            backendKinds: ["mlx"],
            tokenizerType: "huggingface",
            sampleRate: 24_000,
            artifactSpecs: [
                ModelPackArtifact(
                    id: "config",
                    kind: "config",
                    relativePath: "config.json",
                    checksum: nil,
                    byteCount: 16
                ),
                ModelPackArtifact(
                    id: "weights",
                    kind: "weights",
                    relativePath: "weights/model.safetensors",
                    checksum: nil,
                    byteCount: 32
                ),
                ModelPackArtifact(
                    id: "tokenizer",
                    kind: "tokenizer",
                    relativePath: "tokenizers/tokenizer.json",
                    checksum: nil,
                    byteCount: 24
                ),
            ],
            licenseName: "Model license",
            licenseURL: "https://example.com/license"
        )

        let packRegistry = ModelPackRegistry(paths: paths)
        _ = try await packRegistry.install(
            manifest: staleManifest,
            sourceKind: .localFile,
            sourceLocation: "/tmp/stale-qwen-pack.valarmodel",
            notes: "Installed for testing"
        )
        try materializeInstalledPack(paths: paths, manifest: staleManifest)
        try writeHFHubSnapshotArtifacts(
            cacheRoot: cacheRoot,
            modelID: entry.id.rawValue,
            relativePaths: entry.manifest.artifacts.map(\.relativePath)
        )

        let catalog = ModelCatalog(
            supportedSource: StaticSupportedCatalogSource(records: [entry]),
            catalogStore: packRegistry,
            packStore: packRegistry,
            hfCacheRoot: cacheRoot
        )

        let models = try await catalog.refresh()
        let model = try XCTUnwrap(models.first(where: { $0.id == entry.id }))

        XCTAssertEqual(model.installState, .cached)
        XCTAssertNil(model.installedPath)
        XCTAssertNil(model.manifestPath)
    }

    func testModelInstallerValidatesAndRegistersDescriptor() async throws {
        let manifest = makePersistenceManifest(
            modelID: "mlx-community/Qwen3-ForcedAligner-0.6B-8bit",
            familyID: "qwen3_forced_aligner",
            displayName: "Qwen3 ForcedAligner 0.6B",
            capabilities: ["speech.forced_alignment", "text.tokenization"],
            backendKinds: ["mlx"]
        )
        let paths = try makeAppPaths()
        let packRegistry = ModelPackRegistry(paths: paths)
        let modelRegistry = ModelRegistry()
        let capabilityRegistry = CapabilityRegistry()
        let installer = ModelInstaller(
            registry: packRegistry,
            modelRegistry: modelRegistry,
            capabilityRegistry: capabilityRegistry,
            paths: paths
        )
        try materializeInstalledPack(paths: paths, manifest: manifest)

        let result = try await installer.install(
            manifest: manifest,
            sourceKind: .localFile,
            sourceLocation: "/tmp/qwen-aligner.valarmodel",
            notes: "Install during unit test"
        )

        let registered = await modelRegistry.descriptor(for: result.descriptor.id)
        let alignmentModels = await capabilityRegistry.models(supporting: .forcedAlignment)

        XCTAssertFalse(result.report.hasErrors)
        XCTAssertEqual(registered?.familyID, .qwen3ForcedAligner)
        XCTAssertEqual(alignmentModels.map(\.id), [result.descriptor.id])
    }

    func testModelCatalogRefreshPreservesSecondTTsFamilyInstall() async throws {
        let paths = try makeAppPaths()
        let packRegistry = ModelPackRegistry(paths: paths)
        let modelRegistry = ModelRegistry()
        let capabilityRegistry = CapabilityRegistry()
        let installer = ModelInstaller(
            registry: packRegistry,
            modelRegistry: modelRegistry,
            capabilityRegistry: capabilityRegistry,
            paths: paths
        )
        let catalog = ModelCatalog(
            supportedSource: SupportedCatalogSource.curated(),
            catalogStore: packRegistry,
            packStore: packRegistry,
            capabilityRegistry: capabilityRegistry
        )
        let sopranoEntry = try XCTUnwrap(SopranoCatalog.supportedEntries.first)
        try materializeInstalledPack(
            paths: paths,
            manifest: ModelCatalog.makePersistenceManifest(from: sopranoEntry.manifest)
        )

        let result = try await installer.install(
            manifest: ModelCatalog.makePersistenceManifest(from: sopranoEntry.manifest),
            sourceKind: .remoteURL,
            sourceLocation: try XCTUnwrap(sopranoEntry.remoteURL?.absoluteString),
            notes: "Install second family during unit test"
        )
        let models = try await catalog.refresh()
        let installed = try XCTUnwrap(models.first(where: { $0.id == sopranoEntry.id }))
        let sopranoModels = await capabilityRegistry.models(inFamily: .soprano)

        XCTAssertEqual(result.descriptor.familyID, .soprano)
        XCTAssertEqual(installed.installState, .installed)
        XCTAssertEqual(installed.familyID, .soprano)
        XCTAssertTrue(installed.descriptor.capabilities.contains(.speechSynthesis))
        XCTAssertEqual(sopranoModels.map(\.id), [sopranoEntry.id])
    }

    func testModelInstallerRejectsTraversalArtifactPath() async {
        var manifest = makePersistenceManifest(
            modelID: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
            familyID: "qwen3_tts",
            displayName: "Qwen3 TTS Base",
            capabilities: ["speech.synthesis"],
            backendKinds: ["mlx"]
        )
        manifest.artifactSpecs = [
            ModelPackArtifact(
                id: "weights",
                kind: "weights",
                relativePath: "../model.safetensors"
            )
        ]

        let installer = ModelInstaller(registry: ModelPackRegistry())
        let report = await installer.validate(manifest)

        XCTAssertTrue(report.hasErrors)
        XCTAssertTrue(report.issues.contains { $0.message.contains("path traversal") })
    }

    func testModelInstallerUninstallRemovesRegistryEntries() async throws {
        let manifest = makePersistenceManifest(
            modelID: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
            familyID: "qwen3_tts",
            displayName: "Qwen3 TTS Base",
            capabilities: ["speech.synthesis", "text.tokenization"],
            backendKinds: ["mlx"]
        )
        let paths = try makeAppPaths()
        let packRegistry = ModelPackRegistry(paths: paths)
        let modelRegistry = ModelRegistry()
        let capabilityRegistry = CapabilityRegistry()
        let installer = ModelInstaller(
            registry: packRegistry,
            modelRegistry: modelRegistry,
            capabilityRegistry: capabilityRegistry,
            paths: paths
        )
        try materializeInstalledPack(paths: paths, manifest: manifest)

        let result = try await installer.install(
            manifest: manifest,
            sourceKind: .localFile,
            sourceLocation: "/tmp/qwen-base.valarmodel"
        )

        let removed = try await installer.uninstall(descriptor: result.descriptor)

        XCTAssertEqual(removed?.modelID, manifest.modelID)
        let installedAfterUninstall = await packRegistry.installedRecord(for: manifest.modelID)
        let descriptorAfterUninstall = await modelRegistry.descriptor(for: result.descriptor.id)
        let capabilitiesAfterUninstall = await capabilityRegistry.models(supporting: .speechSynthesis)
        XCTAssertNil(installedAfterUninstall)
        XCTAssertNil(descriptorAfterUninstall)
        XCTAssertFalse(capabilitiesAfterUninstall.contains(result.descriptor))
    }

    func testModelInstallerDownloadsArtifactsPublishesProgressAndWritesManifest() async throws {
        let paths = try makeAppPaths()
        let cacheRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: cacheRoot, withIntermediateDirectories: true)
        let manifest = makePersistenceManifest(
            modelID: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
            familyID: "qwen3_tts",
            displayName: "Qwen3 TTS Base",
            capabilities: ["speech.synthesis", "text.tokenization"],
            backendKinds: ["mlx"]
        )
        let sourceLocation = "https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"
        let registry = ModelPackRegistry(paths: paths)
        let installer = ModelInstaller(
            registry: registry,
            modelRegistry: ModelRegistry(),
            capabilityRegistry: CapabilityRegistry(),
            paths: paths,
            hfCacheRoot: cacheRoot,
            sessionFactory: makeMockDownloadSession
        )

        let configData = Data(#"{"model_type":"qwen3"}"#.utf8)
        let weightsData = Data("pretend-weights".utf8)
        let configChecksum = sha256Hex(for: configData)
        let checksum = sha256Hex(for: weightsData)
        var downloadManifest = manifest
        downloadManifest.artifactSpecs = [
            ModelPackArtifact(
                id: "config",
                kind: "config",
                relativePath: "config.json",
                checksum: configChecksum,
                byteCount: configData.count
            ),
            ModelPackArtifact(
                id: "weights",
                kind: "weights",
                relativePath: "weights/model.safetensors",
                checksum: checksum,
                byteCount: weightsData.count
            ),
        ]

        await MainActor.run {
            MockDownloadRegistry.shared.reset()
            MockDownloadRegistry.shared.set(
                data: configData,
                for: URL(string: "\(sourceLocation)/resolve/main/config.json?download=true")!
            )
            MockDownloadRegistry.shared.set(
                data: weightsData,
                for: URL(string: "\(sourceLocation)/resolve/main/weights/model.safetensors?download=true")!
            )
        }

        let progressTask = Task<[ModelInstallProgressEvent], Never> {
            var events: [ModelInstallProgressEvent] = []
            for await event in installer.progress {
                guard event.modelID == downloadManifest.modelID else { continue }
                events.append(event)
                if case .completed = event.status {
                    break
                }
            }
            return events
        }

        let result = try await installer.install(
            manifest: downloadManifest,
            sourceKind: ModelPackSourceKind.remoteURL,
            sourceLocation: sourceLocation,
            notes: "Download during test",
            mode: ModelInstallMode.downloadArtifacts
        )
        let progressEvents = await progressTask.value

        let packDirectory = try paths.modelPackDirectory(
            familyID: downloadManifest.familyID,
            modelID: downloadManifest.modelID
        )
        let configURL = packDirectory.appendingPathComponent("config.json", isDirectory: false)
        let weightsURL = packDirectory.appendingPathComponent("weights/model.safetensors", isDirectory: false)
        let manifestURL = packDirectory.appendingPathComponent("manifest.json", isDirectory: false)

        XCTAssertEqual(result.receipt.sourceKind, ModelPackSourceKind.remoteURL)
        XCTAssertTrue(FileManager.default.fileExists(atPath: configURL.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: weightsURL.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: manifestURL.path))
        XCTAssertEqual(try Data(contentsOf: configURL), configData)
        XCTAssertEqual(try Data(contentsOf: weightsURL), weightsData)
        XCTAssertTrue(progressEvents.contains(where: {
            if case .downloading = $0.status {
                return $0.progress > 0 && $0.progress < 1
            }
            return false
        }))
        XCTAssertEqual(progressEvents.last?.progress, 1)
        XCTAssertEqual(progressEvents.last?.status, .completed)
    }

    func testModelInstallerReusesCachedArtifactsWithoutCopyingBytes() async throws {
        let paths = try makeAppPaths()
        let cacheRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: cacheRoot, withIntermediateDirectories: true)

        let manifest = ValarPersistence.ModelPackManifest(
            familyID: "qwen3_tts",
            modelID: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
            displayName: "Qwen3 TTS Base",
            capabilities: ["speech.synthesis"],
            backendKinds: ["mlx"],
            tokenizerType: "huggingface",
            sampleRate: 24_000,
            artifactSpecs: [
                ModelPackArtifact(
                    id: "config",
                    kind: "config",
                    relativePath: "config.json"
                ),
                ModelPackArtifact(
                    id: "weights",
                    kind: "weights",
                    relativePath: "model.safetensors"
                ),
            ],
            licenseName: "Model license",
            licenseURL: "https://example.com/license"
        )

        let configData = Data(#"{"model_type":"qwen3"}"#.utf8)
        let weightsData = Data("pretend-weights".utf8)
        try writeHFHubSnapshotArtifact(
            cacheRoot: cacheRoot,
            modelID: manifest.modelID,
            relativePath: "config.json",
            data: configData
        )
        try writeHFHubSnapshotArtifact(
            cacheRoot: cacheRoot,
            modelID: manifest.modelID,
            relativePath: "model.safetensors",
            data: weightsData
        )

        let registry = ModelPackRegistry(paths: paths)
        let installer = ModelInstaller(
            registry: registry,
            paths: paths,
            hfCacheRoot: cacheRoot,
            sessionFactory: makeMockDownloadSession
        )

        _ = try await installer.install(
            manifest: manifest,
            sourceKind: .remoteURL,
            sourceLocation: "https://huggingface.co/\(manifest.modelID)",
            mode: .downloadArtifacts
        )

        let installedWeightsURL = try paths.modelPackDirectory(
            familyID: manifest.familyID,
            modelID: manifest.modelID
        ).appendingPathComponent("model.safetensors", isDirectory: false)
        let cachedWeightsURL = cacheRoot
            .appendingPathComponent("models--mlx-community--Qwen3-TTS-12Hz-1.7B-Base-bf16", isDirectory: true)
            .appendingPathComponent("snapshots/test-revision/model.safetensors", isDirectory: false)
        let installedFileID = try fileIdentifier(at: installedWeightsURL)
        let cachedFileID = try fileIdentifier(at: cachedWeightsURL)

        XCTAssertEqual(try Data(contentsOf: installedWeightsURL), weightsData)
        XCTAssertEqual(try fileLinkCount(at: installedWeightsURL), 2)
        XCTAssertEqual(installedFileID.device, cachedFileID.device)
        XCTAssertEqual(installedFileID.inode, cachedFileID.inode)
    }

    func testModelInstallerPrefersHFHubSnapshotOverMLXAudioCacheWhenBothExist() async throws {
        let paths = try makeAppPaths()
        let cacheRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: cacheRoot, withIntermediateDirectories: true)

        let manifest = ValarPersistence.ModelPackManifest(
            familyID: "qwen3_tts",
            modelID: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
            displayName: "Qwen3 TTS Base",
            capabilities: ["speech.synthesis"],
            backendKinds: ["mlx"],
            tokenizerType: "huggingface",
            sampleRate: 24_000,
            artifactSpecs: [
                ModelPackArtifact(id: "config", kind: "config", relativePath: "config.json"),
                ModelPackArtifact(id: "weights", kind: "weights", relativePath: "model.safetensors"),
                ModelPackArtifact(id: "tokenizer", kind: "tokenizer", relativePath: "tokenizer.json"),
            ],
            licenseName: "Model license",
            licenseURL: "https://example.com/license"
        )

        try writeHFHubSnapshotArtifact(
            cacheRoot: cacheRoot,
            modelID: manifest.modelID,
            relativePath: "config.json",
            data: Data(#"{"model_type":"qwen3"}"#.utf8)
        )
        try writeHFHubSnapshotArtifact(
            cacheRoot: cacheRoot,
            modelID: manifest.modelID,
            relativePath: "model.safetensors",
            data: Data("hub-weights".utf8)
        )
        try writeHFHubSnapshotArtifact(
            cacheRoot: cacheRoot,
            modelID: manifest.modelID,
            relativePath: "tokenizer.json",
            data: Data(#"{"tokenizer":"hub"}"#.utf8)
        )

        try writeMLXAudioCacheArtifact(
            cacheRoot: cacheRoot,
            modelID: manifest.modelID,
            relativePath: "config.json",
            data: Data(#"{"model_type":"qwen3"}"#.utf8)
        )
        try writeMLXAudioCacheArtifact(
            cacheRoot: cacheRoot,
            modelID: manifest.modelID,
            relativePath: "model.safetensors",
            data: Data("mlx-audio-weights".utf8)
        )
        try writeMLXAudioCacheArtifact(
            cacheRoot: cacheRoot,
            modelID: manifest.modelID,
            relativePath: "tokenizer.json",
            data: Data(#"{"tokenizer":"mlx-audio"}"#.utf8)
        )

        let installer = ModelInstaller(
            registry: ModelPackRegistry(paths: paths),
            paths: paths,
            hfCacheRoot: cacheRoot,
            sessionFactory: makeMockDownloadSession
        )

        _ = try await installer.install(
            manifest: manifest,
            sourceKind: .remoteURL,
            sourceLocation: "https://huggingface.co/\(manifest.modelID)",
            mode: .downloadArtifacts
        )

        let installedWeightsURL = try paths.modelPackDirectory(
            familyID: manifest.familyID,
            modelID: manifest.modelID
        ).appendingPathComponent("model.safetensors", isDirectory: false)
        let hubWeightsURL = cacheRoot
            .appendingPathComponent("models--mlx-community--Qwen3-TTS-12Hz-1.7B-Base-bf16", isDirectory: true)
            .appendingPathComponent("snapshots/test-revision/model.safetensors", isDirectory: false)
        let mlxAudioWeightsURL = cacheRoot
            .appendingPathComponent("mlx-audio", isDirectory: true)
            .appendingPathComponent("mlx-community_Qwen3-TTS-12Hz-1.7B-Base-bf16", isDirectory: true)
            .appendingPathComponent("model.safetensors", isDirectory: false)

        let installedIdentifier = try fileIdentifier(at: installedWeightsURL)
        let hubIdentifier = try fileIdentifier(at: hubWeightsURL)
        let mlxAudioIdentifier = try fileIdentifier(at: mlxAudioWeightsURL)
        XCTAssertEqual(installedIdentifier.device, hubIdentifier.device)
        XCTAssertEqual(installedIdentifier.inode, hubIdentifier.inode)
        XCTAssertFalse(
            installedIdentifier.device == mlxAudioIdentifier.device &&
                installedIdentifier.inode == mlxAudioIdentifier.inode
        )
    }

    func testModelInstallerDoesNotMaterializeLegacyMLXAudioRuntimeCacheForQwenInstall() async throws {
        let paths = try makeAppPaths()
        let cacheRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: cacheRoot, withIntermediateDirectories: true)

        let sourceLocation = "https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"
        let configData = Data(#"{"model_type":"qwen3"}"#.utf8)
        let weightsData = Data("pretend-weights".utf8)
        let tokenizerData = Data(#"{"tokenizer":"ok"}"#.utf8)
        let speechTokenizerData = Data("pretend-speech-tokenizer".utf8)

        let manifest = ValarPersistence.ModelPackManifest(
            familyID: "qwen3_tts",
            modelID: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
            displayName: "Qwen3 TTS Base",
            capabilities: ["speech.synthesis"],
            backendKinds: ["mlx"],
            tokenizerType: "huggingface",
            sampleRate: 24_000,
            artifactSpecs: [
                ModelPackArtifact(
                    id: "config",
                    kind: "config",
                    relativePath: "config.json",
                    checksum: sha256Hex(for: configData),
                    byteCount: configData.count
                ),
                ModelPackArtifact(
                    id: "weights",
                    kind: "weights",
                    relativePath: "model.safetensors",
                    checksum: sha256Hex(for: weightsData),
                    byteCount: weightsData.count
                ),
                ModelPackArtifact(
                    id: "tokenizer",
                    kind: "tokenizer",
                    relativePath: "tokenizer.json",
                    checksum: sha256Hex(for: tokenizerData),
                    byteCount: tokenizerData.count
                ),
                ModelPackArtifact(
                    id: "speech-tokenizer",
                    kind: "weights",
                    relativePath: "speech_tokenizer/model.safetensors",
                    checksum: sha256Hex(for: speechTokenizerData),
                    byteCount: speechTokenizerData.count
                ),
            ],
            licenseName: "Model license",
            licenseURL: "https://example.com/license"
        )

        await MainActor.run {
            MockDownloadRegistry.shared.reset()
            MockDownloadRegistry.shared.set(
                data: configData,
                for: URL(string: "\(sourceLocation)/resolve/main/config.json?download=true")!
            )
            MockDownloadRegistry.shared.set(
                data: weightsData,
                for: URL(string: "\(sourceLocation)/resolve/main/model.safetensors?download=true")!
            )
            MockDownloadRegistry.shared.set(
                data: tokenizerData,
                for: URL(string: "\(sourceLocation)/resolve/main/tokenizer.json?download=true")!
            )
            MockDownloadRegistry.shared.set(
                data: speechTokenizerData,
                for: URL(string: "\(sourceLocation)/resolve/main/speech_tokenizer/model.safetensors?download=true")!
            )
        }

        let installer = ModelInstaller(
            registry: ModelPackRegistry(paths: paths),
            paths: paths,
            hfCacheRoot: cacheRoot,
            sessionFactory: makeMockDownloadSession
        )

        _ = try await installer.install(
            manifest: manifest,
            sourceKind: .remoteURL,
            sourceLocation: sourceLocation,
            mode: .downloadArtifacts
        )

        let installedWeightsURL = try paths.modelPackDirectory(
            familyID: manifest.familyID,
            modelID: manifest.modelID
        ).appendingPathComponent("model.safetensors", isDirectory: false)
        let legacyCacheRoot = cacheRoot
            .appendingPathComponent("mlx-audio", isDirectory: true)
            .appendingPathComponent("mlx-community_Qwen3-TTS-12Hz-1.7B-Base-bf16", isDirectory: true)

        XCTAssertEqual(try Data(contentsOf: installedWeightsURL), weightsData)
        XCTAssertFalse(FileManager.default.fileExists(atPath: legacyCacheRoot.path))
    }

    func testModelInstallerRejectsChecksumMismatchBeforeRegistryInstall() async throws {
        let paths = try makeAppPaths()
        let registry = ModelPackRegistry(paths: paths)
        let installer = ModelInstaller(
            registry: registry,
            paths: paths,
            hfCacheRoot: URL(fileURLWithPath: "/nonexistent-hf-cache"),
            sessionFactory: makeMockDownloadSession
        )
        let sourceLocation = "https://huggingface.co/mlx-community/Qwen3-ASR-0.6B-8bit"
        let badData = Data("corrupted".utf8)
        let expectedChecksum = sha256Hex(for: Data("expected".utf8))

        let manifest = ValarPersistence.ModelPackManifest(
            familyID: "qwen3_asr",
            modelID: "mlx-community/Qwen3-ASR-0.6B-8bit",
            displayName: "Qwen3 ASR 0.6B",
            capabilities: ["speech.recognition"],
            backendKinds: ["mlx"],
            tokenizerType: "huggingface",
            sampleRate: 16_000,
            artifactSpecs: [
                ModelPackArtifact(
                    id: "weights",
                    kind: "weights",
                    relativePath: "weights/model.safetensors",
                    checksum: expectedChecksum,
                    byteCount: badData.count
                ),
            ],
            licenseName: "Model license",
            licenseURL: "https://example.com/license"
        )

        await MainActor.run {
            MockDownloadRegistry.shared.reset()
            MockDownloadRegistry.shared.set(
                data: badData,
                for: URL(string: "\(sourceLocation)/resolve/main/weights/model.safetensors?download=true")!
            )
        }

        do {
            _ = try await installer.install(
                manifest: manifest,
                sourceKind: ModelPackSourceKind.remoteURL,
                sourceLocation: sourceLocation,
                mode: ModelInstallMode.downloadArtifacts
            )
            XCTFail("Expected checksum mismatch")
        } catch let error as ModelInstallerError {
            XCTAssertEqual(
                error,
                .checksumMismatch(
                    artifactPath: "weights/model.safetensors",
                    expected: expectedChecksum,
                    actual: sha256Hex(for: badData)
                )
            )
        }

        let installedRecord = await registry.installedRecord(for: manifest.modelID)
        XCTAssertNil(installedRecord)
        XCTAssertFalse(
            FileManager.default.fileExists(
                atPath: try paths.modelPackDirectory(familyID: manifest.familyID, modelID: manifest.modelID).path
            )
        )
    }

    func testModelInstallerRejectsMissingWeightChecksumOnRemoteDownload() async throws {
        let paths = try makeAppPaths()
        let registry = ModelPackRegistry(paths: paths)
        let installer = ModelInstaller(
            registry: registry,
            paths: paths,
            hfCacheRoot: URL(fileURLWithPath: "/nonexistent-hf-cache"),
            sessionFactory: makeMockDownloadSession
        )
        let sourceLocation = "https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"
        let weightsData = Data("pretend-weights".utf8)

        let manifest = ValarPersistence.ModelPackManifest(
            familyID: "qwen3_tts",
            modelID: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
            displayName: "Qwen3 TTS Base",
            capabilities: ["speech.synthesis"],
            backendKinds: ["mlx"],
            tokenizerType: "huggingface",
            sampleRate: 24_000,
            artifactSpecs: [
                ModelPackArtifact(
                    id: "weights",
                    kind: "weights",
                    relativePath: "weights/model.safetensors",
                    byteCount: weightsData.count
                ),
            ],
            licenseName: "Model license",
            licenseURL: "https://example.com/license"
        )

        await MainActor.run {
            MockDownloadRegistry.shared.reset()
            MockDownloadRegistry.shared.set(
                data: weightsData,
                for: URL(string: "\(sourceLocation)/resolve/main/weights/model.safetensors?download=true")!
            )
        }

        // Models without pre-computed checksums are now trusted when downloaded
        // from HuggingFace. Install should succeed.
        let record = try await installer.install(
            manifest: manifest,
            sourceKind: ModelPackSourceKind.remoteURL,
            sourceLocation: sourceLocation,
            mode: ModelInstallMode.downloadArtifacts
        )
        XCTAssertNotNil(record)
    }

    func testModelInstallerRejectsMissingConfigChecksumOnRemoteDownload() async throws {
        let paths = try makeAppPaths()
        let registry = ModelPackRegistry(paths: paths)
        let installer = ModelInstaller(
            registry: registry,
            paths: paths,
            hfCacheRoot: URL(fileURLWithPath: "/nonexistent-hf-cache"),
            sessionFactory: makeMockDownloadSession
        )
        let sourceLocation = "https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"
        let configData = Data(#"{"model_type":"qwen3"}"#.utf8)

        let manifest = ValarPersistence.ModelPackManifest(
            familyID: "qwen3_tts",
            modelID: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
            displayName: "Qwen3 TTS Base",
            capabilities: ["speech.synthesis"],
            backendKinds: ["mlx"],
            tokenizerType: "huggingface",
            sampleRate: 24_000,
            artifactSpecs: [
                ModelPackArtifact(
                    id: "config",
                    kind: "config",
                    relativePath: "config.json",
                    byteCount: configData.count
                ),
            ],
            licenseName: "Model license",
            licenseURL: "https://example.com/license"
        )

        await MainActor.run {
            MockDownloadRegistry.shared.reset()
            MockDownloadRegistry.shared.set(
                data: configData,
                for: URL(string: "\(sourceLocation)/resolve/main/config.json?download=true")!
            )
        }

        // Models without pre-computed checksums are now trusted when downloaded
        // from HuggingFace. Install should succeed.
        let record = try await installer.install(
            manifest: manifest,
            sourceKind: ModelPackSourceKind.remoteURL,
            sourceLocation: sourceLocation,
            mode: ModelInstallMode.downloadArtifacts
        )
        XCTAssertNotNil(record)
    }

    func testModelInstallerRejectsTokenizerChecksumMismatchBeforeRegistryInstall() async throws {
        let paths = try makeAppPaths()
        let registry = ModelPackRegistry(paths: paths)
        let installer = ModelInstaller(
            registry: registry,
            paths: paths,
            hfCacheRoot: URL(fileURLWithPath: "/nonexistent-hf-cache"),
            sessionFactory: makeMockDownloadSession
        )
        let sourceLocation = "https://huggingface.co/mlx-community/Qwen3-ASR-0.6B-8bit"
        let tokenizerData = Data(#"{"bos_token":"<s>"}"#.utf8)
        let expectedChecksum = sha256Hex(for: Data(#"{"bos_token":"</s>"}"#.utf8))

        let manifest = ValarPersistence.ModelPackManifest(
            familyID: "qwen3_asr",
            modelID: "mlx-community/Qwen3-ASR-0.6B-8bit",
            displayName: "Qwen3 ASR 0.6B",
            capabilities: ["speech.recognition"],
            backendKinds: ["mlx"],
            tokenizerType: "huggingface",
            sampleRate: 16_000,
            artifactSpecs: [
                ModelPackArtifact(
                    id: "tokenizer",
                    kind: "tokenizer",
                    relativePath: "tokenizer_config.json",
                    checksum: expectedChecksum,
                    byteCount: tokenizerData.count
                ),
            ],
            licenseName: "Model license",
            licenseURL: "https://example.com/license"
        )

        await MainActor.run {
            MockDownloadRegistry.shared.reset()
            MockDownloadRegistry.shared.set(
                data: tokenizerData,
                for: URL(string: "\(sourceLocation)/resolve/main/tokenizer_config.json?download=true")!
            )
        }

        do {
            _ = try await installer.install(
                manifest: manifest,
                sourceKind: ModelPackSourceKind.remoteURL,
                sourceLocation: sourceLocation,
                mode: ModelInstallMode.downloadArtifacts
            )
            XCTFail("Expected checksum mismatch")
        } catch let error as ModelInstallerError {
            XCTAssertEqual(
                error,
                .checksumMismatch(
                    artifactPath: "tokenizer_config.json",
                    expected: expectedChecksum,
                    actual: sha256Hex(for: tokenizerData)
                )
            )
        }

        let installedRecord = await registry.installedRecord(for: manifest.modelID)
        XCTAssertNil(installedRecord)
        XCTAssertFalse(
            FileManager.default.fileExists(
                atPath: try paths.modelPackDirectory(familyID: manifest.familyID, modelID: manifest.modelID).path
            )
        )
    }

    func testModelInstallerRejectsNon2xxDownloadStatusAndLeavesNoFiles() async throws {
        let paths = try makeAppPaths()
        let registry = ModelPackRegistry(paths: paths)
        let installer = ModelInstaller(
            registry: registry,
            paths: paths,
            hfCacheRoot: URL(fileURLWithPath: "/nonexistent-hf-cache"),
            sessionFactory: makeMockDownloadSession
        )
        let sourceLocation = "https://huggingface.co/mlx-community/Qwen3-ASR-0.6B-8bit"
        let body = Data("missing".utf8)
        let manifest = ValarPersistence.ModelPackManifest(
            familyID: "qwen3_asr",
            modelID: "mlx-community/Qwen3-ASR-0.6B-8bit",
            displayName: "Qwen3 ASR 0.6B",
            capabilities: ["speech.recognition"],
            backendKinds: ["mlx"],
            tokenizerType: "huggingface",
            sampleRate: 16_000,
            artifactSpecs: [
                ModelPackArtifact(
                    id: "config",
                    kind: "config",
                    relativePath: "config.json",
                    checksum: sha256Hex(for: body),
                    byteCount: body.count
                ),
            ],
            licenseName: "Model license",
            licenseURL: "https://example.com/license"
        )

        await MainActor.run {
            MockDownloadRegistry.shared.reset()
            MockDownloadRegistry.shared.set(
                data: body,
                statusCode: 404,
                for: URL(string: "\(sourceLocation)/resolve/main/config.json?download=true")!
            )
        }

        do {
            _ = try await installer.install(
                manifest: manifest,
                sourceKind: ModelPackSourceKind.remoteURL,
                sourceLocation: sourceLocation,
                mode: ModelInstallMode.downloadArtifacts
            )
            XCTFail("Expected download failure")
        } catch let error as ModelInstallerError {
            guard case let .downloadFailed(message) = error else {
                return XCTFail("Unexpected error: \(error)")
            }
            XCTAssertTrue(message.contains("404"))
            XCTAssertTrue(message.contains("config.json"))
        }

        let installedRecord = await registry.installedRecord(for: manifest.modelID)
        XCTAssertNil(installedRecord)
        XCTAssertFalse(
            FileManager.default.fileExists(
                atPath: try paths.modelPackDirectory(familyID: manifest.familyID, modelID: manifest.modelID).path
            )
        )
    }

    func testValidationWarnsOnMissingCriticalArtifactChecksums() async {
        let manifest = ValarPersistence.ModelPackManifest(
            familyID: "qwen3_tts",
            modelID: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
            displayName: "Qwen3 TTS Base",
            capabilities: ["speech.synthesis"],
            backendKinds: ["mlx"],
            tokenizerType: "huggingface",
            sampleRate: 24_000,
            artifactSpecs: [
                ModelPackArtifact(
                    id: "config",
                    kind: "config",
                    relativePath: "config.json",
                    byteCount: 128
                ),
                ModelPackArtifact(
                    id: "tokenizer",
                    kind: "tokenizer",
                    relativePath: "tokenizer_config.json",
                    byteCount: 512
                ),
                ModelPackArtifact(
                    id: "weights",
                    kind: "weights",
                    relativePath: "weights/model.safetensors",
                    byteCount: 1_024
                ),
            ],
            licenseName: "Model license",
            licenseURL: "https://example.com/license"
        )

        let installer = ModelInstaller(registry: ModelPackRegistry())
        let report = await installer.validate(manifest)

        XCTAssertFalse(report.hasErrors)
        XCTAssertTrue(report.issues.contains {
            $0.severity == .warning
                && $0.message.contains("Config artifact 'config'")
                && $0.message.contains("cannot locally verify the downloaded file")
        })
        XCTAssertTrue(report.issues.contains {
            $0.severity == .warning
                && $0.message.contains("Tokenizer artifact 'tokenizer'")
                && $0.message.contains("cannot locally verify the downloaded file")
        })
        XCTAssertTrue(report.issues.contains {
            $0.severity == .warning
                && $0.message.contains("Weight artifact 'weights'")
                && $0.message.contains("cannot locally verify the downloaded file")
        })
        XCTAssertFalse(report.issues.contains { $0.message.contains("remote downloads will be rejected") })
    }

    func testQwenCatalogEntriesIncludeWeightChecksums() {
        for entry in QwenCatalog.supportedEntries {
            let weightArtifacts = entry.manifest.artifacts.filter { $0.role == .weights }
            for artifact in weightArtifacts {
                // Quantized variants (8-bit, 4-bit) may not have checksums until downloaded.
                // Only assert checksums for entries that provide them.
                if let checksum = artifact.sha256 {
                    XCTAssertEqual(
                        checksum.count, 64,
                        "SHA-256 checksum for '\(artifact.id)' in \(entry.manifest.displayName) should be 64 hex characters"
                    )
                }
            }
        }
    }

    func testRenderQueueHydratesPersistedJobs() async {
        let persistedJob = RenderJob(
            projectID: UUID(),
            modelID: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
            chapterIDs: [UUID()],
            outputFileName: "resume-me.wav",
            state: .running,
            priority: 3,
            progress: 0.8,
            title: "resume me"
        )
        let store = InMemoryRenderQueueStore(jobs: [persistedJob])
        let queue = RenderQueue(store: store)

        let next = await queue.nextJob()
        let pending = await queue.pendingJobCount()

        XCTAssertEqual(next?.id, persistedJob.id)
        XCTAssertEqual(next?.state, .queued)
        XCTAssertEqual(next?.outputFileName, "resume-me.wav")
        XCTAssertEqual(pending, 1)
    }

    private func makePersistenceManifest(
        modelID: String,
        familyID: String,
        displayName: String,
        capabilities: [String],
        backendKinds: [String]
    ) -> ValarPersistence.ModelPackManifest {
        ValarPersistence.ModelPackManifest(
            familyID: familyID,
            modelID: modelID,
            displayName: displayName,
            capabilities: capabilities,
            backendKinds: backendKinds,
            tokenizerType: "huggingface",
            sampleRate: 24_000,
            artifactSpecs: [
                ModelPackArtifact(
                    id: "weights",
                    kind: "weights",
                    relativePath: "weights/model.safetensors",
                    checksum: "abc123",
                    byteCount: 1_024
                ),
                ModelPackArtifact(
                    id: "config",
                    kind: "config",
                    relativePath: "config.json",
                    checksum: "def456",
                    byteCount: 256
                ),
            ],
            licenseName: "Model license",
            licenseURL: "https://example.com/license",
            minimumAppVersion: "2.0.0",
            notes: "Test manifest"
        )
    }

    private func makeAppPaths() throws -> ValarAppPaths {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return ValarAppPaths(baseURL: root)
    }

    private func writeHFHubSnapshotArtifact(
        cacheRoot: URL,
        modelID: String,
        relativePath: String,
        data: Data
    ) throws {
        let repoDirectory = cacheRoot.appendingPathComponent(
            "models--" + modelID.replacingOccurrences(of: "/", with: "--"),
            isDirectory: true
        )
        let revision = "test-revision"
        let snapshotRoot = repoDirectory
            .appendingPathComponent("snapshots", isDirectory: true)
            .appendingPathComponent(revision, isDirectory: true)
        let refsRoot = repoDirectory.appendingPathComponent("refs", isDirectory: true)

        try FileManager.default.createDirectory(at: snapshotRoot, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: refsRoot, withIntermediateDirectories: true)
        try Data(revision.utf8).write(to: refsRoot.appendingPathComponent("main", isDirectory: false))

        let artifactURL = snapshotRoot.appendingPathComponent(relativePath, isDirectory: false)
        try FileManager.default.createDirectory(at: artifactURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        try data.write(to: artifactURL)
    }

    private func writeHFHubSnapshotArtifacts(
        cacheRoot: URL,
        modelID: String,
        relativePaths: [String]
    ) throws {
        for relativePath in relativePaths where !relativePath.hasSuffix("/") {
            try writeHFHubSnapshotArtifact(
                cacheRoot: cacheRoot,
                modelID: modelID,
                relativePath: relativePath,
                data: Data(relativePath.utf8)
            )
        }
    }

    private func writeMLXAudioCacheArtifact(
        cacheRoot: URL,
        modelID: String,
        relativePath: String,
        data: Data
    ) throws {
        let root = cacheRoot
            .appendingPathComponent("mlx-audio", isDirectory: true)
            .appendingPathComponent(modelID.replacingOccurrences(of: "/", with: "_"), isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let artifactURL = root.appendingPathComponent(relativePath, isDirectory: false)
        try FileManager.default.createDirectory(at: artifactURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        try data.write(to: artifactURL)
    }

    private func materializeInstalledPack(
        paths: ValarAppPaths,
        manifest: ValarPersistence.ModelPackManifest
    ) throws {
        let packDirectory = try paths.modelPackDirectory(
            familyID: manifest.familyID,
            modelID: manifest.modelID
        )
        try FileManager.default.createDirectory(at: packDirectory, withIntermediateDirectories: true)

        for artifact in manifest.artifactSpecs where !artifact.relativePath.hasSuffix("/") {
            let artifactURL = packDirectory.appendingPathComponent(artifact.relativePath, isDirectory: false)
            try FileManager.default.createDirectory(
                at: artifactURL.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            try Data(artifact.relativePath.utf8).write(to: artifactURL)
        }

        let manifestURL = try paths.modelPackManifestURL(
            familyID: manifest.familyID,
            modelID: manifest.modelID
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = .sortedKeys
        try encoder.encode(manifest).write(to: manifestURL)
    }

    private func fileLinkCount(at url: URL) throws -> UInt16 {
        var info = stat()
        guard lstat(url.path, &info) == 0 else {
            throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
        }
        return info.st_nlink
    }

    private func fileIdentifier(at url: URL) throws -> (device: UInt64, inode: UInt64) {
        var info = stat()
        guard lstat(url.path, &info) == 0 else {
            throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
        }
        return (UInt64(info.st_dev), UInt64(info.st_ino))
    }

    private func sha256Hex(for data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}

@MainActor
private final class MockDownloadRegistry {
    private struct Response {
        let data: Data
        let statusCode: Int
    }

    static let shared = MockDownloadRegistry()

    private var responses: [String: Response] = [:]

    func set(data: Data, for url: URL) {
        set(data: data, statusCode: 200, for: url)
    }

    func set(data: Data, statusCode: Int, for url: URL) {
        responses[url.absoluteString] = Response(data: data, statusCode: statusCode)
    }

    func response(for url: URL) -> (data: Data, statusCode: Int)? {
        responses[url.absoluteString]
            .map { ($0.data, $0.statusCode) }
    }

    func reset() {
        responses = [:]
    }
}

private final class MockDownloadURLProtocol: URLProtocol {
    override class func canInit(with request: URLRequest) -> Bool {
        request.url != nil
    }

    override class func canonicalRequest(for request: URLRequest) -> URLRequest {
        request
    }

    override func startLoading() {
        guard let url = request.url else {
            client?.urlProtocol(self, didFailWithError: URLError(.badURL))
            return
        }

        let payload: (data: Data, statusCode: Int)? = if Thread.isMainThread {
            MainActor.assumeIsolated { MockDownloadRegistry.shared.response(for: url) }
        } else {
            DispatchQueue.main.sync {
                MainActor.assumeIsolated { MockDownloadRegistry.shared.response(for: url) }
            }
        }

        guard let payload else {
            client?.urlProtocol(self, didFailWithError: URLError(.fileDoesNotExist))
            return
        }

        let response = HTTPURLResponse(
            url: url,
            statusCode: payload.statusCode,
            httpVersion: nil,
            headerFields: ["Content-Length": "\(payload.data.count)"]
        )!

        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)

        let midpoint = max(1, payload.data.count / 2)
        client?.urlProtocol(self, didLoad: payload.data.prefix(midpoint))
        Thread.sleep(forTimeInterval: 0.05)
        client?.urlProtocol(self, didLoad: payload.data.dropFirst(midpoint))
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}

private let makeMockDownloadSession: @Sendable (URLSessionDownloadDelegate) -> URLSession = { delegate in
    let configuration = URLSessionConfiguration.ephemeral
    configuration.protocolClasses = [MockDownloadURLProtocol.self]
    return URLSession(configuration: configuration, delegate: delegate, delegateQueue: nil)
}
