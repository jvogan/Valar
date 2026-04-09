import Foundation
import XCTest
@testable import ValarCore
import ValarModelKit
import ValarPersistence

private struct EchoTranslator: TranslationProvider {
    func translate(_ request: TranslationRequest) async throws -> String {
        "[\(request.targetLanguage)] \(request.text)"
    }
}

private struct LocalStubInferenceBackend: InferenceBackend {
    var backendKind: BackendKind { .mlx }
    var runtimeCapabilities: BackendCapabilities { BackendCapabilities() }

    func validate(requirement: BackendRequirement) async throws {}

    func loadModel(
        descriptor: ModelDescriptor,
        configuration: ModelRuntimeConfiguration
    ) async throws -> any ValarModel {
        LocalStubModel(descriptor: descriptor, backendKind: configuration.backendKind)
    }

    func unloadModel(_ model: any ValarModel) async throws {}
}

private struct WarmStartStubInferenceBackend: InferenceBackend {
    var backendKind: BackendKind { .mlx }
    var runtimeCapabilities: BackendCapabilities {
        BackendCapabilities(features: [.warmStart], supportedFamilies: [.qwen3TTS, .qwen3ASR])
    }

    func validate(requirement: BackendRequirement) async throws {}

    func loadModel(
        descriptor: ModelDescriptor,
        configuration: ModelRuntimeConfiguration
    ) async throws -> any ValarModel {
        LocalStubModel(descriptor: descriptor, backendKind: configuration.backendKind)
    }

    func unloadModel(_ model: any ValarModel) async throws {}
}

private struct LocalStubModel: ValarModel {
    let descriptor: ModelDescriptor
    let backendKind: BackendKind
}

final class ValarCoreTests: XCTestCase {
    func testCreateVoiceDefaultsToVoiceDesignModelWhenPromptProvided() async throws {
        let baseURL = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        let runtime = try ValarRuntime(
            paths: ValarAppPaths(baseURL: baseURL),
            runtimeConfiguration: RuntimeConfiguration(),
            inferenceBackend: LocalStubInferenceBackend()
        )

        let voice = try await runtime.createVoice(
            VoiceCreateRequest(
                label: "Designed Narrator",
                voicePrompt: "Warm, articulate guide"
            )
        )

        XCTAssertEqual(voice.modelID, ValarRuntime.defaultVoiceDesignModelID.rawValue)
        XCTAssertEqual(voice.runtimeModelID, ValarRuntime.defaultVoiceDesignModelID.rawValue)
        XCTAssertEqual(voice.voicePrompt, "Warm, articulate guide")
        XCTAssertEqual(voice.voiceKind, VoiceKind.legacyPrompt.rawValue)
    }

    func testCreateVoiceDefaultsToCustomVoiceModelWithoutPrompt() async throws {
        let baseURL = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        let runtime = try ValarRuntime(
            paths: ValarAppPaths(baseURL: baseURL),
            runtimeConfiguration: RuntimeConfiguration(),
            inferenceBackend: LocalStubInferenceBackend()
        )

        let voice = try await runtime.createVoice(
            VoiceCreateRequest(label: "Saved Voice")
        )

        XCTAssertEqual(voice.modelID, ValarRuntime.defaultVoiceCreateModelID.rawValue)
        XCTAssertEqual(voice.runtimeModelID, ValarRuntime.defaultVoiceCreateModelID.rawValue)
        XCTAssertNil(voice.voicePrompt)
        XCTAssertNil(voice.voiceKind)
    }

    func testLegacyPromptVoicePrefersVoiceDesignRuntimeModel() {
        let legacyPromptVoice = VoiceLibraryRecord(
            label: "The Architect Dark",
            modelID: ValarRuntime.defaultVoiceCreateModelID.rawValue,
            runtimeModelID: ValarRuntime.defaultVoiceCreateModelID.rawValue,
            voiceKind: VoiceKind.legacyPrompt.rawValue,
            voicePrompt: "A low, deliberate architect with cinematic gravitas."
        )

        XCTAssertEqual(
            legacyPromptVoice.preferredRuntimeModelIdentifier,
            ValarRuntime.defaultVoiceDesignModelID
        )

        let profile = legacyPromptVoice.makeVoiceProfile()
        XCTAssertEqual(profile.sourceModel, ValarRuntime.defaultVoiceDesignModelID)
        XCTAssertEqual(profile.runtimeModel, ValarRuntime.defaultVoiceDesignModelID)
        XCTAssertEqual(profile.label, "A low, deliberate architect with cinematic gravitas.")
        XCTAssertEqual(profile.voiceKind, .legacyPrompt)
        XCTAssertTrue(profile.isLegacyExpressive)
    }

    func testClonePromptVoicePreservesStoredRuntimeModel() {
        let clonePromptVoice = VoiceLibraryRecord(
            label: "Stable Narrator",
            modelID: ValarRuntime.defaultVoiceCloneRuntimeModelID.rawValue,
            runtimeModelID: ValarRuntime.defaultVoiceCloneRuntimeModelID.rawValue,
            referenceAudioAssetName: "reference.wav",
            referenceTranscript: "Hello there.",
            conditioningFormat: VoiceLibraryRecord.qwenClonePromptConditioningFormat,
            voiceKind: VoiceKind.clonePrompt.rawValue
        )

        XCTAssertEqual(
            clonePromptVoice.preferredRuntimeModelIdentifier,
            ValarRuntime.defaultVoiceCloneRuntimeModelID
        )

        let profile = clonePromptVoice.makeVoiceProfile()
        XCTAssertEqual(profile.sourceModel, ValarRuntime.defaultVoiceCloneRuntimeModelID)
        XCTAssertEqual(profile.runtimeModel, ValarRuntime.defaultVoiceCloneRuntimeModelID)
        XCTAssertEqual(profile.voiceKind, .clonePrompt)
        XCTAssertFalse(profile.isLegacyExpressive)
    }

    func testPreferredVoiceBehaviorMatchesRecoveredQwenVoiceKinds() {
        let expressiveVoice = VoiceLibraryRecord(
            label: "The Architect Dark",
            modelID: ValarRuntime.defaultVoiceCreateModelID.rawValue,
            runtimeModelID: ValarRuntime.defaultVoiceDesignModelID.rawValue,
            voiceKind: VoiceKind.legacyPrompt.rawValue,
            voicePrompt: "Low, deliberate, cinematic."
        )
        XCTAssertEqual(expressiveVoice.preferredVoiceBehavior, .expressive)

        let stableVoice = VoiceLibraryRecord(
            label: "Stable Narrator",
            modelID: ValarRuntime.defaultVoiceCloneRuntimeModelID.rawValue,
            runtimeModelID: ValarRuntime.defaultVoiceCloneRuntimeModelID.rawValue,
            referenceAudioAssetName: "reference.wav",
            referenceTranscript: "Hello there.",
            conditioningFormat: VoiceLibraryRecord.qwenClonePromptConditioningFormat,
            voiceKind: VoiceKind.clonePrompt.rawValue
        )
        XCTAssertEqual(stableVoice.preferredVoiceBehavior, .stableNarrator)

        let presetVoice = VoiceLibraryRecord(
            label: "Neutral Female",
            modelID: "mlx-community/Voxtral-4B-TTS-2603-mlx-4bit",
            runtimeModelID: "mlx-community/Voxtral-4B-TTS-2603-mlx-4bit",
            backendVoiceID: "neutral_female"
        )
        XCTAssertEqual(presetVoice.preferredVoiceBehavior, .auto)
    }

    func testReusableQwenClonePromptDetectionRequiresStructuredPayload() throws {
        struct Payload: Encodable {
            let version: Int
            let refSpeakerEmbedding: Data?
            let refCode: Data?
            let numCodeGroups: Int?
            let frameCount: Int?
            let xVectorOnlyMode: Bool
            let iclMode: Bool
        }

        let speakerEmbedding = Data([0x00, 0x00, 0x80, 0x3F])
        let refCodes = withUnsafeBytes(of: Int32(7).littleEndian) { Data($0) }
        let payload = try JSONEncoder().encode(
            Payload(
                version: 1,
                refSpeakerEmbedding: speakerEmbedding,
                refCode: refCodes,
                numCodeGroups: 1,
                frameCount: 1,
                xVectorOnlyMode: false,
                iclMode: true
            )
        )

        let structuredVoice = VoiceLibraryRecord(
            label: "Structured Stable Narrator",
            modelID: ValarRuntime.defaultVoiceCloneRuntimeModelID.rawValue,
            runtimeModelID: ValarRuntime.defaultVoiceCloneRuntimeModelID.rawValue,
            referenceAudioAssetName: "reference.wav",
            referenceTranscript: "Hello there.",
            speakerEmbedding: payload,
            conditioningFormat: VoiceLibraryRecord.qwenClonePromptConditioningFormat,
            voiceKind: VoiceKind.clonePrompt.rawValue
        )
        XCTAssertTrue(structuredVoice.hasReusableQwenClonePrompt)

        let legacyVoice = VoiceLibraryRecord(
            label: "Legacy Stable Narrator",
            modelID: ValarRuntime.defaultVoiceCloneRuntimeModelID.rawValue,
            runtimeModelID: ValarRuntime.defaultVoiceCloneRuntimeModelID.rawValue,
            referenceAudioAssetName: "reference.wav",
            referenceTranscript: "Hello there.",
            speakerEmbedding: speakerEmbedding,
            conditioningFormat: VoiceLibraryRecord.qwenClonePromptConditioningFormat,
            voiceKind: VoiceKind.clonePrompt.rawValue
        )
        XCTAssertFalse(legacyVoice.hasReusableQwenClonePrompt)
    }

    func testWarmStartCandidateIDsPreferPrimaryQwenTTSAndASR() {
        let base = CatalogModel(
            id: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
            descriptor: ModelDescriptor(
                id: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
                familyID: .qwen3TTS,
                displayName: "Qwen3 TTS Base",
                domain: .tts,
                capabilities: [.speechSynthesis, .voiceCloning]
            ),
            familyID: .qwen3TTS,
            installState: .installed,
            providerName: "Valar",
            providerURL: nil,
            sourceKind: .localFile,
            isRecommended: true,
            manifestPath: "/tmp/base/manifest.json",
            installedPath: "/tmp/base",
            artifactCount: 4,
            supportedBackends: [.mlx],
            notes: nil,
            cachedOnDisk: false
        )
        let customVoice = CatalogModel(
            id: "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16",
            descriptor: ModelDescriptor(
                id: "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16",
                familyID: .qwen3TTS,
                displayName: "Qwen3 TTS CustomVoice",
                domain: .tts,
                capabilities: [.speechSynthesis, .voiceCloning]
            ),
            familyID: .qwen3TTS,
            installState: .installed,
            providerName: "Valar",
            providerURL: nil,
            sourceKind: .localFile,
            isRecommended: true,
            manifestPath: "/tmp/customvoice/manifest.json",
            installedPath: "/tmp/customvoice",
            artifactCount: 4,
            supportedBackends: [.mlx],
            notes: nil,
            cachedOnDisk: false
        )
        let asr = CatalogModel(
            id: "mlx-community/Qwen3-ASR-0.6B-8bit",
            descriptor: ModelDescriptor(
                id: "mlx-community/Qwen3-ASR-0.6B-8bit",
                familyID: .qwen3ASR,
                displayName: "Qwen3 ASR 0.6B",
                domain: .stt,
                capabilities: [.speechRecognition]
            ),
            familyID: .qwen3ASR,
            installState: .installed,
            providerName: "Valar",
            providerURL: nil,
            sourceKind: .localFile,
            isRecommended: true,
            manifestPath: "/tmp/asr/manifest.json",
            installedPath: "/tmp/asr",
            artifactCount: 3,
            supportedBackends: [.mlx],
            notes: nil,
            cachedOnDisk: false
        )
        let voxtral = CatalogModel(
            id: "mlx-community/Voxtral-4B-TTS-2603-mlx-4bit",
            descriptor: ModelDescriptor(
                id: "mlx-community/Voxtral-4B-TTS-2603-mlx-4bit",
                familyID: .voxtralTTS,
                displayName: "Voxtral",
                domain: .tts,
                capabilities: [.speechSynthesis, .voiceCloning]
            ),
            familyID: .voxtralTTS,
            installState: .installed,
            providerName: "Valar",
            providerURL: nil,
            sourceKind: .localFile,
            isRecommended: false,
            manifestPath: "/tmp/voxtral/manifest.json",
            installedPath: "/tmp/voxtral",
            artifactCount: 5,
            supportedBackends: [.mlx],
            notes: nil,
            cachedOnDisk: false
        )

        let ids = ValarRuntime.warmStartCandidateIDs(from: [voxtral, asr, customVoice, base])

        XCTAssertEqual(ids, [base.id, asr.id])
    }

    func testRuntimeConfigurationParsesWarmStartEnvironment() {
        let configuration = RuntimeConfiguration.configured(from: [
            RuntimeConfiguration.prewarmEnvVarName: "1",
            RuntimeConfiguration.warmStartModelsEnvVarName: " mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16 , mlx-community/Qwen3-ASR-0.6B-8bit ",
            RuntimeConfiguration.idleTrimSettleGraceSecondsEnvVarName: "12.5",
            RuntimeConfiguration.idleTrimRecentUseGraceSecondsEnvVarName: "34"
        ])

        XCTAssertEqual(configuration.warmPolicy, .eager)
        XCTAssertEqual(
            configuration.warmStartModelIDs,
            [
                ModelIdentifier("mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"),
                ModelIdentifier("mlx-community/Qwen3-ASR-0.6B-8bit"),
            ]
        )
        XCTAssertEqual(configuration.idleTrimSettleGraceSeconds, 12.5)
        XCTAssertEqual(configuration.idleTrimRecentUseGraceSeconds, 34)
    }

    func testRuntimeConfigurationCanDisablePrewarmFromEnvironment() {
        let configuration = RuntimeConfiguration.configured(from: [
            RuntimeConfiguration.prewarmEnvVarName: "0"
        ])

        XCTAssertEqual(configuration.warmPolicy, .lazy)
        XCTAssertNil(configuration.warmStartModelIDs)
    }

    func testRuntimeConfigurationDefaultsToLazyWarmPolicy() {
        let configuration = RuntimeConfiguration.configured(from: [:])

        XCTAssertEqual(configuration.warmPolicy, .lazy)
        XCTAssertNil(configuration.warmStartModelIDs)
        XCTAssertEqual(
            configuration.idleTrimSettleGraceSeconds,
            RuntimeConfiguration.defaultIdleTrimSettleGraceSeconds
        )
        XCTAssertEqual(
            configuration.idleTrimRecentUseGraceSeconds,
            RuntimeConfiguration.defaultIdleTrimRecentUseGraceSeconds
        )
    }

    func testRuntimeConfigurationHonorsExplicitWarmPolicyEnvironment() {
        let configuration = RuntimeConfiguration.configured(from: [
            RuntimeConfiguration.warmPolicyEnvVarName: "eager",
            RuntimeConfiguration.prewarmEnvVarName: "0",
            RuntimeConfiguration.warmStartModelsEnvVarName: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"
        ])

        XCTAssertEqual(configuration.warmPolicy, .eager)
        XCTAssertEqual(
            configuration.warmStartModelIDs,
            [ModelIdentifier("mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16")]
        )
    }

    func testModelRegistryPrefersMeasuredResidentBytesOverEstimates() async {
        let descriptorA = ModelDescriptor(
            id: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
            displayName: "Qwen3 TTS Base",
            domain: .tts,
            capabilities: [.speechSynthesis]
        )
        let descriptorB = ModelDescriptor(
            id: "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16",
            displayName: "Qwen3 TTS VoiceDesign",
            domain: .tts,
            capabilities: [.speechSynthesis, .voiceDesign]
        )

        let registry = ModelRegistry(
            configuration: RuntimeConfiguration(maxResidentBytes: 2_500, maxResidentModels: 2),
            enableMemoryPressureMonitoring: false
        )
        await registry.register(descriptorA, estimatedResidentBytes: 2_048, runtimeConfiguration: nil)
        await registry.register(descriptorB, estimatedResidentBytes: 2_048, runtimeConfiguration: nil)

        await registry.markState(.resident, for: descriptorA.id)
        await registry.setMeasuredResidentBytes(128, for: descriptorA.id)
        await registry.markState(.resident, for: descriptorB.id)
        await registry.setMeasuredResidentBytes(256, for: descriptorB.id)

        let residentBytes = await registry.residentBytes()
        let snapshots = await registry.snapshots()

        XCTAssertEqual(residentBytes, 384)
        XCTAssertEqual(snapshots.first { $0.descriptor.id == descriptorA.id }?.actualResidentBytes, 128)
        XCTAssertEqual(snapshots.first { $0.descriptor.id == descriptorB.id }?.actualResidentBytes, 256)
    }

    func testModelRegistryEvictsLeastRecentlyUsedModelWhenByteBudgetExceeded() async {
        let descriptorA = ModelDescriptor(
            id: "model-a",
            displayName: "Model A",
            domain: .tts,
            capabilities: [.speechSynthesis]
        )
        let descriptorB = ModelDescriptor(
            id: "model-b",
            displayName: "Model B",
            domain: .tts,
            capabilities: [.speechSynthesis]
        )
        let descriptorC = ModelDescriptor(
            id: "model-c",
            displayName: "Model C",
            domain: .tts,
            capabilities: [.speechSynthesis]
        )

        let registry = ModelRegistry(
            configuration: RuntimeConfiguration(maxResidentBytes: 300, maxResidentModels: 3),
            enableMemoryPressureMonitoring: false
        )
        await registry.register(descriptorA, estimatedResidentBytes: 150, runtimeConfiguration: nil)
        await registry.register(descriptorB, estimatedResidentBytes: 150, runtimeConfiguration: nil)
        await registry.register(descriptorC, estimatedResidentBytes: 150, runtimeConfiguration: nil)

        await registry.markState(.resident, for: descriptorA.id)
        await registry.setMeasuredResidentBytes(150, for: descriptorA.id)
        await registry.markState(.resident, for: descriptorB.id)
        await registry.setMeasuredResidentBytes(150, for: descriptorB.id)
        await registry.touch(descriptorA.id)
        await registry.markState(.resident, for: descriptorC.id)
        await registry.setMeasuredResidentBytes(150, for: descriptorC.id)

        let residentModels = await registry.residentModels()
        let snapshots = await registry.snapshots()
        let evictions = await registry.evictionEvents()

        XCTAssertEqual(residentModels.map(\.id), [descriptorA.id, descriptorC.id])
        XCTAssertEqual(snapshots.first { $0.descriptor.id == descriptorB.id }?.state, .unloaded)
        XCTAssertEqual(evictions.last?.descriptor.id, descriptorB.id)
        XCTAssertEqual(evictions.last?.trigger, .budgetExceeded)
    }

    func testModelRegistryEvictsSingleOversizedAutomaticModel() async {
        let descriptor = ModelDescriptor(
            id: "model-oversized",
            displayName: "Oversized",
            domain: .tts,
            capabilities: [.speechSynthesis]
        )

        let registry = ModelRegistry(
            configuration: RuntimeConfiguration(maxResidentBytes: 128, maxResidentModels: 1),
            enableMemoryPressureMonitoring: false
        )
        await registry.register(descriptor, estimatedResidentBytes: 256, runtimeConfiguration: nil)

        await registry.markState(.resident, for: descriptor.id)
        await registry.setMeasuredResidentBytes(256, for: descriptor.id)

        let snapshots = await registry.snapshots()
        let evictions = await registry.evictionEvents()
        let residentModels = await registry.residentModels()

        XCTAssertEqual(snapshots.first { $0.descriptor.id == descriptor.id }?.state, .unloaded)
        XCTAssertEqual(evictions.last?.descriptor.id, descriptor.id)
        XCTAssertEqual(residentModels, [])
    }

    func testModelRegistryProtectsPinnedModelsFromAutoEviction() async {
        let pinned = ModelDescriptor(
            id: "model-pinned",
            displayName: "Pinned",
            domain: .tts,
            capabilities: [.speechSynthesis]
        )
        let automaticA = ModelDescriptor(
            id: "model-auto-a",
            displayName: "Automatic A",
            domain: .tts,
            capabilities: [.speechSynthesis]
        )
        let automaticB = ModelDescriptor(
            id: "model-auto-b",
            displayName: "Automatic B",
            domain: .tts,
            capabilities: [.speechSynthesis]
        )

        let registry = ModelRegistry(
            configuration: RuntimeConfiguration(maxResidentBytes: 300, maxResidentModels: 3),
            enableMemoryPressureMonitoring: false
        )
        await registry.register(
            pinned,
            estimatedResidentBytes: 150,
            runtimeConfiguration: ModelRuntimeConfiguration(backendKind: .mlx, residencyPolicy: .pinned)
        )
        await registry.register(automaticA, estimatedResidentBytes: 150, runtimeConfiguration: nil)
        await registry.register(automaticB, estimatedResidentBytes: 150, runtimeConfiguration: nil)

        await registry.markState(.resident, for: pinned.id)
        await registry.setMeasuredResidentBytes(150, for: pinned.id)
        await registry.markState(.resident, for: automaticA.id)
        await registry.setMeasuredResidentBytes(150, for: automaticA.id)
        await registry.markState(.resident, for: automaticB.id)
        await registry.setMeasuredResidentBytes(150, for: automaticB.id)

        let residentModels = await registry.residentModels()
        let evictions = await registry.evictionEvents()

        XCTAssertTrue(residentModels.map(\.id).contains(pinned.id))
        XCTAssertFalse(evictions.contains { $0.descriptor.id == pinned.id })
        XCTAssertEqual(evictions.last?.descriptor.id, automaticA.id)
    }

    func testModelRegistryProactivelyEvictsOnMemoryPressureAndLogsEvent() async {
        let descriptorA = ModelDescriptor(
            id: "model-pressure-a",
            displayName: "Pressure A",
            domain: .tts,
            capabilities: [.speechSynthesis]
        )
        let descriptorB = ModelDescriptor(
            id: "model-pressure-b",
            displayName: "Pressure B",
            domain: .tts,
            capabilities: [.speechSynthesis]
        )
        let logStream = AsyncStream<String>.makeStream()
        var iterator = logStream.stream.makeAsyncIterator()

        let registry = ModelRegistry(
            configuration: RuntimeConfiguration(maxResidentBytes: 1_024, maxResidentModels: 3),
            enableMemoryPressureMonitoring: false,
            evictionLogger: { message in
                logStream.continuation.yield(message)
            }
        )
        await registry.register(descriptorA, estimatedResidentBytes: 200, runtimeConfiguration: nil)
        await registry.register(descriptorB, estimatedResidentBytes: 200, runtimeConfiguration: nil)

        await registry.markState(.resident, for: descriptorA.id)
        await registry.setMeasuredResidentBytes(200, for: descriptorA.id)
        await registry.markState(.resident, for: descriptorB.id)
        await registry.setMeasuredResidentBytes(200, for: descriptorB.id)

        await registry.handleMemoryPressure(.warning)

        let residentModels = await registry.residentModels()
        let evictions = await registry.evictionEvents()
        let logMessage = await iterator.next()

        XCTAssertEqual(residentModels.map(\.id), [descriptorB.id])
        XCTAssertEqual(evictions.last?.descriptor.id, descriptorA.id)
        XCTAssertEqual(evictions.last?.trigger, .memoryPressureWarning)
        XCTAssertTrue(logMessage?.contains(descriptorA.id.rawValue) == true)
    }

    func testModelRegistryHonorsResidentBudget() async {
        let descriptorA = ModelDescriptor(
            id: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
            displayName: "Qwen3 TTS Base",
            domain: .tts,
            capabilities: [.speechSynthesis, .tokenization]
        )
        let descriptorB = ModelDescriptor(
            id: "mlx-community/Whisper-Large-v3",
            displayName: "Whisper Large v3",
            domain: .stt,
            capabilities: [.speechRecognition]
        )

        let registry = ModelRegistry(configuration: RuntimeConfiguration(maxResidentModels: 1), knownModels: [descriptorA, descriptorB])
        await registry.markState(.resident, for: descriptorA.id)
        await registry.markState(.resident, for: descriptorB.id)

        let snapshots = await registry.snapshots()
        let residentModels = await registry.residentModels()
        XCTAssertEqual(snapshots.first { $0.descriptor.id == descriptorA.id }?.state, .unloaded)
        XCTAssertEqual(snapshots.first { $0.descriptor.id == descriptorB.id }?.state, .resident)
        XCTAssertEqual(residentModels.map(\.id), [descriptorB.id])
    }

    func testRenderQueueOrdersJobsByPriority() async {
        let queue = RenderQueue()
        let projectID = UUID()
        let chapterID = UUID()

        let low = await queue.enqueue(
            projectID: projectID,
            modelID: "model-low",
            chapterIDs: [chapterID],
            outputFileName: "low.wav",
            priority: 1,
            title: "low"
        )
        let high = await queue.enqueue(
            projectID: projectID,
            modelID: "model-high",
            chapterIDs: [UUID()],
            outputFileName: "high.wav",
            priority: 4,
            title: "high"
        )

        let nextJob = await queue.nextJob()
        XCTAssertEqual(nextJob?.id, high.id)
        XCTAssertEqual(nextJob?.outputFileName, "high.wav")
        await queue.cancel(low.id)
        let cancelled = await queue.job(id: low.id)
        XCTAssertEqual(cancelled?.state, .cancelled)
        XCTAssertEqual(cancelled?.chapterIDs, [chapterID])
    }

    func testRenderQueuePublishesJobUpdates() async {
        let queue = RenderQueue()
        let stream = await queue.updates()
        var iterator = stream.makeAsyncIterator()

        let initial = await iterator.next()
        XCTAssertEqual(initial, [])

        let job = await queue.enqueue(
            projectID: UUID(),
            modelID: "model-high",
            chapterIDs: [UUID()],
            outputFileName: "chapter-1.wav",
            priority: 1,
            title: "Chapter 1"
        )
        let enqueued = await iterator.next()

        XCTAssertEqual(enqueued?.count, 1)
        XCTAssertEqual(enqueued?.first?.id, job.id)
        XCTAssertEqual(enqueued?.first?.outputFileName, "chapter-1.wav")

        await queue.transition(job.id, to: .running, progress: 0.5)
        let running = await iterator.next()
        XCTAssertEqual(running?.first?.state, .running)
        XCTAssertEqual(running?.first?.progress, 0.5)
    }

    func testRenderQueuePublishesReplacementUpdates() async {
        let queue = RenderQueue()
        let projectID = UUID()
        let stream = await queue.updates()
        var iterator = stream.makeAsyncIterator()

        let firstValue = await iterator.next()
        XCTAssertEqual(firstValue, [])

        _ = await queue.enqueue(
            projectID: projectID,
            modelID: "model-original",
            chapterIDs: [UUID()],
            outputFileName: "original.wav",
            priority: 1,
            title: "Original"
        )
        _ = await iterator.next()

        let replacement = RenderJob(
            projectID: projectID,
            modelID: "model-replacement",
            chapterIDs: [UUID()],
            outputFileName: "replacement.wav",
            priority: 3,
            title: "Replacement"
        )

        await queue.replaceJobs(for: projectID, with: [replacement])
        let updated = await iterator.next()

        XCTAssertEqual(updated?.map(\.id), [replacement.id])
        XCTAssertEqual(updated?.first?.outputFileName, "replacement.wav")
    }

    func testProjectAndVoiceServicesTrackRecords() async throws {
        let projects = ProjectStore(paths: ValarAppPaths())
        let project = try await projects.create(title: "Book One")
        let bundleURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
            .appendingPathExtension("valarproject")
        await projects.updateBundleURL(bundleURL, for: project.id)
        let allProjects = await projects.allProjects()
        let bundleLocation = await projects.bundleLocation(for: project.id)
        XCTAssertEqual(allProjects.count, 1)
        XCTAssertEqual(bundleLocation?.bundleURL, bundleURL)

        let voices = VoiceLibraryStore()
        _ = try await voices.save(VoiceLibraryRecord(label: "Narrator", modelID: "model"))
        let voiceList = await voices.list()
        XCTAssertEqual(voiceList.count, 1)
    }

    func testDictationLifecycleProducesTranscript() async {
        let dictation = DictationService()
        let session = await dictation.start(modelID: "model", languageHint: "en")
        await dictation.append("Hello", to: session.id)
        await dictation.append("world", to: session.id)

        let snapshotsBeforeFinalize = await dictation.snapshot()
        let transcript = await dictation.finalize(sessionID: session.id)
        let snapshotsAfterFinalize = await dictation.snapshot()

        XCTAssertEqual(snapshotsBeforeFinalize.count, 1)
        XCTAssertEqual(transcript, "Hello world")
        XCTAssertEqual(snapshotsAfterFinalize.count, 0)
    }

    func testTranslationServiceUsesProvider() async throws {
        let service = TranslationService(provider: EchoTranslator())
        let translated = try await service.translate(
            TranslationRequest(sourceLanguage: "en", targetLanguage: "ja", text: "hello")
        )
        XCTAssertEqual(translated, "[ja] hello")
    }

    func testRenderJobFallsBackToUUIDFileNameWhenOutputFileNameIsBlank() {
        let id = UUID(uuidString: "AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE")!
        let job = RenderJob(
            id: id,
            projectID: UUID(uuidString: "11111111-2222-3333-4444-555555555555")!,
            modelID: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
            chapterIDs: [],
            outputFileName: "   "
        )

        XCTAssertEqual(job.outputFileName, "\(id.uuidString).wav")
    }

    func testRenderJobTrimsExplicitOutputFileName() {
        let job = RenderJob(
            projectID: UUID(uuidString: "11111111-2222-3333-4444-555555555555")!,
            modelID: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
            chapterIDs: [],
            outputFileName: "  chapter-01.wav  "
        )

        XCTAssertEqual(job.outputFileName, "chapter-01.wav")
    }

    func testReconcileLocalModelPackStateRemovesStaleInstalledRecords() async throws {
        let baseURL = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        let runtime = try ValarRuntime(
            paths: ValarAppPaths(baseURL: baseURL),
            runtimeConfiguration: RuntimeConfiguration(),
            inferenceBackend: LocalStubInferenceBackend()
        )
        let modelID = ValarRuntime.defaultVoiceCloneRuntimeModelID
        let manifest = try await runtime.modelCatalog.installationManifest(for: modelID)
        XCTAssertNotNil(manifest)
        let unwrappedManifest = try XCTUnwrap(manifest)

        _ = try await runtime.modelPackRegistry.install(
            manifest: unwrappedManifest,
            sourceKind: .remoteURL,
            sourceLocation: "https://example.com/\(modelID.rawValue)",
            notes: nil
        )

        let installedBefore = try await runtime.modelPackRegistry.installedRecord(for: modelID.rawValue)
        XCTAssertNotNil(installedBefore)

        let report = try await runtime.reconcileLocalModelPackState()

        XCTAssertEqual(report.removedStaleModelIDs, [modelID])
        let installedAfter = try await runtime.modelPackRegistry.installedRecord(for: modelID.rawValue)
        XCTAssertNil(installedAfter)
    }

    func testReconcileLocalModelPackStateReportsOrphanedModelPackDirectories() async throws {
        let fileManager = FileManager.default
        let baseURL = fileManager.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        let runtime = try ValarRuntime(
            paths: ValarAppPaths(baseURL: baseURL),
            runtimeConfiguration: RuntimeConfiguration(),
            inferenceBackend: LocalStubInferenceBackend()
        )
        let modelID = ValarRuntime.defaultVoiceCloneRuntimeModelID
        let manifest = try await runtime.modelCatalog.installationManifest(for: modelID)
        XCTAssertNotNil(manifest)
        let unwrappedManifest = try XCTUnwrap(manifest)
        let packDirectory = try runtime.paths.modelPackDirectory(
            familyID: unwrappedManifest.familyID,
            modelID: unwrappedManifest.modelID
        )

        try fileManager.createDirectory(at: packDirectory, withIntermediateDirectories: true)
        let manifestURL = packDirectory.appendingPathComponent("manifest.json", isDirectory: false)
        let manifestData = try JSONEncoder().encode(unwrappedManifest)
        try manifestData.write(to: manifestURL)

        let report = try await runtime.reconcileLocalModelPackState()

        XCTAssertEqual(report.orphanedModelPackPaths, [packDirectory.standardizedFileURL.path])
    }

    func testDaemonReadyStatusIncludesInstalledAvailabilityWhenModelsAreResident() async throws {
        let baseURL = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        let paths = ValarAppPaths(baseURL: baseURL)
        let runtime = try ValarRuntime(
            paths: paths,
            runtimeConfiguration: RuntimeConfiguration(),
            inferenceBackend: LocalStubInferenceBackend()
        )

        let baseID = ValarRuntime.defaultVoiceCloneRuntimeModelID
        let asrID = ModelIdentifier("mlx-community/Qwen3-ASR-0.6B-8bit")
        let baseManifestOptional = try await runtime.modelCatalog.installationManifest(for: baseID)
        let asrManifestOptional = try await runtime.modelCatalog.installationManifest(for: asrID)
        let baseManifest = try XCTUnwrap(baseManifestOptional)
        let asrManifest = try XCTUnwrap(asrManifestOptional)

        _ = try await runtime.modelPackRegistry.install(
            manifest: baseManifest,
            sourceKind: .remoteURL,
            sourceLocation: "https://example.com/\(baseID.rawValue)",
            notes: nil
        )
        try materializeInstalledPack(paths: paths, manifest: baseManifest)

        _ = try await runtime.modelPackRegistry.install(
            manifest: asrManifest,
            sourceKind: .remoteURL,
            sourceLocation: "https://example.com/\(asrID.rawValue)",
            notes: nil
        )
        try materializeInstalledPack(paths: paths, manifest: asrManifest)

        let catalog = try await runtime.modelCatalog.refresh()
        let baseModel = try XCTUnwrap(catalog.first(where: { $0.id == baseID }))
        await runtime.modelRegistry.register(baseModel.descriptor)
        await runtime.modelRegistry.markState(.resident, for: baseID)

        let ready = await runtime.daemonReadyStatus()

        XCTAssertTrue(ready.ready)
        XCTAssertTrue(ready.ttsReady)
        XCTAssertTrue(ready.asrReady)
        XCTAssertFalse(ready.alignmentReady)
        XCTAssertEqual(ready.readinessMode, .resident)
        XCTAssertTrue(ready.residentTTSReady)
        XCTAssertFalse(ready.residentASRReady)
        XCTAssertFalse(ready.residentAlignmentReady)
        XCTAssertEqual(Set(ready.installedModels), Set([baseID.rawValue, asrID.rawValue]))
    }

    func testDaemonReadyStatusRefreshesInstalledModelsWithoutResidentLoads() async throws {
        let baseURL = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        let paths = ValarAppPaths(baseURL: baseURL)
        let runtime = try ValarRuntime(
            paths: paths,
            runtimeConfiguration: RuntimeConfiguration(),
            inferenceBackend: LocalStubInferenceBackend()
        )

        let baseID = ValarRuntime.defaultVoiceCloneRuntimeModelID
        let baseManifestOptional = try await runtime.modelCatalog.installationManifest(for: baseID)
        let baseManifest = try XCTUnwrap(baseManifestOptional)

        _ = try await runtime.modelPackRegistry.install(
            manifest: baseManifest,
            sourceKind: .remoteURL,
            sourceLocation: "https://example.com/\(baseID.rawValue)",
            notes: nil
        )
        try materializeInstalledPack(paths: paths, manifest: baseManifest)

        let ready = await runtime.daemonReadyStatus()

        XCTAssertTrue(ready.ready)
        XCTAssertTrue(ready.ttsReady)
        XCTAssertFalse(ready.asrReady)
        XCTAssertEqual(ready.readinessMode, .loadOnDemand)
        XCTAssertFalse(ready.residentTTSReady)
        XCTAssertFalse(ready.residentASRReady)
        XCTAssertEqual(ready.installedModels, [baseID.rawValue])
    }

    func testWarmStartCatalogModelsUsesRefreshedInstalledState() async throws {
        let baseURL = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        let paths = ValarAppPaths(baseURL: baseURL)
        let runtime = try ValarRuntime(
            paths: paths,
            runtimeConfiguration: RuntimeConfiguration(),
            inferenceBackend: LocalStubInferenceBackend()
        )

        let baseID = ValarRuntime.defaultVoiceCloneRuntimeModelID
        let asrID = ModelIdentifier("mlx-community/Qwen3-ASR-0.6B-8bit")
        let baseManifestOptional = try await runtime.modelCatalog.installationManifest(for: baseID)
        let asrManifestOptional = try await runtime.modelCatalog.installationManifest(for: asrID)
        let baseManifest = try XCTUnwrap(baseManifestOptional)
        let asrManifest = try XCTUnwrap(asrManifestOptional)

        _ = try await runtime.modelPackRegistry.install(
            manifest: baseManifest,
            sourceKind: .remoteURL,
            sourceLocation: "https://example.com/\(baseID.rawValue)",
            notes: nil
        )
        try materializeInstalledPack(paths: paths, manifest: baseManifest)

        _ = try await runtime.modelPackRegistry.install(
            manifest: asrManifest,
            sourceKind: .remoteURL,
            sourceLocation: "https://example.com/\(asrID.rawValue)",
            notes: nil
        )
        try materializeInstalledPack(paths: paths, manifest: asrManifest)

        let warmModels = await runtime.warmStartCatalogModels()

        XCTAssertEqual(
            warmModels.map(\.id),
            [baseID, asrID]
        )
    }

    func testPrewarmInstalledModelsMarksWarmSetResidentInSharedRegistry() async throws {
        let baseURL = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        let paths = ValarAppPaths(baseURL: baseURL)
        let runtime = try ValarRuntime(
            paths: paths,
            runtimeConfiguration: RuntimeConfiguration(warmPolicy: .eager),
            inferenceBackend: WarmStartStubInferenceBackend()
        )

        let baseID = ValarRuntime.defaultVoiceCloneRuntimeModelID
        let asrID = ModelIdentifier("mlx-community/Qwen3-ASR-0.6B-8bit")
        for identifier in [baseID, asrID] {
            let manifestOptional = try await runtime.modelCatalog.installationManifest(for: identifier)
            let manifest = try XCTUnwrap(manifestOptional)
            _ = try await runtime.modelPackRegistry.install(
                manifest: manifest,
                sourceKind: .remoteURL,
                sourceLocation: "https://example.com/\(identifier.rawValue)",
                notes: nil
            )
            try materializeInstalledPack(paths: paths, manifest: manifest)
        }

        await runtime.prewarmInstalledModels()

        let snapshots = await runtime.modelRegistry.snapshots()
        let residentIDs = snapshots
            .filter { $0.state == .resident }
            .map(\.descriptor.id)
            .sorted { $0.rawValue < $1.rawValue }
        XCTAssertEqual(residentIDs, [asrID, baseID].sorted { $0.rawValue < $1.rawValue })
        XCTAssertTrue(
            snapshots
                .filter { residentIDs.contains($0.descriptor.id) }
                .allSatisfy { $0.activeSessionCount == 0 }
        )
    }

    func testRestoreMissingWarmResidentsRehydratesEvictedWarmModel() async throws {
        let baseURL = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        let paths = ValarAppPaths(baseURL: baseURL)
        let runtime = try ValarRuntime(
            paths: paths,
            runtimeConfiguration: RuntimeConfiguration(warmPolicy: .eager),
            inferenceBackend: WarmStartStubInferenceBackend()
        )

        let baseID = ValarRuntime.defaultVoiceCloneRuntimeModelID
        let asrID = ModelIdentifier("mlx-community/Qwen3-ASR-0.6B-8bit")
        for identifier in [baseID, asrID] {
            let manifestOptional = try await runtime.modelCatalog.installationManifest(for: identifier)
            let manifest = try XCTUnwrap(manifestOptional)
            _ = try await runtime.modelPackRegistry.install(
                manifest: manifest,
                sourceKind: .remoteURL,
                sourceLocation: "https://example.com/\(identifier.rawValue)",
                notes: nil
            )
            try materializeInstalledPack(paths: paths, manifest: manifest)
        }

        await runtime.prewarmInstalledModels()
        let evictedBase = await runtime.modelRegistry.evictResident(baseID, trigger: .idleTrim)
        XCTAssertTrue(evictedBase)

        let residentAfterEviction = await runtime.modelRegistry.snapshots()
            .filter { $0.state == .resident }
            .map(\.descriptor.id)
        XCTAssertFalse(residentAfterEviction.contains(baseID))
        XCTAssertTrue(residentAfterEviction.contains(asrID))

        await runtime.restoreMissingWarmResidentsIfNeeded()

        let residentAfterRestore = await runtime.modelRegistry.snapshots()
            .filter { $0.state == .resident }
            .map(\.descriptor.id)
            .sorted { $0.rawValue < $1.rawValue }
        XCTAssertEqual(residentAfterRestore, [asrID, baseID].sorted { $0.rawValue < $1.rawValue })
    }
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
