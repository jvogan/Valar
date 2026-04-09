import XCTest
import ValarAudio
import ValarCore
import ValarModelKit
import ValarPersistence
@testable import ValarTTSMacApp

private actor RenderInvocationTracker {
    struct Invocation: Sendable, Equatable {
        let text: String
        let options: RenderSynthesisOptions
    }

    private var invocations: [Invocation] = []
    private var activeCount = 0
    private var maxConcurrentCount = 0

    func begin(_ text: String, options: RenderSynthesisOptions = RenderSynthesisOptions()) {
        invocations.append(Invocation(text: text, options: options))
        activeCount += 1
        maxConcurrentCount = max(maxConcurrentCount, activeCount)
    }

    func end() {
        activeCount = max(0, activeCount - 1)
    }

    func recordedTexts() -> [String] {
        invocations.map(\.text)
    }

    func recordedOptions() -> [RenderSynthesisOptions] {
        invocations.map(\.options)
    }

    func recordedMaxConcurrency() -> Int {
        maxConcurrentCount
    }
}

private func makeFakePCMChunk(samples: [Float] = [0, 0.25, -0.25, 0.1]) -> AudioChunk {
    AudioChunk(samples: samples, sampleRate: 24_000)
}

private actor StubModelPackRegistry: ModelPackManaging {
    private var manifestsByModelID: [String: ValarPersistence.ModelPackManifest]
    private var installedRecordsByModelID: [String: InstalledModelRecord]
    private var supportedRecordsByModelID: [String: SupportedModelCatalogRecord]

    init(
        manifests: [String: ValarPersistence.ModelPackManifest] = [:],
        installedRecords: [InstalledModelRecord] = [],
        supportedRecords: [SupportedModelCatalogRecord] = []
    ) {
        self.manifestsByModelID = manifests
        self.installedRecordsByModelID = Dictionary(uniqueKeysWithValues: installedRecords.map { ($0.modelID, $0) })
        self.supportedRecordsByModelID = Dictionary(uniqueKeysWithValues: supportedRecords.map { ($0.modelID, $0) })
    }

    func manifest(for modelID: String) async throws -> ValarPersistence.ModelPackManifest? {
        manifestsByModelID[modelID]
    }

    func installedRecord(for modelID: String) async throws -> InstalledModelRecord? {
        installedRecordsByModelID[modelID]
    }

    func receipts() async throws -> [ModelInstallReceipt] {
        installedRecordsByModelID.values.map { record in
            ModelInstallReceipt(
                modelID: record.modelID,
                familyID: record.familyID,
                sourceKind: record.sourceKind,
                sourceLocation: record.installedPath,
                installDate: record.installDate,
                installedModelPath: record.installedPath,
                manifestPath: record.manifestPath,
                checksum: record.checksum,
                artifactCount: record.artifactCount,
                notes: nil
            )
        }
    }

    func supportedModels() async throws -> [SupportedModelCatalogRecord] {
        Array(supportedRecordsByModelID.values)
    }

    func supportedModel(for modelID: String) async throws -> SupportedModelCatalogRecord? {
        supportedRecordsByModelID[modelID]
    }

    func registerSupported(_ record: SupportedModelCatalogRecord) async throws {
        supportedRecordsByModelID[record.modelID] = record
    }

    func install(
        manifest: ValarPersistence.ModelPackManifest,
        sourceKind: ModelPackSourceKind,
        sourceLocation: String,
        notes: String?
    ) async throws -> ModelInstallReceipt {
        manifestsByModelID[manifest.modelID] = manifest
        let installedRecord = InstalledModelRecord(
            familyID: manifest.familyID,
            modelID: manifest.modelID,
            displayName: manifest.displayName,
            installedPath: sourceLocation,
            manifestPath: sourceLocation + "/manifest.json",
            artifactCount: manifest.artifactSpecs.count,
            sourceKind: sourceKind
        )
        installedRecordsByModelID[manifest.modelID] = installedRecord
        return ModelInstallReceipt(
            modelID: manifest.modelID,
            familyID: manifest.familyID,
            sourceKind: sourceKind,
            sourceLocation: sourceLocation,
            installDate: .now,
            installedModelPath: "",
            manifestPath: "",
            checksum: nil,
            artifactCount: manifest.artifactSpecs.count,
            notes: notes
        )
    }

    func uninstall(modelID: String) async throws -> InstalledModelRecord? {
        installedRecordsByModelID.removeValue(forKey: modelID)
    }

    func ledgerEntries() async throws -> [ModelInstallLedgerEntry] { [] }
}

private actor StubProjectStore: ProjectStoring {
    func create(title: String, notes: String?) async throws -> ProjectRecord {
        ProjectRecord(title: title, notes: notes)
    }
    func update(_ project: ProjectRecord) async {}
    func addChapter(_ chapter: ChapterRecord) async {}
    func updateChapter(_ chapter: ChapterRecord) async {}
    func removeChapter(_ id: UUID, from projectID: UUID) async {}
    func addRenderJob(_ job: RenderJobRecord) async {}
    func addExport(_ export: ExportRecord) async {}
    func addSpeaker(_ speaker: ProjectSpeakerRecord) async {}
    func removeSpeaker(_ id: UUID, from projectID: UUID) async {}
    func restore(
        project: ProjectRecord,
        chapters: [ChapterRecord],
        renderJobs: [RenderJobRecord],
        exports: [ExportRecord],
        speakers: [ProjectSpeakerRecord]
    ) async {}
    func allProjects() async -> [ProjectRecord] { [] }
    func chapters(for projectID: UUID) async -> [ChapterRecord] { [] }
    func renderJobs(for projectID: UUID) async -> [RenderJobRecord] { [] }
    func exports(for projectID: UUID) async -> [ExportRecord] { [] }
    func speakers(for projectID: UUID) async -> [ProjectSpeakerRecord] { [] }
    func bundleLocation(for projectID: UUID) async -> ValarProjectBundleLocation? { nil }
    func updateBundleURL(_ bundleURL: URL?, for projectID: UUID) async {}
    func remove(id: UUID) async {}
}

private actor StubVoiceLibraryStore: VoiceLibraryStoring {
    func save(_ voice: VoiceLibraryRecord) async throws -> VoiceLibraryRecord { voice }
    func list() async -> [VoiceLibraryRecord] { [] }
    func delete(_ id: UUID) async throws {}
}

private actor SeededVoiceLibraryStore: VoiceLibraryStoring {
    private var voices: [VoiceLibraryRecord]

    init(voices: [VoiceLibraryRecord]) {
        self.voices = voices
    }

    func save(_ voice: VoiceLibraryRecord) async throws -> VoiceLibraryRecord {
        if let index = voices.firstIndex(where: { $0.id == voice.id }) {
            voices[index] = voice
        } else {
            voices.append(voice)
        }
        return voice
    }

    func list() async -> [VoiceLibraryRecord] {
        voices
    }

    func delete(_ id: UUID) async throws {
        voices.removeAll { $0.id == id }
    }
}

private struct StubTranslationProvider: TranslationProvider {
    func translate(_ request: TranslationRequest) async throws -> String {
        request.text
    }
}

private actor StubProjectExporter: ProjectAudioExporting {
    func exportProjectAudio(
        projectID: UUID,
        modelID: ModelIdentifier,
        synthesisOptions: RenderSynthesisOptions,
        format: ProjectExportFormat,
        mode: ProjectExportMode,
        destinationURL: URL,
        onProgress: @escaping @Sendable (ExportProgress) -> Void
    ) async throws -> ProjectAudioExportResult {
        _ = synthesisOptions
        _ = onProgress
        return ProjectAudioExportResult(files: [], exportedChapterCount: 0)
    }
}

private actor RequestRecorder {
    private var request: SpeechSynthesisRequest?

    func record(_ request: SpeechSynthesisRequest) {
        self.request = request
    }

    func latestRequest() -> SpeechSynthesisRequest? {
        request
    }
}

private struct StubStreamingTTSModel: TextToSpeechWorkflow {
    let descriptor: ModelDescriptor
    let backendKind: BackendKind = .mlx
    let streamFactory: @Sendable () -> AsyncThrowingStream<AudioChunk, Error>
    let requestHandler: (@Sendable (SpeechSynthesisRequest) -> Void)?

    init(
        descriptor: ModelDescriptor,
        streamFactory: @escaping @Sendable () -> AsyncThrowingStream<AudioChunk, Error>,
        requestHandler: (@Sendable (SpeechSynthesisRequest) -> Void)? = nil
    ) {
        self.descriptor = descriptor
        self.streamFactory = streamFactory
        self.requestHandler = requestHandler
    }

    func synthesize(
        request: SpeechSynthesisRequest,
        in session: ModelRuntimeSession
    ) async throws -> AudioChunk {
        requestHandler?(request)
        _ = session

        var iterator = streamFactory().makeAsyncIterator()
        guard let firstChunk = try await iterator.next() else {
            return AudioChunk(samples: [], sampleRate: descriptor.defaultSampleRate ?? 24_000)
        }
        return firstChunk
    }

    func synthesizeStream(request: SpeechSynthesisRequest) async throws -> AsyncThrowingStream<AudioChunk, Error> {
        requestHandler?(request)
        return streamFactory()
    }
}

private actor RecognitionAlignmentRequestRecorder {
    private var transcriptionRequests: [SpeechRecognitionRequest] = []
    private var alignmentRequests: [ForcedAlignmentRequest] = []

    func recordTranscription(_ request: SpeechRecognitionRequest) {
        transcriptionRequests.append(request)
    }

    func recordAlignment(_ request: ForcedAlignmentRequest) {
        alignmentRequests.append(request)
    }

    func latestTranscriptionRequest() -> SpeechRecognitionRequest? {
        transcriptionRequests.last
    }

    func latestAlignmentRequest() -> ForcedAlignmentRequest? {
        alignmentRequests.last
    }
}

private struct StubRecognitionAlignmentModel: SpeechToTextWorkflow, ForcedAlignmentWorkflow {
    let descriptor: ModelDescriptor
    let backendKind: BackendKind = .mlx
    let requestRecorder: RecognitionAlignmentRequestRecorder?

    func transcribe(
        request: SpeechRecognitionRequest,
        in session: ModelRuntimeSession
    ) async throws -> RichTranscriptionResult {
        _ = session
        await requestRecorder?.recordTranscription(request)
        return RichTranscriptionResult(
            text: "Detected speech for \(request.model.rawValue)",
            language: request.languageHint,
            durationSeconds: request.audioChunk.map { Double($0.samples.count) / $0.sampleRate },
            segments: [TranscriptionSegment(text: "Detected speech")],
            backendMetadata: BackendMetadata(
                modelId: request.model.rawValue,
                backendKind: .mlx
            )
        )
    }

    func align(
        request: ForcedAlignmentRequest,
        in session: ModelRuntimeSession
    ) async throws -> ForcedAlignmentResponse {
        _ = session
        await requestRecorder?.recordAlignment(request)
        return ForcedAlignmentResponse(
            transcript: request.transcript,
            segments: [
                AlignmentToken(text: "Detected", startTime: 0, endTime: 0.4),
                AlignmentToken(text: "speech", startTime: 0.4, endTime: 0.8),
            ]
        )
    }
}

private actor StubInferenceBackend: InferenceBackend {
    nonisolated let backendKind: BackendKind = .mlx
    nonisolated let runtimeCapabilities = BackendCapabilities(
        features: [.streamingSynthesis],
        supportedFamilies: [.qwen3TTS, .soprano, .tadaTTS, .voxtralTTS]
    )

    private let model: any ValarModel

    init(model: any ValarModel) {
        self.model = model
    }

    func validate(requirement: BackendRequirement) async throws {
        _ = requirement
    }

    func loadModel(
        descriptor: ModelDescriptor,
        configuration: ModelRuntimeConfiguration
    ) async throws -> any ValarModel {
        _ = descriptor
        _ = configuration
        return model
    }

    func unloadModel(_ model: any ValarModel) async throws {
        _ = model
    }
}

private actor BlockingInferenceBackend: InferenceBackend {
    nonisolated let backendKind: BackendKind = .mlx
    nonisolated let runtimeCapabilities = BackendCapabilities(
        features: [.warmStart, .streamingSynthesis],
        supportedFamilies: [.qwen3TTS, .soprano, .tadaTTS]
    )

    private let model: any ValarModel
    private var didStartLoad = false
    private var didFinishLoad = false
    private var isReleased = false

    init(model: any ValarModel) {
        self.model = model
    }

    func validate(requirement: BackendRequirement) async throws {
        _ = requirement
    }

    func loadModel(
        descriptor: ModelDescriptor,
        configuration: ModelRuntimeConfiguration
    ) async throws -> any ValarModel {
        _ = descriptor
        _ = configuration

        didStartLoad = true
        if !isReleased {
            try await waitUntilReleased()
        }

        didFinishLoad = true
        return model
    }

    func unloadModel(_ model: any ValarModel) async throws {
        _ = model
    }

    func waitForLoadStart(timeoutNanoseconds: UInt64 = 2_000_000_000) async -> Bool {
        guard !didStartLoad else { return true }

        let deadline = DispatchTime.now().uptimeNanoseconds + timeoutNanoseconds
        while !didStartLoad {
            if DispatchTime.now().uptimeNanoseconds >= deadline {
                return false
            }
            try? await Task.sleep(nanoseconds: 10_000_000)
        }
        return true
    }

    func releaseLoad() {
        guard !isReleased else { return }
        isReleased = true
    }

    func hasFinishedLoad() -> Bool {
        didFinishLoad
    }

    private func waitUntilReleased() async throws {
        while !isReleased {
            try Task.checkCancellation()
            try await Task.sleep(nanoseconds: 10_000_000)
        }
    }
}

private actor StubAudioPlayer: AudioPlaying {
    private var playedBuffers: [AudioPCMBuffer] = []
    private var fedBuffers: [AudioPCMBuffer] = []
    private var fedSampleArrays: [[Float]] = []
    private var snapshots: [AudioPlaybackSnapshot]
    private var snapshotIndex = 0
    private(set) var finishStreamCallCount = 0
    private(set) var stopCallCount = 0
    private let playError: (any Error)?

    init(
        snapshots: [AudioPlaybackSnapshot],
        playError: (any Error)? = nil
    ) {
        self.snapshots = snapshots
        self.playError = playError
    }

    func play(_ buffer: AudioPCMBuffer) async throws {
        if let playError {
            throw playError
        }
        playedBuffers.append(buffer)
    }

    func feedChunk(_ buffer: AudioPCMBuffer) async throws {
        fedBuffers.append(buffer)
    }

    func feedSamples(_ samples: [Float], sampleRate: Double) async throws {
        fedSampleArrays.append(samples)
    }

    func finishStream() async {
        finishStreamCallCount += 1
    }

    func playbackSnapshot() async -> AudioPlaybackSnapshot {
        guard !snapshots.isEmpty else {
            return AudioPlaybackSnapshot(
                position: 0,
                queuedDuration: 0,
                isPlaying: false,
                isBuffering: false,
                didFinish: false
            )
        }

        let snapshot = snapshots[min(snapshotIndex, snapshots.count - 1)]
        if snapshotIndex < snapshots.count - 1 {
            snapshotIndex += 1
        }
        return snapshot
    }

    func stop() async {
        stopCallCount += 1
    }

    func recordedFeedCount() -> Int {
        fedBuffers.count
    }

    func allFedSamples() -> [Float] {
        fedSampleArrays.flatMap { $0 }
    }

    func metrics() -> (feedCount: Int, feedSamplesCount: Int, playedCount: Int, finishStreamCallCount: Int, stopCallCount: Int) {
        (
            feedCount: fedBuffers.count,
            feedSamplesCount: fedSampleArrays.count,
            playedCount: playedBuffers.count,
            finishStreamCallCount: finishStreamCallCount,
            stopCallCount: stopCallCount
        )
    }
}

private enum StubAudioPlayerError: Error {
    case playFailed
}

@MainActor
final class ValarTTSMacAppTests: XCTestCase {
    private func makeAppPaths() throws -> ValarAppPaths {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return ValarAppPaths(baseURL: root)
    }

    private func makeDocumentURL() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
            .appendingPathExtension("valarproject")
    }

    private func makeServicesWithBlockingWarmLoad() async throws -> (ValarServiceHub, BlockingInferenceBackend) {
        let appPaths = try makeAppPaths()
        let descriptor = ModelDescriptor(
            id: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
            familyID: .qwen3TTS,
            displayName: "Blocking Warm Qwen Base",
            domain: .tts,
            capabilities: [.speechSynthesis],
            defaultSampleRate: 24_000
        )
        let runtimeConfiguration = RuntimeConfiguration(
            warmPolicy: .eager,
            warmStartModelIDs: [descriptor.id]
        )
        let modelRegistry = ModelRegistry(configuration: runtimeConfiguration)
        let capabilityRegistry = CapabilityRegistry()
        await modelRegistry.register(descriptor)
        await capabilityRegistry.register(descriptor)

        let installRoot = appPaths.modelPacksDirectory.appendingPathComponent("tests", isDirectory: true)
        try FileManager.default.createDirectory(at: installRoot, withIntermediateDirectories: true)
        let installedRecord = try makeInstalledModelRecord(
            modelID: descriptor.id,
            familyID: descriptor.familyID,
            displayName: descriptor.displayName,
            installRoot: installRoot
        )
        let packRegistry = StubModelPackRegistry(installedRecords: [installedRecord])
        let curatedEntry = try XCTUnwrap(SupportedModelCatalog.entry(for: descriptor.id))
        let modelCatalog = ModelCatalog(
            supportedSource: StaticSupportedCatalogSource(records: [curatedEntry]),
            catalogStore: packRegistry,
            packStore: packRegistry,
            capabilityRegistry: capabilityRegistry
        )
        let modelInstaller = ModelInstaller(
            registry: packRegistry,
            modelRegistry: modelRegistry,
            capabilityRegistry: capabilityRegistry,
            paths: appPaths
        )
        let renderQueue = RenderQueue(configuration: runtimeConfiguration)
        let audioPipeline = AudioPipeline()
        let inferenceBackend = BlockingInferenceBackend(
            model: StubStreamingTTSModel(
                descriptor: descriptor,
                streamFactory: {
                    AsyncThrowingStream { continuation in
                        continuation.yield(makeFakePCMChunk())
                        continuation.finish()
                    }
                }
            )
        )

        let services = ValarServiceHub(
            appPaths: appPaths,
            runtimeConfiguration: runtimeConfiguration,
            modelRegistry: modelRegistry,
            capabilityRegistry: capabilityRegistry,
            modelPackRegistry: packRegistry,
            modelCatalog: modelCatalog,
            modelInstaller: modelInstaller,
            inferenceBackend: inferenceBackend,
            renderQueue: renderQueue,
            grdbRenderJobStore: GRDBRenderJobStore(db: try AppDatabase.inMemory()),
            projectRenderService: ProjectRenderService(
                renderQueue: renderQueue,
                projectStore: StubProjectStore(),
                audioPipeline: audioPipeline
            ) { _, _, _ in
                makeFakePCMChunk()
            },
            grdbProjectStore: GRDBProjectStore(db: try AppDatabase.inMemory(), paths: appPaths),
            projectStore: StubProjectStore(),
            grdbVoiceStore: GRDBVoiceStore(db: try AppDatabase.inMemory()),
            voiceLibraryStore: StubVoiceLibraryStore(),
            dictationService: DictationService(),
            translationService: TranslationService(provider: StubTranslationProvider()),
            audioPipeline: audioPipeline,
            audioPlayer: StubAudioPlayer(snapshots: []),
            projectExporter: StubProjectExporter()
        )

        return (services, inferenceBackend)
    }

    func testBootstrapSeedsPublicCuratedCatalogMetadata() async {
        let hub = ValarServiceHub.live(appPaths: try! makeAppPaths())
        let snapshot = await hub.snapshot()
        let diagnosticsSnapshot = await hub.diagnosticsSnapshot()

        XCTAssertEqual(snapshot.modelCount, 7)
        XCTAssertEqual(snapshot.installedModelCount, 0)
        XCTAssertEqual(snapshot.recommendedModelCount, 5)
        XCTAssertEqual(snapshot.availableGenerationModels.count, 5)
        XCTAssertEqual(snapshot.availableRecognitionModels.count, 2)
        XCTAssertTrue(snapshot.compatibilityReport.preservedModelIdentifiers.isEmpty)
        XCTAssertEqual(diagnosticsSnapshot.compatibilityReport.preservedModelIdentifiers.count, 7)
        XCTAssertNotNil(snapshot.availableGenerationModels.first?.id)
        XCTAssertFalse(snapshot.catalogModels.contains(where: { $0.familyID == .tadaTTS }))
    }

    func testGeneratorRuntimeOptionsReadModelRegistryAndVoiceStore() async throws {
        let appPaths = try makeAppPaths()
        let runtimeConfiguration = RuntimeConfiguration()
        let descriptor = ModelDescriptor(
            id: "test/runtime-qwen",
            familyID: .qwen3TTS,
            displayName: "Runtime Registry Model",
            domain: .tts,
            capabilities: [.speechSynthesis],
            defaultSampleRate: 24_000
        )
        let voice = VoiceLibraryRecord(
            label: "Runtime Voice",
            modelID: descriptor.id.rawValue,
            runtimeModelID: descriptor.id.rawValue
        )
        let modelRegistry = ModelRegistry(configuration: runtimeConfiguration)
        let capabilityRegistry = CapabilityRegistry()
        await modelRegistry.register(descriptor)
        await capabilityRegistry.register(descriptor)

        let modelPackRegistry = StubModelPackRegistry()
        let modelCatalog = ModelCatalog(
            supportedSource: StaticSupportedCatalogSource(records: []),
            catalogStore: modelPackRegistry,
            packStore: modelPackRegistry,
            capabilityRegistry: capabilityRegistry
        )
        let modelInstaller = ModelInstaller(
            registry: modelPackRegistry,
            modelRegistry: modelRegistry,
            capabilityRegistry: capabilityRegistry,
            paths: appPaths
        )
        let voiceLibraryStore = SeededVoiceLibraryStore(voices: [voice])
        let renderQueue = RenderQueue(configuration: runtimeConfiguration)
        let audioPipeline = AudioPipeline()
        let services = ValarServiceHub(
            appPaths: appPaths,
            runtimeConfiguration: runtimeConfiguration,
            modelRegistry: modelRegistry,
            capabilityRegistry: capabilityRegistry,
            modelPackRegistry: modelPackRegistry,
            modelCatalog: modelCatalog,
            modelInstaller: modelInstaller,
            inferenceBackend: StubInferenceBackend(
                model: StubStreamingTTSModel(descriptor: descriptor) {
                    AsyncThrowingStream { continuation in
                        continuation.finish()
                    }
                }
            ),
            renderQueue: renderQueue,
            grdbRenderJobStore: GRDBRenderJobStore(db: try AppDatabase.inMemory()),
            projectRenderService: ProjectRenderService(
                renderQueue: renderQueue,
                projectStore: StubProjectStore(),
                audioPipeline: audioPipeline
            ) { _, _, _ in
                makeFakePCMChunk()
            },
            grdbProjectStore: GRDBProjectStore(db: try AppDatabase.inMemory(), paths: appPaths),
            projectStore: StubProjectStore(),
            grdbVoiceStore: GRDBVoiceStore(db: try AppDatabase.inMemory()),
            voiceLibraryStore: voiceLibraryStore,
            dictationService: DictationService(),
            translationService: TranslationService(provider: StubTranslationProvider()),
            audioPipeline: audioPipeline,
            audioPlayer: StubAudioPlayer(snapshots: []),
            projectExporter: StubProjectExporter()
        )

        let state = GeneratorState(services: services, playbackPollInterval: .milliseconds(5))
        await state.reloadRuntimeOptions(selectedModelID: nil)

        XCTAssertEqual(state.availableModels, [
            RuntimeModelPickerOption(
                id: descriptor.id,
                displayName: descriptor.displayName,
                familyID: descriptor.familyID,
                voiceFeatures: [],
                isRecommended: false
            ),
        ])
        XCTAssertEqual(state.availableVoices.map(\.id), [voice.id])
        XCTAssertEqual(state.availableVoices.map(\.label), [voice.label])
        XCTAssertEqual(state.selectedModelID, descriptor.id)
    }

    func testGeneratorStateReloadsVoxtralRandomPresetVoice() async throws {
        let voxtralDescriptor = ModelDescriptor(
            id: "mlx-community/Voxtral-4B-TTS-2603-mlx-4bit",
            familyID: .voxtralTTS,
            displayName: "Voxtral 4B TTS 2603 MLX (4-bit)",
            domain: .tts,
            capabilities: [.speechSynthesis, .presetVoices, .streaming],
            defaultSampleRate: 24_000
        )
        let voxtralManifest = ValarPersistence.ModelPackManifest(
            id: voxtralDescriptor.id.rawValue,
            familyID: voxtralDescriptor.familyID.rawValue,
            modelID: voxtralDescriptor.id.rawValue,
            displayName: voxtralDescriptor.displayName,
            capabilities: [ModelCapability.speechSynthesis.rawValue],
            backendKinds: [BackendKind.mlx.rawValue],
            tokenizerType: "voxtral",
            sampleRate: 24_000,
            artifactSpecs: [
                ModelPackArtifact(id: "weights", kind: "weights", relativePath: "weights/model.safetensors")
            ],
            licenseName: "Mistral Research License",
            notes: "Test Voxtral bundle"
        )
        let voxtralBundleURL = try makeModelBundle(
            manifest: voxtralManifest,
            files: [
                "weights/model.safetensors": Data([0x00, 0x01, 0x02, 0x03])
            ]
        )
        let supportedRecord = SupportedModelCatalogRecord(
            familyID: voxtralDescriptor.familyID.rawValue,
            modelID: voxtralDescriptor.id.rawValue,
            displayName: voxtralDescriptor.displayName,
            providerName: "Valar",
            sourceKind: .localFile,
            isRecommended: false
        )
        let installedRecord = InstalledModelRecord(
            familyID: voxtralDescriptor.familyID.rawValue,
            modelID: voxtralDescriptor.id.rawValue,
            displayName: voxtralDescriptor.displayName,
            installedPath: voxtralBundleURL.path,
            manifestPath: voxtralBundleURL.appendingPathComponent("manifest.json").path,
            artifactCount: 1,
            sourceKind: .localFile
        )

        let (state, _) = try await makeGeneratorState(
            descriptor: voxtralDescriptor,
            supportedRecords: [supportedRecord],
            installedRecords: [installedRecord],
            streamFactory: {
                AsyncThrowingStream { continuation in
                    continuation.yield(makeFakePCMChunk(samples: [0.1, 0.2]))
                    continuation.finish()
                }
            },
            audioPlayer: StubAudioPlayer(snapshots: [])
        )

        await state.reloadRuntimeOptions(selectedModelID: nil)

        let randomVoices = state.availableVoices.filter {
            $0.backendVoiceID == "random"
                && $0.modelID == voxtralDescriptor.id.rawValue
                && $0.runtimeModelID == voxtralDescriptor.id.rawValue
        }

        XCTAssertEqual(state.selectedModelID, voxtralDescriptor.id)
        XCTAssertEqual(randomVoices.count, 1)
        XCTAssertEqual(randomVoices.first?.label, "Random")
        XCTAssertTrue(randomVoices.first?.isModelDeclaredPreset == true)
    }

    func testStoredVoicePickerPartitionsPresetVoicesBySelectedModel() {
        let qwenModel = ModelIdentifier("mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16")
        let voxtralModel = ModelIdentifier("mlx-community/Voxtral-4B-TTS-2603-mlx-4bit")
        let voxtralRandom = VoiceLibraryRecord(
            label: "Random",
            modelID: voxtralModel.rawValue,
            runtimeModelID: voxtralModel.rawValue,
            backendVoiceID: "random"
        )
        let voxtralPreset = VoiceLibraryRecord(
            label: "Emma",
            modelID: voxtralModel.rawValue,
            runtimeModelID: voxtralModel.rawValue,
            backendVoiceID: "neutral_female"
        )
        let qwenPreset = VoiceLibraryRecord(
            label: "Alloy",
            modelID: qwenModel.rawValue,
            runtimeModelID: qwenModel.rawValue,
            backendVoiceID: "Alloy"
        )
        let savedVoice = VoiceLibraryRecord(
            label: "Narrator",
            modelID: qwenModel.rawValue,
            runtimeModelID: qwenModel.rawValue
        )
        let voxtralSavedVoice = VoiceLibraryRecord(
            label: "Claire Memo",
            modelID: voxtralModel.rawValue,
            runtimeModelID: voxtralModel.rawValue
        )

        let partition = StoredVoicePickerView.partitionVoices(
            [voxtralRandom, voxtralPreset, qwenPreset, savedVoice, voxtralSavedVoice],
            selectedModelID: voxtralModel
        )

        XCTAssertEqual(partition.preset.map(\.label), ["Random", "Emma"])
        XCTAssertEqual(partition.saved.map(\.label), ["Claire Memo"])
    }

    func testGeneratorRuntimeOptionsExcludeUninstalledTadaModelsFromPublicSurface() async throws {
        let appPaths = try makeAppPaths()
        let runtimeConfiguration = RuntimeConfiguration()
        let modelRegistry = ModelRegistry(configuration: runtimeConfiguration)
        let capabilityRegistry = CapabilityRegistry()
        let modelPackRegistry = StubModelPackRegistry()
        let modelCatalog = ModelCatalog(
            supportedSource: SupportedCatalogSource.curated(),
            catalogStore: modelPackRegistry,
            packStore: modelPackRegistry,
            capabilityRegistry: capabilityRegistry
        )
        let modelInstaller = ModelInstaller(
            registry: modelPackRegistry,
            modelRegistry: modelRegistry,
            capabilityRegistry: capabilityRegistry,
            paths: appPaths
        )
        let renderQueue = RenderQueue(configuration: runtimeConfiguration)
        let audioPipeline = AudioPipeline()
        let runtimeDescriptor = ModelDescriptor(
            id: "test/runtime-qwen",
            familyID: .qwen3TTS,
            displayName: "Runtime Qwen",
            domain: .tts,
            capabilities: [.speechSynthesis],
            defaultSampleRate: 24_000
        )
        await modelRegistry.register(runtimeDescriptor)
        await capabilityRegistry.register(runtimeDescriptor)

        let services = ValarServiceHub(
            appPaths: appPaths,
            runtimeConfiguration: runtimeConfiguration,
            modelRegistry: modelRegistry,
            capabilityRegistry: capabilityRegistry,
            modelPackRegistry: modelPackRegistry,
            modelCatalog: modelCatalog,
            modelInstaller: modelInstaller,
            inferenceBackend: StubInferenceBackend(
                model: StubStreamingTTSModel(descriptor: runtimeDescriptor) {
                    AsyncThrowingStream { continuation in
                        continuation.finish()
                    }
                }
            ),
            renderQueue: renderQueue,
            grdbRenderJobStore: GRDBRenderJobStore(db: try AppDatabase.inMemory()),
            projectRenderService: ProjectRenderService(
                renderQueue: renderQueue,
                projectStore: StubProjectStore(),
                audioPipeline: audioPipeline
            ) { _, _, _ in
                makeFakePCMChunk()
            },
            grdbProjectStore: GRDBProjectStore(db: try AppDatabase.inMemory(), paths: appPaths),
            projectStore: StubProjectStore(),
            grdbVoiceStore: GRDBVoiceStore(db: try AppDatabase.inMemory()),
            voiceLibraryStore: StubVoiceLibraryStore(),
            dictationService: DictationService(),
            translationService: TranslationService(provider: StubTranslationProvider()),
            audioPipeline: audioPipeline,
            audioPlayer: StubAudioPlayer(snapshots: []),
            projectExporter: StubProjectExporter()
        )

        let state = GeneratorState(services: services)
        await state.reloadRuntimeOptions(selectedModelID: nil)

        XCTAssertEqual(state.availableModels.map(\.id), [runtimeDescriptor.id])
        XCTAssertFalse(state.availableModels.contains(where: { $0.familyID == .tadaTTS }))
    }

    func testAppStateRefreshSnapshotUsesRuntimeModelRegistryForSelection() async throws {
        let appPaths = try makeAppPaths()
        let runtimeConfiguration = RuntimeConfiguration()
        let descriptor = ModelDescriptor(
            id: "test/runtime-only-selection",
            familyID: .qwen3TTS,
            displayName: "Runtime Only Selection",
            domain: .tts,
            capabilities: [.speechSynthesis],
            defaultSampleRate: 24_000
        )
        let modelRegistry = ModelRegistry(configuration: runtimeConfiguration)
        let capabilityRegistry = CapabilityRegistry()
        await modelRegistry.register(descriptor)
        await capabilityRegistry.register(descriptor)

        let modelPackRegistry = StubModelPackRegistry()
        let modelCatalog = ModelCatalog(
            supportedSource: StaticSupportedCatalogSource(records: []),
            catalogStore: modelPackRegistry,
            packStore: modelPackRegistry,
            capabilityRegistry: capabilityRegistry
        )
        let modelInstaller = ModelInstaller(
            registry: modelPackRegistry,
            modelRegistry: modelRegistry,
            capabilityRegistry: capabilityRegistry,
            paths: appPaths
        )
        let renderQueue = RenderQueue(configuration: runtimeConfiguration)
        let audioPipeline = AudioPipeline()
        let services = ValarServiceHub(
            appPaths: appPaths,
            runtimeConfiguration: runtimeConfiguration,
            modelRegistry: modelRegistry,
            capabilityRegistry: capabilityRegistry,
            modelPackRegistry: modelPackRegistry,
            modelCatalog: modelCatalog,
            modelInstaller: modelInstaller,
            inferenceBackend: StubInferenceBackend(
                model: StubStreamingTTSModel(descriptor: descriptor) {
                    AsyncThrowingStream { continuation in
                        continuation.finish()
                    }
                }
            ),
            renderQueue: renderQueue,
            grdbRenderJobStore: GRDBRenderJobStore(db: try AppDatabase.inMemory()),
            projectRenderService: ProjectRenderService(
                renderQueue: renderQueue,
                projectStore: StubProjectStore(),
                audioPipeline: audioPipeline
            ) { _, _, _ in
                makeFakePCMChunk()
            },
            grdbProjectStore: GRDBProjectStore(db: try AppDatabase.inMemory(), paths: appPaths),
            projectStore: StubProjectStore(),
            grdbVoiceStore: GRDBVoiceStore(db: try AppDatabase.inMemory()),
            voiceLibraryStore: StubVoiceLibraryStore(),
            dictationService: DictationService(),
            translationService: TranslationService(provider: StubTranslationProvider()),
            audioPipeline: audioPipeline,
            audioPlayer: StubAudioPlayer(snapshots: []),
            projectExporter: StubProjectExporter()
        )

        let appState = AppState(services: services, sharedServices: services, documentProjectID: UUID())
        await appState.refreshSnapshot()

        XCTAssertEqual(appState.dashboardSnapshot.modelCount, 0)
        XCTAssertEqual(appState.selectedModelID, descriptor.id)
        XCTAssertEqual(appState.generatorState.availableModels.map { $0.id }, [descriptor.id])
        XCTAssertEqual(appState.generatorState.selectedModelID, descriptor.id)
    }

    func testSecondTTsFamilyInstallsSelectsSynthesizesAndExports() async throws {
        let appPaths = try makeAppPaths()
        let runtimeConfiguration = RuntimeConfiguration()
        let modelRegistry = ModelRegistry(configuration: runtimeConfiguration)
        let capabilityRegistry = CapabilityRegistry()
        let modelPackRegistry = ModelPackRegistry()
        let modelCatalog = ModelCatalog(
            supportedSource: SupportedCatalogSource.curated(),
            catalogStore: modelPackRegistry,
            packStore: modelPackRegistry,
            capabilityRegistry: capabilityRegistry
        )
        let modelInstaller = ModelInstaller(
            registry: modelPackRegistry,
            modelRegistry: modelRegistry,
            capabilityRegistry: capabilityRegistry,
            paths: appPaths
        )
        let projectStore = ProjectStore(paths: appPaths)
        let voiceLibraryStore = StubVoiceLibraryStore()
        let renderQueue = RenderQueue(configuration: runtimeConfiguration)
        let audioPipeline = AudioPipeline()
        let audioPlayer = StubAudioPlayer(
            snapshots: [
                AudioPlaybackSnapshot(position: Double(4) / 24_000, queuedDuration: 0, isPlaying: false, isBuffering: false, didFinish: true),
            ]
        )
        let streamFactory: @Sendable () -> AsyncThrowingStream<AudioChunk, Error> = {
            AsyncThrowingStream { continuation in
                continuation.yield(makeFakePCMChunk(samples: [0, 0.2]))
                continuation.yield(makeFakePCMChunk(samples: [-0.1, 0.1]))
                continuation.finish()
            }
        }

        let sopranoEntry = try XCTUnwrap(SopranoCatalog.supportedEntries.first)
        let installResult = try await modelInstaller.install(
            manifest: ModelCatalog.makePersistenceManifest(from: sopranoEntry.manifest),
            sourceKind: .remoteURL,
            sourceLocation: try XCTUnwrap(sopranoEntry.remoteURL?.absoluteString),
            notes: "Installed during pluggability proof"
        )
        let catalogModels = try await modelCatalog.refresh()
        let installedModel = try XCTUnwrap(catalogModels.first(where: { $0.id == sopranoEntry.id }))
        let runtime = BackendSelectionPolicy.Runtime(availableBackends: [.mlx])
        let configuration = try BackendSelectionPolicy().runtimeConfiguration(
            for: installedModel.descriptor,
            runtime: runtime
        )

        XCTAssertEqual(installResult.descriptor.familyID, .soprano)
        XCTAssertEqual(installedModel.installState, .installed)
        XCTAssertEqual(installedModel.familyID, .soprano)
        XCTAssertEqual(configuration.backendKind, .mlx)

        let inferenceBackend = StubInferenceBackend(
            model: StubStreamingTTSModel(descriptor: installResult.descriptor, streamFactory: streamFactory)
        )
        let services = ValarServiceHub(
            appPaths: appPaths,
            runtimeConfiguration: runtimeConfiguration,
            modelRegistry: modelRegistry,
            capabilityRegistry: capabilityRegistry,
            modelPackRegistry: modelPackRegistry,
            modelCatalog: modelCatalog,
            modelInstaller: modelInstaller,
            inferenceBackend: inferenceBackend,
            renderQueue: renderQueue,
            grdbRenderJobStore: GRDBRenderJobStore(db: try AppDatabase.inMemory()),
            projectRenderService: ProjectRenderService(
                renderQueue: renderQueue,
                projectStore: projectStore,
                audioPipeline: audioPipeline
            ) { _, _, _ in
                makeFakePCMChunk()
            },
            grdbProjectStore: GRDBProjectStore(db: try AppDatabase.inMemory(), paths: appPaths),
            projectStore: projectStore,
            grdbVoiceStore: GRDBVoiceStore(db: try AppDatabase.inMemory()),
            voiceLibraryStore: voiceLibraryStore,
            dictationService: DictationService(),
            translationService: TranslationService(provider: StubTranslationProvider()),
            audioPipeline: audioPipeline,
            audioPlayer: audioPlayer,
            projectExporter: StubProjectExporter()
        )
        let state = GeneratorState(services: services, playbackPollInterval: .milliseconds(5))
        state.selectedModelID = sopranoEntry.id
        state.text = "Second family synthesis"

        await state.generate()
        try await Task.sleep(nanoseconds: 30_000_000)

        XCTAssertTrue(state.hasAudio)
        XCTAssertEqual(state.audioDuration, Double(4) / 24_000, accuracy: 0.000_001)

        let project = try await projectStore.create(title: "Soprano Export")
        await projectStore.addChapter(
            ChapterRecord(projectID: project.id, index: 0, title: "Chapter 1", script: "Export this")
        )

        let exportDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: exportDirectory, withIntermediateDirectories: true)
        let tracker = RenderInvocationTracker()

        let coordinator = ProjectAudioExportCoordinator(
            projectStore: projectStore,
            audioPipeline: audioPipeline,
            synthesizeChapter: { modelID, options, text in
                await tracker.begin(text, options: options)
                XCTAssertEqual(modelID, sopranoEntry.id)
                let descriptor = installResult.descriptor
                let configuration = ModelRuntimeConfiguration(
                    backendKind: inferenceBackend.backendKind,
                    residencyPolicy: .automatic
                )
                let loadedModel = try await inferenceBackend.loadModel(
                    descriptor: descriptor,
                    configuration: configuration
                )
                let workflow = try XCTUnwrap(loadedModel as? any TextToSpeechWorkflow)
                let request = SpeechSynthesisRequest(
                    model: modelID,
                    text: text,
                    language: options.normalizedLanguage,
                    sampleRate: descriptor.defaultSampleRate ?? 24_000,
                    responseFormat: "pcm_f32le",
                    temperature: options.temperature.map(Float.init),
                    topP: options.topP.map(Float.init),
                    repetitionPenalty: options.repetitionPenalty.map(Float.init),
                    maxTokens: options.maxTokens,
                    voiceBehavior: options.voiceBehavior
                )
                let session = ModelRuntimeSession(
                    descriptor: descriptor,
                    backendKind: configuration.backendKind,
                    configuration: configuration,
                    state: .resident
                )
                let chunk = try await workflow.synthesize(request: request, in: session)
                await tracker.end()
                return chunk
            }
        )
        let exportOptions = RenderSynthesisOptions(
            language: "fr",
            temperature: 0.65,
            topP: 0.8,
            repetitionPenalty: 1.1,
            maxTokens: 2_048,
            voiceBehavior: .expressive
        )
        let exportResult = try await coordinator.exportProjectAudio(
            projectID: project.id,
            modelID: sopranoEntry.id,
            synthesisOptions: exportOptions,
            format: .wav,
            mode: .chapters,
            destinationURL: exportDirectory,
            onProgress: { _ in }
        )

        XCTAssertEqual(exportResult.exportedChapterCount, 1)
        XCTAssertEqual(exportResult.files.count, 1)
        XCTAssertTrue(FileManager.default.fileExists(atPath: exportResult.files[0].path))
        let exports = await projectStore.exports(for: project.id)
        let recordedOptions = await tracker.recordedOptions()
        XCTAssertEqual(exports.count, 1)
        XCTAssertEqual(recordedOptions, [exportOptions])
    }

    func testSnapshotCompletesBeforeDeferredWarmLoadFinishes() async throws {
        let (services, inferenceBackend) = try await makeServicesWithBlockingWarmLoad()
        defer {
            Task {
                await inferenceBackend.releaseLoad()
            }
        }
        let snapshotFinished = expectation(description: "snapshot finished")
        var didSignalModelsReady = false

        services.onModelsReady = {
            didSignalModelsReady = true
        }

        let snapshotTask = Task {
            let snapshot = await services.snapshot()
            await MainActor.run {
                snapshotFinished.fulfill()
            }
            return snapshot
        }

        let didStartLoad = await inferenceBackend.waitForLoadStart()
        XCTAssertTrue(didStartLoad, "Expected deferred warm load to begin after services.onModelsReady is set.")
        await fulfillment(of: [snapshotFinished], timeout: 0.5)

        _ = await snapshotTask.value
        let didFinishLoadBeforeRelease = await inferenceBackend.hasFinishedLoad()
        XCTAssertFalse(didSignalModelsReady)
        XCTAssertFalse(didFinishLoadBeforeRelease)
    }

    func testModelsReadyNotifiesMultipleObservers() async throws {
        let (services, inferenceBackend) = try await makeServicesWithBlockingWarmLoad()
        defer {
            Task {
                await inferenceBackend.releaseLoad()
            }
        }

        let observer1 = expectation(description: "observer 1")
        let observer2 = expectation(description: "observer 2")

        let observer1ID = services.registerModelsReadyObserver {
            observer1.fulfill()
        }
        let observer2ID = services.registerModelsReadyObserver {
            observer2.fulfill()
        }

        let didStartLoad = await inferenceBackend.waitForLoadStart()
        XCTAssertTrue(didStartLoad)

        await inferenceBackend.releaseLoad()
        await fulfillment(of: [observer1, observer2], timeout: 0.5)

        services.unregisterModelsReadyObserver(observer1ID)
        services.unregisterModelsReadyObserver(observer2ID)
    }

    func testOnModelsReadyWaitsForDeferredWarmLoadRelease() async throws {
        let (services, inferenceBackend) = try await makeServicesWithBlockingWarmLoad()
        let modelsReady = expectation(description: "models ready")
        var didSignalModelsReady = false

        services.onModelsReady = {
            didSignalModelsReady = true
            modelsReady.fulfill()
        }

        let didStartLoad = await inferenceBackend.waitForLoadStart()
        XCTAssertTrue(didStartLoad, "Expected deferred warm load to begin after services.onModelsReady is set.")
        try await Task.sleep(nanoseconds: 50_000_000)
        XCTAssertFalse(didSignalModelsReady)

        await inferenceBackend.releaseLoad()
        await fulfillment(of: [modelsReady], timeout: 0.5)
        let didFinishLoadAfterRelease = await inferenceBackend.hasFinishedLoad()
        XCTAssertTrue(didSignalModelsReady)
        XCTAssertTrue(didFinishLoadAfterRelease)
    }

    func testActionsUpdateWorkspaceState() async throws {
        let hub = ValarServiceHub.live(appPaths: try makeAppPaths())

        // Seed snapshot so models are available
        _ = await hub.snapshot()

        guard let project = await hub.createProject(title: "A New Chronicle") else {
            return XCTFail("Expected valid project title to be accepted")
        }
        XCTAssertEqual(project.title, "A New Chronicle")

        let snapshot1 = await hub.snapshot()
        XCTAssertEqual(snapshot1.projectCount, 1)
        XCTAssertEqual(snapshot1.projects.first?.title, "A New Chronicle")

        guard let modelID = snapshot1.availableGenerationModels.first?.id else {
            XCTFail("No generation model available")
            return
        }

        _ = try await hub.createVoice(label: "Narrator One", modelID: modelID)
        let snapshot2 = await hub.snapshot()
        XCTAssertEqual(snapshot2.voiceCount, 1)

        _ = await hub.queueRender(projectID: project.id, modelID: modelID)
        let snapshot3 = await hub.snapshot()
        XCTAssertEqual(snapshot3.jobCount, 1)
    }

    func testProjectWorkspaceStatePersistsSpeakersAcrossRestart() async throws {
        let appPaths = try makeAppPaths()
        defer { try? FileManager.default.removeItem(at: appPaths.applicationSupport) }

        let hub = ValarServiceHub.live(appPaths: appPaths)
        guard let project = await hub.createProject(title: "Speaker Chronicle") else {
            return XCTFail("Expected project creation to succeed")
        }

        let initialWorkspace = ProjectWorkspaceState(services: hub)
        await initialWorkspace.load(project: project, preferredModelID: nil)
        await initialWorkspace.addSpeaker(
            name: "Narrator",
            voiceModelID: ModelIdentifier("test/narrator"),
            language: "en"
        )

        let reopenedWorkspace = ProjectWorkspaceState(services: ValarServiceHub.live(appPaths: appPaths))
        await reopenedWorkspace.load(project: project, preferredModelID: nil)

        XCTAssertEqual(reopenedWorkspace.speakers.count, 1)
        XCTAssertEqual(reopenedWorkspace.speakers.first?.name, "Narrator")
        XCTAssertEqual(reopenedWorkspace.speakers.first?.voiceModelID, ModelIdentifier("test/narrator"))
        XCTAssertEqual(reopenedWorkspace.speakers.first?.language, "en")

        let speakerID = try XCTUnwrap(reopenedWorkspace.speakers.first?.id)
        await reopenedWorkspace.removeSpeaker(speakerID)

        let reopenedAfterRemoval = ProjectWorkspaceState(services: ValarServiceHub.live(appPaths: appPaths))
        await reopenedAfterRemoval.load(project: project, preferredModelID: nil)
        XCTAssertTrue(reopenedAfterRemoval.speakers.isEmpty)
    }

    func testSaveReopenProjectBundlePreservesChaptersSpeakersAndExports() async throws {
        let sourceAppPaths = try makeAppPaths()
        let reopenedAppPaths = try makeAppPaths()
        let bundleURL = makeDocumentURL()
        defer {
            try? FileManager.default.removeItem(at: sourceAppPaths.applicationSupport)
            try? FileManager.default.removeItem(at: reopenedAppPaths.applicationSupport)
            try? FileManager.default.removeItem(at: bundleURL)
        }

        let sourceServices = ValarServiceHub.live(appPaths: sourceAppPaths)
        guard let project = await sourceServices.createProject(title: "Round Trip Chronicle") else {
            return XCTFail("Expected project creation to succeed")
        }
        await sourceServices.updateProjectBundleURL(bundleURL, for: project.id)

        let sourceWorkspace = ProjectWorkspaceState(services: sourceServices)
        await sourceWorkspace.load(project: project, preferredModelID: nil)
        let expectedRenderOptions = RenderSynthesisOptions(
            language: "fr",
            temperature: 0.65,
            topP: 0.82,
            repetitionPenalty: 1.14,
            maxTokens: 3_072,
            voiceBehavior: .stableNarrator
        )
        sourceWorkspace.language = expectedRenderOptions.language
        sourceWorkspace.temperature = 0.65
        sourceWorkspace.topP = 0.82
        sourceWorkspace.repetitionPenalty = 1.14
        sourceWorkspace.maxTokens = 3_072
        sourceWorkspace.voiceBehavior = expectedRenderOptions.voiceBehavior

        await sourceWorkspace.addChapter()
        var introChapter = try XCTUnwrap(sourceWorkspace.chapters.first)
        introChapter.title = "Intro"
        introChapter.script = "Alpha chapter"
        await sourceWorkspace.updateChapter(introChapter)

        await sourceWorkspace.addChapter()
        var outroChapter = try XCTUnwrap(sourceWorkspace.chapters.last)
        outroChapter.title = "Outro"
        outroChapter.script = "Beta chapter"
        await sourceWorkspace.updateChapter(outroChapter)

        await sourceWorkspace.addSpeaker(
            name: "Narrator",
            voiceModelID: ModelIdentifier("test/narrator"),
            language: "en"
        )

        let renderService = ProjectRenderService(
            renderQueue: sourceServices.renderQueue,
            projectStore: sourceServices.projectStore,
            audioPipeline: AudioPipeline()
        ) { _, _, _ in
            makeFakePCMChunk()
        }
        _ = await renderService.enqueueProjectRender(
            project: project,
            modelID: "test/model",
            synthesisOptions: RenderSynthesisOptions()
        )
        _ = await waitForTerminalJobs(expectedCount: 2, queue: sourceServices.renderQueue)

        let expectedChapters = await sourceServices.projectStore.chapters(for: project.id)
        let expectedSpeakers = await sourceServices.projectStore.speakers(for: project.id)
        let expectedQueuedRenderJobs = await sourceServices.renderQueue.jobs(matching: nil)
            .filter { $0.projectID == project.id }
        let expectedExports = await sourceServices.projectStore.exports(for: project.id)
        XCTAssertEqual(expectedChapters.map(\.title), ["Intro", "Outro"])
        XCTAssertEqual(expectedSpeakers.map(\.name), ["Narrator"])
        XCTAssertEqual(expectedQueuedRenderJobs.count, 2)
        XCTAssertEqual(expectedExports.map(\.fileName), ["001-intro.wav", "002-outro.wav"])

        let savedLocation = try await sourceServices.saveProjectBundle(
            projectID: project.id,
            to: bundleURL,
            modelID: ModelIdentifier("test/model"),
            synthesisOptions: expectedRenderOptions
        )

        for export in expectedExports {
            let exportURL = savedLocation.exportsDirectory.appendingPathComponent(export.fileName, isDirectory: false)
            XCTAssertTrue(
                FileManager.default.fileExists(atPath: exportURL.path),
                "Expected export file at \(exportURL.path)"
            )
        }

        let reopenedServices = ValarServiceHub.live(appPaths: reopenedAppPaths)
        let reopenedSnapshot = try await reopenedServices.openProjectBundle(from: savedLocation.bundleURL)
        let reopenedWorkspace = ProjectWorkspaceState(services: reopenedServices)
        await reopenedWorkspace.load(
            project: reopenedSnapshot.project,
            preferredModelID: reopenedSnapshot.modelID.map { ModelIdentifier($0) },
            preferredSynthesisOptions: reopenedSnapshot.renderSynthesisOptions
        )

        XCTAssertEqual(reopenedSnapshot.chapters, expectedChapters)
        XCTAssertEqual(reopenedSnapshot.speakers, expectedSpeakers)
        XCTAssertEqual(reopenedSnapshot.exports.map(\.fileName), expectedExports.map(\.fileName))
        XCTAssertEqual(reopenedSnapshot.modelID, "test/model")
        XCTAssertEqual(reopenedSnapshot.renderSynthesisOptions, expectedRenderOptions)
        XCTAssertEqual(reopenedSnapshot.renderJobs.count, expectedQueuedRenderJobs.count)
        XCTAssertEqual(
            reopenedSnapshot.renderJobs.map(\.outputFileName),
            expectedQueuedRenderJobs.map(\.outputFileName)
        )
        XCTAssertEqual(
            reopenedSnapshot.renderJobs.map(\.synthesisOptions),
            expectedQueuedRenderJobs.map(\.synthesisOptions)
        )
        XCTAssertEqual(reopenedWorkspace.chapters, expectedChapters)
        XCTAssertEqual(reopenedWorkspace.speakers, expectedSpeakers.map(SpeakerEntry.init(record:)))
        XCTAssertNil(reopenedWorkspace.selectedRenderModelID)
        XCTAssertEqual(reopenedWorkspace.renderSynthesisOptions, expectedRenderOptions)

        let reopenedExports = await reopenedServices.projectStore.exports(for: project.id)
        let reopenedBundleLocation = await reopenedServices.projectStore.bundleLocation(for: project.id)
        let reopenedRenderJobs = await reopenedServices.projectStore.renderJobs(for: project.id)
        XCTAssertEqual(reopenedExports.map(\.fileName), expectedExports.map(\.fileName))
        XCTAssertEqual(reopenedRenderJobs.count, expectedQueuedRenderJobs.count)
        XCTAssertEqual(
            reopenedRenderJobs.map(\.outputFileName),
            expectedQueuedRenderJobs.map(\.outputFileName)
        )
        XCTAssertEqual(
            reopenedRenderJobs.map(\.synthesisOptions),
            expectedQueuedRenderJobs.map(\.synthesisOptions)
        )
        XCTAssertEqual(reopenedBundleLocation?.bundleURL, savedLocation.bundleURL)

        for export in reopenedExports {
            let exportURL = try XCTUnwrap(reopenedBundleLocation?.exportsDirectory.appendingPathComponent(export.fileName, isDirectory: false))
            XCTAssertTrue(
                FileManager.default.fileExists(atPath: exportURL.path),
                "Expected reopened export file at \(exportURL.path)"
            )
        }
    }

    func testSaveProjectBundlePersistsProjectRenderDefaultsInManifest() async throws {
        let appPaths = try makeAppPaths()
        let bundleURL = makeDocumentURL()
        defer {
            try? FileManager.default.removeItem(at: appPaths.applicationSupport)
            try? FileManager.default.removeItem(at: bundleURL)
        }

        let services = ValarServiceHub.live(appPaths: appPaths)
        guard let project = await services.createProject(title: "Manifest Round Trip") else {
            return XCTFail("Expected project creation to succeed")
        }

        let expectedOptions = RenderSynthesisOptions(
            language: "fr",
            temperature: 0.65,
            topP: 0.82,
            repetitionPenalty: 1.14,
            maxTokens: 3_072,
            voiceBehavior: .stableNarrator
        )

        let savedLocation = try await services.saveProjectBundle(
            projectID: project.id,
            to: bundleURL,
            modelID: ModelIdentifier("test/model"),
            synthesisOptions: expectedOptions
        )

        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let manifestData = try Data(contentsOf: savedLocation.manifestURL)
        let manifest = try decoder.decode(ProjectBundleManifest.self, from: manifestData)

        XCTAssertEqual(manifest.modelID, "test/model")
        XCTAssertEqual(manifest.renderSynthesisOptions, expectedOptions)

        let reopened = try await services.openProjectBundle(from: savedLocation.bundleURL)
        XCTAssertEqual(reopened.modelID, "test/model")
        XCTAssertEqual(reopened.renderSynthesisOptions, expectedOptions)
    }

    func testSelectedModelDefaultsToPublicQwenGenerationModel() async {
        let hub = ValarServiceHub.live(appPaths: try! makeAppPaths())
        let snapshot = await hub.snapshot()

        let generationModels = snapshot.availableGenerationModels
        XCTAssertFalse(generationModels.isEmpty)

        XCTAssertTrue(generationModels.allSatisfy { $0.descriptor.capabilities.contains(.speechSynthesis) })
        XCTAssertTrue(generationModels.allSatisfy { $0.installState == .supported })
        XCTAssertTrue(generationModels.contains(where: { $0.familyID == .qwen3TTS }))
    }

    func testLiveCreatesDiskDatabaseAndPersistsAcrossRestart() async throws {
        let appPaths = try makeAppPaths()
        let hub1 = ValarServiceHub.live(appPaths: appPaths)
        let snapshot1 = await hub1.snapshot()

        XCTAssertTrue(FileManager.default.fileExists(atPath: appPaths.databaseURL.path))

        guard let project = await hub1.createProject(title: "Persisted Chronicle") else {
            return XCTFail("Expected valid project title to be accepted")
        }
        let modelID = try XCTUnwrap(snapshot1.availableGenerationModels.first?.id)
        _ = try await hub1.createVoice(label: "Persistent Narrator", modelID: modelID)

        let hub2 = ValarServiceHub.live(appPaths: appPaths)
        let snapshot2 = await hub2.snapshot()

        XCTAssertEqual(snapshot2.projectCount, 1)
        XCTAssertEqual(snapshot2.projects.first?.id, project.id)
        XCTAssertEqual(snapshot2.projects.first?.title, "Persisted Chronicle")
        XCTAssertEqual(snapshot2.voiceCount, 1)
        XCTAssertEqual(snapshot2.voices.first?.label, "Persistent Narrator")
        XCTAssertEqual(snapshot2.installedModelCount, 0)
    }

    func testModelBundleImporterRejectsMissingManifest() async throws {
        let appPaths = try makeAppPaths()
        let services = ValarServiceHub.live(appPaths: appPaths)
        let bundleURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
            .appendingPathExtension("valarmodel")
        try FileManager.default.createDirectory(at: bundleURL, withIntermediateDirectories: true)

        do {
            _ = try await ModelBundleImporter().validateBundle(at: bundleURL, using: services)
            XCTFail("Expected bundle without manifest.json to be rejected")
        } catch let error as ModelImportError {
            XCTAssertEqual(error, .missingManifest)
        }
    }

    func testModelBundleImporterRejectsNonSafetensorsWeights() async throws {
        let appPaths = try makeAppPaths()
        let services = ValarServiceHub.live(appPaths: appPaths)
        let manifest = makeImportManifest(artifactPath: "weights/model.bin")
        let bundleURL = try makeModelBundle(
            manifest: manifest,
            files: ["weights/model.bin": Data("weights".utf8)]
        )

        do {
            _ = try await ModelBundleImporter().validateBundle(at: bundleURL, using: services)
            XCTFail("Expected non-safetensors weight artifact to be rejected")
        } catch let error as ModelImportError {
            XCTAssertEqual(error, .unsupportedWeightExtension("weights/model.bin"))
        }
    }

    func testAppStateImportModelBundleRefreshesCatalogImmediately() async throws {
        let appPaths = try makeAppPaths()
        let services = ValarServiceHub.live(appPaths: appPaths)
        let appState = AppState(services: services, sharedServices: services, documentProjectID: UUID())
        await appState.load()

        let manifest = makeImportManifest(
            modelID: "custom/Imported-Qwen3-TTS-Test",
            displayName: "Imported Qwen3 TTS Test"
        )
        let bundleURL = try makeModelBundle(
            manifest: manifest,
            files: ["weights/model.safetensors": Data("weights".utf8)]
        )

        await appState.importModelBundles(from: [bundleURL])

        XCTAssertNil(appState.importErrorMessage)
        XCTAssertEqual(appState.selectedSection, AppSection.models)
        XCTAssertEqual(appState.selectedModelID?.rawValue, manifest.modelID)

        let importedModel = appState.modelCatalogState.catalogModels.first { $0.id.rawValue == manifest.modelID }
        XCTAssertNotNil(importedModel)
        XCTAssertEqual(importedModel?.installState, .installed)
        XCTAssertEqual(importedModel?.descriptor.displayName, manifest.displayName)

        let installedRecord = try await services.modelPackRegistry.installedRecord(for: manifest.modelID)
        XCTAssertEqual(installedRecord?.sourceKind, .importedArchive)
        XCTAssertTrue(FileManager.default.fileExists(atPath: installedRecord?.manifestPath ?? ""))
    }

    // MARK: - Voice Clone File Validator Tests

    func testFileValidatorRejectsUnsupportedExtension() throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        let mp3URL = tempDir.appendingPathComponent("clip.mp3", isDirectory: false)
        try Data("fake".utf8).write(to: mp3URL)

        XCTAssertThrowsError(try VoiceCloneFileValidator.validateFileSelection(mp3URL)) { error in
            XCTAssertEqual(error as? VoiceCloneError, .unsupportedFileType("mp3"))
        }
    }

    func testFileValidatorRejectsOversizedFile() throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        let bigURL = tempDir.appendingPathComponent("huge.wav", isDirectory: false)

        // Create a file just over the limit (write a small file, then set its size attribute isn't
        // practical, so we'll write a file header and verify the size check logic directly)
        let overLimitSize = VoiceCloneFileValidator.maximumFileSizeBytes + 1
        let fileHandle = FileManager.default.createFile(atPath: bigURL.path, contents: nil)
        XCTAssertTrue(fileHandle)
        let handle = try FileHandle(forWritingTo: bigURL)
        try handle.truncate(atOffset: UInt64(overLimitSize))
        try handle.close()

        XCTAssertThrowsError(try VoiceCloneFileValidator.validateFileSelection(bigURL)) { error in
            XCTAssertEqual(error as? VoiceCloneError, .fileTooLarge(bytes: overLimitSize))
        }
    }

    func testFileValidatorAcceptsValidWAVFileWithinSizeLimit() throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        let wavURL = tempDir.appendingPathComponent("clip.wav", isDirectory: false)
        try Data(count: 1_024).write(to: wavURL)

        XCTAssertNoThrow(try VoiceCloneFileValidator.validateFileSelection(wavURL))
    }

    func testFileValidatorRejectsNonexistentFile() {
        let fakeURL = FileManager.default.temporaryDirectory.appendingPathComponent("nonexistent.wav")
        XCTAssertThrowsError(try VoiceCloneFileValidator.validateFileSelection(fakeURL)) { error in
            XCTAssertEqual(error as? VoiceCloneError, .unreadableFile)
        }
    }

    func testFileValidatorRejectsDirectories() throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        let wavDirectory = tempDir.appendingPathComponent("folder.wav", isDirectory: true)
        try FileManager.default.createDirectory(at: wavDirectory, withIntermediateDirectories: true)

        XCTAssertThrowsError(try VoiceCloneFileValidator.validateFileSelection(wavDirectory)) { error in
            XCTAssertEqual(error as? VoiceCloneError, .unreadableFile)
        }
    }

    func testFileValidatorRejectsNonFileURLs() {
        let remoteURL = URL(string: "https://example.com/reference.wav")!
        XCTAssertThrowsError(try VoiceCloneFileValidator.validateFileSelection(remoteURL)) { error in
            XCTAssertEqual(error as? VoiceCloneError, .unreadableFile)
        }
    }

    func testFileHeaderValidatorRejectsWAVWithWrongMagicBytes() {
        // Data that doesn't start with RIFF...WAVE
        let fakeData = Data("This is not a WAV file at all!!".utf8)
        XCTAssertThrowsError(try VoiceCloneFileValidator.validateFileHeader(fakeData, hint: "wav")) { error in
            XCTAssertEqual(error as? VoiceCloneError, .invalidAudioHeader("WAV"))
        }
    }

    func testFileHeaderValidatorRejectsM4AWithWrongMagicBytes() {
        let fakeData = Data("This is not an M4A file at all!".utf8)
        XCTAssertThrowsError(try VoiceCloneFileValidator.validateFileHeader(fakeData, hint: "m4a")) { error in
            XCTAssertEqual(error as? VoiceCloneError, .invalidAudioHeader("M4A"))
        }
    }

    func testFileHeaderValidatorAcceptsValidWAVHeader() {
        var wavHeader = Data()
        wavHeader.append(contentsOf: "RIFF".utf8)
        wavHeader.append(contentsOf: [0x00, 0x00, 0x00, 0x00]) // file size placeholder
        wavHeader.append(contentsOf: "WAVE".utf8)
        XCTAssertNoThrow(try VoiceCloneFileValidator.validateFileHeader(wavHeader, hint: "wav"))
    }

    func testFileHeaderValidatorAcceptsValidM4AHeader() {
        var m4aHeader = Data()
        m4aHeader.append(contentsOf: [0x00, 0x00, 0x00, 0x20]) // box size
        m4aHeader.append(contentsOf: "ftyp".utf8)
        m4aHeader.append(contentsOf: "M4A ".utf8) // brand
        XCTAssertNoThrow(try VoiceCloneFileValidator.validateFileHeader(m4aHeader, hint: "m4a"))
    }

    func testFileHeaderValidatorRejectsTruncatedData() {
        let tinyData = Data([0x00, 0x01, 0x02])
        XCTAssertThrowsError(try VoiceCloneFileValidator.validateFileHeader(tinyData, hint: "wav")) { error in
            XCTAssertEqual(error as? VoiceCloneError, .invalidAudioHeader("WAV"))
        }
    }

    func testFileHeaderValidatorRejectsUnknownExtension() {
        let data = Data(count: 20)
        XCTAssertThrowsError(try VoiceCloneFileValidator.validateFileHeader(data, hint: "ogg")) { error in
            XCTAssertEqual(error as? VoiceCloneError, .unsupportedFileType("ogg"))
        }
    }

    // MARK: - Voice Clone Audio Validator Tests

    func testVoiceCloneValidatorRejectsInvalidClips() {
        // Too short (2 seconds at 24 kHz)
        let shortSamples = Array(repeating: Float(0.1), count: 48_000)
        let shortBuffer = AudioPCMBuffer(mono: shortSamples, sampleRate: 24_000)
        XCTAssertThrowsError(try VoiceCloneAudioValidator.validate(shortBuffer)) { error in
            XCTAssertEqual(error as? VoiceCloneError, .clipTooShort(actual: 2.0))
        }

        // Too long (35 seconds at 24 kHz)
        let longSamples = Array(repeating: Float(0.1), count: 840_000)
        let longBuffer = AudioPCMBuffer(mono: longSamples, sampleRate: 24_000)
        XCTAssertThrowsError(try VoiceCloneAudioValidator.validate(longBuffer)) { error in
            XCTAssertEqual(error as? VoiceCloneError, .clipTooLong(actual: 35.0))
        }

        // Sample rate too low (8 kHz, 10 seconds)
        let lowRateSamples = Array(repeating: Float(0.1), count: 80_000)
        let lowRateBuffer = AudioPCMBuffer(mono: lowRateSamples, sampleRate: 8_000)
        XCTAssertThrowsError(try VoiceCloneAudioValidator.validate(lowRateBuffer)) { error in
            XCTAssertEqual(error as? VoiceCloneError, .sampleRateTooLow(actual: 8_000))
        }
    }

    func testVoiceCloneValidatorDownmixesStereoAndWarns() throws {
        // 10 seconds of stereo at 24 kHz
        let frameCount = 240_000
        let left = Array(repeating: Float(0.5), count: frameCount)
        let right = Array(repeating: Float(-0.5), count: frameCount)
        let stereoBuffer = AudioPCMBuffer(
            channels: [left, right],
            format: AudioFormatDescriptor(
                sampleRate: 24_000,
                channelCount: 2,
                sampleFormat: .float32,
                interleaved: false,
                container: "pcm"
            )
        )

        let assessment = try VoiceCloneAudioValidator.validate(stereoBuffer)
        XCTAssertEqual(assessment.originalChannelCount, 2)
        XCTAssertNotNil(assessment.warningMessage)
        XCTAssertEqual(assessment.normalizedBuffer.channels.count, 1)
        XCTAssertEqual(assessment.normalizedBuffer.channels[0].count, frameCount)

        // Downmix should average: (0.5 + -0.5) / 2 = 0
        XCTAssertEqual(assessment.normalizedBuffer.channels[0][0], 0.0, accuracy: 0.001)
    }

    func testRapidTypingCoalescesIntoSingleUndoStep() throws {
        let state = GeneratorState(services: ValarServiceHub.live(appPaths: try makeAppPaths()))

        state.applyTextEdit("H")
        state.applyTextEdit("He")
        state.applyTextEdit("Hel")

        XCTAssertEqual(state.text, "Hel")
        XCTAssertTrue(state.canUndo)
        XCTAssertEqual(state.undoMenuTitle, "Undo Typing")

        state.performUndo()

        XCTAssertEqual(state.text, "")
        XCTAssertTrue(state.canRedo)
        XCTAssertEqual(state.redoMenuTitle, "Redo Typing")

        state.performRedo()

        XCTAssertEqual(state.text, "Hel")
    }

    func testUndoRedoRestoresVoiceAndModelSelection() throws {
        let state = GeneratorState(services: ValarServiceHub.live(appPaths: try makeAppPaths()))
        let manualModelID = ModelIdentifier("mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16")
        let runtimeModelID = ModelIdentifier("mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16")
        let voice = VoiceLibraryRecord(
            label: "Narrator",
            modelID: manualModelID.rawValue,
            runtimeModelID: runtimeModelID.rawValue
        )

        state.availableVoices = [voice]
        state.selectModel(manualModelID)

        XCTAssertEqual(state.selectedModelID, manualModelID)
        XCTAssertNil(state.selectedVoiceID)

        state.performUndo()

        XCTAssertNil(state.selectedModelID)
        XCTAssertNil(state.selectedVoiceID)
        XCTAssertTrue(state.canRedo)

        state.performRedo()

        XCTAssertEqual(state.selectedModelID, manualModelID)
        XCTAssertNil(state.selectedVoiceID)

        let voiceState = GeneratorState(services: ValarServiceHub.live(appPaths: try makeAppPaths()))
        voiceState.availableVoices = [voice]
        voiceState.selectVoice(voice.id)

        XCTAssertEqual(voiceState.selectedVoiceID, voice.id)
        XCTAssertEqual(voiceState.selectedModelID, runtimeModelID)

        voiceState.performUndo()

        XCTAssertNil(voiceState.selectedVoiceID)
        XCTAssertNil(voiceState.selectedModelID)
        XCTAssertTrue(voiceState.canRedo)

        voiceState.performRedo()

        XCTAssertEqual(voiceState.selectedVoiceID, voice.id)
        XCTAssertEqual(voiceState.selectedModelID, runtimeModelID)
    }

    func testDocumentScopedAppStateFiltersProjectsToDocumentProject() async throws {
        let services = ValarServiceHub.live(appPaths: try makeAppPaths())
        let maybeDocumentProject = await services.createProject(title: "Document Project")
        let maybeOtherProject = await services.createProject(title: "Other Project")
        let documentProject = try XCTUnwrap(maybeDocumentProject)
        let otherProject = try XCTUnwrap(maybeOtherProject)
        let appState = AppState(services: services, sharedServices: services, documentProjectID: documentProject.id)

        await appState.load()

        XCTAssertEqual(appState.currentProject?.id, documentProject.id)
        XCTAssertEqual(appState.dashboardSnapshot.projectCount, 1)
        XCTAssertEqual(appState.dashboardSnapshot.projects.map { $0.id }, [documentProject.id])
        XCTAssertFalse(appState.dashboardSnapshot.projects.contains(where: { $0.id == otherProject.id }))
    }

    func testDocumentScopedAppStateFiltersRenderJobsToDocumentProject() async throws {
        let services = ValarServiceHub.live(appPaths: try makeAppPaths())
        let maybeDocumentProject = await services.createProject(title: "Document Project")
        let maybeOtherProject = await services.createProject(title: "Other Project")
        let documentProject = try XCTUnwrap(maybeDocumentProject)
        let otherProject = try XCTUnwrap(maybeOtherProject)
        let documentJob = RenderJobRecord(
            projectID: documentProject.id,
            chapterIDs: [],
            outputFileName: "document.wav",
            createdAt: Date(timeIntervalSince1970: 1)
        )
        let otherJob = RenderJobRecord(
            projectID: otherProject.id,
            chapterIDs: [],
            outputFileName: "other.wav",
            createdAt: Date(timeIntervalSince1970: 2)
        )
        try await services.grdbRenderJobStore.save(documentJob)
        try await services.grdbRenderJobStore.save(otherJob)

        let appState = AppState(services: services, sharedServices: services, documentProjectID: documentProject.id)
        await appState.load()

        XCTAssertEqual(appState.dashboardSnapshot.renderJobs.map { $0.id }, [documentJob.id])
        XCTAssertTrue(appState.dashboardSnapshot.renderJobs.allSatisfy { $0.projectID == documentProject.id })
        XCTAssertFalse(appState.dashboardSnapshot.renderJobs.contains(where: { $0.id == otherJob.id }))
    }

    func testDocumentScopedAppStateExportRequiresProjectRenderModelAndRenderableContent() async throws {
        let services = ValarServiceHub.live(appPaths: try makeAppPaths())
        let maybeProject = await services.createProject(title: "Export Project")
        let project = try XCTUnwrap(maybeProject)
        let appState = AppState(services: services, sharedServices: services, documentProjectID: project.id)

        await appState.load()

        appState.selectedModelID = ModelIdentifier("mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16")
        appState.projectHasRenderableContent = true
        XCTAssertFalse(appState.canExportCurrentProject)

        appState.projectRenderModelID = ModelIdentifier("mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16")
        XCTAssertTrue(appState.canExportCurrentProject)

        appState.projectHasRenderableContent = false
        XCTAssertFalse(appState.canExportCurrentProject)
    }

    func testCreateDesignedVoicePersistsPromptMetadata() async throws {
        let hub = ValarServiceHub.live(appPaths: try makeAppPaths())

        let voice = try await hub.createDesignedVoice(
            label: "British Guide",
            prompt: "warm female voice, British accent, mid-30s"
        )
        let voices = await hub.voiceLibraryStore.list()

        XCTAssertEqual(voice.modelID, ValarServiceHub.qwenVoiceDesignModelID.rawValue)
        XCTAssertEqual(voice.voicePrompt, "warm female voice, British accent, mid-30s")
        XCTAssertEqual(voices.map(\.label), ["British Guide"])
        XCTAssertEqual(voices.first?.voicePrompt, "warm female voice, British accent, mid-30s")
    }

    func testLiveRejectsTraversalProjectTitle() async throws {
        let hub = ValarServiceHub.live(appPaths: try makeAppPaths())

        let project = await hub.createProject(title: "../Secrets")
        let snapshot = await hub.snapshot()

        XCTAssertNil(project)
        XCTAssertEqual(snapshot.projectCount, 0)
    }

    func testDocumentFileURLIsCanonicalBundleLocation() async throws {
        let appPaths = try makeAppPaths()
        let testRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        defer {
            try? FileManager.default.removeItem(at: appPaths.applicationSupport)
            try? FileManager.default.removeItem(at: testRoot)
        }

        try FileManager.default.createDirectory(at: testRoot, withIntermediateDirectories: true)

        let document = ValarProjectDocument(services: ValarServiceHub.live(appPaths: appPaths))
        let projectID = document.bundle.snapshot.project.id
        let originalURL = testRoot
            .appendingPathComponent("Story", isDirectory: true)
            .appendingPathExtension("valarproject")
        let movedURL = testRoot
            .appendingPathComponent("MovedStory", isDirectory: true)
            .appendingPathExtension("valarproject")

        await document.prepareForEditing(fileURL: originalURL)
        let originalLocation = await document.services.projectStore.bundleLocation(for: projectID)

        XCTAssertEqual(originalLocation?.bundleURL, originalURL)
        XCTAssertNotEqual(originalLocation?.bundleURL.deletingLastPathComponent(), appPaths.projectsDirectory)

        await document.prepareForEditing(fileURL: movedURL)
        let movedLocation = await document.services.projectStore.bundleLocation(for: projectID)

        XCTAssertEqual(movedLocation?.bundleURL, movedURL)
        XCTAssertEqual(document.appState.projectDocumentURL, movedURL)
    }

    func testProjectRenderServiceProcessesChaptersSequentiallyAndWritesWAVExports() async throws {
        let appPaths = try makeAppPaths()
        let projectStore = ProjectStore(paths: appPaths)
        let renderQueue = RenderQueue()
        let tracker = RenderInvocationTracker()

        let project = try await projectStore.create(title: "Render Queue")
        await projectStore.updateBundleURL(makeDocumentURL(), for: project.id)
        await projectStore.addChapter(
            ChapterRecord(projectID: project.id, index: 0, title: "Intro", script: "Alpha chapter")
        )
        await projectStore.addChapter(
            ChapterRecord(projectID: project.id, index: 1, title: "Middle", script: "Beta chapter")
        )

        let service = ProjectRenderService(
            renderQueue: renderQueue,
            projectStore: projectStore,
            audioPipeline: AudioPipeline()
        ) { _, options, text in
            await tracker.begin(text, options: options)
            try await Task.sleep(nanoseconds: 50_000_000)
            await tracker.end()
            return makeFakePCMChunk()
        }

        let expectedOptions = RenderSynthesisOptions(
            language: "es",
            temperature: 0.6,
            topP: 0.85,
            repetitionPenalty: 1.15,
            maxTokens: 3_072,
            voiceBehavior: .stableNarrator
        )
        let jobs = await service.enqueueProjectRender(
            project: project,
            modelID: "model",
            synthesisOptions: expectedOptions
        )
        XCTAssertEqual(jobs.count, 2)
        XCTAssertTrue(jobs.allSatisfy { $0.synthesisOptions == expectedOptions })

        let finishedJobs = await waitForTerminalJobs(expectedCount: 2, queue: renderQueue)
        let recordedTexts = await tracker.recordedTexts()
        let maxConcurrency = await tracker.recordedMaxConcurrency()
        let exports = await projectStore.exports(for: project.id)
        let recordedOptions = await tracker.recordedOptions()
        XCTAssertEqual(finishedJobs.map(\.state), [.completed, .completed])
        XCTAssertEqual(recordedTexts, ["Alpha chapter", "Beta chapter"])
        XCTAssertEqual(recordedOptions, [expectedOptions, expectedOptions])
        XCTAssertEqual(maxConcurrency, 1)

        let bundleLocation = await projectStore.bundleLocation(for: project.id)
        let exportsDirectory = try XCTUnwrap(bundleLocation?.exportsDirectory)
        let files = try FileManager.default.contentsOfDirectory(
            at: exportsDirectory,
            includingPropertiesForKeys: nil
        ).sorted { $0.lastPathComponent < $1.lastPathComponent }

        XCTAssertEqual(files.map(\.lastPathComponent), ["001-intro.wav", "002-middle.wav"])
        XCTAssertEqual(exports.map(\.fileName), ["001-intro.wav", "002-middle.wav"])

        let header = try Data(contentsOf: files[0]).prefix(4)
        XCTAssertEqual(String(data: header, encoding: .ascii), "RIFF")
    }

    func testProjectRenderServiceSkipsBlankChaptersWhenQueueingProjectRender() async throws {
        let appPaths = try makeAppPaths()
        let projectStore = ProjectStore(paths: appPaths)
        let renderQueue = RenderQueue()
        let tracker = RenderInvocationTracker()

        let project = try await projectStore.create(title: "Renderable Chapters Only")
        await projectStore.updateBundleURL(makeDocumentURL(), for: project.id)
        await projectStore.addChapter(
            ChapterRecord(projectID: project.id, index: 0, title: "Draft", script: "   \n")
        )
        await projectStore.addChapter(
            ChapterRecord(projectID: project.id, index: 1, title: "Narration", script: "Actual content")
        )

        let service = ProjectRenderService(
            renderQueue: renderQueue,
            projectStore: projectStore,
            audioPipeline: AudioPipeline()
        ) { _, options, text in
            await tracker.begin(text, options: options)
            await tracker.end()
            return makeFakePCMChunk()
        }

        let jobs = await service.enqueueProjectRender(
            project: project,
            modelID: "model",
            synthesisOptions: RenderSynthesisOptions()
        )
        XCTAssertEqual(jobs.count, 1)
        XCTAssertEqual(jobs.first?.title, "Narration")

        let finishedJobs = await waitForTerminalJobs(expectedCount: 1, queue: renderQueue)
        let recordedTexts = await tracker.recordedTexts()

        XCTAssertEqual(finishedJobs.map(\.state), [.completed])
        XCTAssertEqual(recordedTexts, ["Actual content"])
    }

    func testProjectRenderServiceCancelsInflightRenderWithoutWritingExport() async throws {
        let appPaths = try makeAppPaths()
        let projectStore = ProjectStore(paths: appPaths)
        let renderQueue = RenderQueue()
        let tracker = RenderInvocationTracker()

        let project = try await projectStore.create(title: "Cancelled Queue")
        await projectStore.updateBundleURL(makeDocumentURL(), for: project.id)
        await projectStore.addChapter(
            ChapterRecord(projectID: project.id, index: 0, title: "Intro", script: "Never finishes")
        )

        let service = ProjectRenderService(
            renderQueue: renderQueue,
            projectStore: projectStore,
            audioPipeline: AudioPipeline()
        ) { _, options, text in
            await tracker.begin(text, options: options)
            do {
                while true {
                    try Task.checkCancellation()
                    try await Task.sleep(nanoseconds: 20_000_000)
                }
            } catch {
                await tracker.end()
                throw error
            }
        }

        let jobs = await service.enqueueProjectRender(
            project: project,
            modelID: "model",
            synthesisOptions: RenderSynthesisOptions()
        )
        let jobID = try XCTUnwrap(jobs.first?.id)

        await waitForJobState(.running, jobID: jobID, queue: renderQueue)
        await service.cancel(jobID)

        let finalJob = await waitForJob(jobID: jobID, queue: renderQueue) { job in
            job?.state == .cancelled
        }
        let exports = await projectStore.exports(for: project.id)
        XCTAssertEqual(finalJob?.state, .cancelled)
        XCTAssertEqual(exports, [])

        let bundleLocation = await projectStore.bundleLocation(for: project.id)
        let outputURL = bundleLocation?.exportsDirectory.appendingPathComponent("001-intro.wav", isDirectory: false)
        XCTAssertFalse(FileManager.default.fileExists(atPath: outputURL?.path ?? ""))
    }

    func testProjectRenderServiceUsesUpdatedChapterText() async throws {
        let appPaths = try makeAppPaths()
        let projectStore = ProjectStore(paths: appPaths)
        let renderQueue = RenderQueue()
        let tracker = RenderInvocationTracker()

        let project = try await projectStore.create(title: "Edited Queue")
        await projectStore.updateBundleURL(makeDocumentURL(), for: project.id)
        var chapter = ChapterRecord(projectID: project.id, index: 0, title: "Intro", script: "Old text")
        await projectStore.addChapter(chapter)
        chapter.script = "New text"
        await projectStore.updateChapter(chapter)

        let service = ProjectRenderService(
            renderQueue: renderQueue,
            projectStore: projectStore,
            audioPipeline: AudioPipeline()
        ) { _, options, text in
            await tracker.begin(text, options: options)
            await tracker.end()
            return makeFakePCMChunk()
        }

        _ = await service.enqueueProjectRender(
            project: project,
            modelID: "model",
            synthesisOptions: RenderSynthesisOptions()
        )
        _ = await waitForTerminalJobs(expectedCount: 1, queue: renderQueue)

        let recordedTexts = await tracker.recordedTexts()
        XCTAssertEqual(recordedTexts, ["New text"])
    }

    func testGeneratorStateStreamsChunksIntoAudioPlayer() async throws {
        let firstChunk = makeFakePCMChunk(samples: [0, 0.25])
        let secondChunk = makeFakePCMChunk(samples: [-0.25, 0.1, 0.2])
        let totalDuration = Double(5) / 24_000

        let audioPlayer = StubAudioPlayer(
            snapshots: [
                AudioPlaybackSnapshot(position: Double(2) / 24_000, queuedDuration: Double(3) / 24_000, isPlaying: true, isBuffering: false, didFinish: false),
                AudioPlaybackSnapshot(position: totalDuration, queuedDuration: 0, isPlaying: false, isBuffering: false, didFinish: true),
            ]
        )
        let streamFactory: @Sendable () -> AsyncThrowingStream<AudioChunk, Error> = {
            AsyncThrowingStream { continuation in
                continuation.yield(firstChunk)
                continuation.yield(secondChunk)
                continuation.finish()
            }
        }

        let (state, modelID) = try await makeGeneratorState(
            streamFactory: streamFactory,
            audioPlayer: audioPlayer
        )
        state.selectedModelID = modelID
        state.text = "Speak now"

        await state.generate()
        try await Task.sleep(nanoseconds: 30_000_000)
        let audioMetrics = await audioPlayer.metrics()

        XCTAssertTrue(state.hasAudio)
        XCTAssertEqual(state.audioDuration, totalDuration, accuracy: 0.000_001)
        XCTAssertEqual(state.playbackPosition, totalDuration, accuracy: 0.000_001)
        XCTAssertFalse(state.isPlaying)
        XCTAssertFalse(state.isPlaybackBuffering)
        XCTAssertEqual(audioMetrics.feedSamplesCount, 2)
        XCTAssertEqual(audioMetrics.finishStreamCallCount, 1)
    }

    func testGeneratorStateShowsBufferingDuringStreamingUnderrun() async throws {
        let firstChunk = makeFakePCMChunk(samples: [0, 0.25])
        let secondChunk = makeFakePCMChunk(samples: [-0.25, 0.1])
        let totalDuration = Double(4) / 24_000

        let audioPlayer = StubAudioPlayer(
            snapshots: [
                AudioPlaybackSnapshot(position: Double(1) / 24_000, queuedDuration: Double(1) / 24_000, isPlaying: true, isBuffering: false, didFinish: false),
                AudioPlaybackSnapshot(position: Double(2) / 24_000, queuedDuration: 0, isPlaying: false, isBuffering: true, didFinish: false),
                AudioPlaybackSnapshot(position: Double(3) / 24_000, queuedDuration: Double(1) / 24_000, isPlaying: true, isBuffering: false, didFinish: false),
                AudioPlaybackSnapshot(position: totalDuration, queuedDuration: 0, isPlaying: false, isBuffering: false, didFinish: true),
            ]
        )
        let streamFactory: @Sendable () -> AsyncThrowingStream<AudioChunk, Error> = {
            AsyncThrowingStream { continuation in
                Task {
                    continuation.yield(firstChunk)
                    try? await Task.sleep(nanoseconds: 200_000_000)
                    continuation.yield(secondChunk)
                    continuation.finish()
                }
            }
        }

        let (state, modelID) = try await makeGeneratorState(
            streamFactory: streamFactory,
            audioPlayer: audioPlayer,
            playbackPollInterval: .milliseconds(30)
        )
        state.selectedModelID = modelID
        state.text = "Wait for more audio"

        let generationTask = Task { @MainActor in
            await state.generate()
        }

        await waitForCondition(timeoutSeconds: 1) {
            state.isPlaybackBuffering
        }

        await generationTask.value
        try await Task.sleep(nanoseconds: 30_000_000)
        let audioMetrics = await audioPlayer.metrics()

        XCTAssertTrue(state.hasAudio)
        XCTAssertEqual(state.audioDuration, totalDuration, accuracy: 0.000_001)
        XCTAssertFalse(state.isPlaybackBuffering)
        XCTAssertFalse(state.isPlaying)
        XCTAssertEqual(audioMetrics.feedSamplesCount, 2)
    }

    func testGeneratorStateStopsReplayImmediatelyWhenToggledOff() async throws {
        let chunk = makeFakePCMChunk(samples: [0, 0.25, -0.25, 0.1])
        let audioPlayer = StubAudioPlayer(
            snapshots: [
                AudioPlaybackSnapshot(position: chunk.sampleRate > 0 ? Double(4) / chunk.sampleRate : 0, queuedDuration: 0, isPlaying: false, isBuffering: false, didFinish: true),
                AudioPlaybackSnapshot(position: chunk.sampleRate > 0 ? Double(2) / chunk.sampleRate : 0, queuedDuration: 0, isPlaying: true, isBuffering: false, didFinish: false),
            ]
        )
        let streamFactory: @Sendable () -> AsyncThrowingStream<AudioChunk, Error> = {
            AsyncThrowingStream { continuation in
                continuation.yield(chunk)
                continuation.finish()
            }
        }

        let (state, modelID) = try await makeGeneratorState(
            streamFactory: streamFactory,
            audioPlayer: audioPlayer
        )
        state.selectedModelID = modelID
        state.text = "Replay and stop"

        await state.generate()
        await waitForCondition(timeoutSeconds: 1) {
            state.hasAudio && !state.isPlaying && !state.isPlaybackBuffering
        }

        state.togglePlayback()
        state.togglePlayback()
        try await Task.sleep(nanoseconds: 20_000_000)

        let audioMetrics = await audioPlayer.metrics()
        XCTAssertFalse(state.isPlaying)
        XCTAssertFalse(state.isPlaybackBuffering)
        XCTAssertEqual(state.playbackPosition, 0, accuracy: 0.000_001)
        XCTAssertGreaterThanOrEqual(audioMetrics.stopCallCount, 2)
        XCTAssertLessThanOrEqual(audioMetrics.playedCount, 1)
    }

    func testGeneratorStateForwardsInlineReferenceAudioForTadaModels() async throws {
        let recorder = RequestRecorder()
        let referenceURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("wav")
        let referenceAudioData = try await AudioPipeline().transcode(
            AudioPCMBuffer(mono: [0.05, -0.05, 0.1, -0.1], sampleRate: 24_000),
            container: "wav"
        ).data
        try referenceAudioData.write(to: referenceURL)

        let tadaDescriptor = ModelDescriptor(
            id: TadaCatalog.tada3BModelIdentifier,
            familyID: .tadaTTS,
            displayName: "TADA 3B",
            domain: .tts,
            capabilities: [.speechSynthesis, .voiceCloning, .audioConditioning, .multilingual],
            defaultSampleRate: 24_000
        )
        let (state, modelID) = try await makeGeneratorState(
            descriptor: tadaDescriptor,
            streamFactory: {
                AsyncThrowingStream { continuation in
                    continuation.yield(makeFakePCMChunk(samples: [0.1, 0.2]))
                    continuation.finish()
                }
            },
            audioPlayer: StubAudioPlayer(snapshots: []),
            requestHandler: { request in
                Task {
                    await recorder.record(request)
                }
            }
        )
        state.selectedModelID = modelID
        state.text = "Clone this"
        state.selectedLanguage = "fr"
        try state.selectReferenceAudio(referenceURL)
        state.referenceTranscript = "Bonjour"

        await state.generate()

        try await Task.sleep(nanoseconds: 20_000_000)
        let recordedRequest = await recorder.latestRequest()
        let request = try XCTUnwrap(recordedRequest)
        XCTAssertNotNil(request.referenceAudioPCMFloat32LE)
        XCTAssertEqual(request.referenceTranscript, "Bonjour")
        XCTAssertEqual(request.referenceAudioAssetName, referenceURL.lastPathComponent)
        XCTAssertEqual(request.language, "fr")
    }

    func testGeneratorStateForwardsSelectedVoxtralPresetVoice() async throws {
        let recorder = RequestRecorder()
        let voxtralDescriptor = ModelDescriptor(
            id: "mlx-community/Voxtral-4B-TTS-2603-mlx-4bit",
            familyID: .voxtralTTS,
            displayName: "Voxtral 4B TTS 2603 MLX (4-bit)",
            domain: .tts,
            capabilities: [.speechSynthesis, .presetVoices, .streaming],
            defaultSampleRate: 24_000
        )
        let voxtralPreset = VoiceLibraryRecord(
            label: "Emma",
            modelID: voxtralDescriptor.id.rawValue,
            runtimeModelID: voxtralDescriptor.id.rawValue,
            backendVoiceID: "neutral_female"
        )
        let (state, _) = try await makeGeneratorState(
            descriptor: voxtralDescriptor,
            streamFactory: {
                AsyncThrowingStream { continuation in
                    continuation.yield(makeFakePCMChunk(samples: [0.1, 0.2]))
                    continuation.finish()
                }
            },
            audioPlayer: StubAudioPlayer(snapshots: []),
            requestHandler: { request in
                Task {
                    await recorder.record(request)
                }
            }
        )
        state.availableVoices = [voxtralPreset]
        state.selectVoice(voxtralPreset.id)
        state.text = "Preset voice path"
        state.selectedLanguage = "en"
        state.voiceBehavior = .stableNarrator

        await state.generate()

        try await Task.sleep(nanoseconds: 20_000_000)
        let latestRequest = await recorder.latestRequest()
        let request = try XCTUnwrap(latestRequest)
        XCTAssertEqual(request.model, voxtralDescriptor.id)
        XCTAssertEqual(request.language, "en")
        XCTAssertEqual(request.voice?.backendVoiceID, "neutral_female")
        XCTAssertEqual(request.voice?.voiceKind, .preset)
        XCTAssertEqual(request.voiceBehavior, .stableNarrator)
    }

    func testGeneratorStateResetsPlaybackStateWhenReplayStartFails() async throws {
        let chunk = makeFakePCMChunk(samples: [0, 0.25, -0.25, 0.1])
        let audioPlayer = StubAudioPlayer(
            snapshots: [
                AudioPlaybackSnapshot(position: chunk.sampleRate > 0 ? Double(4) / chunk.sampleRate : 0, queuedDuration: 0, isPlaying: false, isBuffering: false, didFinish: true),
            ],
            playError: StubAudioPlayerError.playFailed
        )
        let streamFactory: @Sendable () -> AsyncThrowingStream<AudioChunk, Error> = {
            AsyncThrowingStream { continuation in
                continuation.yield(chunk)
                continuation.finish()
            }
        }

        let (state, modelID) = try await makeGeneratorState(
            streamFactory: streamFactory,
            audioPlayer: audioPlayer
        )
        state.selectedModelID = modelID
        state.text = "Replay failure"

        await state.generate()
        await waitForCondition(timeoutSeconds: 1) {
            state.hasAudio && !state.isPlaying && !state.isPlaybackBuffering
        }

        state.togglePlayback()
        await waitForCondition(timeoutSeconds: 1) {
            !state.isPlaying && !state.isPlaybackBuffering && state.playbackPosition == 0
        }

        let audioMetrics = await audioPlayer.metrics()
        XCTAssertEqual(audioMetrics.playedCount, 0)
        XCTAssertEqual(audioMetrics.finishStreamCallCount, 1)
    }

    private func waitForTerminalJobs(
        expectedCount: Int,
        queue: RenderQueue,
        timeoutSeconds: TimeInterval = 2
    ) async -> [RenderJob] {
        let start = Date()
        while true {
            let jobs = await queue.jobs(matching: nil)
            if jobs.count == expectedCount,
               jobs.allSatisfy({ $0.state == .completed || $0.state == .cancelled || $0.state == .failed }) {
                return jobs
            }

            if Date().timeIntervalSince(start) > timeoutSeconds {
                XCTFail("Timed out waiting for terminal render jobs")
                return jobs
            }

            try? await Task.sleep(nanoseconds: 20_000_000)
        }
    }

    private func waitForJobState(
        _ state: RenderJobState,
        jobID: UUID,
        queue: RenderQueue
    ) async {
        _ = await waitForJob(jobID: jobID, queue: queue) { $0?.state == state }
    }

    private func waitForJob(
        jobID: UUID,
        queue: RenderQueue,
        predicate: @escaping (RenderJob?) -> Bool
    ) async -> RenderJob? {
        let start = Date()
        while true {
            let job = await queue.job(id: jobID)
            if predicate(job) {
                return job
            }
            if Date().timeIntervalSince(start) > 2 {
                XCTFail("Timed out waiting for render job \(jobID)")
                return job
            }
            try? await Task.sleep(nanoseconds: 20_000_000)
        }
    }

    @MainActor
    private func waitForCondition(
        timeoutSeconds: TimeInterval = 1,
        predicate: @escaping () -> Bool
    ) async {
        let start = Date()
        while true {
            if predicate() {
                return
            }
            if Date().timeIntervalSince(start) > timeoutSeconds {
                XCTFail("Timed out waiting for condition")
                return
            }
            try? await Task.sleep(nanoseconds: 20_000_000)
        }
    }

    private func makeGeneratorState(
        descriptor: ModelDescriptor? = nil,
        supportedRecords: [SupportedModelCatalogRecord] = [],
        installedRecords: [InstalledModelRecord] = [],
        streamFactory: @escaping @Sendable () -> AsyncThrowingStream<AudioChunk, Error>,
        audioPlayer: StubAudioPlayer,
        playbackPollInterval: Duration = .milliseconds(5),
        requestHandler: (@Sendable (SpeechSynthesisRequest) -> Void)? = nil
    ) async throws -> (GeneratorState, ModelIdentifier) {
        let appPaths = try makeAppPaths()
        let runtimeConfiguration = RuntimeConfiguration()
        let descriptor = descriptor ?? ModelDescriptor(
            id: "test/streaming-qwen",
            familyID: .qwen3TTS,
            displayName: "Streaming Test Model",
            domain: .tts,
            capabilities: [.speechSynthesis],
            defaultSampleRate: 24_000
        )
        let modelRegistry = ModelRegistry(configuration: runtimeConfiguration)
        let capabilityRegistry = CapabilityRegistry()
        await modelRegistry.register(descriptor)
        await capabilityRegistry.register(descriptor)

        let modelPackRegistry = StubModelPackRegistry(
            installedRecords: installedRecords,
            supportedRecords: supportedRecords
        )
        let modelCatalog = ModelCatalog(
            supportedSource: StaticSupportedCatalogSource(records: []),
            catalogStore: modelPackRegistry,
            packStore: modelPackRegistry,
            capabilityRegistry: capabilityRegistry
        )
        let modelInstaller = ModelInstaller(
            registry: modelPackRegistry,
            modelRegistry: modelRegistry,
            capabilityRegistry: capabilityRegistry,
            paths: appPaths
        )
        let projectStore = StubProjectStore()
        let voiceLibraryStore = StubVoiceLibraryStore()
        let renderQueue = RenderQueue(configuration: runtimeConfiguration)
        let audioPipeline = AudioPipeline()
        let projectRenderService = ProjectRenderService(
            renderQueue: renderQueue,
            projectStore: projectStore,
            audioPipeline: audioPipeline
        ) { _, _, _ in
            makeFakePCMChunk()
        }
        let parityHarness = ParityHarness(
            projectStore: projectStore,
            voiceLibraryStore: voiceLibraryStore
        )
        let inferenceBackend = StubInferenceBackend(
            model: StubStreamingTTSModel(
                descriptor: descriptor,
                streamFactory: streamFactory,
                requestHandler: requestHandler
            )
        )
        let grdbRenderJobStore = GRDBRenderJobStore(db: try AppDatabase.inMemory())
        let services = ValarServiceHub(
            appPaths: appPaths,
            runtimeConfiguration: runtimeConfiguration,
            modelRegistry: modelRegistry,
            capabilityRegistry: capabilityRegistry,
            modelPackRegistry: modelPackRegistry,
            modelCatalog: modelCatalog,
            modelInstaller: modelInstaller,
            inferenceBackend: inferenceBackend,
            renderQueue: renderQueue,
            grdbRenderJobStore: grdbRenderJobStore,
            projectRenderService: projectRenderService,
            grdbProjectStore: GRDBProjectStore(db: try AppDatabase.inMemory(), paths: appPaths),
            projectStore: projectStore,
            grdbVoiceStore: GRDBVoiceStore(db: try AppDatabase.inMemory()),
            voiceLibraryStore: voiceLibraryStore,
            dictationService: DictationService(),
            translationService: TranslationService(provider: StubTranslationProvider()),
            parityHarness: parityHarness,
            audioPipeline: audioPipeline,
            audioPlayer: audioPlayer,
            projectExporter: StubProjectExporter()
        )

        return (GeneratorState(services: services, playbackPollInterval: playbackPollInterval), descriptor.id)
    }

    private func makeImportManifest(
        modelID: String = "custom/\(UUID().uuidString)",
        displayName: String = "Imported Model",
        artifactPath: String = "weights/model.safetensors"
    ) -> ValarPersistence.ModelPackManifest {
        ValarPersistence.ModelPackManifest(
            id: modelID,
            familyID: ModelFamilyID.qwen3TTS.rawValue,
            modelID: modelID,
            displayName: displayName,
            capabilities: [ModelCapability.speechSynthesis.rawValue],
            backendKinds: [BackendKind.mlx.rawValue],
            tokenizerType: "qwen",
            sampleRate: 24_000,
            artifactSpecs: [
                ModelPackArtifact(
                    id: "weights",
                    kind: "weights",
                    relativePath: artifactPath
                )
            ],
            licenseName: "Apache-2.0",
            notes: "Imported in unit tests"
        )
    }

    private func makeModelBundle(
        manifest: ValarPersistence.ModelPackManifest,
        files: [String: Data]
    ) throws -> URL {
        let bundleURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
            .appendingPathExtension("valarmodel")
        try FileManager.default.createDirectory(at: bundleURL, withIntermediateDirectories: true)

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(manifest).write(
            to: bundleURL.appendingPathComponent("manifest.json", isDirectory: false),
            options: .atomic
        )

        for (relativePath, data) in files {
            let fileURL = bundleURL.appendingPathComponent(relativePath, isDirectory: false)
            try FileManager.default.createDirectory(
                at: fileURL.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            try data.write(to: fileURL, options: .atomic)
        }

        return bundleURL
    }

    func testProjectWorkspaceAttachAndTranscribeChapterSourceAudioPersistsResults() async throws {
        let recorder = RecognitionAlignmentRequestRecorder()
        let (services, projectStore, asrModelID, _) = try await makeProjectWorkspaceAudioServices(
            requestRecorder: recorder
        )

        let project = try await projectStore.create(title: "Audio Project", notes: nil)
        let chapter = ChapterRecord(projectID: project.id, index: 0, title: "Chapter 1", script: "")
        await projectStore.addChapter(chapter)

        let bundleURL = makeDocumentURL()
        try FileManager.default.createDirectory(at: bundleURL, withIntermediateDirectories: true)
        await projectStore.updateBundleURL(bundleURL, for: project.id)

        let state = ProjectWorkspaceState(services: services)
        await state.load(project: project, preferredModelID: nil)

        let sourceAudioURL = try await makeProjectSourceAudioURL()
        await state.attachSourceAudio(from: sourceAudioURL)

        XCTAssertEqual(state.selectedChapter?.sourceAudioAssetName, "chapter-source-\(chapter.id.uuidString).wav")
        XCTAssertNil(state.selectedChapter?.transcriptionJSON)
        XCTAssertNil(state.selectedChapter?.alignmentJSON)

        state.chapterAudioLanguageHint = "en"
        await state.transcribeSelectedChapter()

        let updatedChapter = try XCTUnwrap(state.selectedChapter)
        XCTAssertEqual(updatedChapter.transcriptionModelID, asrModelID.rawValue)
        let transcriptionJSON = try XCTUnwrap(updatedChapter.transcriptionJSON)
        let transcription = try JSONDecoder().decode(
            RichTranscriptionResult.self,
            from: XCTUnwrap(transcriptionJSON.data(using: .utf8))
        )
        XCTAssertEqual(transcription.language, "en")
        XCTAssertTrue(transcription.text.contains(asrModelID.rawValue))

        let recordedRequest = await recorder.latestTranscriptionRequest()
        XCTAssertEqual(recordedRequest?.model, asrModelID)
        XCTAssertEqual(recordedRequest?.languageHint, "en")
        XCTAssertNotNil(recordedRequest?.audioChunk)

        let persistedAssetURL = bundleURL
            .appendingPathComponent("Assets", isDirectory: true)
            .appendingPathComponent("chapter-source-\(chapter.id.uuidString).wav", isDirectory: false)
        XCTAssertTrue(FileManager.default.fileExists(atPath: persistedAssetURL.path))
    }

    func testProjectWorkspaceAlignSelectedChapterPersistsAlignmentUsingScript() async throws {
        let recorder = RecognitionAlignmentRequestRecorder()
        let (services, projectStore, _, alignModelID) = try await makeProjectWorkspaceAudioServices(
            requestRecorder: recorder
        )

        let project = try await projectStore.create(title: "Aligned Project", notes: nil)
        let script = "Hello aligned world"
        let chapter = ChapterRecord(projectID: project.id, index: 0, title: "Chapter 1", script: script)
        await projectStore.addChapter(chapter)

        let bundleURL = makeDocumentURL()
        try FileManager.default.createDirectory(at: bundleURL, withIntermediateDirectories: true)
        await projectStore.updateBundleURL(bundleURL, for: project.id)

        let state = ProjectWorkspaceState(services: services)
        await state.load(project: project, preferredModelID: nil)

        let sourceAudioURL = try await makeProjectSourceAudioURL()
        await state.attachSourceAudio(from: sourceAudioURL)
        state.chapterAudioLanguageHint = "fr"
        await state.alignSelectedChapter()

        let updatedChapter = try XCTUnwrap(state.selectedChapter)
        XCTAssertEqual(updatedChapter.alignmentModelID, alignModelID.rawValue)
        let alignmentJSON = try XCTUnwrap(updatedChapter.alignmentJSON)
        let alignment = try JSONDecoder().decode(
            ForcedAlignmentResult.self,
            from: XCTUnwrap(alignmentJSON.data(using: .utf8))
        )
        XCTAssertEqual(alignment.transcript, script)
        XCTAssertEqual(alignment.tokens.count, 2)

        let recordedRequest = await recorder.latestAlignmentRequest()
        XCTAssertEqual(recordedRequest?.model, alignModelID)
        XCTAssertEqual(recordedRequest?.languageHint, "fr")
        XCTAssertEqual(recordedRequest?.transcript, script)
    }

    func testProjectWorkspaceAlignFallsBackToStoredTranscriptionWhenScriptIsEmpty() async throws {
        let recorder = RecognitionAlignmentRequestRecorder()
        let (services, projectStore, _, alignModelID) = try await makeProjectWorkspaceAudioServices(
            requestRecorder: recorder
        )

        let project = try await projectStore.create(title: "Fallback Project", notes: nil)
        let chapter = ChapterRecord(projectID: project.id, index: 0, title: "Chapter 1", script: "")
        await projectStore.addChapter(chapter)

        let bundleURL = makeDocumentURL()
        try FileManager.default.createDirectory(at: bundleURL, withIntermediateDirectories: true)
        await projectStore.updateBundleURL(bundleURL, for: project.id)

        let state = ProjectWorkspaceState(services: services)
        await state.load(project: project, preferredModelID: nil)

        let sourceAudioURL = try await makeProjectSourceAudioURL()
        await state.attachSourceAudio(from: sourceAudioURL)
        await state.transcribeSelectedChapter()
        await state.alignSelectedChapter()

        let latestAlignmentRequest = await recorder.latestAlignmentRequest()
        let recordedRequest = try XCTUnwrap(latestAlignmentRequest)
        XCTAssertEqual(recordedRequest.model, alignModelID)
        XCTAssertEqual(recordedRequest.transcript, "Detected speech for mlx-community/Qwen3-ASR-0.6B-8bit")
    }

    private func makeProjectWorkspaceAudioServices(
        requestRecorder: RecognitionAlignmentRequestRecorder
    ) async throws -> (ValarServiceHub, ProjectStore, ModelIdentifier, ModelIdentifier) {
        let appPaths = try makeAppPaths()
        let runtimeConfiguration = RuntimeConfiguration()
        let modelRegistry = ModelRegistry(configuration: runtimeConfiguration)
        let capabilityRegistry = CapabilityRegistry()
        let asrModelID: ModelIdentifier = "mlx-community/Qwen3-ASR-0.6B-8bit"
        let alignModelID: ModelIdentifier = "mlx-community/Qwen3-ForcedAligner-0.6B-8bit"

        let installRoot = appPaths.modelPacksDirectory.appendingPathComponent("tests", isDirectory: true)
        try FileManager.default.createDirectory(at: installRoot, withIntermediateDirectories: true)
        let installedRecords = try [
            makeInstalledModelRecord(
                modelID: asrModelID,
                familyID: .qwen3ASR,
                displayName: "Qwen3-ASR 0.6B 8bit",
                installRoot: installRoot
            ),
            makeInstalledModelRecord(
                modelID: alignModelID,
                familyID: .qwen3ForcedAligner,
                displayName: "Qwen3-ForcedAligner 0.6B 8bit",
                installRoot: installRoot
            ),
        ]
        let modelPackRegistry = StubModelPackRegistry(installedRecords: installedRecords)
        let modelCatalog = ModelCatalog(
            supportedSource: SupportedCatalogSource.curated(),
            catalogStore: modelPackRegistry,
            packStore: modelPackRegistry,
            capabilityRegistry: capabilityRegistry
        )
        let modelInstaller = ModelInstaller(
            registry: modelPackRegistry,
            modelRegistry: modelRegistry,
            capabilityRegistry: capabilityRegistry,
            paths: appPaths
        )
        let projectStore = ProjectStore(paths: appPaths)
        let renderQueue = RenderQueue(configuration: runtimeConfiguration)
        let audioPipeline = AudioPipeline()
        let descriptor = ModelDescriptor(
            id: "test/qwen-analysis",
            familyID: .qwen3ASR,
            displayName: "Stub Qwen Analysis",
            domain: .stt,
            capabilities: [.speechRecognition, .forcedAlignment],
            defaultSampleRate: 16_000
        )
        let services = ValarServiceHub(
            appPaths: appPaths,
            runtimeConfiguration: runtimeConfiguration,
            modelRegistry: modelRegistry,
            capabilityRegistry: capabilityRegistry,
            modelPackRegistry: modelPackRegistry,
            modelCatalog: modelCatalog,
            modelInstaller: modelInstaller,
            inferenceBackend: StubInferenceBackend(
                model: StubRecognitionAlignmentModel(
                    descriptor: descriptor,
                    requestRecorder: requestRecorder
                )
            ),
            renderQueue: renderQueue,
            grdbRenderJobStore: GRDBRenderJobStore(db: try AppDatabase.inMemory()),
            projectRenderService: ProjectRenderService(
                renderQueue: renderQueue,
                projectStore: projectStore,
                audioPipeline: audioPipeline
            ) { _, _, _ in
                makeFakePCMChunk()
            },
            grdbProjectStore: GRDBProjectStore(db: try AppDatabase.inMemory(), paths: appPaths),
            projectStore: projectStore,
            grdbVoiceStore: GRDBVoiceStore(db: try AppDatabase.inMemory()),
            voiceLibraryStore: StubVoiceLibraryStore(),
            dictationService: DictationService(),
            translationService: TranslationService(provider: StubTranslationProvider()),
            audioPipeline: audioPipeline,
            audioPlayer: StubAudioPlayer(snapshots: []),
            projectExporter: StubProjectExporter()
        )

        return (services, projectStore, asrModelID, alignModelID)
    }

    private func makeInstalledModelRecord(
        modelID: ModelIdentifier,
        familyID: ModelFamilyID,
        displayName: String,
        installRoot: URL
    ) throws -> InstalledModelRecord {
        let directory = installRoot
            .appendingPathComponent(modelID.rawValue.replacingOccurrences(of: "/", with: "-"), isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let manifestURL = directory.appendingPathComponent("manifest.json", isDirectory: false)
        try Data("{}".utf8).write(to: manifestURL, options: .atomic)
        let artifactRelativePaths = SupportedModelCatalog.entry(for: modelID)?
            .manifest
            .artifacts
            .map(\.relativePath) ?? ["weights/model.safetensors"]
        for relativePath in artifactRelativePaths {
            let artifactURL = directory.appendingPathComponent(relativePath, isDirectory: false)
            try FileManager.default.createDirectory(
                at: artifactURL.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            if !FileManager.default.fileExists(atPath: artifactURL.path) {
                try Data([0]).write(to: artifactURL, options: .atomic)
            }
        }
        return InstalledModelRecord(
            familyID: familyID.rawValue,
            modelID: modelID.rawValue,
            displayName: displayName,
            installedPath: directory.path,
            manifestPath: manifestURL.path,
            sourceKind: .localFolder
        )
    }

    private func makeProjectSourceAudioURL() async throws -> URL {
        let sourceURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("wav")
        let audioData = try await AudioPipeline().transcode(
            AudioPCMBuffer(mono: [0.05, -0.05, 0.1, -0.1, 0.02, -0.02], sampleRate: 24_000),
            container: "wav"
        )
        try audioData.data.write(to: sourceURL, options: .atomic)
        return sourceURL
    }
}
