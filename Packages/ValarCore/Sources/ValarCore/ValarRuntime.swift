import Foundation
import os
import ValarAudio
import ValarMLX
import ValarModelKit
import ValarPersistence

/// Shared non-UI runtime host for app, CLI, and daemon entry points.
public final class ValarRuntime: Sendable {
    public enum TranscriptionError: Error, Sendable, LocalizedError, Equatable {
        case modelNotFound(ModelIdentifier)
        case modelNotInstalled(ModelIdentifier)
        case unsupportedModelFamily(identifier: ModelIdentifier, family: ModelFamilyID)
        case noCompatibleBackend(ModelIdentifier)
        case loadedModelTypeMismatch(ModelIdentifier)

        public var errorDescription: String? {
            switch self {
            case .modelNotFound(let identifier):
                return "Speech-to-text model not found: \(identifier.rawValue)"
            case .modelNotInstalled(let identifier):
                return "Speech-to-text model is not installed: \(identifier.rawValue)"
            case .unsupportedModelFamily(let identifier, let family):
                return "Model \(identifier.rawValue) belongs to unsupported speech-to-text family \(family.rawValue)."
            case .noCompatibleBackend(let identifier):
                return "No compatible inference backend is available for speech-to-text model \(identifier.rawValue)."
            case .loadedModelTypeMismatch(let identifier):
                return "Loaded model \(identifier.rawValue) does not provide a speech-to-text workflow."
            }
        }
    }

    public enum AlignmentError: Error, Sendable, LocalizedError, Equatable {
        case modelNotFound(ModelIdentifier)
        case modelNotInstalled(ModelIdentifier)
        case unsupportedModelFamily(identifier: ModelIdentifier, family: ModelFamilyID)
        case noCompatibleBackend(ModelIdentifier)
        case loadedModelTypeMismatch(ModelIdentifier)

        public var errorDescription: String? {
            switch self {
            case .modelNotFound(let identifier):
                return "Forced-alignment model not found: \(identifier.rawValue)"
            case .modelNotInstalled(let identifier):
                return "Forced-alignment model is not installed: \(identifier.rawValue)"
            case .unsupportedModelFamily(let identifier, let family):
                return "Model \(identifier.rawValue) belongs to unsupported forced-alignment family \(family.rawValue)."
            case .noCompatibleBackend(let identifier):
                return "No compatible inference backend is available for forced-alignment model \(identifier.rawValue)."
            case .loadedModelTypeMismatch(let identifier):
                return "Loaded model \(identifier.rawValue) does not provide a forced-alignment workflow."
            }
        }
    }

    public let paths: ValarAppPaths
    public let runtimeConfiguration: RuntimeConfiguration
    public let database: AppDatabase
    public let grdbProjectStore: GRDBProjectStore
    public let grdbVoiceStore: GRDBVoiceStore
    public let grdbRenderJobStore: GRDBRenderJobStore
    public let modelRegistry: ModelRegistry
    public let capabilityRegistry: CapabilityRegistry
    public let modelPackRegistry: any ModelPackManaging
    public let modelCatalog: ModelCatalog
    public let modelInstaller: ModelInstaller
    public let inferenceBackend: any InferenceBackend
    public let audioPipeline: AudioPipeline
    public let renderQueue: RenderQueue
    public let projectStore: any ProjectStoring
    public let voiceStore: any VoiceLibraryStoring
    public let voicePromptCache: VoicePromptCache
    public let resourceMonitor: RuntimeResourceSampler
    public let activeSynthesisTracker: ActiveSynthesisTracker
    public let daemonIdleTrimTracker: DaemonIdleTrimTracker
    private let sessionStore: DocumentSessionStore
    let startupMaintenanceCoordinator: RuntimeStartupMaintenanceCoordinator

    public init(
        paths: ValarAppPaths = ValarAppPaths(),
        runtimeConfiguration: RuntimeConfiguration = RuntimeConfiguration.configured(
            from: ProcessInfo.processInfo.environment
        ),
        database: AppDatabase,
        grdbProjectStore: GRDBProjectStore,
        grdbVoiceStore: GRDBVoiceStore,
        grdbRenderJobStore: GRDBRenderJobStore,
        modelRegistry: ModelRegistry,
        capabilityRegistry: CapabilityRegistry,
        modelPackRegistry: any ModelPackManaging,
        modelCatalog: ModelCatalog,
        modelInstaller: ModelInstaller,
        inferenceBackend: any InferenceBackend,
        audioPipeline: AudioPipeline,
        renderQueue: RenderQueue,
        projectStore: any ProjectStoring,
        voiceStore: any VoiceLibraryStoring,
        resourceMonitor: RuntimeResourceSampler = RuntimeResourceSampler(),
        activeSynthesisTracker: ActiveSynthesisTracker = ActiveSynthesisTracker(),
        daemonIdleTrimTracker: DaemonIdleTrimTracker = DaemonIdleTrimTracker()
    ) {
        self.paths = paths
        self.runtimeConfiguration = runtimeConfiguration
        self.database = database
        self.grdbProjectStore = grdbProjectStore
        self.grdbVoiceStore = grdbVoiceStore
        self.grdbRenderJobStore = grdbRenderJobStore
        self.modelRegistry = modelRegistry
        self.capabilityRegistry = capabilityRegistry
        self.modelPackRegistry = modelPackRegistry
        self.modelCatalog = modelCatalog
        self.modelInstaller = modelInstaller
        self.inferenceBackend = inferenceBackend
        self.audioPipeline = audioPipeline
        self.renderQueue = renderQueue
        self.projectStore = projectStore
        self.voiceStore = voiceStore
        self.voicePromptCache = VoicePromptCache()
        self.resourceMonitor = resourceMonitor
        self.activeSynthesisTracker = activeSynthesisTracker
        self.daemonIdleTrimTracker = daemonIdleTrimTracker
        self.sessionStore = DocumentSessionStore()
        self.startupMaintenanceCoordinator = RuntimeStartupMaintenanceCoordinator()
    }

    // MARK: - Document Session Management

    /// Creates a new `DefaultDocumentSession` for the given project bundle, restores state
    /// from the bundle, caches it by project ID, and returns it.  If a session for the
    /// same project ID already exists the existing session is returned without creating a
    /// duplicate.
    ///
    /// Concurrent calls for the same project ID are safe: each caller may construct and
    /// restore a candidate session, but only the first to reach `setSessionIfAbsent` installs
    /// it — subsequent callers receive the already-stored canonical session.
    public func createDocumentSession(for bundle: ProjectBundle) async -> any DocumentSession {
        let projectID = bundle.snapshot.project.id
        if let existing = await sessionStore.session(for: projectID) {
            return existing
        }
        let candidate = DefaultDocumentSession(
            projectStore: projectStore,
            renderQueue: renderQueue,
            projectID: projectID,
            renderSynthesisOptions: bundle.snapshot.renderSynthesisOptions
        )
        await candidate.restore(from: bundle)
        // Atomically install the candidate only if no session was stored in the meantime.
        // Returns the canonical session (either this candidate or a concurrent winner).
        return await sessionStore.setSessionIfAbsent(candidate, for: projectID)
    }

    /// Returns the cached `DocumentSession` for the given project ID, or `nil` if none exists.
    public func documentSession(for projectID: UUID) async -> (any DocumentSession)? {
        await sessionStore.session(for: projectID)
    }

    /// Removes the cached `DocumentSession` for the given project ID.
    public func closeDocumentSession(for projectID: UUID) async {
        await sessionStore.removeSession(for: projectID)
    }

    /// Builds the default runtime using the MLX backend implementation.
    ///
    /// This stays in ValarCore instead of a separate ValarRuntime package because
    /// ValarRuntime is already defined here and the package graph remains acyclic
    /// with a direct ValarCore -> ValarMLX dependency.
    public convenience init(
        paths: ValarAppPaths = ValarAppPaths(),
        runtimeConfiguration: RuntimeConfiguration = RuntimeConfiguration.configured(
            from: ProcessInfo.processInfo.environment
        ),
        fileManager: FileManager = .default
    ) throws {
        let mlxBackend = MLXInferenceBackend(
            evictionPolicy: EvictionPolicy(
                maxResidentBytes: runtimeConfiguration.maxResidentBytes,
                idleTimeoutSeconds: 600
            )
        )
        try self.init(
            paths: paths,
            runtimeConfiguration: runtimeConfiguration,
            inferenceBackend: mlxBackend,
            fileManager: fileManager
        )
    }

    public convenience init(
        paths: ValarAppPaths = ValarAppPaths(),
        runtimeConfiguration: RuntimeConfiguration = RuntimeConfiguration.configured(
            from: ProcessInfo.processInfo.environment
        ),
        inferenceBackend: any InferenceBackend,
        fileManager: FileManager = .default
    ) throws {
        try ValarAppPaths.validateContainment(
            paths.applicationSupport,
            within: paths.applicationSupport.deletingLastPathComponent(),
            fileManager: fileManager
        )

        let database = try AppDatabase(
            path: paths.databaseURL.path,
            allowedDirectories: [paths.applicationSupport],
            fileManager: fileManager
        )
        let modelRegistry = ModelRegistry(
            configuration: runtimeConfiguration,
            evictionHandler: { event in
                guard let mlxBackend = inferenceBackend as? MLXInferenceBackend else { return }
                await mlxBackend.unloadModel(withID: event.descriptor.id)
            }
        )
        let capabilityRegistry = CapabilityRegistry()
        let modelPackRegistry = GRDBBackedModelPackRegistry(
            store: GRDBModelPackStore(db: database),
            paths: paths
        )
        let modelCatalog = ModelCatalog(
            supportedSource: SupportedCatalogSource.curated(),
            catalogStore: modelPackRegistry,
            packStore: modelPackRegistry,
            capabilityRegistry: capabilityRegistry
        )
        let grdbProjectStore = GRDBProjectStore(db: database, paths: paths)
        let projectStore = GRDBBackedProjectStore(store: grdbProjectStore)
        let grdbVoiceStore = GRDBVoiceStore(db: database)
        let voiceStore = GRDBBackedVoiceStore(store: grdbVoiceStore)
        let grdbRenderJobStore = Self.makeRenderJobStore(paths: paths, fileManager: fileManager)
        let renderQueue = RenderQueue(
            configuration: runtimeConfiguration,
            store: GRDBRenderQueueStoreAdapter(store: grdbRenderJobStore)
        )
        let modelInstaller = ModelInstaller(
            registry: modelPackRegistry,
            modelRegistry: modelRegistry,
            capabilityRegistry: capabilityRegistry,
            paths: paths
        )

        self.init(
            paths: paths,
            runtimeConfiguration: runtimeConfiguration,
            database: database,
            grdbProjectStore: grdbProjectStore,
            grdbVoiceStore: grdbVoiceStore,
            grdbRenderJobStore: grdbRenderJobStore,
            modelRegistry: modelRegistry,
            capabilityRegistry: capabilityRegistry,
            modelPackRegistry: modelPackRegistry,
            modelCatalog: modelCatalog,
            modelInstaller: modelInstaller,
            inferenceBackend: inferenceBackend,
            audioPipeline: AudioPipeline(),
            renderQueue: renderQueue,
            projectStore: projectStore,
            voiceStore: voiceStore,
            resourceMonitor: RuntimeResourceSampler()
        )
    }

    public static func live(
        paths: ValarAppPaths = ValarAppPaths(),
        runtimeConfiguration: RuntimeConfiguration = RuntimeConfiguration.configured(
            from: ProcessInfo.processInfo.environment
        ),
        makeInferenceBackend: @Sendable () -> any InferenceBackend,
        fileManager: FileManager = .default
    ) throws -> ValarRuntime {
        try ValarRuntime(
            paths: paths,
            runtimeConfiguration: runtimeConfiguration,
            inferenceBackend: makeInferenceBackend(),
            fileManager: fileManager
        )
    }

    public func transcribe(_ request: SpeechToTextRequest) async throws -> SpeechToTextResponse {
        let descriptor = try await speechToTextDescriptor(for: request.model)
        let selection = try speechToTextBackendSelection(
            for: descriptor,
            preferredSampleRate: request.sampleRate
        )
        try await inferenceBackend.validate(requirement: selection.requirement)
        let configuration = selection.configuration
        do {
            return try await withReservedSpeechToTextWorkflowSession(
                descriptor: descriptor,
                configuration: configuration
            ) { reserved in
                try await reserved.workflow.transcribe(request: request, in: reserved.session)
            }
        } catch let error as MLXBackendError {
            if case .modelNotFound = error {
                throw TranscriptionError.modelNotInstalled(descriptor.id)
            }
            throw error
        } catch let error as WorkflowReservationError {
            switch error {
            case .unsupportedSpeechToText:
                throw TranscriptionError.loadedModelTypeMismatch(descriptor.id)
            default:
                throw error
            }
        }
    }

    /// Transcribes a speech-to-text request by splitting the audio into overlapping
    /// chunks and streaming recognition events as each chunk completes.
    ///
    /// Unlike `transcribe(_:)`, which processes the entire audio in one pass,
    /// `transcribeChunked(_:schedulerConfig:)` splits the input audio using
    /// `ASRChunkScheduler` and processes each chunk sequentially against a
    /// single loaded model instance. Events are emitted as they are produced:
    ///
    /// - `.partial` — running transcript after each chunk.
    /// - `.finalSegment` — high-confidence segment confirmed for a completed chunk.
    /// - `.completed` — full merged result once all chunks are processed.
    /// - `.warning` — informational advisory (e.g. no speech detected).
    ///
    /// The `audioChunk` field of `request` must be non-nil; chunked transcription
    /// does not support asset-name references. When `audioChunk` is nil the method
    /// falls back to single-pass inference and emits a single `.completed` event.
    public func transcribeChunked(
        _ request: SpeechToTextRequest,
        schedulerConfig: ASRChunkSchedulerConfig = .init()
    ) -> AsyncThrowingStream<SpeechRecognitionEvent, Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    let descriptor = try await self.speechToTextDescriptor(for: request.model)
                    let selection = try self.speechToTextBackendSelection(
                        for: descriptor,
                        preferredSampleRate: request.sampleRate
                    )
                    try await self.inferenceBackend.validate(requirement: selection.requirement)
                    let configuration = selection.configuration
                    let stream: AsyncThrowingStream<SpeechRecognitionEvent, Error>
                    do {
                        stream = try await self.withReservedSpeechToTextWorkflowSessionStream(
                            descriptor: descriptor,
                            configuration: configuration
                        ) { reserved in
                            AsyncThrowingStream { streamContinuation in
                                Task {
                                    do {
                                        guard let audioChunk = request.audioChunk else {
                                            streamContinuation.yield(.warning("Chunked transcription requires inline audio; falling back to single-pass inference."))
                                            let result = try await reserved.workflow.transcribe(request: request, in: reserved.session)
                                            streamContinuation.yield(.completed(result))
                                            streamContinuation.finish()
                                            return
                                        }

                                        let samples = audioChunk.samples
                                        let sampleRate = audioChunk.sampleRate
                                        let intSampleRate = max(1, Int(sampleRate))

                                        // Treat entire audio as speech (no VAD available for file-based transcription).
                                        // One VAD frame per second keeps the probabilities array small for any file length.
                                        let vadChunkSize = intSampleRate
                                        let numVadFrames = max(1, (samples.count + vadChunkSize - 1) / vadChunkSize)
                                        let speechProbabilities = Array(repeating: Float(1.0), count: numVadFrames)

                                        let scheduler = ASRChunkScheduler(config: schedulerConfig)
                                        let asrChunks = scheduler.schedule(
                                            audio: samples,
                                            speechProbabilities: speechProbabilities,
                                            sampleRate: intSampleRate,
                                            vadChunkSize: vadChunkSize
                                        )

                                        guard !asrChunks.isEmpty else {
                                            streamContinuation.yield(.warning("No speech content detected in audio; skipping transcription."))
                                            streamContinuation.finish()
                                            return
                                        }

                                        let backendMetadata = BackendMetadata(
                                            modelId: descriptor.id.rawValue,
                                            backendKind: configuration.backendKind
                                        )
                                        let merger = TranscriptionMerger()

                                        for asrChunk in asrChunks {
                                            try Task.checkCancellation()
                                            let chunkRequest = SpeechRecognitionRequest(
                                                model: request.model,
                                                audio: AudioChunk(samples: asrChunk.samples, sampleRate: sampleRate),
                                                languageHint: request.languageHint
                                            )
                                            let result = try await reserved.workflow.transcribe(
                                                request: chunkRequest,
                                                in: reserved.session
                                            )
                                            let events = await merger.merge(
                                                chunk: asrChunk,
                                                result: result,
                                                sampleRate: intSampleRate
                                            )
                                            for event in events {
                                                streamContinuation.yield(event)
                                            }
                                        }

                                        let finalEvent = await merger.finalize(backendMetadata: backendMetadata)
                                        streamContinuation.yield(finalEvent)
                                        streamContinuation.finish()
                                    } catch is CancellationError {
                                        streamContinuation.finish(throwing: CancellationError())
                                    } catch {
                                        streamContinuation.finish(throwing: error)
                                    }
                                }
                            }
                        }
                    } catch let error as MLXBackendError {
                        if case .modelNotFound = error {
                            throw TranscriptionError.modelNotInstalled(descriptor.id)
                        }
                        throw error
                    } catch let error as WorkflowReservationError {
                        switch error {
                        case .unsupportedSpeechToText:
                            throw TranscriptionError.loadedModelTypeMismatch(descriptor.id)
                        default:
                            throw error
                        }
                    }

                    for try await event in stream {
                        continuation.yield(event)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    public func align(_ request: ForcedAlignmentRequest) async throws -> ForcedAlignmentResponse {
        let descriptor = try await forcedAlignmentDescriptor(for: request.model)
        let selection = try forcedAlignmentBackendSelection(
            for: descriptor,
            preferredSampleRate: request.sampleRate
        )
        try await inferenceBackend.validate(requirement: selection.requirement)
        let configuration = selection.configuration
        do {
            return try await withReservedForcedAlignmentWorkflowSession(
                descriptor: descriptor,
                configuration: configuration
            ) { reserved in
                try await reserved.workflow.align(request: request, in: reserved.session)
            }
        } catch let error as MLXBackendError {
            if case .modelNotFound = error {
                throw AlignmentError.modelNotInstalled(descriptor.id)
            }
            throw error
        } catch let error as WorkflowReservationError {
            switch error {
            case .unsupportedForcedAlignment:
                throw AlignmentError.loadedModelTypeMismatch(descriptor.id)
            default:
                throw error
            }
        }
    }

    private func speechToTextDescriptor(for identifier: ModelIdentifier) async throws -> ModelDescriptor {
        guard let model = try await modelCatalog.model(for: identifier) else {
            throw TranscriptionError.modelNotFound(identifier)
        }
        guard model.installState == .installed else {
            throw TranscriptionError.modelNotInstalled(identifier)
        }
        guard model.descriptor.familyID == .qwen3ASR else {
            throw TranscriptionError.unsupportedModelFamily(
                identifier: identifier,
                family: model.descriptor.familyID
            )
        }
        return model.descriptor
    }

    private func forcedAlignmentDescriptor(for identifier: ModelIdentifier) async throws -> ModelDescriptor {
        guard let model = try await modelCatalog.model(for: identifier) else {
            throw AlignmentError.modelNotFound(identifier)
        }
        guard model.installState == .installed else {
            throw AlignmentError.modelNotInstalled(identifier)
        }
        guard model.descriptor.familyID == .qwen3ForcedAligner else {
            throw AlignmentError.unsupportedModelFamily(
                identifier: identifier,
                family: model.descriptor.familyID
            )
        }
        return model.descriptor
    }

    private func speechToTextBackendSelection(
        for descriptor: ModelDescriptor,
        preferredSampleRate: Double?
    ) throws -> (requirement: BackendRequirement, configuration: ModelRuntimeConfiguration) {
        let policy = BackendSelectionPolicy()
        let runtime = Self.selectionRuntime(for: inferenceBackend.backendKind)

        do {
            let requirement = policy.compatibleRequirement(
                for: descriptor,
                runtime: runtime
            )
            guard let requirement else {
                throw BackendSelectionPolicy.SelectionError.noCompatibleBackend(descriptor.id)
            }
            let configuration = try policy.runtimeConfiguration(
                for: descriptor,
                preferredSampleRate: preferredSampleRate,
                runtime: runtime
            )
            return (requirement, configuration)
        } catch let error as BackendSelectionPolicy.SelectionError {
            switch error {
            case .noCompatibleBackend(let identifier):
                throw TranscriptionError.noCompatibleBackend(identifier)
            }
        }
    }

    private func forcedAlignmentBackendSelection(
        for descriptor: ModelDescriptor,
        preferredSampleRate: Double?
    ) throws -> (requirement: BackendRequirement, configuration: ModelRuntimeConfiguration) {
        let policy = BackendSelectionPolicy()
        let runtime = Self.selectionRuntime(for: inferenceBackend.backendKind)

        do {
            let requirement = policy.compatibleRequirement(
                for: descriptor,
                runtime: runtime
            )
            guard let requirement else {
                throw BackendSelectionPolicy.SelectionError.noCompatibleBackend(descriptor.id)
            }
            let configuration = try policy.runtimeConfiguration(
                for: descriptor,
                preferredSampleRate: preferredSampleRate,
                runtime: runtime
            )
            return (requirement, configuration)
        } catch let error as BackendSelectionPolicy.SelectionError {
            switch error {
            case .noCompatibleBackend(let identifier):
                throw AlignmentError.noCompatibleBackend(identifier)
            }
        }
    }

    private static func selectionRuntime(for backendKind: BackendKind) -> BackendSelectionPolicy.Runtime {
        let processInfo = ProcessInfo.processInfo
        let version = processInfo.operatingSystemVersion
        return BackendSelectionPolicy.Runtime(
            availableBackends: [backendKind],
            availableMemoryBytes: Int(clamping: processInfo.physicalMemory),
            supportsLocalExecution: true,
            runtimeVersion: "\(version.majorVersion).\(version.minorVersion).\(version.patchVersion)"
        )
    }

    private static func makeRenderJobStore(
        paths: ValarAppPaths,
        fileManager: FileManager
    ) -> GRDBRenderJobStore {
        let candidateDirectories = [
            (
                directory: paths.applicationSupport,
                parentRoot: paths.applicationSupport.deletingLastPathComponent()
            ),
            (
                directory: fileManager.temporaryDirectory.appendingPathComponent("ValarTTS", isDirectory: true),
                parentRoot: fileManager.temporaryDirectory
            ),
        ]

        for candidate in candidateDirectories {
            do {
                try ValarAppPaths.validateContainment(
                    candidate.directory,
                    within: candidate.parentRoot,
                    fileManager: fileManager
                )
                try fileManager.createDirectory(at: candidate.directory, withIntermediateDirectories: true)
                let databaseURL = candidate.directory.appendingPathComponent("Valar.sqlite", isDirectory: false)
                let database = try AppDatabase(
                    path: databaseURL.path,
                    allowedDirectories: [candidate.directory],
                    fileManager: fileManager
                )
                return GRDBRenderJobStore(db: database)
            } catch {
                continue
            }
        }

        preconditionFailure("Failed to initialize render job database in any writable location.")
    }
}

private actor DocumentSessionStore {
    private var sessions: [UUID: any DocumentSession] = [:]

    func session(for projectID: UUID) -> (any DocumentSession)? {
        sessions[projectID]
    }

    func setSession(_ session: any DocumentSession, for projectID: UUID) {
        sessions[projectID] = session
    }

    /// Atomically installs `session` for `projectID` only if no session is already stored.
    /// Returns the session that is now canonical for that ID — either the incoming one (if it
    /// was the first to arrive) or a pre-existing one created by a concurrent caller.
    func setSessionIfAbsent(_ session: any DocumentSession, for projectID: UUID) -> any DocumentSession {
        if let existing = sessions[projectID] {
            return existing
        }
        sessions[projectID] = session
        return session
    }

    func removeSession(for projectID: UUID) {
        sessions[projectID] = nil
    }
}

private actor GRDBBackedProjectStore: ProjectStoring {
    private static let logger = Logger(subsystem: "com.valar.tts", category: "ProjectStore")

    private let store: GRDBProjectStore
    private var chapters: [UUID: [ChapterRecord]] = [:]
    private var renders: [UUID: [RenderJobRecord]] = [:]
    private var exports: [UUID: [ExportRecord]] = [:]
    private var speakers: [UUID: [ProjectSpeakerRecord]] = [:]

    init(store: GRDBProjectStore) {
        self.store = store
    }

    func create(title: String, notes: String?) async throws -> ProjectRecord {
        let project = try await store.insert(title: title, notes: notes)
        chapters[project.id] = []
        renders[project.id] = []
        exports[project.id] = []
        speakers[project.id] = []
        return project
    }

    func update(_ project: ProjectRecord) async {
        do {
            try await store.update(project)
        } catch {
            Self.logger.error("Failed to update project \(project.id.uuidString, privacy: .public): \(error.localizedDescription, privacy: .private)")
        }
    }

    func addChapter(_ chapter: ChapterRecord) async {
        do {
            try await store.insert(chapter)
        } catch {
            Self.logger.error("Failed to add chapter \(chapter.id.uuidString, privacy: .public): \(error.localizedDescription, privacy: .private)")
            return
        }

        if chapters[chapter.projectID] != nil {
            cacheChapter(chapter)
        }
    }

    func updateChapter(_ chapter: ChapterRecord) async {
        do {
            let persistedChapters = try await store.chapters(for: chapter.projectID)
            guard persistedChapters.contains(where: { $0.id == chapter.id }) else {
                return
            }
            try await store.update(chapter)
        } catch {
            Self.logger.error("Failed to update chapter \(chapter.id.uuidString, privacy: .public): \(error.localizedDescription, privacy: .private)")
            return
        }

        if chapters[chapter.projectID] != nil {
            cacheChapter(chapter)
        }
    }

    func removeChapter(_ id: UUID, from projectID: UUID) async {
        do {
            try await store.removeChapter(id, from: projectID)
        } catch {
            Self.logger.error("Failed to remove chapter \(id.uuidString, privacy: .public): \(error.localizedDescription, privacy: .private)")
            return
        }

        chapters[projectID]?.removeAll { $0.id == id }
    }

    func addRenderJob(_ job: RenderJobRecord) async {
        var records = renders[job.projectID, default: []]
        if let existingIndex = records.firstIndex(where: { $0.id == job.id }) {
            records[existingIndex] = job
        } else {
            records.append(job)
        }
        records.sort { $0.createdAt < $1.createdAt }
        renders[job.projectID] = records
    }

    func addExport(_ export: ExportRecord) async {
        var records = exports[export.projectID, default: []]
        if let existingIndex = records.firstIndex(where: { $0.id == export.id }) {
            records[existingIndex] = export
        } else {
            records.append(export)
        }
        records.sort { $0.createdAt < $1.createdAt }
        exports[export.projectID] = records

        do {
            try await store.insert(export)
        } catch {
            Self.logger.error("Failed to add export \(export.id.uuidString, privacy: .public): \(error.localizedDescription, privacy: .private)")
        }
    }

    func addSpeaker(_ speaker: ProjectSpeakerRecord) async {
        do {
            try await store.save(speaker)
        } catch {
            Self.logger.error("Failed to add speaker \(speaker.id.uuidString, privacy: .public): \(error.localizedDescription, privacy: .private)")
            return
        }

        if speakers[speaker.projectID] != nil {
            cacheSpeaker(speaker)
        }
    }

    func removeSpeaker(_ id: UUID, from projectID: UUID) async {
        do {
            try await store.deleteSpeaker(id)
        } catch {
            Self.logger.error("Failed to remove speaker \(id.uuidString, privacy: .public) from project \(projectID.uuidString, privacy: .public): \(error.localizedDescription, privacy: .private)")
            return
        }

        if speakers[projectID] != nil {
            speakers[projectID]?.removeAll { $0.id == id }
        }
    }

    func restore(
        project: ProjectRecord,
        chapters: [ChapterRecord],
        renderJobs: [RenderJobRecord],
        exports: [ExportRecord],
        speakers: [ProjectSpeakerRecord]
    ) async {
        do {
            try await store.save(project)
            try await store.replaceChapters(for: project.id, with: chapters)
        } catch {
            Self.logger.error("Failed to restore project \(project.id.uuidString, privacy: .public): \(error.localizedDescription, privacy: .private)")
            return
        }

        self.chapters[project.id] = chapters.sorted { $0.index < $1.index }
        self.renders[project.id] = renderJobs.sorted { $0.createdAt < $1.createdAt }
        self.exports[project.id] = exports.sorted { $0.createdAt < $1.createdAt }
        self.speakers[project.id] = speakers
    }

    func allProjects() async -> [ProjectRecord] {
        do {
            return try await store.fetchAll()
        } catch {
            Self.logger.error("Failed to load projects: \(error.localizedDescription, privacy: .private)")
            return []
        }
    }

    func chapters(for projectID: UUID) async -> [ChapterRecord] {
        if let cachedChapters = chapters[projectID] {
            return cachedChapters
        }

        do {
            let persistedChapters = try await store.chapters(for: projectID)
            chapters[projectID] = persistedChapters
            return persistedChapters
        } catch {
            Self.logger.error("Failed to load chapters for project \(projectID.uuidString, privacy: .public): \(error.localizedDescription, privacy: .private)")
            return []
        }
    }

    func renderJobs(for projectID: UUID) async -> [RenderJobRecord] {
        renders[projectID, default: []]
    }

    func exports(for projectID: UUID) async -> [ExportRecord] {
        if let cachedExports = exports[projectID] {
            return cachedExports
        }

        do {
            let persistedExports = try await store.exports(for: projectID)
            exports[projectID] = persistedExports
            return persistedExports
        } catch {
            Self.logger.error("Failed to load exports for project \(projectID.uuidString, privacy: .public): \(error.localizedDescription, privacy: .private)")
            return []
        }
    }

    func speakers(for projectID: UUID) async -> [ProjectSpeakerRecord] {
        if let cachedSpeakers = speakers[projectID], cachedSpeakers.isEmpty == false {
            return cachedSpeakers
        }

        do {
            let persistedSpeakers = try await store.speakers(for: projectID)
            speakers[projectID] = persistedSpeakers
            return persistedSpeakers
        } catch {
            Self.logger.error("Failed to load speakers for project \(projectID.uuidString, privacy: .public): \(error.localizedDescription, privacy: .private)")
            return []
        }
    }

    func bundleLocation(for projectID: UUID) async -> ValarProjectBundleLocation? {
        do {
            return try await store.bundleLocation(for: projectID)
        } catch {
            Self.logger.error("Failed to resolve bundle location for project \(projectID.uuidString, privacy: .public): \(error.localizedDescription, privacy: .private)")
            return nil
        }
    }

    func updateBundleURL(_ bundleURL: URL?, for projectID: UUID) async {
        await store.updateBundleURL(bundleURL, for: projectID)
    }

    func remove(id: UUID) async {
        do {
            try await store.delete(id)
        } catch {
            Self.logger.error("Failed to remove project \(id.uuidString, privacy: .public): \(error.localizedDescription, privacy: .private)")
        }

        chapters[id] = nil
        renders[id] = nil
        exports[id] = nil
        speakers[id] = nil
    }

    private func cacheChapter(_ chapter: ChapterRecord) {
        var records = chapters[chapter.projectID, default: []]
        if let existingIndex = records.firstIndex(where: { $0.id == chapter.id }) {
            records[existingIndex] = chapter
        } else {
            records.append(chapter)
        }
        records.sort { $0.index < $1.index }
        chapters[chapter.projectID] = records
    }

    private func cacheSpeaker(_ speaker: ProjectSpeakerRecord) {
        var records = speakers[speaker.projectID, default: []]
        if let existingIndex = records.firstIndex(where: { $0.id == speaker.id }) {
            records[existingIndex] = speaker
        } else {
            records.append(speaker)
        }
        speakers[speaker.projectID] = records
    }
}

private actor GRDBBackedVoiceStore: VoiceLibraryStoring {
    private static let logger = Logger(subsystem: "com.valar.tts", category: "VoiceLibraryStore")

    private let store: GRDBVoiceStore

    init(store: GRDBVoiceStore) {
        self.store = store
    }

    func save(_ voice: VoiceLibraryRecord) async throws -> VoiceLibraryRecord {
        try await store.insert(voice)
    }

    func list() async -> [VoiceLibraryRecord] {
        do {
            return try await store.fetchAll()
        } catch {
            Self.logger.error("Failed to list voices: \(error.localizedDescription, privacy: .private)")
            return []
        }
    }

    func delete(_ id: UUID) async throws {
        try await store.delete(id)
    }
}

private actor GRDBBackedModelPackRegistry: ModelPackManaging {
    private let paths: ValarAppPaths
    private let store: GRDBModelPackStore

    init(store: GRDBModelPackStore, paths: ValarAppPaths) {
        self.store = store
        self.paths = paths
    }

    func registerSupported(_ record: SupportedModelCatalogRecord) async throws {
        try await store.saveCatalogEntry(
            SupportedModelCatalogRecord(
                id: record.modelID,
                familyID: record.familyID,
                modelID: record.modelID,
                displayName: record.displayName,
                providerName: record.providerName,
                providerURL: record.providerURL,
                installHint: record.installHint,
                sourceKind: record.sourceKind,
                isRecommended: record.isRecommended
            )
        )

        guard try await store.manifest(for: record.modelID) == nil else {
            return
        }

        try await store.saveManifest(
            ModelPackManifest(
                id: record.modelID,
                familyID: record.familyID,
                modelID: record.modelID,
                displayName: record.displayName,
                notes: record.installHint
            )
        )
    }

    func install(
        manifest: ValarPersistence.ModelPackManifest,
        sourceKind: ModelPackSourceKind,
        sourceLocation: String,
        notes: String?
    ) async throws -> ModelInstallReceipt {
        let packDirectory = try paths.modelPackDirectory(
            familyID: manifest.familyID,
            modelID: manifest.modelID
        )
        let manifestURL = try paths.modelPackManifestURL(
            familyID: manifest.familyID,
            modelID: manifest.modelID
        )
        let receipt = ModelInstallReceipt(
            id: manifest.modelID,
            modelID: manifest.modelID,
            familyID: manifest.familyID,
            sourceKind: sourceKind,
            sourceLocation: sourceLocation,
            installedModelPath: packDirectory.path,
            manifestPath: manifestURL.path,
            checksum: manifest.artifactSpecs.first?.checksum,
            artifactCount: manifest.artifactSpecs.count,
            notes: notes
        )
        let record = InstalledModelRecord(
            id: manifest.modelID,
            familyID: manifest.familyID,
            modelID: manifest.modelID,
            displayName: manifest.displayName,
            installedPath: packDirectory.path,
            manifestPath: manifestURL.path,
            artifactCount: manifest.artifactSpecs.count,
            checksum: receipt.checksum,
            sourceKind: sourceKind,
            isEnabled: true
        )

        try await store.saveManifest(manifest)
        try await store.saveInstalledRecord(record)
        try await store.saveReceipt(receipt)
        try await store.saveCatalogEntry(
            SupportedModelCatalogRecord(
                id: manifest.modelID,
                familyID: manifest.familyID,
                modelID: manifest.modelID,
                displayName: manifest.displayName,
                providerName: "Valar",
                providerURL: nil,
                installHint: notes,
                sourceKind: sourceKind,
                isRecommended: manifest.familyID == ModelFamilyID.qwen3TTS.rawValue
            )
        )
        try await store.saveLedgerEntry(
            ModelInstallLedgerEntry(
                receiptID: receipt.id,
                sourceKind: sourceKind,
                sourceLocation: sourceLocation,
                succeeded: true,
                message: notes ?? "Installed \(manifest.displayName)"
            )
        )

        return receipt
    }

    func uninstall(modelID: String) async throws -> InstalledModelRecord? {
        try await store.uninstall(modelID: modelID)
    }

    func manifest(for modelID: String) async throws -> ValarPersistence.ModelPackManifest? {
        try await store.manifest(for: modelID)
    }

    func installedRecord(for modelID: String) async throws -> InstalledModelRecord? {
        try await store.installedRecord(for: modelID)
    }

    func receipts() async throws -> [ModelInstallReceipt] {
        try await store.receipts()
    }

    func supportedModels() async throws -> [SupportedModelCatalogRecord] {
        try await store.supportedModels()
    }

    func supportedModel(for modelID: String) async throws -> SupportedModelCatalogRecord? {
        try await store.supportedModel(for: modelID)
    }

    func ledgerEntries() async throws -> [ModelInstallLedgerEntry] {
        try await store.ledgerEntries()
    }
}
