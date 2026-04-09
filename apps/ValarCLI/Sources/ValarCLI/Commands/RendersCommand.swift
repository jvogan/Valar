import ArgumentParser
import Foundation
import ValarAudio
import ValarCore
import ValarMLX
import ValarModelKit
import ValarPersistence

struct RendersCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "renders",
        abstract: "Inspect and process queued chapter renders for the active project session.",
        subcommands: [Queue.self, Start.self, Status.self, Cancel.self]
    )

    mutating func run() throws {
        throw CleanExit.helpRequest(Self.self)
    }
}

extension RendersCommand {
    struct Queue: ParsableCommand {
        static let configuration = CommandConfiguration(
            commandName: "queue",
            abstract: "Show queued and running renders for the active project session."
        )

        mutating func run() throws {
            let jsonRequested = OutputContext.jsonRequested
            try ProjectsCommand.runAsync {
                let snapshot = try await RendersCommand.loadSnapshot()
                let pendingJobs = RendersCommand.pendingJobs(from: snapshot.jobs)
                if jsonRequested {
                    try OutputFormat.writeSuccess(
                        command: OutputFormat.commandPath("renders queue"),
                        data: RenderStatusPayloadDTO(
                            message: pendingJobs.isEmpty
                                ? "No pending renders for '\(snapshot.activeSession.title)'."
                                : "Loaded \(pendingJobs.count) pending render(s) for '\(snapshot.activeSession.title)'.",
                            projectTitle: snapshot.activeSession.title,
                            renders: pendingJobs.map { RenderJobDTO(from: $0) }
                        )
                    )
                    return
                }

                RendersCommand.printQueue(pendingJobs, title: snapshot.activeSession.title)
            }
        }
    }

    struct Start: ParsableCommand {
        static let configuration = CommandConfiguration(
            commandName: "start",
            abstract: "Process queued renders. If nothing is queued yet, chapter renders are prepared automatically."
        )

        mutating func run() throws {
            let jsonRequested = OutputContext.jsonRequested
            try ProjectsCommand.runAsync {
                let context = try ProjectsCommand.ProjectCommandContext()
                var activeSession = try ProjectsCommand.requireActiveSession(in: context)
                try await ProjectsCommand.hydrateRuntimeSession(for: activeSession, in: context)

                let project = try await ProjectsCommand.requireProjectRecord(
                    for: activeSession.projectID,
                    in: context
                )
                var jobs = await RendersCommand.projectJobs(
                    in: context.runtime.renderQueue,
                    projectID: activeSession.projectID
                )
                var queuedCount = 0

                if RendersCommand.pendingJobs(from: jobs).isEmpty {
                    let chapters = await context.projectStore.chapters(for: activeSession.projectID)
                    let chaptersToQueue = RendersCommand.chaptersNeedingQueuedRenders(
                        chapters: chapters,
                        jobs: jobs
                    )

                    guard chaptersToQueue.isEmpty == false else {
                        let message = "No pending renders to start for '\(activeSession.title)'."
                        if jsonRequested {
                            try OutputFormat.writeSuccess(
                                command: OutputFormat.commandPath("renders start"),
                                data: RenderStatusPayloadDTO(
                                    message: message,
                                    processedCount: 0,
                                    queuedCount: 0,
                                    remainingPendingCount: 0,
                                    renders: []
                                )
                            )
                        } else {
                            print(message)
                        }
                        return
                    }

                    for chapter in chaptersToQueue {
                        let modelID = try await RendersCommand.resolveModelID(
                            explicitModelID: nil,
                            for: chapter,
                            in: context
                        )
                        let synthesisOptions = await RendersCommand.resolveSynthesisOptions(
                            for: chapter,
                            in: context
                        )
                        _ = await context.runtime.renderQueue.enqueue(
                            projectID: project.id,
                            modelID: modelID,
                            chapterIDs: [chapter.id],
                            outputFileName: RendersCommand.outputFileName(for: chapter),
                            priority: 0,
                            title: chapter.title,
                            synthesisOptions: synthesisOptions
                        )
                    }
                    queuedCount = chaptersToQueue.count

                    activeSession = try await ProjectsCommand.persistActiveSession(activeSession, in: context)
                    jobs = await RendersCommand.projectJobs(
                        in: context.runtime.renderQueue,
                        projectID: activeSession.projectID
                    )
                    if jsonRequested == false {
                        print("Queued \(chaptersToQueue.count) render(s) for '\(activeSession.title)'.")
                    }
                }

                let runner = try CLIRenderRunner(
                    context: context,
                    activeSession: activeSession,
                    shouldPrintProgress: jsonRequested == false
                )
                let processedCount = try await runner.runPendingJobs()
                if processedCount == 0 {
                    let message = "No pending renders to start for '\(activeSession.title)'."
                    if jsonRequested {
                        let currentJobs = await RendersCommand.projectJobs(
                            in: context.runtime.renderQueue,
                            projectID: activeSession.projectID
                        )
                        try OutputFormat.writeSuccess(
                            command: OutputFormat.commandPath("renders start"),
                            data: RenderStatusPayloadDTO(
                                message: message,
                                processedCount: 0,
                                queuedCount: 0,
                                remainingPendingCount: RendersCommand.pendingJobs(from: currentJobs).count,
                                renders: currentJobs.map { RenderJobDTO(from: $0) }
                            )
                        )
                    } else {
                        print(message)
                    }
                    return
                }

                let currentJobs = await RendersCommand.projectJobs(
                    in: context.runtime.renderQueue,
                    projectID: activeSession.projectID
                )
                let remainingPending = RendersCommand.pendingJobs(from: currentJobs)
                let message = "Processed \(processedCount) render(s); \(remainingPending.count) pending render(s) remain."
                if jsonRequested {
                    try OutputFormat.writeSuccess(
                        command: OutputFormat.commandPath("renders start"),
                        data: RenderStatusPayloadDTO(
                            message: message,
                            processedCount: processedCount,
                            queuedCount: queuedCount,
                            remainingPendingCount: remainingPending.count,
                            renders: currentJobs.map { RenderJobDTO(from: $0) }
                        )
                    )
                } else {
                    print(message)
                }
            }
        }
    }

    struct Status: ParsableCommand {
        static let configuration = CommandConfiguration(
            commandName: "status",
            abstract: "Show render progress for the active project session."
        )

        @Flag(name: .long, help: "Refresh status until no queued or running renders remain.")
        var watch = false

        @Option(name: .long, help: "Polling interval in seconds when `--watch` is enabled.")
        var interval: Double = 1.0

        mutating func run() throws {
            let watch = self.watch
            let pollInterval = max(interval, 0.2)
            let jsonRequested = OutputContext.jsonRequested

            try ProjectsCommand.runAsync {
                var firstSnapshot = true
                var finalSnapshot: Snapshot?

                while true {
                    let snapshot = try await RendersCommand.loadSnapshot()
                    finalSnapshot = snapshot

                    if jsonRequested == false {
                        if firstSnapshot == false {
                            print("")
                        }
                        firstSnapshot = false

                        print("Render status for '\(snapshot.activeSession.title)' @ \(ProjectsCommand.iso8601String(from: .now))")
                        RendersCommand.printStatus(snapshot.jobs)
                    }

                    if watch == false || RendersCommand.pendingJobs(from: snapshot.jobs).isEmpty {
                        if jsonRequested, let finalSnapshot {
                            try OutputFormat.writeSuccess(
                                command: OutputFormat.commandPath("renders status"),
                                data: RenderStatusPayloadDTO(
                                    message: "Loaded render status for '\(finalSnapshot.activeSession.title)'.",
                                    projectTitle: finalSnapshot.activeSession.title,
                                    watched: watch,
                                    generatedAt: OutputFormat.iso8601String(from: .now),
                                    renders: finalSnapshot.jobs.map { RenderJobDTO(from: $0) }
                                )
                            )
                        }
                        return
                    }

                    try await Task.sleep(for: .seconds(pollInterval))
                }
            }
        }
    }

    struct Cancel: ParsableCommand {
        static let configuration = CommandConfiguration(
            commandName: "cancel",
            abstract: "Cancel a queued or running render."
        )

        @Argument(help: "Render job UUID from `valartts renders queue` or `valartts renders status`.")
        var id: String

        mutating func run() throws {
            let rawID = id
            let jsonRequested = OutputContext.jsonRequested

            try ProjectsCommand.runAsync {
                guard let jobID = UUID(uuidString: rawID.trimmingCharacters(in: .whitespacesAndNewlines)) else {
                    throw ValidationError("Render id must be a UUID.")
                }

                let context = try ProjectsCommand.ProjectCommandContext()
                var activeSession = try ProjectsCommand.requireActiveSession(in: context)
                try await ProjectsCommand.hydrateRuntimeSession(for: activeSession, in: context)

                guard let job = await context.runtime.renderQueue.job(id: jobID), job.projectID == activeSession.projectID else {
                    throw ValidationError("No render exists with id \(rawID) in the active project session.")
                }

                switch job.state {
                case .completed, .cancelled, .failed:
                    let message = "Render \(job.id.uuidString) is already \(job.state.rawValue)."
                    if jsonRequested {
                        try OutputFormat.writeSuccess(
                            command: OutputFormat.commandPath("renders cancel"),
                            data: RenderDetailPayloadDTO(
                                message: message,
                                render: RenderJobDTO(from: job)
                            )
                        )
                    } else {
                        print(message)
                    }
                    return
                case .queued, .running:
                    await context.runtime.renderQueue.cancel(jobID)
                    activeSession = try await ProjectsCommand.persistActiveSession(activeSession, in: context)
                    let updatedJob = await context.runtime.renderQueue.job(id: jobID) ?? job
                    let message = "Cancelled render \(job.id.uuidString)."
                    if jsonRequested {
                        try OutputFormat.writeSuccess(
                            command: OutputFormat.commandPath("renders cancel"),
                            data: RenderDetailPayloadDTO(
                                message: message,
                                render: RenderJobDTO(from: updatedJob)
                            )
                        )
                    } else {
                        print(message)
                    }
                }
            }
        }
    }
}

extension RendersCommand {
    struct Snapshot {
        let activeSession: ProjectsCommand.ActiveProjectSession
        let jobs: [RenderJob]
    }

    static func loadSnapshot() async throws -> Snapshot {
        let context = try ProjectsCommand.ProjectCommandContext()
        let activeSession = try ProjectsCommand.requireActiveSession(in: context)
        try await ProjectsCommand.hydrateRuntimeSession(for: activeSession, in: context)
        let jobs = await projectJobs(in: context.runtime.renderQueue, projectID: activeSession.projectID)
        return Snapshot(activeSession: activeSession, jobs: jobs)
    }

    static func projectJobs(
        in renderQueue: RenderQueue,
        projectID: UUID
    ) async -> [RenderJob] {
        await renderQueue.jobs(matching: nil).filter { $0.projectID == projectID }
    }

    static func pendingJobs(from jobs: [RenderJob]) -> [RenderJob] {
        jobs.filter { $0.state == .queued || $0.state == .running }
    }

    static func chaptersNeedingQueuedRenders(
        chapters: [ChapterRecord],
        jobs: [RenderJob]
    ) -> [ChapterRecord] {
        let latestJobByChapter = jobs.reduce(into: [UUID: RenderJob]()) { partialResult, job in
            guard let chapterID = job.chapterID else {
                return
            }

            let existing = partialResult[chapterID]
            if let existing, existing.createdAt >= job.createdAt {
                return
            }
            partialResult[chapterID] = job
        }

        return chapters.sorted(by: { $0.index < $1.index }).filter { chapter in
            guard let latestJob = latestJobByChapter[chapter.id] else {
                return true
            }

            switch latestJob.state {
            case .queued, .running, .completed:
                return false
            case .cancelled, .failed:
                return true
            }
        }
    }

    static func printQueue(_ jobs: [RenderJob], title: String) {
        guard jobs.isEmpty == false else {
            print("No pending renders for '\(title)'.")
            return
        }

        print("id\tstate\tprogress\tchapter\toutput")
        for job in jobs {
            print(
                [
                    job.id.uuidString,
                    job.state.rawValue,
                    progressLabel(for: job.progress),
                    sanitize(job.title ?? job.chapterID?.uuidString ?? "-"),
                    sanitize(job.outputFileName),
                ].joined(separator: "\t")
            )
        }
    }

    static func printStatus(_ jobs: [RenderJob]) {
        guard jobs.isEmpty == false else {
            print("No renders are recorded for the active project.")
            return
        }

        print("id\tstate\tprogress\tchapter\toutput\tfailure")
        for job in jobs {
            print(
                [
                    job.id.uuidString,
                    job.state.rawValue,
                    progressLabel(for: job.progress),
                    sanitize(job.title ?? job.chapterID?.uuidString ?? "-"),
                    sanitize(job.outputFileName),
                    sanitize(job.failureReason ?? "-"),
                ].joined(separator: "\t")
            )
        }
    }

    static func progressLabel(for value: Double) -> String {
        let clamped = max(0, min(value, 1))
        return "\(Int((clamped * 100).rounded()))%"
    }

    static func sanitize(_ value: String) -> String {
        value
            .replacingOccurrences(of: "\t", with: " ")
            .replacingOccurrences(of: "\n", with: " ")
    }

    static func outputFileName(for chapter: ChapterRecord) -> String {
        let baseName = sanitizeFileStem(
            chapter.title.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                ? "Chapter \(chapter.index + 1)"
                : chapter.title
        )
        let prefix = String(format: "%03d", max(chapter.index + 1, 1))
        return "\(prefix)-\(baseName).wav"
    }

    static func sanitizeFileStem(_ value: String) -> String {
        let allowed = CharacterSet.alphanumerics.union(.whitespaces)
        let collapsed = value
            .unicodeScalars
            .map { allowed.contains($0) ? Character($0) : " " }
            .reduce(into: "") { $0.append($1) }
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()

        let stem = collapsed
            .components(separatedBy: .whitespacesAndNewlines)
            .filter { !$0.isEmpty }
            .joined(separator: "-")

        return stem.isEmpty ? "chapter" : stem
    }

    static func validatedOutputURL(for outputFileName: String, in exportsDirectory: URL) throws -> URL {
        let outputURL = exportsDirectory
            .appendingPathComponent(outputFileName, isDirectory: false)
            .standardizedFileURL
        let standardizedExportsDirectory = exportsDirectory.standardizedFileURL
        let exportsDirectoryPath = standardizedExportsDirectory.path.hasSuffix("/")
            ? standardizedExportsDirectory.path
            : standardizedExportsDirectory.path + "/"

        guard outputURL.path.hasPrefix(exportsDirectoryPath) else {
            throw CLICommandError(message: "Render output '\(outputFileName)' escapes the exports directory.")
        }

        return outputURL
    }

    static func resolveModelID(
        explicitModelID: String?,
        for chapter: ChapterRecord,
        in context: ProjectsCommand.ProjectCommandContext
    ) async throws -> ModelIdentifier {
        if let explicitModelID = explicitModelID?.trimmingCharacters(in: .whitespacesAndNewlines),
           explicitModelID.isEmpty == false {
            return try await validateExplicitModelID(
                ModelIdentifier(explicitModelID),
                in: context
            )
        }

        let jobs = await projectJobs(in: context.runtime.renderQueue, projectID: chapter.projectID)
        if let existingModelID = jobs
            .filter({ $0.chapterID == chapter.id && $0.modelID.rawValue.isEmpty == false })
            .sorted(by: { $0.createdAt > $1.createdAt })
            .first?
            .modelID {
            return existingModelID
        }

        let speakers = await context.projectStore.speakers(for: chapter.projectID)
        if let speakerLabel = chapter.speakerLabel?.trimmingCharacters(in: .whitespacesAndNewlines),
           speakerLabel.isEmpty == false,
           let matchingSpeaker = speakers.first(where: {
               $0.name.compare(speakerLabel, options: [.caseInsensitive, .diacriticInsensitive]) == .orderedSame
           }),
           let voiceModelID = matchingSpeaker.voiceModelID?.trimmingCharacters(in: .whitespacesAndNewlines),
           voiceModelID.isEmpty == false {
            return ModelIdentifier(voiceModelID)
        }

        let distinctSpeakerModels = Array(
            Set(
                speakers.compactMap { speaker in
                    let modelID = speaker.voiceModelID?.trimmingCharacters(in: .whitespacesAndNewlines)
                    return modelID?.isEmpty == false ? modelID : nil
                }
            )
        ).sorted()

        if distinctSpeakerModels.count == 1, let modelID = distinctSpeakerModels.first {
            return ModelIdentifier(modelID)
        }

        let runtime = try ValarRuntime(paths: context.paths, fileManager: context.fileManager)
        let installedSpeechModels = try await runtime.modelCatalog.installedModels()
            .filter { $0.descriptor.capabilities.contains(.speechSynthesis) }

        if installedSpeechModels.count == 1, let modelID = installedSpeechModels.first?.id {
            return modelID
        }

        let recommendedInstalledModels = installedSpeechModels.filter(\.isRecommended)
        if recommendedInstalledModels.count == 1, let modelID = recommendedInstalledModels.first?.id {
            return modelID
        }

        throw ValidationError(
            "Unable to determine a speech model for chapter '\(chapter.title)'. Add a speaker model to the project or use `valartts exports create --chapter <id> --model <model-id>`."
        )
    }

    static func validateExplicitModelID(
        _ identifier: ModelIdentifier,
        in context: ProjectsCommand.ProjectCommandContext
    ) async throws -> ModelIdentifier {
        let runtime = try ValarRuntime(paths: context.paths, fileManager: context.fileManager)
        let descriptor: ModelDescriptor?

        if let registeredDescriptor = await runtime.modelRegistry.descriptor(for: identifier) {
            descriptor = registeredDescriptor
        } else if let capabilityDescriptor = await runtime.capabilityRegistry.descriptor(for: identifier) {
            descriptor = capabilityDescriptor
        } else {
            descriptor = try await runtime.modelCatalog.model(for: identifier)?.descriptor
        }

        guard let descriptor else {
            throw ValidationError("Model '\(identifier.rawValue)' was not found in the runtime catalog.")
        }

        guard descriptor.capabilities.contains(.speechSynthesis) else {
            throw ValidationError("Model '\(descriptor.id.rawValue)' does not support speech synthesis.")
        }

        return descriptor.id
    }

    static func resolveSynthesisOptions(
        for chapter: ChapterRecord,
        in context: ProjectsCommand.ProjectCommandContext
    ) async -> RenderSynthesisOptions {
        let jobs = await projectJobs(in: context.runtime.renderQueue, projectID: chapter.projectID)
        if let existingOptions = jobs
            .filter({ $0.chapterID == chapter.id })
            .sorted(by: { $0.createdAt > $1.createdAt })
            .first?
            .synthesisOptions {
            return existingOptions
        }
        return RenderSynthesisOptions()
    }

    static func failureReason(from error: any Error) -> String {
        let preferredMessage = (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
        let normalizedMessage = preferredMessage
            .components(separatedBy: .whitespacesAndNewlines)
            .filter { !$0.isEmpty }
            .joined(separator: " ")

        guard !normalizedMessage.isEmpty, !normalizedMessage.hasPrefix("The operation couldn't be completed.") else {
            return "Render failed."
        }

        let maxLength = 160
        guard normalizedMessage.count > maxLength else {
            return normalizedMessage
        }

        let truncated = normalizedMessage.prefix(maxLength - 1).trimmingCharacters(in: .whitespacesAndNewlines)
        return "\(truncated)…"
    }
}

private final class CLIRenderRunner {
    private let context: ProjectsCommand.ProjectCommandContext
    private let renderer: CLITextToSpeechRenderer
    private let shouldPrintProgress: Bool
    private var activeSession: ProjectsCommand.ActiveProjectSession

    init(
        context: ProjectsCommand.ProjectCommandContext,
        activeSession: ProjectsCommand.ActiveProjectSession,
        shouldPrintProgress: Bool
    ) throws {
        self.context = context
        self.activeSession = activeSession
        self.shouldPrintProgress = shouldPrintProgress
        self.renderer = try CLITextToSpeechRenderer(
            paths: context.paths,
            fileManager: context.fileManager
        )
    }

    func runPendingJobs() async throws -> Int {
        var processedCount = 0

        while let job = await nextPendingJob() {
            try await process(job)
            processedCount += 1
        }

        return processedCount
    }

    private func nextPendingJob() async -> RenderJob? {
        await context.runtime.renderQueue.jobs(matching: nil).first { job in
            job.projectID == activeSession.projectID && (job.state == .queued || job.state == .running)
        }
    }

    private func process(_ job: RenderJob) async throws {
        if await isCancelledOnDisk(job.id) {
            await context.runtime.renderQueue.cancel(job.id)
            try await persistSession()
            if shouldPrintProgress {
                print("Cancelled render \(job.id.uuidString) before synthesis started.")
            }
            return
        }

        do {
            let chapter = try await requiredChapter(for: job)
            let outputURL = try await requiredOutputURL(for: job)

            await context.runtime.renderQueue.transition(job.id, to: .running, progress: 0.05)
            try await persistSession()

            if await isCancelledOnDisk(job.id) {
                await context.runtime.renderQueue.cancel(job.id)
                try await persistSession()
                if shouldPrintProgress {
                    print("Cancelled render \(job.id.uuidString).")
                }
                return
            }

            await context.runtime.renderQueue.transition(job.id, to: .running, progress: 0.2)
            try await persistSession()

            let audioChunk = try await renderer.synthesize(
                text: chapter.script,
                modelID: job.modelID,
                synthesisOptions: job.synthesisOptions
            )

            if await isCancelledOnDisk(job.id) {
                await context.runtime.renderQueue.cancel(job.id)
                try await persistSession()
                if shouldPrintProgress {
                    print("Cancelled render \(job.id.uuidString).")
                }
                return
            }

            await context.runtime.renderQueue.transition(job.id, to: .running, progress: 0.7)
            try await persistSession()

            let wavData = try await renderer.exportWAV(from: audioChunk)

            if context.fileManager.fileExists(atPath: outputURL.path) {
                try context.fileManager.removeItem(at: outputURL)
            }

            await context.runtime.renderQueue.transition(job.id, to: .running, progress: 0.9)
            try await persistSession()

            if await isCancelledOnDisk(job.id) {
                await context.runtime.renderQueue.cancel(job.id)
                try await persistSession()
                if shouldPrintProgress {
                    print("Cancelled render \(job.id.uuidString).")
                }
                return
            }

            try wavData.write(to: outputURL, options: .atomic)
            await context.runtime.renderQueue.transition(job.id, to: .completed, progress: 1)
            try await persistSession()
            if shouldPrintProgress {
                print("Completed render \(job.id.uuidString) -> \(outputURL.lastPathComponent)")
            }
        } catch {
            await context.runtime.renderQueue.transition(
                job.id,
                to: .failed,
                progress: 1,
                failureReason: RendersCommand.failureReason(from: error)
            )
            try await persistSession()
            if shouldPrintProgress {
                print("Failed render \(job.id.uuidString): \(RendersCommand.failureReason(from: error))")
            }
        }
    }

    private func requiredChapter(for job: RenderJob) async throws -> ChapterRecord {
        guard let chapterID = job.chapterID else {
            throw CLICommandError(message: "Render \(job.id.uuidString) is missing a chapter reference.")
        }

        let chapters = await context.projectStore.chapters(for: job.projectID)
        guard let chapter = chapters.first(where: { $0.id == chapterID }) else {
            throw CLICommandError(message: "Render \(job.id.uuidString) references a missing chapter.")
        }

        return chapter
    }

    private func requiredOutputURL(for job: RenderJob) async throws -> URL {
        guard let bundleLocation = await context.projectStore.bundleLocation(for: job.projectID) else {
            throw CLICommandError(message: "The active project bundle is unavailable for this render.")
        }

        try context.fileManager.createDirectory(
            at: bundleLocation.exportsDirectory,
            withIntermediateDirectories: true
        )

        return try RendersCommand.validatedOutputURL(
            for: job.outputFileName,
            in: bundleLocation.exportsDirectory
        )
    }

    private func isCancelledOnDisk(_ jobID: UUID) async -> Bool {
        let bundleURL = URL(fileURLWithPath: activeSession.bundlePath, isDirectory: true)
        guard let bundle = try? ProjectBundleReader(fileManager: context.fileManager).read(from: bundleURL) else {
            return false
        }

        return bundle.snapshot.renderJobs.first(where: { $0.id == jobID })?.state == RenderJobState.cancelled.rawValue
    }

    private func persistSession() async throws {
        activeSession = try await ProjectsCommand.persistActiveSession(activeSession, in: context)
    }
}

struct CLITextToSpeechRenderer {
    private let runtime: ValarRuntime

    init(
        paths: ValarAppPaths,
        fileManager: FileManager = .default
    ) throws {
        self.runtime = try ValarRuntime(paths: paths, fileManager: fileManager)
    }

    func synthesize(
        text: String,
        modelID: ModelIdentifier,
        synthesisOptions: RenderSynthesisOptions
    ) async throws -> AudioChunk {
        let descriptor = try await resolveDescriptor(for: modelID)
        let configuration = try BackendSelectionPolicy().runtimeConfiguration(
            for: descriptor,
            runtime: BackendSelectionPolicy.Runtime(
                availableBackends: [runtime.inferenceBackend.backendKind]
            )
        )
        do {
            return try await runtime.withReservedTextToSpeechWorkflowSession(
                descriptor: descriptor,
                configuration: configuration
            ) { reserved in
                let request = SpeechSynthesisRequest(
                    model: descriptor.id,
                    text: text,
                    language: synthesisOptions.normalizedLanguage,
                    sampleRate: descriptor.defaultSampleRate ?? 24_000,
                    responseFormat: "pcm_f32le",
                    temperature: synthesisOptions.temperature.map(Float.init),
                    topP: synthesisOptions.topP.map(Float.init),
                    repetitionPenalty: synthesisOptions.repetitionPenalty.map(Float.init),
                    maxTokens: synthesisOptions.maxTokens,
                    voiceBehavior: synthesisOptions.voiceBehavior
                )
                return try await reserved.workflow.synthesize(request: request, in: reserved.session)
            }
        } catch let error as WorkflowReservationError {
            if case .unsupportedTextToSpeech = error {
                throw CLICommandError(
                    message: "Model '\(descriptor.id.rawValue)' did not load as a text-to-speech workflow."
                )
            }
            throw error
        } catch let error as MLXBackendError {
            throw CLICommandError(
                message: "Failed to load model '\(descriptor.id.rawValue)': \(error.localizedDescription)"
            )
        }
    }

    func exportWAV(from chunk: AudioChunk) async throws -> Data {
        let buffer = AudioPCMBuffer(mono: chunk.samples, sampleRate: chunk.sampleRate)
        let exported = try await runtime.audioPipeline.transcode(buffer, container: "wav")
        return exported.data
    }

    private func resolveDescriptor(
        for identifier: ModelIdentifier
    ) async throws -> ModelDescriptor {
        if let descriptor = await runtime.modelRegistry.descriptor(for: identifier) {
            return try validatedSpeechModel(descriptor)
        }

        if let descriptor = await runtime.capabilityRegistry.descriptor(for: identifier) {
            return try validatedSpeechModel(descriptor)
        }

        if let descriptor = (try await runtime.modelCatalog.model(for: identifier))?.descriptor {
            return try validatedSpeechModel(descriptor)
        }

        throw CLICommandError(message: "Model '\(identifier.rawValue)' was not found in the runtime catalog.")
    }

    private func validatedSpeechModel(_ descriptor: ModelDescriptor) throws -> ModelDescriptor {
        guard descriptor.capabilities.contains(.speechSynthesis) else {
            throw CLICommandError(message: "Model '\(descriptor.id.rawValue)' does not support speech synthesis.")
        }
        return descriptor
    }
}
