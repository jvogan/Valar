import Foundation
import ValarAudio
import ValarCore
import ValarModelKit
import ValarPersistence

actor ProjectRenderService {
    typealias ChapterSynthesizer = @Sendable (ModelIdentifier, RenderSynthesisOptions, String) async throws -> AudioChunk

    private let renderQueue: RenderQueue
    private let projectStore: any ProjectStoring
    private let audioPipeline: AudioPipeline
    private let synthesizeChapter: ChapterSynthesizer

    private var processingTask: Task<Void, Never>?
    private var activeJobID: UUID?
    private var activeRenderTask: Task<Void, Error>?

    init(
        renderQueue: RenderQueue,
        projectStore: any ProjectStoring,
        audioPipeline: AudioPipeline,
        synthesizeChapter: @escaping ChapterSynthesizer
    ) {
        self.renderQueue = renderQueue
        self.projectStore = projectStore
        self.audioPipeline = audioPipeline
        self.synthesizeChapter = synthesizeChapter
    }

    func enqueueProjectRender(
        project: ProjectRecord,
        modelID: ModelIdentifier,
        synthesisOptions: RenderSynthesisOptions
    ) async -> [RenderJob] {
        let chapters = await projectStore.chapters(for: project.id)
            .filter(Self.isRenderableChapter)
        var jobs: [RenderJob] = []

        for chapter in chapters.sorted(by: { $0.index < $1.index }) {
            let job = await renderQueue.enqueue(
                projectID: project.id,
                modelID: modelID,
                chapterIDs: [chapter.id],
                outputFileName: Self.outputFileName(for: chapter),
                priority: 0,
                title: chapter.title,
                synthesisOptions: synthesisOptions
            )
            jobs.append(job)
        }

        await ensureProcessing()
        return jobs
    }

    func enqueueChapterRender(
        project: ProjectRecord,
        chapter: ChapterRecord,
        modelID: ModelIdentifier,
        synthesisOptions: RenderSynthesisOptions
    ) async -> RenderJob {
        let job = await renderQueue.enqueue(
            projectID: project.id,
            modelID: modelID,
            chapterIDs: [chapter.id],
            outputFileName: Self.outputFileName(for: chapter),
            priority: 0,
            title: chapter.title,
            synthesisOptions: synthesisOptions
        )

        await ensureProcessing()
        return job
    }

    func cancel(_ jobID: UUID) async {
        if activeJobID == jobID {
            activeRenderTask?.cancel()
        }
        await renderQueue.cancel(jobID)
    }

    func resumePendingJobs() async {
        await ensureProcessing()
    }

    private func ensureProcessing() async {
        guard processingTask == nil else { return }
        processingTask = Task { [weak self] in
            await self?.processLoop()
        }
    }

    private func processLoop() async {
        while let job = await renderQueue.nextJob() {
            if Task.isCancelled {
                processingTask = nil
                return
            }
            await process(job)
        }

        processingTask = nil
        if !Task.isCancelled, await renderQueue.pendingJobCount() > 0 {
            await ensureProcessing()
        }
    }

    private func process(_ job: RenderJob) async {
        let renderTask = Task {
            try await render(job)
        }

        activeJobID = job.id
        activeRenderTask = renderTask
        defer {
            activeJobID = nil
            activeRenderTask = nil
        }
        await renderQueue.transition(job.id, to: .running, progress: 0.05)

        do {
            try await renderTask.value
            await renderQueue.transition(job.id, to: .completed, progress: 1)
        } catch is CancellationError {
            await renderQueue.transition(job.id, to: .cancelled, progress: 1)
        } catch {
            await renderQueue.transition(
                job.id,
                to: .failed,
                progress: 1,
                failureReason: Self.failureReason(from: error)
            )
        }
    }

    private func render(_ job: RenderJob) async throws {
        guard let chapterID = job.chapterID else {
            throw ProjectRenderServiceError.missingChapterReference(job.id)
        }
        let chapters = await projectStore.chapters(for: job.projectID)
        guard let chapter = chapters.first(where: { $0.id == chapterID }) else {
            throw ProjectRenderServiceError.missingChapterReference(job.id)
        }
        guard Self.isRenderableChapter(chapter) else {
            throw ProjectRenderServiceError.emptyChapterScript(chapter.id)
        }
        guard let bundleLocation = await projectStore.bundleLocation(for: job.projectID) else {
            throw ProjectRenderServiceError.missingBundleLocation(job.projectID)
        }

        let exportsDirectory = bundleLocation.exportsDirectory
        try FileManager.default.createDirectory(
            at: exportsDirectory,
            withIntermediateDirectories: true,
            attributes: nil
        )

        try Task.checkCancellation()
        await renderQueue.transition(job.id, to: .running, progress: 0.2)

        let audioChunk = try await synthesizeChapter(job.modelID, job.synthesisOptions, chapter.script)

        try Task.checkCancellation()
        await renderQueue.transition(job.id, to: .running, progress: 0.7)

        let wavData = try await exportWAV(from: audioChunk)
        let outputURL = try Self.validatedOutputURL(
            for: job.outputFileName,
            in: exportsDirectory
        )
        if FileManager.default.fileExists(atPath: outputURL.path) {
            try FileManager.default.removeItem(at: outputURL)
        }

        try Task.checkCancellation()
        await renderQueue.transition(job.id, to: .running, progress: 0.9)

        try wavData.write(to: outputURL, options: .atomic)
        await projectStore.addExport(
            ExportRecord(
                projectID: job.projectID,
                fileName: job.outputFileName
            )
        )
    }

    private func exportWAV(from chunk: AudioChunk) async throws -> Data {
        let buffer = AudioPCMBuffer(mono: chunk.samples, sampleRate: chunk.sampleRate)
        let exported = try await audioPipeline.transcode(buffer, container: "wav")
        return exported.data
    }

    private static func outputFileName(for chapter: ChapterRecord) -> String {
        let baseName = sanitizeFileStem(
            chapter.title.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                ? "Chapter \(chapter.index + 1)"
                : chapter.title
        )
        let prefix = String(format: "%03d", max(chapter.index + 1, 1))
        return "\(prefix)-\(baseName).wav"
    }

    private static func isRenderableChapter(_ chapter: ChapterRecord) -> Bool {
        !chapter.script.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
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
            throw ProjectRenderServiceError.pathTraversal(outputFileName)
        }

        return outputURL
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

    private static func failureReason(from error: any Error) -> String {
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

enum ProjectRenderServiceError: Error {
    case missingBundleLocation(UUID)
    case missingChapterReference(UUID)
    case missingModel(ModelIdentifier)
    case emptyChapterScript(UUID)
    case pathTraversal(String)
    case unsupportedWorkflow(ModelIdentifier)
}

extension ProjectRenderServiceError: LocalizedError {
    var errorDescription: String? {
        switch self {
        case .missingBundleLocation:
            return "The project bundle is missing for this render."
        case .missingChapterReference:
            return "The chapter for this render job could not be found."
        case let .missingModel(modelID):
            return "The model \(modelID.rawValue) is unavailable for rendering."
        case .emptyChapterScript:
            return "Blank chapters can't be rendered."
        case .pathTraversal:
            return "The render output path is invalid."
        case let .unsupportedWorkflow(modelID):
            return "The model \(modelID.rawValue) doesn't support this render workflow."
        }
    }
}
