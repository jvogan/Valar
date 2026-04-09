import Foundation
import ValarModelKit
import ValarPersistence

public protocol DocumentSession: Actor, Sendable {
    func projectID() async -> UUID
    func chapters() async -> [ChapterRecord]
    func addChapter(_ chapter: ChapterRecord) async
    func updateChapter(_ chapter: ChapterRecord) async
    func removeChapter(_ id: UUID) async
    func attachAudio(
        to chapterID: UUID,
        assetName: String?,
        sampleRate: Double?,
        durationSeconds: Double?
    ) async
    func setTranscription(
        for chapterID: UUID,
        transcriptionJSON: String?,
        modelID: String?
    ) async
    func setAlignment(
        for chapterID: UUID,
        alignmentJSON: String?,
        modelID: String?
    ) async
    func restore(from bundle: ProjectBundle) async
    func snapshot(
        preferredModelID: ModelIdentifier?,
        createdAt: Date,
        version: Int
    ) async throws -> ProjectBundle
}

public actor DefaultDocumentSession: DocumentSession {
    private let projectStore: any ProjectStoring
    private let renderQueue: any RenderQueueManaging
    private var activeProjectID: UUID
    private var renderSynthesisOptions: RenderSynthesisOptions

    public init(
        projectStore: any ProjectStoring,
        renderQueue: any RenderQueueManaging,
        projectID: UUID,
        renderSynthesisOptions: RenderSynthesisOptions = RenderSynthesisOptions()
    ) {
        self.projectStore = projectStore
        self.renderQueue = renderQueue
        self.activeProjectID = projectID
        self.renderSynthesisOptions = renderSynthesisOptions
    }

    public func projectID() async -> UUID {
        activeProjectID
    }

    public func chapters() async -> [ChapterRecord] {
        await projectStore.chapters(for: activeProjectID)
    }

    public func addChapter(_ chapter: ChapterRecord) async {
        var projectChapter = chapter
        projectChapter.projectID = activeProjectID
        await projectStore.addChapter(projectChapter)
    }

    public func updateChapter(_ chapter: ChapterRecord) async {
        var projectChapter = chapter
        projectChapter.projectID = activeProjectID
        await projectStore.updateChapter(projectChapter)
    }

    public func removeChapter(_ id: UUID) async {
        await projectStore.removeChapter(id, from: activeProjectID)
    }

    public func attachAudio(
        to chapterID: UUID,
        assetName: String?,
        sampleRate: Double?,
        durationSeconds: Double?
    ) async {
        await projectStore.attachAudio(
            to: chapterID,
            in: activeProjectID,
            assetName: assetName,
            sampleRate: sampleRate,
            durationSeconds: durationSeconds
        )
    }

    public func setTranscription(
        for chapterID: UUID,
        transcriptionJSON: String?,
        modelID: String?
    ) async {
        await projectStore.setTranscription(
            for: chapterID,
            in: activeProjectID,
            transcriptionJSON: transcriptionJSON,
            modelID: modelID
        )
    }

    public func setAlignment(
        for chapterID: UUID,
        alignmentJSON: String?,
        modelID: String?
    ) async {
        await projectStore.setAlignment(
            for: chapterID,
            in: activeProjectID,
            alignmentJSON: alignmentJSON,
            modelID: modelID
        )
    }

    public func restore(from bundle: ProjectBundle) async {
        activeProjectID = bundle.snapshot.project.id
        renderSynthesisOptions = bundle.snapshot.renderSynthesisOptions
        await projectStore.restore(
            project: bundle.snapshot.project,
            chapters: bundle.snapshot.chapters,
            renderJobs: bundle.snapshot.renderJobs,
            exports: bundle.snapshot.exports,
            speakers: bundle.snapshot.speakers
        )
        await renderQueue.replaceJobs(
            for: bundle.snapshot.project.id,
            with: bundle.snapshot.renderJobs.map(Self.makeRenderJob(from:))
        )
    }

    public func snapshot(
        preferredModelID: ModelIdentifier?,
        createdAt: Date,
        version: Int
    ) async throws -> ProjectBundle {
        let projectID = activeProjectID
        guard let project = await projectStore.allProjects().first(where: { $0.id == projectID }) else {
            throw ProjectBundleError.projectNotFound(projectID)
        }

        let storedRenderJobs = await projectStore.renderJobs(for: projectID)
        let queuedRenderJobs = await renderQueue.jobs(matching: nil).filter { $0.projectID == projectID }
        var renderJobsByID = Dictionary(uniqueKeysWithValues: storedRenderJobs.map { ($0.id, $0) })

        for job in queuedRenderJobs {
            let chapterIDs = job.chapterIDs.isEmpty ? (renderJobsByID[job.id]?.chapterIDs ?? []) : job.chapterIDs
            renderJobsByID[job.id] = Self.makeRenderJobRecord(from: job, chapterIDs: chapterIDs)
        }

        let chapters = await projectStore.chapters(for: projectID)
        let renderJobs = renderJobsByID.values.sorted { $0.createdAt < $1.createdAt }
        let exports = await projectStore.exports(for: projectID)
        let speakers = await projectStore.speakers(for: projectID)
        let snapshot = ProjectBundleSnapshot(
            project: project,
            modelID: Self.resolvedBundleModelID(
                projectModelID: preferredModelID,
                renderJobs: renderJobs
            ),
            renderSynthesisOptions: renderSynthesisOptions,
            chapters: chapters,
            renderJobs: renderJobs,
            exports: exports,
            speakers: speakers
        )

        return ProjectBundle(
            manifest: ProjectBundleManifest(
                version: version,
                createdAt: createdAt,
                projectID: project.id,
                title: project.title,
                modelID: snapshot.modelID,
                renderSynthesisOptions: snapshot.renderSynthesisOptions,
                chapters: chapters
                    .sorted(by: { $0.index < $1.index })
                    .map { chapter in
                        ProjectBundleManifest.ChapterSummary(
                            id: chapter.id,
                            index: chapter.index,
                            title: chapter.title
                        )
                    }
            ),
            snapshot: snapshot
        )
    }

    private static func resolvedBundleModelID(
        projectModelID: ModelIdentifier?,
        renderJobs: [RenderJobRecord]
    ) -> String? {
        if let projectModelID, !projectModelID.rawValue.isEmpty {
            return projectModelID.rawValue
        }
        return renderJobs.last(where: { !$0.modelID.isEmpty })?.modelID
    }

    private static func makeRenderJobRecord(from job: RenderJob, chapterIDs: [UUID]) -> RenderJobRecord {
        RenderJobRecord(
            id: job.id,
            projectID: job.projectID,
            modelID: job.modelID.rawValue,
            chapterIDs: chapterIDs,
            outputFileName: job.outputFileName,
            createdAt: job.createdAt,
            updatedAt: .now,
            state: job.state.rawValue,
            priority: job.priority,
            progress: job.progress,
            title: job.title,
            failureReason: job.failureReason,
            queuePosition: job.queuePosition,
            synthesisOptions: job.synthesisOptions
        )
    }

    private static func makeRenderJob(from record: RenderJobRecord) -> RenderJob {
        RenderJob(
            id: record.id,
            projectID: record.projectID,
            modelID: ModelIdentifier(record.modelID),
            chapterIDs: record.chapterIDs,
            outputFileName: record.outputFileName,
            createdAt: record.createdAt,
            state: RenderJobState(rawValue: record.state) ?? .queued,
            priority: record.priority,
            progress: record.progress,
            title: record.title,
            failureReason: record.failureReason,
            queuePosition: record.queuePosition,
            synthesisOptions: record.synthesisOptions
        )
    }
}
