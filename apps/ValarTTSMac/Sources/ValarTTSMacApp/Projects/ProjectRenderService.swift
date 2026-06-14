import Foundation
import ValarAudio
import ValarCore
import ValarModelKit
import ValarPersistence

actor ProjectRenderService {
    typealias ChapterSynthesizer = HeadlessProjectRenderService.ChapterSynthesizer

    private let renderer: HeadlessProjectRenderService

    init(
        renderQueue: RenderQueue,
        projectStore: any ProjectStoring,
        audioPipeline: AudioPipeline,
        synthesizeChapter: @escaping ChapterSynthesizer
    ) {
        self.renderer = HeadlessProjectRenderService(
            renderQueue: renderQueue,
            projectStore: projectStore,
            audioPipeline: audioPipeline,
            synthesizeChapter: synthesizeChapter
        )
    }

    func enqueueProjectRender(
        project: ProjectRecord,
        modelID: ModelIdentifier,
        synthesisOptions: RenderSynthesisOptions
    ) async -> [RenderJob] {
        await renderer.enqueueProjectRender(
            project: project,
            modelID: modelID,
            synthesisOptions: synthesisOptions
        )
    }

    func enqueueChapterRender(
        project: ProjectRecord,
        chapter: ChapterRecord,
        modelID: ModelIdentifier,
        synthesisOptions: RenderSynthesisOptions
    ) async -> RenderJob {
        await renderer.enqueueChapterRender(
            project: project,
            chapter: chapter,
            modelID: modelID,
            synthesisOptions: synthesisOptions
        )
    }

    func cancel(_ jobID: UUID) async {
        await renderer.cancel(jobID)
    }

    func resumePendingJobs() async {
        await renderer.resumePendingJobs()
    }

    static func validatedOutputURL(for outputFileName: String, in exportsDirectory: URL) throws -> URL {
        try HeadlessProjectRenderService.validatedOutputURL(
            for: outputFileName,
            in: exportsDirectory
        )
    }

    static func sanitizeFileStem(_ value: String) -> String {
        HeadlessProjectRenderService.sanitizeFileStem(value)
    }
}
