import Foundation
import XCTest
@testable import ValarCore
import ValarModelKit
import ValarPersistence

private struct StubInferenceBackend: InferenceBackend {
    var backendKind: BackendKind { .mlx }
    var runtimeCapabilities: BackendCapabilities { BackendCapabilities() }

    func validate(requirement: BackendRequirement) async throws {}

    func loadModel(
        descriptor: ModelDescriptor,
        configuration: ModelRuntimeConfiguration
    ) async throws -> any ValarModel {
        StubModel(descriptor: descriptor, backendKind: configuration.backendKind)
    }

    func unloadModel(_ model: any ValarModel) async throws {}
}

private struct StubModel: ValarModel {
    let descriptor: ModelDescriptor
    let backendKind: BackendKind
}

final class DocumentSessionTests: XCTestCase {
    func testDocumentSessionRestoresAndSnapshotsProjectState() async throws {
        let project = ProjectRecord(title: "Book One")
        let expectedRenderOptions = RenderSynthesisOptions(
            language: "es",
            temperature: 0.55,
            topP: 0.82,
            repetitionPenalty: 1.15,
            maxTokens: 3_072,
            voiceBehavior: .stableNarrator
        )
        let chapter = ChapterRecord(
            projectID: project.id,
            index: 0,
            title: "Chapter 1",
            script: "Hello there"
        )
        let speaker = ProjectSpeakerRecord(projectID: project.id, name: "Narrator")
        let export = ExportRecord(projectID: project.id, fileName: "chapter-1.wav")
        let renderJob = RenderJobRecord(
            projectID: project.id,
            modelID: "model-1",
            chapterIDs: [chapter.id],
            outputFileName: "chapter-1.wav",
            state: RenderJobState.queued.rawValue,
            priority: 2,
            progress: 0.25,
            title: "Chapter 1",
            failureReason: "temporary",
            queuePosition: 3,
            synthesisOptions: RenderSynthesisOptions(
                language: "ja",
                temperature: 0.5,
                topP: 0.77,
                repetitionPenalty: 1.2,
                maxTokens: 4_096,
                voiceBehavior: .stableNarrator
            )
        )
        let bundle = ProjectBundle(
            manifest: ProjectBundleManifest(
                version: 4,
                createdAt: .now,
                projectID: project.id,
                title: project.title,
                modelID: nil,
                renderSynthesisOptions: expectedRenderOptions,
                chapters: [
                    .init(id: chapter.id, index: chapter.index, title: chapter.title),
                ]
            ),
            snapshot: ProjectBundleSnapshot(
                project: project,
                modelID: nil,
                renderSynthesisOptions: expectedRenderOptions,
                chapters: [chapter],
                renderJobs: [renderJob],
                exports: [export],
                speakers: [speaker]
            )
        )
        let session = DefaultDocumentSession(
            projectStore: ProjectStore(),
            renderQueue: RenderQueue(),
            projectID: project.id
        )

        await session.restore(from: bundle)
        let snapshot = try await session.snapshot(
            preferredModelID: nil,
            createdAt: bundle.manifest.createdAt,
            version: bundle.manifest.version
        )

        XCTAssertEqual(snapshot.snapshot.project, project)
        XCTAssertEqual(snapshot.snapshot.chapters, [chapter])
        XCTAssertEqual(snapshot.snapshot.speakers, [speaker])
        XCTAssertEqual(snapshot.snapshot.exports, [export])
        XCTAssertEqual(snapshot.snapshot.renderSynthesisOptions, expectedRenderOptions)
        XCTAssertEqual(snapshot.manifest.renderSynthesisOptions, expectedRenderOptions)
        XCTAssertEqual(snapshot.snapshot.renderJobs.count, 1)
        XCTAssertEqual(snapshot.snapshot.renderJobs.first?.id, renderJob.id)
        XCTAssertEqual(snapshot.snapshot.renderJobs.first?.projectID, renderJob.projectID)
        XCTAssertEqual(snapshot.snapshot.renderJobs.first?.modelID, renderJob.modelID)
        XCTAssertEqual(snapshot.snapshot.renderJobs.first?.chapterIDs, renderJob.chapterIDs)
        XCTAssertEqual(snapshot.snapshot.renderJobs.first?.outputFileName, renderJob.outputFileName)
        XCTAssertEqual(snapshot.snapshot.renderJobs.first?.state, renderJob.state)
        XCTAssertEqual(snapshot.snapshot.renderJobs.first?.priority, renderJob.priority)
        XCTAssertEqual(snapshot.snapshot.renderJobs.first?.progress, renderJob.progress)
        XCTAssertEqual(snapshot.snapshot.renderJobs.first?.title, renderJob.title)
        XCTAssertEqual(snapshot.snapshot.renderJobs.first?.failureReason, renderJob.failureReason)
        XCTAssertEqual(snapshot.snapshot.renderJobs.first?.queuePosition, renderJob.queuePosition)
        XCTAssertEqual(snapshot.snapshot.renderJobs.first?.synthesisOptions, renderJob.synthesisOptions)
    }

    func testValarRuntimeCreatesAndCachesDocumentSessions() async throws {
        let baseURL = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        let runtime = try ValarRuntime(
            paths: ValarAppPaths(baseURL: baseURL),
            runtimeConfiguration: RuntimeConfiguration(),
            inferenceBackend: StubInferenceBackend()
        )
        let project = ProjectRecord(title: "Runtime Project")
        let bundle = ProjectBundle(
            manifest: ProjectBundleManifest(
                projectID: project.id,
                title: project.title,
                modelID: nil,
                renderSynthesisOptions: RenderSynthesisOptions(),
                chapters: []
            ),
            snapshot: ProjectBundleSnapshot(
                project: project,
                modelID: nil,
                renderSynthesisOptions: RenderSynthesisOptions(),
                chapters: [],
                renderJobs: [],
                exports: [],
                speakers: []
            )
        )

        let firstSession = await runtime.createDocumentSession(for: bundle)
        let secondSession = await runtime.createDocumentSession(for: bundle)
        let cachedSession = await runtime.documentSession(for: project.id)
        let firstIdentity = ObjectIdentifier(firstSession as AnyObject)
        let secondIdentity = ObjectIdentifier(secondSession as AnyObject)
        let restoredProjectID = await firstSession.projectID()

        XCTAssertEqual(restoredProjectID, project.id)
        XCTAssertEqual(firstIdentity, secondIdentity)
        XCTAssertEqual(cachedSession.map { ObjectIdentifier($0 as AnyObject) }, firstIdentity)

        await runtime.closeDocumentSession(for: project.id)
        let removedSession = await runtime.documentSession(for: project.id)
        XCTAssertNil(removedSession)
    }

    func testDocumentSessionChapterAPIExposesSpeechMetadata() async throws {
        let projectStore = ProjectStore()
        let project = try await projectStore.create(title: "Speech Project", notes: nil)
        let session = DefaultDocumentSession(
            projectStore: projectStore,
            renderQueue: RenderQueue(),
            projectID: project.id
        )

        let chapter = ChapterRecord(
            projectID: project.id,
            index: 0,
            title: "Chapter 1",
            script: "Hello there"
        )

        await session.addChapter(chapter)
        await session.attachAudio(
            to: chapter.id,
            assetName: "chapter-1.wav",
            sampleRate: 24_000,
            durationSeconds: 9.75
        )
        await session.setTranscription(
            for: chapter.id,
            transcriptionJSON: #"{"segments":[{"text":"Hello there"}]}"#,
            modelID: "whisper-large-v3"
        )
        await session.setAlignment(
            for: chapter.id,
            alignmentJSON: #"{"tokens":[{"start":0.0,"end":0.5}]}"#,
            modelID: "ctc-aligner-v1"
        )

        let updatedChapterCandidate = await session.chapters().first
        var updatedChapter = try XCTUnwrap(updatedChapterCandidate)
        updatedChapter.derivedTranslationText = "Bonjour"
        await session.updateChapter(updatedChapter)

        let snapshot = try await session.snapshot(
            preferredModelID: nil,
            createdAt: .now,
            version: 1
        )
        let reloadedChapter = try XCTUnwrap(snapshot.snapshot.chapters.first)

        XCTAssertEqual(reloadedChapter.sourceAudioAssetName, "chapter-1.wav")
        XCTAssertEqual(reloadedChapter.sourceAudioSampleRate, 24_000)
        XCTAssertEqual(reloadedChapter.sourceAudioDurationSeconds, 9.75)
        XCTAssertEqual(reloadedChapter.transcriptionJSON, #"{"segments":[{"text":"Hello there"}]}"#)
        XCTAssertEqual(reloadedChapter.transcriptionModelID, "whisper-large-v3")
        XCTAssertEqual(reloadedChapter.alignmentJSON, #"{"tokens":[{"start":0.0,"end":0.5}]}"#)
        XCTAssertEqual(reloadedChapter.alignmentModelID, "ctc-aligner-v1")
        XCTAssertEqual(reloadedChapter.derivedTranslationText, "Bonjour")
    }
}
