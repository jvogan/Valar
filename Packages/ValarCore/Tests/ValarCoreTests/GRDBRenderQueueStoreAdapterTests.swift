import Foundation
import XCTest
@testable import ValarCore
import ValarModelKit
import ValarPersistence

final class GRDBRenderQueueStoreAdapterTests: XCTestCase {
    func testAdapterRoundTripsRenderJobMetadata() async throws {
        let database = try AppDatabase.inMemory()
        let adapter = GRDBRenderQueueStoreAdapter(store: GRDBRenderJobStore(db: database))
        let job = RenderJob(
            projectID: UUID(),
            modelID: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
            chapterIDs: [UUID()],
            outputFileName: "chapter-1.wav",
            state: .failed,
            priority: 7,
            progress: 0.42,
            title: "Chapter 1",
            synthesisOptions: RenderSynthesisOptions(
                language: "en",
                temperature: 0.55,
                topP: 0.82,
                repetitionPenalty: 1.1,
                maxTokens: 2_048,
                voiceBehavior: .stableNarrator
            )
        )

        try await adapter.save(job)
        let jobs = try await adapter.loadJobs()

        XCTAssertEqual(jobs.count, 1)
        XCTAssertEqual(jobs.first?.id, job.id)
        XCTAssertEqual(jobs.first?.projectID, job.projectID)
        XCTAssertEqual(jobs.first?.modelID, job.modelID)
        XCTAssertEqual(jobs.first?.chapterIDs, job.chapterIDs)
        XCTAssertEqual(jobs.first?.outputFileName, job.outputFileName)
        XCTAssertEqual(jobs.first?.state, job.state)
        XCTAssertEqual(jobs.first?.priority, job.priority)
        XCTAssertEqual(jobs.first?.progress, job.progress)
        XCTAssertEqual(jobs.first?.title, job.title)
        XCTAssertEqual(jobs.first?.synthesisOptions, job.synthesisOptions)
    }

    func testRenderQueuePersistsJobsAcrossDatabaseBackedRestarts() async throws {
        let databaseURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
            .appendingPathComponent("Valar.sqlite", isDirectory: false)
        try FileManager.default.createDirectory(
            at: databaseURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        defer { try? FileManager.default.removeItem(at: databaseURL.deletingLastPathComponent()) }

        let firstQueue = RenderQueue(
            store: GRDBRenderQueueStoreAdapter(
                store: GRDBRenderJobStore(
                    db: try AppDatabase(
                        path: databaseURL.path,
                        allowedDirectories: [databaseURL.deletingLastPathComponent()]
                    )
                )
            )
        )
        let created = await firstQueue.enqueue(
            projectID: UUID(),
            modelID: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
            chapterIDs: [UUID()],
            outputFileName: "long-form-chapter-render.wav",
            priority: 5,
            title: "Long-form chapter render",
            synthesisOptions: RenderSynthesisOptions(
                language: "fr",
                temperature: 0.6,
                topP: 0.88,
                repetitionPenalty: 1.05,
                maxTokens: 3_072,
                voiceBehavior: .expressive
            )
        )
        await firstQueue.transition(created.id, to: .running, progress: 0.6)

        let secondQueue = RenderQueue(
            store: GRDBRenderQueueStoreAdapter(
                store: GRDBRenderJobStore(
                    db: try AppDatabase(
                        path: databaseURL.path,
                        allowedDirectories: [databaseURL.deletingLastPathComponent()]
                    )
                )
            )
        )
        let restored = await secondQueue.job(id: created.id)

        XCTAssertEqual(restored?.id, created.id)
        XCTAssertEqual(restored?.projectID, created.projectID)
        XCTAssertEqual(restored?.modelID, created.modelID)
        XCTAssertEqual(restored?.chapterIDs, created.chapterIDs)
        XCTAssertEqual(restored?.outputFileName, created.outputFileName)
        XCTAssertEqual(restored?.priority, created.priority)
        XCTAssertEqual(restored?.title, created.title)
        XCTAssertEqual(restored?.state, .queued)
        XCTAssertEqual(restored?.progress, 0.6)
        XCTAssertEqual(restored?.synthesisOptions, created.synthesisOptions)
    }
}
