import Foundation
import ValarModelKit
import ValarPersistence

public actor GRDBRenderQueueStoreAdapter: RenderQueueStore {
    private let store: GRDBRenderJobStore

    public init(store: GRDBRenderJobStore) {
        self.store = store
    }

    public func loadJobs() async throws -> [RenderJob] {
        let records = try await store.loadJobs()
        return records.map(Self.makeRenderJob(from:))
    }

    public func save(_ job: RenderJob) async throws {
        try await store.save(Self.makeRecord(from: job))
    }

    public func remove(id: UUID) async throws {
        try await store.remove(id: id)
    }

    static func makeRenderJob(from record: RenderJobRecord) -> RenderJob {
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
            title: record.title ?? normalizedTitle(from: record.outputFileName),
            failureReason: record.failureReason,
            queuePosition: record.queuePosition,
            synthesisOptions: record.synthesisOptions
        )
    }

    static func makeRecord(from job: RenderJob) -> RenderJobRecord {
        let timestamp = Date()
        return RenderJobRecord(
            id: job.id,
            projectID: job.projectID,
            modelID: job.modelID.rawValue,
            chapterIDs: job.chapterIDs,
            outputFileName: normalizedOutputFileName(from: job),
            createdAt: job.createdAt,
            updatedAt: timestamp,
            state: job.state.rawValue,
            priority: job.priority,
            progress: job.progress,
            title: job.title,
            failureReason: job.failureReason,
            queuePosition: job.queuePosition,
            synthesisOptions: job.synthesisOptions
        )
    }

    private static func normalizedOutputFileName(from job: RenderJob) -> String {
        let trimmed = job.outputFileName.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? "\(job.id.uuidString).wav" : trimmed
    }

    private static func normalizedTitle(from value: String?) -> String? {
        let trimmed = value?.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed?.isEmpty == false ? trimmed : nil
    }
}
