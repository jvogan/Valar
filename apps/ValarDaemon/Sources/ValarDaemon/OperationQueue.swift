import Foundation
import ValarCore

actor DaemonOperationQueue {
    typealias OperationWork = @Sendable () async throws -> Void

    private struct QueuedOperation: Sendable {
        let id: String
        let work: OperationWork
    }

    private struct OperationRecord: Sendable {
        let id: String
        let kind: String
        let createdAt: Date
        var status: String
        var startedAt: Date?
        var finishedAt: Date?
        var error: String?
    }

    /// Completed/failed operations older than this are evicted on the next prune pass.
    static let completedTTL: TimeInterval = 30 * 60 // 30 minutes

    private let recentLimit: Int
    private var records: [String: OperationRecord] = [:]
    private var orderedIDs: [String] = []
    private var pending: [QueuedOperation] = []
    private var runnerTask: Task<Void, Never>?

    init(recentLimit: Int = 32) {
        self.recentLimit = recentLimit
    }

    private static let maxPending = 16

    func enqueue(
        kind: String,
        work: @escaping OperationWork
    ) -> DaemonOperationStatusDTO? {
        guard pending.count < Self.maxPending else {
            return nil
        }
        let id = UUID().uuidString
        let record = OperationRecord(
            id: id,
            kind: kind,
            createdAt: Date(),
            status: "queued"
        )
        records[id] = record
        orderedIDs.append(id)
        pending.append(QueuedOperation(id: id, work: work))
        startRunnerIfNeeded()
        return Self.makeDTO(from: record)
    }

    func operation(id: String) -> DaemonOperationStatusDTO? {
        guard let record = records[id] else {
            return nil
        }
        return Self.makeDTO(from: record)
    }

    func queueState() -> DaemonQueueStateDTO {
        let operations = orderedIDs.compactMap { id in
            records[id].map(Self.makeDTO(from:))
        }
        let queuedCount = operations.filter { $0.status == "queued" }.count
        let runningCount = operations.filter { $0.status == "running" }.count
        return DaemonQueueStateDTO(
            operations: operations,
            queuedCount: queuedCount,
            runningCount: runningCount
        )
    }

    private func startRunnerIfNeeded() {
        guard runnerTask == nil else {
            return
        }

        runnerTask = Task {
            await self.runQueuedOperations()
        }
    }

    private func runQueuedOperations() async {
        while let operation = nextOperation() {
            do {
                try await operation.work()
                markFinished(id: operation.id, status: "done", error: nil)
            } catch {
                markFinished(
                    id: operation.id,
                    status: "failed",
                    error: Self.errorMessage(from: error)
                )
            }
            trimCompletedHistory()
        }

        runnerTask = nil
        if !pending.isEmpty {
            startRunnerIfNeeded()
        }
    }

    private func nextOperation() -> QueuedOperation? {
        guard !pending.isEmpty else {
            return nil
        }

        let operation = pending.removeFirst()
        if var record = records[operation.id] {
            record.status = "running"
            record.startedAt = Date()
            records[operation.id] = record
        }
        return operation
    }

    private func markFinished(
        id: String,
        status: String,
        error: String?
    ) {
        guard var record = records[id] else {
            return
        }

        record.status = status
        record.finishedAt = Date()
        record.error = error
        records[id] = record
    }

    /// Removes completed/failed operation records older than `completedTTL`.
    func pruneStale() {
        let cutoff = Date().addingTimeInterval(-Self.completedTTL)
        let staleIDs = orderedIDs.filter { id in
            guard let record = records[id] else { return true }
            guard record.status != "queued", record.status != "running" else { return false }
            return record.finishedAt.map { $0 < cutoff } ?? false
        }
        for id in staleIDs {
            records.removeValue(forKey: id)
        }
        orderedIDs = orderedIDs.filter { records[$0] != nil }
        if !staleIDs.isEmpty {
            print("[DaemonOperationQueue] Pruned \(staleIDs.count) stale operation record(s).")
        }
    }

    private func trimCompletedHistory() {
        var completedToKeep = recentLimit
        var retainedIDs: [String] = []

        for id in orderedIDs.reversed() {
            guard let record = records[id] else {
                continue
            }

            switch record.status {
            case "queued", "running":
                retainedIDs.append(id)
            default:
                guard completedToKeep > 0 else {
                    records.removeValue(forKey: id)
                    continue
                }
                completedToKeep -= 1
                retainedIDs.append(id)
            }
        }

        orderedIDs = retainedIDs.reversed()
    }

    private static func makeDTO(from record: OperationRecord) -> DaemonOperationStatusDTO {
        DaemonOperationStatusDTO(
            operationId: record.id,
            kind: record.kind,
            status: record.status,
            createdAt: record.createdAt.ISO8601Format(),
            startedAt: record.startedAt?.ISO8601Format(),
            finishedAt: record.finishedAt?.ISO8601Format(),
            error: record.error
        )
    }

    private static func errorMessage(from error: any Error) -> String {
        let message = error.localizedDescription.trimmingCharacters(in: .whitespacesAndNewlines)
        if message.isEmpty || message == "The operation could not be completed." {
            return String(describing: error)
        }
        return message
    }
}
