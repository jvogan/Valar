import Foundation

public struct DaemonIdleTrimResult: Sendable, Equatable {
    public let occurredAt: Date
    public let trimmedModelIDs: [String]
    public let reason: String

    public init(
        occurredAt: Date = .now,
        trimmedModelIDs: [String],
        reason: String
    ) {
        self.occurredAt = occurredAt
        self.trimmedModelIDs = trimmedModelIDs
        self.reason = reason
    }
}

public actor DaemonIdleTrimTracker {
    private var lastResult: DaemonIdleTrimResult?

    public init() {}

    public func record(trimmedModelIDs: [String], reason: String, occurredAt: Date = .now) {
        lastResult = DaemonIdleTrimResult(
            occurredAt: occurredAt,
            trimmedModelIDs: trimmedModelIDs.sorted(),
            reason: reason
        )
    }

    public func snapshot() -> DaemonIdleTrimResult? {
        lastResult
    }
}
