import Foundation
import GRDB

// MARK: - GRDB conformance for RenderJobRecord

extension RenderJobRecord: FetchableRecord, PersistableRecord {
    public static var databaseTableName: String { "renderJob" }

    public init(row: Row) throws {
        let dateFormatter = ISO8601DateFormatter()
        let decoder = JSONDecoder()
        let chapterIDs = try decodeJSONColumn(
            [UUID].self,
            from: row["chapterIDs"],
            table: Self.databaseTableName,
            column: "chapterIDs",
            decoder: decoder
        )
        let synthesisOptions: RenderSynthesisOptions
        if row.hasColumn("synthesisOptionsJSON"), let payload: String = row["synthesisOptionsJSON"] {
            synthesisOptions = try decodeJSONColumn(
                RenderSynthesisOptions.self,
                from: payload,
                table: Self.databaseTableName,
                column: "synthesisOptionsJSON",
                decoder: decoder
            )
        } else {
            synthesisOptions = RenderSynthesisOptions()
        }

        guard let id = UUID(uuidString: row["id"]) else {
            throw DecodingError.dataCorrupted(DecodingError.Context(codingPath: [], debugDescription: "Invalid UUID in 'id' column"))
        }
        guard let projectID = UUID(uuidString: row["projectID"]) else {
            throw DecodingError.dataCorrupted(DecodingError.Context(codingPath: [], debugDescription: "Invalid UUID in 'projectID' column"))
        }
        let failureReason: String? = row.hasColumn("failureReason") ? row["failureReason"] : nil
        let queuePosition: Int = row.hasColumn("queuePosition") ? (row["queuePosition"] ?? 0) : 0
        self.init(
            id: id,
            projectID: projectID,
            modelID: row["modelID"] ?? "",
            chapterIDs: chapterIDs,
            outputFileName: row["outputFileName"],
            createdAt: dateFormatter.date(from: row["createdAt"]) ?? .now,
            updatedAt: dateFormatter.date(from: row["updatedAt"]) ?? .now,
            state: row["state"],
            priority: row["priority"] ?? 0,
            progress: row["progress"] ?? 0,
            title: row["title"],
            failureReason: failureReason,
            queuePosition: queuePosition,
            synthesisOptions: synthesisOptions
        )
    }

    public func encode(to container: inout PersistenceContainer) throws {
        let dateFormatter = ISO8601DateFormatter()
        let encoder = JSONEncoder()
        container["id"] = id.uuidString
        container["projectID"] = projectID.uuidString
        container["modelID"] = modelID
        container["chapterIDs"] = try encodeJSONColumn(
            chapterIDs,
            table: Self.databaseTableName,
            column: "chapterIDs",
            encoder: encoder
        )
        container["outputFileName"] = outputFileName
        container["state"] = state
        container["createdAt"] = dateFormatter.string(from: createdAt)
        container["updatedAt"] = dateFormatter.string(from: updatedAt)
        container["priority"] = priority
        container["progress"] = progress
        container["title"] = title
        container["failureReason"] = failureReason
        container["queuePosition"] = queuePosition
        container["synthesisOptionsJSON"] = try encodeJSONColumn(
            synthesisOptions,
            table: Self.databaseTableName,
            column: "synthesisOptionsJSON",
            encoder: encoder
        )
    }
}

// MARK: - GRDBRenderJobStore
// Note: RenderQueueStore protocol is in ValarCore. ValarPersistence cannot import ValarCore,
// so this store has matching methods without formal conformance. A conformance adapter
// bridging RenderJobRecord <-> RenderJob will live in ValarCore.

public final class GRDBRenderJobStore: Sendable {
    private let db: AppDatabase

    public init(db: AppDatabase) {
        self.db = db
    }

    public func loadJobs() async throws -> [RenderJobRecord] {
        try await db.reader.read { db in
            try RenderJobRecord.order(Column("createdAt").asc).fetchAll(db)
        }
    }

    public func save(_ job: RenderJobRecord) async throws {
        try await db.writer.write { db in
            try job.save(db)
        }
    }

    public func remove(id: UUID) async throws {
        _ = try await db.writer.write { db in
            try RenderJobRecord.filter(Column("id") == id.uuidString).deleteAll(db)
        }
    }

    public func job(id: UUID) async throws -> RenderJobRecord? {
        try await db.reader.read { db in
            try RenderJobRecord.filter(Column("id") == id.uuidString).fetchOne(db)
        }
    }

    public func jobs(forProject projectID: UUID) async throws -> [RenderJobRecord] {
        try await db.reader.read { db in
            try RenderJobRecord
                .filter(Column("projectID") == projectID.uuidString)
                .order(Column("createdAt").asc)
                .fetchAll(db)
        }
    }
}
