import Foundation
import GRDB

// MARK: - GRDB conformance for VoiceLibraryRecord

extension VoiceLibraryRecord: FetchableRecord, PersistableRecord {
    public static var databaseTableName: String { "voice" }

    public init(row: Row) throws {
        let dateFormatter = ISO8601DateFormatter()
        let rawID: String = row["id"]
        guard let parsedID = UUID(uuidString: rawID) else {
            throw DatabaseError(
                resultCode: .SQLITE_CORRUPT,
                message: "VoiceLibraryRecord has malformed UUID in 'id' column: '\(rawID)'"
            )
        }
        self.init(
            id: parsedID,
            label: row["label"],
            modelID: row["modelID"],
            runtimeModelID: row["runtimeModelID"],
            backendVoiceID: row["backendVoiceID"],
            sourceAssetName: row["sourceAssetName"],
            referenceAudioAssetName: row["referenceAudioAssetName"],
            referenceTranscript: row["referenceTranscript"],
            referenceDurationSeconds: row["referenceDurationSeconds"],
            referenceSampleRate: row["referenceSampleRate"],
            referenceChannelCount: row["referenceChannelCount"],
            speakerEmbedding: try Self.decryptedBlob(row["speakerEmbedding"]),
            conditioningFormat: row["conditioningFormat"],
            voiceKind: row["voiceKind"],
            voicePrompt: row["voicePrompt"],
            createdAt: dateFormatter.date(from: row["createdAt"]) ?? .now
        )
    }

    public func encode(to container: inout PersistenceContainer) throws {
        let dateFormatter = ISO8601DateFormatter()
        container["id"] = id.uuidString
        container["label"] = label
        container["modelID"] = modelID
        container["runtimeModelID"] = runtimeModelID
        container["backendVoiceID"] = backendVoiceID
        container["sourceAssetName"] = sourceAssetName
        container["referenceAudioAssetName"] = referenceAudioAssetName
        container["referenceTranscript"] = referenceTranscript
        container["referenceDurationSeconds"] = referenceDurationSeconds
        container["referenceSampleRate"] = referenceSampleRate
        container["referenceChannelCount"] = referenceChannelCount
        container["speakerEmbedding"] = try speakerEmbedding.map { try VoiceLibraryProtection.protect($0) }
        container["conditioningFormat"] = conditioningFormat
        container["voiceKind"] = voiceKind
        container["voicePrompt"] = voicePrompt
        container["createdAt"] = dateFormatter.string(from: createdAt)
    }

    private static func decryptedBlob(_ value: Data?) throws -> Data? {
        guard let value else { return nil }
        return try VoiceLibraryProtection.unprotectIfNeeded(value)
    }
}

// MARK: - GRDBVoiceStore

public final class GRDBVoiceStore: Sendable {
    private let db: AppDatabase

    public init(db: AppDatabase) {
        self.db = db
    }

    public func insert(_ voice: VoiceLibraryRecord) async throws -> VoiceLibraryRecord {
        try await db.writer.write { db in
            try voice.save(db)
        }
        return voice
    }

    public func fetchAll() async throws -> [VoiceLibraryRecord] {
        try await db.reader.read { db in
            try VoiceLibraryRecord.order(Column("createdAt").asc).fetchAll(db)
        }
    }

    public func voice(id: UUID) async throws -> VoiceLibraryRecord? {
        try await db.reader.read { db in
            try VoiceLibraryRecord.filter(Column("id") == id.uuidString).fetchOne(db)
        }
    }

    public func delete(_ id: UUID) async throws {
        _ = try await db.writer.write { db in
            try VoiceLibraryRecord.filter(Column("id") == id.uuidString).deleteAll(db)
        }
    }

    public func save(_ voice: VoiceLibraryRecord) async throws -> VoiceLibraryRecord {
        try await insert(voice)
    }

    public func list() async throws -> [VoiceLibraryRecord] {
        try await fetchAll()
    }

    public func remove(id: UUID) async throws {
        try await delete(id)
    }
}
