import Foundation
import GRDB

// MARK: - GRDB conformance for ProjectRecord

extension ProjectRecord: FetchableRecord, PersistableRecord {
    public static var databaseTableName: String { "project" }

    public init(row: Row) throws {
        let dateFormatter = ISO8601DateFormatter()
        guard let id = UUID(uuidString: row["id"]) else {
            throw DecodingError.dataCorrupted(DecodingError.Context(codingPath: [], debugDescription: "Invalid UUID in 'id' column"))
        }
        self.init(
            id: id,
            title: row["title"],
            createdAt: dateFormatter.date(from: row["createdAt"]) ?? .now,
            updatedAt: dateFormatter.date(from: row["updatedAt"]) ?? .now,
            notes: row["notes"]
        )
    }

    public func encode(to container: inout PersistenceContainer) throws {
        let dateFormatter = ISO8601DateFormatter()
        container["id"] = id.uuidString
        container["title"] = title
        container["notes"] = notes
        container["createdAt"] = dateFormatter.string(from: createdAt)
        container["updatedAt"] = dateFormatter.string(from: updatedAt)
    }
}

// MARK: - GRDBProjectStore

public final class GRDBProjectStore: Sendable {
    private actor BundleURLRegistry {
        private var urls: [UUID: URL] = [:]

        func url(for projectID: UUID) -> URL? {
            urls[projectID]
        }

        func update(_ bundleURL: URL?, for projectID: UUID) {
            urls[projectID] = bundleURL
        }
    }

    private let db: AppDatabase
    private let bundleURLs = BundleURLRegistry()

    public init(db: AppDatabase, paths: ValarAppPaths = ValarAppPaths()) {
        self.db = db
        _ = paths
    }

    public func insert(title: String, notes: String? = nil) async throws -> ProjectRecord {
        try ValarAppPaths.validateRelativePath(title, label: "project title")
        let project = ProjectRecord(title: title, notes: notes)
        try await db.writer.write { db in
            try project.save(db)
        }
        return project
    }

    public func fetchAll() async throws -> [ProjectRecord] {
        try await db.reader.read { db in
            try ProjectRecord.order(Column("updatedAt").desc).fetchAll(db)
        }
    }

    public func project(id: UUID) async throws -> ProjectRecord? {
        try await db.reader.read { db in
            try ProjectRecord.filter(Column("id") == id.uuidString).fetchOne(db)
        }
    }

    public func save(_ project: ProjectRecord) async throws {
        try ValarAppPaths.validateRelativePath(project.title, label: "project title")
        try await db.writer.write { db in
            try project.save(db)
        }
    }

    public func update(_ project: ProjectRecord) async throws {
        try ValarAppPaths.validateRelativePath(project.title, label: "project title")
        try await db.writer.write { db in
            try project.update(db)
        }
    }

    public func insert(_ chapter: ChapterRecord) async throws {
        try await db.writer.write { db in
            try chapter.insert(db)
        }
    }

    public func insert(_ export: ExportRecord) async throws {
        let dateFormatter = ISO8601DateFormatter()
        let arguments: StatementArguments = [
            export.id.uuidString,
            export.projectID.uuidString,
            export.fileName,
            dateFormatter.string(from: export.createdAt),
            export.checksum
        ]
        try await db.writer.write { db in
            try db.execute(
                sql: """
                    INSERT INTO "export" (id, projectID, fileName, createdAt, checksum)
                    VALUES (?, ?, ?, ?, ?)
                """,
                arguments: arguments
            )
        }
    }

    public func update(_ chapter: ChapterRecord) async throws {
        try await db.writer.write { db in
            try chapter.update(db)
        }
    }

    public func deleteChapter(_ id: UUID, in projectID: UUID) async throws {
        _ = try await db.writer.write { db in
            try ChapterRecord
                .filter(Column("id") == id.uuidString)
                .filter(Column("projectID") == projectID.uuidString)
                .deleteAll(db)
        }
    }

    public func attachAudio(
        to chapterID: UUID,
        in projectID: UUID,
        assetName: String?,
        sampleRate: Double?,
        durationSeconds: Double?
    ) async throws {
        guard var chapter = try await chapter(id: chapterID, in: projectID) else {
            return
        }

        chapter.sourceAudioAssetName = assetName
        chapter.sourceAudioSampleRate = sampleRate
        chapter.sourceAudioDurationSeconds = durationSeconds
        try await update(chapter)
    }

    public func setTranscription(
        for chapterID: UUID,
        in projectID: UUID,
        transcriptionJSON: String?,
        modelID: String?
    ) async throws {
        guard var chapter = try await chapter(id: chapterID, in: projectID) else {
            return
        }

        chapter.transcriptionJSON = transcriptionJSON
        chapter.transcriptionModelID = modelID
        try await update(chapter)
    }

    public func setAlignment(
        for chapterID: UUID,
        in projectID: UUID,
        alignmentJSON: String?,
        modelID: String?
    ) async throws {
        guard var chapter = try await chapter(id: chapterID, in: projectID) else {
            return
        }

        chapter.alignmentJSON = alignmentJSON
        chapter.alignmentModelID = modelID
        try await update(chapter)
    }

    public func chapters(for projectID: UUID) async throws -> [ChapterRecord] {
        try await db.reader.read { db in
            try ChapterRecord
                .filter(Column("projectID") == projectID.uuidString)
                .order(Column("chapterIndex").asc)
                .fetchAll(db)
        }
    }

    public func exports(for projectID: UUID) async throws -> [ExportRecord] {
        try await db.reader.read { db in
            try ExportRecord.fetchAll(
                db,
                sql: """
                    SELECT id, projectID, fileName, createdAt, checksum
                    FROM "export"
                    WHERE projectID = ?
                    ORDER BY createdAt ASC
                """,
                arguments: [projectID.uuidString]
            )
        }
    }

    public func replaceChapters(for projectID: UUID, with chapters: [ChapterRecord]) async throws {
        try await db.writer.write { db in
            _ = try ChapterRecord
                .filter(Column("projectID") == projectID.uuidString)
                .deleteAll(db)

            for chapter in chapters.sorted(by: { $0.index < $1.index }) {
                try chapter.insert(db)
            }
        }
    }

    public func save(_ speaker: ProjectSpeakerRecord) async throws {
        try await db.writer.write { db in
            let existingPosition = try Int.fetchOne(
                db,
                sql: """
                SELECT position
                FROM projectSpeaker
                WHERE id = ?
                """,
                arguments: [speaker.id.uuidString]
            )
            let position: Int
            if let existingPosition {
                position = existingPosition
            } else {
                let maxPosition = try Int.fetchOne(
                    db,
                    sql: """
                    SELECT MAX(position)
                    FROM projectSpeaker
                    WHERE projectID = ?
                    """,
                    arguments: [speaker.projectID.uuidString]
                ) ?? -1
                position = maxPosition + 1
            }

            try db.execute(
                sql: """
                INSERT INTO projectSpeaker (id, projectID, name, voiceModelID, language, position)
                VALUES (@id, @projectID, @name, @voiceModelID, @language, @position)
                ON CONFLICT(id) DO UPDATE SET
                    projectID = excluded.projectID,
                    name = excluded.name,
                    voiceModelID = excluded.voiceModelID,
                    language = excluded.language,
                    position = excluded.position
                """,
                arguments: [
                    "id": speaker.id.uuidString,
                    "projectID": speaker.projectID.uuidString,
                    "name": speaker.name,
                    "voiceModelID": speaker.voiceModelID,
                    "language": speaker.language,
                    "position": position,
                ]
            )
        }
    }

    public func speakers(for projectID: UUID) async throws -> [ProjectSpeakerRecord] {
        try await db.reader.read { db in
            try ProjectSpeakerRecord
                .filter(Column("projectID") == projectID.uuidString)
                .order(Column("position").asc)
                .fetchAll(db)
        }
    }

    public func deleteSpeaker(_ id: UUID) async throws {
        _ = try await db.writer.write { db in
            try db.execute(
                sql: "DELETE FROM projectSpeaker WHERE id = ?",
                arguments: [id.uuidString]
            )
        }
    }

    public func delete(_ id: UUID) async throws {
        _ = try await db.writer.write { db in
            try ProjectRecord.filter(Column("id") == id.uuidString).deleteAll(db)
        }
        await bundleURLs.update(nil, for: id)
    }

    public func create(title: String, notes: String? = nil) async throws -> ProjectRecord {
        try await insert(title: title, notes: notes)
    }

    public func allProjects() async throws -> [ProjectRecord] {
        try await fetchAll()
    }

    private func chapter(id: UUID, in projectID: UUID) async throws -> ChapterRecord? {
        try await db.reader.read { db in
            try ChapterRecord
                .filter(Column("id") == id.uuidString)
                .filter(Column("projectID") == projectID.uuidString)
                .fetchOne(db)
        }
    }

    public func removeChapter(_ id: UUID, from projectID: UUID) async throws {
        try await deleteChapter(id, in: projectID)
    }

    public func remove(id: UUID) async throws {
        try await delete(id)
    }

    public func bundleLocation(for projectID: UUID) async throws -> ValarProjectBundleLocation? {
        guard let project = try await project(id: projectID),
              let bundleURL = await bundleURLs.url(for: projectID) else { return nil }
        return ValarProjectBundleLocation(projectID: projectID, title: project.title, bundleURL: bundleURL)
    }

    public func updateBundleURL(_ bundleURL: URL?, for projectID: UUID) async {
        await bundleURLs.update(bundleURL, for: projectID)
    }
}
