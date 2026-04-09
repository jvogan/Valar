import Foundation
import GRDB

extension ProjectSpeakerRecord: FetchableRecord, PersistableRecord {
    public static var databaseTableName: String { "projectSpeaker" }

    public init(row: Row) throws {
        guard let id = UUID(uuidString: row["id"]) else {
            throw DecodingError.dataCorrupted(
                DecodingError.Context(
                    codingPath: [],
                    debugDescription: "Invalid UUID in 'id' column"
                )
            )
        }
        guard let projectID = UUID(uuidString: row["projectID"]) else {
            throw DecodingError.dataCorrupted(
                DecodingError.Context(
                    codingPath: [],
                    debugDescription: "Invalid UUID in 'projectID' column"
                )
            )
        }

        self.init(
            id: id,
            projectID: projectID,
            name: row["name"],
            voiceModelID: row["voiceModelID"],
            language: row["language"] ?? "auto"
        )
    }

    public func encode(to container: inout PersistenceContainer) throws {
        container["id"] = id.uuidString
        container["projectID"] = projectID.uuidString
        container["name"] = name
        container["voiceModelID"] = voiceModelID
        container["language"] = language
    }
}

public final class AppDatabase: Sendable {
    private let _reader: any DatabaseReader
    private let _writer: any DatabaseWriter

    public var reader: any DatabaseReader { _reader }
    public var writer: any DatabaseWriter { _writer }

    public init(
        path: String,
        allowedDirectories: [URL] = [],
        fileManager: FileManager = .default
    ) throws {
        let databaseURL = URL(fileURLWithPath: path, isDirectory: false).standardizedFileURL

        if !allowedDirectories.isEmpty {
            let isAllowed = allowedDirectories.contains { allowedDirectory in
                (try? ValarAppPaths.validateContainment(databaseURL, within: allowedDirectory, fileManager: fileManager)) != nil
            }
            guard isAllowed else {
                throw ValarPathValidationError.pathEscapesContainment(
                    path: databaseURL.path,
                    allowedDirectory: allowedDirectories.map(\.path).joined(separator: ", ")
                )
            }
        }

        try fileManager.createDirectory(
            at: databaseURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        var configuration = Configuration()
        configuration.foreignKeysEnabled = true
        let pool = try DatabasePool(path: databaseURL.path, configuration: configuration)
        self._reader = pool
        self._writer = pool
        try Self.migrator.migrate(pool)
    }

    public static func inMemory() throws -> AppDatabase {
        var configuration = Configuration()
        configuration.foreignKeysEnabled = true
        let queue = try DatabaseQueue(configuration: configuration)
        let db = AppDatabase(reader: queue, writer: queue)
        try migrator.migrate(queue)
        return db
    }

    private init(reader: any DatabaseReader, writer: any DatabaseWriter) {
        self._reader = reader
        self._writer = writer
    }

    private static var migrator: DatabaseMigrator {
        var migrator = DatabaseMigrator()

        migrator.registerMigration("v1") { db in
            try db.create(table: "modelPack") { t in
                t.primaryKey("id", .text)
                t.column("schemaVersion", .integer).notNull()
                t.column("familyID", .text).notNull()
                t.column("modelID", .text).notNull()
                t.column("displayName", .text).notNull()
                t.column("capabilities", .text).notNull()
                t.column("backendKinds", .text).notNull()
                t.column("tokenizerType", .text)
                t.column("sampleRate", .double)
                t.column("artifactSpecs", .text).notNull()
                t.column("licenseName", .text)
                t.column("licenseURL", .text)
                t.column("minimumAppVersion", .text)
                t.column("notes", .text)
            }

            try db.create(table: "catalogEntry") { t in
                t.primaryKey("id", .text)
                t.column("familyID", .text).notNull()
                t.column("modelID", .text).notNull()
                t.column("displayName", .text).notNull()
                t.column("providerName", .text).notNull()
                t.column("providerURL", .text)
                t.column("installHint", .text)
                t.column("sourceKind", .text).notNull()
                t.column("isRecommended", .boolean).notNull().defaults(to: false)
            }

            try db.create(table: "installedModel") { t in
                t.primaryKey("id", .text)
                t.column("familyID", .text).notNull()
                t.column("modelID", .text).notNull()
                t.column("displayName", .text).notNull()
                t.column("installDate", .text).notNull()
                t.column("installedPath", .text).notNull()
                t.column("manifestPath", .text).notNull()
                t.column("artifactCount", .integer).notNull().defaults(to: 0)
                t.column("checksum", .text)
                t.column("sourceKind", .text).notNull()
                t.column("isEnabled", .boolean).notNull().defaults(to: true)
            }

            try db.create(table: "installReceipt") { t in
                t.primaryKey("id", .text)
                t.column("modelID", .text).notNull()
                t.column("familyID", .text).notNull()
                t.column("sourceKind", .text).notNull()
                t.column("sourceLocation", .text).notNull()
                t.column("installDate", .text).notNull()
                t.column("installedModelPath", .text).notNull()
                t.column("manifestPath", .text).notNull()
                t.column("checksum", .text)
                t.column("artifactCount", .integer).notNull().defaults(to: 0)
                t.column("notes", .text)
            }

            try db.create(table: "voice") { t in
                t.primaryKey("id", .text)
                t.column("label", .text).notNull()
                t.column("modelID", .text).notNull()
                t.column("sourceAssetName", .text)
                t.column("createdAt", .text).notNull()
            }

            try db.create(table: "renderJob") { t in
                t.primaryKey("id", .text)
                t.column("projectID", .text).notNull()
                t.column("chapterIDs", .text).notNull()
                t.column("outputFileName", .text).notNull()
                t.column("state", .text).notNull().defaults(to: "queued")
                t.column("createdAt", .text).notNull()
                t.column("updatedAt", .text).notNull()
                t.column("synthesisOptionsJSON", .text).notNull().defaults(to: "{}")
            }

            try db.create(table: "installLedger") { t in
                t.primaryKey("id", .text)
                t.column("receiptID", .text)
                t.column("sourceKind", .text).notNull()
                t.column("sourceLocation", .text).notNull()
                t.column("recordedAt", .text).notNull()
                t.column("succeeded", .boolean).notNull()
                t.column("message", .text)
            }

            try db.create(table: "project") { t in
                t.primaryKey("id", .text)
                t.column("title", .text).notNull()
                t.column("notes", .text)
                t.column("createdAt", .text).notNull()
                t.column("updatedAt", .text).notNull()
            }
        }

        migrator.registerMigration("v2_renderJob_metadata") { db in
            try db.alter(table: "renderJob") { t in
                t.add(column: "modelID", .text).notNull().defaults(to: "")
                t.add(column: "priority", .integer).notNull().defaults(to: 0)
                t.add(column: "progress", .double).notNull().defaults(to: 0)
                t.add(column: "title", .text)
            }
        }

        migrator.registerMigration("v3_voice_clone_metadata") { db in
            try db.alter(table: "voice") { t in
                t.add(column: "runtimeModelID", .text)
                t.add(column: "referenceAudioAssetName", .text)
                t.add(column: "referenceTranscript", .text)
                t.add(column: "referenceDurationSeconds", .double)
                t.add(column: "referenceSampleRate", .double)
                t.add(column: "referenceChannelCount", .integer)
                // Stores the cloned-voice speaker embedding bytes derived from a
                // user's reference audio in the local `voice.speakerEmbedding`
                // BLOB column of `valar.db`. Because the value is derived from a
                // person's voiceprint, treat it as biometric data / sensitive
                // personal data.
                t.add(column: "speakerEmbedding", .blob)
            }
        }

        migrator.registerMigration("v4_voice_prompt") { db in
            try db.alter(table: "voice") { t in
                t.add(column: "voicePrompt", .text)
            }
        }

        migrator.registerMigration("v4_renderJob_failure_reason") { db in
            try db.alter(table: "renderJob") { t in
                t.add(column: "failureReason", .text)
            }
        }

        migrator.registerMigration("v5_renderJob_queue_position") { db in
            try db.alter(table: "renderJob") { t in
                t.add(column: "queuePosition", .integer).notNull().defaults(to: 0)
            }
        }

        migrator.registerMigration("v6_chapter_store") { db in
            try db.create(table: "chapter") { t in
                t.primaryKey("id", .text)
                t.column("projectID", .text)
                    .notNull()
                    .references("project", column: "id", onDelete: .cascade)
                t.column("chapterIndex", .integer).notNull()
                t.column("title", .text).notNull()
                t.column("script", .text).notNull()
                t.column("speakerLabel", .text)
                t.column("estimatedDurationSeconds", .double)
            }
        }

        migrator.registerMigration("v8_export_store") { db in
            try db.create(table: "export") { t in
                t.primaryKey("id", .text)
                t.column("projectID", .text)
                    .notNull()
                    .references("project", column: "id", onDelete: .cascade)
                t.column("fileName", .text).notNull()
                t.column("createdAt", .text).notNull()
                t.column("checksum", .text)
            }
        }

        migrator.registerMigration("v9_project_speaker_store") { db in
            try db.create(table: "projectSpeaker") { t in
                t.primaryKey("id", .text)
                t.column("projectID", .text)
                    .notNull()
                    .references("project", column: "id", onDelete: .cascade)
                t.column("name", .text).notNull()
                t.column("voiceModelID", .text)
                t.column("language", .text).notNull().defaults(to: "auto")
                t.column("position", .integer).notNull().defaults(to: 0)
            }
        }

        migrator.registerMigration("v10_model_pack_manifest_json") { db in
            try Self.migrateModelPackTableToManifestJSON(in: db)
        }

        migrator.registerMigration("v11_chapter_speech_metadata") { db in
            try db.alter(table: "chapter") { t in
                t.add(column: "sourceAudioAssetName", .text)
                t.add(column: "sourceAudioSampleRate", .double)
                t.add(column: "sourceAudioDurationSeconds", .double)
                t.add(column: "transcriptionJSON", .text)
                t.add(column: "transcriptionModelID", .text)
                t.add(column: "alignmentJSON", .text)
                t.add(column: "alignmentModelID", .text)
                t.add(column: "derivedTranslationText", .text)
            }
        }

        migrator.registerMigration("v12_voice_backend_voice_id") { db in
            guard try db.tableExists("voice") else {
                try Self.createVoiceTable(in: db)
                return
            }

            do {
                let hasBackendVoiceID = try db.columns(in: "voice")
                    .contains(where: { $0.name == "backendVoiceID" })
                guard hasBackendVoiceID == false else { return }

                try db.alter(table: "voice") { t in
                    t.add(column: "backendVoiceID", .text)
                }
            } catch let error as DatabaseError where error.message?.contains("no such table: voice") == true {
                try Self.createVoiceTable(in: db)
            }
        }

        migrator.registerMigration("v13_voice_conditioning_format") { db in
            guard try db.tableExists("voice") else {
                try Self.createVoiceTable(in: db)
                return
            }

            do {
                let hasConditioningFormat = try db.columns(in: "voice")
                    .contains(where: { $0.name == "conditioningFormat" })
                if hasConditioningFormat == false {
                    try db.alter(table: "voice") { t in
                        t.add(column: "conditioningFormat", .text)
                    }
                }

                try db.execute(
                    sql: """
                    UPDATE voice
                    SET conditioningFormat = ?
                    WHERE conditioningFormat IS NULL
                      AND speakerEmbedding IS NOT NULL
                      AND (
                        LOWER(REPLACE(modelID, '-', '_')) LIKE '%qwen3_tts%'
                        OR LOWER(REPLACE(IFNULL(runtimeModelID, ''), '-', '_')) LIKE '%qwen3_tts%'
                      )
                    """,
                    arguments: [VoiceLibraryRecord.qwenSpeakerEmbeddingConditioningFormat]
                )
            } catch let error as DatabaseError where error.message?.contains("no such table: voice") == true {
                try Self.createVoiceTable(in: db)
            }
        }

        migrator.registerMigration("v14_voice_kind") { db in
            guard try db.tableExists("voice") else {
                try Self.createVoiceTable(in: db)
                return
            }

            do {
                let hasVoiceKind = try db.columns(in: "voice")
                    .contains(where: { $0.name == "voiceKind" })
                if hasVoiceKind == false {
                    try db.alter(table: "voice") { t in
                        t.add(column: "voiceKind", .text)
                    }
                }

                try db.execute(
                    sql: """
                    UPDATE voice
                    SET voiceKind = 'tadaReference'
                    WHERE voiceKind IS NULL
                      AND conditioningFormat = ?
                    """,
                    arguments: [VoiceLibraryRecord.tadaReferenceConditioningFormat]
                )

                try db.execute(
                    sql: """
                    UPDATE voice
                    SET voiceKind = 'clonePrompt'
                    WHERE voiceKind IS NULL
                      AND conditioningFormat = ?
                    """,
                    arguments: [VoiceLibraryRecord.qwenClonePromptConditioningFormat]
                )

                try db.execute(
                    sql: """
                    UPDATE voice
                    SET voiceKind = 'embeddingOnly'
                    WHERE voiceKind IS NULL
                      AND conditioningFormat = ?
                    """,
                    arguments: [VoiceLibraryRecord.qwenSpeakerEmbeddingConditioningFormat]
                )

                try db.execute(
                    sql: """
                    UPDATE voice
                    SET voiceKind = 'legacyPrompt'
                    WHERE voiceKind IS NULL
                      AND IFNULL(TRIM(voicePrompt), '') <> ''
                    """
                )

                try db.execute(
                    sql: """
                    UPDATE voice
                    SET voiceKind = 'preset'
                    WHERE voiceKind IS NULL
                      AND IFNULL(TRIM(backendVoiceID), '') <> ''
                    """
                )
            } catch let error as DatabaseError where error.message?.contains("no such table: voice") == true {
                try Self.createVoiceTable(in: db)
            }
        }

        migrator.registerMigration("v15_render_job_synthesis_options") { db in
            guard try db.tableExists("renderJob") else { return }

            let hasSynthesisOptionsJSON = try db.columns(in: "renderJob")
                .contains(where: { $0.name == "synthesisOptionsJSON" })
            guard hasSynthesisOptionsJSON == false else { return }

            try db.alter(table: "renderJob") { t in
                t.add(column: "synthesisOptionsJSON", .text).notNull().defaults(to: "{}")
            }
        }

        return migrator
    }

    private static func migrateModelPackTableToManifestJSON(in db: Database) throws {
        let legacyRows = try Row.fetchAll(db, sql: "SELECT * FROM modelPack")

        try db.execute(sql: "ALTER TABLE modelPack RENAME TO modelPack_legacy")
        try createModelPackTable(in: db)

        let encoder = JSONEncoder()
        encoder.outputFormatting = .sortedKeys

        for row in legacyRows {
            let manifest = try makeManifest(fromLegacyRow: row)
            let manifestJSON = try encodeJSONColumn(
                manifest,
                table: "modelPack",
                column: "manifestJSON",
                encoder: encoder
            )

            try db.execute(
                sql: """
                INSERT INTO modelPack (id, familyID, displayName, isRecommended, manifestJSON)
                VALUES (?, ?, ?, ?, ?)
                """,
                arguments: [
                    manifest.id,
                    manifest.familyID,
                    manifest.displayName,
                    manifest.isRecommended,
                    manifestJSON,
                ]
            )
        }

        try db.drop(table: "modelPack_legacy")
    }

    private static func createVoiceTable(in db: Database) throws {
        try db.create(table: "voice") { t in
            t.primaryKey("id", .text)
            t.column("label", .text).notNull()
            t.column("modelID", .text).notNull()
            t.column("runtimeModelID", .text)
            t.column("backendVoiceID", .text)
            t.column("sourceAssetName", .text)
            t.column("referenceAudioAssetName", .text)
            t.column("referenceTranscript", .text)
            t.column("referenceDurationSeconds", .double)
            t.column("referenceSampleRate", .double)
            t.column("referenceChannelCount", .integer)
            t.column("speakerEmbedding", .blob)
            t.column("conditioningFormat", .text)
            t.column("voiceKind", .text)
            t.column("voicePrompt", .text)
            t.column("createdAt", .text).notNull()
        }
    }

    private static func createModelPackTable(in db: Database) throws {
        try db.create(table: "modelPack") { t in
            t.primaryKey("id", .text)
            t.column("familyID", .text).notNull()
            t.column("displayName", .text).notNull()
            t.column("isRecommended", .boolean).notNull().defaults(to: false)
            t.column("manifestJSON", .text).notNull()
        }

        try db.create(index: "modelPack_familyID", on: "modelPack", columns: ["familyID"])
        try db.create(index: "modelPack_displayName", on: "modelPack", columns: ["displayName"])
        try db.create(index: "modelPack_isRecommended", on: "modelPack", columns: ["isRecommended"])
    }

    private static func makeManifest(fromLegacyRow row: Row) throws -> ModelPackManifest {
        let decoder = JSONDecoder()
        let isRecommended = row.hasColumn("isRecommended") ? (row["isRecommended"] as Bool) : false

        return try ModelPackManifest(
            id: row["id"],
            schemaVersion: row["schemaVersion"],
            familyID: row["familyID"],
            modelID: row["modelID"],
            displayName: row["displayName"],
            isRecommended: isRecommended,
            capabilities: decodeJSONColumn(
                [String].self,
                from: row["capabilities"],
                table: "modelPack",
                column: "capabilities",
                decoder: decoder
            ),
            backendKinds: decodeJSONColumn(
                [String].self,
                from: row["backendKinds"],
                table: "modelPack",
                column: "backendKinds",
                decoder: decoder
            ),
            tokenizerType: row["tokenizerType"],
            sampleRate: row["sampleRate"],
            artifactSpecs: decodeJSONColumn(
                [ModelPackArtifact].self,
                from: row["artifactSpecs"],
                table: "modelPack",
                column: "artifactSpecs",
                decoder: decoder
            ),
            licenseName: row["licenseName"],
            licenseURL: row["licenseURL"],
            minimumAppVersion: row["minimumAppVersion"],
            notes: row["notes"]
        )
    }
}
