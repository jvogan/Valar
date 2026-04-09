import XCTest
@testable import ValarPersistence
import GRDB

final class GRDBStoreCoverageTests: XCTestCase {
    private let currentMigrationIDs: Set<String> = [
        "v1",
        "v2_renderJob_metadata",
        "v3_voice_clone_metadata",
        "v4_voice_prompt",
        "v4_renderJob_failure_reason",
        "v5_renderJob_queue_position",
        "v6_chapter_store",
        "v8_export_store",
        "v9_project_speaker_store",
        "v10_model_pack_manifest_json",
        "v11_chapter_speech_metadata",
        "v12_voice_backend_voice_id",
        "v13_voice_conditioning_format",
        "v14_voice_kind",
        "v15_render_job_synthesis_options",
    ]

    private func makeTemporaryDirectory() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }

    private func makeFileBackedDatabaseURL(in root: URL) -> URL {
        root.appendingPathComponent("Valar.sqlite", isDirectory: false)
    }

    private func appliedMigrations(in db: AppDatabase) async throws -> Set<String> {
        try await db.reader.read { database in
            Set(try String.fetchAll(database, sql: "SELECT identifier FROM grdb_migrations"))
        }
    }

    private func columnNames(in db: AppDatabase, table: String) async throws -> Set<String> {
        try await db.reader.read { database in
            Set(try database.columns(in: table).map(\.name))
        }
    }

    private func rowCount(in db: AppDatabase, table: String) async throws -> Int {
        try await db.reader.read { database in
            try Int.fetchOne(database, sql: "SELECT COUNT(*) FROM \(table)") ?? 0
        }
    }

    private func createLegacyV1Database(at databaseURL: URL) throws {
        try FileManager.default.createDirectory(
            at: databaseURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )

        var configuration = Configuration()
        configuration.foreignKeysEnabled = true
        let queue = try DatabaseQueue(path: databaseURL.path, configuration: configuration)

        try queue.write { db in
            try db.execute(sql: "CREATE TABLE grdb_migrations (identifier TEXT NOT NULL PRIMARY KEY)")
            try db.execute(sql: "INSERT INTO grdb_migrations (identifier) VALUES ('v1')")

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

            try db.execute(
                sql: """
                INSERT INTO modelPack (
                    id, schemaVersion, familyID, modelID, displayName,
                    capabilities, backendKinds, tokenizerType, sampleRate,
                    artifactSpecs, licenseName, licenseURL, minimumAppVersion, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                arguments: [
                    "legacy-pack-row",
                    1,
                    "qwen3_tts",
                    "legacy-model-id",
                    "Legacy Model",
                    "[\"speech.synthesis\"]",
                    "[\"mlx\"]",
                    "legacy-tokenizer",
                    24_000,
                    "[{\"id\":\"weights\",\"kind\":\"weights\",\"relativePath\":\"weights/model.safetensors\",\"checksum\":\"abc123\",\"byteCount\":4096}]",
                    "MIT",
                    "https://example.com/license",
                    "1.0.0",
                    "legacy notes",
                ]
            )

            try db.execute(
                sql: """
                INSERT INTO voice (id, label, modelID, sourceAssetName, createdAt)
                VALUES (?, ?, ?, ?, ?)
                """,
                arguments: [
                    "00000000-0000-0000-0000-000000000001",
                    "Legacy Voice",
                    "legacy-model",
                    "legacy.wav",
                    "2026-03-18T01:00:00Z",
                ]
            )

            try db.execute(
                sql: """
                INSERT INTO renderJob (id, projectID, chapterIDs, outputFileName, state, createdAt, updatedAt)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                arguments: [
                    UUID(uuidString: "11111111-AAAA-BBBB-CCCC-DDDDDDDDDDDD")!.uuidString,
                    UUID(uuidString: "AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE")!.uuidString,
                    "[\"11111111-2222-3333-4444-555555555555\"]",
                    "legacy.wav",
                    "queued",
                    "2026-03-18T01:05:00Z",
                    "2026-03-18T01:05:00Z",
                ]
            )
        }
    }

    private func createLegacyV10DatabaseWithChapterTable(at databaseURL: URL) throws {
        try FileManager.default.createDirectory(
            at: databaseURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )

        var configuration = Configuration()
        configuration.foreignKeysEnabled = true
        let queue = try DatabaseQueue(path: databaseURL.path, configuration: configuration)
        let appliedMigrations = currentMigrationIDs.subtracting([
            "v11_chapter_speech_metadata",
            "v12_voice_backend_voice_id",
            "v13_voice_conditioning_format",
            "v14_voice_kind",
            "v15_render_job_synthesis_options",
        ]).sorted()
        let projectID = UUID(uuidString: "AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE")!
        let chapterID = UUID(uuidString: "11111111-2222-3333-4444-555555555555")!

        try queue.write { db in
            try db.execute(sql: "CREATE TABLE grdb_migrations (identifier TEXT NOT NULL PRIMARY KEY)")
            for migrationID in appliedMigrations {
                try db.execute(
                    sql: "INSERT INTO grdb_migrations (identifier) VALUES (?)",
                    arguments: [migrationID]
                )
            }

            try db.create(table: "project") { t in
                t.primaryKey("id", .text)
                t.column("title", .text).notNull()
                t.column("notes", .text)
                t.column("createdAt", .text).notNull()
                t.column("updatedAt", .text).notNull()
            }

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

            try db.execute(
                sql: """
                INSERT INTO project (id, title, notes, createdAt, updatedAt)
                VALUES (?, ?, ?, ?, ?)
                """,
                arguments: [
                    projectID.uuidString,
                    "Legacy Project",
                    "Created before speech metadata",
                    "2026-03-18T01:00:00Z",
                    "2026-03-18T01:00:00Z",
                ]
            )

            try db.execute(
                sql: """
                INSERT INTO chapter (
                    id, projectID, chapterIndex, title, script, speakerLabel, estimatedDurationSeconds
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                arguments: [
                    chapterID.uuidString,
                    projectID.uuidString,
                    0,
                    "Legacy Chapter",
                    "Existing script",
                    "Narrator",
                    12.5,
                ]
            )
        }
    }

    private func createLegacyV12VoiceDatabaseWithQwenClone(at databaseURL: URL) throws {
        try FileManager.default.createDirectory(
            at: databaseURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )

        var configuration = Configuration()
        configuration.foreignKeysEnabled = true
        let queue = try DatabaseQueue(path: databaseURL.path, configuration: configuration)
        let appliedMigrations = currentMigrationIDs.subtracting([
            "v13_voice_conditioning_format",
            "v14_voice_kind",
            "v15_render_job_synthesis_options",
        ]).sorted()

        try queue.write { db in
            try db.execute(sql: "CREATE TABLE grdb_migrations (identifier TEXT NOT NULL PRIMARY KEY)")
            for migrationID in appliedMigrations {
                try db.execute(
                    sql: "INSERT INTO grdb_migrations (identifier) VALUES (?)",
                    arguments: [migrationID]
                )
            }

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
                t.column("voicePrompt", .text)
                t.column("createdAt", .text).notNull()
            }

            try db.execute(
                sql: """
                INSERT INTO voice (
                    id, label, modelID, runtimeModelID, sourceAssetName,
                    referenceAudioAssetName, referenceTranscript, referenceDurationSeconds,
                    referenceSampleRate, referenceChannelCount, speakerEmbedding, voicePrompt, createdAt
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                arguments: [
                    "11111111-2222-3333-4444-555555555555",
                    "Migrated Clone",
                    "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16",
                    "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
                    "legacy-reference.wav",
                    "legacy-reference.wav",
                    "Legacy reference transcript.",
                    7.5,
                    24_000.0,
                    1,
                    Data([0x00, 0x00, 0x80, 0x3F]),
                    nil as String?,
                    "2026-03-01T00:00:00Z",
                ]
            )
        }
    }

    private static func makeManifest(
        id: String = UUID().uuidString,
        familyID: String = "qwen3_tts",
        modelID: String,
        displayName: String
    ) -> ModelPackManifest {
        ModelPackManifest(
            id: id,
            familyID: familyID,
            modelID: modelID,
            displayName: displayName,
            capabilities: ["speech.synthesis"],
            backendKinds: ["mlx"],
            artifactSpecs: [
                ModelPackArtifact(
                    id: "weights",
                    kind: "weights",
                    relativePath: "weights/model.safetensors",
                    checksum: "abc123",
                    byteCount: 4_096
                ),
            ],
            licenseName: "MIT",
            licenseURL: "https://example.com/license"
        )
    }

    func testAppDatabaseInMemoryMigratesAllPersistenceTablesAndColumns() async throws {
        let db = try AppDatabase.inMemory()

        for table in [
            "modelPack",
            "catalogEntry",
            "installedModel",
            "installReceipt",
            "voice",
            "renderJob",
            "installLedger",
            "project",
            "chapter",
            "export",
            "projectSpeaker",
        ] {
            let exists = try await db.reader.read { database in
                try database.tableExists(table)
            }
            XCTAssertTrue(exists, "Expected table \(table) to exist after migrations")
        }

        let modelPackColumns = try await columnNames(in: db, table: "modelPack")
        XCTAssertEqual(
            modelPackColumns,
            Set(["displayName", "familyID", "id", "isRecommended", "manifestJSON"])
        )

        let renderJobColumns = try await columnNames(in: db, table: "renderJob")
        XCTAssertTrue(
            renderJobColumns.isSuperset(of: [
                "id",
                "projectID",
                "chapterIDs",
                "outputFileName",
                "state",
                "createdAt",
                "updatedAt",
                "modelID",
                "priority",
                "progress",
                "title",
                "failureReason",
                "queuePosition",
                "synthesisOptionsJSON",
            ])
        )

        let chapterColumns = try await columnNames(in: db, table: "chapter")
        XCTAssertTrue(
            chapterColumns.isSuperset(of: [
                "id",
                "projectID",
                "chapterIndex",
                "title",
                "script",
                "speakerLabel",
                "estimatedDurationSeconds",
                "sourceAudioAssetName",
                "sourceAudioSampleRate",
                "sourceAudioDurationSeconds",
                "transcriptionJSON",
                "transcriptionModelID",
                "alignmentJSON",
                "alignmentModelID",
                "derivedTranslationText",
            ])
        )

        let exportColumns = try await columnNames(in: db, table: "export")
        XCTAssertTrue(
            exportColumns.isSuperset(of: [
                "id",
                "projectID",
                "fileName",
                "createdAt",
                "checksum",
            ])
        )

        let projectSpeakerColumns = try await columnNames(in: db, table: "projectSpeaker")
        XCTAssertTrue(
            projectSpeakerColumns.isSuperset(of: [
                "id",
                "projectID",
                "name",
                "voiceModelID",
                "language",
                "position",
            ])
        )

        let voiceColumns = try await columnNames(in: db, table: "voice")
        XCTAssertTrue(
            voiceColumns.isSuperset(of: [
                "id",
                "label",
                "modelID",
                "sourceAssetName",
                "createdAt",
                "runtimeModelID",
                "referenceAudioAssetName",
                "referenceTranscript",
                "referenceDurationSeconds",
                "referenceSampleRate",
                "referenceChannelCount",
                "speakerEmbedding",
                "voicePrompt",
                "backendVoiceID",
            ])
        )

        let migrations = try await appliedMigrations(in: db)
        XCTAssertEqual(migrations, currentMigrationIDs)
    }

    func testAppDatabaseMigratesEmptyFileDatabaseToCurrentSchema() async throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let databaseURL = makeFileBackedDatabaseURL(in: root)
        let db = try AppDatabase(path: databaseURL.path, allowedDirectories: [root])

        XCTAssertTrue(FileManager.default.fileExists(atPath: databaseURL.path))

        let migrations = try await appliedMigrations(in: db)
        XCTAssertEqual(migrations, currentMigrationIDs)

        let modelPackColumns = try await columnNames(in: db, table: "modelPack")
        XCTAssertEqual(
            modelPackColumns,
            Set(["displayName", "familyID", "id", "isRecommended", "manifestJSON"])
        )

        let renderJobColumns = try await columnNames(in: db, table: "renderJob")
        XCTAssertTrue(renderJobColumns.contains("priority"))
        XCTAssertTrue(renderJobColumns.contains("failureReason"))
        XCTAssertTrue(renderJobColumns.contains("queuePosition"))
        XCTAssertTrue(renderJobColumns.contains("synthesisOptionsJSON"))

        let voiceColumns = try await columnNames(in: db, table: "voice")
        XCTAssertTrue(voiceColumns.contains("voicePrompt"))
        XCTAssertTrue(voiceColumns.contains("backendVoiceID"))

        let exportColumns = try await columnNames(in: db, table: "export")
        XCTAssertEqual(exportColumns, Set(["checksum", "createdAt", "fileName", "id", "projectID"]))

        let projectSpeakerColumns = try await columnNames(in: db, table: "projectSpeaker")
        XCTAssertTrue(projectSpeakerColumns.contains("position"))
    }

    func testAppDatabaseMigratesLegacyChapterTableAndPersistsSpeechMetadata() async throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let databaseURL = makeFileBackedDatabaseURL(in: root)
        try createLegacyV10DatabaseWithChapterTable(at: databaseURL)

        let db = try AppDatabase(path: databaseURL.path, allowedDirectories: [root])
        let store = GRDBProjectStore(db: db)
        let projectID = UUID(uuidString: "AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE")!
        let chapterID = UUID(uuidString: "11111111-2222-3333-4444-555555555555")!

        let migrations = try await appliedMigrations(in: db)
        XCTAssertEqual(migrations, currentMigrationIDs)

        let chapterColumns = try await columnNames(in: db, table: "chapter")
        XCTAssertTrue(
            chapterColumns.isSuperset(of: [
                "sourceAudioAssetName",
                "sourceAudioSampleRate",
                "sourceAudioDurationSeconds",
                "transcriptionJSON",
                "transcriptionModelID",
                "alignmentJSON",
                "alignmentModelID",
                "derivedTranslationText",
            ])
        )

        guard var chapter = try await store.chapters(for: projectID).first else {
            return XCTFail("Expected migrated chapter row")
        }

        XCTAssertEqual(chapter.id, chapterID)
        XCTAssertEqual(chapter.title, "Legacy Chapter")
        XCTAssertNil(chapter.sourceAudioAssetName)
        XCTAssertNil(chapter.transcriptionJSON)
        XCTAssertNil(chapter.derivedTranslationText)

        chapter.sourceAudioAssetName = "chapter-1.wav"
        chapter.sourceAudioSampleRate = 24_000
        chapter.sourceAudioDurationSeconds = 9.75
        chapter.transcriptionJSON = #"{"segments":[{"text":"Hello"}]}"#
        chapter.transcriptionModelID = "whisper-large-v3"
        chapter.alignmentJSON = #"{"tokens":[{"start":0.0,"end":0.4}]}"#
        chapter.alignmentModelID = "ctc-aligner-v1"
        chapter.derivedTranslationText = "Bonjour"

        try await store.update(chapter)

        let reloaded = try await store.chapters(for: projectID)
        XCTAssertEqual(reloaded, [chapter])
    }

    func testAppDatabaseMigratesLegacyV1DatabaseForwardWithoutLosingRows() async throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let databaseURL = makeFileBackedDatabaseURL(in: root)
        try createLegacyV1Database(at: databaseURL)

        let db = try AppDatabase(path: databaseURL.path, allowedDirectories: [root])
        let voiceStore = GRDBVoiceStore(db: db)
        let renderJobStore = GRDBRenderJobStore(db: db)
        let modelPackStore = GRDBModelPackStore(db: db)

        let migrations = try await appliedMigrations(in: db)
        XCTAssertEqual(migrations, currentMigrationIDs)

        let modelPackColumns = try await columnNames(in: db, table: "modelPack")
        XCTAssertEqual(
            modelPackColumns,
            Set(["displayName", "familyID", "id", "isRecommended", "manifestJSON"])
        )

        let renderJobColumns = try await columnNames(in: db, table: "renderJob")
        XCTAssertTrue(renderJobColumns.contains("modelID"))

        let voiceColumns = try await columnNames(in: db, table: "voice")
        XCTAssertTrue(voiceColumns.contains("speakerEmbedding"))
        XCTAssertTrue(voiceColumns.contains("backendVoiceID"))
        XCTAssertTrue(voiceColumns.contains("conditioningFormat"))

        let chapterColumns = try await columnNames(in: db, table: "chapter")
        XCTAssertTrue(chapterColumns.contains("chapterIndex"))

        let exportColumns = try await columnNames(in: db, table: "export")
        XCTAssertEqual(exportColumns, Set(["checksum", "createdAt", "fileName", "id", "projectID"]))

        let projectSpeakerColumns = try await columnNames(in: db, table: "projectSpeaker")
        XCTAssertTrue(projectSpeakerColumns.contains("position"))

        let legacyVoices = try await voiceStore.fetchAll()
        XCTAssertEqual(legacyVoices.count, 1)
        XCTAssertEqual(legacyVoices.first?.label, "Legacy Voice")
        XCTAssertNil(legacyVoices.first?.runtimeModelID)
        XCTAssertNil(legacyVoices.first?.voicePrompt)
        XCTAssertNil(legacyVoices.first?.backendVoiceID)
        XCTAssertNil(legacyVoices.first?.conditioningFormat)

        let legacyJobs = try await renderJobStore.loadJobs()
        XCTAssertEqual(legacyJobs.count, 1)
        XCTAssertEqual(legacyJobs.first?.outputFileName, "legacy.wav")
        XCTAssertEqual(legacyJobs.first?.state, "queued")
        XCTAssertEqual(legacyJobs.first?.modelID, "")
        XCTAssertEqual(legacyJobs.first?.priority, 0)
        XCTAssertEqual(legacyJobs.first?.progress, 0)
        XCTAssertNil(legacyJobs.first?.title)
        XCTAssertEqual(legacyJobs.first?.synthesisOptions, RenderSynthesisOptions())

        let legacyManifest = try await modelPackStore.manifest(for: "legacy-model-id")
        XCTAssertEqual(legacyManifest?.id, "legacy-pack-row")
        XCTAssertEqual(legacyManifest?.modelID, "legacy-model-id")
        XCTAssertEqual(legacyManifest?.displayName, "Legacy Model")
        XCTAssertEqual(legacyManifest?.tokenizerType, "legacy-tokenizer")
        XCTAssertEqual(legacyManifest?.artifactSpecs.first?.checksum, "abc123")
    }

    func testAppDatabaseMigratesLegacyQwenVoiceConditioningFormatForward() async throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let databaseURL = makeFileBackedDatabaseURL(in: root)
        try createLegacyV12VoiceDatabaseWithQwenClone(at: databaseURL)

        let db = try AppDatabase(path: databaseURL.path, allowedDirectories: [root])
        let voiceStore = GRDBVoiceStore(db: db)

        let migrations = try await appliedMigrations(in: db)
        XCTAssertEqual(migrations, currentMigrationIDs)

        let voiceColumns = try await columnNames(in: db, table: "voice")
        XCTAssertTrue(voiceColumns.contains("conditioningFormat"))

        let voices = try await voiceStore.fetchAll()
        XCTAssertEqual(voices.count, 1)
        XCTAssertEqual(
            voices.first?.conditioningFormat,
            VoiceLibraryRecord.qwenSpeakerEmbeddingConditioningFormat
        )
    }

    func testGRDBProjectStoreReturnsEmptyResultsForFreshDatabase() async throws {
        let db = try AppDatabase.inMemory()
        let store = GRDBProjectStore(db: db)

        let projects = try await store.fetchAll()
        XCTAssertTrue(projects.isEmpty)

        let project = try await store.project(id: UUID())
        XCTAssertNil(project)

        let bundleLocation = try await store.bundleLocation(for: UUID())
        XCTAssertNil(bundleLocation)

        try await store.delete(UUID())
        let projectCount = try await rowCount(in: db, table: "project")
        XCTAssertEqual(projectCount, 0)
    }

    func testGRDBProjectStoreSortsByUpdatedAtDescendingAndSaveUpserts() async throws {
        let db = try AppDatabase.inMemory()
        let store = GRDBProjectStore(db: db)

        let oldest = ProjectRecord(
            id: UUID(uuidString: "AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE")!,
            title: "Oldest",
            createdAt: Date(timeIntervalSince1970: 10),
            updatedAt: Date(timeIntervalSince1970: 20),
            notes: "first"
        )
        let middle = ProjectRecord(
            id: UUID(uuidString: "11111111-2222-3333-4444-555555555555")!,
            title: "Middle",
            createdAt: Date(timeIntervalSince1970: 11),
            updatedAt: Date(timeIntervalSince1970: 30)
        )
        let newest = ProjectRecord(
            id: UUID(uuidString: "66666666-7777-8888-9999-AAAAAAAAAAAA")!,
            title: "Newest",
            createdAt: Date(timeIntervalSince1970: 12),
            updatedAt: Date(timeIntervalSince1970: 40)
        )

        try await store.save(oldest)
        try await store.save(middle)
        try await store.save(newest)

        let orderedTitles = try await store.fetchAll().map(\.title)
        XCTAssertEqual(orderedTitles, ["Newest", "Middle", "Oldest"])

        var updatedOldest = oldest
        updatedOldest.title = "Oldest Revised"
        updatedOldest.notes = "updated"
        updatedOldest.updatedAt = Date(timeIntervalSince1970: 50)
        try await store.save(updatedOldest)

        let allProjects = try await store.allProjects()
        XCTAssertEqual(allProjects.count, 3)
        XCTAssertEqual(allProjects.first?.title, "Oldest Revised")

        let projectCount = try await rowCount(in: db, table: "project")
        XCTAssertEqual(projectCount, 3)

        let updatedProject = try await store.project(id: oldest.id)
        XCTAssertEqual(updatedProject?.notes, "updated")
    }

    func testGRDBProjectStoreSupportsConcurrentWrites() async throws {
        let db = try AppDatabase.inMemory()
        let store = GRDBProjectStore(db: db)

        try await withThrowingTaskGroup(of: Void.self) { group in
            for index in 0..<24 {
                group.addTask {
                    _ = try await store.insert(title: "Project \(index)", notes: "Note \(index)")
                }
            }
            try await group.waitForAll()
        }

        let projects = try await store.fetchAll()
        XCTAssertEqual(projects.count, 24)
    }

    func testGRDBProjectStorePersistsExportsInExportTable() async throws {
        let db = try AppDatabase.inMemory()
        let store = GRDBProjectStore(db: db)
        let project = try await store.insert(title: "Exports", notes: nil)
        let createdAt = Date(timeIntervalSince1970: 123)
        let export = ExportRecord(
            id: UUID(uuidString: "BBBBBBBB-CCCC-DDDD-EEEE-FFFFFFFFFFFF")!,
            projectID: project.id,
            fileName: "chapter.wav",
            createdAt: createdAt,
            checksum: "abc123"
        )

        try await store.insert(export)

        let exportCount = try await rowCount(in: db, table: "export")
        XCTAssertEqual(exportCount, 1)

        let persistedRow: Row? = try await db.reader.read { database in
            try Row.fetchOne(
                database,
                sql: "SELECT id, projectID, fileName, createdAt, checksum FROM \"export\""
            )
        }
        let persistedID: String? = persistedRow?["id"]
        let persistedProjectID: String? = persistedRow?["projectID"]
        let persistedFileName: String? = persistedRow?["fileName"]
        let persistedCreatedAt: String? = persistedRow?["createdAt"]
        let persistedChecksum: String? = persistedRow?["checksum"]

        XCTAssertEqual(persistedID, export.id.uuidString)
        XCTAssertEqual(persistedProjectID, export.projectID.uuidString)
        XCTAssertEqual(persistedFileName, export.fileName)
        XCTAssertEqual(persistedCreatedAt, ISO8601DateFormatter().string(from: createdAt))
        XCTAssertEqual(persistedChecksum, export.checksum)
    }

    func testGRDBVoiceStoreReturnsEmptyResultsForFreshDatabase() async throws {
        let db = try AppDatabase.inMemory()
        let store = GRDBVoiceStore(db: db)

        let voices = try await store.fetchAll()
        XCTAssertTrue(voices.isEmpty)

        let listedVoices = try await store.list()
        XCTAssertTrue(listedVoices.isEmpty)

        let voice = try await store.voice(id: UUID())
        XCTAssertNil(voice)

        try await store.delete(UUID())
        let voiceCount = try await rowCount(in: db, table: "voice")
        XCTAssertEqual(voiceCount, 0)
    }

    func testGRDBVoiceStoreOrdersPresetAndCustomVoicesAndUpsertsDuplicateInsert() async throws {
        let db = try AppDatabase.inMemory()
        let store = GRDBVoiceStore(db: db)

        let custom = VoiceLibraryRecord(
            id: UUID(uuidString: "AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE")!,
            label: "Studio Clone",
            modelID: "qwen3-custom",
            runtimeModelID: "qwen3-base",
            referenceAudioAssetName: "voice.wav",
            referenceTranscript: "Custom transcript",
            createdAt: Date(timeIntervalSince1970: 10)
        )
        let preset = VoiceLibraryRecord(
            id: UUID(uuidString: "11111111-2222-3333-4444-555555555555")!,
            label: "Preset Narrator",
            modelID: "qwen3-base",
            sourceAssetName: "preset.wav",
            createdAt: Date(timeIntervalSince1970: 20)
        )

        _ = try await store.insert(preset)
        _ = try await store.insert(custom)

        let orderedVoices = try await store.fetchAll()
        XCTAssertEqual(orderedVoices.map(\.label), ["Studio Clone", "Preset Narrator"])
        XCTAssertTrue(orderedVoices[0].isClonedVoice)
        XCTAssertFalse(orderedVoices[1].isClonedVoice)

        var updatedPreset = preset
        updatedPreset.voicePrompt = "documentary tone"
        _ = try await store.insert(updatedPreset)

        let voiceCount = try await rowCount(in: db, table: "voice")
        XCTAssertEqual(voiceCount, 2)

        let updatedVoice = try await store.voice(id: preset.id)
        XCTAssertEqual(updatedVoice?.voicePrompt, "documentary tone")
    }

    func testGRDBVoiceStoreSupportsConcurrentWrites() async throws {
        let db = try AppDatabase.inMemory()
        let store = GRDBVoiceStore(db: db)

        try await withThrowingTaskGroup(of: Void.self) { group in
            for index in 0..<18 {
                group.addTask {
                    _ = try await store.save(
                        VoiceLibraryRecord(
                            label: "Voice \(index)",
                            modelID: "model-\(index)",
                            voicePrompt: "Prompt \(index)"
                        )
                    )
                }
            }
            try await group.waitForAll()
        }

        let voices = try await store.fetchAll()
        XCTAssertEqual(voices.count, 18)
    }

    func testGRDBRenderJobStoreReturnsEmptyResultsForFreshDatabase() async throws {
        let db = try AppDatabase.inMemory()
        let store = GRDBRenderJobStore(db: db)
        let projectID = UUID()

        let jobs = try await store.loadJobs()
        XCTAssertTrue(jobs.isEmpty)

        let projectJobs = try await store.jobs(forProject: projectID)
        XCTAssertTrue(projectJobs.isEmpty)

        let job = try await store.job(id: UUID())
        XCTAssertNil(job)

        try await store.remove(id: UUID())
        let renderJobCount = try await rowCount(in: db, table: "renderJob")
        XCTAssertEqual(renderJobCount, 0)
    }

    func testGRDBRenderJobStoreFiltersByProjectAndMaintainsQueueOrdering() async throws {
        let db = try AppDatabase.inMemory()
        let store = GRDBRenderJobStore(db: db)
        let primaryProjectID = UUID(uuidString: "AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE")!
        let secondaryProjectID = UUID(uuidString: "11111111-2222-3333-4444-555555555555")!

        let second = RenderJobRecord(
            id: UUID(uuidString: "66666666-7777-8888-9999-AAAAAAAAAAAA")!,
            projectID: primaryProjectID,
            modelID: "qwen3-base",
            chapterIDs: [UUID()],
            outputFileName: "second.wav",
            createdAt: Date(timeIntervalSince1970: 20),
            updatedAt: Date(timeIntervalSince1970: 20),
            priority: 2
        )
        let first = RenderJobRecord(
            id: UUID(uuidString: "BBBBBBBB-CCCC-DDDD-EEEE-FFFFFFFFFFFF")!,
            projectID: primaryProjectID,
            modelID: "qwen3-base",
            chapterIDs: [UUID()],
            outputFileName: "first.wav",
            createdAt: Date(timeIntervalSince1970: 10),
            updatedAt: Date(timeIntervalSince1970: 10),
            priority: 1
        )
        let third = RenderJobRecord(
            id: UUID(uuidString: "12345678-90AB-CDEF-1234-567890ABCDEF")!,
            projectID: secondaryProjectID,
            modelID: "qwen3-other",
            chapterIDs: [UUID()],
            outputFileName: "third.wav",
            createdAt: Date(timeIntervalSince1970: 30),
            updatedAt: Date(timeIntervalSince1970: 30)
        )

        try await store.save(second)
        try await store.save(first)
        try await store.save(third)

        let allJobs = try await store.loadJobs()
        XCTAssertEqual(allJobs.map(\.outputFileName), ["first.wav", "second.wav", "third.wav"])

        let primaryJobs = try await store.jobs(forProject: primaryProjectID)
        XCTAssertEqual(primaryJobs.map(\.outputFileName), ["first.wav", "second.wav"])

        let secondaryJobs = try await store.jobs(forProject: secondaryProjectID)
        XCTAssertEqual(secondaryJobs.map(\.outputFileName), ["third.wav"])
    }

    func testGRDBRenderJobStoreUpsertsStateTransitionsWithoutDuplicateRows() async throws {
        let db = try AppDatabase.inMemory()
        let store = GRDBRenderJobStore(db: db)
        let jobID = UUID(uuidString: "AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE")!

        let queued = RenderJobRecord(
            id: jobID,
            projectID: UUID(),
            modelID: "qwen3-base",
            chapterIDs: [UUID()],
            outputFileName: "queued.wav",
            createdAt: Date(timeIntervalSince1970: 10),
            updatedAt: Date(timeIntervalSince1970: 10),
            state: "queued",
            priority: 1,
            progress: 0
        )
        try await store.save(queued)

        var running = queued
        running.state = "running"
        running.updatedAt = Date(timeIntervalSince1970: 20)
        running.progress = 0.6
        running.title = "Batch Render"
        try await store.save(running)

        let renderJobCount = try await rowCount(in: db, table: "renderJob")
        XCTAssertEqual(renderJobCount, 1)
        let fetched = try await store.job(id: jobID)
        XCTAssertEqual(fetched?.state, "running")
        XCTAssertEqual(fetched?.progress, 0.6)
        XCTAssertEqual(fetched?.title, "Batch Render")
    }

    func testGRDBRenderJobStoreSupportsConcurrentWrites() async throws {
        let db = try AppDatabase.inMemory()
        let store = GRDBRenderJobStore(db: db)
        let projectID = UUID()

        try await withThrowingTaskGroup(of: Void.self) { group in
            for index in 0..<20 {
                group.addTask {
                    try await store.save(
                        RenderJobRecord(
                            projectID: projectID,
                            modelID: "model-\(index)",
                            chapterIDs: [UUID()],
                            outputFileName: "output-\(index).wav",
                            createdAt: Date(timeIntervalSince1970: TimeInterval(index)),
                            updatedAt: Date(timeIntervalSince1970: TimeInterval(index))
                        )
                    )
                }
            }
            try await group.waitForAll()
        }

        let jobs = try await store.jobs(forProject: projectID)
        XCTAssertEqual(jobs.count, 20)
    }

    func testGRDBModelPackStoreReturnsEmptyResultsForFreshDatabase() async throws {
        let db = try AppDatabase.inMemory()
        let store = GRDBModelPackStore(db: db)

        let manifest = try await store.manifest(for: "missing-model")
        XCTAssertNil(manifest)

        let installedRecord = try await store.installedRecord(for: "missing-model")
        XCTAssertNil(installedRecord)

        let supportedModel = try await store.supportedModel(for: "missing-model")
        XCTAssertNil(supportedModel)

        let supportedModels = try await store.supportedModels()
        XCTAssertTrue(supportedModels.isEmpty)

        let receipts = try await store.receipts()
        XCTAssertTrue(receipts.isEmpty)

        let ledgerEntries = try await store.ledgerEntries()
        XCTAssertTrue(ledgerEntries.isEmpty)

        let removedRecord = try await store.uninstall(modelID: "missing-model")
        XCTAssertNil(removedRecord)
    }

    func testGRDBModelPackStoreSortsCatalogReceiptsLedgerAndUpsertsInstalledStatus() async throws {
        let db = try AppDatabase.inMemory()
        let store = GRDBModelPackStore(db: db)

        let zeta = SupportedModelCatalogRecord(
            id: "catalog-zeta",
            familyID: "qwen3_tts",
            modelID: "model-zeta",
            displayName: "Zeta Voice",
            providerName: "Valar",
            sourceKind: .remoteURL
        )
        let alpha = SupportedModelCatalogRecord(
            id: "catalog-alpha",
            familyID: "qwen3_tts",
            modelID: "model-alpha",
            displayName: "Alpha Voice",
            providerName: "Valar",
            sourceKind: .localFile,
            isRecommended: true
        )
        try await store.saveCatalogEntry(zeta)
        try await store.saveCatalogEntry(alpha)

        let supportedModels = try await store.supportedModels()
        XCTAssertEqual(supportedModels.map(\.displayName), ["Alpha Voice", "Zeta Voice"])

        let supportedModel = try await store.supportedModel(for: "model-alpha")
        XCTAssertEqual(supportedModel?.providerName, "Valar")

        let installed = InstalledModelRecord(
            id: "installed-alpha",
            familyID: "qwen3_tts",
            modelID: "model-alpha",
            displayName: "Alpha Voice",
            installDate: Date(timeIntervalSince1970: 30),
            installedPath: "/tmp/model-alpha",
            manifestPath: "/tmp/model-alpha/manifest.json",
            artifactCount: 1,
            checksum: "abc123",
            sourceKind: .localFile,
            isEnabled: true
        )
        try await store.saveInstalledRecord(installed)

        var disabledInstalled = installed
        disabledInstalled.artifactCount = 4
        disabledInstalled.checksum = "def456"
        disabledInstalled.isEnabled = false
        try await store.saveInstalledRecord(disabledInstalled)

        let olderReceipt = ModelInstallReceipt(
            id: "receipt-1",
            modelID: "model-alpha",
            familyID: "qwen3_tts",
            sourceKind: .localFile,
            sourceLocation: "/tmp/alpha-1.valarmodel",
            installDate: Date(timeIntervalSince1970: 10),
            installedModelPath: installed.installedPath,
            manifestPath: installed.manifestPath,
            artifactCount: 1
        )
        let newerReceipt = ModelInstallReceipt(
            id: "receipt-2",
            modelID: "model-alpha",
            familyID: "qwen3_tts",
            sourceKind: .localFile,
            sourceLocation: "/tmp/alpha-2.valarmodel",
            installDate: Date(timeIntervalSince1970: 20),
            installedModelPath: installed.installedPath,
            manifestPath: installed.manifestPath,
            artifactCount: 4
        )
        try await store.saveReceipt(newerReceipt)
        try await store.saveReceipt(olderReceipt)

        let newerLedger = ModelInstallLedgerEntry(
            id: "ledger-2",
            receiptID: newerReceipt.id,
            sourceKind: .localFile,
            sourceLocation: newerReceipt.sourceLocation,
            recordedAt: Date(timeIntervalSince1970: 20),
            succeeded: true,
            message: "completed"
        )
        let olderLedger = ModelInstallLedgerEntry(
            id: "ledger-1",
            receiptID: olderReceipt.id,
            sourceKind: .localFile,
            sourceLocation: olderReceipt.sourceLocation,
            recordedAt: Date(timeIntervalSince1970: 10),
            succeeded: false,
            message: "retry"
        )
        try await store.saveLedgerEntry(newerLedger)
        try await store.saveLedgerEntry(olderLedger)

        let fetchedInstalled = try await store.installedRecord(for: "model-alpha")
        XCTAssertEqual(fetchedInstalled?.artifactCount, 4)
        XCTAssertEqual(fetchedInstalled?.checksum, "def456")
        XCTAssertFalse(fetchedInstalled?.isEnabled ?? true)
        let installedCount = try await rowCount(in: db, table: "installedModel")
        XCTAssertEqual(installedCount, 1)

        let receipts = try await store.receipts()
        XCTAssertEqual(receipts.map(\.id), ["receipt-1", "receipt-2"])

        let ledgerEntries = try await store.ledgerEntries()
        XCTAssertEqual(ledgerEntries.map(\.id), ["ledger-1", "ledger-2"])
    }

    func testGRDBModelPackStoreSupportsConcurrentManifestWrites() async throws {
        let db = try AppDatabase.inMemory()
        let store = GRDBModelPackStore(db: db)

        let makeManifest: @Sendable (String, String, String) -> ModelPackManifest = { id, modelID, displayName in
            Self.makeManifest(id: id, modelID: modelID, displayName: displayName)
        }

        try await withThrowingTaskGroup(of: Void.self) { group in
            for index in 0..<16 {
                group.addTask {
                    try await store.saveManifest(
                        makeManifest("manifest-\(index)", "model-\(index)", "Model \(index)")
                    )
                }
            }
            try await group.waitForAll()
        }

        let manifestCount = try await rowCount(in: db, table: "modelPack")
        XCTAssertEqual(manifestCount, 16)
    }

    func testGRDBRecordCodingReportsEncodingFailures() throws {
        enum SentinelError: Error {
            case encodingFailed
        }

        struct FailingEncodable: Encodable {
            func encode(to encoder: Encoder) throws {
                throw SentinelError.encodingFailed
            }
        }

        XCTAssertThrowsError(
            try encodeJSONColumn(FailingEncodable(), table: "modelPack", column: "artifactSpecs")
        ) { error in
            XCTAssertEqual(
                error as? GRDBRecordCodingError,
                .jsonEncodingFailed(
                    table: "modelPack",
                    column: "artifactSpecs",
                    underlying: SentinelError.encodingFailed.localizedDescription
                )
            )
        }
    }

    func testValarPathValidationErrorsExposeReadableDescriptions() throws {
        XCTAssertEqual(
            ValarPathValidationError.emptyPathValue("path").errorDescription,
            "path must not be empty"
        )
        XCTAssertEqual(
            ValarPathValidationError.absolutePathNotAllowed(label: "voice", value: "/tmp/file.wav").errorDescription,
            "voice must stay relative"
        )
        XCTAssertEqual(
            ValarPathValidationError.pathTraversalDetected(label: "model", value: "../secret").errorDescription,
            "model contains path traversal components"
        )
        XCTAssertEqual(
            ValarPathValidationError.pathEscapesContainment(path: "/tmp/out", allowedDirectory: "/tmp/in").errorDescription,
            "Resolved path escapes the allowed directory"
        )

        XCTAssertThrowsError(try ValarAppPaths.validateRelativePath("   ", label: "title")) { error in
            XCTAssertEqual(error as? ValarPathValidationError, .emptyPathValue("title"))
        }
        XCTAssertThrowsError(try ValarAppPaths.validateRelativePath("/tmp/absolute", label: "title")) { error in
            XCTAssertEqual(
                error as? ValarPathValidationError,
                .absolutePathNotAllowed(label: "title", value: "/tmp/absolute")
            )
        }
        XCTAssertThrowsError(try ValarAppPaths(baseURL: URL(fileURLWithPath: "/tmp")).modelPackDirectory(familyID: " ", modelID: "model")) { error in
            XCTAssertEqual(error as? ValarPathValidationError, .emptyPathValue("model family"))
        }
    }

    func testValarPersistenceValueTypesExposeExpectedDerivedValues() {
        let manifest = Self.makeManifest(
            id: "manifest-value",
            familyID: "qwen3_tts",
            modelID: "model-value",
            displayName: "Value Model"
        )
        XCTAssertEqual(manifest.canonicalPackURL, "valarmodel://qwen3_tts/model-value")

        let importedAsset = ImportedModelAsset(
            id: "asset-1",
            modelID: "model-value",
            assetName: "weights",
            relativePath: "weights/model.safetensors",
            byteCount: 128,
            checksum: "sum"
        )
        XCTAssertEqual(importedAsset.assetName, "weights")

        let legacySource = LegacyImportSource(kind: .pythonWorkspace, rootPath: "/tmp/legacy")
        let step = MigrationStep(title: "copy", succeeded: true, message: "done")
        let plan = MigrationPlan(source: legacySource, steps: [step], notes: "planned")
        XCTAssertEqual(plan.steps, [step])

        let outcome = ImportOutcome(
            source: legacySource,
            importedProjectCount: 2,
            importedVoiceCount: 3,
            warnings: ["check assets"]
        )
        XCTAssertEqual(outcome.warnings, ["check assets"])

        let layout = ModelPackDirectoryLayout(
            rootDirectory: URL(fileURLWithPath: "/tmp/root"),
            manifestURL: URL(fileURLWithPath: "/tmp/root/manifest.json"),
            artifactsDirectory: URL(fileURLWithPath: "/tmp/root/artifacts"),
            tokenizerDirectory: URL(fileURLWithPath: "/tmp/root/tokenizer"),
            licenseDirectory: URL(fileURLWithPath: "/tmp/root/license"),
            checksumsDirectory: URL(fileURLWithPath: "/tmp/root/checksums")
        )
        XCTAssertEqual(layout.manifestURL.lastPathComponent, "manifest.json")
    }

    func testProjectStoreSupportsLifecycleOrderingAndRemoval() async throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        let store = ProjectStore(paths: ValarAppPaths(baseURL: root))

        let firstProject = try await store.create(title: "First", notes: "draft")
        var updatedFirstProject = firstProject
        updatedFirstProject.updatedAt = Date(timeIntervalSince1970: 30)
        updatedFirstProject.notes = "updated"
        await store.update(updatedFirstProject)

        let secondProject = try await store.create(title: "Second", notes: nil)
        var updatedSecondProject = secondProject
        updatedSecondProject.updatedAt = Date(timeIntervalSince1970: 40)
        await store.update(updatedSecondProject)

        let firstChapter = ChapterRecord(
            id: UUID(uuidString: "AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE")!,
            projectID: firstProject.id,
            index: 1,
            title: "Second Chapter",
            script: "later"
        )
        let replacementFirstChapter = ChapterRecord(
            id: firstChapter.id,
            projectID: firstProject.id,
            index: 0,
            title: "Opening Chapter",
            script: "earlier"
        )
        let anotherChapter = ChapterRecord(
            id: UUID(uuidString: "11111111-2222-3333-4444-555555555555")!,
            projectID: firstProject.id,
            index: 2,
            title: "Final Chapter",
            script: "end"
        )
        await store.addChapter(firstChapter)
        await store.updateChapter(ChapterRecord(id: UUID(), projectID: firstProject.id, index: 9, title: "Missing", script: "noop"))
        await store.updateChapter(replacementFirstChapter)
        await store.addChapter(anotherChapter)

        let firstJob = RenderJobRecord(
            id: UUID(uuidString: "66666666-7777-8888-9999-AAAAAAAAAAAA")!,
            projectID: firstProject.id,
            chapterIDs: [replacementFirstChapter.id],
            outputFileName: "later.wav",
            createdAt: Date(timeIntervalSince1970: 20)
        )
        let earlierReplacementJob = RenderJobRecord(
            id: firstJob.id,
            projectID: firstProject.id,
            chapterIDs: [replacementFirstChapter.id],
            outputFileName: "earlier.wav",
            createdAt: Date(timeIntervalSince1970: 10)
        )
        await store.addRenderJob(firstJob)
        await store.addRenderJob(earlierReplacementJob)

        let export = ExportRecord(
            id: UUID(uuidString: "BBBBBBBB-CCCC-DDDD-EEEE-FFFFFFFFFFFF")!,
            projectID: firstProject.id,
            fileName: "chapter.wav",
            createdAt: Date(timeIntervalSince1970: 15)
        )
        await store.addExport(export)
        let narrator = ProjectSpeakerRecord(
            id: UUID(uuidString: "ABCDEFAB-CDEF-ABCD-EFAB-CDEFABCDEFAB")!,
            projectID: firstProject.id,
            name: "Narrator",
            voiceModelID: "test/narrator",
            language: "en"
        )
        let companion = ProjectSpeakerRecord(
            id: UUID(uuidString: "FEDCBAFE-DCBA-FEDC-BAFE-DCBAFEDCBAFE")!,
            projectID: firstProject.id,
            name: "Companion",
            voiceModelID: nil,
            language: "auto"
        )
        await store.addSpeaker(narrator)
        await store.addSpeaker(companion)
        await store.removeSpeaker(narrator.id, from: firstProject.id)

        let restoredProject = ProjectRecord(
            id: UUID(uuidString: "12345678-90AB-CDEF-1234-567890ABCDEF")!,
            title: "Restored",
            createdAt: Date(timeIntervalSince1970: 1),
            updatedAt: Date(timeIntervalSince1970: 50)
        )
        let restoredChapter = ChapterRecord(projectID: restoredProject.id, index: 0, title: "Restored Chapter", script: "restored")
        let restoredJob = RenderJobRecord(projectID: restoredProject.id, chapterIDs: [restoredChapter.id], outputFileName: "restored.wav", createdAt: Date(timeIntervalSince1970: 2))
        let restoredExport = ExportRecord(projectID: restoredProject.id, fileName: "restored-export.wav", createdAt: Date(timeIntervalSince1970: 3))
        let restoredSpeaker = ProjectSpeakerRecord(
            projectID: restoredProject.id,
            name: "Restored Speaker",
            voiceModelID: "test/restored",
            language: "fr"
        )
        await store.restore(
            project: restoredProject,
            chapters: [restoredChapter],
            renderJobs: [restoredJob],
            exports: [restoredExport],
            speakers: [restoredSpeaker]
        )

        let orderedProjects = await store.allProjects()
        XCTAssertEqual(orderedProjects.map(\.title), ["Restored", "Second", "First"])

        let chapters = await store.chapters(for: firstProject.id)
        XCTAssertEqual(chapters.map(\.title), ["Opening Chapter", "Final Chapter"])

        let renderJobs = await store.renderJobs(for: firstProject.id)
        XCTAssertEqual(renderJobs.map(\.outputFileName), ["earlier.wav"])

        let exports = await store.exports(for: firstProject.id)
        XCTAssertEqual(exports.map(\.fileName), ["chapter.wav"])

        let speakers = await store.speakers(for: firstProject.id)
        XCTAssertEqual(speakers.map(\.name), ["Companion"])

        let missingBundleURL = await store.bundleURL(for: UUID())
        XCTAssertNil(missingBundleURL)

        let existingBundleURL = await store.bundleURL(for: firstProject.id)
        XCTAssertNil(existingBundleURL)

        let missingBundleLocation = await store.bundleLocation(for: UUID())
        XCTAssertNil(missingBundleLocation)

        let documentURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
            .appendingPathExtension("valarproject")
        await store.updateBundleURL(documentURL, for: firstProject.id)

        let updatedBundleURL = await store.bundleURL(for: firstProject.id)
        XCTAssertEqual(updatedBundleURL, documentURL)

        let existingBundleLocation = await store.bundleLocation(for: firstProject.id)
        XCTAssertEqual(existingBundleLocation?.bundleURL, documentURL)

        await store.remove(id: firstProject.id)
        let remainingChapters = await store.chapters(for: firstProject.id)
        XCTAssertTrue(remainingChapters.isEmpty)

        let remainingJobs = await store.renderJobs(for: firstProject.id)
        XCTAssertTrue(remainingJobs.isEmpty)

        let remainingExports = await store.exports(for: firstProject.id)
        XCTAssertTrue(remainingExports.isEmpty)

        let remainingSpeakers = await store.speakers(for: firstProject.id)
        XCTAssertTrue(remainingSpeakers.isEmpty)
    }

    func testVoiceLibraryStoreMaintainsOrderingAndDelete() async throws {
        let first = VoiceLibraryRecord(
            id: UUID(uuidString: "AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE")!,
            label: "Later",
            modelID: "later",
            createdAt: Date(timeIntervalSince1970: 20)
        )
        let second = VoiceLibraryRecord(
            id: UUID(uuidString: "11111111-2222-3333-4444-555555555555")!,
            label: "Earlier",
            modelID: "earlier",
            createdAt: Date(timeIntervalSince1970: 10)
        )
        let store = VoiceLibraryStore(records: [first])

        _ = try await store.save(first)
        _ = try await store.save(second)
        let orderedVoices = await store.list()
        XCTAssertEqual(orderedVoices.map(\.label), ["Earlier", "Later"])

        try await store.delete(first.id)
        let remainingVoices = await store.list()
        XCTAssertEqual(remainingVoices.map(\.label), ["Earlier"])
    }

    func testMigrationLedgerSupportsManualRecordEntries() async {
        let step = MigrationStep(title: "Copy Legacy Files", succeeded: false, message: "missing file")
        let ledger = MigrationLedger()

        await ledger.record(step)

        let steps = await ledger.steps()
        XCTAssertEqual(steps, [step])
    }

    func testModelPackRegistryQueriesAndUninstallReturnExpectedValues() async throws {
        let seededManifest = Self.makeManifest(
            id: "seed-manifest",
            familyID: "seed-family",
            modelID: "seed-model",
            displayName: "Seed Model"
        )
        let seededRecord = InstalledModelRecord(
            id: "seed-record",
            familyID: "seed-family",
            modelID: "seed-model",
            displayName: "Seed Model",
            installedPath: "/tmp/seed",
            manifestPath: "/tmp/seed/manifest.json",
            artifactCount: 1,
            sourceKind: .localFolder
        )
        let seededReceipt = ModelInstallReceipt(
            id: "seed-receipt",
            modelID: "seed-model",
            familyID: "seed-family",
            sourceKind: .importedArchive,
            sourceLocation: "/tmp/seed.valarmodel",
            installedModelPath: seededRecord.installedPath,
            manifestPath: seededRecord.manifestPath,
            artifactCount: 1,
            notes: "seeded"
        )

        let registry = ModelPackRegistry(
            manifests: [seededManifest],
            records: [seededRecord],
            receipts: [seededReceipt]
        )

        let supported = await registry.supportedModel(for: "seed-model")
        XCTAssertEqual(supported?.displayName, "seed-model")

        let manifest = await registry.manifest(for: "seed-model")
        XCTAssertEqual(manifest?.displayName, "Seed Model")

        let installed = await registry.installedRecord(for: "seed-model")
        XCTAssertEqual(installed?.id, "seed-record")

        let receipt = await registry.receipt(for: "seed-model")
        XCTAssertEqual(receipt?.id, "seed-receipt")

        let removed = await registry.uninstall(modelID: "seed-model")
        XCTAssertEqual(removed?.id, "seed-record")

        let manifestAfterRemoval = await registry.manifest(for: "seed-model")
        XCTAssertNil(manifestAfterRemoval)

        let receiptAfterRemoval = await registry.receipt(for: "seed-model")
        XCTAssertNil(receiptAfterRemoval)
    }
}
