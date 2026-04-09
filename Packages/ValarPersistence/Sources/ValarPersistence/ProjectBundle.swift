import Foundation
import GRDB

public enum ProjectBundleError: Error, LocalizedError, Equatable {
    case invalidBundle(URL)
    case missingManifest(URL)
    case missingSQLiteDatabase(URL)
    case missingProjectRecord(URL)
    case projectNotFound(UUID)
    case projectMismatch(expected: UUID, actual: UUID)

    public var errorDescription: String? {
        switch self {
        case let .invalidBundle(url):
            return "Project bundle at '\(url.path)' is invalid."
        case let .missingManifest(url):
            return "Project bundle manifest is missing at '\(url.path)'."
        case let .missingSQLiteDatabase(url):
            return "Project bundle database is missing at '\(url.path)'."
        case let .missingProjectRecord(url):
            return "Project bundle database at '\(url.path)' does not contain a project record."
        case let .projectNotFound(projectID):
            return "Project \(projectID.uuidString) was not found."
        case let .projectMismatch(expected, actual):
            return "Project bundle manifest expected project \(expected.uuidString) but database contained \(actual.uuidString)."
        }
    }
}

public struct ProjectBundleManifest: Codable, Sendable, Equatable {
    public struct ChapterSummary: Codable, Sendable, Equatable, Hashable, Identifiable {
        public let id: UUID
        public var index: Int
        public var title: String

        public init(id: UUID, index: Int, title: String) {
            self.id = id
            self.index = index
            self.title = title
        }
    }

    public var version: Int
    public var createdAt: Date
    public var projectID: UUID
    public var title: String
    public var modelID: String?
    public var renderSynthesisOptions: RenderSynthesisOptions
    public var chapters: [ChapterSummary]

    public init(
        version: Int = 1,
        createdAt: Date = .now,
        projectID: UUID,
        title: String,
        modelID: String? = nil,
        renderSynthesisOptions: RenderSynthesisOptions = RenderSynthesisOptions(),
        chapters: [ChapterSummary]
    ) {
        self.version = version
        self.createdAt = createdAt
        self.projectID = projectID
        self.title = title
        self.modelID = modelID
        self.renderSynthesisOptions = renderSynthesisOptions
        self.chapters = chapters
    }

    private enum CodingKeys: String, CodingKey {
        case version
        case createdAt
        case projectID
        case title
        case modelID
        case renderSynthesisOptions
        case chapters
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        version = try container.decode(Int.self, forKey: .version)
        createdAt = try container.decode(Date.self, forKey: .createdAt)
        projectID = try container.decode(UUID.self, forKey: .projectID)
        title = try container.decode(String.self, forKey: .title)
        modelID = try container.decodeIfPresent(String.self, forKey: .modelID)
        renderSynthesisOptions = try container.decodeIfPresent(
            RenderSynthesisOptions.self,
            forKey: .renderSynthesisOptions
        ) ?? RenderSynthesisOptions()
        chapters = try container.decode([ChapterSummary].self, forKey: .chapters)
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(version, forKey: .version)
        try container.encode(createdAt, forKey: .createdAt)
        try container.encode(projectID, forKey: .projectID)
        try container.encode(title, forKey: .title)
        try container.encodeIfPresent(modelID, forKey: .modelID)
        try container.encode(renderSynthesisOptions, forKey: .renderSynthesisOptions)
        try container.encode(chapters, forKey: .chapters)
    }
}

public struct ProjectBundleSnapshot: Codable, Sendable, Equatable {
    public var project: ProjectRecord
    public var modelID: String?
    public var renderSynthesisOptions: RenderSynthesisOptions
    public var chapters: [ChapterRecord]
    public var renderJobs: [RenderJobRecord]
    public var exports: [ExportRecord]
    public var speakers: [ProjectSpeakerRecord]

    public init(
        project: ProjectRecord,
        modelID: String? = nil,
        renderSynthesisOptions: RenderSynthesisOptions = RenderSynthesisOptions(),
        chapters: [ChapterRecord],
        renderJobs: [RenderJobRecord],
        exports: [ExportRecord],
        speakers: [ProjectSpeakerRecord] = []
    ) {
        self.project = project
        self.modelID = modelID
        self.renderSynthesisOptions = renderSynthesisOptions
        self.chapters = chapters
        self.renderJobs = renderJobs
        self.exports = exports
        self.speakers = speakers
    }

    private enum CodingKeys: String, CodingKey {
        case project
        case modelID
        case renderSynthesisOptions
        case chapters
        case renderJobs
        case exports
        case speakers
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        project = try container.decode(ProjectRecord.self, forKey: .project)
        modelID = try container.decodeIfPresent(String.self, forKey: .modelID)
        renderSynthesisOptions = try container.decodeIfPresent(
            RenderSynthesisOptions.self,
            forKey: .renderSynthesisOptions
        ) ?? RenderSynthesisOptions()
        chapters = try container.decode([ChapterRecord].self, forKey: .chapters)
        renderJobs = try container.decode([RenderJobRecord].self, forKey: .renderJobs)
        exports = try container.decode([ExportRecord].self, forKey: .exports)
        speakers = try container.decodeIfPresent([ProjectSpeakerRecord].self, forKey: .speakers) ?? []
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(project, forKey: .project)
        try container.encodeIfPresent(modelID, forKey: .modelID)
        try container.encode(renderSynthesisOptions, forKey: .renderSynthesisOptions)
        try container.encode(chapters, forKey: .chapters)
        try container.encode(renderJobs, forKey: .renderJobs)
        try container.encode(exports, forKey: .exports)
        try container.encode(speakers, forKey: .speakers)
    }
}

public struct ProjectBundle: Codable, Sendable, Equatable {
    public var manifest: ProjectBundleManifest
    public var snapshot: ProjectBundleSnapshot

    public init(manifest: ProjectBundleManifest, snapshot: ProjectBundleSnapshot) {
        self.manifest = manifest
        self.snapshot = snapshot
    }
}

extension ChapterRecord: FetchableRecord, PersistableRecord {
    public static var databaseTableName: String { "chapter" }

    public init(row: Row) throws {
        self.init(
            id: UUID(uuidString: row["id"]) ?? UUID(),
            projectID: UUID(uuidString: row["projectID"]) ?? UUID(),
            index: row["chapterIndex"],
            title: row["title"],
            script: row["script"],
            speakerLabel: row["speakerLabel"],
            estimatedDurationSeconds: row["estimatedDurationSeconds"],
            sourceAudioAssetName: row["sourceAudioAssetName"],
            sourceAudioSampleRate: row["sourceAudioSampleRate"],
            sourceAudioDurationSeconds: row["sourceAudioDurationSeconds"],
            transcriptionJSON: row["transcriptionJSON"],
            transcriptionModelID: row["transcriptionModelID"],
            alignmentJSON: row["alignmentJSON"],
            alignmentModelID: row["alignmentModelID"],
            derivedTranslationText: row["derivedTranslationText"]
        )
    }

    public func encode(to container: inout PersistenceContainer) throws {
        container["id"] = id.uuidString
        container["projectID"] = projectID.uuidString
        container["chapterIndex"] = index
        container["title"] = title
        container["script"] = script
        container["speakerLabel"] = speakerLabel
        container["estimatedDurationSeconds"] = estimatedDurationSeconds
        container["sourceAudioAssetName"] = sourceAudioAssetName
        container["sourceAudioSampleRate"] = sourceAudioSampleRate
        container["sourceAudioDurationSeconds"] = sourceAudioDurationSeconds
        container["transcriptionJSON"] = transcriptionJSON
        container["transcriptionModelID"] = transcriptionModelID
        container["alignmentJSON"] = alignmentJSON
        container["alignmentModelID"] = alignmentModelID
        container["derivedTranslationText"] = derivedTranslationText
    }
}

extension ExportRecord: FetchableRecord, PersistableRecord {
    public static var databaseTableName: String { "exportMetadata" }

    public init(row: Row) throws {
        let dateFormatter = ISO8601DateFormatter()
        self.init(
            id: UUID(uuidString: row["id"]) ?? UUID(),
            projectID: UUID(uuidString: row["projectID"]) ?? UUID(),
            fileName: row["fileName"],
            createdAt: dateFormatter.date(from: row["createdAt"]) ?? .now,
            checksum: row["checksum"]
        )
    }

    public func encode(to container: inout PersistenceContainer) throws {
        let dateFormatter = ISO8601DateFormatter()
        container["id"] = id.uuidString
        container["projectID"] = projectID.uuidString
        container["fileName"] = fileName
        container["createdAt"] = dateFormatter.string(from: createdAt)
        container["checksum"] = checksum
    }
}

private struct BundledProjectSpeakerRecord: FetchableRecord, PersistableRecord {
    static let databaseTableName = "projectSpeaker"

    let id: UUID
    let projectID: UUID
    let name: String
    let voiceModelID: String?
    let language: String
    let position: Int

    init(record: ProjectSpeakerRecord, position: Int) {
        self.id = record.id
        self.projectID = record.projectID
        self.name = record.name
        self.voiceModelID = record.voiceModelID
        self.language = record.language
        self.position = position
    }

    init(row: Row) {
        self.id = UUID(uuidString: row["id"]) ?? UUID()
        self.projectID = UUID(uuidString: row["projectID"]) ?? UUID()
        self.name = row["name"]
        self.voiceModelID = row["voiceModelID"]
        self.language = row["language"]
        self.position = row["position"]
    }

    func encode(to container: inout PersistenceContainer) {
        container["id"] = id.uuidString
        container["projectID"] = projectID.uuidString
        container["name"] = name
        container["voiceModelID"] = voiceModelID
        container["language"] = language
        container["position"] = position
    }

    var record: ProjectSpeakerRecord {
        ProjectSpeakerRecord(
            id: id,
            projectID: projectID,
            name: name,
            voiceModelID: voiceModelID,
            language: language
        )
    }
}

private final class ProjectBundleDatabase {
    private let queue: DatabaseQueue
    private let databaseURL: URL

    init(path: String, fileManager: FileManager = .default) throws {
        self.databaseURL = URL(fileURLWithPath: path, isDirectory: false)
        try fileManager.createDirectory(
            at: databaseURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )

        var configuration = Configuration()
        configuration.foreignKeysEnabled = true
        self.queue = try DatabaseQueue(path: databaseURL.path, configuration: configuration)
        try Self.migrator.migrate(queue)
    }

    func write(snapshot: ProjectBundleSnapshot) throws {
        try queue.write { db in
            try snapshot.project.save(db)
            try BundledProjectSpeakerRecord.deleteAll(db)

            for chapter in snapshot.chapters.sorted(by: { $0.index < $1.index }) {
                try chapter.save(db)
            }

            for renderJob in snapshot.renderJobs.sorted(by: { $0.createdAt < $1.createdAt }) {
                try renderJob.save(db)
            }

            for export in snapshot.exports.sorted(by: { $0.createdAt < $1.createdAt }) {
                try export.save(db)
            }

            for (position, speaker) in snapshot.speakers.enumerated() {
                try BundledProjectSpeakerRecord(record: speaker, position: position).save(db)
            }
        }
    }

    func readSnapshot(
        modelID: String?,
        renderSynthesisOptions: RenderSynthesisOptions
    ) throws -> ProjectBundleSnapshot {
        try queue.read { db in
            guard let project = try ProjectRecord.fetchOne(db, sql: "SELECT * FROM project LIMIT 1") else {
                throw ProjectBundleError.missingProjectRecord(databaseURL)
            }

            let chapters = try ChapterRecord.fetchAll(
                db,
                sql: "SELECT * FROM chapter ORDER BY chapterIndex ASC"
            )
            let renderJobs = try RenderJobRecord.fetchAll(
                db,
                sql: "SELECT * FROM renderJob ORDER BY queuePosition ASC, createdAt ASC"
            )
            let exports = try ExportRecord.fetchAll(
                db,
                sql: "SELECT * FROM exportMetadata ORDER BY createdAt ASC"
            )
            let speakers = try BundledProjectSpeakerRecord.fetchAll(
                db,
                sql: "SELECT * FROM projectSpeaker ORDER BY position ASC"
            )

            return ProjectBundleSnapshot(
                project: project,
                modelID: modelID,
                renderSynthesisOptions: renderSynthesisOptions,
                chapters: chapters,
                renderJobs: renderJobs,
                exports: exports,
                speakers: speakers.map(\.record)
            )
        }
    }

    private static var migrator: DatabaseMigrator {
        var migrator = DatabaseMigrator()

        migrator.registerMigration("v1_bundle") { db in
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

            try db.create(table: "renderJob") { t in
                t.primaryKey("id", .text)
                t.column("projectID", .text)
                    .notNull()
                    .references("project", column: "id", onDelete: .cascade)
                t.column("chapterIDs", .text).notNull()
                t.column("outputFileName", .text).notNull()
                t.column("state", .text).notNull().defaults(to: "queued")
                t.column("createdAt", .text).notNull()
                t.column("updatedAt", .text).notNull()
                t.column("modelID", .text).notNull().defaults(to: "")
                t.column("priority", .integer).notNull().defaults(to: 0)
                t.column("progress", .double).notNull().defaults(to: 0)
                t.column("title", .text)
                t.column("failureReason", .text)
                t.column("synthesisOptionsJSON", .text).notNull().defaults(to: "{}")
            }

            try db.create(table: "exportMetadata") { t in
                t.primaryKey("id", .text)
                t.column("projectID", .text)
                    .notNull()
                    .references("project", column: "id", onDelete: .cascade)
                t.column("fileName", .text).notNull()
                t.column("createdAt", .text).notNull()
                t.column("checksum", .text)
            }
        }

        migrator.registerMigration("v2_bundle_speakers") { db in
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

        migrator.registerMigration("v3_bundle_render_job_queue_position") { db in
            try db.alter(table: "renderJob") { t in
                t.add(column: "queuePosition", .integer).notNull().defaults(to: 0)
            }
        }

        migrator.registerMigration("v4_bundle_chapter_speech_metadata") { db in
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

        migrator.registerMigration("v5_bundle_render_job_synthesis_options") { db in
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
}

public final class ProjectBundleWriter {
    private let fileManager: FileManager
    private let bundleCommitter: (FileManager, ValarProjectBundleLocation, ValarProjectBundleLocation) throws -> Void

    public convenience init(fileManager: FileManager = .default) {
        self.init(
            fileManager: fileManager,
            bundleCommitter: { fileManager, temporaryLocation, destinationLocation in
                try Self.commitBundle(
                    using: fileManager,
                    from: temporaryLocation,
                    to: destinationLocation
                )
            }
        )
    }

    init(
        fileManager: FileManager = .default,
        bundleCommitter: @escaping (FileManager, ValarProjectBundleLocation, ValarProjectBundleLocation) throws -> Void
    ) {
        self.fileManager = fileManager
        self.bundleCommitter = bundleCommitter
    }

    @discardableResult
    public func write(
        _ snapshot: ProjectBundleSnapshot,
        to location: ValarProjectBundleLocation,
        createdAt: Date = .now
    ) throws -> ProjectBundleManifest {
        let normalizedLocation = ValarProjectBundleLocation(
            projectID: snapshot.project.id,
            title: snapshot.project.title,
            bundleURL: normalizedBundleURL(location.bundleURL)
        )
        let temporaryLocation = ValarProjectBundleLocation(
            projectID: snapshot.project.id,
            title: snapshot.project.title,
            bundleURL: temporaryBundleURL(for: normalizedLocation.bundleURL)
        )
        let manifest = bundleManifest(
            for: snapshot,
            createdAt: createdAt
        )

        try removeItemIfPresent(at: temporaryLocation.bundleURL)

        do {
            try prepareTemporaryBundle(at: temporaryLocation)

            // Write a fresh project-only bundle database instead of copying the
            // app-wide `valar.db`. That excludes saved voice conditioning
            // payloads (including Qwen speaker embeddings) and cloned-voice
            // reference audio stored in the separate voice library from
            // `.valarproject` exports.
            let database = try ProjectBundleDatabase(
                path: temporaryLocation.sqliteURL.path,
                fileManager: fileManager
            )
            try database.write(snapshot: snapshot)
            try writeManifest(manifest, to: temporaryLocation.manifestURL)
            try commitTemporaryBundle(
                from: temporaryLocation,
                to: normalizedLocation
            )
            try removeItemIfPresent(at: temporaryLocation.bundleURL)
            return manifest
        } catch {
            try? removeItemIfPresent(at: temporaryLocation.bundleURL)
            throw error
        }
    }

    private func normalizedBundleURL(_ bundleURL: URL) -> URL {
        guard bundleURL.pathExtension != "valarproject" else {
            return bundleURL
        }
        return bundleURL.appendingPathExtension("valarproject")
    }

    private func temporaryBundleURL(for bundleURL: URL) -> URL {
        bundleURL.appendingPathExtension("saving")
    }

    private func bundleManifest(
        for snapshot: ProjectBundleSnapshot,
        createdAt: Date
    ) -> ProjectBundleManifest {
        ProjectBundleManifest(
            createdAt: createdAt,
            projectID: snapshot.project.id,
            title: snapshot.project.title,
            modelID: snapshot.modelID,
            renderSynthesisOptions: snapshot.renderSynthesisOptions,
            chapters: snapshot.chapters
                .sorted(by: { $0.index < $1.index })
                .map { chapter in
                    ProjectBundleManifest.ChapterSummary(
                        id: chapter.id,
                        index: chapter.index,
                        title: chapter.title
                    )
                }
        )
    }

    private func prepareTemporaryBundle(at temporaryLocation: ValarProjectBundleLocation) throws {
        try fileManager.createDirectory(
            at: temporaryLocation.bundleURL,
            withIntermediateDirectories: true
        )
        try fileManager.createDirectory(at: temporaryLocation.assetsDirectory, withIntermediateDirectories: true)
        try fileManager.createDirectory(at: temporaryLocation.exportsDirectory, withIntermediateDirectories: true)
        try fileManager.createDirectory(at: temporaryLocation.cacheDirectory, withIntermediateDirectories: true)
    }

    private func writeManifest(
        _ manifest: ProjectBundleManifest,
        to manifestURL: URL
    ) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        let data = try encoder.encode(manifest)
        try data.write(to: manifestURL, options: .atomic)
    }

    private func commitTemporaryBundle(
        from temporaryLocation: ValarProjectBundleLocation,
        to destinationLocation: ValarProjectBundleLocation
    ) throws {
        try bundleCommitter(fileManager, temporaryLocation, destinationLocation)
    }

    private static func commitBundle(
        using fileManager: FileManager,
        from temporaryLocation: ValarProjectBundleLocation,
        to destinationLocation: ValarProjectBundleLocation
    ) throws {
        if !fileManager.fileExists(atPath: destinationLocation.bundleURL.path) {
            try fileManager.moveItem(at: temporaryLocation.bundleURL, to: destinationLocation.bundleURL)
            return
        }

        try ensureDirectoryExists(using: fileManager, at: destinationLocation.assetsDirectory)
        try ensureDirectoryExists(using: fileManager, at: destinationLocation.exportsDirectory)
        try ensureDirectoryExists(using: fileManager, at: destinationLocation.cacheDirectory)

        try replaceFile(
            using: fileManager,
            from: temporaryLocation.sqliteURL,
            to: destinationLocation.sqliteURL
        )
        try replaceFile(
            using: fileManager,
            from: temporaryLocation.manifestURL,
            to: destinationLocation.manifestURL
        )
    }

    private static func ensureDirectoryExists(
        using fileManager: FileManager,
        at directoryURL: URL
    ) throws {
        var isDirectory: ObjCBool = false
        if fileManager.fileExists(atPath: directoryURL.path, isDirectory: &isDirectory), isDirectory.boolValue {
            return
        }

        try fileManager.createDirectory(at: directoryURL, withIntermediateDirectories: true)
    }

    private static func replaceFile(
        using fileManager: FileManager,
        from sourceURL: URL,
        to destinationURL: URL
    ) throws {
        if fileManager.fileExists(atPath: destinationURL.path) {
            _ = try fileManager.replaceItemAt(
                destinationURL,
                withItemAt: sourceURL,
                backupItemName: nil,
                options: []
            )
            return
        }

        try fileManager.moveItem(at: sourceURL, to: destinationURL)
    }

    private func removeItemIfPresent(at url: URL) throws {
        guard fileManager.fileExists(atPath: url.path) else {
            return
        }
        try fileManager.removeItem(at: url)
    }
}

public final class ProjectBundleReader {
    private let fileManager: FileManager

    public init(fileManager: FileManager = .default) {
        self.fileManager = fileManager
    }

    public func read(from bundleURL: URL) throws -> ProjectBundle {
        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: bundleURL.path, isDirectory: &isDirectory), isDirectory.boolValue else {
            throw ProjectBundleError.invalidBundle(bundleURL)
        }

        let manifestURL = bundleURL.appendingPathComponent("manifest.json", isDirectory: false)
        guard fileManager.fileExists(atPath: manifestURL.path) else {
            throw ProjectBundleError.missingManifest(manifestURL)
        }

        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let manifest = try decoder.decode(ProjectBundleManifest.self, from: Data(contentsOf: manifestURL))
        let location = ValarProjectBundleLocation(
            projectID: manifest.projectID,
            title: manifest.title,
            bundleURL: bundleURL
        )

        guard fileManager.fileExists(atPath: location.sqliteURL.path) else {
            throw ProjectBundleError.missingSQLiteDatabase(location.sqliteURL)
        }

        let database = try ProjectBundleDatabase(
            path: location.sqliteURL.path,
            fileManager: fileManager
        )
        let snapshot = try database.readSnapshot(
            modelID: manifest.modelID,
            renderSynthesisOptions: manifest.renderSynthesisOptions
        )

        guard snapshot.project.id == manifest.projectID else {
            throw ProjectBundleError.projectMismatch(
                expected: manifest.projectID,
                actual: snapshot.project.id
            )
        }

        return ProjectBundle(manifest: manifest, snapshot: snapshot)
    }
}
