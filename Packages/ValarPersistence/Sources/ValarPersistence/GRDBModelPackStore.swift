import Foundation
import GRDB

// MARK: - GRDB conformances for model pack types

extension ModelPackManifest: FetchableRecord, PersistableRecord {
    public static var databaseTableName: String { "modelPack" }

    public init(row: Row) throws {
        let decoder = JSONDecoder()
        self = try decodeJSONColumn(
            Self.self,
            from: row["manifestJSON"],
            table: Self.databaseTableName,
            column: "manifestJSON",
            decoder: decoder
        )
    }

    public func encode(to container: inout PersistenceContainer) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = .sortedKeys
        container["id"] = id
        container["familyID"] = familyID
        container["displayName"] = displayName
        container["isRecommended"] = isRecommended
        container["manifestJSON"] = try encodeJSONColumn(
            self,
            table: Self.databaseTableName,
            column: "manifestJSON",
            encoder: encoder
        )
    }
}

extension SupportedModelCatalogRecord: FetchableRecord, PersistableRecord {
    public static var databaseTableName: String { "catalogEntry" }

    public init(row: Row) throws {
        self.init(
            id: row["id"],
            familyID: row["familyID"],
            modelID: row["modelID"],
            displayName: row["displayName"],
            providerName: row["providerName"],
            providerURL: row["providerURL"],
            installHint: row["installHint"],
            sourceKind: ModelPackSourceKind(rawValue: row["sourceKind"]) ?? .localFile,
            isRecommended: row["isRecommended"]
        )
    }

    public func encode(to container: inout PersistenceContainer) throws {
        container["id"] = id
        container["familyID"] = familyID
        container["modelID"] = modelID
        container["displayName"] = displayName
        container["providerName"] = providerName
        container["providerURL"] = providerURL
        container["installHint"] = installHint
        container["sourceKind"] = sourceKind.rawValue
        container["isRecommended"] = isRecommended
    }
}

extension InstalledModelRecord: FetchableRecord, PersistableRecord {
    public static var databaseTableName: String { "installedModel" }

    public init(row: Row) throws {
        let dateFormatter = ISO8601DateFormatter()
        self.init(
            id: row["id"],
            familyID: row["familyID"],
            modelID: row["modelID"],
            displayName: row["displayName"],
            installDate: dateFormatter.date(from: row["installDate"]) ?? .now,
            installedPath: row["installedPath"],
            manifestPath: row["manifestPath"],
            artifactCount: row["artifactCount"],
            checksum: row["checksum"],
            sourceKind: ModelPackSourceKind(rawValue: row["sourceKind"]) ?? .localFile,
            isEnabled: row["isEnabled"]
        )
    }

    public func encode(to container: inout PersistenceContainer) throws {
        let dateFormatter = ISO8601DateFormatter()
        container["id"] = id
        container["familyID"] = familyID
        container["modelID"] = modelID
        container["displayName"] = displayName
        container["installDate"] = dateFormatter.string(from: installDate)
        container["installedPath"] = installedPath
        container["manifestPath"] = manifestPath
        container["artifactCount"] = artifactCount
        container["checksum"] = checksum
        container["sourceKind"] = sourceKind.rawValue
        container["isEnabled"] = isEnabled
    }
}

extension ModelInstallReceipt: FetchableRecord, PersistableRecord {
    public static var databaseTableName: String { "installReceipt" }

    public init(row: Row) throws {
        let dateFormatter = ISO8601DateFormatter()
        self.init(
            id: row["id"],
            modelID: row["modelID"],
            familyID: row["familyID"],
            sourceKind: ModelPackSourceKind(rawValue: row["sourceKind"]) ?? .localFile,
            sourceLocation: row["sourceLocation"],
            installDate: dateFormatter.date(from: row["installDate"]) ?? .now,
            installedModelPath: row["installedModelPath"],
            manifestPath: row["manifestPath"],
            checksum: row["checksum"],
            artifactCount: row["artifactCount"],
            notes: row["notes"]
        )
    }

    public func encode(to container: inout PersistenceContainer) throws {
        let dateFormatter = ISO8601DateFormatter()
        container["id"] = id
        container["modelID"] = modelID
        container["familyID"] = familyID
        container["sourceKind"] = sourceKind.rawValue
        container["sourceLocation"] = sourceLocation
        container["installDate"] = dateFormatter.string(from: installDate)
        container["installedModelPath"] = installedModelPath
        container["manifestPath"] = manifestPath
        container["checksum"] = checksum
        container["artifactCount"] = artifactCount
        container["notes"] = notes
    }
}

extension ModelInstallLedgerEntry: FetchableRecord, PersistableRecord {
    public static var databaseTableName: String { "installLedger" }

    public init(row: Row) throws {
        let dateFormatter = ISO8601DateFormatter()
        self.init(
            id: row["id"],
            receiptID: row["receiptID"],
            sourceKind: ModelPackSourceKind(rawValue: row["sourceKind"]) ?? .localFile,
            sourceLocation: row["sourceLocation"],
            recordedAt: dateFormatter.date(from: row["recordedAt"]) ?? .now,
            succeeded: row["succeeded"],
            message: row["message"]
        )
    }

    public func encode(to container: inout PersistenceContainer) throws {
        let dateFormatter = ISO8601DateFormatter()
        container["id"] = id
        container["receiptID"] = receiptID
        container["sourceKind"] = sourceKind.rawValue
        container["sourceLocation"] = sourceLocation
        container["recordedAt"] = dateFormatter.string(from: recordedAt)
        container["succeeded"] = succeeded
        container["message"] = message
    }
}

// MARK: - GRDBModelPackStore

public final class GRDBModelPackStore: ModelPackStore, ModelCatalogStore, Sendable {
    private let db: AppDatabase

    public init(db: AppDatabase) {
        self.db = db
    }

    // MARK: ModelPackStore

    public func manifest(for modelID: String) async throws -> ModelPackManifest? {
        try await db.reader.read { db in
            try self.manifest(in: db, matching: modelID)
        }
    }

    public func installedRecord(for modelID: String) async throws -> InstalledModelRecord? {
        try await db.reader.read { db in
            try InstalledModelRecord.filter(Column("modelID") == modelID).fetchOne(db)
        }
    }

    public func receipts() async throws -> [ModelInstallReceipt] {
        try await db.reader.read { db in
            try ModelInstallReceipt.order(Column("installDate").asc).fetchAll(db)
        }
    }

    // MARK: ModelCatalogStore

    public func supportedModels() async throws -> [SupportedModelCatalogRecord] {
        try await db.reader.read { db in
            try SupportedModelCatalogRecord.order(Column("displayName").asc).fetchAll(db)
        }
    }

    public func supportedModel(for modelID: String) async throws -> SupportedModelCatalogRecord? {
        try await db.reader.read { db in
            try SupportedModelCatalogRecord.filter(Column("modelID") == modelID).fetchOne(db)
        }
    }

    // MARK: Write operations

    public func saveManifest(_ manifest: ModelPackManifest) async throws {
        try await db.writer.write { db in
            try manifest.save(db)
        }
    }

    public func saveCatalogEntry(_ entry: SupportedModelCatalogRecord) async throws {
        try await db.writer.write { db in
            try entry.save(db)
        }
    }

    public func saveInstalledRecord(_ record: InstalledModelRecord) async throws {
        try await db.writer.write { db in
            try record.save(db)
        }
    }

    public func saveReceipt(_ receipt: ModelInstallReceipt) async throws {
        try await db.writer.write { db in
            try receipt.save(db)
        }
    }

    public func saveLedgerEntry(_ entry: ModelInstallLedgerEntry) async throws {
        try await db.writer.write { db in
            try entry.save(db)
        }
    }

    public func uninstall(modelID: String) async throws -> InstalledModelRecord? {
        try await db.writer.write { db in
            let record = try InstalledModelRecord
                .filter(Column("modelID") == modelID)
                .fetchOne(db)
            let manifestIDsToDelete = try self.modelPackIDs(in: db, matching: modelID)

            _ = try InstalledModelRecord
                .filter(Column("modelID") == modelID)
                .deleteAll(db)
            _ = try ModelInstallReceipt
                .filter(Column("modelID") == modelID)
                .deleteAll(db)
            for manifestID in manifestIDsToDelete {
                try db.execute(
                    sql: "DELETE FROM modelPack WHERE id = ?",
                    arguments: [manifestID]
                )
            }
            return record
        }
    }

    public func ledgerEntries() async throws -> [ModelInstallLedgerEntry] {
        try await db.reader.read { db in
            try ModelInstallLedgerEntry.order(Column("recordedAt").asc).fetchAll(db)
        }
    }

    private func manifest(in db: Database, matching modelID: String) throws -> ModelPackManifest? {
        for row in try Row.fetchAll(db, sql: "SELECT * FROM modelPack") {
            let manifest = try ModelPackManifest(row: row)
            let storedID: String = row["id"]
            if storedID == modelID || manifest.modelID == modelID {
                return manifest
            }
        }

        return nil
    }

    private func modelPackIDs(in db: Database, matching modelID: String) throws -> [String] {
        var matchingIDs: [String] = []

        for row in try Row.fetchAll(db, sql: "SELECT * FROM modelPack") {
            let storedID: String = row["id"]
            if storedID == modelID {
                matchingIDs.append(storedID)
                continue
            }

            let manifest = try ModelPackManifest(row: row)
            if manifest.modelID == modelID {
                matchingIDs.append(storedID)
            }
        }

        return matchingIDs
    }
}
