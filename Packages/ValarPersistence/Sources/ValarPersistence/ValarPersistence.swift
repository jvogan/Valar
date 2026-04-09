import Foundation
import ValarModelKit

public enum ValarPathValidationError: Error, LocalizedError, Equatable {
    case emptyPathValue(String)
    case absolutePathNotAllowed(label: String, value: String)
    case pathTraversalDetected(label: String, value: String)
    case pathEscapesContainment(path: String, allowedDirectory: String)
    case applicationSupportDirectoryUnavailable

    public var errorDescription: String? {
        switch self {
        case let .emptyPathValue(label):
            return "\(label) must not be empty"
        case let .absolutePathNotAllowed(label, _):
            return "\(label) must stay relative"
        case let .pathTraversalDetected(label, _):
            return "\(label) contains path traversal components"
        case .pathEscapesContainment:
            return "Resolved path escapes the allowed directory"
        case .applicationSupportDirectoryUnavailable:
            return "Application Support directory is unavailable; cannot safely determine storage location"
        }
    }
}

public struct ValarAppPaths: Sendable, Equatable {
    public let applicationSupport: URL
    public let modelPacksDirectory: URL
    public let projectsDirectory: URL
    public let voiceLibraryDirectory: URL
    public let cacheDirectory: URL
    public let importsDirectory: URL
    public let snapshotsDirectory: URL
    public let databaseURL: URL

    public init(fileManager: FileManager = .default) {
        if let override = Self.overrideBaseURL() {
            self.init(baseURL: override)
            return
        }
        let base = Self.defaultApplicationSupportDirectory(fileManager: fileManager)
            .appendingPathComponent("ValarTTS", isDirectory: true)
        self.init(baseURL: base)
    }

    public init(baseURL: URL) {
        self.applicationSupport = baseURL
        self.modelPacksDirectory = baseURL.appendingPathComponent("ModelPacks", isDirectory: true)
        self.projectsDirectory = baseURL.appendingPathComponent("Projects", isDirectory: true)
        self.voiceLibraryDirectory = baseURL.appendingPathComponent("VoiceLibrary", isDirectory: true)
        self.cacheDirectory = baseURL.appendingPathComponent("Cache", isDirectory: true)
        self.importsDirectory = baseURL.appendingPathComponent("Imports", isDirectory: true)
        self.snapshotsDirectory = baseURL.appendingPathComponent("Snapshots", isDirectory: true)
        self.databaseURL = baseURL.appendingPathComponent("valar.db", isDirectory: false)
    }

    public static func resolveSystemPaths(
        fileManager: FileManager = .default
    ) throws -> ValarAppPaths {
        guard let applicationSupportDirectory = fileManager.urls(
            for: .applicationSupportDirectory,
            in: .userDomainMask
        ).first else {
            throw ValarPathValidationError.applicationSupportDirectoryUnavailable
        }
        let base = applicationSupportDirectory
            .appendingPathComponent("ValarTTS", isDirectory: true)
        return ValarAppPaths(baseURL: base)
    }

    private static func defaultApplicationSupportDirectory(fileManager: FileManager) -> URL {
        if let applicationSupportDirectory = fileManager.urls(
            for: .applicationSupportDirectory,
            in: .userDomainMask
        ).first {
            return applicationSupportDirectory
        }

        return fileManager.homeDirectoryForCurrentUser
            .appendingPathComponent("Library", isDirectory: true)
            .appendingPathComponent("Application Support", isDirectory: true)
    }

    private static func overrideBaseURL(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> URL? {
        for key in ["VALARTTS_CLI_HOME", "VALARTTS_HOME"] {
            guard let rawValue = environment[key]?.trimmingCharacters(in: .whitespacesAndNewlines),
                  rawValue.isEmpty == false else {
                continue
            }

            return URL(fileURLWithPath: rawValue, isDirectory: true).standardizedFileURL
        }

        return nil
    }

    public static func validateContainment(
        _ candidate: URL,
        within allowedDirectory: URL,
        fileManager: FileManager = .default
    ) throws {
        let canonicalCandidate = try canonicalize(candidate, fileManager: fileManager)
        let canonicalAllowedDirectory = try canonicalize(allowedDirectory, fileManager: fileManager)

        guard contains(canonicalCandidate, within: canonicalAllowedDirectory) else {
            throw ValarPathValidationError.pathEscapesContainment(
                path: canonicalCandidate.path,
                allowedDirectory: canonicalAllowedDirectory.path
            )
        }
    }

    public func validateContainment(
        _ candidate: URL,
        within allowedDirectory: URL,
        fileManager: FileManager = .default
    ) throws {
        try Self.validateContainment(candidate, within: allowedDirectory, fileManager: fileManager)
    }

    public static func validateRelativePath(_ path: String, label: String = "path") throws {
        let trimmed = path.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw ValarPathValidationError.emptyPathValue(label)
        }
        guard !trimmed.hasPrefix("/") && !trimmed.hasPrefix("\\") else {
            throw ValarPathValidationError.absolutePathNotAllowed(label: label, value: path)
        }

        if pathComponents(in: trimmed).contains("..") {
            throw ValarPathValidationError.pathTraversalDetected(label: label, value: path)
        }
    }

    public func sanitizeBundleName(_ title: String) -> String {
        let allowed = CharacterSet.alphanumerics.union(.whitespaces)
        let collapsed = title
            .unicodeScalars
            .map { allowed.contains($0) ? Character($0) : " " }
            .reduce(into: "") { $0.append($1) }
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()

        return collapsed
            .components(separatedBy: .whitespacesAndNewlines)
            .filter { !$0.isEmpty }
            .joined(separator: "-")
    }

    public func modelPackDirectory(familyID: String, modelID: String) throws -> URL {
        try Self.validatePathIdentifier(familyID, label: "model family")
        try Self.validatePathIdentifier(modelID, label: "model identifier")
        let safeFamily = sanitizeBundleName(familyID)
        let safeModel = sanitizeBundleName(modelID)
        let directory = modelPacksDirectory
            .appendingPathComponent(safeFamily, isDirectory: true)
            .appendingPathComponent(safeModel, isDirectory: true)
        try validateContainment(directory, within: modelPacksDirectory)
        return directory
    }

    public func modelPackManifestURL(familyID: String, modelID: String) throws -> URL {
        let manifestURL = try modelPackDirectory(familyID: familyID, modelID: modelID)
            .appendingPathComponent("manifest.json", isDirectory: false)
        try validateContainment(manifestURL, within: modelPacksDirectory)
        return manifestURL
    }

    public func modelPackAssetsDirectory(familyID: String, modelID: String) throws -> URL {
        let assetsDirectory = try modelPackDirectory(familyID: familyID, modelID: modelID)
            .appendingPathComponent("assets", isDirectory: true)
        try validateContainment(assetsDirectory, within: modelPacksDirectory)
        return assetsDirectory
    }

    public func importedModelAssetDirectory(modelID: String) throws -> URL {
        try Self.validatePathIdentifier(modelID, label: "model identifier")
        let directory = importsDirectory.appendingPathComponent(sanitizeBundleName(modelID), isDirectory: true)
        try validateContainment(directory, within: importsDirectory)
        return directory
    }

    private static let allowedVoiceExtensions: Set<String> = ["wav", "aiff", "m4a", "mp3", "caf"]

    /// Returns the URL of the directory that stores an asset-backed conditioning bundle
    /// for a given voice. The directory name follows the convention `"<voiceID>-tada"`.
    public func voiceConditioningAssetURL(voiceID: UUID) throws -> URL {
        let assetURL = voiceLibraryDirectory
            .appendingPathComponent("\(voiceID.uuidString)-tada", isDirectory: true)
        try validateContainment(assetURL, within: voiceLibraryDirectory)
        return assetURL
    }

    public func voiceAssetURL(voiceID: UUID, fileExtension: String = "wav") throws -> URL {
        let sanitizedExtension = fileExtension
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .replacingOccurrences(of: ".", with: "")
            .lowercased()
        let resolvedExtension = Self.allowedVoiceExtensions.contains(sanitizedExtension) ? sanitizedExtension : "wav"
        let assetURL = voiceLibraryDirectory
            .appendingPathComponent(voiceID.uuidString, isDirectory: false)
            .appendingPathExtension(resolvedExtension)
        try validateContainment(assetURL, within: voiceLibraryDirectory)
        return assetURL
    }

    private static func validatePathIdentifier(_ value: String, label: String) throws {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw ValarPathValidationError.emptyPathValue(label)
        }
        if pathComponents(in: trimmed).contains("..") {
            throw ValarPathValidationError.pathTraversalDetected(label: label, value: value)
        }
    }

    private static func canonicalize(_ url: URL, fileManager: FileManager) throws -> URL {
        let standardized = url.standardizedFileURL
        var existingAncestor = standardized
        var unresolvedComponents: [String] = []

        while !fileManager.fileExists(atPath: existingAncestor.path) {
            let parent = existingAncestor.deletingLastPathComponent()
            if parent.path == existingAncestor.path {
                break
            }
            unresolvedComponents.insert(existingAncestor.lastPathComponent, at: 0)
            existingAncestor = parent
        }

        let resolvedAncestor = existingAncestor.resolvingSymlinksInPath().standardizedFileURL
        return unresolvedComponents.reduce(resolvedAncestor) { partial, component in
            partial.appendingPathComponent(component, isDirectory: false)
        }
    }

    private static func contains(_ candidate: URL, within allowedDirectory: URL) -> Bool {
        let candidateComponents = candidate.standardizedFileURL.pathComponents
        let allowedComponents = allowedDirectory.standardizedFileURL.pathComponents

        guard candidateComponents.count >= allowedComponents.count else {
            return false
        }

        return zip(allowedComponents, candidateComponents).allSatisfy(==)
    }

    private static func pathComponents(in value: String) -> [String] {
        value
            .replacingOccurrences(of: "\\", with: "/")
            .split(separator: "/")
            .map(String.init)
    }
}

public struct RenderSynthesisOptions: Codable, Sendable, Equatable {
    public var language: String
    public var temperature: Double?
    public var topP: Double?
    public var repetitionPenalty: Double?
    public var maxTokens: Int?
    public var voiceBehavior: SpeechSynthesisVoiceBehavior

    public init(
        language: String = "auto",
        temperature: Double? = 0.7,
        topP: Double? = 0.9,
        repetitionPenalty: Double? = 1.0,
        maxTokens: Int? = 8_192,
        voiceBehavior: SpeechSynthesisVoiceBehavior = .auto
    ) {
        self.language = language
        self.temperature = temperature
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
        self.maxTokens = maxTokens
        self.voiceBehavior = voiceBehavior
    }

    public var normalizedLanguage: String? {
        let trimmed = language.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.isEmpty == false, trimmed.caseInsensitiveCompare("auto") != .orderedSame else {
            return nil
        }
        return trimmed
    }

    private enum CodingKeys: String, CodingKey {
        case language
        case temperature
        case topP
        case repetitionPenalty
        case maxTokens
        case voiceBehavior
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.language = try container.decodeIfPresent(String.self, forKey: .language) ?? "auto"
        self.temperature = try container.decodeIfPresent(Double.self, forKey: .temperature) ?? 0.7
        self.topP = try container.decodeIfPresent(Double.self, forKey: .topP) ?? 0.9
        self.repetitionPenalty = try container.decodeIfPresent(Double.self, forKey: .repetitionPenalty) ?? 1.0
        self.maxTokens = try container.decodeIfPresent(Int.self, forKey: .maxTokens) ?? 8_192
        self.voiceBehavior = try container.decodeIfPresent(SpeechSynthesisVoiceBehavior.self, forKey: .voiceBehavior) ?? .auto
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(language, forKey: .language)
        try container.encodeIfPresent(temperature, forKey: .temperature)
        try container.encodeIfPresent(topP, forKey: .topP)
        try container.encodeIfPresent(repetitionPenalty, forKey: .repetitionPenalty)
        try container.encodeIfPresent(maxTokens, forKey: .maxTokens)
        try container.encode(voiceBehavior, forKey: .voiceBehavior)
    }
}

public enum DocumentBundleRole: String, CaseIterable, Codable, Sendable {
    case manifest
    case sqliteDatabase
    case assets
    case exports
    case cache
    case metadata
}

public struct DocumentBundleManifest: Codable, Sendable, Equatable, Identifiable {
    public let id: UUID
    public var title: String
    public var schemaVersion: Int
    public var projectVersion: Int
    public var createdAt: Date
    public var updatedAt: Date
    public var chapterCount: Int
    public var voiceCount: Int
    public var exportCount: Int
    public var modelPackCount: Int

    public init(
        id: UUID = UUID(),
        title: String,
        schemaVersion: Int = 1,
        projectVersion: Int = 1,
        createdAt: Date = .now,
        updatedAt: Date = .now,
        chapterCount: Int = 0,
        voiceCount: Int = 0,
        exportCount: Int = 0,
        modelPackCount: Int = 0
    ) {
        self.id = id
        self.title = title
        self.schemaVersion = schemaVersion
        self.projectVersion = projectVersion
        self.createdAt = createdAt
        self.updatedAt = updatedAt
        self.chapterCount = chapterCount
        self.voiceCount = voiceCount
        self.exportCount = exportCount
        self.modelPackCount = modelPackCount
    }
}

public enum ModelPackSourceKind: String, CaseIterable, Codable, Sendable {
    case localFile
    case localFolder
    case remoteURL
    case importedArchive
}

public struct ModelPackArtifact: Codable, Sendable, Equatable, Hashable, Identifiable {
    public let id: String
    public var kind: String
    public var relativePath: String
    public var checksum: String?
    public var byteCount: Int?
    public var required: Bool

    public init(
        id: String,
        kind: String,
        relativePath: String,
        checksum: String? = nil,
        byteCount: Int? = nil,
        required: Bool = true
    ) {
        self.id = id
        self.kind = kind
        self.relativePath = relativePath
        self.checksum = checksum
        self.byteCount = byteCount
        self.required = required
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(String.self, forKey: .id)
        kind = try container.decode(String.self, forKey: .kind)
        relativePath = try container.decode(String.self, forKey: .relativePath)
        checksum = try container.decodeIfPresent(String.self, forKey: .checksum)
        byteCount = try container.decodeIfPresent(Int.self, forKey: .byteCount)
        required = try container.decodeIfPresent(Bool.self, forKey: .required) ?? true
    }
}

public struct SupportedModelCatalogRecord: Codable, Sendable, Equatable, Hashable, Identifiable {
    public let id: String
    public var familyID: String
    public var modelID: String
    public var displayName: String
    public var providerName: String
    public var providerURL: String?
    public var installHint: String?
    public var sourceKind: ModelPackSourceKind
    public var isRecommended: Bool

    public init(
        id: String = UUID().uuidString,
        familyID: String,
        modelID: String,
        displayName: String,
        providerName: String,
        providerURL: String? = nil,
        installHint: String? = nil,
        sourceKind: ModelPackSourceKind = .localFile,
        isRecommended: Bool = false
    ) {
        self.id = id
        self.familyID = familyID
        self.modelID = modelID
        self.displayName = displayName
        self.providerName = providerName
        self.providerURL = providerURL
        self.installHint = installHint
        self.sourceKind = sourceKind
        self.isRecommended = isRecommended
    }
}

public struct InstalledModelRecord: Codable, Sendable, Equatable, Hashable, Identifiable {
    public let id: String
    public var familyID: String
    public var modelID: String
    public var displayName: String
    public var installDate: Date
    public var installedPath: String
    public var manifestPath: String
    public var artifactCount: Int
    public var checksum: String?
    public var sourceKind: ModelPackSourceKind
    public var isEnabled: Bool

    public init(
        id: String = UUID().uuidString,
        familyID: String,
        modelID: String,
        displayName: String,
        installDate: Date = .now,
        installedPath: String,
        manifestPath: String,
        artifactCount: Int = 0,
        checksum: String? = nil,
        sourceKind: ModelPackSourceKind,
        isEnabled: Bool = true
    ) {
        self.id = id
        self.familyID = familyID
        self.modelID = modelID
        self.displayName = displayName
        self.installDate = installDate
        self.installedPath = installedPath
        self.manifestPath = manifestPath
        self.artifactCount = artifactCount
        self.checksum = checksum
        self.sourceKind = sourceKind
        self.isEnabled = isEnabled
    }
}

public struct ModelPackManifest: Codable, Sendable, Equatable, Hashable, Identifiable {
    public let id: String
    public var schemaVersion: Int
    public var familyID: String
    public var modelID: String
    public var displayName: String
    public var isRecommended: Bool
    public var capabilities: [String]
    public var backendKinds: [String]
    public var tokenizerType: String?
    public var sampleRate: Double?
    public var artifactSpecs: [ModelPackArtifact]
    public var licenseName: String?
    public var licenseURL: String?
    public var minimumAppVersion: String?
    public var notes: String?

    public init(
        id: String = UUID().uuidString,
        schemaVersion: Int = 1,
        familyID: String,
        modelID: String,
        displayName: String,
        isRecommended: Bool = false,
        capabilities: [String] = [],
        backendKinds: [String] = [],
        tokenizerType: String? = nil,
        sampleRate: Double? = nil,
        artifactSpecs: [ModelPackArtifact] = [],
        licenseName: String? = nil,
        licenseURL: String? = nil,
        minimumAppVersion: String? = nil,
        notes: String? = nil
    ) {
        self.id = id
        self.schemaVersion = schemaVersion
        self.familyID = familyID
        self.modelID = modelID
        self.displayName = displayName
        self.isRecommended = isRecommended
        self.capabilities = capabilities
        self.backendKinds = backendKinds
        self.tokenizerType = tokenizerType
        self.sampleRate = sampleRate
        self.artifactSpecs = artifactSpecs
        self.licenseName = licenseName
        self.licenseURL = licenseURL
        self.minimumAppVersion = minimumAppVersion
        self.notes = notes
    }

    public var canonicalPackURL: String {
        "valarmodel://\(familyID)/\(modelID)"
    }
}

public struct ModelInstallReceipt: Codable, Sendable, Equatable, Hashable, Identifiable {
    public let id: String
    public var modelID: String
    public var familyID: String
    public var sourceKind: ModelPackSourceKind
    public var sourceLocation: String
    public var installDate: Date
    public var installedModelPath: String
    public var manifestPath: String
    public var checksum: String?
    public var artifactCount: Int
    public var notes: String?

    public init(
        id: String = UUID().uuidString,
        modelID: String,
        familyID: String,
        sourceKind: ModelPackSourceKind,
        sourceLocation: String,
        installDate: Date = .now,
        installedModelPath: String,
        manifestPath: String,
        checksum: String? = nil,
        artifactCount: Int = 0,
        notes: String? = nil
    ) {
        self.id = id
        self.modelID = modelID
        self.familyID = familyID
        self.sourceKind = sourceKind
        self.sourceLocation = sourceLocation
        self.installDate = installDate
        self.installedModelPath = installedModelPath
        self.manifestPath = manifestPath
        self.checksum = checksum
        self.artifactCount = artifactCount
        self.notes = notes
    }
}

public struct ModelInstallLedgerEntry: Codable, Sendable, Equatable, Hashable, Identifiable {
    public let id: String
    public var receiptID: String?
    public var sourceKind: ModelPackSourceKind
    public var sourceLocation: String
    public var recordedAt: Date
    public var succeeded: Bool
    public var message: String?

    public init(
        id: String = UUID().uuidString,
        receiptID: String? = nil,
        sourceKind: ModelPackSourceKind,
        sourceLocation: String,
        recordedAt: Date = .now,
        succeeded: Bool,
        message: String? = nil
    ) {
        self.id = id
        self.receiptID = receiptID
        self.sourceKind = sourceKind
        self.sourceLocation = sourceLocation
        self.recordedAt = recordedAt
        self.succeeded = succeeded
        self.message = message
    }
}

public struct ImportedModelAsset: Codable, Sendable, Equatable, Hashable, Identifiable {
    public let id: String
    public var modelID: String
    public var assetName: String
    public var relativePath: String
    public var byteCount: Int?
    public var checksum: String?
    public var importedAt: Date

    public init(
        id: String = UUID().uuidString,
        modelID: String,
        assetName: String,
        relativePath: String,
        byteCount: Int? = nil,
        checksum: String? = nil,
        importedAt: Date = .now
    ) {
        self.id = id
        self.modelID = modelID
        self.assetName = assetName
        self.relativePath = relativePath
        self.byteCount = byteCount
        self.checksum = checksum
        self.importedAt = importedAt
    }
}

public struct ValarProjectBundleLocation: Codable, Sendable, Equatable, Hashable {
    public let projectID: UUID
    public let title: String
    public let bundleURL: URL
    public let manifestURL: URL
    public let sqliteURL: URL
    public let assetsDirectory: URL
    public let exportsDirectory: URL
    public let cacheDirectory: URL

    public init(projectID: UUID, title: String, bundleURL: URL, manifestURL: URL, sqliteURL: URL, assetsDirectory: URL, exportsDirectory: URL, cacheDirectory: URL) {
        self.projectID = projectID
        self.title = title
        self.bundleURL = bundleURL
        self.manifestURL = manifestURL
        self.sqliteURL = sqliteURL
        self.assetsDirectory = assetsDirectory
        self.exportsDirectory = exportsDirectory
        self.cacheDirectory = cacheDirectory
    }

    public init(projectID: UUID, title: String, bundleURL: URL) {
        self.init(
            projectID: projectID,
            title: title,
            bundleURL: bundleURL,
            manifestURL: bundleURL.appendingPathComponent("manifest.json", isDirectory: false),
            sqliteURL: bundleURL.appendingPathComponent("\(projectID.uuidString).sqlite", isDirectory: false),
            assetsDirectory: bundleURL.appendingPathComponent("Assets", isDirectory: true),
            exportsDirectory: bundleURL.appendingPathComponent("Exports", isDirectory: true),
            cacheDirectory: bundleURL.appendingPathComponent("Cache", isDirectory: true)
        )
    }
}

public struct ModelPackDirectoryLayout: Codable, Sendable, Equatable, Hashable {
    public let rootDirectory: URL
    public let manifestURL: URL
    public let artifactsDirectory: URL
    public let tokenizerDirectory: URL
    public let licenseDirectory: URL
    public let checksumsDirectory: URL

    public init(rootDirectory: URL, manifestURL: URL, artifactsDirectory: URL, tokenizerDirectory: URL, licenseDirectory: URL, checksumsDirectory: URL) {
        self.rootDirectory = rootDirectory
        self.manifestURL = manifestURL
        self.artifactsDirectory = artifactsDirectory
        self.tokenizerDirectory = tokenizerDirectory
        self.licenseDirectory = licenseDirectory
        self.checksumsDirectory = checksumsDirectory
    }
}

public protocol ModelPackStore: Sendable {
    func manifest(for modelID: String) async throws -> ModelPackManifest?
    func installedRecord(for modelID: String) async throws -> InstalledModelRecord?
    func receipts() async throws -> [ModelInstallReceipt]
}

public protocol ModelCatalogStore: Sendable {
    func supportedModels() async throws -> [SupportedModelCatalogRecord]
    func supportedModel(for modelID: String) async throws -> SupportedModelCatalogRecord?
}

public protocol ModelPackManaging: ModelPackStore, ModelCatalogStore {
    func registerSupported(_ record: SupportedModelCatalogRecord) async throws
    func install(
        manifest: ModelPackManifest,
        sourceKind: ModelPackSourceKind,
        sourceLocation: String,
        notes: String?
    ) async throws -> ModelInstallReceipt
    func uninstall(modelID: String) async throws -> InstalledModelRecord?
    func ledgerEntries() async throws -> [ModelInstallLedgerEntry]
}

public protocol ProjectStoring: Sendable {
    func create(title: String, notes: String?) async throws -> ProjectRecord
    func update(_ project: ProjectRecord) async
    func addChapter(_ chapter: ChapterRecord) async
    func updateChapter(_ chapter: ChapterRecord) async
    func removeChapter(_ id: UUID, from projectID: UUID) async
    func attachAudio(
        to chapterID: UUID,
        in projectID: UUID,
        assetName: String?,
        sampleRate: Double?,
        durationSeconds: Double?
    ) async
    func setTranscription(
        for chapterID: UUID,
        in projectID: UUID,
        transcriptionJSON: String?,
        modelID: String?
    ) async
    func setAlignment(
        for chapterID: UUID,
        in projectID: UUID,
        alignmentJSON: String?,
        modelID: String?
    ) async
    func addRenderJob(_ job: RenderJobRecord) async
    func addExport(_ export: ExportRecord) async
    func addSpeaker(_ speaker: ProjectSpeakerRecord) async
    func removeSpeaker(_ id: UUID, from projectID: UUID) async
    func restore(
        project: ProjectRecord,
        chapters: [ChapterRecord],
        renderJobs: [RenderJobRecord],
        exports: [ExportRecord],
        speakers: [ProjectSpeakerRecord]
    ) async
    func allProjects() async -> [ProjectRecord]
    func chapters(for projectID: UUID) async -> [ChapterRecord]
    func renderJobs(for projectID: UUID) async -> [RenderJobRecord]
    func exports(for projectID: UUID) async -> [ExportRecord]
    func speakers(for projectID: UUID) async -> [ProjectSpeakerRecord]
    func bundleLocation(for projectID: UUID) async -> ValarProjectBundleLocation?
    func updateBundleURL(_ bundleURL: URL?, for projectID: UUID) async
    func remove(id: UUID) async
}

public extension ProjectStoring {
    func attachAudio(
        to chapterID: UUID,
        in projectID: UUID,
        assetName: String?,
        sampleRate: Double?,
        durationSeconds: Double?
    ) async {
        guard var chapter = await chapters(for: projectID).first(where: { $0.id == chapterID }) else {
            return
        }

        chapter.sourceAudioAssetName = assetName
        chapter.sourceAudioSampleRate = sampleRate
        chapter.sourceAudioDurationSeconds = durationSeconds
        await updateChapter(chapter)
    }

    func setTranscription(
        for chapterID: UUID,
        in projectID: UUID,
        transcriptionJSON: String?,
        modelID: String?
    ) async {
        guard var chapter = await chapters(for: projectID).first(where: { $0.id == chapterID }) else {
            return
        }

        chapter.transcriptionJSON = transcriptionJSON
        chapter.transcriptionModelID = modelID
        await updateChapter(chapter)
    }

    func setAlignment(
        for chapterID: UUID,
        in projectID: UUID,
        alignmentJSON: String?,
        modelID: String?
    ) async {
        guard var chapter = await chapters(for: projectID).first(where: { $0.id == chapterID }) else {
            return
        }

        chapter.alignmentJSON = alignmentJSON
        chapter.alignmentModelID = modelID
        await updateChapter(chapter)
    }
}

public protocol VoiceLibraryStoring: Sendable {
    func save(_ voice: VoiceLibraryRecord) async throws -> VoiceLibraryRecord
    func list() async -> [VoiceLibraryRecord]
    func delete(_ id: UUID) async throws
}

public struct ProjectRecord: Codable, Sendable, Equatable, Identifiable {
    public let id: UUID
    public var title: String
    public var createdAt: Date
    public var updatedAt: Date
    public var notes: String?

    public init(
        id: UUID = UUID(),
        title: String,
        createdAt: Date = .now,
        updatedAt: Date = .now,
        notes: String? = nil
    ) {
        self.id = id
        self.title = title
        self.createdAt = createdAt
        self.updatedAt = updatedAt
        self.notes = notes
    }
}

public struct ChapterRecord: Codable, Sendable, Equatable, Identifiable {
    public let id: UUID
    public var projectID: UUID
    public var index: Int
    public var title: String
    public var script: String
    public var speakerLabel: String?
    public var estimatedDurationSeconds: Double?
    public var sourceAudioAssetName: String?
    public var sourceAudioSampleRate: Double?
    public var sourceAudioDurationSeconds: Double?
    public var transcriptionJSON: String?
    public var transcriptionModelID: String?
    public var alignmentJSON: String?
    public var alignmentModelID: String?
    public var derivedTranslationText: String?

    public var hasSourceAudio: Bool {
        sourceAudioAssetName != nil
    }

    public init(
        id: UUID = UUID(),
        projectID: UUID,
        index: Int,
        title: String,
        script: String,
        speakerLabel: String? = nil,
        estimatedDurationSeconds: Double? = nil,
        sourceAudioAssetName: String? = nil,
        sourceAudioSampleRate: Double? = nil,
        sourceAudioDurationSeconds: Double? = nil,
        transcriptionJSON: String? = nil,
        transcriptionModelID: String? = nil,
        alignmentJSON: String? = nil,
        alignmentModelID: String? = nil,
        derivedTranslationText: String? = nil
    ) {
        self.id = id
        self.projectID = projectID
        self.index = index
        self.title = title
        self.script = script
        self.speakerLabel = speakerLabel
        self.estimatedDurationSeconds = estimatedDurationSeconds
        self.sourceAudioAssetName = sourceAudioAssetName
        self.sourceAudioSampleRate = sourceAudioSampleRate
        self.sourceAudioDurationSeconds = sourceAudioDurationSeconds
        self.transcriptionJSON = transcriptionJSON
        self.transcriptionModelID = transcriptionModelID
        self.alignmentJSON = alignmentJSON
        self.alignmentModelID = alignmentModelID
        self.derivedTranslationText = derivedTranslationText
    }
}

public struct ProjectSpeakerRecord: Codable, Sendable, Equatable, Identifiable {
    public let id: UUID
    public var projectID: UUID
    public var name: String
    public var voiceModelID: String?
    public var language: String

    public init(
        id: UUID = UUID(),
        projectID: UUID,
        name: String,
        voiceModelID: String? = nil,
        language: String = "auto"
    ) {
        self.id = id
        self.projectID = projectID
        self.name = name
        self.voiceModelID = voiceModelID
        self.language = language
    }
}

public struct VoiceLibraryRecord: Codable, Sendable, Equatable, Identifiable {
    public static let qwenSpeakerEmbeddingConditioningFormat = "qwen.speaker_embedding/v1"
    public static let qwenClonePromptConditioningFormat = "qwen.clone_prompt/v1"
    public static let tadaReferenceConditioningFormat = "tada.reference/v1"

    public let id: UUID
    public var label: String
    public var modelID: String
    public var runtimeModelID: String?
    public var backendVoiceID: String?
    public var sourceAssetName: String?
    public var referenceAudioAssetName: String?
    public var referenceTranscript: String?
    public var referenceDurationSeconds: Double?
    public var referenceSampleRate: Double?
    public var referenceChannelCount: Int?
    public var speakerEmbedding: Data?
    public var conditioningFormat: String?
    public var voiceKind: String?
    public var voicePrompt: String?
    public var createdAt: Date

    public var isClonedVoice: Bool {
        referenceAudioAssetName != nil
    }

    public var isModelDeclaredPreset: Bool {
        guard let backendVoiceID else { return false }
        return runtimeModelID == modelID
            && sourceAssetName == nil
            && referenceAudioAssetName == nil
            && referenceTranscript == nil
            && speakerEmbedding == nil
            && voicePrompt == nil
            && !backendVoiceID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    public var isMutable: Bool {
        !isModelDeclaredPreset
    }

    public init(
        id: UUID = UUID(),
        label: String,
        modelID: String,
        runtimeModelID: String? = nil,
        backendVoiceID: String? = nil,
        sourceAssetName: String? = nil,
        referenceAudioAssetName: String? = nil,
        referenceTranscript: String? = nil,
        referenceDurationSeconds: Double? = nil,
        referenceSampleRate: Double? = nil,
        referenceChannelCount: Int? = nil,
        speakerEmbedding: Data? = nil,
        conditioningFormat: String? = nil,
        voiceKind: String? = nil,
        voicePrompt: String? = nil,
        createdAt: Date = .now
    ) {
        self.id = id
        self.label = label
        self.modelID = modelID
        self.runtimeModelID = runtimeModelID
        self.backendVoiceID = backendVoiceID
        self.sourceAssetName = sourceAssetName
        self.referenceAudioAssetName = referenceAudioAssetName
        self.referenceTranscript = referenceTranscript
        self.referenceDurationSeconds = referenceDurationSeconds
        self.referenceSampleRate = referenceSampleRate
        self.referenceChannelCount = referenceChannelCount
        self.speakerEmbedding = speakerEmbedding
        self.conditioningFormat = conditioningFormat ?? Self.inferredConditioningFormat(
            modelID: modelID,
            runtimeModelID: runtimeModelID,
            speakerEmbedding: speakerEmbedding
        )
        self.voiceKind = voiceKind ?? Self.inferredVoiceKind(
            backendVoiceID: backendVoiceID,
            voicePrompt: voicePrompt,
            referenceAudioAssetName: referenceAudioAssetName,
            referenceTranscript: referenceTranscript,
            conditioningFormat: self.conditioningFormat,
            runtimeModelID: runtimeModelID,
            modelID: modelID
        )
        self.voicePrompt = voicePrompt
        self.createdAt = createdAt
    }

    private static func inferredConditioningFormat(
        modelID: String,
        runtimeModelID: String?,
        speakerEmbedding: Data?
    ) -> String? {
        guard speakerEmbedding != nil else {
            return nil
        }

        if isQwenTTSModelID(modelID) || runtimeModelID.map(isQwenTTSModelID) == true {
            return qwenSpeakerEmbeddingConditioningFormat
        }

        return nil
    }

    private static func inferredVoiceKind(
        backendVoiceID: String?,
        voicePrompt: String?,
        referenceAudioAssetName: String?,
        referenceTranscript: String?,
        conditioningFormat: String?,
        runtimeModelID: String?,
        modelID: String
    ) -> String? {
        let familyHint = runtimeModelID.map(inferFamilyFromModelID) ?? inferFamilyFromModelID(modelID)
        if let backendVoiceID,
           !backendVoiceID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            if familyHint == "qwen3_tts" {
                return "namedSpeaker"
            }
            return "preset"
        }

        switch conditioningFormat {
        case Self.qwenClonePromptConditioningFormat:
            return "clonePrompt"
        case Self.qwenSpeakerEmbeddingConditioningFormat:
            return "embeddingOnly"
        case Self.tadaReferenceConditioningFormat:
            return "tadaReference"
        default:
            break
        }

        if let voicePrompt,
           !voicePrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return "legacyPrompt"
        }

        if familyHint == "qwen3_tts",
           (referenceAudioAssetName != nil || referenceTranscript != nil) {
            return "embeddingOnly"
        }

        return nil
    }

    private static func isQwenTTSModelID(_ modelID: String) -> Bool {
        modelID
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .replacingOccurrences(of: "-", with: "_")
            .lowercased()
            .contains("qwen3_tts")
    }

    private static func inferFamilyFromModelID(_ modelID: String) -> String {
        let lower = modelID.lowercased().replacingOccurrences(of: "-", with: "_")
        if lower.contains("qwen3_tts") { return "qwen3_tts" }
        if lower.contains("tada") { return "tada_tts" }
        if lower.contains("voxtral") { return "voxtral_tts" }
        if lower.contains("chatterbox") { return "chatterbox_tts" }
        if lower.contains("orpheus") { return "orpheus_tts" }
        if lower.contains("soprano") { return "soprano" }
        return "unknown"
    }

    private static func isTADAModelID(_ modelID: String) -> Bool {
        let normalized = modelID
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .replacingOccurrences(of: "-", with: "_")
            .lowercased()
        return normalized.contains("tada") || normalized.hasPrefix("mlx_tada")
    }

    /// Infers the model family as a raw string (e.g. "qwen3_tts", "tada_tts").
    /// Compare against `ModelFamilyID` raw values in higher-level modules.
    public var inferredFamilyID: String {
        if let conditioningFormat {
            switch conditioningFormat {
            case Self.qwenSpeakerEmbeddingConditioningFormat:
                return "qwen3_tts"
            case Self.qwenClonePromptConditioningFormat:
                return "qwen3_tts"
            case Self.tadaReferenceConditioningFormat:
                return "tada_tts"
            default:
                break
            }
        }
        if Self.isQwenTTSModelID(modelID) || runtimeModelID.map(Self.isQwenTTSModelID) == true {
            return "qwen3_tts"
        }
        if Self.isTADAModelID(modelID) || runtimeModelID.map(Self.isTADAModelID) == true {
            return "tada_tts"
        }
        return Self.inferFamilyFromModelID(modelID)
    }

    public var effectiveConditioningFormat: String? {
        conditioningFormat
    }

    /// Returns the directory name used to store the conditioning bundle for this voice.
    /// For TADA voices this is `"<id>-tada"` under the voice library directory.
    /// Returns `nil` for voice types that don't use an asset-backed conditioning bundle.
    public var conditioningAssetName: String? {
        guard conditioningFormat == Self.tadaReferenceConditioningFormat else { return nil }
        return "\(id.uuidString)-tada"
    }

    public var familyDisplayName: String {
        switch inferredFamilyID {
        case "qwen3_tts": return "Qwen"
        case "tada_tts": return "TADA"
        case "voxtral_tts": return "Voxtral"
        case "chatterbox_tts": return "Chatterbox"
        case "orpheus_tts": return "Orpheus"
        case "soprano": return "Soprano"
        default: return inferredFamilyID
        }
    }

    public var typeDisplayName: String {
        switch effectiveVoiceKind {
        case "preset": return "Preset"
        case "namedSpeaker": return "Named Speaker"
        case "legacyPrompt": return "Designed"
        case "clonePrompt": return "Stable Narrator"
        case "embeddingOnly": return "Cloned"
        case "tadaReference": return "Cloned"
        default:
            if isDesignedVoice { return "Designed" }
            if isClonedVoice { return "Cloned" }
            return "Custom"
        }
    }

    public var isDesignedVoice: Bool {
        voicePrompt != nil && speakerEmbedding == nil && referenceAudioAssetName == nil
    }

    public var effectiveVoiceKind: String? {
        voiceKind ?? Self.inferredVoiceKind(
            backendVoiceID: backendVoiceID,
            voicePrompt: voicePrompt,
            referenceAudioAssetName: referenceAudioAssetName,
            referenceTranscript: referenceTranscript,
            conditioningFormat: conditioningFormat,
            runtimeModelID: runtimeModelID,
            modelID: modelID
        )
    }

    public var isLegacyExpressive: Bool {
        effectiveVoiceKind == "legacyPrompt"
    }
}

public struct RenderJobRecord: Codable, Sendable, Equatable, Identifiable {
    public let id: UUID
    public var projectID: UUID
    public var modelID: String
    public var chapterIDs: [UUID]
    public var outputFileName: String
    public var createdAt: Date
    public var updatedAt: Date
    public var state: String
    public var priority: Int
    public var progress: Double
    public var title: String?
    public var failureReason: String?
    public var queuePosition: Int
    public var synthesisOptions: RenderSynthesisOptions

    public init(
        id: UUID = UUID(),
        projectID: UUID,
        modelID: String = "",
        chapterIDs: [UUID],
        outputFileName: String,
        createdAt: Date = .now,
        updatedAt: Date = .now,
        state: String = "queued",
        priority: Int = 0,
        progress: Double = 0,
        title: String? = nil,
        failureReason: String? = nil,
        queuePosition: Int = 0,
        synthesisOptions: RenderSynthesisOptions = RenderSynthesisOptions()
    ) {
        self.id = id
        self.projectID = projectID
        self.modelID = modelID
        self.chapterIDs = chapterIDs
        self.outputFileName = outputFileName
        self.createdAt = createdAt
        self.updatedAt = updatedAt
        self.state = state
        self.priority = priority
        self.progress = progress
        self.title = title
        self.failureReason = failureReason
        self.queuePosition = queuePosition
        self.synthesisOptions = synthesisOptions
    }
}

public struct ExportRecord: Codable, Sendable, Equatable, Identifiable {
    public let id: UUID
    public var projectID: UUID
    public var fileName: String
    public var createdAt: Date
    public var checksum: String?

    public init(
        id: UUID = UUID(),
        projectID: UUID,
        fileName: String,
        createdAt: Date = .now,
        checksum: String? = nil
    ) {
        self.id = id
        self.projectID = projectID
        self.fileName = fileName
        self.createdAt = createdAt
        self.checksum = checksum
    }
}

public enum LegacySourceKind: String, CaseIterable, Codable, Sendable {
    case pythonWorkspace
    case pythonProject
    case voiceLibraryFolder
    case audioAssetBundle
    case unknown
}

public struct LegacyImportSource: Codable, Sendable, Equatable, Identifiable {
    public let id: UUID
    public let kind: LegacySourceKind
    public let rootPath: String
    public let discoveredAt: Date

    public init(
        id: UUID = UUID(),
        kind: LegacySourceKind,
        rootPath: String,
        discoveredAt: Date = .now
    ) {
        self.id = id
        self.kind = kind
        self.rootPath = rootPath
        self.discoveredAt = discoveredAt
    }
}

public struct MigrationStep: Codable, Sendable, Equatable, Identifiable {
    public let id: UUID
    public let title: String
    public let performedAt: Date
    public let succeeded: Bool
    public let message: String?

    public init(
        id: UUID = UUID(),
        title: String,
        performedAt: Date = .now,
        succeeded: Bool,
        message: String? = nil
    ) {
        self.id = id
        self.title = title
        self.performedAt = performedAt
        self.succeeded = succeeded
        self.message = message
    }
}

public struct MigrationPlan: Codable, Sendable, Equatable, Identifiable {
    public let id: UUID
    public let source: LegacyImportSource
    public var steps: [MigrationStep]
    public var notes: String?

    public init(
        id: UUID = UUID(),
        source: LegacyImportSource,
        steps: [MigrationStep] = [],
        notes: String? = nil
    ) {
        self.id = id
        self.source = source
        self.steps = steps
        self.notes = notes
    }
}

public struct ImportOutcome: Codable, Sendable, Equatable, Identifiable {
    public let id: UUID
    public let source: LegacyImportSource
    public let importedProjectCount: Int
    public let importedVoiceCount: Int
    public let warnings: [String]

    public init(
        id: UUID = UUID(),
        source: LegacyImportSource,
        importedProjectCount: Int,
        importedVoiceCount: Int,
        warnings: [String] = []
    ) {
        self.id = id
        self.source = source
        self.importedProjectCount = importedProjectCount
        self.importedVoiceCount = importedVoiceCount
        self.warnings = warnings
    }
}

public protocol MigrationJournal: Sendable {
    func record(_ step: MigrationStep) async
    func steps() async -> [MigrationStep]
}

public actor ProjectStore: ProjectStoring {
    private let paths: ValarAppPaths
    private var projects: [UUID: ProjectRecord]
    private var chapters: [UUID: [ChapterRecord]]
    private var renders: [UUID: [RenderJobRecord]]
    private var exports: [UUID: [ExportRecord]]
    private var speakers: [UUID: [ProjectSpeakerRecord]]
    private var bundleURLs: [UUID: URL]

    public init(paths: ValarAppPaths = ValarAppPaths(), projects: [ProjectRecord] = []) {
        self.paths = paths
        self.projects = Dictionary(uniqueKeysWithValues: projects.map { ($0.id, $0) })
        self.chapters = [:]
        self.renders = [:]
        self.exports = [:]
        self.speakers = [:]
        self.bundleURLs = [:]
    }

    public func create(title: String, notes: String? = nil) async throws -> ProjectRecord {
        let project = ProjectRecord(title: title, notes: notes)
        projects[project.id] = project
        chapters[project.id] = []
        renders[project.id] = []
        exports[project.id] = []
        speakers[project.id] = []
        return project
    }

    public func update(_ project: ProjectRecord) async {
        projects[project.id] = project
    }

    public func addChapter(_ chapter: ChapterRecord) async {
        var records = chapters[chapter.projectID, default: []]
        if let existingIndex = records.firstIndex(where: { $0.id == chapter.id }) {
            records[existingIndex] = chapter
        } else {
            records.append(chapter)
        }
        records.sort { $0.index < $1.index }
        chapters[chapter.projectID] = records
    }

    public func updateChapter(_ chapter: ChapterRecord) async {
        guard var projectChapters = chapters[chapter.projectID],
              let index = projectChapters.firstIndex(where: { $0.id == chapter.id }) else {
            return
        }
        projectChapters[index] = chapter
        projectChapters.sort { $0.index < $1.index }
        chapters[chapter.projectID] = projectChapters
    }

    public func removeChapter(_ id: UUID, from projectID: UUID) async {
        chapters[projectID]?.removeAll { $0.id == id }
    }

    public func addRenderJob(_ job: RenderJobRecord) async {
        var records = renders[job.projectID, default: []]
        if let existingIndex = records.firstIndex(where: { $0.id == job.id }) {
            records[existingIndex] = job
        } else {
            records.append(job)
        }
        records.sort { $0.createdAt < $1.createdAt }
        renders[job.projectID] = records
    }

    public func addExport(_ export: ExportRecord) async {
        var records = exports[export.projectID, default: []]
        if let existingIndex = records.firstIndex(where: { $0.id == export.id }) {
            records[existingIndex] = export
        } else {
            records.append(export)
        }
        records.sort { $0.createdAt < $1.createdAt }
        exports[export.projectID] = records
    }

    public func addSpeaker(_ speaker: ProjectSpeakerRecord) async {
        var records = speakers[speaker.projectID, default: []]
        if let existingIndex = records.firstIndex(where: { $0.id == speaker.id }) {
            records[existingIndex] = speaker
        } else {
            records.append(speaker)
        }
        speakers[speaker.projectID] = records
    }

    public func removeSpeaker(_ id: UUID, from projectID: UUID) async {
        speakers[projectID]?.removeAll { $0.id == id }
    }

    public func restore(
        project: ProjectRecord,
        chapters: [ChapterRecord],
        renderJobs: [RenderJobRecord],
        exports: [ExportRecord],
        speakers: [ProjectSpeakerRecord]
    ) async {
        projects[project.id] = project
        self.chapters[project.id] = chapters.sorted { $0.index < $1.index }
        self.renders[project.id] = renderJobs.sorted { $0.createdAt < $1.createdAt }
        self.exports[project.id] = exports.sorted { $0.createdAt < $1.createdAt }
        self.speakers[project.id] = speakers
    }

    public func allProjects() async -> [ProjectRecord] {
        projects.values.sorted { $0.updatedAt > $1.updatedAt }
    }

    public func chapters(for projectID: UUID) async -> [ChapterRecord] {
        chapters[projectID, default: []]
    }

    public func renderJobs(for projectID: UUID) async -> [RenderJobRecord] {
        renders[projectID, default: []]
    }

    public func exports(for projectID: UUID) async -> [ExportRecord] {
        exports[projectID, default: []]
    }

    public func speakers(for projectID: UUID) async -> [ProjectSpeakerRecord] {
        speakers[projectID, default: []]
    }

    public func bundleURL(for projectID: UUID) -> URL? {
        guard projects[projectID] != nil else { return nil }
        return bundleURLs[projectID]
    }

    public func bundleLocation(for projectID: UUID) async -> ValarProjectBundleLocation? {
        guard let project = projects[projectID],
              let bundleURL = bundleURLs[projectID] else {
            return nil
        }
        return ValarProjectBundleLocation(projectID: projectID, title: project.title, bundleURL: bundleURL)
    }

    public func updateBundleURL(_ bundleURL: URL?, for projectID: UUID) async {
        guard projects[projectID] != nil else { return }
        bundleURLs[projectID] = bundleURL
    }

    public func remove(id: UUID) async {
        projects[id] = nil
        chapters[id] = nil
        renders[id] = nil
        exports[id] = nil
        speakers[id] = nil
        bundleURLs[id] = nil
    }
}

public actor VoiceLibraryStore: VoiceLibraryStoring {
    private var voices: [UUID: VoiceLibraryRecord]

    public init(records: [VoiceLibraryRecord] = []) {
        self.voices = Dictionary(uniqueKeysWithValues: records.map { ($0.id, $0) })
    }

    public func save(_ voice: VoiceLibraryRecord) async throws -> VoiceLibraryRecord {
        voices[voice.id] = voice
        return voice
    }

    public func list() async -> [VoiceLibraryRecord] {
        voices.values.sorted { $0.createdAt < $1.createdAt }
    }

    public func delete(_ id: UUID) async throws {
        voices[id] = nil
    }
}

public actor MigrationLedger: MigrationJournal {
    private(set) public var imports: [MigrationStep]

    public init(imports: [MigrationStep] = []) {
        self.imports = imports
    }

    public func record(_ step: MigrationStep) {
        imports.append(step)
    }

    public func record(sourcePath: String, note: String) -> MigrationStep {
        let step = MigrationStep(title: sourcePath, succeeded: true, message: note)
        imports.append(step)
        return step
    }

    public func steps() -> [MigrationStep] {
        imports
    }
}

public actor ModelPackRegistry: ModelPackManaging {
    private let paths: ValarAppPaths
    private var manifests: [String: ModelPackManifest]
    private var records: [String: InstalledModelRecord]
    private var installedReceipts: [String: ModelInstallReceipt]
    private var catalog: [String: SupportedModelCatalogRecord]
    private var ledger: [ModelInstallLedgerEntry]

    public init(
        paths: ValarAppPaths = ValarAppPaths(),
        manifests: [ModelPackManifest] = [],
        records: [InstalledModelRecord] = [],
        receipts: [ModelInstallReceipt] = []
    ) {
        self.paths = paths
        self.manifests = Dictionary(uniqueKeysWithValues: manifests.map { ($0.modelID, $0) })
        self.records = Dictionary(uniqueKeysWithValues: records.map { ($0.modelID, $0) })
        self.installedReceipts = Dictionary(uniqueKeysWithValues: receipts.map { ($0.modelID, $0) })
        self.catalog = Dictionary(uniqueKeysWithValues: manifests.map {
            (
                $0.modelID,
                SupportedModelCatalogRecord(
                    id: $0.id,
                    familyID: $0.familyID,
                    modelID: $0.modelID,
                    displayName: $0.displayName,
                    providerName: "Valar",
                    providerURL: nil,
                    installHint: $0.notes,
                    sourceKind: .localFile,
                    isRecommended: $0.isRecommended
                )
            )
        })
        for receipt in receipts {
            catalog[receipt.modelID] = SupportedModelCatalogRecord(
                id: receipt.id,
                familyID: receipt.familyID,
                modelID: receipt.modelID,
                displayName: receipt.modelID,
                providerName: "Valar",
                providerURL: nil,
                installHint: receipt.notes,
                sourceKind: receipt.sourceKind,
                isRecommended: false
            )
        }
        self.ledger = []
    }

    public func registerSupported(_ record: SupportedModelCatalogRecord) async {
        catalog[record.modelID] = record
        manifests[record.modelID] = manifests[record.modelID]
            ?? ModelPackManifest(
                id: record.id,
                familyID: record.familyID,
                modelID: record.modelID,
                displayName: record.displayName,
                isRecommended: record.isRecommended,
                notes: record.installHint
            )
    }

    public func install(
        manifest: ModelPackManifest,
        sourceKind: ModelPackSourceKind,
        sourceLocation: String,
        notes: String? = nil
    ) async throws -> ModelInstallReceipt {
        let packDirectory = try paths.modelPackDirectory(familyID: manifest.familyID, modelID: manifest.modelID)
        let manifestURL = try paths.modelPackManifestURL(familyID: manifest.familyID, modelID: manifest.modelID)
        let receipt = ModelInstallReceipt(
            modelID: manifest.modelID,
            familyID: manifest.familyID,
            sourceKind: sourceKind,
            sourceLocation: sourceLocation,
            installedModelPath: packDirectory.path,
            manifestPath: manifestURL.path,
            checksum: manifest.artifactSpecs.first?.checksum,
            artifactCount: manifest.artifactSpecs.count,
            notes: notes
        )
        let record = InstalledModelRecord(
            id: receipt.id,
            familyID: manifest.familyID,
            modelID: manifest.modelID,
            displayName: manifest.displayName,
            installedPath: packDirectory.path,
            manifestPath: manifestURL.path,
            artifactCount: manifest.artifactSpecs.count,
            checksum: receipt.checksum,
            sourceKind: sourceKind,
            isEnabled: true
        )
        manifests[manifest.modelID] = manifest
        records[manifest.modelID] = record
        installedReceipts[manifest.modelID] = receipt
        catalog[manifest.modelID] = SupportedModelCatalogRecord(
            id: receipt.id,
            familyID: manifest.familyID,
            modelID: manifest.modelID,
            displayName: manifest.displayName,
            providerName: "Valar",
            providerURL: nil,
            installHint: notes,
            sourceKind: sourceKind,
            isRecommended: manifest.isRecommended
        )
        ledger.append(
            ModelInstallLedgerEntry(
                receiptID: receipt.id,
                sourceKind: sourceKind,
                sourceLocation: sourceLocation,
                succeeded: true,
                message: notes ?? "Installed \(manifest.displayName)"
            )
        )
        return receipt
    }

    public func supportedModels() async -> [SupportedModelCatalogRecord] {
        catalog.values
            .sorted { $0.displayName < $1.displayName }
    }

    public func supportedModel(for modelID: String) async -> SupportedModelCatalogRecord? {
        catalog[modelID]
    }

    public func manifest(for modelID: String) async -> ModelPackManifest? {
        manifests[modelID]
    }

    public func installedRecord(for modelID: String) async -> InstalledModelRecord? {
        records[modelID]
    }

    public func receipt(for modelID: String) async -> ModelInstallReceipt? {
        installedReceipts[modelID]
    }

    public func receipts() async -> [ModelInstallReceipt] {
        installedReceipts.values.sorted { $0.installDate < $1.installDate }
    }

    public func uninstall(modelID: String) async -> InstalledModelRecord? {
        let record = records.removeValue(forKey: modelID)
        manifests.removeValue(forKey: modelID)
        installedReceipts.removeValue(forKey: modelID)
        return record
    }

    public func ledgerEntries() async -> [ModelInstallLedgerEntry] {
        ledger.sorted { $0.recordedAt < $1.recordedAt }
    }
}
