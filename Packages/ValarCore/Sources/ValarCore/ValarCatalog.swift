import CryptoKit
import Darwin
import Foundation
import ValarModelKit
import ValarPersistence

public protocol CapabilityRegistryManaging: Sendable {
    func register(_ descriptor: ModelDescriptor) async
    func unregister(_ identifier: ModelIdentifier) async
    func descriptor(for identifier: ModelIdentifier) async -> ModelDescriptor?
    func capabilities(for identifier: ModelIdentifier) async -> Set<ModelCapability>
    func models(supporting capability: ModelCapability) async -> [ModelDescriptor]
    func models(inFamily familyID: ModelFamilyID) async -> [ModelDescriptor]
    func allModels() async -> [ModelDescriptor]
}

public actor CapabilityRegistry: CapabilityRegistryManaging {
    private var descriptors: [ModelIdentifier: ModelDescriptor]
    private var capabilityIndex: [ModelCapability: Set<ModelIdentifier>]
    private var familyIndex: [ModelFamilyID: Set<ModelIdentifier>]

    public init(descriptors: [ModelDescriptor] = []) {
        self.descriptors = Dictionary(uniqueKeysWithValues: descriptors.map { ($0.id, $0) })
        self.capabilityIndex = [:]
        self.familyIndex = [:]

        for descriptor in descriptors {
            for capability in descriptor.capabilities {
                capabilityIndex[capability, default: []].insert(descriptor.id)
            }
            familyIndex[descriptor.familyID, default: []].insert(descriptor.id)
        }
    }

    public func register(_ descriptor: ModelDescriptor) {
        unregisterImmediately(descriptor.id)
        descriptors[descriptor.id] = descriptor
        for capability in descriptor.capabilities {
            capabilityIndex[capability, default: []].insert(descriptor.id)
        }
        familyIndex[descriptor.familyID, default: []].insert(descriptor.id)
    }

    public func unregister(_ identifier: ModelIdentifier) {
        unregisterImmediately(identifier)
    }

    public func descriptor(for identifier: ModelIdentifier) -> ModelDescriptor? {
        descriptors[identifier]
    }

    public func capabilities(for identifier: ModelIdentifier) -> Set<ModelCapability> {
        descriptors[identifier]?.capabilities ?? []
    }

    public func models(supporting capability: ModelCapability) -> [ModelDescriptor] {
        let identifiers = capabilityIndex[capability] ?? []
        return identifiers
            .compactMap { descriptors[$0] }
            .sorted { $0.displayName < $1.displayName }
    }

    public func models(inFamily familyID: ModelFamilyID) -> [ModelDescriptor] {
        let identifiers = familyIndex[familyID] ?? []
        return identifiers
            .compactMap { descriptors[$0] }
            .sorted { $0.displayName < $1.displayName }
    }

    public func allModels() -> [ModelDescriptor] {
        descriptors.values.sorted { $0.displayName < $1.displayName }
    }

    private func unregisterImmediately(_ identifier: ModelIdentifier) {
        guard let existing = descriptors.removeValue(forKey: identifier) else { return }

        for capability in existing.capabilities {
            capabilityIndex[capability]?.remove(identifier)
            if capabilityIndex[capability]?.isEmpty == true {
                capabilityIndex[capability] = nil
            }
        }

        familyIndex[existing.familyID]?.remove(identifier)
        if familyIndex[existing.familyID]?.isEmpty == true {
            familyIndex[existing.familyID] = nil
        }
    }
}

public protocol SupportedCatalogSourcing: Sendable {
    func supportedEntries() async throws -> [SupportedModelCatalogEntry]
}

public struct StaticSupportedCatalogSource: SupportedCatalogSourcing {
    public let records: [SupportedModelCatalogEntry]

    public init(records: [SupportedModelCatalogEntry]) {
        self.records = records
    }

    public func supportedEntries() async throws -> [SupportedModelCatalogEntry] {
        records
    }
}

public enum SupportedCatalogSource {
    public static func curated() -> StaticSupportedCatalogSource {
        StaticSupportedCatalogSource(
            records: SupportedModelCatalog.allSupportedEntries
        )
    }

    public static func qwenFirst() -> StaticSupportedCatalogSource {
        curated()
    }
}

public struct CatalogVisibilityPolicy: Sendable, Equatable {
    public static let nonCommercialEnvVarName = "VALARTTS_ENABLE_NONCOMMERCIAL_MODELS"

    public let allowsNonCommercialModels: Bool

    public init(allowsNonCommercialModels: Bool) {
        self.allowsNonCommercialModels = allowsNonCommercialModels
    }

    public static func currentProcess() -> CatalogVisibilityPolicy {
        let enabled: Bool
        if let raw = getenv(nonCommercialEnvVarName) {
            enabled = Self.parseBooleanFlag(String(cString: raw))
        } else {
            enabled = false
        }
        return CatalogVisibilityPolicy(allowsNonCommercialModels: enabled)
    }

    public func allows(_ entry: SupportedModelCatalogEntry) -> Bool {
        allowsNonCommercialModels || !entry.manifest.licenses.contains(where: \.isNonCommercial)
    }

    public func hiddenReason(for identifier: ModelIdentifier) -> String? {
        guard let entry = SupportedModelCatalog.entry(for: identifier), allows(entry) == false else {
            return nil
        }

        return "Model '\(identifier.rawValue)' is hidden by default because it is licensed for non-commercial use only. Set \(Self.nonCommercialEnvVarName)=1 to enable non-commercial models intentionally."
    }

    public static func parseBooleanFlag(_ rawValue: String) -> Bool {
        switch rawValue.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "1", "true", "yes", "on":
            return true
        default:
            return false
        }
    }
}

public enum CatalogInstallState: String, Codable, Sendable {
    case supported
    case cached
    case installed
}

public enum CatalogInstallPathStatus: String, Codable, Sendable {
    case valid
    case missingInstalledPath
    case missingManifest
    case missingArtifacts

    public var isValid: Bool {
        self == .valid
    }
}

public struct CatalogModel: Codable, Sendable, Equatable, Identifiable {
    public let id: ModelIdentifier
    public let descriptor: ModelDescriptor
    public let familyID: ModelFamilyID
    public let installState: CatalogInstallState
    public let providerName: String
    public let providerURL: URL?
    public let sourceKind: ModelPackSourceKind?
    public let isRecommended: Bool
    public let manifestPath: String?
    public let installedPath: String?
    public let artifactCount: Int
    public let supportedBackends: [BackendKind]
    public let licenseName: String?
    public let licenseURL: URL?
    public let supportTier: ModelSupportTier
    public let releaseEligible: Bool
    public let qualityTierByLanguage: [String: ModelLanguageSupportTier]
    public let distributionTier: ModelDistributionTier
    public let notes: String?
    public let cachedOnDisk: Bool
    public let installPathStatus: CatalogInstallPathStatus?

    public init(
        id: ModelIdentifier,
        descriptor: ModelDescriptor,
        familyID: ModelFamilyID,
        installState: CatalogInstallState,
        providerName: String,
        providerURL: URL?,
        sourceKind: ModelPackSourceKind?,
        isRecommended: Bool,
        manifestPath: String?,
        installedPath: String?,
        artifactCount: Int,
        supportedBackends: [BackendKind],
        licenseName: String? = nil,
        licenseURL: URL? = nil,
        supportTier: ModelSupportTier? = nil,
        releaseEligible: Bool? = nil,
        qualityTierByLanguage: [String: ModelLanguageSupportTier]? = nil,
        distributionTier: ModelDistributionTier = .optionalInstall,
        notes: String?,
        cachedOnDisk: Bool = false,
        installPathStatus: CatalogInstallPathStatus? = nil
    ) {
        self.id = id
        self.descriptor = descriptor
        self.familyID = familyID
        self.installState = installState
        self.providerName = providerName
        self.providerURL = providerURL
        self.sourceKind = sourceKind
        self.isRecommended = isRecommended
        self.manifestPath = manifestPath
        self.installedPath = installedPath
        self.artifactCount = artifactCount
        self.supportedBackends = supportedBackends
        self.licenseName = licenseName
        self.licenseURL = licenseURL
        self.supportTier = supportTier ?? descriptor.supportTier
        self.releaseEligible = releaseEligible ?? descriptor.releaseEligible
        self.qualityTierByLanguage = qualityTierByLanguage ?? descriptor.qualityTierByLanguage
        self.distributionTier = distributionTier
        self.notes = notes
        self.cachedOnDisk = cachedOnDisk
        self.installPathStatus = installPathStatus
    }
}

public actor ModelCatalog {
    private let supportedSource: any SupportedCatalogSourcing
    private let catalogStore: (any ModelCatalogStore)?
    private let packStore: (any ModelPackStore)?
    private let capabilityRegistry: (any CapabilityRegistryManaging)?
    private let visibilityPolicyProvider: @Sendable () -> CatalogVisibilityPolicy
    private let hfCacheRoot: URL?
    private var cachedModels: [ModelIdentifier: CatalogModel]

    public init(
        supportedSource: any SupportedCatalogSourcing = SupportedCatalogSource.curated(),
        catalogStore: (any ModelCatalogStore)? = nil,
        packStore: (any ModelPackStore)? = nil,
        capabilityRegistry: (any CapabilityRegistryManaging)? = nil,
        hfCacheRoot: URL? = nil,
        visibilityPolicyProvider: @escaping @Sendable () -> CatalogVisibilityPolicy = CatalogVisibilityPolicy.currentProcess
    ) {
        self.supportedSource = supportedSource
        self.catalogStore = catalogStore
        self.packStore = packStore
        self.capabilityRegistry = capabilityRegistry
        self.visibilityPolicyProvider = visibilityPolicyProvider
        self.hfCacheRoot = hfCacheRoot
        self.cachedModels = [:]
    }

    public func refresh() async throws -> [CatalogModel] {
        var supported = try await supportedSource.supportedEntries()
        if let catalogStore {
            let persisted = try await catalogStore.supportedModels()
            supported.append(contentsOf: persisted.map(Self.makeSupportedEntry(from:)))
        }

        var uniqueSupported: [ModelIdentifier: SupportedModelCatalogEntry] = [:]
        for entry in supported {
            if uniqueSupported[entry.id] == nil {
                uniqueSupported[entry.id] = entry
            }
        }
        let visibilityPolicy = visibilityPolicyProvider()
        var models: [ModelIdentifier: CatalogModel] = [:]

        for entry in uniqueSupported.values where visibilityPolicy.allows(entry) {
            let installedRecord = try await packStore?.installedRecord(for: entry.id.rawValue)
            let supportedPersistenceManifest = Self.makePersistenceManifest(from: entry.manifest)
            let installPathStatus = installedRecord.map {
                Self.installPathStatus($0, manifest: supportedPersistenceManifest)
            }
            let materializedInstalledRecord = installPathStatus?.isValid == true ? installedRecord : nil
            let resolvedManifest = entry.manifest
            let descriptor = ModelDescriptor(manifest: resolvedManifest)

            let cachedOnDisk: Bool
            if materializedInstalledRecord == nil {
                cachedOnDisk = Self.hasCachedArtifacts(for: entry, hfCacheRoot: hfCacheRoot)
            } else {
                cachedOnDisk = false
            }
            let installState: CatalogInstallState = materializedInstalledRecord != nil ? .installed
                : cachedOnDisk ? .cached
                : .supported

            let catalogModel = CatalogModel(
                id: descriptor.id,
                descriptor: descriptor,
                familyID: descriptor.familyID,
                installState: installState,
                providerName: entry.remoteURL?.host() ?? "Valar",
                providerURL: entry.remoteURL,
                sourceKind: materializedInstalledRecord?.sourceKind,
                isRecommended: entry.isRecommended,
                manifestPath: materializedInstalledRecord?.manifestPath,
                installedPath: materializedInstalledRecord?.installedPath,
                artifactCount: materializedInstalledRecord?.artifactCount ?? resolvedManifest.artifacts.count,
                supportedBackends: resolvedManifest.supportedBackends.map(\.backendKind),
                licenseName: resolvedManifest.licenses.first?.name,
                licenseURL: resolvedManifest.licenses.first?.sourceURL,
                supportTier: resolvedManifest.supportTier,
                releaseEligible: resolvedManifest.releaseEligible,
                qualityTierByLanguage: resolvedManifest.qualityTierByLanguage,
                distributionTier: entry.distributionTier,
                notes: resolvedManifest.notes,
                cachedOnDisk: cachedOnDisk,
                installPathStatus: installPathStatus
            )
            models[catalogModel.id] = catalogModel

            if let capabilityRegistry {
                await capabilityRegistry.register(descriptor)
            }
        }

        cachedModels = models
        return ordered(models.values)
    }

    public func supportedModels() async throws -> [CatalogModel] {
        if cachedModels.isEmpty {
            return try await refresh()
        }
        return ordered(cachedModels.values)
    }

    public func installedModels() async throws -> [CatalogModel] {
        try await supportedModels().filter { $0.installState == .installed }
    }

    public func staleInstalledModels() async throws -> [CatalogModel] {
        try await supportedModels().filter { model in
            guard let status = model.installPathStatus else { return false }
            return !status.isValid
        }
    }

    public func recommendedModels() async throws -> [CatalogModel] {
        try await supportedModels().filter(\.isRecommended)
    }

    public func model(for identifier: ModelIdentifier) async throws -> CatalogModel? {
        if cachedModels.isEmpty {
            _ = try await refresh()
        }
        return cachedModels[identifier]
    }

    public func installationManifest(for identifier: ModelIdentifier) async throws -> ValarPersistence.ModelPackManifest? {
        if let entry = try await supportedSource.supportedEntries().first(where: { $0.id == identifier }) {
            return Self.makePersistenceManifest(from: entry.manifest)
        }

        return try await packStore?.manifest(for: identifier.rawValue)
    }

    public func models(supporting capability: ModelCapability) async throws -> [CatalogModel] {
        try await supportedModels().filter { $0.descriptor.capabilities.contains(capability) }
    }

    private func ordered<S: Sequence>(_ models: S) -> [CatalogModel] where S.Element == CatalogModel {
        // Sort order: installed (0) → cached (1) → supported (2)
        func sortOrder(_ state: CatalogInstallState) -> Int {
            switch state {
            case .installed: return 0
            case .cached: return 1
            case .supported: return 2
            }
        }
        return models.sorted { lhs, rhs in
            let lhsStale = lhs.installPathStatus.map { !$0.isValid } ?? false
            let rhsStale = rhs.installPathStatus.map { !$0.isValid } ?? false
            if lhsStale != rhsStale {
                return !lhsStale && rhsStale
            }
            if lhs.installState != rhs.installState {
                return sortOrder(lhs.installState) < sortOrder(rhs.installState)
            }
            if lhs.isRecommended != rhs.isRecommended {
                return lhs.isRecommended && !rhs.isRecommended
            }
            return lhs.descriptor.displayName < rhs.descriptor.displayName
        }
    }

    /// Returns the expected mlx-audio HuggingFace cache directory for a catalog entry,
    /// or nil if the entry has no remote provider URL.
    private static func hfMLXAudioCachePath(
        for entry: SupportedModelCatalogEntry,
        fileManager: FileManager = .default,
        hfCacheRoot: URL? = nil
    ) -> String? {
        guard entry.remoteURL != nil else { return nil }
        let dirName = entry.id.rawValue.replacingOccurrences(of: "/", with: "_")
        return resolveHFHubCacheRoot(fileManager: fileManager, hfCacheRoot: hfCacheRoot)
            .appendingPathComponent("mlx-audio", isDirectory: true)
            .appendingPathComponent(dirName)
            .path
    }

    private static func hasCachedArtifacts(
        for entry: SupportedModelCatalogEntry,
        fileManager: FileManager = .default,
        hfCacheRoot: URL? = nil
    ) -> Bool {
        let hubArtifactURL: (String, String) -> URL? = { modelID, relativePath in
            guard let snapshotDirectory = hfHubSnapshotDirectory(
                modelID: modelID,
                fileManager: fileManager,
                hfCacheRoot: hfCacheRoot
            ) else {
                return nil
            }
            let candidate = snapshotDirectory.appendingPathComponent(relativePath, isDirectory: false)
            return fileManager.fileExists(atPath: candidate.path) ? candidate : nil
        }
        let requiredArtifacts = entry.manifest.artifacts
            .filter { $0.required && !$0.relativePath.hasSuffix("/") }
        if !requiredArtifacts.isEmpty,
           requiredArtifacts.allSatisfy({ artifact in
               if let direct = hubArtifactURL(entry.id.rawValue, artifact.relativePath) {
                   return nonEmptyFileExists(at: direct, fileManager: fileManager)
               }

               let fallbackModelID: String?
               switch entry.manifest.familyID {
               case .vibevoiceRealtimeTTS
                   where Set(["tokenizer.json", "tokenizer_config.json"]).contains(artifact.relativePath):
                   fallbackModelID = VibeVoiceCatalog.tokenizerSourceModelIdentifier.rawValue
               default:
                   fallbackModelID = nil
               }

               guard let fallbackModelID,
                     let fallback = hubArtifactURL(fallbackModelID, artifact.relativePath) else {
                   return false
               }

               return nonEmptyFileExists(at: fallback, fileManager: fileManager)
           }) {
            return true
        }

        if let cachePath = hfMLXAudioCachePath(for: entry, fileManager: fileManager, hfCacheRoot: hfCacheRoot),
           artifactFilesExist(
               relativePaths: requiredArtifactRelativePaths(for: entry.manifest),
               under: URL(fileURLWithPath: cachePath, isDirectory: true),
               fileManager: fileManager,
               allowBasenameFallback: true
           ) {
            return true
        }

        return false
    }

    private static func installPathStatus(
        _ record: InstalledModelRecord,
        manifest: ValarPersistence.ModelPackManifest,
        fileManager: FileManager = .default
    ) -> CatalogInstallPathStatus {
        let installedRoot = URL(fileURLWithPath: record.installedPath, isDirectory: true)
        let manifestURL = URL(fileURLWithPath: record.manifestPath, isDirectory: false)
        guard fileManager.fileExists(atPath: installedRoot.path) else {
            return .missingInstalledPath
        }
        guard fileManager.fileExists(atPath: manifestURL.path) else {
            return .missingManifest
        }
        guard artifactFilesExist(
            relativePaths: requiredArtifactRelativePaths(for: manifest),
            under: installedRoot,
            fileManager: fileManager
        ) else {
            return .missingArtifacts
        }
        return .valid
    }

    private static func materializedInstalledRecord(
        _ record: InstalledModelRecord,
        manifest: ValarPersistence.ModelPackManifest,
        fileManager: FileManager = .default
    ) -> InstalledModelRecord? {
        guard installPathStatus(record, manifest: manifest, fileManager: fileManager).isValid else {
            return nil
        }

        return record
    }

    private static func requiredArtifactRelativePaths(for manifest: ValarModelKit.ModelPackManifest) -> [String] {
        manifest.artifacts
            .filter { $0.required && !$0.relativePath.hasSuffix("/") }
            .map(\.relativePath)
    }

    private static func requiredArtifactRelativePaths(for manifest: ValarPersistence.ModelPackManifest) -> [String] {
        manifest.artifactSpecs
            .filter { $0.required && !$0.relativePath.hasSuffix("/") }
            .map(\.relativePath)
    }

    private static func artifactFilesExist(
        relativePaths: [String],
        under root: URL,
        fileManager: FileManager = .default,
        allowBasenameFallback: Bool = false
    ) -> Bool {
        guard fileManager.fileExists(atPath: root.path) else {
            return false
        }

        return relativePaths.allSatisfy { relativePath in
            let artifactURL = root.appendingPathComponent(relativePath, isDirectory: false)
            if nonEmptyFileExists(at: artifactURL, fileManager: fileManager) {
                return true
            }

            guard allowBasenameFallback else {
                return false
            }

            let basenameURL = root.appendingPathComponent(URL(fileURLWithPath: relativePath).lastPathComponent, isDirectory: false)
            return nonEmptyFileExists(at: basenameURL, fileManager: fileManager)
        }
    }

    private static func nonEmptyFileExists(at url: URL, fileManager: FileManager = .default) -> Bool {
        guard fileManager.fileExists(atPath: url.path) else {
            return false
        }

        let size = (try? url.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0
        return size > 0
    }

    private static func hfHubSnapshotDirectory(
        modelID: String,
        fileManager: FileManager = .default,
        hfCacheRoot: URL? = nil
    ) -> URL? {
        let cacheBase = resolveHFHubCacheRoot(fileManager: fileManager, hfCacheRoot: hfCacheRoot)
        let repoDirectoryName = "models--" + modelID.replacingOccurrences(of: "/", with: "--")
        let repoDirectory = cacheBase.appendingPathComponent(repoDirectoryName, isDirectory: true)
        let snapshotsDirectory = repoDirectory.appendingPathComponent("snapshots", isDirectory: true)
        guard fileManager.fileExists(atPath: snapshotsDirectory.path) else {
            return nil
        }

        let refsMain = repoDirectory.appendingPathComponent("refs/main", isDirectory: false)
        if let revision = try? String(contentsOf: refsMain, encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines),
           !revision.isEmpty
        {
            let candidate = snapshotsDirectory.appendingPathComponent(revision, isDirectory: true)
            if fileManager.fileExists(atPath: candidate.path) {
                return candidate
            }
        }

        let snapshots = (try? fileManager.contentsOfDirectory(
            at: snapshotsDirectory,
            includingPropertiesForKeys: [.contentModificationDateKey],
            options: [.skipsHiddenFiles]
        )) ?? []
        return snapshots.max(by: {
            let lhs = (try? $0.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
            let rhs = (try? $1.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
            return lhs < rhs
        })
    }

    private static func resolveHFHubCacheRoot(
        fileManager: FileManager = .default,
        hfCacheRoot: URL? = nil
    ) -> URL {
        if let hfCacheRoot {
            return hfCacheRoot
        }

        let environment = ProcessInfo.processInfo.environment
        if let explicit = environment["HF_HUB_CACHE"]?.trimmingCharacters(in: .whitespacesAndNewlines),
           !explicit.isEmpty {
            return URL(fileURLWithPath: explicit, isDirectory: true)
        }
        if let explicit = environment["HUGGINGFACE_HUB_CACHE"]?.trimmingCharacters(in: .whitespacesAndNewlines),
           !explicit.isEmpty {
            return URL(fileURLWithPath: explicit, isDirectory: true)
        }
        if let hfHome = environment["HF_HOME"]?.trimmingCharacters(in: .whitespacesAndNewlines),
           !hfHome.isEmpty {
            return URL(fileURLWithPath: hfHome, isDirectory: true)
                .appendingPathComponent("hub", isDirectory: true)
        }

        return fileManager.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/hub", isDirectory: true)
    }

    public static func makeSupportedRecord(from entry: SupportedModelCatalogEntry) -> SupportedModelCatalogRecord {
        SupportedModelCatalogRecord(
            id: entry.id.rawValue,
            familyID: entry.manifest.familyID.rawValue,
            modelID: entry.id.rawValue,
            displayName: entry.manifest.displayName,
            providerName: entry.remoteURL?.host() ?? "Valar",
            providerURL: entry.remoteURL?.absoluteString,
            installHint: entry.requiresManualDownload
                ? "Import a .valarmodel pack or fetch the curated provider package."
                : "Curated first-party Valar core model.",
            sourceKind: entry.remoteURL == nil ? .localFile : .remoteURL,
            isRecommended: entry.isRecommended
        )
    }

    public static func makePersistenceManifest(from manifest: ValarModelKit.ModelPackManifest) -> ValarPersistence.ModelPackManifest {
        ValarPersistence.ModelPackManifest(
            id: manifest.id.rawValue,
            schemaVersion: manifest.schemaVersion,
            familyID: manifest.familyID.rawValue,
            modelID: manifest.id.rawValue,
            displayName: manifest.displayName,
            capabilities: manifest.capabilities.map(\.rawValue).sorted(),
            backendKinds: manifest.supportedBackends.map(\.backendKind.rawValue),
            tokenizerType: manifest.tokenizer?.kind,
            sampleRate: manifest.audio?.defaultSampleRate,
            artifactSpecs: manifest.artifacts.map {
                ModelPackArtifact(
                    id: $0.id,
                    kind: $0.role.rawValue,
                    relativePath: $0.relativePath,
                    checksum: $0.sha256,
                    byteCount: $0.sizeBytes,
                    required: $0.required
                )
            },
            licenseName: manifest.licenses.first?.name,
            licenseURL: manifest.licenses.first?.sourceURL?.absoluteString,
            minimumAppVersion: manifest.minimumAppVersion,
            notes: manifest.notes
        )
    }

    static func makeSupportedEntry(from record: SupportedModelCatalogRecord) -> SupportedModelCatalogEntry {
        let familyID = makeFamilyID(from: record.familyID)
        let manifest = ValarModelKit.ModelPackManifest(
            id: ModelIdentifier(record.modelID),
            familyID: familyID,
            displayName: record.displayName,
            domain: inferredDomain(for: familyID),
            capabilities: inferredCapabilities(from: [], familyID: familyID, modelID: record.modelID, displayName: record.displayName),
            supportedBackends: [BackendRequirement(backendKind: .mlx)],
            artifacts: [],
            licenses: [],
            notes: record.installHint
        )
        return SupportedModelCatalogEntry(
            manifest: manifest,
            remoteURL: record.providerURL.flatMap(URL.init(string:)),
            requiresManualDownload: record.sourceKind == .remoteURL,
            distributionTier: .optionalInstall,
            tags: [familyID.rawValue]
        )
    }

    static func makeModelManifest(from manifest: ValarPersistence.ModelPackManifest) -> ValarModelKit.ModelPackManifest {
        let familyID = makeFamilyID(from: manifest.familyID)
        return ValarModelKit.ModelPackManifest(
            id: ModelIdentifier(manifest.modelID),
            familyID: familyID,
            displayName: manifest.displayName,
            domain: inferredDomain(for: familyID),
            capabilities: inferredCapabilities(
                from: manifest.capabilities,
                familyID: familyID,
                modelID: manifest.modelID,
                displayName: manifest.displayName
            ),
            supportedBackends: parseBackendRequirements(manifest.backendKinds),
            artifacts: manifest.artifactSpecs.map { artifact in
                ArtifactSpec(
                    id: artifact.id,
                    role: artifactRole(for: artifact.kind),
                    relativePath: artifact.relativePath,
                    sha256: artifact.checksum,
                    sizeBytes: artifact.byteCount,
                    required: artifact.required
                )
            },
            tokenizer: manifest.tokenizerType.map { TokenizerSpec(kind: $0) },
            audio: AudioConstraint(defaultSampleRate: manifest.sampleRate),
            licenses: makeLicenseSpecs(from: manifest),
            minimumAppVersion: manifest.minimumAppVersion,
            notes: manifest.notes
        )
    }

    static func makeFamilyID(from value: String) -> ModelFamilyID {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        switch trimmed {
        case ModelFamilyID.qwen3TTS.rawValue:
            return .qwen3TTS
        case ModelFamilyID.qwen3ASR.rawValue:
            return .qwen3ASR
        case ModelFamilyID.qwen3ForcedAligner.rawValue:
            return .qwen3ForcedAligner
        case ModelFamilyID.voxtralTTS.rawValue:
            return .voxtralTTS
        case ModelFamilyID.tadaTTS.rawValue:
            return .tadaTTS
        case ModelFamilyID.soprano.rawValue:
            return .soprano
        case ModelFamilyID.whisper.rawValue:
            return .whisper
        default:
            return trimmed.isEmpty ? .unknown : ModelFamilyID(trimmed)
        }
    }

    static func inferredDomain(for familyID: ModelFamilyID) -> ModelDomain {
        switch familyID {
        case .qwen3TTS, .voxtralTTS, .tadaTTS, .soprano:
            return .tts
        case .qwen3ASR, .whisper:
            return .stt
        case .qwen3ForcedAligner:
            return .stt
        case .unknown:
            return .utility
        default:
            return .utility
        }
    }

    static func inferredCapabilities(
        from rawValues: [String],
        familyID: ModelFamilyID,
        modelID: String,
        displayName: String
    ) -> Set<ModelCapability> {
        let explicit = Set(rawValues.map { CapabilityID(rawValue: $0) })
        if !explicit.isEmpty {
            return explicit
        }

        let label = "\(modelID) \(displayName)".lowercased()
        switch familyID {
        case .qwen3TTS:
            var capabilities: Set<ModelCapability> = [.speechSynthesis, .tokenization, .longFormRendering]
            if label.contains("customvoice") {
                capabilities.insert(.voiceCloning)
                capabilities.insert(.audioConditioning)
            }
            if label.contains("voicedesign") {
                capabilities.insert(.voiceDesign)
                capabilities.insert(.audioConditioning)
            }
            return capabilities
        case .voxtralTTS:
            return [.speechSynthesis, .tokenization, .longFormRendering]
        case .tadaTTS:
            var capabilities: Set<ModelCapability> = [.speechSynthesis, .voiceCloning, .audioConditioning]
            if label.contains("3b") {
                capabilities.insert(.multilingual)
            }
            return capabilities
        case .soprano:
            return [.speechSynthesis, .tokenization, .longFormRendering]
        case .qwen3ASR:
            return [.speechRecognition, .tokenization, .translation]
        case .qwen3ForcedAligner:
            return [.speechRecognition, .forcedAlignment, .tokenization]
        case .whisper:
            return [.speechRecognition, .tokenization]
        case .unknown:
            return []
        default:
            return []
        }
    }

    static func parseBackendRequirements(_ values: [String]) -> [BackendRequirement] {
        let requirements = values.compactMap { value in
            BackendKind(rawValue: value).map { BackendRequirement(backendKind: $0) }
        }
        return requirements.isEmpty ? [BackendRequirement(backendKind: .mlx)] : requirements
    }

    static func makeLicenseSpecs(from manifest: ValarPersistence.ModelPackManifest) -> [LicenseSpec] {
        if let licenseName = manifest.licenseName {
            return [
                LicenseSpec(
                    name: licenseName,
                    sourceURL: manifest.licenseURL.flatMap(URL.init(string:)),
                    requiresAttribution: true
                ),
            ]
        }
        if let licenseURL = manifest.licenseURL {
            return [LicenseSpec(name: "Model license", sourceURL: URL(string: licenseURL), requiresAttribution: true)]
        }
        return []
    }

    static func artifactRole(for value: String) -> ArtifactRole {
        switch value.lowercased() {
        case "weights":
            return .weights
        case "config":
            return .config
        case "tokenizer":
            return .tokenizer
        case "vocabulary":
            return .vocabulary
        case "prompttemplate", "prompt_template":
            return .promptTemplate
        case "conditioning":
            return .conditioning
        case "voiceasset", "voice_asset":
            return .voiceAsset
        case "checksum":
            return .checksum
        case "license":
            return .license
        default:
            return .auxiliary
        }
    }
}

public enum ModelInstallValidationSeverity: String, Codable, Sendable {
    case warning
    case error
}

public struct ModelInstallValidationIssue: Codable, Sendable, Equatable, Identifiable {
    public let id: UUID
    public let severity: ModelInstallValidationSeverity
    public let message: String

    public init(
        id: UUID = UUID(),
        severity: ModelInstallValidationSeverity,
        message: String
    ) {
        self.id = id
        self.severity = severity
        self.message = message
    }
}

public struct ModelInstallValidationReport: Sendable, Equatable {
    public let manifest: ValarPersistence.ModelPackManifest
    public let issues: [ModelInstallValidationIssue]

    public init(manifest: ValarPersistence.ModelPackManifest, issues: [ModelInstallValidationIssue]) {
        self.manifest = manifest
        self.issues = issues
    }

    public var hasErrors: Bool {
        issues.contains { $0.severity == .error }
    }
}

public enum ModelInstallerError: Error, Equatable {
    case validationFailed([String])
    case installedRecordMissing(String)
    case installedPackMissing(String)
    case invalidRemoteSourceLocation(String)
    case downloadFailed(String)
    case checksumMismatch(artifactPath: String, expected: String, actual: String)
    case missingChecksum(artifactPath: String)
}

public enum ModelInstallMode: Sendable, Equatable {
    case metadataOnly
    case downloadArtifacts
}

public enum ModelInstallProgressStatus: Sendable, Equatable {
    case starting
    case downloading
    case verifying
    case completed
    case failed(String)
}

public struct ModelInstallProgressEvent: Sendable, Equatable, Identifiable {
    public let id: UUID
    public let modelID: String
    public let progress: Double
    public let status: ModelInstallProgressStatus

    public init(
        id: UUID = UUID(),
        modelID: String,
        progress: Double,
        status: ModelInstallProgressStatus
    ) {
        self.id = id
        self.modelID = modelID
        self.progress = progress
        self.status = status
    }
}

public struct ModelInstallationResult: Sendable, Equatable {
    public let report: ModelInstallValidationReport
    public let receipt: ModelInstallReceipt
    public let record: InstalledModelRecord
    public let descriptor: ModelDescriptor

    public init(
        report: ModelInstallValidationReport,
        receipt: ModelInstallReceipt,
        record: InstalledModelRecord,
        descriptor: ModelDescriptor
    ) {
        self.report = report
        self.receipt = receipt
        self.record = record
        self.descriptor = descriptor
    }
}

public actor ModelInstaller {
    private static let remoteChecksumRequiredKinds: Set<String> =
        Set(ArtifactRole.allCases.map(\.rawValue))
    private static let voxtralConverterRelativePath = "scripts/voxtral/convert_voice_embeddings.py"
    private static let voxtralToolingBootstrapRelativePath = "scripts/voxtral/bootstrap_env.sh"
    private static let voxtralToolingPythonRelativePaths = [
        "scripts/voxtral/.venv/bin/python3",
        "scripts/voxtral/.venv/bin/python",
    ]
    private static let tadaTokenizerPreferredModelIDs = [
        "HumeAI/mlx-tada-1b",
        "HumeAI/mlx-tada-3b",
    ]
    private static let tadaTokenizerFallbackModelIDs = [
        "meta-llama/Llama-3.2-1B",
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-3B",
        "meta-llama/Llama-3.2-3B-Instruct",
    ]
    private static let tadaTokenizerRemediationMessage =
        "TADA install is missing tokenizer.json and no cached fallback was found. " +
        "Re-download the HumeAI/mlx-tada-* repo or place tokenizer.json in your local Hugging Face cache, then retry."
    private static let vibeVoiceTokenizerSourceModelID =
        VibeVoiceCatalog.tokenizerSourceModelIdentifier.rawValue
    private static let vibeVoiceTokenizerArtifactRelativePaths: Set<String> = [
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    private let registry: any ModelPackManaging
    private let modelRegistry: (any ModelRegistryManaging)?
    private let capabilityRegistry: (any CapabilityRegistryManaging)?
    private let paths: ValarAppPaths
    private let fileManager: FileManager
    private let sessionFactory: @Sendable (URLSessionDownloadDelegate) -> URLSession
    private let hfCacheRoot: URL?
    public nonisolated let progress: AsyncStream<ModelInstallProgressEvent>
    private let progressContinuation: AsyncStream<ModelInstallProgressEvent>.Continuation

    public init(
        registry: any ModelPackManaging,
        modelRegistry: (any ModelRegistryManaging)? = nil,
        capabilityRegistry: (any CapabilityRegistryManaging)? = nil,
        paths: ValarAppPaths = ValarAppPaths(),
        fileManager: FileManager = .default,
        hfCacheRoot: URL? = nil,
        sessionFactory: @escaping @Sendable (URLSessionDownloadDelegate) -> URLSession = { delegate in
            let configuration = URLSessionConfiguration.default
            configuration.requestCachePolicy = .reloadIgnoringLocalCacheData
            return URLSession(configuration: configuration, delegate: delegate, delegateQueue: nil)
        }
    ) {
        self.registry = registry
        self.modelRegistry = modelRegistry
        self.capabilityRegistry = capabilityRegistry
        self.paths = paths
        self.fileManager = fileManager
        self.hfCacheRoot = hfCacheRoot
        self.sessionFactory = sessionFactory
        let stream = AsyncStream<ModelInstallProgressEvent>.makeStream()
        self.progress = stream.stream
        self.progressContinuation = stream.continuation
    }

    public func validate(_ manifest: ValarPersistence.ModelPackManifest) -> ModelInstallValidationReport {
        let whitespace = CharacterSet.whitespacesAndNewlines
        var issues: [ModelInstallValidationIssue] = []

        if manifest.familyID.trimmingCharacters(in: whitespace).isEmpty {
            issues.append(.init(severity: .error, message: "familyID is required"))
        }
        if manifest.modelID.trimmingCharacters(in: whitespace).isEmpty {
            issues.append(.init(severity: .error, message: "modelID is required"))
        }
        if manifest.displayName.trimmingCharacters(in: whitespace).isEmpty {
            issues.append(.init(severity: .error, message: "displayName is required"))
        }
        if manifest.capabilities.isEmpty {
            issues.append(.init(severity: .error, message: "At least one capability is required"))
        }
        if manifest.backendKinds.isEmpty {
            issues.append(.init(severity: .error, message: "At least one backend kind is required"))
        }
        if manifest.artifactSpecs.isEmpty {
            issues.append(.init(severity: .error, message: "At least one artifact spec is required"))
        }
        if manifest.licenseName == nil && manifest.licenseURL == nil {
            issues.append(.init(severity: .error, message: "License metadata is required for installable model packs"))
        }
        if manifest.familyID == ModelFamilyID.qwen3TTS.rawValue && manifest.tokenizerType == nil {
            issues.append(.init(severity: .warning, message: "Qwen3 TTS packs should declare tokenizerType"))
        }
        if manifest.sampleRate == nil {
            issues.append(.init(severity: .warning, message: "sampleRate is missing; runtime defaults will be used"))
        }
        if manifest.artifactSpecs.contains(where: { $0.relativePath.trimmingCharacters(in: whitespace).isEmpty }) {
            issues.append(.init(severity: .error, message: "Artifact relative paths must be non-empty"))
        }
        for artifact in manifest.artifactSpecs {
            do {
                try ValarAppPaths.validateRelativePath(
                    artifact.relativePath,
                    label: "Artifact relative path '\(artifact.id)'"
                )
            } catch {
                issues.append(.init(severity: .error, message: error.localizedDescription))
            }
        }
        let uncheckedArtifacts = manifest.artifactSpecs.filter {
            Self.remoteChecksumRequiredKinds.contains($0.kind)
                && !$0.relativePath.hasSuffix("/")
                && $0.checksum == nil
        }
        for artifact in uncheckedArtifacts {
            issues.append(.init(
                severity: .warning,
                message: "\(Self.checksumWarningLabel(for: artifact.kind)) artifact '\(artifact.id)' is missing a SHA-256 checksum; Valar can install it, but cannot locally verify the downloaded file"
            ))
        }
        do {
            try ValarAppPaths.validateRelativePath(manifest.familyID, label: "familyID")
        } catch {
            issues.append(.init(severity: .error, message: error.localizedDescription))
        }
        do {
            try ValarAppPaths.validateRelativePath(manifest.modelID, label: "modelID")
        } catch {
            issues.append(.init(severity: .error, message: error.localizedDescription))
        }

        return ModelInstallValidationReport(manifest: manifest, issues: issues)
    }

    public func install(
        manifest: ValarPersistence.ModelPackManifest,
        sourceKind: ModelPackSourceKind,
        sourceLocation: String,
        notes: String? = nil,
        mode: ModelInstallMode = .metadataOnly
    ) async throws -> ModelInstallationResult {
        let report = validate(manifest)
        if report.hasErrors {
            throw ModelInstallerError.validationFailed(report.issues.map { $0.message })
        }

        if mode == .downloadArtifacts {
            try preflightInstallToolingIfNeeded(manifest: manifest)
            do {
                try await downloadArtifactsIfNeeded(
                    manifest: manifest,
                    sourceKind: sourceKind,
                    sourceLocation: sourceLocation
                )
            } catch {
                progressContinuation.yield(
                    ModelInstallProgressEvent(
                        modelID: manifest.modelID,
                        progress: 0,
                        status: .failed(String(describing: error))
                    )
                )
                throw error
            }
        }

        let receipt = try await registry.install(
            manifest: manifest,
            sourceKind: sourceKind,
            sourceLocation: sourceLocation,
            notes: notes
        )

        guard let record = try await registry.installedRecord(for: manifest.modelID) else {
            throw ModelInstallerError.installedRecordMissing(manifest.modelID)
        }
        guard fileManager.fileExists(atPath: record.installedPath) else {
            _ = try? await registry.uninstall(modelID: manifest.modelID)
            throw ModelInstallerError.installedPackMissing(record.installedPath)
        }

        let descriptor = ModelDescriptor(manifest: ModelCatalog.makeModelManifest(from: manifest))
        let estimatedBytes = manifest.artifactSpecs.compactMap { $0.byteCount }.reduce(0, +)
        let runtimeConfiguration = ModelRuntimeConfiguration(
            backendKind: ModelCatalog.parseBackendRequirements(manifest.backendKinds).first?.backendKind ?? .mlx,
            residencyPolicy: .automatic,
            preferredSampleRate: manifest.sampleRate,
            memoryBudgetBytes: estimatedBytes > 0 ? estimatedBytes : nil,
            allowQuantizedWeights: true,
            allowWarmStart: true
        )

        if let modelRegistry {
            await modelRegistry.register(
                descriptor,
                estimatedResidentBytes: estimatedBytes > 0 ? estimatedBytes : nil,
                runtimeConfiguration: runtimeConfiguration
            )
        }
        if let capabilityRegistry {
            await capabilityRegistry.register(descriptor)
        }

        return ModelInstallationResult(
            report: report,
            receipt: receipt,
            record: record,
            descriptor: descriptor
        )
    }

    public nonisolated static func tadaTokenizerInstallIssue(
        preferredModelID: String? = nil,
        fileManager: FileManager = .default,
        hfCacheRoot: URL? = nil
    ) -> String? {
        resolveTadaTokenizerSource(
            preferredModelID: preferredModelID,
            fileManager: fileManager,
            hfCacheRoot: hfCacheRoot
        ) == nil
            ? tadaTokenizerRemediationMessage
            : nil
    }

    private func preflightInstallToolingIfNeeded(manifest: ValarPersistence.ModelPackManifest) throws {
        switch manifest.familyID {
        case ModelFamilyID.voxtralTTS.rawValue:
            let requiresManagedNormalization = manifest.artifactSpecs.contains { artifact in
                artifact.relativePath.lowercased().hasSuffix(".pt")
            }
            if requiresManagedNormalization {
                _ = try Self.resolveManagedVoxtralPython(
                    fileManager: fileManager,
                    validateTorchImport: true
                )
            }
        default:
            return
        }
    }

    public func uninstall(descriptor: ModelDescriptor) async throws -> InstalledModelRecord? {
        try await uninstall(modelID: descriptor.id)
    }

    public func uninstall(modelID: ModelIdentifier) async throws -> InstalledModelRecord? {
        guard let existingRecord = try await registry.installedRecord(for: modelID.rawValue) else {
            return nil
        }

        if FileManager.default.fileExists(atPath: existingRecord.installedPath) {
            try FileManager.default.removeItem(atPath: existingRecord.installedPath)
            try pruneEmptyModelPackDirectories(startingAt: URL(fileURLWithPath: existingRecord.installedPath, isDirectory: true))
        }

        guard let record = try await registry.uninstall(modelID: modelID.rawValue) else {
            return existingRecord
        }

        if let modelRegistry {
            await modelRegistry.unregister(modelID)
        }
        if let capabilityRegistry {
            await capabilityRegistry.unregister(modelID)
        }

        return record
    }

    public func purgeSharedCaches(for modelID: ModelIdentifier) throws -> [String] {
        let hubRoot = Self.resolveHFHubCacheRoot(fileManager: fileManager, hfCacheRoot: hfCacheRoot)
        let standardDirectory = hubRoot.appendingPathComponent(Self.hfHubRepoDirectoryName(for: modelID.rawValue), isDirectory: true)
        let legacyDirectory = hubRoot
            .appendingPathComponent("mlx-audio", isDirectory: true)
            .appendingPathComponent(Self.hfMLXAudioDirectoryName(for: modelID.rawValue), isDirectory: true)

        var removedPaths: [String] = []
        if fileManager.fileExists(atPath: standardDirectory.path) {
            try fileManager.removeItem(at: standardDirectory)
            removedPaths.append(standardDirectory.path)
        }
        if fileManager.fileExists(atPath: legacyDirectory.path) {
            try fileManager.removeItem(at: legacyDirectory)
            removedPaths.append(legacyDirectory.path)
        }
        return removedPaths
    }

    private func downloadArtifactsIfNeeded(
        manifest: ValarPersistence.ModelPackManifest,
        sourceKind: ModelPackSourceKind,
        sourceLocation: String
    ) async throws {
        guard sourceKind == .remoteURL else { return }
        guard let baseURL = URL(string: sourceLocation) else {
            throw ModelInstallerError.invalidRemoteSourceLocation(sourceLocation)
        }
        guard let scheme = baseURL.scheme?.lowercased(), scheme == "https" else {
            throw ModelInstallerError.invalidRemoteSourceLocation(sourceLocation)
        }

        let packDirectory = try paths.modelPackDirectory(familyID: manifest.familyID, modelID: manifest.modelID)
        let stagingDirectory = packDirectory
            .deletingLastPathComponent()
            .appendingPathComponent(".\(packDirectory.lastPathComponent)-downloading", isDirectory: true)

        try removeIfPresent(stagingDirectory)
        try fileManager.createDirectory(at: stagingDirectory, withIntermediateDirectories: true)

        do {
            let downloadableArtifacts = manifest.artifactSpecs.filter { !$0.relativePath.hasSuffix("/") }
            let placeholderDirectories = manifest.artifactSpecs.filter { $0.relativePath.hasSuffix("/") }

            for artifact in placeholderDirectories {
                let directoryURL = stagingDirectory.appendingPathComponent(artifact.relativePath, isDirectory: true)
                try ValarAppPaths.validateContainment(directoryURL, within: stagingDirectory)
                try fileManager.createDirectory(at: directoryURL, withIntermediateDirectories: true)
            }

            progressContinuation.yield(
                ModelInstallProgressEvent(
                    modelID: manifest.modelID,
                    progress: 0,
                    status: .starting
                )
            )

            if !downloadableArtifacts.isEmpty {
                let delegateState = ModelInstallerDownloadState()
                let delegate = ModelInstallerDownloadDelegate(state: delegateState)
                let session = sessionFactory(delegate)
                defer { session.finishTasksAndInvalidate() }

                let totalArtifacts = Double(downloadableArtifacts.count)
                for (index, artifact) in downloadableArtifacts.enumerated() {
                    let baseProgress = Double(index) / totalArtifacts
                    let weight = 1 / totalArtifacts
                    let destinationURL = stagingDirectory.appendingPathComponent(artifact.relativePath, isDirectory: false)
                    try ValarAppPaths.validateContainment(destinationURL, within: stagingDirectory)

                    try fileManager.createDirectory(
                        at: destinationURL.deletingLastPathComponent(),
                        withIntermediateDirectories: true
                    )

                    // Prefer the standard Hugging Face snapshot cache when present.
                    // Fall back to the legacy mlx-audio cache layout for compatibility.
                    let hfCached = resolvedCachedArtifactURL(manifest: manifest, artifact: artifact)

                    if let cachedURL = hfCached {
                        try copyCachedArtifact(at: cachedURL, to: destinationURL)
                    } else {
                        let artifactURL = try remoteArtifactURL(
                            for: manifest,
                            artifact: artifact,
                            primaryBaseURL: baseURL
                        )

                        let onProgress: @Sendable (Double) -> Void = { [progressContinuation] fraction in
                            let overallProgress = min(1, max(0, baseProgress + (fraction * weight)))
                            progressContinuation.yield(
                                ModelInstallProgressEvent(
                                    modelID: manifest.modelID,
                                    progress: overallProgress,
                                    status: .downloading
                                )
                            )
                        }

                        do {
                            _ = try await downloadArtifact(
                                from: artifactURL,
                                to: destinationURL,
                                session: session,
                                state: delegateState,
                                onProgress: onProgress
                            )
                        } catch ModelInstallerError.downloadFailed(let msg)
                            where !artifact.required && msg.contains("404") {
                            // Optional artifact not present on remote — skip without aborting.
                            continue
                        }
                    }

                    if let checksum = artifact.checksum {
                        progressContinuation.yield(
                            ModelInstallProgressEvent(
                                modelID: manifest.modelID,
                                progress: min(1, baseProgress + weight),
                                status: .verifying
                            )
                        )
                        let actualChecksum = try sha256Hex(for: destinationURL)
                        guard actualChecksum.caseInsensitiveCompare(checksum) == .orderedSame else {
                            throw ModelInstallerError.checksumMismatch(
                                artifactPath: artifact.relativePath,
                                expected: checksum,
                                actual: actualChecksum
                            )
                        }
                    } else if artifact.checksum != nil && hfCached == nil {
                        // Catalog declared a checksum but it wasn't verified above —
                        // this shouldn't happen, but guard against it.
                        throw ModelInstallerError.missingChecksum(artifactPath: artifact.relativePath)
                        // Note: models without pre-computed checksums (checksum == nil)
                        // are trusted when downloaded directly from HuggingFace.
                    }
                }
            }

            try prepareArtifactsForRuntimeIfNeeded(manifest: manifest, stagingDirectory: stagingDirectory)

            try writeManifest(manifest, to: stagingDirectory)
            try removeIfPresent(packDirectory)
            try fileManager.moveItem(at: stagingDirectory, to: packDirectory)
            progressContinuation.yield(
                ModelInstallProgressEvent(
                    modelID: manifest.modelID,
                    progress: 1,
                    status: .completed
                )
            )
        } catch {
            try? removeIfPresent(stagingDirectory)
            throw error
        }
    }

    private func prepareArtifactsForRuntimeIfNeeded(
        manifest: ValarPersistence.ModelPackManifest,
        stagingDirectory: URL
    ) throws {
        switch manifest.familyID {
        case ModelFamilyID.voxtralTTS.rawValue:
            try normalizeVoxtralVoiceEmbeddingsIfNeeded(in: stagingDirectory)
        case ModelFamilyID.tadaTTS.rawValue:
            try materializeTadaTokenizerIfNeeded(in: stagingDirectory, preferredModelID: manifest.modelID)
        case ModelFamilyID.vibevoiceRealtimeTTS.rawValue:
            try synthesizeVibeVoiceTokenizerMetadataIfNeeded(in: stagingDirectory)
        default:
            break
        }
    }

    private func normalizeVoxtralVoiceEmbeddingsIfNeeded(in modelRoot: URL) throws {
        let safeDirectory = modelRoot.appendingPathComponent("voice_embedding_safe", isDirectory: true)
        let safeManifestURL = safeDirectory.appendingPathComponent("index.json", isDirectory: false)
        if fileManager.fileExists(atPath: safeManifestURL.path) {
            return
        }

        let rawDirectory = modelRoot.appendingPathComponent("voice_embedding", isDirectory: true)
        let rawVoiceFiles = (try? fileManager.contentsOfDirectory(
            at: rawDirectory,
            includingPropertiesForKeys: nil
        ).filter { $0.pathExtension.lowercased() == "pt" }) ?? []
        let safetensorsVoiceFiles = (try? fileManager.contentsOfDirectory(
            at: rawDirectory,
            includingPropertiesForKeys: nil
        ).filter { $0.pathExtension.lowercased() == "safetensors" }) ?? []
        if !safetensorsVoiceFiles.isEmpty {
            return
        }
        guard rawVoiceFiles.isEmpty == false else {
            throw ModelInstallerError.downloadFailed(
                "Voxtral install is missing preset voice embedding assets at \(rawDirectory.path)."
            )
        }

        let converterURL = try Self.resolveVoxtralConverterScript(fileManager: fileManager)
        let pythonURL = try Self.resolveManagedVoxtralPython(
            fileManager: fileManager,
            validateTorchImport: true
        )
        let outputPipe = Pipe()
        let process = Process()
        process.executableURL = pythonURL
        process.arguments = [
            converterURL.path,
            modelRoot.path,
            "--output",
            safeDirectory.path,
        ]
        process.standardOutput = outputPipe
        process.standardError = outputPipe

        do {
            try process.run()
        } catch {
            throw ModelInstallerError.downloadFailed(
                "Failed to launch the managed Voxtral voice normalizer at \(pythonURL.path). " +
                    "Run 'bash \(try Self.resolveVoxtralSetupScript(fileManager: fileManager).path)' to repair the toolchain."
            )
        }

        process.waitUntilExit()
        let outputData = outputPipe.fileHandleForReading.readDataToEndOfFile()
        let output = String(data: outputData, encoding: .utf8)?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""

        guard process.terminationStatus == 0 else {
            let detail = output.isEmpty ? "No output captured." : output
            throw ModelInstallerError.downloadFailed(
                "Voxtral voice normalization failed. \(detail)"
            )
        }

        guard fileManager.fileExists(atPath: safeManifestURL.path) else {
            throw ModelInstallerError.downloadFailed(
                "Voxtral voice normalization completed without producing \(safeManifestURL.lastPathComponent)."
            )
        }

        if fileManager.fileExists(atPath: rawDirectory.path) {
            try fileManager.removeItem(at: rawDirectory)
        }
    }

    private func materializeTadaTokenizerIfNeeded(
        in modelRoot: URL,
        preferredModelID: String
    ) throws {
        let tokenizerURL = modelRoot.appendingPathComponent("tokenizer.json", isDirectory: false)
        if fileManager.fileExists(atPath: tokenizerURL.path) {
            return
        }

        guard let sourceURL = resolveTadaTokenizerSource(preferredModelID: preferredModelID) else {
            throw ModelInstallerError.downloadFailed(Self.tadaTokenizerRemediationMessage)
        }

        try ValarAppPaths.validateContainment(tokenizerURL, within: modelRoot)
        try fileManager.copyItem(at: sourceURL, to: tokenizerURL)
    }

    private func resolveTadaTokenizerSource(preferredModelID: String? = nil) -> URL? {
        Self.resolveTadaTokenizerSource(
            preferredModelID: preferredModelID,
            fileManager: fileManager,
            hfCacheRoot: hfCacheRoot
        )
    }

    private nonisolated static func resolveTadaTokenizerSource(
        preferredModelID: String? = nil,
        fileManager: FileManager,
        hfCacheRoot: URL?
    ) -> URL? {
        let candidateModelIDs = tadaTokenizerCandidateIDs(preferredModelID: preferredModelID)
        for modelID in candidateModelIDs {
            if let cached = hfHubArtifactURL(
                modelID: modelID,
                relativePath: "tokenizer.json",
                fileManager: fileManager,
                hfCacheRoot: hfCacheRoot
            ) {
                return cached
            }
        }
        return nil
    }

    private nonisolated static func tadaTokenizerCandidateIDs(preferredModelID: String?) -> [String] {
        var ordered: [String] = []
        if let preferredModelID,
           !preferredModelID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            ordered.append(preferredModelID)
        }
        ordered.append(contentsOf: tadaTokenizerPreferredModelIDs)
        ordered.append(contentsOf: tadaTokenizerFallbackModelIDs)

        var seen: Set<String> = []
        return ordered.filter { seen.insert($0).inserted }
    }

    private func synthesizeVibeVoiceTokenizerMetadataIfNeeded(in modelRoot: URL) throws {
        let tokenizerConfigURL = modelRoot.appendingPathComponent("tokenizer_config.json", isDirectory: false)
        guard fileManager.fileExists(atPath: tokenizerConfigURL.path) else {
            return
        }

        let configData = try Data(contentsOf: tokenizerConfigURL)
        guard let root = try JSONSerialization.jsonObject(with: configData) as? [String: Any] else {
            return
        }

        let addedTokensURL = modelRoot.appendingPathComponent("added_tokens.json", isDirectory: false)
        if !fileManager.fileExists(atPath: addedTokensURL.path),
           let decoder = root["added_tokens_decoder"] as? [String: Any] {
            var mapping: [String: Int] = [:]
            for (rawID, value) in decoder {
                guard let tokenID = Int(rawID),
                      let token = value as? [String: Any],
                      let content = token["content"] as? String,
                      !content.isEmpty else {
                    continue
                }
                mapping[content] = tokenID
            }
            if !mapping.isEmpty {
                try writeJSON(mapping, to: addedTokensURL)
            }
        }

        let specialTokensURL = modelRoot.appendingPathComponent("special_tokens_map.json", isDirectory: false)
        if !fileManager.fileExists(atPath: specialTokensURL.path) {
            let scalarKeys = [
                "bos_token",
                "eos_token",
                "unk_token",
                "sep_token",
                "pad_token",
                "cls_token",
                "mask_token",
            ]
            var specialTokensMap: [String: Any] = [:]
            for key in scalarKeys {
                if let value = root[key] {
                    specialTokensMap[key] = value
                }
            }
            if let additional = root["additional_special_tokens"] {
                specialTokensMap["additional_special_tokens"] = additional
            }
            if !specialTokensMap.isEmpty {
                try writeJSON(specialTokensMap, to: specialTokensURL)
            }
        }
    }

    private nonisolated static func fallbackHubSourceModelID(
        familyID: String,
        relativePath: String
    ) -> String? {
        switch familyID {
        case ModelFamilyID.vibevoiceRealtimeTTS.rawValue
            where vibeVoiceTokenizerArtifactRelativePaths.contains(relativePath):
            return vibeVoiceTokenizerSourceModelID
        default:
            return nil
        }
    }

    private static func repositoryRootURL() -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    private static func resolveVoxtralConverterScript(fileManager: FileManager) throws -> URL {
        let repositoryRoot = repositoryRootURL()
        let scriptURL = repositoryRoot.appendingPathComponent(voxtralConverterRelativePath)
        if fileManager.fileExists(atPath: scriptURL.path) {
            return scriptURL
        }

        throw ModelInstallerError.downloadFailed(
            "Unable to locate Voxtral converter script at \(scriptURL.path)."
        )
    }

    private static func resolveVoxtralSetupScript(fileManager: FileManager) throws -> URL {
        let repositoryRoot = repositoryRootURL()
        let scriptURL = repositoryRoot.appendingPathComponent(voxtralToolingBootstrapRelativePath)
        if fileManager.fileExists(atPath: scriptURL.path) {
            return scriptURL
        }

        throw ModelInstallerError.downloadFailed(
            "Unable to locate Voxtral setup script at \(scriptURL.path)."
        )
    }

    private static func resolveManagedVoxtralPython(
        fileManager: FileManager,
        validateTorchImport: Bool
    ) throws -> URL {
        let repositoryRoot = repositoryRootURL()
        let candidates = voxtralToolingPythonRelativePaths.map { repositoryRoot.appendingPathComponent($0) }

        guard let pythonURL = candidates.first(where: { fileManager.isExecutableFile(atPath: $0.path) }) else {
            let setupScript = try resolveVoxtralSetupScript(fileManager: fileManager)
            throw ModelInstallerError.downloadFailed(
                "Voxtral installs require the managed normalization toolchain under scripts/voxtral/.venv. " +
                    "Run 'bash \(setupScript.path)' once, then retry."
            )
        }

        guard validateTorchImport else {
            return pythonURL
        }

        let outputPipe = Pipe()
        let process = Process()
        process.executableURL = pythonURL
        process.arguments = ["-c", "import torch"]
        process.standardOutput = outputPipe
        process.standardError = outputPipe

        do {
            try process.run()
        } catch {
            let setupScript = try resolveVoxtralSetupScript(fileManager: fileManager)
            throw ModelInstallerError.downloadFailed(
                "Failed to launch the managed Voxtral toolchain at \(pythonURL.path). " +
                    "Run 'bash \(setupScript.path)' to recreate it."
            )
        }

        process.waitUntilExit()
        guard process.terminationStatus == 0 else {
            let detail = String(data: outputPipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8)?
                .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            let setupScript = try resolveVoxtralSetupScript(fileManager: fileManager)
            let suffix = detail.isEmpty ? "" : " (\(detail))"
            throw ModelInstallerError.downloadFailed(
                "The managed Voxtral toolchain is missing torch or is otherwise broken\(suffix). " +
                    "Run 'bash \(setupScript.path)' to repair it."
            )
        }

        return pythonURL
    }

    /// Returns the URL of an artifact in the mlx-audio HuggingFace cache, or nil if absent.
    ///
    /// The mlx-audio Python library caches models in a flat layout (all files at root),
    /// while ValarTTS catalog artifacts use subdirectories (e.g. `weights/model.safetensors`).
    /// This method checks the exact subpath first, then falls back to the flat basename,
    /// and also maps known directory aliases (e.g. `conditioning/` → `speech_tokenizer/`).
    private func hfMLXAudioArtifactURL(modelID: String, relativePath: String) -> URL? {
        let dirName = modelID.replacingOccurrences(of: "/", with: "_")
        let cacheBase = Self.resolveMLXAudioCacheRoot(
            fileManager: fileManager,
            hfCacheRoot: hfCacheRoot
        )
        let rootDir = cacheBase.appendingPathComponent(dirName)

        // 1. Exact subpath match (ValarTTS layout)
        let exactURL = rootDir.appendingPathComponent(relativePath)
        if fileManager.fileExists(atPath: exactURL.path) { return exactURL }

        // 2. Flat basename fallback (mlx-audio flat cache)
        let basename = (relativePath as NSString).lastPathComponent
        if basename != relativePath {
            let flatURL = rootDir.appendingPathComponent(basename)
            if fileManager.fileExists(atPath: flatURL.path) { return flatURL }
        }

        // 3. Known directory aliases
        let aliases: [String: String] = [
            "conditioning": "speech_tokenizer",
        ]
        let leadingComponent = relativePath.split(separator: "/", maxSplits: 1).first.map(String.init) ?? relativePath
        if let alias = aliases[leadingComponent] {
            let aliasedPath = relativePath.replacingOccurrences(
                of: leadingComponent, with: alias,
                range: relativePath.range(of: leadingComponent)
            )
            let aliasURL = rootDir.appendingPathComponent(aliasedPath)
            if fileManager.fileExists(atPath: aliasURL.path) { return aliasURL }
        }

        return nil
    }

    private func resolvedCachedArtifactURL(
        manifest: ValarPersistence.ModelPackManifest,
        artifact: ModelPackArtifact
    ) -> URL? {
        if let direct = hfHubArtifactURL(modelID: manifest.modelID, relativePath: artifact.relativePath) {
            return direct
        }
        if let fallbackModelID = Self.fallbackHubSourceModelID(
            familyID: manifest.familyID,
            relativePath: artifact.relativePath
        ),
        let fallback = hfHubArtifactURL(modelID: fallbackModelID, relativePath: artifact.relativePath) {
            return fallback
        }
        return hfMLXAudioArtifactURL(modelID: manifest.modelID, relativePath: artifact.relativePath)
    }

    /// Returns the URL of an artifact in the standard Hugging Face Hub cache, or nil if absent.
    ///
    /// This allows `valartts models install ...` to register files that were downloaded earlier
    /// by `huggingface-cli`, `hf`, or another tool that uses the default hub snapshot cache.
    private func hfHubArtifactURL(modelID: String, relativePath: String) -> URL? {
        Self.hfHubArtifactURL(
            modelID: modelID,
            relativePath: relativePath,
            fileManager: fileManager,
            hfCacheRoot: hfCacheRoot
        )
    }

    private nonisolated static func hfHubArtifactURL(
        modelID: String,
        relativePath: String,
        fileManager: FileManager,
        hfCacheRoot: URL?
    ) -> URL? {
        guard let snapshotDirectory = hfHubSnapshotDirectory(
            modelID: modelID,
            fileManager: fileManager,
            hfCacheRoot: hfCacheRoot
        ) else {
            return nil
        }

        let exactURL = snapshotDirectory.appendingPathComponent(relativePath, isDirectory: false)
        return fileManager.fileExists(atPath: exactURL.path) ? exactURL : nil
    }

    private nonisolated static func hfHubSnapshotDirectory(
        modelID: String,
        fileManager: FileManager,
        hfCacheRoot: URL?
    ) -> URL? {
        let cacheBase = resolveHFHubCacheRoot(
            fileManager: fileManager,
            hfCacheRoot: hfCacheRoot
        )
        let repoDirectoryName = "models--" + modelID.replacingOccurrences(of: "/", with: "--")
        let repoDirectory = cacheBase.appendingPathComponent(repoDirectoryName, isDirectory: true)
        let snapshotsDirectory = repoDirectory.appendingPathComponent("snapshots", isDirectory: true)
        guard fileManager.fileExists(atPath: snapshotsDirectory.path) else {
            return nil
        }

        let snapshotDirectory: URL?
        let refsMain = repoDirectory.appendingPathComponent("refs/main", isDirectory: false)
        if let revision = try? String(contentsOf: refsMain, encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines),
           !revision.isEmpty
        {
            let candidate = snapshotsDirectory.appendingPathComponent(revision, isDirectory: true)
            snapshotDirectory = fileManager.fileExists(atPath: candidate.path) ? candidate : nil
        } else {
            let snapshots = (try? fileManager.contentsOfDirectory(
                at: snapshotsDirectory,
                includingPropertiesForKeys: [.contentModificationDateKey],
                options: [.skipsHiddenFiles]
            )) ?? []
            snapshotDirectory = snapshots.max(by: {
                let lhs = (try? $0.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
                let rhs = (try? $1.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
                return lhs < rhs
            })
        }

        guard let snapshotDirectory else {
            return nil
        }

        return snapshotDirectory
    }

    private nonisolated static func resolveMLXAudioCacheRoot(
        fileManager: FileManager,
        hfCacheRoot: URL?
    ) -> URL {
        resolveHFHubCacheRoot(fileManager: fileManager, hfCacheRoot: hfCacheRoot)
            .appendingPathComponent("mlx-audio", isDirectory: true)
    }

    private nonisolated static func resolveHFHubCacheRoot(
        fileManager: FileManager,
        hfCacheRoot: URL?
    ) -> URL {
        if let hfCacheRoot {
            return hfCacheRoot
        }

        let environment = ProcessInfo.processInfo.environment
        if let explicit = environment["HF_HUB_CACHE"]?.trimmingCharacters(in: .whitespacesAndNewlines),
           !explicit.isEmpty {
            return URL(fileURLWithPath: explicit, isDirectory: true)
        }
        if let explicit = environment["HUGGINGFACE_HUB_CACHE"]?.trimmingCharacters(in: .whitespacesAndNewlines),
           !explicit.isEmpty {
            return URL(fileURLWithPath: explicit, isDirectory: true)
        }
        if let hfHome = environment["HF_HOME"]?.trimmingCharacters(in: .whitespacesAndNewlines),
           !hfHome.isEmpty {
            return URL(fileURLWithPath: hfHome, isDirectory: true)
                .appendingPathComponent("hub", isDirectory: true)
        }

        return fileManager.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/hub", isDirectory: true)
    }

    private nonisolated static func hfMLXAudioDirectoryName(for modelID: String) -> String {
        modelID.replacingOccurrences(of: "/", with: "_")
    }

    private func downloadArtifact(
        from sourceURL: URL,
        to destinationURL: URL,
        session: URLSession,
        state: ModelInstallerDownloadState,
        onProgress: @escaping @Sendable (Double) -> Void
    ) async throws -> URL {
        let task = session.downloadTask(with: sourceURL)
        task.taskDescription = destinationURL.path

        return try await withCheckedThrowingContinuation { continuation in
            Task {
                await state.register(
                    taskIdentifier: task.taskIdentifier,
                    continuation: continuation,
                    onProgress: onProgress
                )
                task.resume()
            }
        }
    }

    private func remoteArtifactURL(baseURL: URL, relativePath: String) throws -> URL {
        let cleanRelativePath = relativePath.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        guard !cleanRelativePath.isEmpty else {
            throw ModelInstallerError.downloadFailed("Artifact path is empty")
        }

        var artifactURL = baseURL
        artifactURL.append(path: "resolve")
        artifactURL.append(path: "main")
        for component in cleanRelativePath.split(separator: "/").map(String.init) {
            artifactURL.append(path: component)
        }

        guard var components = URLComponents(url: artifactURL, resolvingAgainstBaseURL: false) else {
            throw ModelInstallerError.downloadFailed("Failed to resolve remote artifact URL for \(relativePath)")
        }
        components.queryItems = [URLQueryItem(name: "download", value: "true")]
        guard let resolvedURL = components.url else {
            throw ModelInstallerError.downloadFailed("Failed to encode remote artifact URL for \(relativePath)")
        }
        return resolvedURL
    }

    private func remoteArtifactURL(
        for manifest: ValarPersistence.ModelPackManifest,
        artifact: ModelPackArtifact,
        primaryBaseURL: URL
    ) throws -> URL {
        if let fallbackModelID = Self.fallbackHubSourceModelID(
            familyID: manifest.familyID,
            relativePath: artifact.relativePath
        ) {
            let fallbackBaseURL = URL(string: "https://huggingface.co/\(fallbackModelID)")!
            return try remoteArtifactURL(baseURL: fallbackBaseURL, relativePath: artifact.relativePath)
        }

        return try remoteArtifactURL(baseURL: primaryBaseURL, relativePath: artifact.relativePath)
    }

    private func writeManifest(_ manifest: ValarPersistence.ModelPackManifest, to directory: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        let manifestURL = directory.appendingPathComponent("manifest.json", isDirectory: false)
        try encoder.encode(manifest).write(to: manifestURL, options: .atomic)
    }

    private func writeJSON(_ value: Any, to url: URL) throws {
        let parent = url.deletingLastPathComponent()
        try ValarAppPaths.validateContainment(url, within: parent)
        let data = try JSONSerialization.data(withJSONObject: value, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: url, options: .atomic)
    }

    private func removeIfPresent(_ url: URL) throws {
        guard fileManager.fileExists(atPath: url.path) else { return }
        try fileManager.removeItem(at: url)
    }

    private func copyCachedArtifact(at sourceURL: URL, to destinationURL: URL) throws {
        let resolvedSource = sourceURL.resolvingSymlinksInPath()
        guard fileManager.fileExists(atPath: resolvedSource.path) else {
            throw ModelInstallerError.downloadFailed(
                "Cached artifact is missing or points to a broken symlink: \(sourceURL.path)"
            )
        }
        do {
            try fileManager.linkItem(at: resolvedSource, to: destinationURL)
            return
        } catch {
            try fileManager.copyItem(at: resolvedSource, to: destinationURL)
        }
    }

    private func pruneEmptyModelPackDirectories(startingAt directory: URL) throws {
        let allowedRoot = paths.modelPacksDirectory.standardizedFileURL
        var current = directory.deletingLastPathComponent().standardizedFileURL

        while current.path != allowedRoot.path {
            try ValarAppPaths.validateContainment(current, within: allowedRoot, fileManager: fileManager)
            let children = try fileManager.contentsOfDirectory(
                at: current,
                includingPropertiesForKeys: nil,
                options: [.skipsHiddenFiles]
            )
            guard children.isEmpty else {
                return
            }
            try fileManager.removeItem(at: current)
            current = current.deletingLastPathComponent().standardizedFileURL
        }
    }

    private func sha256Hex(for fileURL: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: fileURL)
        defer { try? handle.close() }

        var hasher = SHA256()
        while let chunk = try handle.read(upToCount: 1_048_576), !chunk.isEmpty {
            hasher.update(data: chunk)
        }

        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    private static func checksumWarningLabel(for kind: String) -> String {
        switch kind {
        case ArtifactRole.weights.rawValue:
            return "Weight"
        case ArtifactRole.config.rawValue:
            return "Config"
        case ArtifactRole.tokenizer.rawValue:
            return "Tokenizer"
        default:
            return kind.capitalized
        }
    }

    private nonisolated static func hfHubRepoDirectoryName(for modelID: String) -> String {
        "models--" + modelID.replacingOccurrences(of: "/", with: "--")
    }
}

private actor ModelInstallerDownloadState {
    private struct DownloadContext {
        let continuation: CheckedContinuation<URL, Error>
        let onProgress: @Sendable (Double) -> Void
    }

    private var contexts: [Int: DownloadContext] = [:]

    func register(
        taskIdentifier: Int,
        continuation: CheckedContinuation<URL, Error>,
        onProgress: @escaping @Sendable (Double) -> Void
    ) {
        contexts[taskIdentifier] = DownloadContext(
            continuation: continuation,
            onProgress: onProgress
        )
    }

    func reportProgress(taskIdentifier: Int, totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64) {
        guard totalBytesExpectedToWrite > 0 else { return }
        let fraction = min(1, max(0, Double(totalBytesWritten) / Double(totalBytesExpectedToWrite)))
        contexts[taskIdentifier]?.onProgress(fraction)
    }

    func finish(taskIdentifier: Int, fileURL: URL) {
        guard let context = contexts.removeValue(forKey: taskIdentifier) else { return }
        context.onProgress(1)
        context.continuation.resume(returning: fileURL)
    }

    func fail(taskIdentifier: Int, error: Error) {
        guard let context = contexts.removeValue(forKey: taskIdentifier) else { return }
        context.continuation.resume(throwing: error)
    }
}

private final class ModelInstallerDownloadDelegate: NSObject, URLSessionDownloadDelegate {
    private let state: ModelInstallerDownloadState

    init(state: ModelInstallerDownloadState) {
        self.state = state
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64,
        totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        Task {
            await state.reportProgress(
                taskIdentifier: downloadTask.taskIdentifier,
                totalBytesWritten: totalBytesWritten,
                totalBytesExpectedToWrite: totalBytesExpectedToWrite
            )
        }
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didFinishDownloadingTo location: URL
    ) {
        guard let response = downloadTask.response as? HTTPURLResponse else {
            Task {
                await state.fail(
                    taskIdentifier: downloadTask.taskIdentifier,
                    error: ModelInstallerError.downloadFailed("Missing HTTP response for download task")
                )
            }
            return
        }

        guard (200 ..< 300).contains(response.statusCode) else {
            let target = downloadTask.originalRequest?.url?.absoluteString
                ?? downloadTask.currentRequest?.url?.absoluteString
                ?? response.url?.absoluteString
                ?? "unknown URL"
            Task {
                await state.fail(
                    taskIdentifier: downloadTask.taskIdentifier,
                    error: ModelInstallerError.downloadFailed(
                        "Unexpected HTTP status \(response.statusCode) while downloading \(target)"
                    )
                )
            }
            return
        }

        guard let destinationPath = downloadTask.taskDescription else {
            Task {
                await state.fail(
                    taskIdentifier: downloadTask.taskIdentifier,
                    error: ModelInstallerError.downloadFailed("Missing destination path for download task")
                )
            }
            return
        }

        let destinationURL = URL(fileURLWithPath: destinationPath, isDirectory: false)
        do {
            let destinationDirectory = destinationURL.deletingLastPathComponent()
            try FileManager.default.createDirectory(at: destinationDirectory, withIntermediateDirectories: true)
            if FileManager.default.fileExists(atPath: destinationURL.path) {
                try FileManager.default.removeItem(at: destinationURL)
            }
            try FileManager.default.moveItem(at: location, to: destinationURL)
            Task {
                await state.finish(taskIdentifier: downloadTask.taskIdentifier, fileURL: destinationURL)
            }
        } catch {
            Task {
                await state.fail(taskIdentifier: downloadTask.taskIdentifier, error: error)
            }
        }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        guard let error else { return }
        Task {
            await state.fail(taskIdentifier: task.taskIdentifier, error: error)
        }
    }
}
