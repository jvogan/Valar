import Foundation
import SwiftUI
import ValarAudio
import ValarCore
import ValarMLX
import ValarModelKit
import ValarPersistence
import os

struct GoldenCorpusManifest: Codable, Sendable, Equatable {
    let fixtureFamilies: [String]
    let qwenFamilyIdentifiers: [String]
    let qwenPreservedModelIdentifiers: [ModelIdentifier]
    let retiredNamespaces: [String]
    let createdAt: Date
    let modelIdentifiers: [ModelIdentifier]
    let notes: String?

    init(
        fixtureFamilies: [String],
        qwenFamilyIdentifiers: [String] = [],
        qwenPreservedModelIdentifiers: [ModelIdentifier] = [],
        retiredNamespaces: [String] = [],
        createdAt: Date = .now,
        modelIdentifiers: [ModelIdentifier] = [],
        notes: String? = nil
    ) {
        self.fixtureFamilies = fixtureFamilies
        self.qwenFamilyIdentifiers = qwenFamilyIdentifiers
        self.qwenPreservedModelIdentifiers = qwenPreservedModelIdentifiers
        self.retiredNamespaces = retiredNamespaces
        self.createdAt = createdAt
        self.modelIdentifiers = modelIdentifiers
        self.notes = notes
    }
}

struct LegacyImportPlan: Codable, Sendable, Equatable {
    let sourceRoot: String
    let projectsDiscovered: Int
    let voiceAssetsDiscovered: Int
    let bundleCount: Int
    let qwenProjectsDiscovered: Int
    let qwenVoiceAssetsDiscovered: Int
    let qwenModelIdentifiers: [ModelIdentifier]
    let retiredNamespaces: [String]
    let namespaceRemapSummary: String

    init(
        sourceRoot: String,
        projectsDiscovered: Int,
        voiceAssetsDiscovered: Int,
        bundleCount: Int = 0,
        qwenProjectsDiscovered: Int = 0,
        qwenVoiceAssetsDiscovered: Int = 0,
        qwenModelIdentifiers: [ModelIdentifier] = [],
        retiredNamespaces: [String] = [],
        namespaceRemapSummary: String = ""
    ) {
        self.sourceRoot = sourceRoot
        self.projectsDiscovered = projectsDiscovered
        self.voiceAssetsDiscovered = voiceAssetsDiscovered
        self.bundleCount = bundleCount
        self.qwenProjectsDiscovered = qwenProjectsDiscovered
        self.qwenVoiceAssetsDiscovered = qwenVoiceAssetsDiscovered
        self.qwenModelIdentifiers = qwenModelIdentifiers
        self.retiredNamespaces = retiredNamespaces
        self.namespaceRemapSummary = namespaceRemapSummary
    }
}

struct CompatibilityReport: Codable, Sendable, Equatable {
    let preservedModelIdentifiers: [ModelIdentifier]
    let retiredNamespaces: [String]
    let pendingImports: Int
    let qwenModelIdentifiers: [ModelIdentifier]
    let qwenProjectCount: Int
    let qwenVoiceAssetCount: Int
    let namespaceRemapSummary: String

    init(
        preservedModelIdentifiers: [ModelIdentifier],
        retiredNamespaces: [String],
        pendingImports: Int,
        qwenModelIdentifiers: [ModelIdentifier] = [],
        qwenProjectCount: Int = 0,
        qwenVoiceAssetCount: Int = 0,
        namespaceRemapSummary: String = ""
    ) {
        self.preservedModelIdentifiers = preservedModelIdentifiers
        self.retiredNamespaces = retiredNamespaces
        self.pendingImports = pendingImports
        self.qwenModelIdentifiers = qwenModelIdentifiers
        self.qwenProjectCount = qwenProjectCount
        self.qwenVoiceAssetCount = qwenVoiceAssetCount
        self.namespaceRemapSummary = namespaceRemapSummary
    }
}

actor ParityHarness {
    private struct LegacyNamespaceMap: Sendable {
        let sourceNamespace: String
        let targetNamespace: String

        init(
            sourceNamespace: String = "mlx_audio",
            targetNamespace: String = "Valar"
        ) {
            self.sourceNamespace = sourceNamespace
            self.targetNamespace = targetNamespace
        }
    }

    private let projectStore: any ProjectStoring
    private let voiceLibraryStore: any VoiceLibraryStoring
    private var expectedModels: [ModelIdentifier] = []
    private var namespaceMap = LegacyNamespaceMap()

    init(projectStore: any ProjectStoring, voiceLibraryStore: any VoiceLibraryStoring) {
        self.projectStore = projectStore
        self.voiceLibraryStore = voiceLibraryStore
    }

    func registerExpectedModels(_ identifiers: [ModelIdentifier]) {
        expectedModels = identifiers
    }

    func scaffoldGoldenCorpus() -> GoldenCorpusManifest {
        let qwenIdentifiers = expectedModels.filter { $0.rawValue.lowercased().contains("qwen") }
        let fixtureFamilies = expectedModels.map { $0.inferredFamilyHint.rawValue }.sorted()
        let qwenFamilies = Array(Set(qwenIdentifiers.map { $0.inferredFamilyHint.rawValue })).sorted()
        return GoldenCorpusManifest(
            fixtureFamilies: fixtureFamilies,
            qwenFamilyIdentifiers: qwenFamilies,
            qwenPreservedModelIdentifiers: qwenIdentifiers.sorted { $0.rawValue < $1.rawValue },
            retiredNamespaces: [namespaceMap.sourceNamespace],
            modelIdentifiers: expectedModels.sorted { $0.rawValue < $1.rawValue },
            notes: "Qwen-first golden corpus preserved for parity, import, and migration."
        )
    }

    func planLegacyImport(from sourceRoot: String) async -> LegacyImportPlan {
        let projects = await projectStore.allProjects().count
        let voices = await voiceLibraryStore.list().count
        let qwenIdentifiers = expectedModels.filter { $0.rawValue.lowercased().contains("qwen") }
        return LegacyImportPlan(
            sourceRoot: sourceRoot,
            projectsDiscovered: projects,
            voiceAssetsDiscovered: voices,
            bundleCount: projects,
            qwenProjectsDiscovered: projects,
            qwenVoiceAssetsDiscovered: voices,
            qwenModelIdentifiers: qwenIdentifiers.sorted { $0.rawValue < $1.rawValue },
            retiredNamespaces: [namespaceMap.sourceNamespace],
            namespaceRemapSummary: "\(namespaceMap.sourceNamespace) -> \(namespaceMap.targetNamespace)"
        )
    }

    func compatibilityReport() async -> CompatibilityReport {
        let projects = await projectStore.allProjects().count
        let voices = await voiceLibraryStore.list().count
        let qwenIdentifiers = expectedModels.filter { $0.rawValue.lowercased().contains("qwen") }
        return CompatibilityReport(
            preservedModelIdentifiers: expectedModels,
            retiredNamespaces: [namespaceMap.sourceNamespace],
            pendingImports: projects + voices,
            qwenModelIdentifiers: qwenIdentifiers.sorted { $0.rawValue < $1.rawValue },
            qwenProjectCount: projects,
            qwenVoiceAssetCount: voices,
            namespaceRemapSummary: "\(namespaceMap.sourceNamespace) -> \(namespaceMap.targetNamespace)"
        )
    }
}

actor GRDBBackedProjectStore: ProjectStoring {
    private static let logger = Logger(subsystem: "com.valar.tts", category: "ProjectStore")
    private let store: GRDBProjectStore
    private var chapters: [UUID: [ChapterRecord]] = [:]
    private var renders: [UUID: [RenderJobRecord]] = [:]
    private var exports: [UUID: [ExportRecord]] = [:]
    private var speakers: [UUID: [ProjectSpeakerRecord]] = [:]
    private var bundleURLs: [UUID: URL] = [:]

    init(store: GRDBProjectStore) {
        self.store = store
    }

    func create(title: String, notes: String?) async throws -> ProjectRecord {
        let project = try await store.insert(title: title, notes: notes)
        chapters[project.id] = []
        renders[project.id] = []
        exports[project.id] = []
        speakers[project.id] = []
        return project
    }

    func update(_ project: ProjectRecord) async {
        do {
            try await store.update(project)
        } catch {
            Self.logger.error("Failed to update project \(project.id.uuidString, privacy: .public): \(error.localizedDescription, privacy: .private)")
        }
    }

    func addChapter(_ chapter: ChapterRecord) async {
        do {
            try await store.insert(chapter)
        } catch {
            Self.logger.error("Failed to add chapter \(chapter.id.uuidString, privacy: .public): \(error.localizedDescription, privacy: .private)")
            return
        }
        if chapters[chapter.projectID] != nil {
            cacheChapter(chapter)
        }
    }

    func updateChapter(_ chapter: ChapterRecord) async {
        do {
            let persistedChapters = try await store.chapters(for: chapter.projectID)
            guard persistedChapters.contains(where: { $0.id == chapter.id }) else {
                return
            }
            try await store.update(chapter)
        } catch {
            Self.logger.error("Failed to update chapter \(chapter.id.uuidString, privacy: .public): \(error.localizedDescription, privacy: .private)")
            return
        }
        if chapters[chapter.projectID] != nil {
            cacheChapter(chapter)
        }
    }

    func removeChapter(_ id: UUID, from projectID: UUID) async {
        do {
            try await store.deleteChapter(id, in: projectID)
        } catch {
            Self.logger.error("Failed to remove chapter \(id.uuidString, privacy: .public): \(error.localizedDescription, privacy: .private)")
            return
        }
        chapters[projectID]?.removeAll { $0.id == id }
    }

    func addRenderJob(_ job: RenderJobRecord) async {
        var records = renders[job.projectID, default: []]
        if let existingIndex = records.firstIndex(where: { $0.id == job.id }) {
            records[existingIndex] = job
        } else {
            records.append(job)
        }
        records.sort { $0.createdAt < $1.createdAt }
        renders[job.projectID] = records
    }

    func addExport(_ export: ExportRecord) async {
        var records = exports[export.projectID, default: []]
        if let existingIndex = records.firstIndex(where: { $0.id == export.id }) {
            records[existingIndex] = export
        } else {
            records.append(export)
        }
        records.sort { $0.createdAt < $1.createdAt }
        exports[export.projectID] = records

        do {
            try await store.insert(export)
        } catch {
            Self.logger.error("Failed to add export \(export.id.uuidString, privacy: .public): \(error.localizedDescription, privacy: .private)")
        }
    }

    func addSpeaker(_ speaker: ProjectSpeakerRecord) async {
        do {
            try await store.save(speaker)
        } catch {
            Self.logger.error("Failed to add speaker \(speaker.id.uuidString, privacy: .public): \(error.localizedDescription, privacy: .private)")
            return
        }
        if speakers[speaker.projectID] != nil {
            cacheSpeaker(speaker)
        }
    }

    func removeSpeaker(_ id: UUID, from projectID: UUID) async {
        do {
            try await store.deleteSpeaker(id)
        } catch {
            Self.logger.error("Failed to remove speaker \(id.uuidString, privacy: .public) from project \(projectID.uuidString, privacy: .public): \(error.localizedDescription, privacy: .private)")
            return
        }
        if speakers[projectID] != nil {
            speakers[projectID]?.removeAll { $0.id == id }
        }
    }

    func restore(
        project: ProjectRecord,
        chapters: [ChapterRecord],
        renderJobs: [RenderJobRecord],
        exports: [ExportRecord],
        speakers: [ProjectSpeakerRecord]
    ) async {
        do {
            try await store.save(project)
            try await store.replaceChapters(for: project.id, with: chapters)
        } catch {
            Self.logger.error("Failed to restore project \(project.id.uuidString, privacy: .public): \(error.localizedDescription, privacy: .private)")
            return
        }
        self.chapters[project.id] = chapters.sorted { $0.index < $1.index }
        self.renders[project.id] = renderJobs.sorted { $0.createdAt < $1.createdAt }
        self.exports[project.id] = exports.sorted { $0.createdAt < $1.createdAt }
        self.speakers[project.id] = speakers
    }

    func allProjects() async -> [ProjectRecord] {
        do {
            return try await store.fetchAll()
        } catch {
            Self.logger.error("Failed to load projects: \(error.localizedDescription, privacy: .private)")
            return []
        }
    }

    func chapters(for projectID: UUID) async -> [ChapterRecord] {
        if let cachedChapters = chapters[projectID] {
            return cachedChapters
        }

        do {
            let persistedChapters = try await store.chapters(for: projectID)
            chapters[projectID] = persistedChapters
            return persistedChapters
        } catch {
            Self.logger.error("Failed to load chapters for project \(projectID.uuidString, privacy: .public): \(error.localizedDescription, privacy: .private)")
            return []
        }
    }

    func renderJobs(for projectID: UUID) async -> [RenderJobRecord] {
        renders[projectID, default: []]
    }

    func exports(for projectID: UUID) async -> [ExportRecord] {
        if let cachedExports = exports[projectID] {
            return cachedExports
        }

        do {
            let persistedExports = try await store.exports(for: projectID)
            exports[projectID] = persistedExports
            return persistedExports
        } catch {
            Self.logger.error("Failed to load exports for project \(projectID.uuidString, privacy: .public): \(error.localizedDescription, privacy: .private)")
            return []
        }
    }

    func speakers(for projectID: UUID) async -> [ProjectSpeakerRecord] {
        if let cachedSpeakers = speakers[projectID], cachedSpeakers.isEmpty == false {
            return cachedSpeakers
        }

        do {
            let persistedSpeakers = try await store.speakers(for: projectID)
            speakers[projectID] = persistedSpeakers
            return persistedSpeakers
        } catch {
            Self.logger.error("Failed to load speakers for project \(projectID.uuidString, privacy: .public): \(error.localizedDescription, privacy: .private)")
            return []
        }
    }

    func bundleLocation(for projectID: UUID) async -> ValarProjectBundleLocation? {
        do {
            guard let bundleURL = bundleURLs[projectID],
                  let project = try await store.project(id: projectID) else {
                return nil
            }
            return ValarProjectBundleLocation(
                projectID: projectID,
                title: project.title,
                bundleURL: bundleURL
            )
        } catch {
            Self.logger.error("Failed to resolve bundle location for project \(projectID.uuidString, privacy: .public): \(error.localizedDescription, privacy: .private)")
            return nil
        }
    }

    func updateBundleURL(_ bundleURL: URL?, for projectID: UUID) {
        bundleURLs[projectID] = bundleURL
    }

    func remove(id: UUID) async {
        do {
            try await store.delete(id)
        } catch {
            Self.logger.error("Failed to remove project \(id.uuidString, privacy: .public): \(error.localizedDescription, privacy: .private)")
        }
        chapters[id] = nil
        renders[id] = nil
        exports[id] = nil
        speakers[id] = nil
        bundleURLs[id] = nil
    }

    private func cacheChapter(_ chapter: ChapterRecord) {
        var records = chapters[chapter.projectID, default: []]
        if let existingIndex = records.firstIndex(where: { $0.id == chapter.id }) {
            records[existingIndex] = chapter
        } else {
            records.append(chapter)
        }
        records.sort { $0.index < $1.index }
        chapters[chapter.projectID] = records
    }

    private func cacheSpeaker(_ speaker: ProjectSpeakerRecord) {
        var records = speakers[speaker.projectID, default: []]
        if let existingIndex = records.firstIndex(where: { $0.id == speaker.id }) {
            records[existingIndex] = speaker
        } else {
            records.append(speaker)
        }
        speakers[speaker.projectID] = records
    }
}

actor GRDBBackedVoiceLibraryStore: VoiceLibraryStoring {
    private static let logger = Logger(subsystem: "com.valar.tts", category: "VoiceLibraryStore")
    private let store: GRDBVoiceStore

    init(store: GRDBVoiceStore) {
        self.store = store
    }

    func save(_ voice: VoiceLibraryRecord) async throws -> VoiceLibraryRecord {
        return try await store.insert(voice)
    }

    func list() async -> [VoiceLibraryRecord] {
        do {
            return try await store.fetchAll()
        } catch {
            Self.logger.error("Failed to list voices: \(error.localizedDescription, privacy: .private)")
            return []
        }
    }

    func delete(_ id: UUID) async throws {
        try await store.delete(id)
    }
}

actor GRDBBackedModelPackRegistry: ModelPackManaging {
    private let paths: ValarAppPaths
    private let store: GRDBModelPackStore

    init(store: GRDBModelPackStore, paths: ValarAppPaths) {
        self.store = store
        self.paths = paths
    }

    func registerSupported(_ record: SupportedModelCatalogRecord) async throws {
        try await store.saveCatalogEntry(
            SupportedModelCatalogRecord(
                id: record.modelID,
                familyID: record.familyID,
                modelID: record.modelID,
                displayName: record.displayName,
                providerName: record.providerName,
                providerURL: record.providerURL,
                installHint: record.installHint,
                sourceKind: record.sourceKind,
                isRecommended: record.isRecommended
            )
        )

        guard try await store.manifest(for: record.modelID) == nil else { return }

        try await store.saveManifest(
            ValarPersistence.ModelPackManifest(
                id: record.modelID,
                familyID: record.familyID,
                modelID: record.modelID,
                displayName: record.displayName,
                notes: record.installHint
            )
        )
    }

    func install(
        manifest: ValarPersistence.ModelPackManifest,
        sourceKind: ModelPackSourceKind,
        sourceLocation: String,
        notes: String?
    ) async throws -> ModelInstallReceipt {
        let packDirectory = try paths.modelPackDirectory(familyID: manifest.familyID, modelID: manifest.modelID)
        let manifestURL = try paths.modelPackManifestURL(familyID: manifest.familyID, modelID: manifest.modelID)
        let receipt = ModelInstallReceipt(
            id: manifest.modelID,
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
            id: manifest.modelID,
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

        try await store.saveManifest(manifest)
        try await store.saveInstalledRecord(record)
        try await store.saveReceipt(receipt)
        try await store.saveCatalogEntry(
            SupportedModelCatalogRecord(
                id: manifest.modelID,
                familyID: manifest.familyID,
                modelID: manifest.modelID,
                displayName: manifest.displayName,
                providerName: "Valar",
                providerURL: nil,
                installHint: notes,
                sourceKind: sourceKind,
                isRecommended: manifest.familyID == ModelFamilyID.qwen3TTS.rawValue
            )
        )
        try await store.saveLedgerEntry(
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

    func uninstall(modelID: String) async throws -> InstalledModelRecord? {
        try await store.uninstall(modelID: modelID)
    }

    func manifest(for modelID: String) async throws -> ValarPersistence.ModelPackManifest? {
        try await store.manifest(for: modelID)
    }

    func installedRecord(for modelID: String) async throws -> InstalledModelRecord? {
        try await store.installedRecord(for: modelID)
    }

    func receipts() async throws -> [ModelInstallReceipt] {
        try await store.receipts()
    }

    func supportedModels() async throws -> [SupportedModelCatalogRecord] {
        try await store.supportedModels()
    }

    func supportedModel(for modelID: String) async throws -> SupportedModelCatalogRecord? {
        try await store.supportedModel(for: modelID)
    }

    func ledgerEntries() async throws -> [ModelInstallLedgerEntry] {
        try await store.ledgerEntries()
    }
}

struct ValarDashboardSnapshot: Equatable {
    let catalogModels: [CatalogModel]
    let modelSnapshots: [ModelResidencySnapshot]
    let projects: [ProjectRecord]
    let voices: [VoiceLibraryRecord]
    let renderJobs: [RenderJob]
    let goldenCorpus: GoldenCorpusManifest
    let legacyImportPlan: LegacyImportPlan
    let compatibilityReport: CompatibilityReport
    let runtimeConfiguration: RuntimeConfiguration
    let appPaths: ValarAppPaths
    let lastUpdatedAt: Date

    static let empty = ValarDashboardSnapshot(
        catalogModels: [],
        modelSnapshots: [],
        projects: [],
        voices: [],
        renderJobs: [],
        goldenCorpus: GoldenCorpusManifest(fixtureFamilies: []),
        legacyImportPlan: LegacyImportPlan(sourceRoot: "", projectsDiscovered: 0, voiceAssetsDiscovered: 0),
        compatibilityReport: CompatibilityReport(
            preservedModelIdentifiers: [],
            retiredNamespaces: [],
            pendingImports: 0
        ),
        runtimeConfiguration: RuntimeConfiguration(),
        appPaths: ValarAppPaths(),
        lastUpdatedAt: Date()
    )

    var modelCount: Int { catalogModels.count }
    var installedModelCount: Int { installedCatalogModels.count }
    var cachedModelCount: Int { cachedCatalogModels.count }
    var recommendedModelCount: Int { recommendedModels.count }
    var projectCount: Int { projects.count }
    var voiceCount: Int { voices.count }
    var jobCount: Int { renderJobs.count }

    var installedCatalogModels: [CatalogModel] {
        catalogModels.filter { $0.installState == .installed }
    }

    var cachedCatalogModels: [CatalogModel] {
        catalogModels.filter { $0.installState == .cached }
    }

    var recommendedModels: [CatalogModel] {
        catalogModels.filter(\.isRecommended)
    }

    var availableGenerationModels: [CatalogModel] {
        let installed = installedCatalogModels.filter {
            $0.descriptor.capabilities.contains(.speechSynthesis)
        }
        if !installed.isEmpty {
            return installed
        }
        return catalogModels.filter { $0.descriptor.capabilities.contains(.speechSynthesis) }
    }

    var availableRecognitionModels: [CatalogModel] {
        let installed = installedCatalogModels.filter {
            $0.descriptor.capabilities.contains(.speechRecognition)
        }
        if !installed.isEmpty {
            return installed
        }
        return catalogModels.filter { $0.descriptor.capabilities.contains(.speechRecognition) }
    }

    var ttsModels: [ModelResidencySnapshot] {
        modelSnapshots.filter { $0.descriptor.domain == .tts }
    }

    var sttModels: [ModelResidencySnapshot] {
        modelSnapshots.filter { $0.descriptor.domain == .stt }
    }

    func catalogModel(for identifier: ModelIdentifier?) -> CatalogModel? {
        guard let identifier else { return nil }
        return catalogModels.first(where: { $0.id == identifier })
    }
}

struct RuntimeModelPickerOption: Identifiable, Equatable {
    let id: ModelIdentifier
    let displayName: String
    let familyID: ModelFamilyID
    let voiceFeatures: [ModelVoiceFeature]
    let isRecommended: Bool
    let supportTier: ModelSupportTier
    let distributionTier: ModelDistributionTier

    init(
        id: ModelIdentifier,
        displayName: String,
        familyID: ModelFamilyID,
        voiceFeatures: [ModelVoiceFeature],
        isRecommended: Bool,
        supportTier: ModelSupportTier = .supported,
        distributionTier: ModelDistributionTier = .optionalInstall
    ) {
        self.id = id
        self.displayName = displayName
        self.familyID = familyID
        self.voiceFeatures = voiceFeatures
        self.isRecommended = isRecommended
        self.supportTier = supportTier
        self.distributionTier = distributionTier
    }

    var supportsReferenceAudio: Bool {
        ModelVoiceSupport(features: voiceFeatures).supportsReferenceAudio
    }

    var selectionPriority: Int {
        let recommendationBias = isRecommended ? 0 : 100
        let distributionBias: Int = switch distributionTier {
        case .bundledFirstRun: 0
        case .optionalInstall: 10
        case .compatibilityPreview: 20
        }
        let supportBias: Int = switch supportTier {
        case .supported: 0
        case .preview: 1
        case .experimental: 2
        }
        return recommendationBias + distributionBias + supportBias
    }
}

enum ProjectChapterAudioError: LocalizedError {
    case missingProjectBundleStorage(UUID)
    case missingSourceAudio(UUID)
    case sourceAudioAssetNotFound(String)
    case emptyAlignmentTranscript

    var errorDescription: String? {
        switch self {
        case .missingProjectBundleStorage:
            return "Save the project document before attaching or processing chapter source audio."
        case .missingSourceAudio:
            return "Attach source audio to the chapter before transcribing or aligning it."
        case let .sourceAudioAssetNotFound(assetName):
            return "The chapter source audio asset '\(assetName)' could not be found."
        case .emptyAlignmentTranscript:
            return "Enter chapter text or transcribe the source audio before aligning."
        }
    }
}

protocol AudioPlaying: Sendable {
    func play(_ buffer: AudioPCMBuffer) async throws
    func feedChunk(_ buffer: AudioPCMBuffer) async throws
    func feedSamples(_ samples: [Float], sampleRate: Double) async throws
    func finishStream() async
    func playbackSnapshot() async -> AudioPlaybackSnapshot
    func stop() async
}

actor AudioPlayerService: AudioPlaying {
    private let player: AudioEnginePlayer

    init(player: AudioEnginePlayer = AudioEnginePlayer()) {
        self.player = player
    }

    func play(_ buffer: AudioPCMBuffer) async throws {
        try await player.play(buffer)
    }

    func feedChunk(_ buffer: AudioPCMBuffer) async throws {
        try await player.feedChunk(buffer)
    }

    func feedSamples(_ samples: [Float], sampleRate: Double) async throws {
        try await player.feedSamples(samples, sampleRate: sampleRate)
    }

    func finishStream() async {
        await player.finishStream()
    }

    func playbackSnapshot() async -> AudioPlaybackSnapshot {
        await player.playbackSnapshot()
    }

    func stop() async {
        await player.stop()
    }
}

actor SilentAudioPlayer: AudioPlaying {
    func play(_ buffer: AudioPCMBuffer) async throws {}
    func feedChunk(_ buffer: AudioPCMBuffer) async throws {}
    func feedSamples(_ samples: [Float], sampleRate: Double) async throws {}
    func finishStream() async {}
    func playbackSnapshot() async -> AudioPlaybackSnapshot {
        AudioPlaybackSnapshot(position: 0, queuedDuration: 0, isPlaying: false, isBuffering: false, didFinish: true)
    }
    func stop() async {}
}

extension ValarRuntime {
    private static let diagnosticsLogger = Logger(subsystem: "com.valar.tts", category: "RuntimeDiagnostics")

    static var appDefaultConfiguration: RuntimeConfiguration {
        RuntimeConfiguration(
            warmPolicy: .lazy,
            maxResidentBytes: 10 * 1_024 * 1_024 * 1_024,
            maxResidentModels: 2,
            maxQueuedRenderJobs: 16
        )
    }

    static func live(appPaths: ValarAppPaths = ValarAppPaths()) throws -> ValarRuntime {
        try ValarRuntime.live(
            paths: appPaths,
            runtimeConfiguration: appDefaultConfiguration,
            makeInferenceBackend: { MLXInferenceBackend() }
        )
    }

    func generationModelOptions() async -> [RuntimeModelPickerOption] {
        _ = await ensureStartupMaintenance()
        let catalogModels = (try? await modelCatalog.supportedModels()) ?? []
        let catalogOptions = catalogModels
            .filter { $0.installState == .installed }
            .filter { $0.descriptor.capabilities.contains(.speechSynthesis) }
            .map { model in
                let voiceSupport = model.descriptor.voiceSupport
                return RuntimeModelPickerOption(
                    id: model.id,
                    displayName: model.descriptor.displayName,
                    familyID: model.familyID,
                    voiceFeatures: voiceSupport.features,
                    isRecommended: model.isRecommended,
                    supportTier: model.supportTier,
                    distributionTier: model.distributionTier
                )
            }

        var optionsByID = Dictionary(uniqueKeysWithValues: catalogOptions.map { ($0.id, $0) })
        let runtimeSnapshots = await modelRegistry.snapshots()
        for snapshot in runtimeSnapshots where snapshot.descriptor.capabilities.contains(.speechSynthesis) {
            guard optionsByID[snapshot.descriptor.id] == nil else {
                continue
            }

            let descriptor = snapshot.descriptor
            let voiceSupport = descriptor.voiceSupport
            optionsByID[descriptor.id] = RuntimeModelPickerOption(
                id: descriptor.id,
                displayName: descriptor.displayName,
                familyID: descriptor.familyID,
                voiceFeatures: voiceSupport.features,
                isRecommended: false
            )
        }

        return optionsByID.values.sorted { lhs, rhs in
            if lhs.selectionPriority != rhs.selectionPriority {
                return lhs.selectionPriority < rhs.selectionPriority
            }
            return lhs.displayName.localizedCaseInsensitiveCompare(rhs.displayName) == .orderedAscending
        }
    }

    func storedVoices() async -> [VoiceLibraryRecord] {
        _ = await ensureStartupMaintenance()
        return await voiceStore
            .list()
            .sorted { $0.label.localizedCaseInsensitiveCompare($1.label) == .orderedAscending }
    }

    func dashboardSnapshot(
        parityHarness: ParityHarness? = nil,
        includeParityData: Bool = false
    ) async -> ValarDashboardSnapshot {
        _ = await ensureStartupMaintenance()
        let catalogModels: [CatalogModel]
        do {
            catalogModels = try await modelCatalog.refresh()
        } catch {
            Self.diagnosticsLogger.error("Failed to load runtime catalog models for diagnostics: \(error.localizedDescription, privacy: .private)")
            catalogModels = []
        }

        let modelSnapshots = await modelRegistry.snapshots()
        let projects = await projectStore.allProjects()
        let voices = await storedVoices()

        let renderJobs: [RenderJob]
        do {
            renderJobs = try await grdbRenderJobStore.loadJobs().map(Self.makeRenderJob(from:))
        } catch {
            Self.diagnosticsLogger.error("Failed to load render queue diagnostics: \(error.localizedDescription, privacy: .private)")
            renderJobs = []
        }

        let paritySnapshot: ValarServiceHub.ParitySnapshotData
        if includeParityData, let parityHarness {
            paritySnapshot = await ValarServiceHub.makeParitySnapshotData(
                harness: parityHarness,
                appPaths: paths,
                catalogModels: catalogModels
            )
        } else {
            paritySnapshot = .empty
        }

        return ValarDashboardSnapshot(
            catalogModels: catalogModels,
            modelSnapshots: modelSnapshots,
            projects: projects,
            voices: voices,
            renderJobs: renderJobs,
            goldenCorpus: paritySnapshot.goldenCorpus,
            legacyImportPlan: paritySnapshot.legacyImportPlan,
            compatibilityReport: paritySnapshot.compatibilityReport,
            runtimeConfiguration: runtimeConfiguration,
            appPaths: paths,
            lastUpdatedAt: Date()
        )
    }

    func diagnostics(
        parityHarness: ParityHarness? = nil,
        includeParityData: Bool = false
    ) async -> ValarDashboardSnapshot {
        await dashboardSnapshot(
            parityHarness: parityHarness,
            includeParityData: includeParityData
        )
    }

    private static func makeRenderJob(from record: RenderJobRecord) -> RenderJob {
        RenderJob(
            id: record.id,
            projectID: record.projectID,
            modelID: ModelIdentifier(record.modelID),
            chapterIDs: record.chapterIDs,
            outputFileName: record.outputFileName,
            createdAt: record.createdAt,
            state: RenderJobState(rawValue: record.state) ?? .queued,
            priority: record.priority,
            progress: record.progress,
            title: record.title,
            synthesisOptions: record.synthesisOptions
        )
    }
}

@MainActor
final class ValarServiceHub {
    typealias ModelsReadyObserver = @MainActor @Sendable () async -> Void

    private enum CatalogSeedAction: Sendable {
        case bootstrapAndInstall
        case reseedMetadataOnly
        case skipSeeding
    }

    fileprivate struct ParitySnapshotData {
        let goldenCorpus: GoldenCorpusManifest
        let legacyImportPlan: LegacyImportPlan
        let compatibilityReport: CompatibilityReport

        static let empty = ParitySnapshotData(
            goldenCorpus: GoldenCorpusManifest(fixtureFamilies: []),
            legacyImportPlan: LegacyImportPlan(sourceRoot: "", projectsDiscovered: 0, voiceAssetsDiscovered: 0),
            compatibilityReport: CompatibilityReport(
                preservedModelIdentifiers: [],
                retiredNamespaces: [],
                pendingImports: 0
            )
        )
    }

    private static let logger = Logger(subsystem: "com.valar.tts", category: "ModelInstall")
    private nonisolated static let bootstrapVersion = 2
    private nonisolated static let catalogSeedVersionKey = "catalogSeedVersion"

    static let qwenCustomVoiceModelID: ModelIdentifier = "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16"
    static let qwenVoiceDesignModelID: ModelIdentifier = "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16"
    let runtime: ValarRuntime
    let projectRenderService: ProjectRenderService
    let dictationService: DictationService
    let translationService: TranslationService
    let parityHarness: ParityHarness?
    let audioPlayer: any AudioPlaying
    let projectExporter: any ProjectAudioExporting
    var appPaths: ValarAppPaths { runtime.paths }
    var runtimeConfiguration: RuntimeConfiguration { runtime.runtimeConfiguration }
    var modelRegistry: ModelRegistry { runtime.modelRegistry }
    var capabilityRegistry: CapabilityRegistry { runtime.capabilityRegistry }
    var modelPackRegistry: any ModelPackManaging { runtime.modelPackRegistry }
    var modelCatalog: ModelCatalog { runtime.modelCatalog }
    var modelInstaller: ModelInstaller { runtime.modelInstaller }
    var inferenceBackend: any InferenceBackend { runtime.inferenceBackend }
    var renderQueue: RenderQueue { runtime.renderQueue }
    var grdbRenderJobStore: GRDBRenderJobStore { runtime.grdbRenderJobStore }
    var grdbProjectStore: GRDBProjectStore { runtime.grdbProjectStore }
    var projectStore: any ProjectStoring { runtime.projectStore }
    var grdbVoiceStore: GRDBVoiceStore { runtime.grdbVoiceStore }
    var voiceLibraryStore: any VoiceLibraryStoring { runtime.voiceStore }
    var audioPipeline: AudioPipeline { runtime.audioPipeline }
    var projectCreationErrorMessage: String?
    var onModelsReady: ModelsReadyObserver? {
        didSet {
            if let legacyModelsReadyObserverID {
                modelsReadyObservers.removeValue(forKey: legacyModelsReadyObserverID)
                self.legacyModelsReadyObserverID = nil
            }
            guard let onModelsReady else { return }
            legacyModelsReadyObserverID = registerModelsReadyObserver(onModelsReady)
        }
    }
    private var modelsReadyObservers: [UUID: ModelsReadyObserver] = [:]
    private var legacyModelsReadyObserverID: UUID?
    private var didFinishModelsReady = false
    private var modelsReadyTask: Task<Void, Never>?

    init(
        runtime: ValarRuntime,
        projectRenderService: ProjectRenderService,
        dictationService: DictationService,
        translationService: TranslationService,
        parityHarness: ParityHarness? = nil,
        audioPlayer: any AudioPlaying,
        projectExporter: any ProjectAudioExporting
    ) {
        self.runtime = runtime
        self.projectRenderService = projectRenderService
        self.dictationService = dictationService
        self.translationService = translationService
        self.parityHarness = parityHarness
        self.audioPlayer = audioPlayer
        self.projectExporter = projectExporter
    }

    deinit {
        modelsReadyTask?.cancel()
    }

    convenience init(
        appPaths: ValarAppPaths,
        runtimeConfiguration: RuntimeConfiguration,
        modelRegistry: ModelRegistry,
        capabilityRegistry: CapabilityRegistry,
        modelPackRegistry: any ModelPackManaging,
        modelCatalog: ModelCatalog,
        modelInstaller: ModelInstaller,
        inferenceBackend: any InferenceBackend,
        renderQueue: RenderQueue,
        grdbRenderJobStore: GRDBRenderJobStore,
        projectRenderService: ProjectRenderService,
        grdbProjectStore: GRDBProjectStore,
        projectStore: any ProjectStoring,
        grdbVoiceStore: GRDBVoiceStore,
        voiceLibraryStore: any VoiceLibraryStoring,
        dictationService: DictationService,
        translationService: TranslationService,
        parityHarness: ParityHarness? = nil,
        audioPipeline: AudioPipeline,
        audioPlayer: any AudioPlaying,
        projectExporter: any ProjectAudioExporting
    ) {
        let runtime = ValarRuntime(
            paths: appPaths,
            runtimeConfiguration: runtimeConfiguration,
            database: Self.makeInjectedRuntimeDatabase(),
            grdbProjectStore: grdbProjectStore,
            grdbVoiceStore: grdbVoiceStore,
            grdbRenderJobStore: grdbRenderJobStore,
            modelRegistry: modelRegistry,
            capabilityRegistry: capabilityRegistry,
            modelPackRegistry: modelPackRegistry,
            modelCatalog: modelCatalog,
            modelInstaller: modelInstaller,
            inferenceBackend: inferenceBackend,
            audioPipeline: audioPipeline,
            renderQueue: renderQueue,
            projectStore: projectStore,
            voiceStore: voiceLibraryStore
        )
        self.init(
            runtime: runtime,
            projectRenderService: projectRenderService,
            dictationService: dictationService,
            translationService: translationService,
            parityHarness: parityHarness,
            audioPlayer: audioPlayer,
            projectExporter: projectExporter
        )
    }

    private static func makeInjectedRuntimeDatabase() -> AppDatabase {
        do {
            return try AppDatabase.inMemory()
        } catch {
            fatalError("Unable to initialize in-memory AppDatabase for injected ValarServiceHub runtime: \(error)")
        }
    }

    static func live(appPaths: ValarAppPaths = ValarAppPaths()) -> ValarServiceHub {
        let runtime: ValarRuntime
        do {
            runtime = try ValarRuntime.live(appPaths: appPaths)
        } catch {
            fatalError("Unable to initialize AppDatabase at \(PathRedaction.redact(appPaths.databaseURL.path)): \(error)")
        }
        let dictationService = DictationService()
        let translationService = TranslationService(provider: AppTranslator())
        let parityHarness: ParityHarness?
#if DEBUG
        parityHarness = ParityHarness(projectStore: runtime.projectStore, voiceLibraryStore: runtime.voiceStore)
#else
        parityHarness = nil
#endif
        let audioPlayer: any AudioPlaying = if Self.isRunningUnitTests {
            SilentAudioPlayer()
        } else {
            AudioPlayerService()
        }
        let synthesizeChapter: @Sendable (ModelIdentifier, RenderSynthesisOptions, String) async throws -> AudioChunk = { modelID, options, text in
            let descriptor = try await Self.resolveDescriptor(
                for: modelID,
                modelRegistry: runtime.modelRegistry,
                capabilityRegistry: runtime.capabilityRegistry,
                modelCatalog: runtime.modelCatalog
            )
            let backendRuntime = BackendSelectionPolicy.Runtime(
                availableBackends: [runtime.inferenceBackend.backendKind]
            )
            let configuration = try BackendSelectionPolicy().runtimeConfiguration(
                for: descriptor,
                residencyPolicy: .automatic,
                runtime: backendRuntime
            )
            let request = SpeechSynthesisRequest(
                model: modelID,
                text: text,
                language: options.normalizedLanguage,
                sampleRate: descriptor.defaultSampleRate ?? 24_000,
                responseFormat: "pcm_f32le",
                temperature: options.temperature.map(Float.init),
                topP: options.topP.map(Float.init),
                repetitionPenalty: options.repetitionPenalty.map(Float.init),
                maxTokens: options.maxTokens,
                voiceBehavior: options.voiceBehavior
            )
            return try await runtime.withReservedTextToSpeechWorkflowSession(
                descriptor: descriptor,
                configuration: configuration
            ) { reserved in
                try await reserved.workflow.synthesize(request: request, in: reserved.session)
            }
        }
        let projectRenderService = ProjectRenderService(
            renderQueue: runtime.renderQueue,
            projectStore: runtime.projectStore,
            audioPipeline: runtime.audioPipeline,
            synthesizeChapter: synthesizeChapter
        )
        let projectExporter = ProjectAudioExportCoordinator(
            projectStore: runtime.projectStore,
            audioPipeline: runtime.audioPipeline,
            synthesizeChapter: synthesizeChapter
        )

        return ValarServiceHub(
            runtime: runtime,
            projectRenderService: projectRenderService,
            dictationService: dictationService,
            translationService: translationService,
            parityHarness: parityHarness,
            audioPlayer: audioPlayer,
            projectExporter: projectExporter
        )
    }

    private static var isRunningUnitTests: Bool {
        NSClassFromString("XCTestCase") != nil
            || ProcessInfo.processInfo.environment["XCTestConfigurationFilePath"] != nil
    }

    func snapshot() async -> ValarDashboardSnapshot {
        await runtime.dashboardSnapshot()
    }

    func diagnosticsSnapshot() async -> ValarDashboardSnapshot {
#if DEBUG
        await runtime.diagnostics(parityHarness: parityHarness, includeParityData: true)
#else
        await runtime.diagnostics()
#endif
    }

    fileprivate static func makeParitySnapshotData(
        harness: ParityHarness,
        appPaths: ValarAppPaths,
        catalogModels: [CatalogModel]
    ) async -> ParitySnapshotData {
        await harness.registerExpectedModels(catalogModels.map(\.id))
        let goldenCorpus = await harness.scaffoldGoldenCorpus()
        let legacyImportPlan = await harness.planLegacyImport(from: appPaths.applicationSupport.path)
        let compatibilityReport = await harness.compatibilityReport()
        return ParitySnapshotData(
            goldenCorpus: goldenCorpus,
            legacyImportPlan: legacyImportPlan,
            compatibilityReport: compatibilityReport
        )
    }

    func createProject(title: String) async -> ProjectRecord? {
        projectCreationErrorMessage = nil

        do {
            try ValarAppPaths.validateRelativePath(title, label: "project title")
            return try await projectStore.create(title: title, notes: nil)
        } catch {
            let debugMessage = (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
            projectCreationErrorMessage = "Unable to create project. Please check the title and try again."
            Self.logger.error("Failed to create project: \(debugMessage, privacy: .private)")
            return nil
        }
    }

    func updateProjectBundleURL(_ bundleURL: URL?, for projectID: UUID) async {
        await projectStore.updateBundleURL(bundleURL, for: projectID)
    }

    func deleteProject(_ projectID: UUID) async throws {
        let bundleLocation = await projectStore.bundleLocation(for: projectID)
        let projectJobs = await renderQueue.jobs(matching: nil).filter { $0.projectID == projectID }

        for job in projectJobs {
            await projectRenderService.cancel(job.id)
        }

        if let bundleURL = bundleLocation?.bundleURL {
            try removeProjectBundleIfPresent(at: bundleURL)
        }

        await renderQueue.replaceJobs(for: projectID, with: [])
        try await grdbProjectStore.delete(projectID)
        await projectStore.remove(id: projectID)
    }

    func saveProjectBundle(
        projectID: UUID,
        to bundleURL: URL,
        modelID: ModelIdentifier?,
        synthesisOptions: RenderSynthesisOptions = RenderSynthesisOptions()
    ) async throws -> ValarProjectBundleLocation {
        guard let project = await projectStore.allProjects().first(where: { $0.id == projectID }) else {
            throw ProjectBundleError.projectNotFound(projectID)
        }

        let storedRenderJobs = await projectStore.renderJobs(for: projectID)
        let queuedRenderJobs = await renderQueue.jobs(matching: nil).filter { $0.projectID == projectID }
        var renderJobsByID = Dictionary(uniqueKeysWithValues: storedRenderJobs.map { ($0.id, $0) })

        for job in queuedRenderJobs {
            let chapterIDs = renderJobsByID[job.id]?.chapterIDs ?? []
            renderJobsByID[job.id] = makeRenderJobRecord(from: job, chapterIDs: chapterIDs)
        }

        let renderJobs = renderJobsByID.values.sorted { $0.createdAt < $1.createdAt }
        let speakers = try await grdbProjectStore.speakers(for: projectID)
        let snapshot = ProjectBundleSnapshot(
            project: project,
            modelID: resolvedBundleModelID(projectModelID: modelID, renderJobs: renderJobs),
            renderSynthesisOptions: synthesisOptions,
            chapters: await projectStore.chapters(for: projectID),
            renderJobs: renderJobs,
            exports: await projectStore.exports(for: projectID),
            speakers: speakers
        )
        let location = ValarProjectBundleLocation(
            projectID: project.id,
            title: project.title,
            bundleURL: bundleURL
        )

        _ = try ProjectBundleWriter().write(snapshot, to: location)
        let normalizedLocation = ValarProjectBundleLocation(
            projectID: project.id,
            title: project.title,
            bundleURL: location.bundleURL.pathExtension == "valarproject"
                ? location.bundleURL
                : location.bundleURL.appendingPathExtension("valarproject")
        )
        await updateProjectBundleURL(normalizedLocation.bundleURL, for: project.id)
        return normalizedLocation
    }

    func openProjectBundle(from bundleURL: URL) async throws -> ProjectBundleSnapshot {
        let bundle = try ProjectBundleReader().read(from: bundleURL)
        await projectStore.restore(
            project: bundle.snapshot.project,
            chapters: bundle.snapshot.chapters,
            renderJobs: bundle.snapshot.renderJobs,
            exports: bundle.snapshot.exports,
            speakers: bundle.snapshot.speakers
        )
        await updateProjectBundleURL(bundleURL, for: bundle.snapshot.project.id)
        await renderQueue.replaceJobs(
            for: bundle.snapshot.project.id,
            with: bundle.snapshot.renderJobs.map(makeRenderJob(from:))
        )
        return bundle.snapshot
    }

    func attachChapterSourceAudio(
        chapter: ChapterRecord,
        from sourceURL: URL
    ) async throws -> ChapterRecord {
        let location = try await requiredProjectBundleLocation(for: chapter.projectID)
        let sourceData = try readSecurityScopedData(at: sourceURL)
        let sourceHint = sourceURL.pathExtension.isEmpty ? nil : sourceURL.pathExtension
        let decodedBuffer = try await audioPipeline.decode(sourceData, hint: sourceHint)
        let exportedAsset = try await audioPipeline.export(
            decodedBuffer,
            as: AudioFormatDescriptor(
                sampleRate: decodedBuffer.format.sampleRate,
                channelCount: decodedBuffer.format.channelCount,
                sampleFormat: decodedBuffer.format.sampleFormat,
                interleaved: decodedBuffer.format.interleaved,
                container: "wav"
            )
        )

        let assetName = Self.chapterSourceAudioAssetName(for: chapter.id)
        let destinationURL = try projectAssetURL(assetName: assetName, within: location.assetsDirectory)
        try persistProjectAsset(exportedAsset.data, to: destinationURL, within: location.assetsDirectory)

        if let previousAssetName = chapter.sourceAudioAssetName,
           previousAssetName != assetName {
            let previousAssetURL = try projectAssetURL(
                assetName: previousAssetName,
                within: location.assetsDirectory
            )
            try? removeItemIfPresent(at: previousAssetURL, within: location.assetsDirectory)
        }

        if let session = await runtime.documentSession(for: chapter.projectID) {
            await session.attachAudio(
                to: chapter.id,
                assetName: assetName,
                sampleRate: decodedBuffer.format.sampleRate,
                durationSeconds: decodedBuffer.duration
            )
            await session.setTranscription(for: chapter.id, transcriptionJSON: nil, modelID: nil)
            await session.setAlignment(for: chapter.id, alignmentJSON: nil, modelID: nil)
        } else {
            await projectStore.attachAudio(
                to: chapter.id,
                in: chapter.projectID,
                assetName: assetName,
                sampleRate: decodedBuffer.format.sampleRate,
                durationSeconds: decodedBuffer.duration
            )
            await projectStore.setTranscription(
                for: chapter.id,
                in: chapter.projectID,
                transcriptionJSON: nil,
                modelID: nil
            )
            await projectStore.setAlignment(
                for: chapter.id,
                in: chapter.projectID,
                alignmentJSON: nil,
                modelID: nil
            )
        }

        var updated = chapter
        updated.sourceAudioAssetName = assetName
        updated.sourceAudioSampleRate = decodedBuffer.format.sampleRate
        updated.sourceAudioDurationSeconds = decodedBuffer.duration
        updated.transcriptionJSON = nil
        updated.transcriptionModelID = nil
        updated.alignmentJSON = nil
        updated.alignmentModelID = nil
        return updated
    }

    func transcribeChapterSourceAudio(
        chapter: ChapterRecord,
        modelID: ModelIdentifier,
        languageHint: String?
    ) async throws -> ChapterRecord {
        let audioChunk = try await sourceAudioChunk(for: chapter)
        let result = try await runtime.transcribe(
            SpeechRecognitionRequest(
                model: modelID,
                audio: audioChunk,
                languageHint: normalizedLanguageHint(languageHint)
            )
        )
        let transcriptionJSON = try Self.encodeJSONString(result)

        if let session = await runtime.documentSession(for: chapter.projectID) {
            await session.setTranscription(
                for: chapter.id,
                transcriptionJSON: transcriptionJSON,
                modelID: modelID.rawValue
            )
        } else {
            await projectStore.setTranscription(
                for: chapter.id,
                in: chapter.projectID,
                transcriptionJSON: transcriptionJSON,
                modelID: modelID.rawValue
            )
        }

        var updated = chapter
        updated.transcriptionJSON = transcriptionJSON
        updated.transcriptionModelID = modelID.rawValue
        return updated
    }

    func alignChapterSourceAudio(
        chapter: ChapterRecord,
        transcript: String,
        modelID: ModelIdentifier,
        languageHint: String?
    ) async throws -> ChapterRecord {
        let trimmedTranscript = transcript.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedTranscript.isEmpty else {
            throw ProjectChapterAudioError.emptyAlignmentTranscript
        }

        let audioChunk = try await sourceAudioChunk(for: chapter)
        let result = try await runtime.align(
            ForcedAlignmentRequest(
                model: modelID,
                audio: audioChunk,
                transcript: trimmedTranscript,
                languageHint: normalizedLanguageHint(languageHint)
            )
        )
        let alignmentJSON = try Self.encodeJSONString(result)

        if let session = await runtime.documentSession(for: chapter.projectID) {
            await session.setAlignment(
                for: chapter.id,
                alignmentJSON: alignmentJSON,
                modelID: modelID.rawValue
            )
        } else {
            await projectStore.setAlignment(
                for: chapter.id,
                in: chapter.projectID,
                alignmentJSON: alignmentJSON,
                modelID: modelID.rawValue
            )
        }

        var updated = chapter
        updated.alignmentJSON = alignmentJSON
        updated.alignmentModelID = modelID.rawValue
        return updated
    }

    func descriptor(for identifier: ModelIdentifier) async throws -> ModelDescriptor {
        if let descriptor = await modelRegistry.descriptor(for: identifier) {
            return descriptor
        }

        if let descriptor = await capabilityRegistry.descriptor(for: identifier) {
            return descriptor
        }

        let supportedModels = try await modelCatalog.supportedModels()
        if let descriptor = supportedModels.first(where: { $0.id == identifier })?.descriptor {
            return descriptor
        }

        throw VoiceCloneServiceError.missingModel(identifier.rawValue)
    }

    func createVoice(label: String, modelID: ModelIdentifier) async throws -> VoiceLibraryRecord {
        try await voiceLibraryStore.save(
            VoiceLibraryRecord(
                label: label,
                modelID: modelID.rawValue,
                runtimeModelID: modelID.rawValue
            )
        )
    }

    func createDesignedVoice(label: String, prompt: String) async throws -> VoiceLibraryRecord {
        try await voiceLibraryStore.save(
            VoiceLibraryRecord(
                label: label,
                modelID: Self.qwenVoiceDesignModelID.rawValue,
                runtimeModelID: Self.qwenVoiceDesignModelID.rawValue,
                voiceKind: VoiceKind.legacyPrompt.rawValue,
                voicePrompt: prompt
            )
        )
    }

    func deleteVoice(_ voiceID: UUID) async throws {
        try await runtime.deleteVoice(voiceID)
    }

    func cloneVoice(_ draft: VoiceCloneDraft) async throws -> VoiceLibraryRecord {
        let trimmedLabel = draft.label.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedLabel.isEmpty else {
            throw VoiceCloneError.emptyLabel
        }

        let trimmedTranscript = draft.referenceTranscript.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedTranscript.isEmpty else {
            throw VoiceCloneError.emptyTranscript
        }

        let fileExtension = draft.referenceAudioURL.pathExtension.lowercased()
        guard VoiceCloneFileValidator.allowedExtensions.contains(fileExtension) else {
            throw VoiceCloneError.unsupportedFileType(fileExtension)
        }

        let fileData = try readSecurityScopedData(at: draft.referenceAudioURL)

        guard fileData.count <= VoiceCloneFileValidator.maximumFileSizeBytes else {
            throw VoiceCloneError.fileTooLarge(bytes: fileData.count)
        }

        try VoiceCloneFileValidator.validateFileHeader(fileData, hint: fileExtension)

        let decodedBuffer = try await audioPipeline.decode(fileData, hint: fileExtension)
        let assessment = try VoiceCloneAudioValidator.validate(decodedBuffer)
        let selectedRuntimeModelID = draft.modelID ?? VoiceCloneModels.defaultRuntimeModelID
        let runtimeDescriptor = try await descriptor(for: selectedRuntimeModelID)
        let targetSampleRate = runtimeDescriptor.defaultSampleRate ?? assessment.normalizedBuffer.format.sampleRate
        let preparedBuffer = if assessment.normalizedBuffer.format.sampleRate == targetSampleRate {
            assessment.normalizedBuffer
        } else {
            try await audioPipeline.resample(assessment.normalizedBuffer, to: targetSampleRate)
        }

        let exportedAsset = try await audioPipeline.export(
            preparedBuffer,
            as: AudioFormatDescriptor(
                sampleRate: preparedBuffer.format.sampleRate,
                channelCount: 1,
                sampleFormat: .float32,
                interleaved: false,
                container: "wav"
            )
        )

        let voiceID = UUID()
        let assetURL = try appPaths.voiceAssetURL(voiceID: voiceID, fileExtension: "wav")
        try persistVoiceAsset(exportedAsset.data, to: assetURL)

        let monoSamples = preparedBuffer.channels.first ?? []
        let record: VoiceLibraryRecord
        if runtimeDescriptor.familyID == .tadaTTS {
            guard let conditioningBackend = inferenceBackend as? any VoiceConditioningInferenceBackend else {
                throw VoiceCloneServiceError.missingModel(VoiceCloneModels.profileModelID.rawValue)
            }
            let conditioning = try await conditioningBackend.extractVoiceConditioning(
                VoiceConditioningExtractionRequest(
                    descriptor: runtimeDescriptor,
                    monoReferenceSamples: monoSamples,
                    sampleRate: preparedBuffer.format.sampleRate,
                    referenceTranscript: trimmedTranscript
                )
            )
            let bundleURL = try appPaths.voiceConditioningAssetURL(voiceID: voiceID)
            _ = try TADAConditioningBundleIO.write(conditioning: conditioning, to: bundleURL)
            record = VoiceLibraryRecord(
                id: voiceID,
                label: trimmedLabel,
                modelID: runtimeDescriptor.id.rawValue,
                runtimeModelID: runtimeDescriptor.id.rawValue,
                sourceAssetName: draft.referenceAudioURL.lastPathComponent,
                referenceAudioAssetName: assetURL.lastPathComponent,
                referenceTranscript: trimmedTranscript,
                referenceDurationSeconds: assessment.durationSeconds,
                referenceSampleRate: preparedBuffer.format.sampleRate,
                referenceChannelCount: assessment.originalChannelCount,
                conditioningFormat: VoiceLibraryRecord.tadaReferenceConditioningFormat,
                voiceKind: VoiceKind.tadaReference.rawValue
            )
        } else {
            let conditioning: VoiceConditioning?
            if runtimeDescriptor.familyID == .qwen3TTS,
               let conditioningBackend = inferenceBackend as? any VoiceConditioningInferenceBackend {
                conditioning = try? await conditioningBackend.extractVoiceConditioning(
                    VoiceConditioningExtractionRequest(
                        descriptor: runtimeDescriptor,
                        monoReferenceSamples: monoSamples,
                        sampleRate: preparedBuffer.format.sampleRate,
                        referenceTranscript: trimmedTranscript
                    )
                )
            } else {
                conditioning = nil
            }

            let embedding: Data?
            if let conditioning {
                embedding = conditioning.payload
            } else {
                guard let mlxBackend = inferenceBackend as? MLXInferenceBackend else {
                    throw VoiceCloneServiceError.missingModel(VoiceCloneModels.profileModelID.rawValue)
                }
                embedding = try await mlxBackend.extractSpeakerEmbedding(
                    descriptor: runtimeDescriptor,
                    monoReferenceSamples: monoSamples
                )
            }
            let conditioningFormat = conditioning?.format ?? VoiceLibraryRecord.qwenSpeakerEmbeddingConditioningFormat
            let voiceKind = conditioningFormat == VoiceLibraryRecord.qwenClonePromptConditioningFormat
                ? VoiceKind.clonePrompt.rawValue
                : VoiceKind.embeddingOnly.rawValue
            record = VoiceLibraryRecord(
                id: voiceID,
                label: trimmedLabel,
                modelID: runtimeDescriptor.id.rawValue,
                runtimeModelID: runtimeDescriptor.id.rawValue,
                sourceAssetName: draft.referenceAudioURL.lastPathComponent,
                referenceAudioAssetName: assetURL.lastPathComponent,
                referenceTranscript: trimmedTranscript,
                referenceDurationSeconds: assessment.durationSeconds,
                referenceSampleRate: preparedBuffer.format.sampleRate,
                referenceChannelCount: assessment.originalChannelCount,
                speakerEmbedding: embedding,
                conditioningFormat: conditioningFormat,
                voiceKind: voiceKind
            )
        }

        return try await voiceLibraryStore.save(record)
    }

    func referencePromptPayload(for voice: VoiceLibraryRecord) async throws -> (pcmData: Data, sampleRate: Double, transcript: String)? {
        guard
            let assetName = voice.referenceAudioAssetName,
            let transcript = voice.referenceTranscript,
            !transcript.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            return nil
        }

        if voice.hasReusableQwenClonePrompt {
            return nil
        }

        let assetURL = try resolvedVoiceAssetURL(assetName: assetName)
        guard FileManager.default.fileExists(atPath: assetURL.path) else {
            throw VoiceCloneError.missingReferenceAudio(assetName)
        }

        let assetData = try VoiceLibraryProtection.readProtectedFile(from: assetURL)
        let buffer = try await audioPipeline.decode(assetData, hint: assetURL.pathExtension)
        let monoSamples = buffer.channels.first ?? []
        return (
            pcmData: audioPCMFloat32LEData(from: monoSamples),
            sampleRate: buffer.format.sampleRate,
            transcript: transcript
        )
    }

    func voiceReadyForSynthesis(_ voice: VoiceLibraryRecord?) async throws -> VoiceLibraryRecord? {
        guard let voice else { return nil }
        return try await runtime.upgradeVoiceForSynthesisIfNeeded(voice)
    }

    func inlineReferencePromptPayload(
        audioURL: URL,
        transcript: String?
    ) async throws -> (pcmData: Data, sampleRate: Double, transcript: String?) {
        try VoiceCloneFileValidator.validateFileSelection(audioURL)
        let assetData = try readSecurityScopedData(at: audioURL)
        try VoiceCloneFileValidator.validateFileHeader(assetData, hint: audioURL.pathExtension)
        let buffer = try await audioPipeline.decode(assetData, hint: audioURL.pathExtension)
        let monoSamples = buffer.channels.first ?? []
        let trimmedTranscript = transcript?.trimmingCharacters(in: .whitespacesAndNewlines)
        return (
            pcmData: audioPCMFloat32LEData(from: monoSamples),
            sampleRate: buffer.format.sampleRate,
            transcript: trimmedTranscript?.isEmpty == false ? trimmedTranscript : nil
        )
    }

    func synthesizePreview(
        text: String,
        modelID: ModelIdentifier,
        voiceRecord: VoiceLibraryRecord?
    ) async throws -> AudioPCMBuffer {
        let preparedVoiceRecord = try await voiceReadyForSynthesis(voiceRecord)
        let resolvedDescriptor = try await descriptor(for: modelID)
        let backendRuntime = BackendSelectionPolicy.Runtime(
            availableBackends: [inferenceBackend.backendKind]
        )
        let configuration = try BackendSelectionPolicy().runtimeConfiguration(
            for: resolvedDescriptor,
            residencyPolicy: .automatic,
            runtime: backendRuntime
        )

        let promptPayload: (pcmData: Data, sampleRate: Double, transcript: String)? = if let preparedVoiceRecord {
            try await referencePromptPayload(for: preparedVoiceRecord)
        } else {
            nil
        }

        let request = SpeechSynthesisRequest(
            model: modelID,
            text: text,
            voice: resolvedVoiceProfile(from: preparedVoiceRecord),
            referenceAudioAssetName: preparedVoiceRecord?.referenceAudioAssetName,
            referenceAudioPCMFloat32LE: promptPayload?.pcmData,
            referenceAudioSampleRate: promptPayload?.sampleRate,
            referenceTranscript: promptPayload?.transcript,
            sampleRate: resolvedDescriptor.defaultSampleRate ?? 24_000,
            responseFormat: "pcm_f32le",
            voiceBehavior: .auto
        )
        do {
            return try await self.runtime.withReservedTextToSpeechWorkflowSession(
                descriptor: resolvedDescriptor,
                configuration: configuration
            ) { reserved in
                let chunk = try await reserved.workflow.synthesize(request: request, in: reserved.session)
                return AudioPCMBuffer(mono: chunk.samples, sampleRate: chunk.sampleRate)
            }
        } catch {
            if let localizedError = error as? LocalizedError,
               localizedError.errorDescription?.contains("does not provide a text-to-speech workflow") == true {
                throw AudioPreviewError.unsupportedWorkflow(modelID)
            }
            throw error
        }
    }

    func queueRender(projectID: UUID, modelID: ModelIdentifier) async -> RenderJob {
        return await renderQueue.enqueue(projectID: projectID, modelID: modelID)
    }

    private func resolvedBundleModelID(
        projectModelID: ModelIdentifier?,
        renderJobs: [RenderJobRecord]
    ) -> String? {
        if let projectModelID, !projectModelID.rawValue.isEmpty {
            return projectModelID.rawValue
        }
        return renderJobs.last(where: { !$0.modelID.isEmpty })?.modelID
    }

    private func makeRenderJobRecord(from job: RenderJob, chapterIDs: [UUID]) -> RenderJobRecord {
        RenderJobRecord(
            id: job.id,
            projectID: job.projectID,
            modelID: job.modelID.rawValue,
            chapterIDs: chapterIDs,
            outputFileName: normalizedOutputFileName(from: job),
            createdAt: job.createdAt,
            updatedAt: .now,
            state: job.state.rawValue,
            priority: job.priority,
            progress: job.progress,
            title: job.title,
            synthesisOptions: job.synthesisOptions
        )
    }

    private func normalizedOutputFileName(from job: RenderJob) -> String {
        let trimmedOutputFileName = job.outputFileName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmedOutputFileName.isEmpty else {
            return trimmedOutputFileName
        }

        let trimmedTitle = (job.title ?? "render")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let baseName = ProjectRenderService.sanitizeFileStem(
            trimmedTitle.isEmpty ? "render" : trimmedTitle
        )
        return "\(baseName)-\(job.id.uuidString).wav"
    }

    private func makeRenderJob(from record: RenderJobRecord) -> RenderJob {
        RenderJob(
            id: record.id,
            projectID: record.projectID,
            modelID: ModelIdentifier(record.modelID),
            chapterIDs: record.chapterIDs,
            outputFileName: record.outputFileName,
            createdAt: record.createdAt,
            state: RenderJobState(rawValue: record.state) ?? .queued,
            priority: record.priority,
            progress: record.progress,
            title: record.title,
            synthesisOptions: record.synthesisOptions
        )
    }

    private static func resolveDescriptor(
        for identifier: ModelIdentifier,
        modelRegistry: ModelRegistry,
        capabilityRegistry: CapabilityRegistry,
        modelCatalog: ModelCatalog
    ) async throws -> ModelDescriptor {
        if let descriptor = await modelRegistry.descriptor(for: identifier) {
            return descriptor
        }

        if let descriptor = await capabilityRegistry.descriptor(for: identifier) {
            return descriptor
        }

        let supportedModels = try await modelCatalog.supportedModels()
        if let descriptor = supportedModels.first(where: { $0.id == identifier })?.descriptor {
            return descriptor
        }

        throw ProjectRenderServiceError.missingModel(identifier)
    }

    private func runtimeConfiguration(
        for manifest: ValarModelKit.ModelPackManifest,
        residencyPolicy: ResidencyPolicy = .automatic
    ) throws -> ModelRuntimeConfiguration {
        let estimatedBytes = manifest.artifacts.compactMap(\.sizeBytes).reduce(0, +)
        let descriptor = ModelDescriptor(manifest: manifest)
        let runtime = BackendSelectionPolicy.Runtime(
            availableBackends: [inferenceBackend.backendKind]
        )
        return try BackendSelectionPolicy().runtimeConfiguration(
            for: descriptor,
            residencyPolicy: residencyPolicy,
            preferredSampleRate: manifest.audio?.defaultSampleRate,
            memoryBudgetBytes: estimatedBytes > 0 ? estimatedBytes : nil,
            allowQuantizedWeights: true,
            allowWarmStart: true,
            runtime: runtime
        )
    }

    private func resolvedVoiceProfile(from record: VoiceLibraryRecord?) -> VoiceProfile? {
        guard let record else { return nil }
        return record.makeVoiceProfile()
    }

    private func audioBuffer(from chunk: AudioChunk) -> AudioPCMBuffer {
        AudioPCMBuffer(mono: chunk.samples, sampleRate: chunk.sampleRate)
    }

    private func readSecurityScopedData(at url: URL) throws -> Data {
        let needsScopedAccess = url.startAccessingSecurityScopedResource()
        defer {
            if needsScopedAccess {
                url.stopAccessingSecurityScopedResource()
            }
        }
        return try Data(contentsOf: url)
    }

    private func persistVoiceAsset(_ data: Data, to url: URL) throws {
        let directory = url.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try ValarAppPaths.validateContainment(url, within: appPaths.voiceLibraryDirectory)
        try VoiceLibraryProtection.writeProtectedFile(data, to: url)
    }

    private static func chapterSourceAudioAssetName(for chapterID: UUID) -> String {
        "chapter-source-\(chapterID.uuidString).wav"
    }

    private static func encodeJSONString<T: Encodable>(_ value: T) throws -> String {
        let data = try JSONEncoder().encode(value)
        guard let string = String(data: data, encoding: .utf8) else {
            throw CocoaError(.fileWriteUnknown)
        }
        return string
    }

    private func normalizedLanguageHint(_ value: String?) -> String? {
        guard let trimmed = value?.trimmingCharacters(in: .whitespacesAndNewlines),
              !trimmed.isEmpty
        else {
            return nil
        }
        return trimmed
    }

    private func requiredProjectBundleLocation(for projectID: UUID) async throws -> ValarProjectBundleLocation {
        guard let location = await projectStore.bundleLocation(for: projectID) else {
            throw ProjectChapterAudioError.missingProjectBundleStorage(projectID)
        }
        return location
    }

    private func sourceAudioChunk(for chapter: ChapterRecord) async throws -> AudioChunk {
        guard let assetName = chapter.sourceAudioAssetName else {
            throw ProjectChapterAudioError.missingSourceAudio(chapter.id)
        }

        let location = try await requiredProjectBundleLocation(for: chapter.projectID)
        let assetURL = try projectAssetURL(assetName: assetName, within: location.assetsDirectory)
        guard FileManager.default.fileExists(atPath: assetURL.path) else {
            throw ProjectChapterAudioError.sourceAudioAssetNotFound(assetName)
        }

        let data = try Data(contentsOf: assetURL)
        let buffer = try await audioPipeline.decode(data, hint: assetURL.pathExtension)
        return AudioChunk(
            samples: monoSamples(from: buffer),
            sampleRate: buffer.format.sampleRate
        )
    }

    private func monoSamples(from buffer: AudioPCMBuffer) -> [Float] {
        guard let firstChannel = buffer.channels.first else {
            return []
        }
        guard buffer.channels.count > 1 else {
            return firstChannel
        }

        var mono = [Float](repeating: 0, count: firstChannel.count)
        let scale = 1 / Float(buffer.channels.count)
        for channel in buffer.channels {
            for index in 0..<min(mono.count, channel.count) {
                mono[index] += channel[index] * scale
            }
        }
        return mono
    }

    private func persistProjectAsset(_ data: Data, to url: URL, within root: URL) throws {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try ValarAppPaths.validateContainment(url, within: root)
        try data.write(to: url, options: .atomic)
    }

    private func projectAssetURL(assetName: String, within assetsDirectory: URL) throws -> URL {
        let assetURL = assetsDirectory.appendingPathComponent(assetName, isDirectory: false)
        try ValarAppPaths.validateContainment(assetURL, within: assetsDirectory)
        return assetURL
    }

    private func resolvedVoiceAssetURL(assetName: String) throws -> URL {
        let assetURL = appPaths.voiceLibraryDirectory.appendingPathComponent(assetName, isDirectory: false)
        try ValarAppPaths.validateContainment(assetURL, within: appPaths.voiceLibraryDirectory)
        return assetURL
    }

    private func removeItemIfPresent(at url: URL, within root: URL) throws {
        try ValarAppPaths.validateContainment(url, within: root)
        guard FileManager.default.fileExists(atPath: url.path) else { return }
        try FileManager.default.removeItem(at: url)
    }

    private func removeProjectBundleIfPresent(at bundleURL: URL) throws {
        guard bundleURL.pathExtension == "valarproject" else { return }
        guard FileManager.default.fileExists(atPath: bundleURL.path) else { return }
        try FileManager.default.removeItem(at: bundleURL)
    }

    private func audioPCMFloat32LEData(from samples: [Float]) -> Data {
        var data = Data()
        data.reserveCapacity(samples.count * MemoryLayout<Float>.size)
        for sample in samples {
            var bits = sample.bitPattern.littleEndian
            withUnsafeBytes(of: &bits) { rawBuffer in
                data.append(contentsOf: rawBuffer)
            }
        }
        return data
    }

    private static func makeSupportedManifest(from record: SupportedModelCatalogRecord) -> ValarModelKit.ModelPackManifest {
        let familyID = makeFamilyID(from: record.familyID)
        return ValarModelKit.ModelPackManifest(
            id: ModelIdentifier(record.modelID),
            familyID: familyID,
            displayName: record.displayName,
            domain: inferredDomain(for: familyID),
            capabilities: inferredCapabilities(
                from: [],
                familyID: familyID,
                modelID: record.modelID,
                displayName: record.displayName
            ),
            supportedBackends: [BackendRequirement(backendKind: .mlx)],
            artifacts: [],
            licenses: [],
            notes: record.installHint
        )
    }

    private static func makeModelManifest(from manifest: ValarPersistence.ModelPackManifest) -> ValarModelKit.ModelPackManifest {
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
                    sizeBytes: artifact.byteCount
                )
            },
            tokenizer: manifest.tokenizerType.map { TokenizerSpec(kind: $0) },
            audio: AudioConstraint(defaultSampleRate: manifest.sampleRate),
            licenses: makeLicenseSpecs(from: manifest),
            minimumAppVersion: manifest.minimumAppVersion,
            notes: manifest.notes
        )
    }

    private static func makeFamilyID(from value: String) -> ModelFamilyID {
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

    private static func inferredDomain(for familyID: ModelFamilyID) -> ModelDomain {
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

    private static func inferredCapabilities(
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

    private static func parseBackendRequirements(_ values: [String]) -> [BackendRequirement] {
        let requirements = values.compactMap { value in
            BackendKind(rawValue: value).map { BackendRequirement(backendKind: $0) }
        }
        return requirements.isEmpty ? [BackendRequirement(backendKind: .mlx)] : requirements
    }

    private static func makeLicenseSpecs(from manifest: ValarPersistence.ModelPackManifest) -> [LicenseSpec] {
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

    private static func artifactRole(for value: String) -> ArtifactRole {
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

    @discardableResult
    func registerModelsReadyObserver(
        _ observer: @escaping ModelsReadyObserver
    ) -> UUID {
        let token = UUID()
        if didFinishModelsReady {
            Task { @MainActor in
                await observer()
            }
            return token
        }
        modelsReadyObservers[token] = observer
        ensureModelsReady()
        return token
    }

    func unregisterModelsReadyObserver(_ token: UUID) {
        modelsReadyObservers.removeValue(forKey: token)
    }

    private func ensureModelsReady() {
        guard !didFinishModelsReady, modelsReadyTask == nil else { return }

        let modelPackRegistry = modelPackRegistry
        let modelInstaller = modelInstaller
        let runtime = runtime
        let log = Logger(subsystem: "com.valar.tts", category: "ModelInstall")
        let catalogSeedAction = Self.catalogSeedAction()

        modelsReadyTask = Task.detached(priority: .utility) {
            let didSeedCatalog = await Self.seedCatalogMetadata(
                using: catalogSeedAction,
                modelPackRegistry: modelPackRegistry,
                modelInstaller: modelInstaller,
                log: log
            )
            if didSeedCatalog {
                Self.storeCatalogSeedVersion(Self.bootstrapVersion)
            }

            await runtime.hydrateInstalledCatalogDescriptors()
            if runtime.runtimeConfiguration.warmPolicy == .eager {
                await runtime.prewarmInstalledModels()
            }

            let observers = await MainActor.run { () -> [ModelsReadyObserver] in
                self.didFinishModelsReady = true
                self.modelsReadyTask = nil
                let observers = Array(self.modelsReadyObservers.values)
                self.modelsReadyObservers.removeAll()
                self.legacyModelsReadyObserverID = nil
                return observers
            }
            for observer in observers {
                await observer()
            }
        }
    }

    private nonisolated static func catalogSeedAction(defaults: UserDefaults = .standard) -> CatalogSeedAction {
        guard let storedValue = defaults.object(forKey: Self.catalogSeedVersionKey) else {
            return .bootstrapAndInstall
        }

        let storedVersion = (storedValue as? NSNumber)?.intValue ?? 0
        if storedVersion == Self.bootstrapVersion {
            return .skipSeeding
        }
        if storedVersion < Self.bootstrapVersion {
            return .reseedMetadataOnly
        }
        return .skipSeeding
    }

    private nonisolated static func seedCatalogMetadata(
        using action: CatalogSeedAction,
        modelPackRegistry: any ModelPackManaging,
        modelInstaller: ModelInstaller,
        log: Logger
    ) async -> Bool {
        switch action {
        case .skipSeeding:
            return false
        case .reseedMetadataOnly:
            return await registerCatalogEntries(modelPackRegistry: modelPackRegistry, log: log)
        case .bootstrapAndInstall:
            let didRegisterEntries = await registerCatalogEntries(modelPackRegistry: modelPackRegistry, log: log)
            let didInstallEntries = await installMissingCatalogEntries(
                modelPackRegistry: modelPackRegistry,
                modelInstaller: modelInstaller,
                log: log
            )
            return didRegisterEntries && didInstallEntries
        }
    }

    private nonisolated static func registerCatalogEntries(
        modelPackRegistry: any ModelPackManaging,
        log: Logger
    ) async -> Bool {
        var didRegisterEntries = true

        for entry in curatedCatalogEntries() {
            do {
                try await modelPackRegistry.registerSupported(ModelCatalog.makeSupportedRecord(from: entry))
            } catch {
                didRegisterEntries = false
                log.error("Failed to register supported model \(entry.id.rawValue, privacy: .public): \(error.localizedDescription, privacy: .private)")
            }
        }

        return didRegisterEntries
    }

    private nonisolated static func installMissingCatalogEntries(
        modelPackRegistry: any ModelPackManaging,
        modelInstaller: ModelInstaller,
        log: Logger
    ) async -> Bool {
        var didInstallEntries = true

        for entry in curatedCatalogEntries() {
            // The app should only auto-install bundled/local model packs here.
            // Remote Hugging Face models must stay explicit user actions in the
            // Models UI; otherwise first launch can create misleading install
            // receipts or trigger surprise downloads.
            guard entry.remoteURL == nil else {
                continue
            }
            do {
                guard try await modelPackRegistry.installedRecord(for: entry.id.rawValue) == nil else {
                    continue
                }
            } catch {
                didInstallEntries = false
                log.error("Failed to query installed record for \(entry.id.rawValue, privacy: .public): \(error.localizedDescription, privacy: .private)")
                continue
            }

            do {
                _ = try await modelInstaller.install(
                    manifest: ModelCatalog.makePersistenceManifest(from: entry.manifest),
                    sourceKind: .localFile,
                    sourceLocation: entry.id.rawValue,
                    notes: "Curated Qwen-first core model"
                )
            } catch {
                didInstallEntries = false
                log.error("Failed to install model \(entry.id.rawValue, privacy: .public): \(error.localizedDescription, privacy: .private)")
            }
        }

        return didInstallEntries
    }

    private nonisolated static func curatedCatalogEntries() -> [SupportedModelCatalogEntry] {
        SupportedModelCatalog.curatedEntries(
            includeNonCommercial: CatalogVisibilityPolicy.currentProcess().allowsNonCommercialModels
        )
    }

    private nonisolated static func storeCatalogSeedVersion(_ version: Int, defaults: UserDefaults = .standard) {
        defaults.set(version, forKey: Self.catalogSeedVersionKey)
    }
}

private struct AppTranslator: TranslationProvider {
    func translate(_ request: TranslationRequest) async throws -> String {
        request.text
    }
}

private enum VoiceCloneServiceError: LocalizedError {
    case missingModel(String)

    var errorDescription: String? {
        switch self {
        case .missingModel(let identifier):
            return "Model '\(identifier)' is not available for voice cloning."
        }
    }
}

private enum AudioPreviewError: LocalizedError {
    case unsupportedWorkflow(ModelIdentifier)

    var errorDescription: String? {
        switch self {
        case .unsupportedWorkflow(let modelID):
            return "Audio preview failed: model '\(modelID.rawValue)' does not support text-to-speech."
        }
    }
}
