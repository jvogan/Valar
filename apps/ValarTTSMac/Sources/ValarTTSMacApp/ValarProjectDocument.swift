import AppKit
import Foundation
import Observation
import SwiftUI
import UniformTypeIdentifiers
import ValarCore
import ValarModelKit
import ValarPersistence

// Stable path for the parent directory of all per-session isolated runtimes.
// Defined at file scope so it can be referenced by the atexit handler without capture.
private let _valarSessionsParentPath: String =
    NSTemporaryDirectory() + "ValarTTSDocumentSessions"

// Free function with C-compatible signature for use with atexit().
private func _valarCleanupSessionsOnExit() {
    try? FileManager.default.removeItem(atPath: _valarSessionsParentPath)
}

extension UTType {
    static let valarProject = UTType(exportedAs: "com.valar.tts.project", conformingTo: .package)
}

@MainActor
final class ValarProjectDocument: @preconcurrency ReferenceFileDocument {
    typealias Snapshot = ProjectBundle

    static var readableContentTypes: [UTType] { [.valarProject] }
    static let sharedServices = ValarServiceHub.live()

    @Published private(set) var bundle: ProjectBundle

    let services: ValarServiceHub
    let sharedServices: ValarServiceHub
    let appState: AppState
    private let runtimeRootURL: URL?

    private var hasLoadedIntoServices = false
    private var isTrackingDocumentChanges = false
    private var cachedSnapshot: ProjectBundle?
    private var snapshotRefreshTask: Task<Void, Never>?
    private var activeBundleURL: URL?

    // MARK: - Biometric data cleanup

    /// Removes all stale per-session temp directories left by prior sessions.
    /// Call once at app launch before any documents are opened.
    static func cleanupStaleSessionDirectories() {
        try? FileManager.default.removeItem(atPath: _valarSessionsParentPath)
    }

    /// Installed exactly once (lazy static). Registers an atexit handler and an
    /// applicationWillTerminate observer so session temp data (which contains
    /// biometric speaker embeddings) is purged on normal exit and — combined
    /// with cleanupStaleSessionDirectories() at next launch — on crash/force-quit.
    private static let _cleanupHandlersRegistered: Bool = {
        atexit(_valarCleanupSessionsOnExit)
        NotificationCenter.default.addObserver(
            forName: NSApplication.willTerminateNotification,
            object: nil,
            queue: nil
        ) { _ in
            try? FileManager.default.removeItem(atPath: _valarSessionsParentPath)
        }
        return true
    }()

    convenience init() {
        let bundle = Self.makeEmptyBundle()
        let runtimeContext = Self.makeIsolatedRuntimeContext(sharedAppPaths: Self.sharedServices.appPaths)
        self.init(
            services: runtimeContext.services,
            sharedServices: Self.sharedServices,
            bundle: bundle,
            runtimeRootURL: runtimeContext.runtimeRootURL
        )
    }

    convenience init(configuration: ReadConfiguration) throws {
        let bundle = try Self.readBundle(from: configuration.file)
        let runtimeContext = Self.makeIsolatedRuntimeContext(sharedAppPaths: Self.sharedServices.appPaths)
        self.init(
            services: runtimeContext.services,
            sharedServices: Self.sharedServices,
            bundle: bundle,
            runtimeRootURL: runtimeContext.runtimeRootURL
        )
    }

    convenience init(services: ValarServiceHub) {
        self.init(
            services: services,
            sharedServices: services,
            bundle: Self.makeEmptyBundle(),
            runtimeRootURL: nil
        )
    }

    private init(
        services: ValarServiceHub,
        sharedServices: ValarServiceHub,
        bundle: ProjectBundle,
        runtimeRootURL: URL?
    ) {
        self.services = services
        self.sharedServices = sharedServices
        self.appState = AppState(
            services: services,
            sharedServices: sharedServices,
            documentProjectID: bundle.snapshot.project.id
        )
        self.appState.projectRenderModelID = bundle.snapshot.modelID.map { ModelIdentifier($0) }
        self.appState.projectRenderSynthesisOptions = bundle.snapshot.renderSynthesisOptions
        self.bundle = bundle
        self.runtimeRootURL = runtimeRootURL
    }

    deinit {
        snapshotRefreshTask?.cancel()
        guard let runtimeRootURL else { return }
        try? FileManager.default.removeItem(at: runtimeRootURL)
    }

    func prepareForEditing(fileURL: URL?) async {
        appState.updateProjectDocumentURL(fileURL)
        let workingBundleURL = resolvedWorkingBundleURL(fileURL: fileURL)
        activeBundleURL = workingBundleURL
        await services.updateProjectBundleURL(workingBundleURL, for: bundle.snapshot.project.id)

        guard !hasLoadedIntoServices else { return }

        _ = await services.runtime.createDocumentSession(for: bundle)
        await appState.load()
        appState.selectedSection = .project
        hasLoadedIntoServices = true
        cachedSnapshot = bundle
        beginTrackingDocumentChanges()
    }

    func snapshot(contentType: UTType) throws -> ProjectBundle {
        // Return the eagerly-cached snapshot to avoid blocking the main thread.
        // The snapshot is refreshed asynchronously on every mutation via markEdited().
        if let cachedSnapshot {
            bundle = cachedSnapshot
            return cachedSnapshot
        }
        return bundle
    }

    func fileWrapper(snapshot: ProjectBundle, configuration: WriteConfiguration) throws -> FileWrapper {
        bundle = snapshot

        let temporaryBundleURL = try Self.makeTemporaryBundleURL()
        let location = ValarProjectBundleLocation(
            projectID: snapshot.snapshot.project.id,
            title: snapshot.snapshot.project.title,
            bundleURL: temporaryBundleURL
        )

        try ProjectBundleWriter().write(
            snapshot.snapshot,
            to: location,
            createdAt: snapshot.manifest.createdAt
        )
        try copyAssetsIfPresent(
            from: activeBundleURL,
            to: temporaryBundleURL,
            projectID: snapshot.snapshot.project.id
        )

        let wrapper = try FileWrapper(url: temporaryBundleURL, options: .immediate)
        try? FileManager.default.removeItem(at: temporaryBundleURL)
        return wrapper
    }

    private static func currentBundle(
        from services: ValarServiceHub,
        projectID: UUID,
        preferredModelID: ModelIdentifier?,
        preferredSynthesisOptions: RenderSynthesisOptions,
        createdAt: Date,
        version: Int
    ) async throws -> ProjectBundle {
        guard let project = await services.projectStore.allProjects().first(where: { $0.id == projectID }) else {
            throw ProjectBundleError.projectNotFound(projectID)
        }

        let storedRenderJobs = await services.projectStore.renderJobs(for: projectID)
        let queuedRenderJobs = await services.renderQueue.jobs(matching: nil).filter { $0.projectID == projectID }
        var renderJobsByID = Dictionary(uniqueKeysWithValues: storedRenderJobs.map { ($0.id, $0) })

        for job in queuedRenderJobs {
            let chapterIDs = renderJobsByID[job.id]?.chapterIDs ?? []
            renderJobsByID[job.id] = makeRenderJobRecord(from: job, chapterIDs: chapterIDs)
        }

        let renderJobs = renderJobsByID.values.sorted { $0.createdAt < $1.createdAt }
        let chapters = await services.projectStore.chapters(for: projectID)
        let exports = await services.projectStore.exports(for: projectID)
        let speakers = await services.projectStore.speakers(for: projectID)
        let snapshot = ProjectBundleSnapshot(
            project: project,
            modelID: resolvedBundleModelID(projectModelID: preferredModelID, renderJobs: renderJobs),
            renderSynthesisOptions: preferredSynthesisOptions,
            chapters: chapters,
            renderJobs: renderJobs,
            exports: exports,
            speakers: speakers
        )

        return ProjectBundle(
            manifest: ProjectBundleManifest(
                version: version,
                createdAt: createdAt,
                projectID: project.id,
                title: project.title,
                modelID: snapshot.modelID,
                renderSynthesisOptions: snapshot.renderSynthesisOptions,
                chapters: chapters
                    .sorted(by: { $0.index < $1.index })
                    .map { chapter in
                        ProjectBundleManifest.ChapterSummary(
                            id: chapter.id,
                            index: chapter.index,
                            title: chapter.title
                        )
                    }
            ),
            snapshot: snapshot
        )
    }

    private static func makeEmptyBundle() -> ProjectBundle {
        let project = ProjectRecord(title: "Untitled Project")
        let snapshot = ProjectBundleSnapshot(
            project: project,
            renderSynthesisOptions: RenderSynthesisOptions(),
            chapters: [],
            renderJobs: [],
            exports: [],
            speakers: []
        )
        return ProjectBundle(
            manifest: ProjectBundleManifest(
                projectID: project.id,
                title: project.title,
                renderSynthesisOptions: snapshot.renderSynthesisOptions,
                chapters: []
            ),
            snapshot: snapshot
        )
    }

    private static func readBundle(from fileWrapper: FileWrapper) throws -> ProjectBundle {
        let temporaryBundleURL = try makeTemporaryBundleURL()
        defer { try? FileManager.default.removeItem(at: temporaryBundleURL) }
        try fileWrapper.write(to: temporaryBundleURL, options: .atomic, originalContentsURL: nil)
        return try ProjectBundleReader().read(from: temporaryBundleURL)
    }

    private static func makeTemporaryBundleURL() throws -> URL {
        let parentDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("ValarTTSDocuments", isDirectory: true)
        try FileManager.default.createDirectory(
            at: parentDirectory,
            withIntermediateDirectories: true
        )
        return parentDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
            .appendingPathExtension("valarproject")
    }

    private func resolvedWorkingBundleURL(fileURL: URL?) -> URL? {
        if let fileURL {
            return fileURL
        }

        guard let runtimeRootURL else {
            return nil
        }

        return runtimeRootURL
            .appendingPathComponent(bundle.snapshot.project.id.uuidString, isDirectory: true)
            .appendingPathExtension("valarproject")
    }

    private func copyAssetsIfPresent(
        from sourceBundleURL: URL?,
        to destinationBundleURL: URL,
        projectID: UUID
    ) throws {
        guard let sourceBundleURL else { return }

        let sourceAssetsDirectory = ValarProjectBundleLocation(
            projectID: projectID,
            title: bundle.snapshot.project.title,
            bundleURL: sourceBundleURL
        ).assetsDirectory
        guard FileManager.default.fileExists(atPath: sourceAssetsDirectory.path) else {
            return
        }

        let destinationAssetsDirectory = ValarProjectBundleLocation(
            projectID: projectID,
            title: bundle.snapshot.project.title,
            bundleURL: destinationBundleURL
        ).assetsDirectory
        try FileManager.default.createDirectory(
            at: destinationAssetsDirectory,
            withIntermediateDirectories: true
        )

        let assetURLs = try FileManager.default.contentsOfDirectory(
            at: sourceAssetsDirectory,
            includingPropertiesForKeys: nil
        )
        for assetURL in assetURLs {
            let destinationURL = destinationAssetsDirectory.appendingPathComponent(
                assetURL.lastPathComponent,
                isDirectory: false
            )
            if FileManager.default.fileExists(atPath: destinationURL.path) {
                try FileManager.default.removeItem(at: destinationURL)
            }
            try FileManager.default.copyItem(at: assetURL, to: destinationURL)
        }
    }

    private struct IsolatedRuntimeContext {
        let services: ValarServiceHub
        let runtimeRootURL: URL
    }

    private static func makeIsolatedRuntimeContext(
        sharedAppPaths: ValarAppPaths = ValarAppPaths(),
        fileManager: FileManager = .default
    ) -> IsolatedRuntimeContext {
        _ = _cleanupHandlersRegistered
        let runtimeRootURL = fileManager.temporaryDirectory
            .appendingPathComponent("ValarTTSDocumentSessions", isDirectory: true)
            .appendingPathComponent(UUID().uuidString, isDirectory: true)

        do {
            try bootstrapIsolatedRuntime(
                at: runtimeRootURL,
                from: sharedAppPaths,
                fileManager: fileManager
            )
        } catch {
            fatalError("Unable to bootstrap isolated document runtime: \(error.localizedDescription)")
        }

        return IsolatedRuntimeContext(
            services: ValarServiceHub.live(appPaths: ValarAppPaths(baseURL: runtimeRootURL)),
            runtimeRootURL: runtimeRootURL
        )
    }

    private static func bootstrapIsolatedRuntime(
        at runtimeRootURL: URL,
        from sharedAppPaths: ValarAppPaths,
        fileManager: FileManager
    ) throws {
        let isolatedAppPaths = ValarAppPaths(baseURL: runtimeRootURL)
        try fileManager.createDirectory(at: runtimeRootURL, withIntermediateDirectories: true)
        try copyDatabaseIfPresent(
            from: sharedAppPaths.databaseURL,
            to: isolatedAppPaths.databaseURL,
            fileManager: fileManager
        )
        try copyItemIfPresent(
            from: sharedAppPaths.voiceLibraryDirectory,
            to: isolatedAppPaths.voiceLibraryDirectory,
            fileManager: fileManager
        )
    }

    private static func copyItemIfPresent(
        from sourceURL: URL,
        to destinationURL: URL,
        fileManager: FileManager
    ) throws {
        guard fileManager.fileExists(atPath: sourceURL.path) else {
            return
        }

        let parentDirectory = destinationURL.deletingLastPathComponent()
        try fileManager.createDirectory(at: parentDirectory, withIntermediateDirectories: true)

        try fileManager.copyItem(at: sourceURL, to: destinationURL)
    }

    private static func copyDatabaseIfPresent(
        from sourceURL: URL,
        to destinationURL: URL,
        fileManager: FileManager
    ) throws {
        try copyItemIfPresent(
            from: sourceURL,
            to: destinationURL,
            fileManager: fileManager
        )

        for suffix in ["-wal", "-shm"] {
            try copyItemIfPresent(
                from: URL(fileURLWithPath: sourceURL.path + suffix),
                to: URL(fileURLWithPath: destinationURL.path + suffix),
                fileManager: fileManager
            )
        }
    }

    private static func resolvedBundleModelID(
        projectModelID: ModelIdentifier?,
        renderJobs: [RenderJobRecord]
    ) -> String? {
        if let projectModelID, !projectModelID.rawValue.isEmpty {
            return projectModelID.rawValue
        }
        return renderJobs.last(where: { !$0.modelID.isEmpty })?.modelID
    }

    private static func makeRenderJobRecord(from job: RenderJob, chapterIDs: [UUID]) -> RenderJobRecord {
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

    private static func normalizedOutputFileName(from job: RenderJob) -> String {
        let trimmedTitle = (job.title ?? "render")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let baseName = ProjectRenderService.sanitizeFileStem(
            trimmedTitle.isEmpty ? "render" : trimmedTitle
        )
        return "\(baseName)-\(job.id.uuidString).wav"
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

    func markEdited() {
        guard hasLoadedIntoServices else { return }
        objectWillChange.send()
        refreshCachedSnapshot()
    }

    private func refreshCachedSnapshot() {
        snapshotRefreshTask?.cancel()
        snapshotRefreshTask = Task { @MainActor [weak self] in
            guard let self, !Task.isCancelled else { return }
            let projectID = self.appState.currentProject?.id ?? self.bundle.snapshot.project.id
            let preferredModelID = self.appState.projectRenderModelID ?? self.appState.selectedModelID
            let preferredSynthesisOptions = self.appState.projectRenderSynthesisOptions
            let createdAt = self.bundle.manifest.createdAt
            let version = self.bundle.manifest.version
            do {
                let freshBundle = try await Self.currentBundle(
                    from: self.services,
                    projectID: projectID,
                    preferredModelID: preferredModelID,
                    preferredSynthesisOptions: preferredSynthesisOptions,
                    createdAt: createdAt,
                    version: version
                )
                guard !Task.isCancelled else { return }
                self.cachedSnapshot = freshBundle
            } catch {
                // Snapshot refresh failed — keep previous cached version.
                // The document system will use the last good snapshot on save.
            }
        }
    }

    private func beginTrackingDocumentChanges() {
        guard !isTrackingDocumentChanges else { return }
        isTrackingDocumentChanges = true
        trackDocumentMetadataChanges()
    }

    private func trackDocumentMetadataChanges() {
        withObservationTracking { [weak self] in
            guard let self else { return }
            _ = self.appState.selectedModelID
            _ = self.appState.projectRenderModelID
            _ = self.appState.projectRenderSynthesisOptions
            _ = self.appState.currentProject
            _ = self.appState.voiceLibraryState.voices
        } onChange: { [weak self] in
            Task { @MainActor [weak self] in
                guard let self else { return }
                self.markEdited()
                self.trackDocumentMetadataChanges()
            }
        }
    }
}

struct DocumentEditAction {
    private let handler: @MainActor @Sendable () -> Void

    init(_ handler: @escaping @MainActor @Sendable () -> Void) {
        self.handler = handler
    }

    @MainActor
    func callAsFunction() {
        handler()
    }
}

private struct DocumentEditActionKey: EnvironmentKey {
    static let defaultValue: DocumentEditAction? = nil
}

extension EnvironmentValues {
    var documentEditAction: DocumentEditAction? {
        get { self[DocumentEditActionKey.self] }
        set { self[DocumentEditActionKey.self] = newValue }
    }
}

struct ValarProjectDocumentSceneView: View {
    @ObservedObject var document: ValarProjectDocument
    let fileURL: URL?

    var body: some View {
        AppShellView()
            .environment(document.appState)
            .environment(\.documentEditAction, DocumentEditAction {
                document.markEdited()
            })
            .task(id: fileURL) {
                await document.prepareForEditing(fileURL: fileURL)
            }
            .onDisappear {
                let projectID = document.bundle.snapshot.project.id
                Task {
                    await document.services.runtime.closeDocumentSession(for: projectID)
                }
            }
    }
}
