import Foundation
import ValarCore
import ValarPersistence

/// Process-local session registry that bridges HTTP session IDs to ValarRuntime document sessions.
/// Sessions survive in-memory but are not persisted across daemon restarts.
actor SessionManager {

    // MARK: - Types

    private struct SessionEntry {
        let projectID: UUID
        let bundleURL: URL
        var lastAccessedAt: Date
    }

    private struct SessionCreation {
        let sessionID: UUID
        let projectID: UUID
        let bundleURL: URL
    }

    /// Sessions idle longer than this are evicted on the next prune pass.
    static let sessionTTL: TimeInterval = 60 * 60 // 1 hour

    // MARK: - State

    private var sessions: [UUID: SessionEntry] = [:]
    private var openingSessions: [String: Task<SessionCreation, Error>] = [:]

    // MARK: - Session lifecycle

    /// Opens an existing `.valarproject` bundle or creates a new one at `rawPath`.
    /// Returns a session ID that callers use for subsequent requests.
    /// If the bundle URL is already open the existing session ID is returned.
    func openOrCreate(path rawPath: String, runtime: ValarRuntime) async throws -> UUID {
        let bundleURL = try Self.normalizedProjectBundleURL(rawPath, runtime: runtime)
        let bundleKey = bundleURL.path

        // Return existing session ID if this bundle is already registered.
        if let existing = sessions.first(where: { $0.value.bundleURL.path == bundleKey }) {
            sessions[existing.key]?.lastAccessedAt = Date()
            return existing.key
        }

        if openingSessions[bundleKey] == nil {
            let sessionID = UUID()
            openingSessions[bundleKey] = Task {
                try await Self.createSession(
                    sessionID: sessionID,
                    bundleURL: bundleURL,
                    runtime: runtime
                )
            }
        }

        guard let creationTask = openingSessions[bundleKey] else {
            throw SessionManagerError.sessionOpeningFailed
        }

        do {
            let creation = try await creationTask.value
            if let existing = sessions.first(where: { $0.value.bundleURL.path == bundleKey }) {
                openingSessions[bundleKey] = nil
                sessions[existing.key]?.lastAccessedAt = Date()
                return existing.key
            }

            sessions[creation.sessionID] = SessionEntry(
                projectID: creation.projectID,
                bundleURL: creation.bundleURL,
                lastAccessedAt: Date()
            )
            openingSessions[bundleKey] = nil
            return creation.sessionID
        } catch {
            openingSessions[bundleKey] = nil
            if let existing = sessions.first(where: { $0.value.bundleURL.path == bundleKey }) {
                sessions[existing.key]?.lastAccessedAt = Date()
                return existing.key
            }
            throw error
        }
    }

    private nonisolated static func normalizedProjectBundleURL(
        _ rawPath: String,
        runtime: ValarRuntime
    ) throws -> URL {
        let standardized = URL(fileURLWithPath: rawPath).standardizedFileURL
        try ValarAppPaths.validateContainment(standardized, within: runtime.paths.projectsDirectory)
        return try canonicalizedURL(standardized, fileManager: .default)
    }

    private nonisolated static func canonicalizedURL(
        _ url: URL,
        fileManager: FileManager
    ) throws -> URL {
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

    private nonisolated static func createSession(
        sessionID: UUID,
        bundleURL: URL,
        runtime: ValarRuntime
    ) async throws -> SessionCreation {
        let projectID: UUID

        if FileManager.default.fileExists(atPath: bundleURL.path) {
            let reader = ProjectBundleReader()
            let bundle = try reader.read(from: bundleURL)
            let docSession = await runtime.createDocumentSession(for: bundle)
            projectID = await docSession.projectID()
        } else {
            let title = bundleURL.deletingPathExtension().lastPathComponent
            let project = try await runtime.projectStore.create(
                title: title.isEmpty ? "Untitled" : title,
                notes: nil
            )
            projectID = project.id
            let bundle = ProjectBundle(
                manifest: ProjectBundleManifest(
                    projectID: project.id,
                    title: project.title,
                    chapters: []
                ),
                snapshot: ProjectBundleSnapshot(
                    project: project,
                    chapters: [],
                    renderJobs: [],
                    exports: []
                )
            )
            _ = await runtime.createDocumentSession(for: bundle)
        }

        return SessionCreation(sessionID: sessionID, projectID: projectID, bundleURL: bundleURL)
    }

    /// Returns the project ID and bundle URL for the given session ID, updating its last-accessed timestamp.
    func entry(for sessionID: UUID) throws -> (projectID: UUID, bundleURL: URL) {
        guard let e = sessions[sessionID] else {
            throw SessionManagerError.sessionNotFound(sessionID)
        }
        sessions[sessionID]?.lastAccessedAt = Date()
        return (e.projectID, e.bundleURL)
    }

    /// Removes the session from the registry and tears down the document session in the runtime.
    func close(sessionID: UUID, runtime: ValarRuntime) async {
        guard let entry = sessions.removeValue(forKey: sessionID) else { return }
        await runtime.closeDocumentSession(for: entry.projectID)
    }

    /// Evicts sessions that have been idle longer than `sessionTTL` and tears them down in the runtime.
    func pruneStale(runtime: ValarRuntime) async {
        let cutoff = Date().addingTimeInterval(-Self.sessionTTL)
        let stale = sessions.filter { $0.value.lastAccessedAt < cutoff }.map(\.key)
        for sessionID in stale {
            await close(sessionID: sessionID, runtime: runtime)
        }
        if !stale.isEmpty {
            print("[SessionManager] Pruned \(stale.count) idle session(s).")
        }
    }
}

// MARK: - Errors

enum SessionManagerError: LocalizedError, Sendable {
    case sessionNotFound(UUID)
    case sessionOpeningFailed

    var errorDescription: String? {
        switch self {
        case .sessionNotFound(let id):
            return "Session '\(id.uuidString)' not found."
        case .sessionOpeningFailed:
            return "Session could not be opened."
        }
    }
}
