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

    /// Sessions idle longer than this are evicted on the next prune pass.
    static let sessionTTL: TimeInterval = 60 * 60 // 1 hour

    // MARK: - State

    private var sessions: [UUID: SessionEntry] = [:]

    // MARK: - Session lifecycle

    /// Opens an existing `.valarproject` bundle or creates a new one at `rawPath`.
    /// Returns a session ID that callers use for subsequent requests.
    /// If the bundle URL is already open the existing session ID is returned.
    func openOrCreate(path rawPath: String, runtime: ValarRuntime) async throws -> UUID {
        let bundleURL = URL(fileURLWithPath: rawPath).standardizedFileURL

        // Reject paths outside the allowed projects directory to prevent path traversal.
        try ValarAppPaths.validateContainment(bundleURL, within: runtime.paths.projectsDirectory)

        // Return existing session ID if this bundle is already registered.
        if let existing = sessions.first(where: { $0.value.bundleURL == bundleURL }) {
            sessions[existing.key]?.lastAccessedAt = Date()
            return existing.key
        }

        let sessionID = UUID()
        let projectID: UUID

        if FileManager.default.fileExists(atPath: bundleURL.path) {
            let reader = ProjectBundleReader()
            let bundle = try reader.read(from: bundleURL)
            let docSession = await runtime.createDocumentSession(for: bundle)
            projectID = await docSession.projectID()
        } else {
            // Create a fresh project and seed an empty bundle in the runtime.
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

        // After await — re-check for duplicate (another request may have inserted during our await)
        if let existing = sessions.first(where: { $0.value.bundleURL == bundleURL }) {
            // Another caller already created a session for this bundle during our await
            return existing.key
        }

        sessions[sessionID] = SessionEntry(projectID: projectID, bundleURL: bundleURL, lastAccessedAt: Date())
        return sessionID
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

    var errorDescription: String? {
        switch self {
        case .sessionNotFound(let id):
            return "Session '\(id.uuidString)' not found."
        }
    }
}
