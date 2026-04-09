import Foundation
import Hummingbird
import ValarCore
import ValarPersistence

extension ValarDaemonRouter {

    // MARK: - Route registration

    static func registerSessionRoutes(
        on router: HTTPRouteGroup,
        runtime: ValarRuntime,
        sessionManager: SessionManager
    ) {
        let sessions = router.group("sessions")

        // POST /v1/sessions/new
        sessions.post("new") { request, _ async -> Response in
            do {
                let rawBody = try await request.body.collect(upTo: 1_000_000)
                let body = try JSONDecoder().decode(SessionNewRequestDTO.self, from: rawBody)
                let sessionID = try await sessionManager.openOrCreate(path: body.path, runtime: runtime)
                return try jsonResponse(SessionNewResponseDTO(sessionId: sessionID.uuidString))
            } catch {
                return sessionErrorResponse(message(for: error), status: status(for: error))
            }
        }

        // All session-scoped routes share the ":id" prefix
        let session = sessions.group(":id")

        // GET /v1/sessions/:id/chapters
        session.get("chapters") { _, context async -> Response in
            do {
                let (projectID, _) = try await sessionManager.resolvedEntry(from: context)
                guard let docSession = await runtime.documentSession(for: projectID) else {
                    throw SessionRouteError.sessionDocumentNotFound
                }
                let chapters = await docSession.chapters()
                return try jsonResponse(chapters.map(ChapterDTO.init(from:)))
            } catch {
                return sessionErrorResponse(message(for: error), status: status(for: error))
            }
        }

        // POST /v1/sessions/:id/chapters
        session.post("chapters") { request, context async -> Response in
            do {
                let (projectID, _) = try await sessionManager.resolvedEntry(from: context)
                guard let docSession = await runtime.documentSession(for: projectID) else {
                    throw SessionRouteError.sessionDocumentNotFound
                }
                let rawBody = try await request.body.collect(upTo: 1_000_000)
                let body = try JSONDecoder().decode(ChapterCreateRequestDTO.self, from: rawBody)
                let existing = await docSession.chapters()
                let resolvedIndex = body.index >= 0
                    ? body.index
                    : (existing.map(\.index).max().map { $0 + 1 } ?? 0)
                let chapter = ChapterRecord(
                    projectID: projectID,
                    index: resolvedIndex,
                    title: body.title,
                    script: body.text,
                    speakerLabel: body.speakerLabel
                )
                await docSession.addChapter(chapter)
                return try jsonResponse(ChapterDTO(from: chapter), status: .created)
            } catch {
                return sessionErrorResponse(message(for: error), status: status(for: error))
            }
        }

        // PATCH /v1/sessions/:id/chapters/:chid
        session.patch("chapters/:chid") { request, context async -> Response in
            do {
                let (projectID, _) = try await sessionManager.resolvedEntry(from: context)
                guard let docSession = await runtime.documentSession(for: projectID) else {
                    throw SessionRouteError.sessionDocumentNotFound
                }
                let chidString = try context.parameters.require("chid")
                guard let chapterID = UUID(uuidString: chidString) else {
                    throw SessionRouteError.invalidChapterID(chidString)
                }
                let rawBody = try await request.body.collect(upTo: 1_000_000)
                let body = try JSONDecoder().decode(ChapterUpdateRequestDTO.self, from: rawBody)
                let chapters = await docSession.chapters()
                guard var chapter = chapters.first(where: { $0.id == chapterID }) else {
                    throw SessionRouteError.chapterNotFound(chidString)
                }
                if let title = body.title { chapter.title = title }
                if let text = body.text { chapter.script = text }
                await docSession.updateChapter(chapter)
                return try jsonResponse(ChapterDTO(from: chapter))
            } catch {
                return sessionErrorResponse(message(for: error), status: status(for: error))
            }
        }

        // POST /v1/sessions/:id/save
        session.post("save") { _, context async -> Response in
            do {
                let (projectID, bundleURL) = try await sessionManager.resolvedEntry(from: context)
                guard let docSession = await runtime.documentSession(for: projectID) else {
                    throw SessionRouteError.sessionDocumentNotFound
                }
                let bundle = try await docSession.snapshot(
                    preferredModelID: nil,
                    createdAt: .now,
                    version: 1
                )
                let location = ValarProjectBundleLocation(
                    projectID: bundle.snapshot.project.id,
                    title: bundle.snapshot.project.title,
                    bundleURL: bundleURL
                )
                try ProjectBundleWriter().write(bundle.snapshot, to: location)
                return try jsonResponse(OKResponseDTO(ok: true))
            } catch {
                return sessionErrorResponse(message(for: error), status: status(for: error))
            }
        }

        // POST /v1/sessions/:id/close
        session.post("close") { _, context async -> Response in
            do {
                let idString = try context.parameters.require("id")
                guard let sessionID = UUID(uuidString: idString) else {
                    throw SessionRouteError.invalidSessionID(idString)
                }
                await sessionManager.close(sessionID: sessionID, runtime: runtime)
                return try jsonResponse(OKResponseDTO(ok: true))
            } catch {
                return sessionErrorResponse(message(for: error), status: status(for: error))
            }
        }
    }

    private static func sessionErrorResponse(
        _ message: String,
        status: HTTPResponse.Status
    ) -> Response {
        daemonErrorResponse(message: message, status: status, kind: "session_error")
    }

    private static func status(for error: Error) -> HTTPResponse.Status {
        switch error {
        case let routeError as SessionRouteError:
            return routeError.httpStatus
        case let httpError as HTTPError:
            return httpError.status
        case is DecodingError, is ValarPathValidationError, is ProjectBundleError:
            return .badRequest
        default:
            return .internalServerError
        }
    }

    private static func message(for error: Error) -> String {
        switch error {
        case let routeError as SessionRouteError:
            return routeError.localizedDescription
        case let httpError as HTTPError:
            return httpError.body ?? httpError.localizedDescription
        case let bundleError as ProjectBundleError:
            return bundleError.localizedDescription
        case let validationError as ValarPathValidationError:
            return validationError.localizedDescription
        case is DecodingError:
            return "Invalid JSON request body."
        default:
            return "Session request failed due to an internal daemon error."
        }
    }
}

// MARK: - SessionManager + context helpers

private extension SessionManager {
    /// Extracts and validates the session ID from `:id` in the URL, then resolves to a session entry.
    func resolvedEntry(from context: ValarDaemonRouter.Context) async throws -> (projectID: UUID, bundleURL: URL) {
        let idString = try context.parameters.require("id")
        guard let sessionID = UUID(uuidString: idString) else {
            throw SessionRouteError.invalidSessionID(idString)
        }
        do {
            return try entry(for: sessionID)
        } catch {
            throw SessionRouteError.sessionNotFound("Session '\(sessionID.uuidString)' not found.")
        }
    }
}

private enum SessionRouteError: LocalizedError {
    case invalidSessionID(String)
    case sessionNotFound(String)
    case sessionDocumentNotFound
    case invalidChapterID(String)
    case chapterNotFound(String)

    var httpStatus: HTTPResponse.Status {
        switch self {
        case .invalidSessionID, .invalidChapterID:
            return .badRequest
        case .sessionNotFound, .sessionDocumentNotFound, .chapterNotFound:
            return .notFound
        }
    }

    var errorDescription: String? {
        switch self {
        case .invalidSessionID(let value):
            return "Invalid session ID '\(value)'"
        case .sessionNotFound(let message):
            return message
        case .sessionDocumentNotFound:
            return "Session document not found in runtime"
        case .invalidChapterID(let value):
            return "Invalid chapter ID '\(value)'"
        case .chapterNotFound(let value):
            return "Chapter '\(value)' not found"
        }
    }
}
