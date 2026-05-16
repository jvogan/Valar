import Hummingbird
import Foundation
import HTTPTypes
import NIOCore
import NIOHTTPTypes
import ValarCore
import ValarPersistence

actor ClientInputCloseRegistry {
    static let shared = ClientInputCloseRegistry()

    private var futures: [ObjectIdentifier: EventLoopFuture<Void>] = [:]

    func store(_ future: EventLoopFuture<Void>, for channel: any Channel) {
        futures[ObjectIdentifier(channel as AnyObject)] = future
    }

    func future(for channel: any Channel) -> EventLoopFuture<Void>? {
        futures[ObjectIdentifier(channel as AnyObject)]
    }

    func remove(for channel: any Channel) {
        futures.removeValue(forKey: ObjectIdentifier(channel as AnyObject))
    }
}

final class ClientInputCloseHandler: ChannelInboundHandler, RemovableChannelHandler {
    typealias InboundIn = HTTPRequestPart

    private var disconnectPromise: EventLoopPromise<Void>?

    func handlerAdded(context: ChannelHandlerContext) {
        let promise = context.eventLoop.makePromise(of: Void.self)
        self.disconnectPromise = promise
        let channel = context.channel
        let future = promise.futureResult
        Task {
            await ClientInputCloseRegistry.shared.store(
                future,
                for: channel
            )
        }
    }

    func userInboundEventTriggered(context: ChannelHandlerContext, event: Any) {
        if case ChannelEvent.inputClosed = event {
            signalDisconnect()
        }
        context.fireUserInboundEventTriggered(event)
    }

    func channelInactive(context: ChannelHandlerContext) {
        signalDisconnect()
        let channel = context.channel
        Task {
            await ClientInputCloseRegistry.shared.remove(for: channel)
        }
        context.fireChannelInactive()
    }

    private func signalDisconnect() {
        disconnectPromise?.succeed(())
        disconnectPromise = nil
    }
}

private extension HTTPField.Name {
    static let secFetchSite = Self("Sec-Fetch-Site")!
    static let secFetchMode = Self("Sec-Fetch-Mode")!
}

struct LocalDaemonRequestGuardMiddleware<Context: RequestContext>: RouterMiddleware {
    private let loopbackHosts = Set(["127.0.0.1", "::1", "[::1]", "localhost"])

    func handle(
        _ request: Request,
        context: Context,
        next: (Request, Context) async throws -> Response
    ) async throws -> Response {
        guard isAllowedHost(request.head.authority) else {
            return ValarDaemonRouter.daemonErrorResponse(
                message: "Request Host must be loopback.",
                status: .forbidden,
                kind: "local_request_guard"
            )
        }
        guard isAllowedBrowserFetchMetadata(request) else {
            return ValarDaemonRouter.daemonErrorResponse(
                message: "Cross-site browser requests are not allowed.",
                status: .forbidden,
                kind: "local_request_guard"
            )
        }
        guard isAllowedOrigin(request.headers[.origin]) else {
            return ValarDaemonRouter.daemonErrorResponse(
                message: "Cross-origin browser requests are not allowed.",
                status: .forbidden,
                kind: "local_request_guard"
            )
        }
        guard hasAllowedContentType(request) else {
            return ValarDaemonRouter.daemonErrorResponse(
                message: "Unsupported Content-Type for this endpoint.",
                status: .unsupportedMediaType,
                kind: "local_request_guard"
            )
        }

        return try await next(request, context)
    }

    private func isAllowedHost(_ hostHeader: String?) -> Bool {
        guard let hostHeader, hostHeader.isEmpty == false else {
            return true
        }
        return loopbackHosts.contains(hostWithoutPort(hostHeader).lowercased())
    }

    private func hostWithoutPort(_ value: String) -> String {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.hasPrefix("[") {
            guard let end = trimmed.firstIndex(of: "]") else { return trimmed }
            return String(trimmed[...end])
        }
        return trimmed.split(separator: ":", maxSplits: 1).first.map(String.init) ?? trimmed
    }

    private func isAllowedBrowserFetchMetadata(_ request: Request) -> Bool {
        let site = request.headers[.secFetchSite]?.lowercased()
        if site == "cross-site" || site == "same-site" {
            return false
        }
        if request.method != .get {
            let mode = request.headers[.secFetchMode]?.lowercased()
            if mode == "no-cors" || mode == "navigate" {
                return false
            }
        }
        return true
    }

    private func isAllowedOrigin(_ originHeader: String?) -> Bool {
        guard let originHeader, originHeader.isEmpty == false else {
            return true
        }
        guard let origin = URL(string: originHeader), let host = origin.host else {
            return false
        }
        return loopbackHosts.contains(host.lowercased())
    }

    private func hasAllowedContentType(_ request: Request) -> Bool {
        switch request.method {
        case .get, .head, .delete:
            return true
        default:
            guard let contentType = request.headers[.contentType]?.lowercased() else {
                return requestContentLength(request) == 0
            }
            return contentType.hasPrefix("application/json")
                || contentType.hasPrefix("multipart/form-data")
        }
    }

    private func requestContentLength(_ request: Request) -> Int {
        guard let rawLength = request.headers[.contentLength]?.trimmingCharacters(in: .whitespacesAndNewlines),
              rawLength.isEmpty == false
        else {
            return 0
        }
        return Int(rawLength) ?? -1
    }
}

enum ValarDaemonRouter {
    struct DaemonRequestContext: RequestContext {
        var coreContext: CoreRequestContextStorage
        let channel: any Channel

        init(source: Source) {
            self.coreContext = .init(source: source)
            self.channel = source.channel
        }
    }

    struct BuiltRouter {
        let router: HTTPRouter
        let backgroundTasks: [Task<Void, Never>]
    }

    typealias Context = DaemonRequestContext
    typealias HTTPRouter = Router<Context>
    typealias HTTPRouteGroup = RouterGroup<Context>

    static func build(runtime: ValarRuntime, startedAt: Date = .now) -> BuiltRouter {
        let router = HTTPRouter(context: Context.self)
        let operations = DaemonOperationQueue()
        let sessionManager = SessionManager()
        registerV1Routes(on: router, runtime: runtime, operations: operations, sessionManager: sessionManager, startedAt: startedAt)
        let evictionTask = startEvictionLoop(
            runtime: runtime,
            operations: operations,
            sessionManager: sessionManager,
            startedAt: startedAt
        )
        return BuiltRouter(router: router, backgroundTasks: [evictionTask])
    }

    /// Starts a background Task that prunes stale sessions and operations every 5 minutes.
    private static func startEvictionLoop(
        runtime: ValarRuntime,
        operations: DaemonOperationQueue,
        sessionManager: SessionManager,
        startedAt: Date
    ) -> Task<Void, Never> {
        Task {
            await sessionManager.pruneStale(runtime: runtime)
            await operations.pruneStale()
            await runtime.trimIdleNonWarmResidentsIfNeeded(startedAt: startedAt)
            await runtime.restoreMissingWarmResidentsIfNeeded()
            let interval: UInt64 = 60 * 1_000_000_000
            while true {
                try? await Task.sleep(nanoseconds: interval)
                guard !Task.isCancelled else { return }
                await sessionManager.pruneStale(runtime: runtime)
                await operations.pruneStale()
                await runtime.trimIdleNonWarmResidentsIfNeeded(startedAt: startedAt)
                await runtime.restoreMissingWarmResidentsIfNeeded()
            }
        }
    }

    private static func registerV1Routes(
        on router: HTTPRouter,
        runtime: ValarRuntime,
        operations: DaemonOperationQueue,
        sessionManager: SessionManager,
        startedAt: Date
    ) {
        let v1 = router.group("v1")
        v1.addMiddleware {
            LocalDaemonRequestGuardMiddleware<Context>()
        }
        registerHealthRoutes(on: v1, runtime: runtime)
        registerReadyRoutes(on: v1, runtime: runtime)
        registerRuntimeRoutes(on: v1, runtime: runtime, startedAt: startedAt)
        registerModelRoutes(on: v1, runtime: runtime, operations: operations)
        registerOperationRoutes(on: v1, operations: operations)
        registerVoiceRoutes(on: v1, runtime: runtime)
        registerSessionRoutes(on: v1, runtime: runtime, sessionManager: sessionManager)
        registerAudioRoutes(on: v1, runtime: runtime)
        registerAlignmentRoutes(on: v1, runtime: runtime)
        registerCapabilitiesRoutes(on: v1, runtime: runtime)
    }

    private static func registerHealthRoutes(on router: HTTPRouteGroup, runtime: ValarRuntime) {
        router.get("health") { _, _ -> Response in
            try jsonResponse(runtime.daemonHealthStatus())
        }
    }

    static func jsonResponse<T: Encodable>(_ value: T, status: HTTPResponse.Status = .ok) throws -> Response {
        let data = try JSONEncoder().encode(value)
        var body = ByteBuffer()
        body.writeBytes(data)
        return Response(status: status, headers: [.contentType: "application/json"], body: .init(byteBuffer: body))
    }

    static func daemonErrorResponse(
        message: String,
        status: HTTPResponse.Status,
        kind: String = "daemon_error",
        help: String? = nil
    ) -> Response {
        let sanitizedMessage = ValarPathRedaction.sanitizeMessage(message)
        let sanitizedHelp = help.map(ValarPathRedaction.sanitizeMessage)
        do {
            return try jsonResponse(
                DaemonErrorEnvelopeDTO(
                    error: ValarCommandErrorDTO(
                        code: Int(status.code),
                        kind: kind,
                        message: sanitizedMessage,
                        help: sanitizedHelp
                    )
                ),
                status: status
            )
        } catch {
            var body = ByteBuffer()
            body.writeString("{\"ok\":false,\"error\":{\"code\":500,\"kind\":\"daemon_error\",\"message\":\"Internal encoding error.\"}}")
            return Response(
                status: status,
                headers: [.contentType: "application/json"],
                body: .init(byteBuffer: body)
            )
        }
    }
}
