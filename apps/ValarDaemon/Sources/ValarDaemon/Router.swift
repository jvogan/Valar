import Hummingbird
import Foundation
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
