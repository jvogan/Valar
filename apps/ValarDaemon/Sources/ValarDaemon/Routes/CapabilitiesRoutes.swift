import Foundation
import Hummingbird
import ValarCore

extension ValarDaemonRouter {
    static func registerCapabilitiesRoutes(
        on router: HTTPRouteGroup,
        runtime: ValarRuntime
    ) {
        router.get("capabilities") { _, _ async throws -> Response in
            do {
                let metallibAvailable = checkMetallibAvailable()
                let ready = await runtime.daemonReadyStatus()
                let snapshot = try await runtime.capabilitySnapshot(
                    daemonReachable: true,
                    daemonReady: ready.ready,
                    metallibAvailable: metallibAvailable
                )
                return try jsonResponse(snapshot)
            } catch {
                return daemonErrorResponse(
                    message: "Failed to serialize daemon capability snapshot.",
                    status: .internalServerError
                )
            }
        }
    }

    private static func checkMetallibAvailable() -> Bool {
        let binaryURL = URL(fileURLWithPath: CommandLine.arguments[0]).standardizedFileURL
        let binaryDir = binaryURL.deletingLastPathComponent()
        let candidates = [
            binaryDir.appendingPathComponent("default.metallib"),
            binaryDir.appendingPathComponent("mlx.metallib"),
        ]
        for url in candidates {
            if FileManager.default.fileExists(atPath: url.path) {
                return true
            }
        }
        return false
    }
}
