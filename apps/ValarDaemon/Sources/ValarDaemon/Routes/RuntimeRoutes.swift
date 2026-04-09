import Foundation
import Hummingbird
import ValarCore

extension ValarDaemonRouter {
    static func registerReadyRoutes(
        on router: HTTPRouteGroup,
        runtime: ValarRuntime
    ) {
        router.get("ready") { _, _ async throws -> Response in
            do {
                let dto = await runtime.daemonReadyStatus()
                let status: HTTPResponse.Status = dto.ready ? .ok : .serviceUnavailable
                return try jsonResponse(dto, status: status)
            } catch {
                return daemonErrorResponse(
                    message: "Failed to serialize daemon readiness snapshot.",
                    status: .internalServerError
                )
            }
        }
    }

    static func registerRuntimeRoutes(
        on router: HTTPRouteGroup,
        runtime: ValarRuntime,
        startedAt: Date
    ) {
        router.get("runtime") { _, _ async throws -> Response in
            do {
                let dto = await runtime.daemonRuntimeStatus(startedAt: startedAt)
                return try jsonResponse(dto)
            } catch {
                return daemonErrorResponse(
                    message: "Failed to serialize daemon runtime snapshot.",
                    status: .internalServerError
                )
            }
        }

        router.post("runtime/trim") { request, _ async throws -> Response in
            let payload: DaemonRuntimeTrimRequestDTO
            do {
                let body = try await request.body.collect(upTo: 1_048_576)
                payload = try JSONDecoder().decode(DaemonRuntimeTrimRequestDTO.self, from: body)
            } catch {
                return daemonErrorResponse(
                    message: "Invalid runtime trim payload. Expected JSON with optional 'modelIDs' and 'includeWarmStartModels'.",
                    status: .badRequest,
                    kind: "runtime_trim_error"
                )
            }

            let result = await runtime.trimRouteRuntimeResidents(
                startedAt: startedAt,
                modelIDs: payload.modelIDs,
                includeWarmStartModels: payload.includeWarmStartModels
            )
            return try jsonResponse(result)
        }
    }
}
