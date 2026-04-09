import Foundation
import Hummingbird
import ValarCore

extension ValarDaemonRouter {
    static func registerModelRoutes(
        on router: HTTPRouteGroup,
        runtime: ValarRuntime,
        operations: DaemonOperationQueue
    ) {
        router.get("models") { _, _ async throws -> Response in
            do {
                return try jsonResponse(await runtime.listRouteModels())
            } catch {
                return modelErrorResponse(
                    "Failed to serialize model list response.",
                    status: .internalServerError
                )
            }
        }

        router.post("models/install") { request, _ async throws -> Response in
            let payload: ModelInstallRequestDTO
            do {
                let body = try await request.body.collect(upTo: 1_048_576)
                payload = try JSONDecoder().decode(ModelInstallRequestDTO.self, from: body)
            } catch {
                return modelErrorResponse(
                    "Invalid install payload. Expected JSON with 'model' (or legacy 'model_id').",
                    status: .badRequest
                )
            }
            do {
                // Daemon only auto-downloads when allow_download=true is set explicitly.
                let catalog = try await runtime.modelCatalog.supportedModels()
                if let model = catalog.first(where: { $0.id.rawValue == payload.model }),
                   payload.allowDownload == false,
                   model.providerURL != nil,
                   (model.installState == .supported || payload.refreshCache) {
                    return modelErrorResponse(
                        "Model requires download. Retry with allow_download=true or use the CLI with --allow-download.",
                        status: .conflict,
                        kind: "download_required",
                        help: "Retry with allow_download=true or use the CLI with --allow-download."
                    )
                }

                let operation = await operations.enqueue(kind: "model.install") {
                    try await runtime.installRouteModel(
                        id: payload.model,
                        allowDownload: payload.allowDownload,
                        refreshCache: payload.refreshCache
                    )
                }
                guard let operation else {
                    return operationErrorResponse(
                        "Operation queue is full.",
                        status: .tooManyRequests,
                        help: "Retry after in-flight operations finish."
                    )
                }

                return try jsonResponse(
                    ModelInstallOperationDTO(
                        operationId: operation.operationId,
                        status: operation.status
                    )
                )
            } catch let error as RouteModelError {
                return modelErrorResponse(
                    error.localizedDescription,
                    status: status(for: error),
                    help: help(for: error)
                )
            } catch {
                return modelErrorResponse("Model installation failed due to an internal daemon error.", status: .internalServerError)
            }
        }

        router.post("models/remove") { request, _ async -> Response in
            let payload: ModelRemoveRequestDTO
            do {
                payload = try await JSONDecoder().decode(
                    ModelRemoveRequestDTO.self,
                    from: request.body.collect(upTo: 1_048_576)
                )
            } catch {
                return modelErrorResponse(
                    "Invalid remove payload. Expected JSON with 'model'.",
                    status: .badRequest
                )
            }

            do {
                try await runtime.removeRouteModel(id: payload.model)
                return try jsonResponse(OKResponseDTO(ok: true))
            } catch let error as RouteModelError {
                return modelErrorResponse(
                    error.localizedDescription,
                    status: status(for: error),
                    help: help(for: error)
                )
            } catch {
                return modelErrorResponse("Model removal failed due to an internal daemon error.", status: .internalServerError)
            }
        }

        router.post("models/purge-cache") { request, _ async -> Response in
            let payload: ModelRemoveRequestDTO
            do {
                payload = try await JSONDecoder().decode(
                    ModelRemoveRequestDTO.self,
                    from: request.body.collect(upTo: 1_048_576)
                )
            } catch {
                return modelErrorResponse(
                    "Invalid purge-cache payload. Expected JSON with 'model'.",
                    status: .badRequest
                )
            }

            do {
                let result = try await runtime.purgeRouteModelSharedCache(id: payload.model)
                return try jsonResponse(result)
            } catch let error as RouteModelError {
                return modelErrorResponse(
                    error.localizedDescription,
                    status: status(for: error),
                    help: help(for: error)
                )
            } catch {
                return modelErrorResponse("Model cache purge failed due to an internal daemon error.", status: .internalServerError)
            }
        }
    }

    static func registerOperationRoutes(
        on router: HTTPRouteGroup,
        operations: DaemonOperationQueue
    ) {
        router.get("operations/:id") { _, context async -> Response in
            do {
                guard let id = context.parameters.get("id") else {
                    return operationErrorResponse(
                        "Missing operation ID parameter.",
                        status: .badRequest
                    )
                }
                guard let operation = await operations.operation(id: id) else {
                    return operationErrorResponse(
                        "Operation '\(id)' not found.",
                        status: .notFound
                    )
                }
                return try jsonResponse(operation)
            } catch {
                return operationErrorResponse(
                    "Failed to encode operation response.",
                    status: .internalServerError
                )
            }
        }

        router.get("queue") { _, _ async throws -> Response in
            do {
                return try jsonResponse(await operations.queueState())
            } catch {
                return operationErrorResponse(
                    "Failed to serialize operation queue state.",
                    status: .internalServerError
                )
            }
        }
    }

    private static func modelErrorResponse(
        _ message: String,
        status: HTTPResponse.Status,
        kind: String = "model_error",
        help: String? = nil
    ) -> Response {
        daemonErrorResponse(message: message, status: status, kind: kind, help: help)
    }

    private static func operationErrorResponse(
        _ message: String,
        status: HTTPResponse.Status,
        help: String? = nil
    ) -> Response {
        daemonErrorResponse(message: message, status: status, kind: "operation_error", help: help)
    }

    private static func status(for error: RouteModelError) -> HTTPResponse.Status {
        switch error {
        case .modelNotFound:
            return .notFound
        case .modelHidden:
            return .forbidden
        case .refreshRequiresDownload:
            return .conflict
        }
    }

    private static func help(for error: RouteModelError) -> String? {
        switch error {
        case .modelNotFound:
            return nil
        case .modelHidden:
            return nil
        case .refreshRequiresDownload:
            return "Retry with allow_download=true or use the CLI with --allow-download."
        }
    }
}
