import Foundation
import ValarMLX
import ValarModelKit

protocol RuntimeBackendInventory: Sendable {
    var availableBackendKinds: Set<BackendKind> { get }
}

protocol ModelIDUnloadingInferenceBackend: Sendable {
    func unloadModel(withID identifier: ModelIdentifier) async
}

protocol ModelSessionTrackingInferenceBackend: Sendable {
    func beginModelSession(for identifier: ModelIdentifier) async
    func endModelSession(for identifier: ModelIdentifier) async
}

extension MLXInferenceBackend: ModelIDUnloadingInferenceBackend {}

extension MLXInferenceBackend: ModelSessionTrackingInferenceBackend {
    public func beginModelSession(for identifier: ModelIdentifier) async {
        beginSession(for: identifier)
    }

    public func endModelSession(for identifier: ModelIdentifier) async {
        endSession(for: identifier)
    }
}

public enum CompositeInferenceBackendError: LocalizedError, Sendable, Equatable {
    case missingBackend(BackendKind)

    public var errorDescription: String? {
        switch self {
        case .missingBackend(let backendKind):
            return "No inference backend is registered for '\(backendKind.rawValue)'."
        }
    }
}

public actor CompositeInferenceBackend: InferenceBackend, RuntimeBackendInventory, ModelIDUnloadingInferenceBackend, ModelSessionTrackingInferenceBackend {
    public nonisolated let backendKind: BackendKind
    public nonisolated let availableBackendKinds: Set<BackendKind>
    public nonisolated let runtimeCapabilities: BackendCapabilities

    private let backends: [BackendKind: any InferenceBackend]

    public init(primary: any InferenceBackend, additional: [any InferenceBackend] = []) {
        var backendsByKind: [BackendKind: any InferenceBackend] = [primary.backendKind: primary]
        for backend in additional {
            backendsByKind[backend.backendKind] = backend
        }

        self.backendKind = primary.backendKind
        self.availableBackendKinds = Set(backendsByKind.keys)
        self.runtimeCapabilities = Self.mergeCapabilities(Array(backendsByKind.values))
        self.backends = backendsByKind
    }

    public func validate(requirement: BackendRequirement) async throws {
        guard let backend = backends[requirement.backendKind] else {
            throw CompositeInferenceBackendError.missingBackend(requirement.backendKind)
        }
        try await backend.validate(requirement: requirement)
    }

    public func prewarm(
        descriptor: ModelDescriptor,
        configuration: ModelRuntimeConfiguration
    ) async {
        guard let backend = backends[configuration.backendKind] else { return }
        await backend.prewarm(descriptor: descriptor, configuration: configuration)
    }

    public func loadModel(
        descriptor: ModelDescriptor,
        configuration: ModelRuntimeConfiguration
    ) async throws -> any ValarModel {
        guard let backend = backends[configuration.backendKind] else {
            throw CompositeInferenceBackendError.missingBackend(configuration.backendKind)
        }
        return try await backend.loadModel(descriptor: descriptor, configuration: configuration)
    }

    public func unloadModel(_ model: any ValarModel) async throws {
        guard let backend = backends[model.backendKind] else {
            throw CompositeInferenceBackendError.missingBackend(model.backendKind)
        }
        try await backend.unloadModel(model)
    }

    public func unloadModel(withID identifier: ModelIdentifier) async {
        for backend in backends.values {
            guard let unloadingBackend = backend as? any ModelIDUnloadingInferenceBackend else {
                continue
            }
            await unloadingBackend.unloadModel(withID: identifier)
        }
    }

    public func beginModelSession(for identifier: ModelIdentifier) async {
        for backend in backends.values {
            guard let sessionTrackingBackend = backend as? any ModelSessionTrackingInferenceBackend else {
                continue
            }
            await sessionTrackingBackend.beginModelSession(for: identifier)
        }
    }

    public func endModelSession(for identifier: ModelIdentifier) async {
        for backend in backends.values {
            guard let sessionTrackingBackend = backend as? any ModelSessionTrackingInferenceBackend else {
                continue
            }
            await sessionTrackingBackend.endModelSession(for: identifier)
        }
    }

    private static func mergeCapabilities(_ backends: [any InferenceBackend]) -> BackendCapabilities {
        let features = backends.reduce(into: Set<BackendFeatureID>()) { partial, backend in
            partial.formUnion(backend.runtimeCapabilities.features)
        }
        let families = backends.reduce(into: Set<ModelFamilyID>()) { partial, backend in
            partial.formUnion(backend.runtimeCapabilities.supportedFamilies)
        }
        let maximumConcurrentSessions = backends
            .compactMap(\.runtimeCapabilities.maximumConcurrentSessions)
            .reduce(0, +)

        return BackendCapabilities(
            features: features,
            supportedFamilies: families,
            maximumConcurrentSessions: maximumConcurrentSessions == 0 ? nil : maximumConcurrentSessions
        )
    }
}
