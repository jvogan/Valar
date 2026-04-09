import Foundation

public struct BackendSelectionPolicy: Sendable {
    public enum SelectionError: Error, Equatable, Sendable {
        case noCompatibleBackend(ModelIdentifier)
    }

    public struct Runtime: Hashable, Sendable {
        public let availableBackends: Set<BackendKind>
        public let availableMemoryBytes: Int?
        public let supportsLocalExecution: Bool
        public let runtimeVersion: String?

        public init(
            availableBackends: Set<BackendKind>,
            availableMemoryBytes: Int? = nil,
            supportsLocalExecution: Bool = true,
            runtimeVersion: String? = nil
        ) {
            self.availableBackends = availableBackends
            self.availableMemoryBytes = availableMemoryBytes
            self.supportsLocalExecution = supportsLocalExecution
            self.runtimeVersion = runtimeVersion
        }

        public static var current: Self {
            let processInfo = ProcessInfo.processInfo
            let version = processInfo.operatingSystemVersion
            return Self(
                availableBackends: defaultAvailableBackends(),
                availableMemoryBytes: Int(clamping: processInfo.physicalMemory),
                supportsLocalExecution: true,
                runtimeVersion: "\(version.majorVersion).\(version.minorVersion).\(version.patchVersion)"
            )
        }

        private static func defaultAvailableBackends() -> Set<BackendKind> {
            var backends: Set<BackendKind> = [.cpu, .mock]
#if os(macOS)
            backends.insert(.coreml)
            backends.insert(.metal)
#if arch(arm64)
            backends.insert(.mlx)
#endif
#endif
            return backends
        }
    }

    public init() {}

    public func compatibleRequirement(
        for descriptor: ModelDescriptor,
        runtime: Runtime = .current
    ) -> BackendRequirement? {
        descriptor.supportedBackends.first { isCompatible($0, runtime: runtime) }
    }

    public func backend(
        for descriptor: ModelDescriptor,
        runtime: Runtime = .current
    ) throws -> BackendKind {
        guard let requirement = compatibleRequirement(for: descriptor, runtime: runtime) else {
            throw SelectionError.noCompatibleBackend(descriptor.id)
        }
        return requirement.backendKind
    }

    public func runtimeConfiguration(
        for descriptor: ModelDescriptor,
        residencyPolicy: ResidencyPolicy = .automatic,
        preferredSampleRate: Double? = nil,
        memoryBudgetBytes: Int? = nil,
        allowQuantizedWeights: Bool = true,
        allowWarmStart: Bool = true,
        runtime: Runtime = .current
    ) throws -> ModelRuntimeConfiguration {
        ModelRuntimeConfiguration(
            backendKind: try backend(for: descriptor, runtime: runtime),
            residencyPolicy: residencyPolicy,
            preferredSampleRate: preferredSampleRate ?? descriptor.defaultSampleRate,
            memoryBudgetBytes: memoryBudgetBytes,
            allowQuantizedWeights: allowQuantizedWeights,
            allowWarmStart: allowWarmStart
        )
    }

    public func isCompatible(
        _ requirement: BackendRequirement,
        runtime: Runtime = .current
    ) -> Bool {
        guard runtime.availableBackends.contains(requirement.backendKind) else {
            return false
        }
        if let minimumMemoryBytes = requirement.minimumMemoryBytes,
           let availableMemoryBytes = runtime.availableMemoryBytes,
           availableMemoryBytes < minimumMemoryBytes {
            return false
        }
        if requirement.requiresLocalExecution && !runtime.supportsLocalExecution {
            return false
        }
        if let minimumRuntimeVersion = requirement.minimumRuntimeVersion,
           let runtimeVersion = runtime.runtimeVersion,
           compareVersions(runtimeVersion, minimumRuntimeVersion) == .orderedAscending {
            return false
        }
        return true
    }

    private func compareVersions(_ lhs: String, _ rhs: String) -> ComparisonResult {
        let lhsParts = lhs.split(separator: ".").map { Int($0) ?? 0 }
        let rhsParts = rhs.split(separator: ".").map { Int($0) ?? 0 }
        let count = max(lhsParts.count, rhsParts.count)

        for index in 0..<count {
            let lhsValue = index < lhsParts.count ? lhsParts[index] : 0
            let rhsValue = index < rhsParts.count ? rhsParts[index] : 0
            if lhsValue < rhsValue {
                return .orderedAscending
            }
            if lhsValue > rhsValue {
                return .orderedDescending
            }
        }

        return .orderedSame
    }
}
