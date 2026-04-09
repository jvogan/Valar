import Darwin
import Darwin.Mach
import Dispatch
import Foundation
import os
import ValarAudio
import ValarModelKit
import ValarPersistence

public enum WarmPolicy: String, Codable, Sendable {
    case lazy
    case eager
}

public struct ModelResidencyBudget: Codable, Sendable, Equatable {
    public let maxResidentBytes: Int
    public let maxResidentModels: Int

    public init(
        maxResidentBytes: Int = 10 * 1_024 * 1_024 * 1_024,
        maxResidentModels: Int = 2
    ) {
        self.maxResidentBytes = maxResidentBytes
        self.maxResidentModels = maxResidentModels
    }
}

public struct RuntimeConfiguration: Codable, Sendable, Equatable {
    public static let warmPolicyEnvVarName = "VALARTTS_WARM_POLICY"
    public static let prewarmEnvVarName = "VALARTTS_PREWARM"
    public static let warmStartModelsEnvVarName = "VALARTTS_WARM_START_MODELS"
    public static let idleTrimSettleGraceSecondsEnvVarName = "VALARTTS_IDLE_TRIM_SETTLE_GRACE_SECONDS"
    public static let idleTrimRecentUseGraceSecondsEnvVarName = "VALARTTS_IDLE_TRIM_RECENT_USE_GRACE_SECONDS"
    public static let defaultIdleTrimSettleGraceSeconds: TimeInterval = 300
    public static let defaultIdleTrimRecentUseGraceSeconds: TimeInterval = 600

    public let warmPolicy: WarmPolicy
    public let residencyBudget: ModelResidencyBudget
    public let maxQueuedRenderJobs: Int
    public let warmStartModelIDs: [ModelIdentifier]?
    public let idleTrimSettleGraceSeconds: TimeInterval
    public let idleTrimRecentUseGraceSeconds: TimeInterval

    public var maxResidentBytes: Int { residencyBudget.maxResidentBytes }
    public var maxResidentModels: Int { residencyBudget.maxResidentModels }

    public init(
        warmPolicy: WarmPolicy = .lazy,
        maxResidentBytes: Int = 10 * 1_024 * 1_024 * 1_024,
        maxResidentModels: Int = 2,
        maxQueuedRenderJobs: Int = 16,
        warmStartModelIDs: [ModelIdentifier]? = nil,
        idleTrimSettleGraceSeconds: TimeInterval = RuntimeConfiguration.defaultIdleTrimSettleGraceSeconds,
        idleTrimRecentUseGraceSeconds: TimeInterval = RuntimeConfiguration.defaultIdleTrimRecentUseGraceSeconds
    ) {
        self.warmPolicy = warmPolicy
        self.residencyBudget = ModelResidencyBudget(
            maxResidentBytes: maxResidentBytes,
            maxResidentModels: maxResidentModels
        )
        self.maxQueuedRenderJobs = maxQueuedRenderJobs
        self.warmStartModelIDs = warmStartModelIDs
        self.idleTrimSettleGraceSeconds = max(0, idleTrimSettleGraceSeconds)
        self.idleTrimRecentUseGraceSeconds = max(0, idleTrimRecentUseGraceSeconds)
    }

    public static func configured(
        from environment: [String: String],
        defaultWarmPolicy: WarmPolicy = .lazy
    ) -> RuntimeConfiguration {
        let defaults = RuntimeConfiguration()
        let warmPolicy: WarmPolicy
        if let raw = environment[warmPolicyEnvVarName]?.trimmingCharacters(in: .whitespacesAndNewlines),
           !raw.isEmpty {
            warmPolicy = Self.parseWarmPolicy(raw, default: defaultWarmPolicy)
        } else if let raw = environment[prewarmEnvVarName] {
            warmPolicy = CatalogVisibilityPolicy.parseBooleanFlag(raw) ? .eager : .lazy
        } else {
            warmPolicy = defaultWarmPolicy
        }

        let warmStartModelIDs: [ModelIdentifier]? = environment[warmStartModelsEnvVarName]?
            .split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            .map { ModelIdentifier($0) }
            .nonEmpty
        let idleTrimSettleGraceSeconds = parsePositiveTimeInterval(
            environment[idleTrimSettleGraceSecondsEnvVarName],
            defaultValue: defaultIdleTrimSettleGraceSeconds
        )
        let idleTrimRecentUseGraceSeconds = parsePositiveTimeInterval(
            environment[idleTrimRecentUseGraceSecondsEnvVarName],
            defaultValue: defaultIdleTrimRecentUseGraceSeconds
        )

        return RuntimeConfiguration(
            warmPolicy: warmPolicy,
            maxResidentBytes: defaults.maxResidentBytes,
            maxResidentModels: defaults.maxResidentModels,
            maxQueuedRenderJobs: defaults.maxQueuedRenderJobs,
            warmStartModelIDs: warmStartModelIDs,
            idleTrimSettleGraceSeconds: idleTrimSettleGraceSeconds,
            idleTrimRecentUseGraceSeconds: idleTrimRecentUseGraceSeconds
        )
    }

    private static func parseWarmPolicy(
        _ rawValue: String,
        `default`: WarmPolicy
    ) -> WarmPolicy {
        switch rawValue.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "eager", "warm", "prewarm", "1", "true", "yes", "on":
            return .eager
        case "lazy", "cold", "0", "false", "no", "off":
            return .lazy
        default:
            return `default`
        }
    }

    private static func parsePositiveTimeInterval(
        _ rawValue: String?,
        defaultValue: TimeInterval
    ) -> TimeInterval {
        guard let trimmed = rawValue?.trimmingCharacters(in: .whitespacesAndNewlines),
              !trimmed.isEmpty,
              let value = TimeInterval(trimmed),
              value >= 0 else {
            return defaultValue
        }
        return value
    }
}

private extension Array {
    var nonEmpty: Self? {
        isEmpty ? nil : self
    }
}

public enum RuntimeResourceMonitor {
    public static func currentProcessFootprintBytes() -> Int {
        guard let taskVMInfoRev1Offset = MemoryLayout.offset(of: \task_vm_info_data_t.min_address) else {
            assertionFailure(
                "RuntimeResourceMonitor could not determine task_vm_info_data_t.min_address offset."
            )
            return 0
        }

        let taskVMInfoRev1Count = mach_msg_type_number_t(
            taskVMInfoRev1Offset / MemoryLayout<integer_t>.size
        )
        let taskVMInfoCount = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size
        )
        var info = task_vm_info_data_t()
        var count = taskVMInfoCount

        let result = withUnsafeMutablePointer(to: &info) { infoPointer in
            infoPointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { integerPointer in
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), integerPointer, &count)
            }
        }

        guard result == KERN_SUCCESS else {
            assertionFailure("RuntimeResourceMonitor.task_info(TASK_VM_INFO) failed with status \(result).")
            return 0
        }

        guard count >= taskVMInfoRev1Count else {
            assertionFailure(
                "RuntimeResourceMonitor.task_info(TASK_VM_INFO) returned \(count) words; expected at least \(taskVMInfoRev1Count)."
            )
            return 0
        }

        return Int(clamping: info.phys_footprint)
    }

    public static func currentProcessCPUTimeSeconds() -> Double? {
        var usage = rusage()
        guard getrusage(RUSAGE_SELF, &usage) == 0 else {
            return nil
        }

        let userSeconds = Double(usage.ru_utime.tv_sec) + Double(usage.ru_utime.tv_usec) / 1_000_000
        let systemSeconds = Double(usage.ru_stime.tv_sec) + Double(usage.ru_stime.tv_usec) / 1_000_000
        return userSeconds + systemSeconds
    }

    public static func currentProcessCPUPercent(uptimeSeconds: Double) -> Double? {
        guard uptimeSeconds > 0,
              let cpuSeconds = currentProcessCPUTimeSeconds() else {
            return nil
        }
        return max(0, (cpuSeconds / uptimeSeconds) * 100)
    }

    public static func availableDiskBytes(at url: URL) -> Int? {
        guard let values = try? url.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]),
              let capacity = values.volumeAvailableCapacityForImportantUsage else {
            return nil
        }
        return Int(clamping: capacity)
    }
}

public struct ModelResidencySnapshot: Codable, Sendable, Equatable {
    public let descriptor: ModelDescriptor
    public let state: ModelResidencyState
    public let lastTouchedAt: Date?
    public let residentRank: Int?
    public let estimatedResidentBytes: Int
    public let actualResidentBytes: Int?
    public let residencyPolicy: ResidencyPolicy
    public let activeSessionCount: Int

    public init(
        descriptor: ModelDescriptor,
        state: ModelResidencyState,
        lastTouchedAt: Date? = nil,
        residentRank: Int? = nil,
        estimatedResidentBytes: Int = 0,
        actualResidentBytes: Int? = nil,
        residencyPolicy: ResidencyPolicy = .automatic,
        activeSessionCount: Int = 0
    ) {
        self.descriptor = descriptor
        self.state = state
        self.lastTouchedAt = lastTouchedAt
        self.residentRank = residentRank
        self.estimatedResidentBytes = estimatedResidentBytes
        self.actualResidentBytes = actualResidentBytes
        self.residencyPolicy = residencyPolicy
        self.activeSessionCount = activeSessionCount
    }
}

public enum ModelMemoryPressureLevel: String, Codable, Sendable, Equatable {
    case warning
    case critical
}

public enum ModelResidencyEvictionTrigger: String, Codable, Sendable, Equatable {
    case budgetExceeded
    case idleTrim
    case memoryPressureWarning
    case memoryPressureCritical
}

public struct ModelResidencyEvictionEvent: Codable, Sendable, Equatable {
    public let descriptor: ModelDescriptor
    public let trigger: ModelResidencyEvictionTrigger
    public let reclaimedBytes: Int
    public let residentBytesAfterEviction: Int
    public let residentModelCountAfterEviction: Int
    public let occurredAt: Date

    public init(
        descriptor: ModelDescriptor,
        trigger: ModelResidencyEvictionTrigger,
        reclaimedBytes: Int,
        residentBytesAfterEviction: Int,
        residentModelCountAfterEviction: Int,
        occurredAt: Date = .now
    ) {
        self.descriptor = descriptor
        self.trigger = trigger
        self.reclaimedBytes = reclaimedBytes
        self.residentBytesAfterEviction = residentBytesAfterEviction
        self.residentModelCountAfterEviction = residentModelCountAfterEviction
        self.occurredAt = occurredAt
    }
}

public protocol ModelRegistryManaging: Sendable {
    func register(_ descriptor: ModelDescriptor) async
    func register(
        _ descriptor: ModelDescriptor,
        estimatedResidentBytes: Int?,
        runtimeConfiguration: ModelRuntimeConfiguration?
    ) async
    func unregister(_ identifier: ModelIdentifier) async
    func descriptor(for identifier: ModelIdentifier) async -> ModelDescriptor?
    func markState(_ state: ModelResidencyState, for identifier: ModelIdentifier) async
    func setResidencyPolicy(_ policy: ResidencyPolicy, for identifier: ModelIdentifier) async
    func setMeasuredResidentBytes(_ bytes: Int, for identifier: ModelIdentifier) async
    func reserveSession(for identifier: ModelIdentifier) async
    func releaseSession(for identifier: ModelIdentifier) async
    func touch(_ identifier: ModelIdentifier) async
    func handleMemoryPressure(_ level: ModelMemoryPressureLevel) async
    func evictResident(_ identifier: ModelIdentifier, trigger: ModelResidencyEvictionTrigger) async -> Bool
    func evictionEvents() async -> [ModelResidencyEvictionEvent]
    func snapshots() async -> [ModelResidencySnapshot]
    func residentModels() async -> [ModelDescriptor]
    func residentBytes() async -> Int
}

public actor ModelRegistry: ModelRegistryManaging {
    private static let logger = Logger(subsystem: "com.valar.tts", category: "ModelRegistry")

    private struct Entry: Sendable {
        var descriptor: ModelDescriptor
        var state: ModelResidencyState
        var lastTouchedAt: Date?
        var estimatedResidentBytes: Int
        var actualResidentBytes: Int?
        var runtimeConfiguration: ModelRuntimeConfiguration
        var activeSessionCount: Int
        var loadStartedFootprintBytes: Int?
    }

    private final class MemoryPressureMonitor {
        private let source: any DispatchSourceMemoryPressure

        init(handler: @escaping @Sendable (ModelMemoryPressureLevel) -> Void) {
            let source = DispatchSource.makeMemoryPressureSource(
                eventMask: [.warning, .critical],
                queue: DispatchQueue.global(qos: .utility)
            )
            self.source = source
            source.setEventHandler {
                let pressure = source.data
                if pressure.contains(.critical) {
                    handler(.critical)
                } else if pressure.contains(.warning) {
                    handler(.warning)
                }
            }
            source.resume()
        }
    }

    private let configuration: RuntimeConfiguration
    private let memoryFootprintProvider: @Sendable () -> Int
    private let evictionLogger: @Sendable (String) -> Void
    private let evictionHandler: @Sendable (ModelResidencyEvictionEvent) async -> Void
    private var entries: [ModelIdentifier: Entry]
    private var residentOrder: [ModelIdentifier]
    private var evictionHistory: [ModelResidencyEvictionEvent]
    private var memoryPressureMonitor: MemoryPressureMonitor?

    public init(
        configuration: RuntimeConfiguration = RuntimeConfiguration(),
        knownModels: [ModelDescriptor] = [],
        enableMemoryPressureMonitoring: Bool = true,
        memoryFootprintProvider: (@Sendable () -> Int)? = nil,
        evictionLogger: (@Sendable (String) -> Void)? = nil,
        evictionHandler: (@Sendable (ModelResidencyEvictionEvent) async -> Void)? = nil
    ) {
        self.configuration = configuration
        self.memoryFootprintProvider = memoryFootprintProvider ?? Self.currentProcessFootprintBytes
        self.evictionLogger = evictionLogger ?? Self.defaultEvictionLogger
        self.evictionHandler = evictionHandler ?? { _ in }
        self.entries = Dictionary(
            uniqueKeysWithValues: knownModels.map { descriptor in
                (
                    descriptor.id,
                    Entry(
                        descriptor: descriptor,
                        state: .unloaded,
                        lastTouchedAt: nil,
                        estimatedResidentBytes: Self.defaultEstimatedResidentBytes(for: descriptor),
                        actualResidentBytes: nil,
                        runtimeConfiguration: Self.defaultRuntimeConfiguration(for: descriptor),
                        activeSessionCount: 0,
                        loadStartedFootprintBytes: nil
                    )
                )
            }
        )
        self.residentOrder = []
        self.evictionHistory = []
        self.memoryPressureMonitor = nil

        if enableMemoryPressureMonitoring {
            Task {
                await self.installMemoryPressureMonitor()
            }
        }
    }

    public func register(_ descriptor: ModelDescriptor) {
        register(
            descriptor,
            estimatedResidentBytes: nil,
            runtimeConfiguration: nil
        )
    }

    public func register(
        _ descriptor: ModelDescriptor,
        estimatedResidentBytes: Int?,
        runtimeConfiguration: ModelRuntimeConfiguration?
    ) {
        let current = entries[descriptor.id]
        entries[descriptor.id] = Entry(
            descriptor: descriptor,
            state: current?.state ?? .unloaded,
            lastTouchedAt: current?.lastTouchedAt,
            estimatedResidentBytes: estimatedResidentBytes
                ?? current?.estimatedResidentBytes
                ?? Self.defaultEstimatedResidentBytes(for: descriptor),
            actualResidentBytes: current?.actualResidentBytes,
            runtimeConfiguration: runtimeConfiguration
                ?? current?.runtimeConfiguration
                ?? Self.defaultRuntimeConfiguration(for: descriptor),
            activeSessionCount: current?.activeSessionCount ?? 0,
            loadStartedFootprintBytes: current?.loadStartedFootprintBytes
        )
    }

    public func unregister(_ identifier: ModelIdentifier) {
        entries[identifier] = nil
        residentOrder.removeAll { $0 == identifier }
    }

    public func descriptor(for identifier: ModelIdentifier) -> ModelDescriptor? {
        entries[identifier]?.descriptor
    }

    public func markState(_ state: ModelResidencyState, for identifier: ModelIdentifier) async {
        guard var entry = entries[identifier] else { return }
        entry.state = state
        entry.lastTouchedAt = .now
        switch state {
        case .warming:
            entry.loadStartedFootprintBytes = memoryFootprintProvider()
        case .resident:
            if let baseline = entry.loadStartedFootprintBytes {
                let measuredBytes = max(0, memoryFootprintProvider() - baseline)
                if measuredBytes > 0 {
                    entry.actualResidentBytes = measuredBytes
                }
            }
            entry.loadStartedFootprintBytes = nil
        case .cooling:
            entry.loadStartedFootprintBytes = nil
        case .unloaded:
            entry.actualResidentBytes = 0
            entry.loadStartedFootprintBytes = nil
        }
        entries[identifier] = entry
        updateResidencyOrder(for: identifier, state: state)
        await enforceResidentBudgetIfNeeded()
    }

    public func setResidencyPolicy(_ policy: ResidencyPolicy, for identifier: ModelIdentifier) async {
        guard var entry = entries[identifier] else { return }
        let existing = entry.runtimeConfiguration
        entry.runtimeConfiguration = ModelRuntimeConfiguration(
            backendKind: existing.backendKind,
            residencyPolicy: policy,
            preferredSampleRate: existing.preferredSampleRate,
            memoryBudgetBytes: existing.memoryBudgetBytes,
            allowQuantizedWeights: existing.allowQuantizedWeights,
            allowWarmStart: existing.allowWarmStart
        )
        entries[identifier] = entry
        await enforceResidentBudgetIfNeeded()
    }

    public func setMeasuredResidentBytes(_ bytes: Int, for identifier: ModelIdentifier) async {
        guard var entry = entries[identifier] else { return }
        entry.actualResidentBytes = max(0, bytes)
        entry.lastTouchedAt = .now
        entries[identifier] = entry
        await enforceResidentBudgetIfNeeded()
    }

    public func reserveSession(for identifier: ModelIdentifier) {
        guard var entry = entries[identifier] else { return }
        entry.activeSessionCount += 1
        entry.lastTouchedAt = .now
        entries[identifier] = entry
        if entry.state == .resident {
            moveToBack(identifier)
        }
    }

    public func releaseSession(for identifier: ModelIdentifier) async {
        guard var entry = entries[identifier] else { return }
        entry.activeSessionCount = max(0, entry.activeSessionCount - 1)
        entry.lastTouchedAt = .now
        entries[identifier] = entry
        await enforceResidentBudgetIfNeeded()
    }

    public func touch(_ identifier: ModelIdentifier) {
        guard var entry = entries[identifier] else { return }
        entry.lastTouchedAt = .now
        entries[identifier] = entry
        if entry.state == .resident {
            moveToBack(identifier)
        }
    }

    public func handleMemoryPressure(_ level: ModelMemoryPressureLevel) async {
        switch level {
        case .warning:
            guard let candidate = evictionCandidate() else { return }
            await evict(candidate, trigger: .memoryPressureWarning)
            await enforceResidentBudgetIfNeeded()
        case .critical:
            while let candidate = evictionCandidate() {
                await evict(candidate, trigger: .memoryPressureCritical)
            }
        }
    }

    @discardableResult
    public func evictResident(
        _ identifier: ModelIdentifier,
        trigger: ModelResidencyEvictionTrigger
    ) async -> Bool {
        guard let entry = entries[identifier],
              entry.state == .resident,
              entry.activeSessionCount == 0 else {
            return false
        }
        let policy = entry.runtimeConfiguration.residencyPolicy
        guard policy != .pinned else {
            return false
        }
        await evict(identifier, trigger: trigger)
        return true
    }

    public func evictionEvents() -> [ModelResidencyEvictionEvent] {
        evictionHistory
    }

    public func snapshots() -> [ModelResidencySnapshot] {
        entries.values
            .map(\.descriptor)
            .sorted { $0.displayName < $1.displayName }
            .map { descriptor in
                let entry = entries[descriptor.id]
                return ModelResidencySnapshot(
                    descriptor: descriptor,
                    state: entry?.state ?? .unloaded,
                    lastTouchedAt: entry?.lastTouchedAt,
                    residentRank: residentOrder.firstIndex(of: descriptor.id),
                    estimatedResidentBytes: entry?.estimatedResidentBytes ?? 0,
                    actualResidentBytes: entry?.actualResidentBytes,
                    residencyPolicy: entry?.runtimeConfiguration.residencyPolicy ?? .automatic,
                    activeSessionCount: entry?.activeSessionCount ?? 0
                )
            }
    }

    public func residentModels() -> [ModelDescriptor] {
        residentOrder.compactMap { entries[$0]?.descriptor }
    }

    public func residentBytes() -> Int {
        residentOrder.reduce(into: 0) { result, identifier in
            if let entry = entries[identifier] {
                result += accountedResidentBytes(for: entry)
            }
        }
    }

    private func updateResidencyOrder(for identifier: ModelIdentifier, state: ModelResidencyState) {
        switch state {
        case .resident:
            moveToBack(identifier)
        case .warming, .cooling, .unloaded:
            residentOrder.removeAll { $0 == identifier }
        }
    }

    private func moveToBack(_ identifier: ModelIdentifier) {
        residentOrder.removeAll { $0 == identifier }
        residentOrder.append(identifier)
    }

    private func enforceResidentBudgetIfNeeded() async {
        guard configuration.maxResidentModels >= 0 else { return }

        while shouldEvictResident() {
            guard let evicted = evictionCandidate() else { return }
            await evict(evicted, trigger: .budgetExceeded)
        }
    }

    private func shouldEvictResident() -> Bool {
        if residentOrder.count > configuration.residencyBudget.maxResidentModels {
            return true
        }

        let totalResidentBytes = residentBytes()
        return totalResidentBytes > configuration.residencyBudget.maxResidentBytes && !residentOrder.isEmpty
    }

    private func evictionCandidate() -> ModelIdentifier? {
        residentOrder.first { identifier in
            guard let entry = entries[identifier] else { return false }
            let policy = entry.runtimeConfiguration.residencyPolicy
            return policy != .pinned && policy != .eager && entry.activeSessionCount == 0
        }
    }

    private func evict(_ identifier: ModelIdentifier, trigger: ModelResidencyEvictionTrigger) async {
        guard var entry = entries[identifier] else { return }

        let reclaimedBytes = accountedResidentBytes(for: entry)
        residentOrder.removeAll { $0 == identifier }
        entry.state = .cooling
        entry.lastTouchedAt = .now
        entry.loadStartedFootprintBytes = nil
        entries[identifier] = entry

        let event = ModelResidencyEvictionEvent(
            descriptor: entry.descriptor,
            trigger: trigger,
            reclaimedBytes: reclaimedBytes,
            residentBytesAfterEviction: residentBytes(),
            residentModelCountAfterEviction: residentOrder.count
        )
        evictionHistory.append(event)
        evictionLogger(Self.formatEvictionLog(event))
        await evictionHandler(event)

        // Re-check state after suspension — a concurrent reload may have occurred
        guard var unloadedEntry = entries[identifier] else { return }
        guard unloadedEntry.state != .resident && unloadedEntry.activeSessionCount == 0 else {
            // Model was reloaded or session started during eviction — abort
            return
        }
        unloadedEntry.state = .unloaded
        unloadedEntry.lastTouchedAt = .now
        unloadedEntry.actualResidentBytes = 0
        unloadedEntry.loadStartedFootprintBytes = nil
        entries[identifier] = unloadedEntry
    }

    private func accountedResidentBytes(for entry: Entry) -> Int {
        entry.actualResidentBytes ?? entry.estimatedResidentBytes
    }

    private func installMemoryPressureMonitor() {
        guard memoryPressureMonitor == nil else { return }
        memoryPressureMonitor = MemoryPressureMonitor { [weak self] level in
            guard let self else { return }
            Task {
                await self.handleMemoryPressure(level)
            }
        }
    }

    private static func defaultRuntimeConfiguration(for descriptor: ModelDescriptor) -> ModelRuntimeConfiguration {
        ModelRuntimeConfiguration(
            backendKind: .mlx,
            residencyPolicy: .automatic,
            preferredSampleRate: descriptor.defaultSampleRate,
            memoryBudgetBytes: defaultEstimatedResidentBytes(for: descriptor),
            allowQuantizedWeights: true,
            allowWarmStart: true
        )
    }

    private static func defaultEstimatedResidentBytes(for descriptor: ModelDescriptor) -> Int {
        switch descriptor.domain {
        case .tts:
            if descriptor.displayName.localizedCaseInsensitiveContains("voice") {
                return 6 * 1_024 * 1_024 * 1_024
            }
            return 4 * 1_024 * 1_024 * 1_024
        case .stt:
            return 2 * 1_024 * 1_024 * 1_024
        case .sts:
            return 3 * 1_024 * 1_024 * 1_024
        case .codec:
            return 1 * 1_024 * 1_024 * 1_024
        case .utility:
            return 512 * 1_024 * 1_024
        }
    }

    private static func defaultEvictionLogger(_ message: String) {
        logger.info("\(message, privacy: .public)")
    }

    private static func formatEvictionLog(_ event: ModelResidencyEvictionEvent) -> String {
        let reclaimed = ByteCountFormatter.string(
            fromByteCount: Int64(event.reclaimedBytes),
            countStyle: .memory
        )
        let remaining = ByteCountFormatter.string(
            fromByteCount: Int64(event.residentBytesAfterEviction),
            countStyle: .memory
        )
        return "[ModelRegistry] Evicted \(event.descriptor.id.rawValue) via \(event.trigger.rawValue); reclaimed \(reclaimed); remaining resident bytes \(remaining); resident models \(event.residentModelCountAfterEviction)"
    }

    private static func currentProcessFootprintBytes() -> Int {
        guard let taskVMInfoRev1Offset = MemoryLayout.offset(of: \task_vm_info_data_t.min_address) else {
            return 0
        }
        let taskVMInfoRev1Count = mach_msg_type_number_t(
            taskVMInfoRev1Offset / MemoryLayout<integer_t>.size
        )
        let taskVMInfoCount = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size
        )
        var info = task_vm_info_data_t()
        var count = taskVMInfoCount

        let result = withUnsafeMutablePointer(to: &info) { infoPointer in
            infoPointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { integerPointer in
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), integerPointer, &count)
            }
        }

        guard result == KERN_SUCCESS, count >= taskVMInfoRev1Count else {
            return 0
        }

        return Int(clamping: info.phys_footprint)
    }
}

public enum RenderJobState: String, Codable, Sendable {
    case queued
    case running
    case completed
    case cancelled
    case failed
}

public struct RenderJob: Codable, Sendable, Equatable, Identifiable {
    public let id: UUID
    public let projectID: UUID
    public let modelID: ModelIdentifier
    public let createdAt: Date
    public var chapterIDs: [UUID]
    public var outputFileName: String
    public var state: RenderJobState
    public var priority: Int
    public var progress: Double
    public var title: String?
    public var failureReason: String?
    public var queuePosition: Int
    public var synthesisOptions: RenderSynthesisOptions

    public init(
        id: UUID = UUID(),
        projectID: UUID,
        modelID: ModelIdentifier,
        chapterIDs: [UUID] = [],
        outputFileName: String? = nil,
        createdAt: Date = .now,
        state: RenderJobState = .queued,
        priority: Int = 0,
        progress: Double = 0,
        title: String? = nil,
        failureReason: String? = nil,
        queuePosition: Int = 0,
        synthesisOptions: RenderSynthesisOptions = RenderSynthesisOptions()
    ) {
        self.id = id
        self.projectID = projectID
        self.modelID = modelID
        self.chapterIDs = chapterIDs
        self.outputFileName = Self.normalizedOutputFileName(outputFileName, fallbackID: id)
        self.createdAt = createdAt
        self.state = state
        self.priority = priority
        self.progress = progress
        self.title = title
        self.failureReason = Self.normalizedFailureReason(failureReason)
        self.queuePosition = queuePosition
        self.synthesisOptions = synthesisOptions
    }

    public var chapterID: UUID? {
        chapterIDs.first
    }

    private static func normalizedOutputFileName(_ value: String?, fallbackID: UUID) -> String {
        let trimmed = value?.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let trimmed, !trimmed.isEmpty else {
            return "\(fallbackID.uuidString).wav"
        }
        return trimmed
    }

    static func normalizedFailureReason(_ value: String?) -> String? {
        let collapsed = value?
            .components(separatedBy: .whitespacesAndNewlines)
            .filter { !$0.isEmpty }
            .joined(separator: " ")
        guard let collapsed, !collapsed.isEmpty else {
            return nil
        }
        return collapsed
    }
}

public protocol RenderQueueManaging: Sendable {
    func enqueue(
        projectID: UUID,
        modelID: ModelIdentifier,
        chapterIDs: [UUID],
        outputFileName: String?,
        priority: Int,
        title: String?,
        synthesisOptions: RenderSynthesisOptions
    ) async -> RenderJob
    func transition(_ jobID: UUID, to state: RenderJobState, progress: Double?, failureReason: String?) async
    func cancel(_ jobID: UUID) async
    func reorderQueuedJobs(for projectID: UUID, orderedJobIDs: [UUID]) async
    func replaceJobs(for projectID: UUID, with jobs: [RenderJob]) async
    func job(id: UUID) async -> RenderJob?
    func jobs(matching state: RenderJobState?) async -> [RenderJob]
    func nextJob() async -> RenderJob?
    func pendingJobCount() async -> Int
}

public protocol RenderQueueStore: Sendable {
    func loadJobs() async throws -> [RenderJob]
    func save(_ job: RenderJob) async throws
    func remove(id: UUID) async throws
}

public actor InMemoryRenderQueueStore: RenderQueueStore {
    private var jobs: [UUID: RenderJob]

    public init(jobs: [RenderJob] = []) {
        self.jobs = Dictionary(uniqueKeysWithValues: jobs.map { ($0.id, $0) })
    }

    public func loadJobs() async throws -> [RenderJob] {
        jobs.values.sorted { $0.createdAt < $1.createdAt }
    }

    public func save(_ job: RenderJob) async throws {
        jobs[job.id] = job
    }

    public func remove(id: UUID) async throws {
        jobs[id] = nil
    }
}

public actor RenderQueue: RenderQueueManaging {
    private let configuration: RuntimeConfiguration
    private let store: any RenderQueueStore
    private var jobs: [UUID: RenderJob] = [:]
    private var hydrated = false
    private var updateContinuations: [UUID: AsyncStream<[RenderJob]>.Continuation] = [:]

    // Sort cache: nil means dirty, rebuilt lazily. Only invalidated when sort-relevant
    // fields change (state, priority, structural add/remove). Progress ticks reuse the cache.
    private var sortedJobIDs: [UUID]?

    // Debounce task for progress-only updates. State-change transitions cancel this and
    // publish immediately; progress-only transitions schedule a deferred publish instead.
    private var pendingProgressPublish: Task<Void, Never>?
    private static let progressPublishDelay: Duration = .milliseconds(100)

    public init(
        configuration: RuntimeConfiguration = RuntimeConfiguration(),
        store: any RenderQueueStore = InMemoryRenderQueueStore()
    ) {
        self.configuration = configuration
        self.store = store
    }

    public func enqueue(
        projectID: UUID,
        modelID: ModelIdentifier,
        chapterIDs: [UUID] = [],
        outputFileName: String? = nil,
        priority: Int = 0,
        title: String? = nil,
        synthesisOptions: RenderSynthesisOptions = RenderSynthesisOptions()
    ) async -> RenderJob {
        await hydrateIfNeeded()
        if configuration.maxQueuedRenderJobs > 0 {
            let queuedJobs = jobs.values.filter { $0.state == .queued }
            if queuedJobs.count >= configuration.maxQueuedRenderJobs,
               let oldestQueued = queuedJobs.sorted(by: { $0.createdAt < $1.createdAt }).first {
                jobs[oldestQueued.id] = nil
                try? await store.remove(id: oldestQueued.id)
            }
        }

        let job = RenderJob(
            projectID: projectID,
            modelID: modelID,
            chapterIDs: chapterIDs,
            outputFileName: outputFileName,
            priority: priority,
            title: title,
            queuePosition: nextQueuePosition(for: projectID),
            synthesisOptions: synthesisOptions
        )
        jobs[job.id] = job
        try? await store.save(job)
        sortedJobIDs = nil
        publishUpdates()
        return job
    }

    public func transition(
        _ jobID: UUID,
        to state: RenderJobState,
        progress: Double? = nil,
        failureReason: String? = nil
    ) async {
        await hydrateIfNeeded()
        guard var job = jobs[jobID] else { return }
        let stateChanged = job.state != state
        job.state = state
        if let progress {
            job.progress = progress
        }
        if state == .failed {
            if let failureReason = RenderJob.normalizedFailureReason(failureReason) {
                job.failureReason = failureReason
            } else if stateChanged {
                job.failureReason = nil
            }
        } else {
            job.failureReason = nil
        }
        jobs[jobID] = job
        try? await store.save(job)

        if stateChanged {
            // Sort order may have changed — cancel any pending debounce and publish immediately.
            pendingProgressPublish?.cancel()
            pendingProgressPublish = nil
            sortedJobIDs = nil
            publishUpdates()
        } else {
            // Progress-only update: sort order is unchanged; debounce to avoid per-tick re-sorts.
            scheduleProgressPublish()
        }
    }

    public func cancel(_ jobID: UUID) async {
        await transition(jobID, to: .cancelled, progress: 1)
    }

    public func reorderQueuedJobs(for projectID: UUID, orderedJobIDs: [UUID]) async {
        await hydrateIfNeeded()

        let currentQueuedJobs = orderedJobs().filter { job in
            job.projectID == projectID && job.state == .queued
        }
        guard !currentQueuedJobs.isEmpty else { return }

        var reorderedIDs: [UUID] = []
        reorderedIDs.reserveCapacity(currentQueuedJobs.count)

        for jobID in orderedJobIDs where currentQueuedJobs.contains(where: { $0.id == jobID }) {
            if !reorderedIDs.contains(jobID) {
                reorderedIDs.append(jobID)
            }
        }

        for job in currentQueuedJobs where !reorderedIDs.contains(job.id) {
            reorderedIDs.append(job.id)
        }

        var hasChanges = false
        for (position, jobID) in reorderedIDs.enumerated() {
            guard var job = jobs[jobID], job.queuePosition != position else { continue }
            job.queuePosition = position
            jobs[jobID] = job
            hasChanges = true
            try? await store.save(job)
        }

        guard hasChanges else { return }
        pendingProgressPublish?.cancel()
        pendingProgressPublish = nil
        sortedJobIDs = nil
        publishUpdates()
    }

    public func replaceJobs(for projectID: UUID, with replacementJobs: [RenderJob]) async {
        await hydrateIfNeeded()

        let existingJobIDs = jobs.values
            .filter { $0.projectID == projectID }
            .map(\.id)

        for jobID in existingJobIDs {
            jobs[jobID] = nil
            try? await store.remove(id: jobID)
        }

        for job in replacementJobs {
            jobs[job.id] = job
            try? await store.save(job)
        }
        pendingProgressPublish?.cancel()
        pendingProgressPublish = nil
        sortedJobIDs = nil
        publishUpdates()
    }

    public func job(id: UUID) async -> RenderJob? {
        await hydrateIfNeeded()
        return jobs[id]
    }

    public func jobs(matching state: RenderJobState? = nil) async -> [RenderJob] {
        await hydrateIfNeeded()
        return orderedJobs().filter { job in
            state.map { $0 == job.state } ?? true
        }
    }

    public func nextJob() async -> RenderJob? {
        await hydrateIfNeeded()
        return orderedJobs().first { $0.state == .queued }
    }

    public func pendingJobCount() async -> Int {
        await hydrateIfNeeded()
        return jobs.values.filter { $0.state == .queued || $0.state == .running }.count
    }

    public func updates() async -> AsyncStream<[RenderJob]> {
        await hydrateIfNeeded()
        let observerID = UUID()
        let snapshot = orderedJobs()
        let stream = AsyncStream<[RenderJob]>.makeStream()
        updateContinuations[observerID] = stream.continuation
        stream.continuation.yield(snapshot)
        stream.continuation.onTermination = { @Sendable [weak self] _ in
            Task {
                await self?.removeUpdateContinuation(observerID)
            }
        }
        return stream.stream
    }

    private func orderedJobs() -> [RenderJob] {
        if sortedJobIDs == nil {
            sortedJobIDs = jobs.values.sorted { lhs, rhs in
                if stateOrder(lhs.state) != stateOrder(rhs.state) {
                    return stateOrder(lhs.state) < stateOrder(rhs.state)
                }
                if lhs.priority != rhs.priority {
                    return lhs.priority > rhs.priority
                }
                if lhs.state == .queued,
                   rhs.state == .queued,
                   lhs.projectID == rhs.projectID,
                   lhs.queuePosition != rhs.queuePosition {
                    return lhs.queuePosition < rhs.queuePosition
                }
                return lhs.createdAt < rhs.createdAt
            }.map(\.id)
        }
        return sortedJobIDs!.compactMap { jobs[$0] }
    }

    private func scheduleProgressPublish() {
        pendingProgressPublish?.cancel()
        pendingProgressPublish = Task {
            try? await Task.sleep(for: Self.progressPublishDelay)
            guard !Task.isCancelled else { return }
            self.publishUpdates()
        }
    }

    private func hydrateIfNeeded() async {
        guard !hydrated else { return }
        do {
            let persistedJobs = try await store.loadJobs()
            jobs = Dictionary(uniqueKeysWithValues: persistedJobs.map { ($0.id, Self.recoveredJob($0)) })
            hydrated = true  // set AFTER successful load
        } catch {
            // Log the error so storage failures aren't silent
            print("RenderQueue: Failed to hydrate from store: \(error)")
            // Don't set hydrated = true — allow retry on next call
        }
    }

    private func publishUpdates() {
        let snapshot = orderedJobs()
        for continuation in updateContinuations.values {
            continuation.yield(snapshot)
        }
    }

    private func removeUpdateContinuation(_ observerID: UUID) {
        updateContinuations[observerID] = nil
    }

    private static func recoveredJob(_ job: RenderJob) -> RenderJob {
        guard job.state == .running else { return job }
        var recovered = job
        recovered.state = .queued
        recovered.progress = min(recovered.progress, 0.95)
        return recovered
    }

    private func stateOrder(_ state: RenderJobState) -> Int {
        switch state {
        case .queued: return 0
        case .running: return 1
        case .completed: return 2
        case .cancelled: return 3
        case .failed: return 4
        }
    }

    private func nextQueuePosition(for projectID: UUID) -> Int {
        jobs.values
            .filter { $0.projectID == projectID }
            .map(\.queuePosition)
            .max()
            .map { $0 + 1 } ?? 0
    }
}

public struct DictationSession: Codable, Sendable, Equatable, Identifiable {
    public let id: UUID
    public let modelID: ModelIdentifier
    public let languageHint: String?
    public var buffer: [String]
    public var startedAt: Date
    public var finishedAt: Date?

    public init(
        id: UUID = UUID(),
        modelID: ModelIdentifier,
        languageHint: String? = nil,
        buffer: [String] = [],
        startedAt: Date = .now,
        finishedAt: Date? = nil
    ) {
        self.id = id
        self.modelID = modelID
        self.languageHint = languageHint
        self.buffer = buffer
        self.startedAt = startedAt
        self.finishedAt = finishedAt
    }
}

public protocol DictationManaging: Sendable {
    func start(modelID: ModelIdentifier, languageHint: String?) async -> DictationSession
    func append(_ text: String, to sessionID: UUID) async
    func finalize(sessionID: UUID) async -> String
    func cancel(sessionID: UUID) async
    func snapshot() async -> [DictationSession]
}

public actor DictationService: DictationManaging {
    private var sessions: [UUID: DictationSession] = [:]

    public init() {}

    public func start(modelID: ModelIdentifier, languageHint: String? = nil) -> DictationSession {
        let session = DictationSession(modelID: modelID, languageHint: languageHint)
        sessions[session.id] = session
        return session
    }

    public func append(_ text: String, to sessionID: UUID) {
        guard var session = sessions[sessionID] else { return }
        session.buffer.append(text)
        sessions[sessionID] = session
    }

    public func finalize(sessionID: UUID) -> String {
        guard var session = sessions[sessionID] else { return "" }
        session.finishedAt = .now
        sessions[sessionID] = session
        let transcript = session.buffer.joined(separator: " ")
        sessions[sessionID] = nil
        return transcript
    }

    public func cancel(sessionID: UUID) {
        sessions[sessionID] = nil
    }

    public func snapshot() -> [DictationSession] {
        sessions.values.sorted { $0.startedAt < $1.startedAt }
    }
}

public protocol TranslationProvider: Sendable {
    func translate(_ request: TranslationRequest) async throws -> String
}

public protocol TranslationManaging: Sendable {
    func translate(_ request: TranslationRequest) async throws -> String
}

public actor TranslationService: TranslationManaging {
    private let provider: any TranslationProvider

    public init(provider: any TranslationProvider) {
        self.provider = provider
    }

    public func translate(_ request: TranslationRequest) async throws -> String {
        try await provider.translate(request)
    }
}
