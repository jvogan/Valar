import AppKit
import SwiftUI
import ValarCore
import ValarPersistence
import os

enum RuntimeComputeUnits: String, CaseIterable, Codable, Identifiable, Sendable {
    case automatic
    case cpuOnly
    case cpuAndGPU
    case cpuAndNeuralEngine
    case all

    var id: String { rawValue }

    var title: String {
        switch self {
        case .automatic:
            "Automatic"
        case .cpuOnly:
            "CPU Only"
        case .cpuAndGPU:
            "CPU and GPU"
        case .cpuAndNeuralEngine:
            "CPU and Neural Engine"
        case .all:
            "All Available"
        }
    }
}

struct AppSettingsSnapshot: Codable, Equatable, Sendable {
    var runtimeComputeUnits: RuntimeComputeUnits
    var runtimeThreadCount: Int
    var runtimeMemoryBudgetGiB: Double
    var modelsDirectoryPath: String
    var defaultSampleRate: Int
    var outputDeviceID: String
    var bufferSize: Int
    var debugLoggingEnabled: Bool
    var telemetryOptOutEnabled: Bool
    var showNonCommercialModels: Bool

    private enum CodingKeys: String, CodingKey {
        case runtimeComputeUnits
        case runtimeThreadCount
        case runtimeMemoryBudgetGiB
        case modelsDirectoryPath
        case defaultSampleRate
        case outputDeviceID
        case bufferSize
        case debugLoggingEnabled
        case telemetryOptOutEnabled
        case showNonCommercialModels
    }

    init(
        runtimeComputeUnits: RuntimeComputeUnits,
        runtimeThreadCount: Int,
        runtimeMemoryBudgetGiB: Double,
        modelsDirectoryPath: String,
        defaultSampleRate: Int,
        outputDeviceID: String,
        bufferSize: Int,
        debugLoggingEnabled: Bool,
        telemetryOptOutEnabled: Bool,
        showNonCommercialModels: Bool
    ) {
        self.runtimeComputeUnits = runtimeComputeUnits
        self.runtimeThreadCount = runtimeThreadCount
        self.runtimeMemoryBudgetGiB = runtimeMemoryBudgetGiB
        self.modelsDirectoryPath = modelsDirectoryPath
        self.defaultSampleRate = defaultSampleRate
        self.outputDeviceID = outputDeviceID
        self.bufferSize = bufferSize
        self.debugLoggingEnabled = debugLoggingEnabled
        self.telemetryOptOutEnabled = telemetryOptOutEnabled
        self.showNonCommercialModels = showNonCommercialModels
    }

    init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        runtimeComputeUnits = try container.decode(RuntimeComputeUnits.self, forKey: .runtimeComputeUnits)
        runtimeThreadCount = try container.decode(Int.self, forKey: .runtimeThreadCount)
        runtimeMemoryBudgetGiB = try container.decode(Double.self, forKey: .runtimeMemoryBudgetGiB)
        modelsDirectoryPath = try container.decode(String.self, forKey: .modelsDirectoryPath)
        defaultSampleRate = try container.decode(Int.self, forKey: .defaultSampleRate)
        outputDeviceID = try container.decode(String.self, forKey: .outputDeviceID)
        bufferSize = try container.decode(Int.self, forKey: .bufferSize)
        debugLoggingEnabled = try container.decode(Bool.self, forKey: .debugLoggingEnabled)
        telemetryOptOutEnabled = try container.decode(Bool.self, forKey: .telemetryOptOutEnabled)
        showNonCommercialModels = try container.decodeIfPresent(Bool.self, forKey: .showNonCommercialModels) ?? false
    }
}

struct AppSettingsDefaults: Equatable, Sendable {
    let snapshot: AppSettingsSnapshot
    let cacheDirectoryURL: URL

    init(
        appPaths: ValarAppPaths,
        runtimeConfiguration: RuntimeConfiguration,
        processorCount: Int = ProcessInfo.processInfo.activeProcessorCount
    ) {
        let threadCount = max(1, min(processorCount, 32))
        let memoryBudgetGiB = max(
            1,
            Int(
                round(
                    Double(runtimeConfiguration.maxResidentBytes)
                        / Double(1_024 * 1_024 * 1_024)
                )
            )
        )

        self.snapshot = AppSettingsSnapshot(
            runtimeComputeUnits: .automatic,
            runtimeThreadCount: threadCount,
            runtimeMemoryBudgetGiB: Double(memoryBudgetGiB),
            modelsDirectoryPath: appPaths.modelPacksDirectory.path,
            defaultSampleRate: 24_000,
            outputDeviceID: AudioOutputDevice.systemDefaultID,
            bufferSize: 1_024,
            debugLoggingEnabled: false,
            telemetryOptOutEnabled: false,
            showNonCommercialModels: false
        )
        self.cacheDirectoryURL = appPaths.cacheDirectory
    }
}

struct AppSettingsStore {
    static let storageKey = "valar.settings.v1"

    private let userDefaults: UserDefaults

    init(userDefaults: UserDefaults) {
        self.userDefaults = userDefaults
    }

    func load(defaults: AppSettingsSnapshot) -> AppSettingsSnapshot {
        guard let data = userDefaults.data(forKey: Self.storageKey) else {
            return defaults
        }

        guard var snapshot = try? JSONDecoder().decode(AppSettingsSnapshot.self, from: data) else {
            return defaults
        }
        snapshot.modelsDirectoryPath = defaults.modelsDirectoryPath
        return snapshot
    }

    func save(_ snapshot: AppSettingsSnapshot) {
        do {
            let data = try JSONEncoder().encode(snapshot)
            userDefaults.set(data, forKey: Self.storageKey)
        } catch {
            Logger(subsystem: "com.valar.tts", category: "SettingsState")
                .error("Failed to encode settings snapshot: \(error.localizedDescription, privacy: .private)")
        }
    }

    func reset() {
        userDefaults.removeObject(forKey: Self.storageKey)
    }
}

@Observable
@MainActor
final class SettingsState {
    var runtimeComputeUnits: RuntimeComputeUnits { didSet { persistIfNeeded() } }
    var runtimeThreadCount: Int { didSet { persistIfNeeded() } }
    var runtimeMemoryBudgetGiB: Double { didSet { persistIfNeeded() } }
    var modelsDirectoryPath: String { didSet { persistIfNeeded() } }
    var defaultSampleRate: Int { didSet { persistIfNeeded() } }
    var outputDeviceID: String { didSet { persistIfNeeded() } }
    var bufferSize: Int { didSet { persistIfNeeded() } }
    var debugLoggingEnabled: Bool { didSet { persistIfNeeded() } }
    var telemetryOptOutEnabled: Bool { didSet { persistIfNeeded() } }
    var showNonCommercialModels: Bool {
        didSet {
            applyCatalogVisibilityEnvironment()
            persistIfNeeded()
            if oldValue != showNonCommercialModels {
                Task { @MainActor [weak self] in
                    guard let self else { return }
                    await self.onCatalogVisibilityChanged?(showNonCommercialModels)
                }
            }
        }
    }

    var outputDevices: [AudioOutputDevice] = []
    var cacheSizeDescription = "Zero KB"
    var statusMessage = "Changes are saved automatically."
    var onCatalogVisibilityChanged: (@MainActor @Sendable (Bool) async -> Void)?

    let defaults: AppSettingsDefaults

    private let store: AppSettingsStore
    private let fileManager: FileManager
    private let outputDevicesProvider: @MainActor () -> [AudioOutputDevice]
    private var isApplyingSnapshot = false

    init(
        appPaths: ValarAppPaths = ValarAppPaths(),
        runtimeConfiguration: RuntimeConfiguration = RuntimeConfiguration(),
        userDefaults: UserDefaults = .standard,
        fileManager: FileManager = .default,
        outputDevicesProvider: @escaping @MainActor () -> [AudioOutputDevice] = { SystemAudioDeviceCatalog.outputDevices() }
    ) {
        self.defaults = AppSettingsDefaults(
            appPaths: appPaths,
            runtimeConfiguration: runtimeConfiguration
        )
        self.store = AppSettingsStore(userDefaults: userDefaults)
        self.fileManager = fileManager
        self.outputDevicesProvider = outputDevicesProvider

        let snapshot = store.load(defaults: defaults.snapshot)
        self.runtimeComputeUnits = snapshot.runtimeComputeUnits
        self.runtimeThreadCount = snapshot.runtimeThreadCount
        self.runtimeMemoryBudgetGiB = snapshot.runtimeMemoryBudgetGiB
        self.modelsDirectoryPath = snapshot.modelsDirectoryPath
        self.defaultSampleRate = snapshot.defaultSampleRate
        self.outputDeviceID = snapshot.outputDeviceID
        self.bufferSize = snapshot.bufferSize
        self.debugLoggingEnabled = snapshot.debugLoggingEnabled
        self.telemetryOptOutEnabled = snapshot.telemetryOptOutEnabled
        self.showNonCommercialModels = snapshot.showNonCommercialModels

        refreshOutputDevices(persistSelection: false)
        refreshCacheSize()
        applyCatalogVisibilityEnvironment()
    }

    convenience init(
        services: ValarServiceHub,
        userDefaults: UserDefaults = .standard,
        fileManager: FileManager = .default,
        outputDevicesProvider: @escaping @MainActor () -> [AudioOutputDevice] = { SystemAudioDeviceCatalog.outputDevices() }
    ) {
        self.init(
            appPaths: services.appPaths,
            runtimeConfiguration: services.runtimeConfiguration,
            userDefaults: userDefaults,
            fileManager: fileManager,
            outputDevicesProvider: outputDevicesProvider
        )
    }

    func chooseModelsDirectory() {
        modelsDirectoryPath = defaults.snapshot.modelsDirectoryPath
        statusMessage = "Custom model directories are not yet supported in the Mac app."
    }

    func clearCache() {
        do {
            if fileManager.fileExists(atPath: defaults.cacheDirectoryURL.path) {
                let entries = try fileManager.contentsOfDirectory(
                    at: defaults.cacheDirectoryURL,
                    includingPropertiesForKeys: nil
                )
                for entry in entries {
                    try fileManager.removeItem(at: entry)
                }
            }
            try fileManager.createDirectory(
                at: defaults.cacheDirectoryURL,
                withIntermediateDirectories: true
            )

            refreshCacheSize()
            statusMessage = "Cleared cached data."
        } catch {
            statusMessage = "Failed to clear cache: \(PathRedaction.redactMessage(error.localizedDescription))"
        }
    }

    func resetToDefaults() {
        store.reset()
        apply(defaults.snapshot, persist: false)
        refreshOutputDevices(persistSelection: true)
        refreshCacheSize()
        statusMessage = "Restored default settings."
    }

    func refreshCacheSize() {
        let bytes = Self.directorySize(at: defaults.cacheDirectoryURL, fileManager: fileManager)
        cacheSizeDescription = ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
    }

    private func persistIfNeeded() {
        guard !isApplyingSnapshot else { return }
        store.save(snapshot())
    }

    private func apply(_ snapshot: AppSettingsSnapshot, persist: Bool) {
        isApplyingSnapshot = true
        runtimeComputeUnits = snapshot.runtimeComputeUnits
        runtimeThreadCount = snapshot.runtimeThreadCount
        runtimeMemoryBudgetGiB = snapshot.runtimeMemoryBudgetGiB
        modelsDirectoryPath = snapshot.modelsDirectoryPath
        defaultSampleRate = snapshot.defaultSampleRate
        outputDeviceID = snapshot.outputDeviceID
        bufferSize = snapshot.bufferSize
        debugLoggingEnabled = snapshot.debugLoggingEnabled
        telemetryOptOutEnabled = snapshot.telemetryOptOutEnabled
        showNonCommercialModels = snapshot.showNonCommercialModels
        isApplyingSnapshot = false

        if persist {
            store.save(self.snapshot())
        }
    }

    private func refreshOutputDevices(persistSelection: Bool) {
        let devices = outputDevicesProvider()
        outputDevices = devices.isEmpty ? [AudioOutputDevice.systemDefault] : devices

        let resolvedSelection = outputDevices.contains(where: { $0.id == outputDeviceID })
            ? outputDeviceID
            : AudioOutputDevice.systemDefaultID

        if resolvedSelection != outputDeviceID {
            outputDeviceID = resolvedSelection
        }

        if persistSelection {
            store.save(snapshot())
        }
    }

    private func snapshot() -> AppSettingsSnapshot {
        AppSettingsSnapshot(
            runtimeComputeUnits: runtimeComputeUnits,
            runtimeThreadCount: max(1, min(runtimeThreadCount, 32)),
            runtimeMemoryBudgetGiB: max(1, min(runtimeMemoryBudgetGiB, 64)),
            modelsDirectoryPath: defaults.snapshot.modelsDirectoryPath,
            defaultSampleRate: defaultSampleRate,
            outputDeviceID: outputDeviceID,
            bufferSize: bufferSize,
            debugLoggingEnabled: debugLoggingEnabled,
            telemetryOptOutEnabled: telemetryOptOutEnabled,
            showNonCommercialModels: showNonCommercialModels
        )
    }

    private func applyCatalogVisibilityEnvironment() {
        if showNonCommercialModels {
            setenv(CatalogVisibilityPolicy.nonCommercialEnvVarName, "1", 1)
        } else {
            unsetenv(CatalogVisibilityPolicy.nonCommercialEnvVarName)
        }
    }

    static func applyPersistedCatalogVisibilityEnvironment(
        userDefaults: UserDefaults = .standard
    ) {
        let defaults = AppSettingsDefaults(
            appPaths: ValarAppPaths(),
            runtimeConfiguration: RuntimeConfiguration()
        ).snapshot
        let snapshot = AppSettingsStore(userDefaults: userDefaults).load(defaults: defaults)
        if snapshot.showNonCommercialModels {
            setenv(CatalogVisibilityPolicy.nonCommercialEnvVarName, "1", 1)
        } else {
            unsetenv(CatalogVisibilityPolicy.nonCommercialEnvVarName)
        }
    }

    private static func directorySize(at url: URL, fileManager: FileManager) -> Int64 {
        guard let enumerator = fileManager.enumerator(
            at: url,
            includingPropertiesForKeys: [.isRegularFileKey, .fileSizeKey],
            options: [.skipsHiddenFiles]
        ) else {
            return 0
        }

        var totalSize: Int64 = 0
        for case let fileURL as URL in enumerator {
            guard let values = try? fileURL.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey]),
                  values.isRegularFile == true,
                  let fileSize = values.fileSize else {
                continue
            }
            totalSize += Int64(fileSize)
        }
        return totalSize
    }
}
