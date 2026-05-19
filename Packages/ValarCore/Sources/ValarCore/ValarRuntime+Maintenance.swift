import Foundation
import ValarModelKit
import ValarPersistence

public struct RuntimeStartupMaintenanceReport: Sendable, Equatable {
    public let modelPackState: ModelPackStateReconciliationReport
    public let voiceLibrary: VoiceLibraryMaintenanceReport

    public init(
        modelPackState: ModelPackStateReconciliationReport = .init(),
        voiceLibrary: VoiceLibraryMaintenanceReport = .init()
    ) {
        self.modelPackState = modelPackState
        self.voiceLibrary = voiceLibrary
    }

    public var didChangeState: Bool {
        modelPackState.didChangeState || voiceLibrary.didUpgradeVoices
    }
}

public struct ModelPackStateReconciliationReport: Sendable, Equatable {
    public let removedStaleModelIDs: [ModelIdentifier]
    public let orphanedModelPackPaths: [String]

    public init(
        removedStaleModelIDs: [ModelIdentifier] = [],
        orphanedModelPackPaths: [String] = []
    ) {
        self.removedStaleModelIDs = removedStaleModelIDs
        self.orphanedModelPackPaths = orphanedModelPackPaths
    }

    public var didChangeState: Bool {
        !removedStaleModelIDs.isEmpty || !orphanedModelPackPaths.isEmpty
    }
}

public struct VoiceLibraryMaintenanceReport: Sendable, Equatable {
    public let upgradedReusableQwenClonePromptVoiceIDs: [UUID]

    public init(upgradedReusableQwenClonePromptVoiceIDs: [UUID] = []) {
        self.upgradedReusableQwenClonePromptVoiceIDs = upgradedReusableQwenClonePromptVoiceIDs
    }

    public var didUpgradeVoices: Bool {
        !upgradedReusableQwenClonePromptVoiceIDs.isEmpty
    }
}

actor RuntimeStartupMaintenanceCoordinator {
    private var task: Task<RuntimeStartupMaintenanceReport, Never>?

    func value(
        start: @escaping @Sendable () async -> RuntimeStartupMaintenanceReport
    ) async -> RuntimeStartupMaintenanceReport {
        if let task {
            return await task.value
        }

        let newTask = Task {
            await start()
        }
        task = newTask
        return await newTask.value
    }
}

public extension ValarRuntime {
    func hydrateInstalledCatalogDescriptors() async {
        _ = await ensureStartupMaintenance()
        let catalogModels: [CatalogModel]
        do {
            catalogModels = try await modelCatalog.refresh()
        } catch {
            return
        }

        let policy = BackendSelectionPolicy()
        let runtime = BackendSelectionPolicy.Runtime(
            availableBackends: [inferenceBackend.backendKind]
        )

        for model in catalogModels where model.installState == .installed {
            let runtimeConfiguration = try? policy.runtimeConfiguration(
                for: model.descriptor,
                runtime: runtime
            )
            await modelRegistry.register(
                model.descriptor,
                estimatedResidentBytes: runtimeConfiguration?.memoryBudgetBytes,
                runtimeConfiguration: runtimeConfiguration
            )
            await capabilityRegistry.register(model.descriptor)
        }
    }

    func ensureStartupMaintenance() async -> RuntimeStartupMaintenanceReport {
        await startupMaintenanceCoordinator.value {
            let modelPackState = (try? await self.reconcileLocalModelPackState()) ?? .init()
            let voiceLibrary = await self.performVoiceLibraryMaintenance()
            return RuntimeStartupMaintenanceReport(
                modelPackState: modelPackState,
                voiceLibrary: voiceLibrary
            )
        }
    }

    func auditLocalModelPackState(
        fileManager: FileManager = .default
    ) async throws -> ModelPackStateReconciliationReport {
        let models = try await modelCatalog.supportedModels()
        let staleModelIDs = try await staleInstalledModelIDs(from: models)
        let orphanedModelPackPaths = try await self.orphanedModelPackPaths(fileManager: fileManager)
        return ModelPackStateReconciliationReport(
            removedStaleModelIDs: staleModelIDs.sorted { $0.rawValue < $1.rawValue },
            orphanedModelPackPaths: orphanedModelPackPaths.sorted()
        )
    }

    func reconcileLocalModelPackState(
        fileManager: FileManager = .default
    ) async throws -> ModelPackStateReconciliationReport {
        let report = try await auditLocalModelPackState(fileManager: fileManager)

        for modelID in report.removedStaleModelIDs {
            _ = try await modelPackRegistry.uninstall(modelID: modelID.rawValue)
            await modelRegistry.unregister(modelID)
            await capabilityRegistry.unregister(modelID)
        }

        return report
    }

    func removeOrphanedModelPacks(
        paths orphanedPaths: [String],
        fileManager: FileManager = .default
    ) async throws -> [String] {
        let currentOrphans = Set(try await orphanedModelPackPaths(fileManager: fileManager))
        var removedPaths: [String] = []
        for path in orphanedPaths {
            let url = URL(fileURLWithPath: path, isDirectory: true).standardizedFileURL
            guard fileManager.fileExists(atPath: url.path) else { continue }
            try validateModelPackDirectoryForRemoval(url, fileManager: fileManager)
            guard currentOrphans.contains(url.path) else {
                continue
            }
            try fileManager.removeItem(at: url)
            removedPaths.append(url.path)
            try pruneEmptyModelPackDirectories(
                startingAt: url,
                fileManager: fileManager
            )
        }
        return removedPaths.sorted()
    }

    func installedModelPackPaths() async throws -> Set<String> {
        let receipts = try await modelPackRegistry.receipts()
        var paths: Set<String> = []
        for receipt in receipts {
            do {
                let packDirectory = try self.paths.modelPackDirectory(
                    familyID: receipt.familyID,
                    modelID: receipt.modelID
                )
                paths.formUnion(modelPackPathComparisonKeys(for: packDirectory))
            } catch {
                continue
            }
        }
        return paths
    }

    func performVoiceLibraryMaintenance(
        fileManager: FileManager = .default
    ) async -> VoiceLibraryMaintenanceReport {
        var upgradedVoiceIDs: [UUID] = []
        let voices = await voiceStore.list()

        for voice in voices {
            let kind = voice.resolvedVoiceKind
            guard voice.inferredFamilyID == ModelFamilyID.qwen3TTS.rawValue,
                  kind == .clonePrompt || kind == .embeddingOnly,
                  voice.hasReusableQwenClonePrompt == false,
                  voice.referenceAudioAssetName != nil,
                  voice.referenceTranscript?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false
            else {
                continue
            }

            do {
                let upgradedVoice = try await upgradeVoiceForSynthesisIfNeeded(voice)
                if upgradedVoice.hasReusableQwenClonePrompt {
                    upgradedVoiceIDs.append(upgradedVoice.id)
                }
            } catch {
                continue
            }
        }

        return VoiceLibraryMaintenanceReport(
            upgradedReusableQwenClonePromptVoiceIDs: upgradedVoiceIDs.sorted { $0.uuidString < $1.uuidString }
        )
    }
}

private extension ValarRuntime {
    func staleInstalledModelIDs(
        from models: [CatalogModel]
    ) async throws -> [ModelIdentifier] {
        var staleIDs: [ModelIdentifier] = []
        for model in models {
            guard let status = model.installPathStatus, !status.isValid else {
                continue
            }
            guard try await modelPackRegistry.installedRecord(for: model.id.rawValue) != nil else {
                continue
            }
            staleIDs.append(model.id)
        }
        return staleIDs
    }

    func orphanedModelPackPaths(fileManager: FileManager) async throws -> [String] {
        let registeredPaths = try await installedModelPackPaths()
        let root = paths.modelPacksDirectory
        guard fileManager.fileExists(atPath: root.path) else {
            return []
        }

        let familyDirectories = try fileManager.contentsOfDirectory(
            at: root,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        )

        var orphanedPaths: [String] = []
        for familyDirectory in familyDirectories {
            var isDirectory: ObjCBool = false
            guard fileManager.fileExists(atPath: familyDirectory.path, isDirectory: &isDirectory), isDirectory.boolValue else {
                continue
            }
            guard !ValarAppPaths.isSymbolicLink(familyDirectory, fileManager: fileManager) else {
                continue
            }
            try ValarAppPaths.validateContainment(
                familyDirectory,
                within: root,
                fileManager: fileManager
            )

            let modelDirectories = try fileManager.contentsOfDirectory(
                at: familyDirectory,
                includingPropertiesForKeys: nil,
                options: [.skipsHiddenFiles]
            )
            for modelDirectory in modelDirectories {
                guard !ValarAppPaths.isSymbolicLink(modelDirectory, fileManager: fileManager) else {
                    continue
                }
                try ValarAppPaths.validateContainment(
                    modelDirectory,
                    within: root,
                    fileManager: fileManager
                )
                guard fileManager.fileExists(
                    atPath: modelDirectory.appendingPathComponent("manifest.json").path
                ) else {
                    continue
                }
                let pathKeys = modelPackPathComparisonKeys(for: modelDirectory)
                if registeredPaths.isDisjoint(with: pathKeys) {
                    let standardizedPath = modelDirectory.standardizedFileURL.path
                    orphanedPaths.append(standardizedPath)
                }
            }
        }

        return orphanedPaths
    }

    func modelPackPathComparisonKeys(for url: URL) -> Set<String> {
        let standardized = url.standardizedFileURL
        return [
            standardized.path,
            standardized.resolvingSymlinksInPath().standardizedFileURL.path
        ]
    }

    func validateModelPackDirectoryForRemoval(_ url: URL, fileManager: FileManager) throws {
        let root = paths.modelPacksDirectory
        try ValarAppPaths.validateContainment(url, within: root, fileManager: fileManager)
        try ValarAppPaths.validateDirectoryIsNotSymbolicLink(url, fileManager: fileManager)
    }

    func pruneEmptyModelPackDirectories(
        startingAt url: URL,
        fileManager: FileManager = .default
    ) throws {
        let root = paths.modelPacksDirectory.resolvingSymlinksInPath().standardizedFileURL
        var current = url.deletingLastPathComponent().resolvingSymlinksInPath().standardizedFileURL
        while current != root {
            try ValarAppPaths.validateContainment(current, within: root, fileManager: fileManager)
            try ValarAppPaths.validateDirectoryIsNotSymbolicLink(current, fileManager: fileManager)
            let contents = try fileManager.contentsOfDirectory(
                at: current,
                includingPropertiesForKeys: nil,
                options: [.skipsHiddenFiles]
            )
            guard contents.isEmpty else { break }
            try fileManager.removeItem(at: current)
            let parent = current.deletingLastPathComponent()
            if parent.path == current.path { break }
            current = parent
        }
    }
}

public extension ValarAppPaths {
    var daemonPIDFileURL: URL {
        applicationSupport.appendingPathComponent("valarttsd.pid", isDirectory: false)
    }
}
