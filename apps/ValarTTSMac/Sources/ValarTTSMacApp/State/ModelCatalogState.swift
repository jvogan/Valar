import SwiftUI
import ValarCore
import ValarModelKit
import ValarPersistence
import os

@Observable
@MainActor
final class ModelCatalogState {
    private static let logger = Logger(subsystem: "com.valar.tts", category: "ModelCatalog")

    private struct InstallTaskEntry {
        let id: UUID
        let task: Task<Void, Never>
    }

    var catalogModels: [CatalogModel] = []
    var searchText = ""
    var selectedModel: CatalogModel?
    var downloads: [ModelIdentifier: Double] = [:]
    var installErrors: [ModelIdentifier: String] = [:]
    private(set) var diskUsageBytes: [ModelIdentifier: Int] = [:]
    private var processMemoryFootprintBytes = 0
    private var availableSystemMemoryBytes = 0
    private var installTasks: [ModelIdentifier: InstallTaskEntry] = [:]
    var onCatalogMutation: (@MainActor @Sendable () async -> Void)?

    var filteredModels: [CatalogModel] {
        guard !searchText.isEmpty else { return catalogModels }
        let query = searchText.lowercased()
        return catalogModels.filter {
            $0.descriptor.displayName.lowercased().contains(query)
            || $0.id.rawValue.lowercased().contains(query)
            || $0.familyID.rawValue.lowercased().contains(query)
        }
    }

    var installedModels: [CatalogModel] {
        filteredModels.filter { $0.installState == .installed }
    }

    var cachedModels: [CatalogModel] {
        filteredModels.filter { $0.installState == .cached }
    }

    var availableModels: [CatalogModel] {
        filteredModels.filter { $0.installState == .supported }
    }

    var memoryUsedBytes: Int {
        processMemoryFootprintBytes
    }

    var memoryBudgetBytes: Int {
        guard availableSystemMemoryBytes > 0 else { return 0 }
        return processMemoryFootprintBytes + availableSystemMemoryBytes
    }

    private let services: ValarServiceHub
    private var progressObserverTask: Task<Void, Never>?

    init(services: ValarServiceHub) {
        self.services = services
        self.progressObserverTask = Task { [weak self, services] in
            guard let self else { return }
            for await event in services.modelInstaller.progress {
                await MainActor.run {
                    self.apply(progressEvent: event)
                }
            }
        }
    }

    func apply(snapshot: ValarDashboardSnapshot, selectedModelID: ModelIdentifier?) {
        catalogModels = snapshot.catalogModels
        if let selectedModelID {
            selectedModel = snapshot.catalogModel(for: selectedModelID)
        } else if let selectedModel,
                  snapshot.catalogModel(for: selectedModel.id) == nil {
            self.selectedModel = nil
        }
        refreshResourceMetrics()
    }

    func loadCatalog() async {
        let refreshedCatalog: [CatalogModel]
        do {
            refreshedCatalog = try await services.modelCatalog.refresh()
        } catch {
            Self.logger.error("Failed to refresh model catalog: \(error.localizedDescription, privacy: .private)")
            refreshedCatalog = []
        }
        catalogModels = refreshedCatalog
        if let selectedModel {
            self.selectedModel = refreshedCatalog.first(where: { $0.id == selectedModel.id })
        }
        refreshResourceMetrics()
    }

    func startInstall(_ model: CatalogModel) {
        let modelID = model.id
        guard installTasks[modelID] == nil else { return }
        let installID = UUID()

        installTasks[modelID] = InstallTaskEntry(id: installID, task: Task { @MainActor [weak self, modelID, installID] in
            guard let self else { return }
            guard let model = self.catalogModel(for: modelID) else {
                self.finishInstallTask(for: modelID, installID: installID)
                return
            }
            await self.installModel(model)
            self.finishInstallTask(for: modelID, installID: installID)
        })
    }

    func cancelInstall(_ model: CatalogModel) {
        let modelID = model.id
        installTasks.removeValue(forKey: modelID)?.task.cancel()
        downloads[modelID] = nil
        installErrors[modelID] = nil
    }

    func installModel(_ model: CatalogModel) async {
        installErrors[model.id] = nil
        guard let manifest = await installManifest(for: model) else {
            downloads[model.id] = nil
            installErrors[model.id] = "Model manifest unavailable — cannot install"
            Self.logger.error("Manifest unavailable for model \(model.id.rawValue, privacy: .public); install aborted")
            return
        }

        let sourceKind: ModelPackSourceKind = model.providerURL == nil ? .localFile : .remoteURL
        let sourceLocation = model.providerURL?.absoluteString ?? model.id.rawValue
        let mode: ModelInstallMode = sourceKind == .remoteURL ? .downloadArtifacts : .metadataOnly

        downloads[model.id] = 0
        do {
            _ = try await services.modelInstaller.install(
                manifest: manifest,
                sourceKind: sourceKind,
                sourceLocation: sourceLocation,
                notes: model.notes,
                mode: mode
            )
            installErrors[model.id] = nil
        } catch is CancellationError {
            downloads[model.id] = nil
            installErrors[model.id] = nil
        } catch {
            downloads[model.id] = nil
            installErrors[model.id] = error.localizedDescription
        }
        await loadCatalog()
        await onCatalogMutation?()
    }

    func uninstallModel(_ model: CatalogModel) async {
        installErrors[model.id] = nil
        do {
            _ = try await services.modelInstaller.uninstall(descriptor: model.descriptor)
            await loadCatalog()
            await onCatalogMutation?()
        } catch {
            Self.logger.error("Failed to uninstall model \(model.id.rawValue, privacy: .public): \(error.localizedDescription, privacy: .private)")
            installErrors[model.id] = error.localizedDescription
        }
    }

    func diskUsageBytes(for model: CatalogModel) -> Int {
        diskUsageBytes[model.id] ?? 0
    }

    private func installManifest(for model: CatalogModel) async -> ValarPersistence.ModelPackManifest? {
        do {
            if let persistedManifest = try await services.modelPackRegistry.manifest(for: model.id.rawValue) {
                return persistedManifest
            }
        } catch {
            Self.logger.error("Failed to fetch manifest for \(model.id.rawValue, privacy: .public): \(error.localizedDescription, privacy: .private)")
        }

        return try? await services.modelCatalog.installationManifest(for: model.id)
    }

    private func apply(progressEvent event: ModelInstallProgressEvent) {
        let modelID = ModelIdentifier(event.modelID)

        switch event.status {
        case .starting:
            downloads[modelID] = max(event.progress, 0.01)
        case .downloading, .verifying:
            downloads[modelID] = event.progress
        case .completed, .failed:
            downloads[modelID] = nil
        }
    }

    private func catalogModel(for modelID: ModelIdentifier) -> CatalogModel? {
        catalogModels.first(where: { $0.id == modelID }) ?? selectedModel.flatMap { model in
            model.id == modelID ? model : nil
        }
    }

    private func finishInstallTask(for modelID: ModelIdentifier, installID: UUID) {
        guard installTasks[modelID]?.id == installID else { return }
        installTasks[modelID] = nil
    }

    private func refreshResourceMetrics() {
        processMemoryFootprintBytes = SystemResourceMonitor.currentProcessFootprintBytes()
        availableSystemMemoryBytes = SystemResourceMonitor.availableMemoryBytes()
        diskUsageBytes = Dictionary(
            uniqueKeysWithValues: catalogModels.map { model in
                (model.id, SystemResourceMonitor.diskUsageBytes(at: model.installedPath))
            }
        )
    }
}
