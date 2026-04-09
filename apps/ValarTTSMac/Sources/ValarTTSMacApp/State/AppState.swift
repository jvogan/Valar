import AppKit
import SwiftUI
import ValarCore
import ValarModelKit
import ValarPersistence

@Observable
@MainActor
final class AppState {
    var selectedSection: AppSection = .generate
    var selectedModelID: ModelIdentifier?
    var projectRenderModelID: ModelIdentifier?
    var projectRenderSynthesisOptions: RenderSynthesisOptions = RenderSynthesisOptions()
    var projectHasRenderableContent = false
    var statusMessage = "Welcome to Valar."
    var importErrorMessage: String?
    var exportErrorMessage: String?
    var isLoading = false
    var isExportingProjectAudio = false
    var exportProgress: Double? = nil
    var exportProgressLabel: String? = nil
    var showWelcome: Bool
    var dashboardSnapshot: ValarDashboardSnapshot = .empty
    var preferredProjectExportFormat: ProjectExportFormat = .wav
    var preferredProjectExportMode: ProjectExportMode = .concatenated
    var projectDocumentURL: URL?

    let services: ValarServiceHub
    let sharedServices: ValarServiceHub
    let documentProjectID: UUID
    let generatorState: GeneratorState
    let modelCatalogState: ModelCatalogState
    let voiceLibraryState: VoiceLibraryState
    let settingsState: SettingsState

    private static let onboardingKey = "ValarTTS.hasCompletedOnboarding"
    private var modelsReadyObserverID: UUID?
    private var refreshGeneration = 0

    init(
        services: ValarServiceHub,
        sharedServices: ValarServiceHub,
        documentProjectID: UUID
    ) {
        self.services = services
        self.sharedServices = sharedServices
        self.documentProjectID = documentProjectID
        self.showWelcome = !UserDefaults.standard.bool(forKey: Self.onboardingKey)
        self.generatorState = GeneratorState(
            services: services,
            modelDiscoveryServices: sharedServices
        )
        self.modelCatalogState = ModelCatalogState(services: sharedServices)
        self.voiceLibraryState = VoiceLibraryState(services: services)
        self.settingsState = SettingsState(services: sharedServices)
        self.settingsState.onCatalogVisibilityChanged = { [weak self] _ in
            guard let self else { return }
            await self.refreshSnapshot()
        }
        self.modelCatalogState.onCatalogMutation = { [weak self] in
            guard let self else { return }
            await self.refreshSnapshot()
        }
        self.generatorState.onSelectionChange = { [weak self] modelID in
            self?.applyGeneratorSelection(modelID: modelID)
        }
        self.modelsReadyObserverID = self.sharedServices.registerModelsReadyObserver { [weak self] in
            guard let self else { return }
            await self.refreshSnapshot()
        }
    }

    func completeOnboarding() {
        showWelcome = false
        UserDefaults.standard.set(true, forKey: Self.onboardingKey)
        selectedSection = .models
    }

    func load() async {
        isLoading = true
        defer { isLoading = false }
        statusMessage = "Loading..."

        await refreshSnapshot()

        statusMessage = "Ready: \(dashboardSnapshot.modelCount) models, \(formattedProjectCount), \(dashboardSnapshot.voiceCount) voices."
    }

    func refreshSnapshot() async {
        refreshGeneration += 1
        while true {
            let currentRefreshGeneration = refreshGeneration
            let previousProjectID = currentProject?.id
            let runtimeModelOptions = await sharedServices.runtime.generationModelOptions()
            let localSnapshot = await services.snapshot()
            let sharedSnapshot = await sharedServices.snapshot()
            guard currentRefreshGeneration == refreshGeneration else { continue }

            let snapshot = mergedSnapshot(local: scopedSnapshot(from: localSnapshot), shared: sharedSnapshot)
            dashboardSnapshot = snapshot

            selectedModelID = resolvedModelID(from: runtimeModelOptions)
            if previousProjectID != nil, currentProject?.id != previousProjectID {
                generatorState.clearUndoHistory()
            }

            await generatorState.reloadRuntimeOptions(selectedModelID: selectedModelID)
            guard currentRefreshGeneration == refreshGeneration else { continue }

            modelCatalogState.apply(snapshot: snapshot, selectedModelID: selectedModelID)
            voiceLibraryState.apply(snapshot: snapshot)
            return
        }
    }

    var currentProject: ProjectRecord? {
        dashboardSnapshot.projects.first(where: { $0.id == documentProjectID })
    }

    func updateProjectDocumentURL(_ url: URL?) {
        projectDocumentURL = url
    }

    func importModelBundles(from bundleURLs: [URL]) async {
        let importer = ModelBundleImporter()
        let candidates = bundleURLs.filter { $0.pathExtension.lowercased() == "valarmodel" }
        importErrorMessage = nil

        guard !candidates.isEmpty else {
            let rejectedName = bundleURLs.first?.lastPathComponent ?? "selection"
            presentImportError(ModelImportError.unsupportedBundleExtension(rejectedName))
            return
        }

        selectedSection = .models

        var importedResults: [ImportedModelBundleResult] = []
        for bundleURL in candidates {
            do {
                let result = try await importer.importBundle(from: bundleURL, using: sharedServices)
                importedResults.append(result)
                selectedModelID = result.modelID
            } catch {
                presentImportError(error)
                return
            }
        }

        await refreshSnapshot()

        if let selectedModelID {
            modelCatalogState.selectedModel = modelCatalogState.catalogModels.first(where: { $0.id == selectedModelID })
        }

        if let result = importedResults.last {
            statusMessage = importedResults.count == 1
                ? "Imported \(result.displayName)."
                : "Imported \(importedResults.count) model bundles."
        }
    }

    func dismissImportError() {
        importErrorMessage = nil
    }

    var canExportCurrentProject: Bool {
        currentProject != nil
            && projectHasRenderableContent
            && projectRenderModelID != nil
            && !isExportingProjectAudio
    }

    func exportCurrentProject(
        preferredModelID: ModelIdentifier? = nil,
        preferredSynthesisOptions: RenderSynthesisOptions = RenderSynthesisOptions()
    ) async {
        guard let project = currentProject else {
            statusMessage = "This document's project is unavailable for export."
            return
        }
        guard projectHasRenderableContent else {
            statusMessage = "Add script text to at least one chapter before exporting."
            return
        }
        guard let modelID = preferredModelID ?? projectRenderModelID else {
            statusMessage = "Choose a render model before exporting."
            return
        }

        guard let destinationURL = await chooseExportDestination(for: project) else {
            return
        }

        isExportingProjectAudio = true
        exportProgress = 0.0
        exportProgressLabel = "Preparing export…"
        defer {
            isExportingProjectAudio = false
            exportProgress = nil
            exportProgressLabel = nil
        }

        do {
            let result = try await services.projectExporter.exportProjectAudio(
                projectID: project.id,
                modelID: modelID,
                synthesisOptions: preferredSynthesisOptions,
                format: preferredProjectExportFormat,
                mode: preferredProjectExportMode,
                destinationURL: destinationURL,
                onProgress: { [weak self] progress in
                    Task { @MainActor [weak self] in
                        self?.exportProgress = progress.fraction
                        self?.exportProgressLabel = progress.statusLabel
                    }
                }
            )

            let exportedNames = result.files.map(\.lastPathComponent).joined(separator: ", ")
            statusMessage = "Exported \(result.exportedChapterCount) chapter(s): \(exportedNames)"
        } catch {
            exportErrorMessage = "Export failed: \(PathRedaction.redactMessage(error.localizedDescription))"
        }
    }

    private func resolvedModelID(from runtimeModelOptions: [RuntimeModelPickerOption]) -> ModelIdentifier? {
        if let selectedModelID,
           runtimeModelOptions.contains(where: { $0.id == selectedModelID }) {
            return selectedModelID
        }

        return runtimeModelOptions.first?.id
    }

    private func scopedSnapshot(from snapshot: ValarDashboardSnapshot) -> ValarDashboardSnapshot {
        ValarDashboardSnapshot(
            catalogModels: snapshot.catalogModels,
            modelSnapshots: snapshot.modelSnapshots,
            projects: snapshot.projects.filter { $0.id == documentProjectID },
            voices: snapshot.voices,
            renderJobs: snapshot.renderJobs.filter { $0.projectID == documentProjectID },
            goldenCorpus: snapshot.goldenCorpus,
            legacyImportPlan: snapshot.legacyImportPlan,
            compatibilityReport: snapshot.compatibilityReport,
            runtimeConfiguration: snapshot.runtimeConfiguration,
            appPaths: snapshot.appPaths,
            lastUpdatedAt: snapshot.lastUpdatedAt
        )
    }

    private func mergedSnapshot(
        local: ValarDashboardSnapshot,
        shared: ValarDashboardSnapshot
    ) -> ValarDashboardSnapshot {
        ValarDashboardSnapshot(
            catalogModels: shared.catalogModels,
            modelSnapshots: shared.modelSnapshots,
            projects: local.projects,
            voices: local.voices,
            renderJobs: local.renderJobs,
            goldenCorpus: shared.goldenCorpus,
            legacyImportPlan: shared.legacyImportPlan,
            compatibilityReport: shared.compatibilityReport,
            runtimeConfiguration: shared.runtimeConfiguration,
            appPaths: shared.appPaths,
            lastUpdatedAt: max(local.lastUpdatedAt, shared.lastUpdatedAt)
        )
    }

    private func applyGeneratorSelection(modelID: ModelIdentifier?) {
        selectedModelID = modelID
        if let modelID {
            modelCatalogState.selectedModel =
                dashboardSnapshot.catalogModel(for: modelID)
                ?? modelCatalogState.catalogModels.first(where: { $0.id == modelID })
        } else {
            modelCatalogState.selectedModel = nil
        }
    }

    private func chooseExportDestination(for project: ProjectRecord) async -> URL? {
        let suggestedDirectory = projectDocumentURL?
            .deletingLastPathComponent()
            .appendingPathComponent("Exports", isDirectory: true)

        switch preferredProjectExportMode {
        case .concatenated:
            let panel = NSSavePanel()
            panel.canCreateDirectories = true
            panel.prompt = "Export Audio"
            panel.directoryURL = suggestedDirectory
            panel.nameFieldStringValue = "\(sanitizedFileStem(project.title)).\(preferredProjectExportFormat.fileExtension)"
            return panel.runModal() == .OK ? panel.url : nil
        case .chapters:
            let panel = NSOpenPanel()
            panel.canChooseFiles = false
            panel.canChooseDirectories = true
            panel.canCreateDirectories = true
            panel.allowsMultipleSelection = false
            panel.prompt = "Choose Export Folder"
            panel.directoryURL = suggestedDirectory
            return panel.runModal() == .OK ? panel.url : nil
        }
    }

    private func sanitizedFileStem(_ value: String) -> String {
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "-_ "))
        let collapsed = value.unicodeScalars.map { allowed.contains($0) ? Character($0) : " " }
            .reduce(into: "") { $0.append($1) }
            .trimmingCharacters(in: .whitespacesAndNewlines)

        let stem = collapsed
            .components(separatedBy: .whitespacesAndNewlines)
            .filter { !$0.isEmpty }
            .joined(separator: "-")
            .lowercased()

        return stem.isEmpty ? "project" : stem
    }

    private func presentImportError(_ error: any Error) {
        let message = PathRedaction.redactMessage(
            (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
        )
        importErrorMessage = message
        statusMessage = "Model import failed: \(message)"
    }

    private var formattedProjectCount: String {
        let count = dashboardSnapshot.projectCount
        return count == 1 ? "1 project" : "\(count) projects"
    }
}

// Environment key for focused binding
struct AppStateFocusedKey: FocusedValueKey {
    typealias Value = Binding<AppState>
}

extension FocusedValues {
    var appState: Binding<AppState>? {
        get { self[AppStateFocusedKey.self] }
        set { self[AppStateFocusedKey.self] = newValue }
    }
}

// Environment support
extension EnvironmentValues {
    @Entry var appStateValue: AppState? = nil
}
