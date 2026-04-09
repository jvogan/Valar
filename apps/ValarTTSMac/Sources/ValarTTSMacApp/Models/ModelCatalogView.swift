import SwiftUI
import ValarCore
import ValarModelKit
import ValarPersistence

struct ModelCatalogView: View {
    @Environment(AppState.self) private var appState
    @State private var isDropTargeted = false

    var body: some View {
        let modelState = appState.modelCatalogState

        return Group {
            if appState.isLoading && modelState.catalogModels.isEmpty {
                List {
                    Section("Installed") {
                        ForEach(0..<3, id: \.self) { _ in
                            SkeletonModelCard()
                        }
                    }
                    Section("Cached") {
                        ForEach(0..<2, id: \.self) { _ in
                            SkeletonModelCard()
                        }
                    }
                    Section("Available") {
                        ForEach(0..<4, id: \.self) { _ in
                            SkeletonModelCard()
                        }
                    }
                }
                .listStyle(.inset(alternatesRowBackgrounds: true))
                .transition(.opacity)
            } else {
                content(modelState)
                    .transition(.opacity)
            }
        }
        .animation(.easeInOut(duration: 0.35), value: appState.isLoading)
        .navigationTitle("Models")
        .dropDestination(for: URL.self) { items, _ in
            let bundleURLs = items.filter { $0.pathExtension.lowercased() == "valarmodel" }
            guard !bundleURLs.isEmpty else { return false }
            Task { await appState.importModelBundles(from: bundleURLs) }
            return true
        } isTargeted: { isTargeted in
            self.isDropTargeted = isTargeted
        }
        .overlay {
            if isDropTargeted {
                ZStack {
                    RoundedRectangle(cornerRadius: 16, style: .continuous)
                        .fill(Color.accentColor.opacity(0.05))
                    RoundedRectangle(cornerRadius: 16, style: .continuous)
                        .strokeBorder(
                            Color.accentColor,
                            style: StrokeStyle(lineWidth: 2.5, dash: [10, 7])
                        )
                    VStack(spacing: 8) {
                        Image(systemName: "cpu.fill")
                            .font(.largeTitle)
                            .foregroundStyle(Color.accentColor)
                            .symbolEffect(.pulse, options: .repeating)
                        Text("Drop .valarmodel bundle to import")
                            .font(.headline)
                            .foregroundStyle(.primary)
                    }
                    .padding(20)
                    .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
                }
                .padding(8)
                .transition(.opacity.combined(with: .scale(scale: 0.97)))
                .allowsHitTesting(false)
            }
        }
        .animation(.spring(response: 0.3, dampingFraction: 0.75), value: isDropTargeted)
    }

    private func modelRow(_ model: CatalogModel, state: ModelCatalogState) -> some View {
        let progress = state.downloads[model.id]
        return ModelCardView(
            model: model,
            downloadProgress: progress,
            diskUsageBytes: state.diskUsageBytes(for: model),
            installError: state.installErrors[model.id]
        )
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(modelCardAccessibilityLabel(for: model, downloadProgress: progress))
        .tag(model.id)
    }

    private func content(_ state: ModelCatalogState) -> some View {
        @Bindable var s = state
        let selectionBinding = Binding<ModelIdentifier?>(
            get: { state.selectedModel?.id },
            set: { id in state.selectedModel = state.catalogModels.first(where: { $0.id == id }) }
        )
        return List(selection: selectionBinding) {
            if !state.installedModels.isEmpty {
                Section {
                    ForEach(state.installedModels) { model in
                        modelRow(model, state: state)
                    }
                } header: {
                    Label("Installed", systemImage: "checkmark.circle.fill")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.green)
                }
            }

            if !state.cachedModels.isEmpty {
                Section {
                    ForEach(state.cachedModels) { model in
                        modelRow(model, state: state)
                    }
                } header: {
                    Label("Cached", systemImage: "externaldrive.fill")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.orange)
                }
            }

            if !state.availableModels.isEmpty {
                Section {
                    ForEach(state.availableModels) { model in
                        modelRow(model, state: state)
                    }
                } header: {
                    Label("Available", systemImage: "arrow.down.circle")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.secondary)
                }
            }

            if state.filteredModels.isEmpty {
                ContentUnavailableView.search(text: state.searchText)
            }
        }
        .listStyle(.inset(alternatesRowBackgrounds: true))
        .searchable(text: $s.searchText, prompt: "Search models")
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                MemoryPressureView(
                    usedBytes: state.memoryUsedBytes,
                    budgetBytes: state.memoryBudgetBytes
                )
                .accessibilityElement(children: .ignore)
                .accessibilityLabel(memoryPressureAccessibilityLabel(
                    usedBytes: state.memoryUsedBytes,
                    budgetBytes: state.memoryBudgetBytes
                ))
            }
        }
        .inspector(isPresented: Binding(
            get: { state.selectedModel != nil },
            set: { if !$0 { state.selectedModel = nil } }
        )) {
            if let model = state.selectedModel {
                ModelDetailView(
                    model: model,
                    downloadProgress: state.downloads[model.id],
                    diskUsageBytes: state.diskUsageBytes(for: model)
                ) {
                    state.startInstall(model)
                } onUninstall: {
                    Task { await state.uninstallModel(model) }
                } onCancelDownload: {
                    state.cancelInstall(model)
                }
                .inspectorColumnWidth(min: 280, ideal: 320, max: 420)
            }
        }
    }

    private func modelCardAccessibilityLabel(for model: CatalogModel, downloadProgress: Double?) -> String {
        "\(model.descriptor.displayName), \(modelInstallStateText(for: model, downloadProgress: downloadProgress))"
    }

    private func modelInstallStateText(for model: CatalogModel, downloadProgress: Double?) -> String {
        if downloadProgress != nil {
            return "downloading"
        }

        switch model.installState {
        case .installed:
            return "installed"
        case .cached:
            return "cached in shared storage; local install may still use disk"
        case .supported:
            return "not installed"
        }
    }

    private func memoryPressureAccessibilityLabel(usedBytes: Int, budgetBytes: Int) -> String {
        let percentageUsed: Int
        if budgetBytes > 0 {
            percentageUsed = Int((Double(usedBytes) / Double(budgetBytes)) * 100)
        } else {
            percentageUsed = 0
        }

        let formatter = ByteCountFormatter()
        formatter.countStyle = .memory

        let usedText = formatter.string(fromByteCount: Int64(usedBytes))
        let budgetText = formatter.string(fromByteCount: Int64(budgetBytes))
        return "Memory usage \(usedText) of \(budgetText), \(percentageUsed) percent used"
    }
}
