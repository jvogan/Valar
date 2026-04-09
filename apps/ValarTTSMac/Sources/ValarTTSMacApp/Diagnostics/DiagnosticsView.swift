import SwiftUI

struct DiagnosticsView: View {
    private enum LoadState {
        case loading
        case loaded(ValarDashboardSnapshot)
        case failed
    }

    @Environment(AppState.self) private var appState
    @State private var loadState: LoadState = .loading
    @State private var loadAttempt = 0

    var body: some View {
        Group {
            switch loadState {
            case .loading:
                ProgressView()
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            case let .loaded(snapshot):
                ScrollView {
                    VStack(alignment: .leading, spacing: 24) {
                        header
                        metricsGrid(snapshot: snapshot)
                        serviceHealthCard(snapshot: snapshot)
                    }
                    .padding(24)
                }
            case .failed:
                EmptyStateView(
                    icon: "exclamationmark.triangle",
                    title: "Diagnostics unavailable",
                    message: "The diagnostics snapshot could not be loaded.",
                    actionLabel: "Retry",
                    action: retryLoad
                )
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .navigationTitle("Diagnostics")
        .task(id: loadAttempt) {
            await loadSnapshot()
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Diagnostics")
                .font(.largeTitle.bold())
            Text("Check app health, available models, and local service readiness.")
                .foregroundStyle(.secondary)
        }
    }

    private func metricsGrid(snapshot: ValarDashboardSnapshot) -> some View {
        LazyVGrid(columns: [GridItem(.adaptive(minimum: 180), spacing: 16)], spacing: 16) {
            MetricCard(
                title: "Installed",
                value: "\(snapshot.installedModelCount)",
                detail: "catalog models installed locally",
                symbol: "internaldrive"
            )
            MetricCard(
                title: "Cached",
                value: "\(snapshot.cachedModelCount)",
                detail: "shared-cache models ready to register",
                symbol: "externaldrive"
            )
            MetricCard(
                title: "Recommended",
                value: "\(snapshot.recommendedModelCount)",
                detail: "featured public models",
                symbol: "sparkles"
            )
            MetricCard(
                title: "Projects",
                value: "\(snapshot.projectCount)",
                detail: "local Valar projects",
                symbol: "folder"
            )
            MetricCard(
                title: "Voices",
                value: "\(snapshot.voiceCount)",
                detail: "saved local voices",
                symbol: "waveform"
            )
        }
    }

    private func serviceHealthCard(snapshot: ValarDashboardSnapshot) -> some View {
        SurfaceCard(title: "Service health", symbol: "heart.text.square") {
            VStack(alignment: .leading, spacing: 10) {
                KeyValueRow(label: "Catalog models", value: "\(snapshot.modelCount) total")
                KeyValueRow(label: "Installed", value: "\(snapshot.installedModelCount) installed locally")
                KeyValueRow(label: "Cached", value: "\(snapshot.cachedModelCount) available from shared cache")
                KeyValueRow(label: "Projects", value: "\(snapshot.projectCount)")
                KeyValueRow(label: "Voices", value: "\(snapshot.voiceCount)")
                KeyValueRow(label: "Render jobs", value: "\(snapshot.jobCount) queued")
                KeyValueRow(label: "Last updated", value: snapshot.lastUpdatedAt.formatted(date: .abbreviated, time: .shortened))
            }
        }
    }

    @MainActor
    private func retryLoad() {
        loadAttempt += 1
    }

    @MainActor
    private func loadSnapshot() async {
        loadState = .loading
        let attempt = loadAttempt
        let timeoutTask = Task { @MainActor in
            do {
                try await Task.sleep(for: .seconds(5))
            } catch {
                return
            }

            guard loadAttempt == attempt else { return }
            guard case .loading = loadState else { return }
            loadState = .failed
        }

        #if DEBUG
        let snapshot = await appState.services.runtime.diagnostics(
            parityHarness: appState.services.parityHarness,
            includeParityData: false
        )
        #else
        let snapshot = await appState.services.runtime.diagnostics()
        #endif
        timeoutTask.cancel()

        guard !Task.isCancelled else { return }
        guard loadAttempt == attempt else { return }

        if snapshot == .empty {
            loadState = .failed
            return
        }

        guard case .loading = loadState else { return }
        loadState = .loaded(snapshot)
    }
}
