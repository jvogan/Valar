import SwiftUI
import ValarPersistence

struct ProjectExportView: View {
    let project: ProjectRecord
    let state: ProjectWorkspaceState

    @Environment(AppState.self) private var appState

    var body: some View {
        @Bindable var appState = appState
        let exportProgress = min(max(appState.exportProgress ?? 0, 0), 1)
        let exportProgressLabel = appState.exportProgressLabel ?? "Preparing export…"

        VStack(alignment: .leading, spacing: 12) {
            Label("Export", systemImage: "square.and.arrow.up")
                .font(.headline)

            Picker("Format", selection: $appState.preferredProjectExportFormat) {
                ForEach(ProjectExportFormat.allCases, id: \.self) { format in
                    Text(format.rawValue).tag(format)
                }
            }
            .pickerStyle(.segmented)

            Picker("Mode", selection: $appState.preferredProjectExportMode) {
                ForEach(ProjectExportMode.allCases, id: \.self) { mode in
                    Text(mode.rawValue).tag(mode)
                }
            }
            .pickerStyle(.segmented)

            Button {
                Task {
                    await appState.exportCurrentProject(
                        preferredModelID: state.selectedRenderModelID,
                        preferredSynthesisOptions: state.renderSynthesisOptions
                    )
                }
            } label: {
                HStack(spacing: 6) {
                    Label(
                        appState.isExportingProjectAudio ? "Exporting \(project.title)…" : "Export Project",
                        systemImage: "arrow.down.doc"
                    )
                }
                .frame(maxWidth: .infinity)
                .contentTransition(.interpolate)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.regular)
            .disabled(!appState.canExportCurrentProject)
            .animation(.easeInOut(duration: 0.2), value: appState.isExportingProjectAudio)

            if appState.isExportingProjectAudio {
                VStack(alignment: .leading, spacing: 6) {
                    ProgressView(value: exportProgress)
                        .progressViewStyle(.linear)

                    HStack(alignment: .firstTextBaseline) {
                        Text(exportProgressLabel)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        Spacer()
                        Text(exportProgress, format: .percent.precision(.fractionLength(0)))
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(.secondary)
                    }
                }
                .transition(.opacity.combined(with: .move(edge: .top)))
            }
        }
    }
}
