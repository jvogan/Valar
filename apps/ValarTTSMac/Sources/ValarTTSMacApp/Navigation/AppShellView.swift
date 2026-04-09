import SwiftUI

struct AppShellView: View {
    @Environment(AppState.self) private var appState

    var body: some View {
        @Bindable var state = appState
        NavigationSplitView {
            AppSidebarView(selection: $state.selectedSection)
                .disabled(appState.showWelcome)
        } detail: {
            if appState.showWelcome {
                WelcomeView {
                    appState.completeOnboarding()
                }
            } else {
                switch appState.selectedSection {
                case .generate:
                    GeneratorView()
                case .project:
                    DocumentProjectView()
                case .voices:
                    VoiceLibraryView()
                case .models:
                    ModelCatalogView()
                case .diagnostics:
                    DiagnosticsView()
                }
            }
        }
        .navigationSplitViewStyle(.balanced)
        .frame(minWidth: 900, minHeight: 600)
        .focusedSceneValue(\.appState, Binding(get: { appState }, set: { _ in }))
        .alert(
            "Model Import Failed",
            isPresented: Binding(
                get: { appState.importErrorMessage != nil },
                set: { isPresented in
                    if !isPresented {
                        appState.dismissImportError()
                    }
                }
            )
        ) {
            Button("OK", role: .cancel) {
                appState.dismissImportError()
            }
        } message: {
            Text(appState.importErrorMessage ?? "")
        }
        .alert(
            "Export Failed",
            isPresented: Binding(
                get: { appState.exportErrorMessage != nil },
                set: { if !$0 { appState.exportErrorMessage = nil } }
            )
        ) {
            Button("OK", role: .cancel) {
                appState.exportErrorMessage = nil
            }
        } message: {
            Text(appState.exportErrorMessage ?? "")
        }
    }
}
