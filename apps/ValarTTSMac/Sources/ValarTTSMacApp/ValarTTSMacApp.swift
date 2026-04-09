import SwiftUI

@MainActor
@main
struct ValarTTSMacApp: App {
    @State private var standaloneSettingsState = SettingsState(services: ValarProjectDocument.sharedServices)

    init() {
        SettingsState.applyPersistedCatalogVisibilityEnvironment()
        // Remove any session temp dirs (which contain biometric speaker embeddings)
        // left behind by a prior crash or force-quit.
        ValarProjectDocument.cleanupStaleSessionDirectories()
    }

    var body: some Scene {
        DocumentGroup(newDocument: {
            MainActor.assumeIsolated {
                ValarProjectDocument()
            }
        }) { configuration in
            ValarProjectDocumentSceneView(
                document: configuration.document,
                fileURL: configuration.fileURL
            )
        }
        .commands { AppCommands() }
        .defaultSize(width: 1100, height: 720)

        Settings {
            ValarSettingsSceneView(fallback: standaloneSettingsState)
        }
    }
}

private struct ValarSettingsSceneView: View {
    @FocusedValue(\.appState) private var focusedAppState
    let fallback: SettingsState

    var body: some View {
        if let appState = focusedAppState?.wrappedValue {
            SettingsView(state: appState.settingsState)
        } else {
            SettingsView(state: fallback)
        }
    }
}
