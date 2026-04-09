import SwiftUI

struct DocumentProjectView: View {
    @Environment(AppState.self) private var appState

    var body: some View {
        Group {
            if let project = appState.currentProject {
                ProjectWorkspaceView(
                    project: project,
                    services: appState.services,
                    preferredRenderModelID: appState.projectRenderModelID ?? appState.selectedModelID,
                    preferredRenderSynthesisOptions: appState.projectRenderSynthesisOptions,
                    onChapterCountChange: nil
                )
            } else if appState.isLoading {
                ProgressView("Loading project...")
            } else {
                ContentUnavailableView(
                    "Project Unavailable",
                    systemImage: "book.pages",
                    description: Text("This document's project could not be loaded.")
                )
            }
        }
        .navigationTitle(appState.currentProject?.title ?? "Project")
    }
}
