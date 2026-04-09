import AppKit
import SwiftUI
import ValarPersistence

struct AppCommands: Commands {
    @FocusedValue(\.appState) private var focusedAppState
    @FocusedValue(\.generatorGenerateAction) private var generatorGenerateAction
    @FocusedValue(\.generatorCanGenerate) private var generatorCanGenerate
    @FocusedValue(\.generatorTogglePlaybackAction) private var generatorTogglePlaybackAction
    @FocusedValue(\.generatorCanTogglePlayback) private var generatorCanTogglePlayback
    @FocusedValue(\.generatorToggleInspectorAction) private var generatorToggleInspectorAction
    @FocusedValue(\.projectRenderChapterAction) private var projectRenderChapterAction
    @FocusedValue(\.projectCanRenderChapter) private var projectCanRenderChapter
    @FocusedValue(\.projectRenderChapterDisabledReason) private var projectRenderChapterDisabledReason
    @FocusedValue(\.projectRenderProjectAction) private var projectRenderProjectAction
    @FocusedValue(\.projectCanRenderProject) private var projectCanRenderProject
    @FocusedValue(\.projectRenderProjectDisabledReason) private var projectRenderProjectDisabledReason
    @FocusedValue(\.projectRenderModelID) private var projectRenderModelID
    @FocusedValue(\.projectRenderSynthesisOptions) private var projectRenderSynthesisOptions

    var body: some Commands {
        let appState = focusedAppState?.wrappedValue
        let generator = appState?.generatorState

        CommandGroup(replacing: .appInfo) {
            Button("About Valar") {
                AboutWindowController.show()
            }
        }

        CommandGroup(replacing: .undoRedo) {
            Button(generator?.undoMenuTitle ?? "Undo") {
                generator?.performUndo()
            }
            .keyboardShortcut("z", modifiers: .command)
            .disabled(generator?.canUndo != true)

            Button(generator?.redoMenuTitle ?? "Redo") {
                generator?.performRedo()
            }
            .keyboardShortcut("Z", modifiers: [.command, .shift])
            .disabled(generator?.canRedo != true)
        }

        CommandGroup(after: .saveItem) {
            Button("Import Model…") {
                guard let appState,
                      let bundleURL = openModelBundleURL() else {
                    return
                }
                Task { await appState.importModelBundles(from: [bundleURL]) }
            }
            .disabled(appState == nil)
        }

        CommandMenu("Generate") {
            Button("Generate Speech") {
                generatorGenerateAction?.callAsFunction()
            }
                .keyboardShortcut(.generate)
                .disabled(generatorGenerateAction == nil || generatorCanGenerate != true)
            Divider()
            Button("Toggle Inspector") {
                generatorToggleInspectorAction?.callAsFunction()
            }
                .keyboardShortcut(.toggleInspector)
                .disabled(generatorToggleInspectorAction == nil)
        }

        CommandMenu("Playback") {
            Button("Play / Pause") {
                generatorTogglePlaybackAction?.callAsFunction()
            }
                .disabled(generatorTogglePlaybackAction == nil || generatorCanTogglePlayback != true)
        }

        CommandMenu("Render") {
            Button("Render Chapter") {
                projectRenderChapterAction?.callAsFunction()
            }
                .keyboardShortcut(.renderChapter)
                .disabled(projectRenderChapterAction == nil || projectCanRenderChapter != true)
                .help(renderChapterMenuHelp)
            Button("Render Project") {
                projectRenderProjectAction?.callAsFunction()
            }
                .keyboardShortcut(.renderProject)
                .disabled(projectRenderProjectAction == nil || projectCanRenderProject != true)
                .help(renderProjectMenuHelp)
            Divider()
            Button("Export...") {
                guard let appState else { return }
                Task {
                    await appState.exportCurrentProject(
                        preferredModelID: projectRenderModelID,
                        preferredSynthesisOptions: projectRenderSynthesisOptions ?? RenderSynthesisOptions()
                    )
                }
            }
                .keyboardShortcut(.exportProject)
                .disabled(appState?.canExportCurrentProject != true)
        }
    }

    private func openModelBundleURL() -> URL? {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.allowedContentTypes = ValarDragDrop.acceptedModelImportTypes
        panel.prompt = "Import Model"
        panel.message = "Choose a .valarmodel bundle to import."

        guard panel.runModal() == .OK,
              let selectedURL = panel.url else {
            return nil
        }

        return selectedURL
    }

    private var renderChapterMenuHelp: String {
        if projectRenderChapterAction == nil {
            return "Open a project workspace to render a chapter."
        }
        return projectRenderChapterDisabledReason ?? "Render the selected chapter."
    }

    private var renderProjectMenuHelp: String {
        if projectRenderProjectAction == nil {
            return "Open a project workspace to render the project."
        }
        return projectRenderProjectDisabledReason ?? "Render every chapter in the current project."
    }
}
