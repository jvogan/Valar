import SwiftUI
import ValarCore
import ValarModelKit
import ValarPersistence

struct ProjectWorkspaceView: View {
    @Environment(AppState.self) private var appState
    @Environment(\.documentEditAction) private var documentEditAction
    let project: ProjectRecord
    let services: ValarServiceHub
    let preferredRenderModelID: ModelIdentifier?
    let preferredRenderSynthesisOptions: RenderSynthesisOptions
    let onChapterCountChange: (@MainActor @Sendable (Int) -> Void)?

    @State private var state: ProjectWorkspaceState?
    @State private var showScriptIngestion = false
    @State private var showSpeakerForm = false
    @State private var isScriptDropTargeted = false
    @State private var droppedScriptLabel: String?

    var body: some View {
        Group {
            if let state {
                workspaceContent(state)
            } else {
                ProgressView("Loading workspace...")
            }
        }
        .navigationTitle(project.title)
        .toolbar {
            ToolbarItemGroup(placement: .primaryAction) {
                Button {
                    showScriptIngestion = true
                } label: {
                    Label("Import Script", systemImage: "doc.text")
                }
                Button {
                    Task { await state?.addChapter() }
                } label: {
                    Label("Add Chapter", systemImage: "plus")
                }
            }
        }
        .sheet(isPresented: $showScriptIngestion) {
            if let state {
                ScriptIngestionView(workspaceState: state)
            }
        }
        .sheet(isPresented: $showSpeakerForm) {
            if let state {
                SpeakerFormView(workspaceState: state)
            }
        }
        .onDisappear {
            state?.teardown()
        }
        .task {
            let ws = ProjectWorkspaceState(
                services: services,
                onChapterCountChange: onChapterCountChange
            )
            await ws.load(
                project: project,
                preferredModelID: preferredRenderModelID,
                preferredSynthesisOptions: preferredRenderSynthesisOptions
            )
            appState.projectRenderModelID = ws.selectedRenderModelID
            appState.projectRenderSynthesisOptions = ws.renderSynthesisOptions
            state = ws
        }
        .alert(
            "Workspace Error",
            isPresented: Binding(
                get: { state?.errorMessage != nil },
                set: { isPresented in
                    if !isPresented {
                        state?.dismissError()
                    }
                }
            )
        ) {
            Button("OK", role: .cancel) {
                state?.dismissError()
            }
        } message: {
            Text(state?.errorMessage ?? "")
        }
        .overlay {
            if isScriptDropTargeted {
                ZStack {
                    RoundedRectangle(cornerRadius: 16, style: .continuous)
                        .fill(Color.accentColor.opacity(0.05))
                    RoundedRectangle(cornerRadius: 16, style: .continuous)
                        .strokeBorder(
                            Color.accentColor,
                            style: StrokeStyle(lineWidth: 2.5, dash: [10, 7])
                        )
                    VStack(spacing: 8) {
                        Image(systemName: "doc.on.doc.fill")
                            .font(.largeTitle)
                            .foregroundStyle(Color.accentColor)
                        Text("Drop script files to create chapters")
                            .font(.headline)
                            .foregroundStyle(.primary)
                        Text("Text split on \"---\" delimiter")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    .padding(20)
                    .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
                }
                .padding(6)
                .transition(.opacity.combined(with: .scale(scale: 0.97)))
                .allowsHitTesting(false)
            }
        }
        .animation(.spring(response: 0.3, dampingFraction: 0.75), value: isScriptDropTargeted)
        .dropDestination(for: URL.self) { items, _ in
            guard let ws = state else { return false }
            let textFiles = items.filter { ValarDragDrop.isTextFile($0) }
            guard !textFiles.isEmpty else { return false }
            Task {
                for file in textFiles {
                    await ws.ingestScriptFile(file)
                }
                showScriptDropConfirmation(textFiles.count)
            }
            return true
        } isTargeted: { targeted in
            isScriptDropTargeted = targeted
        }
        .overlay(alignment: .bottom) {
            if let label = droppedScriptLabel {
                HStack(spacing: 6) {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundStyle(.green)
                    Text(label)
                        .font(.caption.weight(.medium))
                }
                .padding(.horizontal, 14)
                .padding(.vertical, 8)
                .background(.regularMaterial, in: Capsule())
                .shadow(color: .black.opacity(0.1), radius: 8, y: 2)
                .padding(.bottom, 12)
                .transition(.move(edge: .bottom).combined(with: .opacity))
            }
        }
        .animation(.spring(response: 0.4, dampingFraction: 0.8), value: droppedScriptLabel != nil)
    }

    @ViewBuilder
    private func workspaceContent(_ ws: ProjectWorkspaceState) -> some View {
        VStack(spacing: 0) {
            // DAW-style transport bar
            TransportBarView(state: ws)

            HSplitView {
                // Main content: timeline + editor
                mainContent(ws)
                    .frame(minWidth: 500)

                // Right inspector: speakers, defaults, render queue, export
                inspectorColumn(ws)
                    .frame(minWidth: 260, idealWidth: 320, maxWidth: 400)
            }
        }
        .focusedSceneValue(
            \.projectRenderChapterAction,
            ProjectRenderCommandAction {
                Task { await ws.startSelectedChapterRender() }
            }
        )
        .focusedSceneValue(\.projectCanRenderChapter, ws.canRenderSelectedChapter)
        .focusedSceneValue(\.projectRenderChapterDisabledReason, ws.renderChapterDisabledReason)
        .focusedSceneValue(
            \.projectRenderProjectAction,
            ProjectRenderCommandAction {
                Task { await ws.startRender() }
            }
        )
        .focusedSceneValue(\.projectCanRenderProject, ws.renderProjectDisabledReason == nil)
        .focusedSceneValue(\.projectRenderProjectDisabledReason, ws.renderProjectDisabledReason)
        .focusedSceneValue(\.projectRenderModelID, ws.selectedRenderModelID)
        .focusedSceneValue(\.projectRenderSynthesisOptions, ws.renderSynthesisOptions)
        .task {
            appState.projectHasRenderableContent = ws.projectHasRenderableContent
        }
        .onChange(of: ws.project) { _, _ in
            appState.projectHasRenderableContent = ws.projectHasRenderableContent
            documentEditAction?()
        }
        .onChange(of: ws.chapters) { _, _ in
            appState.projectHasRenderableContent = ws.projectHasRenderableContent
            documentEditAction?()
        }
        .onChange(of: ws.speakers) { _, _ in
            documentEditAction?()
        }
        .onChange(of: ws.renderJobs) { _, _ in
            documentEditAction?()
        }
        .onChange(of: ws.selectedRenderModelID) { _, newValue in
            appState.projectRenderModelID = newValue
            documentEditAction?()
        }
        .onChange(of: ws.renderSynthesisOptions) { _, newValue in
            appState.projectRenderSynthesisOptions = newValue
            documentEditAction?()
        }
    }

    @ViewBuilder
    private func mainContent(_ ws: ProjectWorkspaceState) -> some View {
        VSplitView {
            // Chapter timeline (DAW lanes)
            ChapterTimelineView(state: ws)
                .frame(minHeight: 120, idealHeight: 200)

            // Segment editor for selected chapter
            SegmentEditorView(state: ws)
                .frame(minHeight: 200)
        }
    }

    @ViewBuilder
    private func inspectorColumn(_ ws: ProjectWorkspaceState) -> some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                SpeakerCastView(state: ws, onAddSpeaker: { showSpeakerForm = true })
                Divider()
                ProjectDefaultsView(state: ws)
                Divider()
                RenderQueueView(state: ws)
                Divider()
                ProjectExportView(project: project, state: ws)
            }
            .padding(16)
        }
    }

    private func showScriptDropConfirmation(_ count: Int) {
        droppedScriptLabel = "Imported \(count) script file\(count == 1 ? "" : "s")"
        Task { @MainActor in
            try? await Task.sleep(for: .seconds(2.5))
            withAnimation { droppedScriptLabel = nil }
        }
    }
}
