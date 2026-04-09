import SwiftUI
import ValarCore
import ValarModelKit

struct GeneratorCommandAction {
    private let handler: @MainActor @Sendable () -> Void

    init(_ handler: @escaping @MainActor @Sendable () -> Void) {
        self.handler = handler
    }

    @MainActor
    func callAsFunction() {
        handler()
    }
}

private struct GeneratorGenerateActionKey: FocusedValueKey {
    typealias Value = GeneratorCommandAction
}

private struct GeneratorCanGenerateKey: FocusedValueKey {
    typealias Value = Bool
}

private struct GeneratorTogglePlaybackActionKey: FocusedValueKey {
    typealias Value = GeneratorCommandAction
}

private struct GeneratorCanTogglePlaybackKey: FocusedValueKey {
    typealias Value = Bool
}

private struct GeneratorToggleInspectorActionKey: FocusedValueKey {
    typealias Value = GeneratorCommandAction
}

extension FocusedValues {
    var generatorGenerateAction: GeneratorCommandAction? {
        get { self[GeneratorGenerateActionKey.self] }
        set { self[GeneratorGenerateActionKey.self] = newValue }
    }

    var generatorCanGenerate: Bool? {
        get { self[GeneratorCanGenerateKey.self] }
        set { self[GeneratorCanGenerateKey.self] = newValue }
    }

    var generatorTogglePlaybackAction: GeneratorCommandAction? {
        get { self[GeneratorTogglePlaybackActionKey.self] }
        set { self[GeneratorTogglePlaybackActionKey.self] = newValue }
    }

    var generatorCanTogglePlayback: Bool? {
        get { self[GeneratorCanTogglePlaybackKey.self] }
        set { self[GeneratorCanTogglePlaybackKey.self] = newValue }
    }

    var generatorToggleInspectorAction: GeneratorCommandAction? {
        get { self[GeneratorToggleInspectorActionKey.self] }
        set { self[GeneratorToggleInspectorActionKey.self] = newValue }
    }
}

struct GeneratorView: View {
    @Environment(AppState.self) private var appState
    @State private var showInspector = false

    var body: some View {
        let generator = appState.generatorState
        let inspectorVisibility = $showInspector

        VStack(spacing: 0) {
            if generator.showsInlineReferenceAudioControls {
                GeneratorReferenceAudioSection(state: generator)
                Divider()
            }

            // Text editor area
            GeneratorTextEditor(state: generator)

            Divider()

            // Audio player (visible after generation)
            if generator.hasAudio {
                GeneratorPlayerView(state: generator)
                    .frame(height: 60)
                    .padding(.horizontal)
                    .padding(.vertical, 8)
            }

            // Generation progress
            if generator.isGenerating {
                ProgressView(value: generator.generationProgress)
                    .progressViewStyle(.linear)
                    .padding(.horizontal)
            }
        }
        .inspector(isPresented: $showInspector) {
            GeneratorInspectorView(state: generator)
                .inspectorColumnWidth(min: 240, ideal: 280, max: 360)
        }
        .toolbar {
            GeneratorToolbar(state: generator, showInspector: $showInspector)
        }
        .focusedSceneValue(
            \.generatorGenerateAction,
            GeneratorCommandAction {
                Task { await generator.generate() }
            }
        )
        .focusedSceneValue(\.generatorCanGenerate, generator.canGenerate)
        .focusedSceneValue(
            \.generatorTogglePlaybackAction,
            GeneratorCommandAction {
                generator.togglePlayback()
            }
        )
        .focusedSceneValue(\.generatorCanTogglePlayback, generator.hasAudio)
        .focusedSceneValue(
            \.generatorToggleInspectorAction,
            GeneratorCommandAction {
                inspectorVisibility.wrappedValue.toggle()
            }
        )
        .alert(
            "Generation Failed",
            isPresented: Binding(
                get: { generator.errorMessage != nil },
                set: { isPresented in
                    if !isPresented {
                        generator.errorMessage = nil
                    }
                }
            )
        ) {
            Button("OK", role: .cancel) {
                generator.errorMessage = nil
            }
        } message: {
            Text(generator.errorMessage ?? "")
        }
    }
}
