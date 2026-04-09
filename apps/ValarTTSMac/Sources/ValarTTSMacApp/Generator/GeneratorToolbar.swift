import SwiftUI
import ValarModelKit
import ValarPersistence

struct GeneratorToolbar: ToolbarContent {
    @Bindable var state: GeneratorState
    @Binding var showInspector: Bool
    @Environment(AppState.self) private var appState

    var body: some ToolbarContent {
        ToolbarItemGroup(placement: .primaryAction) {
            VoicePickerView(selectedModelID: Binding(
                get: { state.selectedModelID },
                set: { newValue in
                    state.selectModel(newValue)
                }
            ))

            StoredVoicePickerView(
                voices: state.availableVoices,
                selectedModelID: state.selectedModelID,
                selectedFamilyID: state.selectedModelOption?.familyID,
                selectedVoiceID: Binding(
                    get: { state.selectedVoiceID },
                    set: { state.selectVoice($0) }
                )
            )
            .accessibilityLabel("Voice")

            Picker("Language", selection: $state.selectedLanguage) {
                Text("Auto").tag("auto")
                Divider()
                Text("English").tag("en")
                Text("French").tag("fr")
                Text("German").tag("de")
                Text("Spanish").tag("es")
                Text("Portuguese").tag("pt")
                Text("Italian").tag("it")
                Text("Polish").tag("pl")
                Text("Japanese").tag("ja")
                Text("Chinese").tag("zh")
                Text("Arabic").tag("ar")
            }
            .frame(width: 100)
            .accessibilityLabel("Output language")

            Button {
                Task { await state.generate() }
            } label: {
                Label("Generate", systemImage: "waveform.badge.plus")
            }
            .accessibilityLabel("Generate")
            .keyboardShortcut(.generate)
            .disabled(!state.canGenerate)
            .help(
                state.canGenerate
                    ? "Generate speech from text"
                    : state.requiresInlineReferenceAudio
                        ? "Add a reference clip to use the selected TADA model."
                        : "Choose a speech model before generating."
            )

            Button {
                showInspector.toggle()
            } label: {
                Image(systemName: "sidebar.trailing")
            }
            .accessibilityLabel("Toggle Inspector")
            .keyboardShortcut(.toggleInspector)
            .help("Toggle Inspector")
        }
    }
}
