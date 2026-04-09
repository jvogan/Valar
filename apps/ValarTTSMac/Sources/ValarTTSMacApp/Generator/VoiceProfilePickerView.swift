import SwiftUI
import ValarPersistence

struct VoiceProfilePickerView: View {
    @Binding var selectedVoiceID: UUID?

    let voices: [VoiceLibraryRecord]
    let onSelect: (VoiceLibraryRecord?) -> Void

    var body: some View {
        Picker("Voice", selection: selectionBinding) {
            Text("Default voice").tag(nil as UUID?)
            if !presetVoices.isEmpty {
                Section("Preset Voices") {
                    ForEach(presetVoices) { voice in
                        Text(voice.label).tag(Optional(voice.id))
                    }
                }
            }
            if !savedVoices.isEmpty {
                Section("Saved Voices") {
                    ForEach(savedVoices) { voice in
                        Text(voice.label).tag(Optional(voice.id))
                    }
                }
            }
        }
        .frame(width: 180)
    }

    private var selectionBinding: Binding<UUID?> {
        Binding(
            get: { selectedVoiceID },
            set: { newValue in
                selectedVoiceID = newValue
                let selectedVoice = voices.first(where: { $0.id == newValue })
                onSelect(selectedVoice)
            }
        )
    }

    private var presetVoices: [VoiceLibraryRecord] {
        voices.filter(\.isModelDeclaredPreset)
    }

    private var savedVoices: [VoiceLibraryRecord] {
        voices.filter { !$0.isModelDeclaredPreset }
    }
}
