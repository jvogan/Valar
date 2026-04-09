import SwiftUI
import ValarModelKit

struct SpeakerFormView: View {
    @Bindable var workspaceState: ProjectWorkspaceState
    @Environment(\.dismiss) private var dismiss

    @State private var name = ""
    @State private var voiceModelRaw = ""
    @State private var language = "auto"

    private let languages = ["auto", "en", "zh", "ja", "ko", "fr", "de", "es", "it", "pt", "ru", "ar"]

    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            Text("Add Speaker")
                .font(.title2.bold())

            VStack(alignment: .leading, spacing: 12) {
                TextField("Speaker name", text: $name)
                    .textFieldStyle(.roundedBorder)

                TextField("Voice model ID (optional)", text: $voiceModelRaw)
                    .textFieldStyle(.roundedBorder)

                Picker("Language", selection: $language) {
                    ForEach(languages, id: \.self) { lang in
                        Text(lang == "auto" ? "Auto-detect" : lang.uppercased()).tag(lang)
                    }
                }
                .pickerStyle(.menu)
            }

            HStack {
                Spacer()
                Button("Cancel", role: .cancel) { dismiss() }
                    .buttonStyle(.bordered)
                Button("Add") {
                    let modelID: ModelIdentifier? = voiceModelRaw.isEmpty ? nil : ModelIdentifier(voiceModelRaw)
                    Task {
                        await workspaceState.addSpeaker(name: name, voiceModelID: modelID, language: language)
                        dismiss()
                    }
                }
                .buttonStyle(.borderedProminent)
                .disabled(name.trimmingCharacters(in: .whitespaces).isEmpty)
            }
        }
        .padding(24)
        .frame(minWidth: 360)
    }
}
