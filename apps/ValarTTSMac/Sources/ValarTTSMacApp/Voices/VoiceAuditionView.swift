import SwiftUI
import ValarModelKit
import ValarPersistence

struct VoiceAuditionView: View {
    @Environment(AppState.self) private var appState
    let voice: VoiceLibraryRecord

    @State private var auditionText = "Hello, this is a preview of the selected voice."
    @State private var isGenerating = false
    @State private var statusText = "Generate a quick preview with this saved voice."
    @State private var isVisible = false

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Audition")
                .font(.headline)

            TextField("Enter text to preview", text: $auditionText)
                .textFieldStyle(.roundedBorder)

            HStack {
                Button {
                    Task { await previewVoice() }
                } label: {
                    Label("Preview", systemImage: "play.fill")
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.small)
                .disabled(isGenerating || auditionText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)

                if isGenerating {
                    ProgressView()
                        .controlSize(.small)
                }
            }

            Text(statusText)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .onAppear {
            isVisible = true
        }
        .onDisappear {
            isVisible = false
            Task {
                await appState.services.audioPlayer.stop()
            }
        }
    }

    private func previewVoice() async {
        let prompt = auditionText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty else { return }

        isGenerating = true
        statusText = "Generating preview..."
        defer { isGenerating = false }

        do {
            await appState.services.audioPlayer.stop()
            let modelID = voice.preferredRuntimeModelIdentifier ?? ModelIdentifier(voice.modelID)
            let buffer = try await appState.services.synthesizePreview(
                text: prompt,
                modelID: modelID,
                voiceRecord: voice
            )
            guard isVisible else { return }
            try await appState.services.audioPlayer.play(buffer)
            statusText = "Preview ready."
        } catch {
            statusText = "Preview failed: \(error.localizedDescription)"
        }
    }
}
