import AppKit
import SwiftUI
import UniformTypeIdentifiers

struct GeneratorPlayerView: View {
    @Bindable var state: GeneratorState
    @State private var saveErrorMessage: String?

    var body: some View {
        HStack(spacing: 12) {
            // Play/Pause button
            Button {
                state.togglePlayback()
            } label: {
                ZStack {
                    Image(systemName: state.isPlaying ? "pause.fill" : "play.fill")
                        .font(.title2)
                        .opacity(state.isPlaybackBuffering ? 0 : 1)

                    if state.isPlaybackBuffering {
                        ProgressView()
                            .controlSize(.small)
                    }
                }
            }
            .accessibilityLabel(state.isPlaying ? "Pause" : "Play")
            .buttonStyle(.plain)
            .keyboardShortcut(.space, modifiers: .option)

            // Waveform visualization with playback progress
            WaveformVisualizationView(
                samples: state.waveformSamples,
                progress: state.audioDuration > 0 ? state.playbackPosition / state.audioDuration : 0
            )
            .frame(height: 32)

            // Duration
            VStack(alignment: .trailing, spacing: 2) {
                Text(formatDuration(state.audioDuration))
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)

                if state.isPlaybackBuffering {
                    Text("Buffering…")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }
            }

            // Save button
            Button {
                saveAudio()
            } label: {
                Image(systemName: "square.and.arrow.down")
            }
            .accessibilityLabel("Save Audio")
            .buttonStyle(.plain)
            .disabled(!state.canSaveGeneratedAudio)
            .help(state.canSaveGeneratedAudio ? "Save audio file" : "Finish generation before saving")
        }
        .alert(
            "Save Audio Failed",
            isPresented: Binding(
                get: { saveErrorMessage != nil },
                set: { isPresented in
                    if !isPresented {
                        saveErrorMessage = nil
                    }
                }
            )
        ) {
            Button("OK", role: .cancel) {
                saveErrorMessage = nil
            }
        } message: {
            Text(saveErrorMessage ?? "")
        }
    }

    private func formatDuration(_ seconds: Double) -> String {
        let mins = Int(seconds) / 60
        let secs = Int(seconds) % 60
        return String(format: "%d:%02d", mins, secs)
    }

    private func saveAudio() {
        guard let destinationURL = saveAudioURL() else { return }

        Task { @MainActor in
            do {
                try await state.saveGeneratedAudio(to: destinationURL)
            } catch {
                let message = (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
                saveErrorMessage = PathRedaction.redactMessage(message)
            }
        }
    }

    private func saveAudioURL() -> URL? {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.wav]
        panel.canCreateDirectories = true
        panel.prompt = "Save Audio"
        panel.nameFieldStringValue = state.suggestedSaveAudioFilename

        guard panel.runModal() == .OK,
              let selectedURL = panel.url else {
            return nil
        }

        let wavExtension = UTType.wav.preferredFilenameExtension ?? "wav"
        guard selectedURL.pathExtension.lowercased() == wavExtension else {
            return selectedURL.appendingPathExtension(wavExtension)
        }

        return selectedURL
    }
}
