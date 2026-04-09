import SwiftUI
import ValarPersistence

struct VoiceDetailSheet: View {
    let voice: VoiceLibraryRecord
    let onDelete: () -> Void
    let onUseForGeneration: () -> Void
    @State private var isShowingDeleteConfirmation = false

    var body: some View {
        ScrollView {
        VStack(alignment: .leading, spacing: 20) {
            HStack {
                Image(systemName: "person.wave.2.fill")
                    .font(.largeTitle)
                    .foregroundStyle(Color.accentColor)
                VStack(alignment: .leading, spacing: 4) {
                    Text(voice.label)
                        .font(.title3.bold())
                    Text("Voice profile")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            Divider()

            Group {
                LabeledContent("Model ID") {
                    Text(voice.modelID)
                        .font(.caption)
                        .textSelection(.enabled)
                }
                LabeledContent("Voice Kind") {
                    Text(voice.typeDisplayName)
                        .font(.caption)
                }
                if voice.isModelDeclaredPreset, let backendVoiceID = voice.backendVoiceID {
                    LabeledContent("Preset Voice ID") {
                        Text(backendVoiceID)
                            .font(.caption)
                            .textSelection(.enabled)
                    }
                    LabeledContent("Mutability") {
                        Text("Immutable model preset")
                            .font(.caption)
                    }
                } else {
                    LabeledContent("Created") {
                        Text(voice.createdAt.formatted(date: .abbreviated, time: .shortened))
                            .font(.caption)
                    }
                }
                if let source = voice.sourceAssetName {
                    LabeledContent("Source") {
                        Text(source)
                            .font(.caption)
                    }
                }
                if let duration = voice.referenceDurationSeconds {
                    LabeledContent("Reference Duration") {
                        Text(String(format: "%.1f s", duration))
                            .font(.caption)
                    }
                }
                if let sampleRate = voice.referenceSampleRate {
                    LabeledContent("Sample Rate") {
                        Text(String(format: "%.0f Hz", sampleRate))
                            .font(.caption)
                    }
                }
                if let channelCount = voice.referenceChannelCount {
                    LabeledContent("Channels") {
                        Text(channelCount == 1 ? "Mono" : "\(channelCount)")
                            .font(.caption)
                    }
                }
                if let embedding = voice.speakerEmbedding {
                    LabeledContent("Speaker Embedding") {
                        Text("\(embedding.count / MemoryLayout<Float>.size) floats")
                            .font(.caption)
                    }
                }
                if voice.isLegacyExpressive {
                    LabeledContent("Continuity") {
                        Text("Expressive legacy prompt voice")
                            .font(.caption)
                    }
                }
                if let prompt = voice.voicePrompt, !prompt.isEmpty {
                    LabeledContent("Design Prompt") {
                        Text(prompt)
                            .font(.caption)
                            .textSelection(.enabled)
                    }
                }
            }

            Divider()

            VoiceAuditionView(voice: voice)

            Spacer()

            Button {
                onUseForGeneration()
            } label: {
                Label("Use For Generation", systemImage: "waveform")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)

            if voice.isMutable {
                Button(role: .destructive) {
                    isShowingDeleteConfirmation = true
                } label: {
                    Label("Delete Voice", systemImage: "trash")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
            }
        }
        }
        .padding()
        .confirmationDialog(
            "Delete voice \"\(voice.label)\"?",
            isPresented: $isShowingDeleteConfirmation,
            titleVisibility: .visible
        ) {
            Button("Delete", role: .destructive) {
                onDelete()
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("This permanently removes the saved voice profile. If this voice is used in any project chapters, those renders will need a different voice.")
        }
    }
}
