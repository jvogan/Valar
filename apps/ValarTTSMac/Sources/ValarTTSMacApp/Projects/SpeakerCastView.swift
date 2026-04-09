import SwiftUI

struct SpeakerCastView: View {
    @Bindable var state: ProjectWorkspaceState
    let onAddSpeaker: () -> Void
    @State private var speakerToRemove: SpeakerEntry?

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            header
            Divider()
            speakerList
        }
        .frame(minHeight: 120, idealHeight: 180)
        .confirmationDialog(
            "Remove \(speakerToRemove?.name ?? "speaker")?",
            isPresented: Binding(
                get: { speakerToRemove != nil },
                set: { if !$0 { speakerToRemove = nil } }
            )
        ) {
            Button("Remove", role: .destructive) {
                if let id = speakerToRemove?.id {
                    Task { await state.removeSpeaker(id) }
                }
                speakerToRemove = nil
            }
        }
    }

    private var header: some View {
        HStack {
            Label("Cast", systemImage: "person.2")
                .font(.headline)
            Spacer()
            Button(action: onAddSpeaker) {
                Image(systemName: "plus.circle.fill")
                    .imageScale(.medium)
            }
            .buttonStyle(.plain)
            .foregroundStyle(Color.accentColor)
            .accessibilityLabel("Add Speaker")
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
    }

    private var speakerList: some View {
        ScrollView {
            LazyVStack(spacing: 2) {
                if state.speakers.isEmpty {
                    Text("No speakers assigned")
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                        .frame(maxWidth: .infinity, alignment: .center)
                        .padding(.vertical, 16)
                } else {
                    ForEach(state.speakers) { speaker in
                        speakerRow(speaker)
                    }
                }
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
        }
    }

    private func speakerRow(_ speaker: SpeakerEntry) -> some View {
        HStack(spacing: 8) {
            Image(systemName: "person.circle.fill")
                .foregroundStyle(Color.accentColor.opacity(0.7))
                .imageScale(.large)

            VStack(alignment: .leading, spacing: 2) {
                Text(speaker.name)
                    .font(.subheadline.weight(.medium))
                    .lineLimit(1)
                HStack(spacing: 6) {
                    if let modelID = speaker.voiceModelID {
                        Text(modelID.rawValue)
                            .font(.caption2)
                            .foregroundStyle(.tertiary)
                            .lineLimit(1)
                    }
                    if speaker.language != "auto" {
                        Text(speaker.language.uppercased())
                            .font(.caption2.weight(.semibold))
                            .foregroundStyle(.secondary)
                    }
                }
            }

            Spacer()

            Button {
                speakerToRemove = speaker
            } label: {
                Image(systemName: "xmark.circle")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Remove Speaker")
        }
        .padding(.vertical, 6)
        .padding(.horizontal, 10)
    }
}
