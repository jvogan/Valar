import SwiftUI
import ValarPersistence

struct VoiceCardView: View {
    let voice: VoiceLibraryRecord
    let isSelected: Bool

    @State private var isHovering = false

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Image(systemName: "person.wave.2.fill")
                    .font(.title2)
                    .foregroundStyle(Color.accentColor)
                Spacer()
                Image(systemName: "chevron.right")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(isHovering || isSelected ? AnyShapeStyle(.secondary) : AnyShapeStyle(.tertiary))
            }

            Text(voice.label)
                .font(.headline)
                .lineLimit(2)
                .fixedSize(horizontal: false, vertical: true)

            Text(voice.modelID)
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(1)

            HStack {
                if voice.isModelDeclaredPreset, let backendVoiceID = voice.backendVoiceID {
                    Text(backendVoiceID)
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                } else {
                    Text(voice.createdAt.formatted(date: .abbreviated, time: .omitted))
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }
                Spacer()
                Label(voice.typeDisplayName, systemImage: badgeSystemImage)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(ValarSpacing.cardPadding)
        .interactiveCardBackground(isSelected: isSelected, isHovered: isHovering)
        .onHover { isHovering = $0 }
    }

    private var badgeSystemImage: String {
        switch voice.effectiveVoiceKind {
        case "preset":
            return "lock.fill"
        case "clonePrompt", "embeddingOnly", "tadaReference":
            return "waveform"
        case "legacyPrompt":
            return "wand.and.stars"
        case "namedSpeaker":
            return "person.fill.checkmark"
        default:
            return "person.wave.2.fill"
        }
    }
}
