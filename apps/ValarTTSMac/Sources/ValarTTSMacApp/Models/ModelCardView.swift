import SwiftUI
import ValarCore
import ValarModelKit

struct ModelCardView: View {
    let model: CatalogModel
    let downloadProgress: Double?
    let diskUsageBytes: Int
    let installError: String?

    var body: some View {
        HStack(spacing: 14) {
            domainIconView

            VStack(alignment: .leading, spacing: 4) {
                HStack(alignment: .firstTextBaseline, spacing: 8) {
                    Text(model.descriptor.displayName)
                        .font(.body.weight(.semibold))
                        .lineLimit(1)
                    if let badge = supportBadge {
                        Text(badge.text)
                            .font(.system(size: 9, weight: .bold, design: .rounded))
                            .foregroundStyle(badge.color)
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(badge.color.opacity(0.12))
                            .clipShape(Capsule())
                    }
                    if model.isRecommended {
                        Image(systemName: "star.fill")
                            .font(.caption2)
                            .foregroundStyle(.orange)
                    }
                }

                HStack(spacing: 6) {
                    familyBadge
                    domainLabel
                    if let licenseName = model.licenseName {
                        licenseBadge(licenseName)
                    }
                }

                HStack(spacing: 4) {
                    Text(model.id.rawValue)
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                        .lineLimit(1)
                    if diskUsageBytes > 0 {
                        Text("·")
                            .font(.caption)
                            .foregroundStyle(.quaternary)
                        Text(ByteCountFormatter.string(fromByteCount: Int64(diskUsageBytes), countStyle: .file))
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(.tertiary)
                    }
                }

                if let installError, !installError.isEmpty {
                    Label(installError, systemImage: "exclamationmark.triangle.fill")
                        .font(.caption)
                        .foregroundStyle(.red)
                        .lineLimit(2)
                }
            }

            Spacer(minLength: 8)

            if let progress = downloadProgress {
                CompactDownloadIndicator(progress: progress)
            } else {
                StatusBadge(state: model.installState)
            }
        }
        .padding(.vertical, 6)
    }

    private var domainIconView: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .fill(domainColor.opacity(0.12))
                .frame(width: 36, height: 36)

            Image(systemName: domainIcon)
                .font(.system(size: 15, weight: .medium))
                .foregroundStyle(domainColor)
        }
    }

    private var domainIcon: String {
        switch model.descriptor.domain {
        case .tts: return "waveform"
        case .stt: return "mic.fill"
        case .sts: return "arrow.triangle.swap"
        case .codec: return "doc.zipper"
        case .utility: return "gearshape.fill"
        }
    }

    private var domainColor: Color {
        switch model.descriptor.domain {
        case .tts: return .blue
        case .stt: return .purple
        case .sts: return .teal
        case .codec: return .indigo
        case .utility: return .gray
        }
    }

    private var domainLabel: some View {
        Text(model.descriptor.domain.rawValue.uppercased())
            .font(.system(size: 9, weight: .bold, design: .rounded))
            .foregroundStyle(domainColor.opacity(0.7))
            .padding(.horizontal, 5)
            .padding(.vertical, 1.5)
            .background(domainColor.opacity(0.08))
            .clipShape(RoundedRectangle(cornerRadius: 3, style: .continuous))
    }

    private var familyBadge: some View {
        Text(familyDisplayName(model.familyID.rawValue))
            .font(.caption2.weight(.semibold))
            .foregroundStyle(.secondary)
            .padding(.horizontal, 7)
            .padding(.vertical, 2)
            .background(.quaternary.opacity(0.5))
            .clipShape(Capsule())
    }

    private func familyDisplayName(_ rawValue: String) -> String {
        switch rawValue {
        case "qwen3_tts": return "Qwen3 TTS"
        case "tada_tts": return "TADA"
        case "voxtral_tts": return "Voxtral"
        case "orpheus_tts": return "Orpheus"
        case "marvis_tts": return "Marvis"
        case "chatterbox_tts": return "Chatterbox"
        case "soprano": return "Soprano"
        case "pocket_tts": return "Pocket TTS"
        case "qwen3_asr": return "Qwen3 ASR"
        case "qwen3_forced_aligner": return "Qwen3 Aligner"
        case "whisper": return "Whisper"
        case "lfm_audio": return "LFM Audio"
        case "vibevoice_realtime_tts": return "VibeVoice"
        default: return rawValue
        }
    }

    private var supportBadge: (text: String, color: Color)? {
        if model.distributionTier == .compatibilityPreview {
            return ("Compatibility", .teal)
        }
        switch model.supportTier {
        case .preview:
            return ("Preview", .orange)
        case .experimental:
            return ("Experimental", .mint)
        case .supported:
            return nil
        }
    }

    private func licenseBadge(_ licenseName: String) -> some View {
        Text(licenseName)
            .font(.caption2.weight(.semibold))
            .foregroundStyle(.orange)
            .padding(.horizontal, 7)
            .padding(.vertical, 2)
            .background(.orange.opacity(0.1))
            .clipShape(Capsule())
    }
}

/// Compact inline progress ring shown on the card row during downloads.
private struct CompactDownloadIndicator: View {
    let progress: Double

    var body: some View {
        ZStack {
            Circle()
                .stroke(Color.accentColor.opacity(0.15), lineWidth: 3)
            Circle()
                .trim(from: 0, to: CGFloat(max(progress, 0.02)))
                .stroke(Color.accentColor, style: StrokeStyle(lineWidth: 3, lineCap: .round))
                .rotationEffect(.degrees(-90))
            Text("\(Int(progress * 100))")
                .font(.system(size: 8, weight: .bold, design: .rounded).monospacedDigit())
                .foregroundStyle(.secondary)
        }
        .frame(width: 28, height: 28)
    }
}
