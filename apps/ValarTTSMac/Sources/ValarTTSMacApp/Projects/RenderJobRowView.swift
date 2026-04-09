import SwiftUI
import ValarCore

struct RenderJobRowView: View {
    let job: RenderJob
    let onCancel: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                stateIcon
                Text(job.title ?? job.id.uuidString.prefix(8).description)
                    .font(.subheadline.weight(.medium))
                    .lineLimit(1)
                Spacer()
                stateBadge
                if job.state == .queued || job.state == .running {
                    Button(action: onCancel) {
                        Image(systemName: "xmark.circle")
                            .font(.caption)
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(.secondary)
                    .accessibilityLabel("Cancel Job")
                    .transition(.scale.combined(with: .opacity))
                }
            }

            if job.state == .running {
                GeometryReader { geo in
                    ZStack(alignment: .leading) {
                        Capsule()
                            .fill(.quaternary)

                        Capsule()
                            .fill(StatusColor.info.gradient)
                            .frame(width: max(0, geo.size.width * job.progress))
                    }
                }
                .frame(height: 5)
                .clipShape(Capsule())
                .transition(.opacity.combined(with: .scale(scale: 0.95, anchor: .leading)))
                .animation(.easeInOut(duration: 0.35), value: job.progress)
            }

            Text(job.outputFileName)
                .font(.caption2.monospaced())
                .foregroundStyle(.secondary)

            if job.state == .failed, let failureReason = job.failureReason {
                Text(failureReason)
                    .font(.caption)
                    .foregroundStyle(StatusColor.error)
                    .lineLimit(2)
                    .fixedSize(horizontal: false, vertical: true)
            }

            Text(job.createdAt.formatted(date: .abbreviated, time: .shortened))
                .font(.caption2)
                .foregroundStyle(.tertiary)
        }
        .padding(.vertical, 8)
        .padding(.horizontal, 12)
        .background(.surfacePrimary)
        .clipShape(RoundedRectangle(cornerRadius: ValarRadius.field, style: .continuous))
        .animation(.spring(duration: 0.4, bounce: 0.15), value: job.state)
        .animation(.spring(duration: 0.3), value: job.state == .queued || job.state == .running)
    }

    @ViewBuilder
    private var stateIcon: some View {
        switch job.state {
        case .queued:
            Image(systemName: "clock")
                .foregroundStyle(StatusColor.warning)
                .contentTransition(.symbolEffect(.replace))
        case .running:
            Image(systemName: "bolt.fill")
                .foregroundStyle(StatusColor.info)
                .symbolEffect(.pulse, isActive: true)
        case .completed:
            Image(systemName: "checkmark.circle.fill")
                .foregroundStyle(StatusColor.success)
                .contentTransition(.symbolEffect(.replace))
        case .cancelled:
            Image(systemName: "slash.circle")
                .foregroundStyle(StatusColor.neutral)
                .contentTransition(.symbolEffect(.replace))
        case .failed:
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(StatusColor.error)
                .contentTransition(.symbolEffect(.replace))
        }
    }

    private var stateBadge: some View {
        Text(stateLabel)
            .font(.caption2.weight(.semibold))
            .padding(.horizontal, 8)
            .padding(.vertical, 3)
            .background(StatusColor.badge(badgeColor))
            .clipShape(Capsule())
            .contentTransition(.interpolate)
    }

    private var badgeColor: Color {
        switch job.state {
        case .queued: return StatusColor.warning
        case .running: return StatusColor.info
        case .completed: return StatusColor.success
        case .cancelled: return StatusColor.neutral
        case .failed: return StatusColor.error
        }
    }

    private var stateLabel: String {
        switch job.state {
        case .queued: return "Queued"
        case .running: return "Rendering"
        case .completed: return "Complete"
        case .cancelled: return "Cancelled"
        case .failed: return "Failed"
        }
    }
}
