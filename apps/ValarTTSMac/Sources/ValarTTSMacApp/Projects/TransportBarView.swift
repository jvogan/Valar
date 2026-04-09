import SwiftUI
import ValarCore

struct TransportBarView: View {
    @Bindable var state: ProjectWorkspaceState

    var body: some View {
        HStack(spacing: 0) {
            transportControls
                .padding(.horizontal, 14)

            Divider()
                .frame(height: 24)

            statusLCD
                .padding(.horizontal, 14)

            Spacer()

            if let progress = state.overallRenderProgress {
                masterProgressBar(progress)
                    .padding(.trailing, 16)
            }
        }
        .frame(height: 44)
        .background(.ultraThinMaterial)
        .overlay(alignment: .bottom) {
            Rectangle()
                .fill(.quaternary.opacity(0.15))
                .frame(height: 1)
        }
    }

    // MARK: - Transport controls

    private var transportControls: some View {
        HStack(spacing: 6) {
            controlButton(icon: "arrow.left.to.line.compact", size: 11) {
                state.selectedChapterID = state.chapters.first?.id
            }
            .accessibilityLabel("Select First Chapter")
            .help("Select First Chapter")
            .disabled(state.chapters.isEmpty)

            controlButton(
                icon: state.hasActiveRender ? "xmark.circle.fill" : "sparkles.rectangle.stack",
                size: state.hasActiveRender ? 12 : 11,
                accent: true
            ) {
                if state.hasActiveRender {
                    Task { await state.stopAllRenders() }
                } else {
                    Task { await state.startRender() }
                }
            }
            .accessibilityLabel(state.hasActiveRender ? "Cancel Render" : "Start Render")
            .help(state.hasActiveRender ? "Cancel Render" : "Start Render")
            .disabled(!state.canStartRender && !state.hasActiveRender)

            controlButton(icon: "xmark.circle", size: 11) {
                Task { await state.stopAllRenders() }
            }
            .accessibilityLabel("Cancel Render")
            .help("Cancel Render")
            .disabled(!state.hasActiveRender)
        }
    }

    private func controlButton(
        icon: String,
        size: CGFloat,
        accent: Bool = false,
        action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            Image(systemName: icon)
                .font(.system(size: size, weight: .semibold))
                .frame(width: 28, height: 28)
                .background(
                    accent
                        ? AnyShapeStyle(Color.accentColor.opacity(0.12))
                        : AnyShapeStyle(Color.secondary.opacity(0.08))
                )
                .clipShape(Circle())
        }
        .buttonStyle(.plain)
        .foregroundStyle(accent ? Color.accentColor : .primary)
    }

    // MARK: - LCD status display

    private var statusLCD: some View {
        VStack(alignment: .leading, spacing: 1) {
            Text(state.renderSummary)
                .font(.system(size: 10, weight: .medium, design: .monospaced))
                .lineLimit(1)
                .foregroundStyle(.primary)

            if state.hasActiveRender {
                HStack(spacing: 3) {
                    Circle()
                        .fill(.blue)
                        .frame(width: 4, height: 4)
                    Text("RENDER")
                        .font(.system(size: 8, weight: .heavy, design: .monospaced))
                        .foregroundStyle(.blue)
                }
            } else {
                Text("\(state.chapters.count) track\(state.chapters.count == 1 ? "" : "s")")
                    .font(.system(size: 8, weight: .medium, design: .monospaced))
                    .foregroundStyle(.tertiary)
            }
        }
    }

    // MARK: - Master progress bar

    @ViewBuilder
    private func masterProgressBar(_ progress: Double) -> some View {
        HStack(spacing: 8) {
            Text("\(Int(progress * 100))%")
                .font(.system(size: 10, weight: .bold, design: .monospaced))
                .foregroundStyle(.secondary)
                .frame(width: 30, alignment: .trailing)

            ZStack(alignment: .leading) {
                RoundedRectangle(cornerRadius: 2.5, style: .continuous)
                    .fill(.quaternary.opacity(0.2))
                    .frame(width: 100, height: 5)

                RoundedRectangle(cornerRadius: 2.5, style: .continuous)
                    .fill(
                        LinearGradient(
                            colors: [.blue, .cyan],
                            startPoint: .leading,
                            endPoint: .trailing
                        )
                    )
                    .frame(width: max(0, 100 * progress), height: 5)
                    .animation(.easeInOut(duration: 0.3), value: progress)
            }
        }
    }
}
