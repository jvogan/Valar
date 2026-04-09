import SwiftUI
import ValarCore
import ValarPersistence

struct ChapterTimelineView: View {
    @Bindable var state: ProjectWorkspaceState

    @State private var dropTargetChapterID: UUID?

    private let headerWidth: CGFloat = 150
    private let laneHeight: CGFloat = 52

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            rulerBar
            Divider()
            if state.chapters.isEmpty {
                emptyState
            } else {
                ScrollView(.vertical, showsIndicators: true) {
                    LazyVStack(spacing: 0) {
                        ForEach(state.chapters) { chapter in
                            chapterLane(chapter)
                        }
                    }
                }
                .animation(
                    .spring(response: 0.35, dampingFraction: 0.75),
                    value: state.chapters.map(\.id)
                )
            }
        }
        .background(Color(nsColor: .controlBackgroundColor))
    }

    // MARK: - Ruler

    private var rulerBar: some View {
        HStack(spacing: 0) {
            Text("TRACKS")
                .font(.system(size: 9, weight: .bold, design: .rounded))
                .foregroundStyle(.quaternary)
                .frame(width: headerWidth, alignment: .leading)
                .padding(.leading, 14)

            Rectangle()
                .fill(.quaternary.opacity(0.2))
                .frame(width: 1)

            GeometryReader { geo in
                let count = max(2, Int(geo.size.width / 72))
                HStack(spacing: 0) {
                    ForEach(0..<count, id: \.self) { i in
                        rulerMark(i, total: count)
                    }
                }
                .padding(.horizontal, 8)
            }
        }
        .frame(height: 22)
        .background(.quaternary.opacity(0.04))
    }

    private func rulerMark(_ index: Int, total: Int) -> some View {
        HStack(spacing: 0) {
            Rectangle()
                .fill(.quaternary.opacity(0.25))
                .frame(width: 1, height: 10)
            Text(rulerLabel(index))
                .font(.system(size: 9, weight: .medium, design: .monospaced))
                .foregroundStyle(.quaternary)
                .padding(.leading, 3)
            Spacer(minLength: 0)
        }
        .frame(maxWidth: .infinity)
    }

    // MARK: - Lanes

    @ViewBuilder
    private func chapterLane(_ chapter: ChapterRecord) -> some View {
        let isSelected = state.selectedChapterID == chapter.id
        let status = state.renderStatus(for: chapter.id)
        let isDropTarget = dropTargetChapterID == chapter.id

        Button {
            state.selectedChapterID = chapter.id
        } label: {
            HStack(spacing: 0) {
                trackHeader(chapter, status: status, isSelected: isSelected)
                    .frame(width: headerWidth)

                Rectangle()
                    .fill(.quaternary.opacity(0.2))
                    .frame(width: 1)

                GeometryReader { geo in
                    durationBar(chapter, status: status, availableWidth: geo.size.width)
                        .padding(.vertical, 8)
                        .padding(.horizontal, 6)
                }
            }
        }
        .buttonStyle(.plain)
        .frame(height: laneHeight)
        .background(laneBackground(isSelected: isSelected))
        .overlay(alignment: .top) {
            if isDropTarget {
                Rectangle()
                    .fill(Color.accentColor)
                    .frame(height: 2)
                    .transition(.opacity)
            }
        }
        .overlay(alignment: .bottom) {
            Rectangle()
                .fill(.quaternary.opacity(0.12))
                .frame(height: 1)
        }
        .contentShape(Rectangle())
        .accessibilityLabel(chapter.title)
        .draggable(chapter.id.uuidString) {
            dragPreview(chapter)
        }
        .dropDestination(for: String.self) { items, _ in
            handleDrop(items, onto: chapter)
        } isTargeted: { targeted in
            withAnimation(.easeInOut(duration: 0.12)) {
                dropTargetChapterID = targeted ? chapter.id : nil
            }
        }
    }

    @ViewBuilder
    private func trackHeader(
        _ chapter: ChapterRecord,
        status: ChapterRenderStatus,
        isSelected: Bool
    ) -> some View {
        HStack(spacing: 8) {
            // Status indicator
            Circle()
                .fill(statusColor(status))
                .frame(width: 7, height: 7)
                .modifier(PulseEffect(active: status.isRendering))
                .accessibilityLabel(statusLabel(status))

            VStack(alignment: .leading, spacing: 1) {
                Text(chapter.title)
                    .font(.system(size: 11, weight: isSelected ? .bold : .medium))
                    .lineLimit(1)

                HStack(spacing: 4) {
                    if let speaker = chapter.speakerLabel {
                        Text(speaker)
                            .font(.system(size: 9, weight: .semibold))
                            .foregroundStyle(.secondary)
                    }
                    if let dur = chapter.estimatedDurationSeconds {
                        Text(formatDuration(dur))
                            .font(.system(size: 9, design: .monospaced))
                            .foregroundStyle(.tertiary)
                    }
                }
            }

            Spacer(minLength: 0)
        }
        .padding(.horizontal, 12)
    }

    @ViewBuilder
    private func durationBar(
        _ chapter: ChapterRecord,
        status: ChapterRenderStatus,
        availableWidth: CGFloat
    ) -> some View {
        let width = barWidth(for: chapter, available: availableWidth)

        ZStack(alignment: .leading) {
            // Background bar
            RoundedRectangle(cornerRadius: 5, style: .continuous)
                .fill(statusColor(status).opacity(0.14))
                .frame(width: width)

            // Filled portion (progress or complete)
            if let progress = status.progress {
                RoundedRectangle(cornerRadius: 5, style: .continuous)
                    .fill(
                        LinearGradient(
                            colors: statusGradient(status),
                            startPoint: .leading,
                            endPoint: .trailing
                        )
                    )
                    .frame(width: max(0, width * progress))
                    .animation(.easeInOut(duration: 0.3), value: progress)
            }

            // Title label
            Text(chapter.title)
                .font(.system(size: 9, weight: .semibold))
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .padding(.horizontal, 8)
                .frame(width: width, alignment: .leading)
        }
        .frame(maxHeight: .infinity)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func dragPreview(_ chapter: ChapterRecord) -> some View {
        HStack(spacing: 6) {
            Image(systemName: "line.3.horizontal")
                .font(.system(size: 10))
                .foregroundStyle(.secondary)
            Text(chapter.title)
                .font(.system(size: 11, weight: .medium))
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .background(.ultraThinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
    }

    // MARK: - Empty state

    private var emptyState: some View {
        VStack(spacing: 6) {
            Image(systemName: "waveform")
                .font(.title2)
                .foregroundStyle(.quaternary)
            Text("No tracks yet")
                .font(.caption)
                .foregroundStyle(.tertiary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding(20)
    }

    // MARK: - Helpers

    private func handleDrop(_ items: [String], onto target: ChapterRecord) -> Bool {
        guard let idString = items.first,
              let droppedID = UUID(uuidString: idString),
              let from = state.chapters.firstIndex(where: { $0.id == droppedID }),
              let to = state.chapters.firstIndex(where: { $0.id == target.id }),
              from != to else { return false }
        Task { await state.moveChapter(from: from, to: to) }
        dropTargetChapterID = nil
        return true
    }

    private func laneBackground(isSelected: Bool) -> Color {
        isSelected ? Color.accentColor.opacity(0.07) : .clear
    }

    private func statusColor(_ status: ChapterRenderStatus) -> Color {
        switch status {
        case .idle: return .secondary
        case .queued: return .gray
        case .rendering: return .blue
        case .complete: return .green
        case .failed: return .red
        }
    }

    private func statusLabel(_ status: ChapterRenderStatus) -> String {
        switch status {
        case .idle: return "Draft"
        case .queued: return "Queued"
        case .rendering: return "Rendering"
        case .complete: return "Complete"
        case .failed: return "Failed"
        }
    }

    private func statusGradient(_ status: ChapterRenderStatus) -> [Color] {
        switch status {
        case .rendering: return [.blue.opacity(0.6), .cyan.opacity(0.5)]
        case .complete: return [.green.opacity(0.5), .mint.opacity(0.4)]
        case .failed: return [.red.opacity(0.5), .orange.opacity(0.4)]
        default: return [statusColor(status).opacity(0.3)]
        }
    }

    private func barWidth(for chapter: ChapterRecord, available: CGFloat) -> CGFloat {
        let maxDur = state.chapters.compactMap(\.estimatedDurationSeconds).max() ?? 30
        let dur = chapter.estimatedDurationSeconds ?? estimateDuration(chapter.script)
        let ratio = min(1.0, dur / max(maxDur, 1))
        return max(50, available * ratio)
    }

    private func estimateDuration(_ script: String) -> Double {
        max(3, Double(script.split(separator: " ").count) / 150.0 * 60.0)
    }

    private func rulerLabel(_ i: Int) -> String {
        let sec = i * 10
        return String(format: "%d:%02d", sec / 60, sec % 60)
    }

    private func formatDuration(_ s: Double) -> String {
        String(format: "%d:%02d", Int(s) / 60, Int(s) % 60)
    }
}

// MARK: - Pulse animation

private struct PulseEffect: ViewModifier {
    let active: Bool
    @State private var on = false

    func body(content: Content) -> some View {
        content
            .opacity(on ? 0.35 : 1)
            .onChange(of: active) { _, newValue in
                if newValue {
                    withAnimation(
                        .easeInOut(duration: 0.7)
                        .repeatForever(autoreverses: true)
                    ) {
                        on = true
                    }
                } else {
                    withAnimation(.easeInOut(duration: 0.2)) {
                        on = false
                    }
                }
            }
            .onAppear {
                if active {
                    withAnimation(
                        .easeInOut(duration: 0.7)
                        .repeatForever(autoreverses: true)
                    ) {
                        on = true
                    }
                }
            }
    }
}
