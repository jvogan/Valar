import SwiftUI
import ValarCore

struct RenderQueueView: View {
    @Bindable var state: ProjectWorkspaceState
    @State private var draggedJob: RenderJob?
    @State private var dropTargetJobID: UUID?

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Label("Render Queue", systemImage: "play.rectangle")
                    .font(.headline)
                Spacer()
                Text("\(state.renderJobs.count)")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 2)
                    .background(.surfaceBadge)
                    .clipShape(Capsule())
            }

            if let progress = state.overallRenderProgress {
                VStack(alignment: .leading, spacing: 6) {
                    ProgressView(value: progress)
                        .progressViewStyle(.linear)
                    Text(state.renderSummary)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            if state.renderJobs.isEmpty {
                VStack(spacing: 8) {
                    Image(systemName: "play.rectangle")
                        .font(.system(size: 20))
                        .foregroundStyle(.quaternary)
                    Text("No render jobs queued")
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                }
                .frame(maxWidth: .infinity, alignment: .center)
                .padding(.vertical, 16)
            } else {
                ForEach(state.renderJobs) { job in
                    renderJobRow(job)
                }
            }
        }
    }

    private func renderJobRow(_ job: RenderJob) -> some View {
        let isDragging = draggedJob?.id == job.id
        let isDropTarget = dropTargetJobID == job.id
        let isReorderable = job.state == .queued

        return RenderJobRowView(job: job) {
            Task { await state.cancelRender(job.id) }
        }
        .opacity(isDragging ? 0.4 : 1)
        .overlay(alignment: .top) {
            if isDropTarget && !isDragging {
                DropInsertionIndicator()
                    .offset(y: -3)
            }
        }
        .animation(.easeInOut(duration: 0.2), value: isDragging)
        .animation(.easeInOut(duration: 0.15), value: isDropTarget)
        .if(isReorderable) { view in
            view
                .draggable(job.id.uuidString) {
                    renderJobDragPreview(job)
                }
                .onDrop(
                    of: [.text],
                    delegate: ReorderDropDelegate(
                        item: job,
                        items: state.renderJobs,
                        onReorder: { source, destination in
                            withAnimation(.spring(response: 0.3, dampingFraction: 0.8)) {
                                state.moveRenderJobs(from: source, to: destination)
                            }
                        },
                        draggedItem: $draggedJob,
                        isTargeted: $dropTargetJobID
                    )
                )
        }
    }

    private func renderJobDragPreview(_ job: RenderJob) -> some View {
        HStack(spacing: 6) {
            Image(systemName: "clock")
                .font(.caption)
                .foregroundStyle(.orange)
            Text(job.title ?? job.outputFileName)
                .font(.caption.weight(.medium))
                .lineLimit(1)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .fill(.regularMaterial)
                .shadow(color: .black.opacity(0.15), radius: 8, y: 3)
        )
        .onAppear { draggedJob = job }
    }
}

// MARK: - Conditional modifier

private extension View {
    @ViewBuilder
    func `if`<Content: View>(_ condition: Bool, transform: (Self) -> Content) -> some View {
        if condition {
            transform(self)
        } else {
            self
        }
    }
}
