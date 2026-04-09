import SwiftUI
import ValarPersistence

struct ProjectCardView: View {
    let project: ProjectRecord
    let isSelected: Bool
    let onSelect: () -> Void
    let onDelete: () -> Void

    @State private var isHovering = false
    @State private var showDeleteConfirmation = false

    var body: some View {
        HStack(spacing: 10) {
            Button(action: onSelect) {
                HStack(spacing: 12) {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(project.title)
                            .font(.headline)
                            .lineLimit(2)
                            .fixedSize(horizontal: false, vertical: true)
                        Text(project.updatedAt.formatted(date: .abbreviated, time: .omitted))
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }

                    Spacer(minLength: 12)

                    Image(systemName: isSelected ? "checkmark.circle.fill" : "circle")
                        .foregroundStyle(isSelected ? Color.accentColor : Color.secondary.opacity(0.45))
                        .imageScale(.medium)

                    Image(systemName: "chevron.right")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(isHovering || isSelected ? AnyShapeStyle(.secondary) : AnyShapeStyle(.tertiary))
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)

            if isHovering {
                Button(role: .destructive) {
                    showDeleteConfirmation = true
                } label: {
                    Image(systemName: "trash")
                        .font(.caption.weight(.semibold))
                        .frame(width: 28, height: 28)
                }
                .buttonStyle(.plain)
                .background(.surfaceRecessed, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
                .foregroundStyle(.secondary)
            }
        }
        .padding(ValarSpacing.cardPadding)
        .interactiveCardBackground(isSelected: isSelected, isHovered: isHovering)
        .buttonStyle(.plain)
        .onHover { isHovering = $0 }
        .confirmationDialog("Delete this project?", isPresented: $showDeleteConfirmation) {
            Button("Delete", role: .destructive, action: onDelete)
        }
    }
}
