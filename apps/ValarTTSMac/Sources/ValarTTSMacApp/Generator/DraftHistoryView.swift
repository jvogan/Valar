import SwiftUI

struct DraftHistoryView: View {
    @Bindable var state: GeneratorState

    var body: some View {
        List(state.drafts) { draft in
            DraftHistoryRow(draft: draft) {
                state.restoreDraft(draft)
            }
            .listRowInsets(EdgeInsets(top: 4, leading: 8, bottom: 4, trailing: 8))
            .listRowSeparator(.hidden)
            .listRowBackground(Color.clear)
        }
        .listStyle(.plain)
    }
}

struct DraftHistoryRow: View {
    let draft: GeneratorState.DraftEntry
    let action: () -> Void

    @State private var isHovering = false

    var body: some View {
        Button(action: action) {
            HStack(alignment: .top, spacing: 12) {
                VStack(alignment: .leading, spacing: 6) {
                    Text(previewText)
                        .font(.callout)
                        .lineLimit(3)
                        .multilineTextAlignment(.leading)

                    HStack(spacing: 8) {
                        if let modelName {
                            Text(modelName)
                                .font(.caption2.weight(.semibold))
                                .foregroundStyle(.secondary)
                                .padding(.horizontal, 7)
                                .padding(.vertical, 3)
                                .background(.surfaceBadge, in: Capsule())
                        }

                        Text(draft.timestamp, style: .relative)
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                }

                Spacer(minLength: 8)

                Image(systemName: "chevron.right")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(isHovering ? .secondary : .tertiary)
            }
            .padding(12)
            .frame(maxWidth: .infinity, alignment: .leading)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .interactiveCardBackground(isHovered: isHovering, cornerRadius: 10)
        .onHover { isHovering = $0 }
    }

    private var previewText: String {
        let trimmed = draft.text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return "Empty draft" }
        return trimmed
    }

    private var modelName: String? {
        guard let modelID = draft.modelID else { return nil }
        let name = modelID.rawValue.split(separator: "/").last.map(String.init) ?? modelID.rawValue
        return name.isEmpty ? nil : name
    }
}
