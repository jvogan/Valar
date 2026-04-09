import SwiftUI
import UniformTypeIdentifiers

// MARK: - Accepted types

enum ValarDragDrop {
    static let modelBundleType = UTType(filenameExtension: "valarmodel") ?? .package
    static let acceptedTextTypes: [UTType] = [.plainText, .utf8PlainText]
    static let acceptedAudioTypes: [UTType] = [.audio, .wav, .mpeg4Audio, .mp3]
    static let acceptedFileTypes: [UTType] = [.text, .plainText]
    static let acceptedModelImportTypes: [UTType] = [modelBundleType]
    static let acceptedScriptExtensions: Set<String> = ["txt", "md", "text", "fountain", "fdx"]
    static let acceptedAudioExtensions: Set<String> = ["wav", "m4a", "mp3", "aiff", "aif", "flac", "caf"]

    static func isTextFile(_ url: URL) -> Bool {
        acceptedScriptExtensions.contains(url.pathExtension.lowercased())
    }

    static func isAudioFile(_ url: URL) -> Bool {
        acceptedAudioExtensions.contains(url.pathExtension.lowercased())
    }
}

// MARK: - Reorder drop delegate

struct ReorderDropDelegate<Item: Identifiable>: DropDelegate where Item.ID: Equatable {
    let item: Item
    let items: [Item]
    let onReorder: (_ from: IndexSet, _ to: Int) -> Void
    let draggedItem: Binding<Item?>
    let isTargeted: Binding<Item.ID?>

    func dropEntered(info: DropInfo) {
        isTargeted.wrappedValue = item.id
        guard let dragged = draggedItem.wrappedValue,
              dragged.id != item.id,
              let fromIndex = items.firstIndex(where: { $0.id == dragged.id }),
              let toIndex = items.firstIndex(where: { $0.id == item.id }) else {
            return
        }
        let destination = fromIndex < toIndex ? toIndex + 1 : toIndex
        onReorder(IndexSet(integer: fromIndex), destination)
    }

    func dropExited(info: DropInfo) {
        if isTargeted.wrappedValue == item.id {
            isTargeted.wrappedValue = nil
        }
    }

    func dropUpdated(info: DropInfo) -> DropProposal? {
        DropProposal(operation: .move)
    }

    func performDrop(info: DropInfo) -> Bool {
        draggedItem.wrappedValue = nil
        isTargeted.wrappedValue = nil
        return true
    }
}

// MARK: - Drop indicator line

struct DropInsertionIndicator: View {
    var body: some View {
        HStack(spacing: 4) {
            Circle()
                .fill(Color.accentColor)
                .frame(width: 6, height: 6)
            Rectangle()
                .fill(Color.accentColor)
                .frame(height: 2)
            Circle()
                .fill(Color.accentColor)
                .frame(width: 6, height: 6)
        }
        .padding(.horizontal, 4)
        .transition(.opacity.combined(with: .scale(scale: 0.8, anchor: .leading)))
    }
}

