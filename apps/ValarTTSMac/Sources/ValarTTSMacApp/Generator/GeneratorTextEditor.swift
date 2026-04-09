import AppKit
import SwiftUI

struct GeneratorTextEditor: View {
    @Bindable var state: GeneratorState

    var body: some View {
        VStack(spacing: 0) {
            GeneratorEditorRepresentable(state: state)
                .frame(maxWidth: .infinity, maxHeight: .infinity)

            // Stats bar
            HStack {
                Text("\(state.wordCount) words")
                    .foregroundStyle(.secondary)
                Text("\(state.text.count) characters")
                    .foregroundStyle(.secondary)
                Spacer()
                if state.isGenerating {
                    Text("Generating...")
                        .foregroundStyle(.secondary)
                }
            }
            .font(.caption)
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .background(.bar)
        }
    }
}

private struct GeneratorEditorRepresentable: NSViewRepresentable {
    @Bindable var state: GeneratorState

    func makeCoordinator() -> Coordinator {
        Coordinator(state: state)
    }

    func makeNSView(context: Context) -> NSScrollView {
        let scrollView = NSScrollView()
        scrollView.drawsBackground = false
        scrollView.hasVerticalScroller = true
        scrollView.hasHorizontalScroller = false
        scrollView.autohidesScrollers = true
        scrollView.borderType = .noBorder

        let textView = GeneratorUndoTextView()
        textView.delegate = context.coordinator
        textView.externalUndoManager = state.undoManager
        textView.allowsUndo = false
        textView.isRichText = false
        textView.importsGraphics = false
        textView.isAutomaticQuoteSubstitutionEnabled = false
        textView.isAutomaticDashSubstitutionEnabled = false
        textView.drawsBackground = false
        textView.backgroundColor = .clear
        textView.font = .systemFont(ofSize: NSFont.systemFontSize)
        textView.minSize = .zero
        textView.maxSize = NSSize(
            width: CGFloat.greatestFiniteMagnitude,
            height: CGFloat.greatestFiniteMagnitude
        )
        textView.isVerticallyResizable = true
        textView.isHorizontallyResizable = false
        textView.autoresizingMask = [.width]
        textView.textContainerInset = NSSize(width: 12, height: 12)
        textView.string = state.text
        textView.textContainer?.containerSize = NSSize(width: scrollView.contentSize.width, height: .greatestFiniteMagnitude)
        textView.textContainer?.widthTracksTextView = true

        scrollView.documentView = textView
        return scrollView
    }

    func updateNSView(_ scrollView: NSScrollView, context: Context) {
        guard let textView = scrollView.documentView as? GeneratorUndoTextView else { return }

        textView.externalUndoManager = state.undoManager

        guard textView.string != state.text else { return }

        context.coordinator.isApplyingStateUpdate = true
        let selectedRange = textView.selectedRange()
        textView.string = state.text
        textView.setSelectedRange(
            NSRange(
                location: min(selectedRange.location, textView.string.utf16.count),
                length: 0
            )
        )
        context.coordinator.isApplyingStateUpdate = false
    }

    final class Coordinator: NSObject, NSTextViewDelegate {
        @MainActor
        private let state: GeneratorState
        var isApplyingStateUpdate = false

        @MainActor
        init(state: GeneratorState) {
            self.state = state
        }

        func textDidChange(_ notification: Notification) {
            guard !isApplyingStateUpdate,
                  let textView = notification.object as? NSTextView else {
                return
            }

            Task { @MainActor [state] in
                state.applyTextEdit(textView.string)
            }
        }
    }
}

private final class GeneratorUndoTextView: NSTextView {
    weak var externalUndoManager: UndoManager?

    override var undoManager: UndoManager? {
        externalUndoManager
    }
}
