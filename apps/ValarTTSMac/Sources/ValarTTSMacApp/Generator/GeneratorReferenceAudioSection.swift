import SwiftUI
import UniformTypeIdentifiers

struct GeneratorReferenceAudioSection: View {
    @Bindable var state: GeneratorState
    @State private var isImportingFile = false
    @State private var isDropTargeted = false

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .firstTextBaseline) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Reference Audio")
                        .font(.headline)
                    Text(
                        state.requiresInlineReferenceAudio
                            ? "TADA needs a reference clip at generation time. Transcript is optional for English and recommended elsewhere."
                            : "Optional reference clip for audio-conditioned speech models."
                    )
                    .font(.caption)
                    .foregroundStyle(.secondary)
                }
                Spacer()
                if state.requiresInlineReferenceAudio {
                    Text("Required")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.secondary)
                }
            }

            dropZone

            VStack(alignment: .leading, spacing: 6) {
                Text("Reference Transcript")
                    .font(.subheadline.weight(.medium))
                TextEditor(text: $state.referenceTranscript)
                    .frame(minHeight: 68, maxHeight: 96)
                    .padding(8)
                    .background(.surfaceRecessed)
                    .clipShape(RoundedRectangle(cornerRadius: ValarRadius.field, style: .continuous))
            }
        }
        .padding(.horizontal)
        .padding(.vertical, 14)
        .fileImporter(
            isPresented: $isImportingFile,
            allowedContentTypes: [.wav, UTType(filenameExtension: "m4a")!],
            allowsMultipleSelection: false
        ) { result in
            handleImport(result)
        }
    }

    private var dropZone: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 12) {
                Image(systemName: state.selectedReferenceAudioURL == nil ? "waveform.badge.plus" : "waveform")
                    .font(.title3)
                    .foregroundStyle(isDropTargeted ? Color.accentColor : .secondary)
                VStack(alignment: .leading, spacing: 3) {
                    Text(state.selectedReferenceAudioURL?.lastPathComponent ?? "Drop a WAV or M4A clip here")
                        .font(.body.weight(.medium))
                    Text(state.selectedReferenceAudioURL == nil ? "or choose a file" : "Selected reference clip")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                if state.selectedReferenceAudioURL != nil {
                    Button("Clear") {
                        state.clearReferenceAudio()
                    }
                }
                Button("Choose File…") {
                    isImportingFile = true
                }
            }
        }
        .padding(14)
        .frame(maxWidth: .infinity, minHeight: 86)
        .background(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .fill(isDropTargeted ? Color.accentTintLight : .surfaceRecessed)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .strokeBorder(
                    isDropTargeted ? Color.accentColor : Color.secondary.opacity(0.25),
                    style: StrokeStyle(lineWidth: 1.5, dash: [8, 6])
                )
        )
        .animation(.spring(response: 0.3, dampingFraction: 0.7), value: isDropTargeted)
        .dropDestination(for: URL.self) { items, _ in
            guard let url = items.first else { return false }
            return selectFile(url)
        } isTargeted: { isTargeted in
            isDropTargeted = isTargeted
        }
    }

    private func handleImport(_ result: Result<[URL], Error>) {
        switch result {
        case .success(let urls):
            if let url = urls.first {
                _ = selectFile(url)
            }
        case .failure(let error):
            state.errorMessage = PathRedaction.redactMessage(error.localizedDescription)
        }
    }

    @discardableResult
    private func selectFile(_ url: URL) -> Bool {
        do {
            try state.selectReferenceAudio(url)
            state.errorMessage = nil
            return true
        } catch {
            state.errorMessage = PathRedaction.redactMessage(error.localizedDescription)
            return false
        }
    }
}
