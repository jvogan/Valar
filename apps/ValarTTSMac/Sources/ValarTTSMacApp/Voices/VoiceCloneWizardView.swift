import SwiftUI
import UniformTypeIdentifiers
import ValarModelKit

struct VoiceCloneWizardView: View {
    @State private var voiceLabel = ""
    @State private var referenceTranscript = ""
    @State private var selectedFileURL: URL?
    @State private var selectedModelID: ModelIdentifier?
    @State private var errorMessage: String?
    @State private var isImportingFile = false
    @State private var isSaving = false
    @State private var isDropTargeted = false
    @Environment(\.dismiss) private var dismiss

    var prefilledAudioURL: URL? = nil
    let availableModels: [RuntimeModelPickerOption]
    let onSave: @MainActor (VoiceCloneDraft) async throws -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            Text("Save a Cloned Voice")
                .font(.title2.bold())
            Text("Drop a WAV or M4A reference clip, choose the voice-cloning model, then provide the exact transcript so Valar can materialize a reusable saved voice.")
                .font(.callout)
                .foregroundStyle(.secondary)

            Divider()

            VStack(alignment: .leading, spacing: 8) {
                Text("Voice Label")
                    .font(.headline)
                TextField("My Custom Voice", text: $voiceLabel)
                    .textFieldStyle(.roundedBorder)
            }

            VStack(alignment: .leading, spacing: 10) {
                Text("Reference Audio")
                    .font(.headline)
                dropZone
                Text("Requirements: 5-30 seconds, sample rate 16 kHz or higher, mono preferred. Stereo clips will be downmixed before cloning.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            VStack(alignment: .leading, spacing: 8) {
                Text("Clone Model")
                    .font(.headline)
                if availableModels.isEmpty {
                    Text("Install a clone-capable model first. TADA is the strongest multilingual cloning lane; Qwen remains the fallback saved-voice lane.")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                } else {
                    Picker(
                        "Clone Model",
                        selection: Binding(
                            get: { (selectedModelID ?? defaultCloneModelID(in: availableModels))! },
                            set: { selectedModelID = $0 }
                        )
                    ) {
                        ForEach(availableModels) { model in
                            Text(model.displayName).tag(model.id)
                        }
                    }
                    .pickerStyle(.menu)

                    if let selectedOption = selectedModelOption {
                        Text(selectedOption.familyID == .tadaTTS
                            ? "TADA is the strongest multilingual saved-voice path. Valar stores a reusable local conditioning bundle for the saved voice."
                            : "Qwen stores a reusable local cloned voice from the reference clip and transcript.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            }

            VStack(alignment: .leading, spacing: 8) {
                Text("Reference Transcript")
                    .font(.headline)
                TextEditor(text: $referenceTranscript)
                    .frame(minHeight: 96)
                    .padding(8)
                    .background(.surfaceRecessed)
                    .clipShape(RoundedRectangle(cornerRadius: ValarRadius.field, style: .continuous))
                Text("Use the exact spoken text from the clip. Valar currently requires the transcript when saving a cloned voice.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            if let errorMessage {
                Label(errorMessage, systemImage: "exclamationmark.triangle.fill")
                    .font(.callout)
                    .foregroundStyle(StatusColor.error)
            }

            Spacer()

            HStack {
                Button("Cancel") { dismiss() }
                    .keyboardShortcut(.cancelAction)
                Spacer()
                Button {
                    saveVoice()
                } label: {
                    HStack(spacing: 6) {
                        if isSaving {
                            ProgressView()
                                .controlSize(.small)
                                .transition(.scale.combined(with: .opacity))
                        }
                        Text(isSaving ? "Saving…" : "Save Voice")
                            .contentTransition(.interpolate)
                    }
                }
                .buttonStyle(.borderedProminent)
                .disabled(!canSave)
                .keyboardShortcut(.defaultAction)
                .animation(.easeInOut(duration: 0.2), value: isSaving)
            }
        }
        .padding(24)
        .frame(minWidth: 520, minHeight: 500)
        .onAppear {
            if let prefilledAudioURL {
                _ = selectFile(prefilledAudioURL)
            }
            selectedModelID = defaultCloneModelID(in: availableModels)
        }
        .fileImporter(
            isPresented: $isImportingFile,
            allowedContentTypes: [.wav, UTType(filenameExtension: "m4a")!],
            allowsMultipleSelection: false
        ) { result in
            handleImport(result)
        }
    }

    private var canSave: Bool {
        !voiceLabel.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && selectedFileURL != nil
            && !referenceTranscript.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && selectedModelOption != nil
            && !isSaving
    }

    private var selectedModelOption: RuntimeModelPickerOption? {
        let preferredID = selectedModelID ?? defaultCloneModelID(in: availableModels)
        return availableModels.first(where: { $0.id == preferredID })
    }

    private func defaultCloneModelID(in models: [RuntimeModelPickerOption]) -> ModelIdentifier? {
        models.first(where: { $0.familyID == .tadaTTS })?.id ?? models.first?.id
    }

    private var dropZone: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .center, spacing: 12) {
                Image(systemName: selectedFileURL == nil ? "waveform.badge.plus" : "waveform")
                    .font(.title2)
                    .foregroundStyle(isDropTargeted ? Color.accentColor : .secondary)
                    .symbolEffect(.bounce, value: isDropTargeted)
                VStack(alignment: .leading, spacing: 4) {
                    Text(selectedFileURL?.lastPathComponent ?? "Drop a WAV or M4A clip here")
                        .font(.body.weight(.medium))
                    Text(selectedFileURL == nil ? "or use the file picker" : "Selected reference audio")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Button("Choose File…") {
                    isImportingFile = true
                }
            }
        }
        .padding(16)
        .frame(maxWidth: .infinity, minHeight: 92)
        .background(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .fill(isDropTargeted ? Color.accentTintLight : .surfaceRecessed)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .strokeBorder(isDropTargeted ? Color.accentColor : Color.secondary.opacity(0.25), style: StrokeStyle(lineWidth: 1.5, dash: [8, 6]))
        )
        .scaleEffect(isDropTargeted ? 1.01 : 1)
        .animation(.spring(response: 0.3, dampingFraction: 0.7), value: isDropTargeted)
        .dropDestination(for: URL.self) { items, _ in
            guard let url = items.first else { return false }
            return selectFile(url)
        } isTargeted: { isTargeted in
            isDropTargeted = isTargeted
        }
    }

    private func saveVoice() {
        guard let selectedFileURL else { return }
        errorMessage = nil
        isSaving = true

        let draft = VoiceCloneDraft(
            label: voiceLabel,
            referenceAudioURL: selectedFileURL,
            referenceTranscript: referenceTranscript,
            modelID: selectedModelOption?.id
        )

        Task {
            do {
                try await onSave(draft)
                dismiss()
            } catch {
                errorMessage = error.localizedDescription
                isSaving = false
            }
        }
    }

    private func handleImport(_ result: Result<[URL], Error>) {
        switch result {
        case .success(let urls):
            if let url = urls.first {
                _ = selectFile(url)
            }
        case .failure(let error):
            errorMessage = error.localizedDescription
        }
    }

    @discardableResult
    private func selectFile(_ url: URL) -> Bool {
        do {
            try VoiceCloneFileValidator.validateFileSelection(url)
        } catch {
            errorMessage = error.localizedDescription
            return false
        }

        selectedFileURL = url
        errorMessage = nil
        return true
    }
}
