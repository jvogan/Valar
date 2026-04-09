import SwiftUI

struct VoiceDesignView: View {
    @Environment(AppState.self) private var appState
    @State private var designState: VoiceDesignState?
    @State private var isShowingDiscardConfirmation = false
    @State private var isSaving = false
    @State private var initialPreviewScript = ""
    @Environment(\.dismiss) private var dismiss

    let onSave: (String, String) -> Void

    var body: some View {
        Group {
            if let designState {
                content(designState)
                    .transition(.opacity)
            } else {
                VStack(alignment: .leading, spacing: 20) {
                    SkeletonLine(width: 160, height: 22)
                    SkeletonLine(width: 280, height: 12)
                    Divider()
                    SkeletonFormField(labelWidth: 80)
                    SkeletonFormField(labelWidth: 120)
                        .frame(height: 140)
                    SkeletonFormField(labelWidth: 100)
                        .frame(height: 100)
                    Spacer()
                }
                .transition(.opacity)
            }
        }
        .animation(.easeInOut(duration: 0.3), value: designState != nil)
        .task {
            if designState == nil {
                let state = VoiceDesignState(services: appState.services)
                designState = state
                initialPreviewScript = state.previewScript
            }
        }
        .confirmationDialog(
            "Discard unsaved changes?",
            isPresented: $isShowingDiscardConfirmation,
            titleVisibility: .visible
        ) {
            Button("Discard Changes", role: .destructive) {
                dismiss()
            }
            Button("Cancel", role: .cancel) {}
        }
        .padding(24)
        .frame(minWidth: 520, minHeight: 520)
    }

    private func content(_ state: VoiceDesignState) -> some View {
        @Bindable var s = state

        return VStack(alignment: .leading, spacing: 20) {
            Text("Design a Voice")
                .font(.title2.bold())
            Text("Describe the voice you want, generate a sample with Qwen, then save the accepted design as a reusable library voice.")
                .font(.callout)
                .foregroundStyle(.secondary)

            Divider()

            VStack(alignment: .leading, spacing: 8) {
                Text("Voice Label")
                    .font(.headline)
                TextField("Studio Narrator", text: $s.voiceLabel)
                    .textFieldStyle(.roundedBorder)
            }

            VStack(alignment: .leading, spacing: 8) {
                Text("Voice Description")
                    .font(.headline)
                TextEditor(text: $s.instruction)
                    .frame(minHeight: 120)
                    .padding(8)
                    .background(.surfaceRecessed)
                    .clipShape(RoundedRectangle(cornerRadius: ValarRadius.field, style: .continuous))
                Text("Qwen is the local design lane here. Example: warm female voice, British accent, mid-30s, calm but confident delivery.")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            }

            VStack(alignment: .leading, spacing: 8) {
                Text("Preview Script")
                    .font(.headline)
                TextEditor(text: $s.previewScript)
                    .frame(minHeight: 80)
                    .padding(8)
                    .background(.surfaceRecessed)
                    .clipShape(RoundedRectangle(cornerRadius: ValarRadius.field, style: .continuous))
            }

            VStack(alignment: .leading, spacing: 10) {
                Text(s.previewStatus)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .contentTransition(.interpolate)
                    .animation(.easeInOut(duration: 0.2), value: s.previewStatus)

                HStack(spacing: 8) {
                    Button {
                        Task { await state.generatePreview() }
                    } label: {
                        HStack(spacing: 6) {
                            if state.isGeneratingPreview {
                                ProgressView()
                                    .controlSize(.small)
                                    .transition(.scale.combined(with: .opacity))
                            }
                            Label(state.hasPreview ? "Preview Again" : "Generate Preview", systemImage: "play.circle.fill")
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(!state.canGeneratePreview)
                    .animation(.easeInOut(duration: 0.2), value: state.isGeneratingPreview)
                }
            }

            Spacer()

            HStack {
                Button("Cancel") {
                    if hasUnsavedChanges(state) {
                        isShowingDiscardConfirmation = true
                    } else {
                        dismiss()
                    }
                }
                    .keyboardShortcut(.cancelAction)
                Spacer()
                VStack(alignment: .trailing, spacing: 4) {
                    Button("Accept Voice") {
                        isSaving = true
                        Task {
                            onSave(state.voiceLabel.trimmingCharacters(in: .whitespacesAndNewlines),
                                   state.instruction.trimmingCharacters(in: .whitespacesAndNewlines))
                            isSaving = false
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(!state.canAccept || isSaving)
                    .help("Preview the current instruction first to enable Accept.")
                    .keyboardShortcut(.defaultAction)

                    if !state.canAccept {
                        Text("Generate a preview first to accept.")
                            .foregroundStyle(.secondary)
                            .font(.caption)
                    }
                }
            }
        }
    }

    private func hasUnsavedChanges(_ state: VoiceDesignState) -> Bool {
        !state.voiceLabel.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            || !state.instruction.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            || state.previewScript != initialPreviewScript
    }
}
