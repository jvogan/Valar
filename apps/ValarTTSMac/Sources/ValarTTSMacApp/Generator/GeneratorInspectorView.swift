import SwiftUI
import ValarModelKit

struct GeneratorInspectorView: View {
    @Bindable var state: GeneratorState
    @State private var pendingDraftRestore: GeneratorState.DraftEntry?

    var body: some View {
        Form {
            Section("Generation") {
                VStack(alignment: .leading, spacing: 8) {
                    LabeledContent("Temperature") {
                        Text(state.temperature, format: .number.precision(.fractionLength(2)))
                            .monospacedDigit()
                    }
                    Slider(value: $state.temperature, in: 0...2, step: 0.05)
                }

                VStack(alignment: .leading, spacing: 8) {
                    LabeledContent("Top P") {
                        Text(state.topP, format: .number.precision(.fractionLength(2)))
                            .monospacedDigit()
                    }
                    Slider(value: $state.topP, in: 0.05...1, step: 0.01)
                }

                Picker("Voice behavior", selection: $state.voiceBehavior) {
                    ForEach(SpeechSynthesisVoiceBehavior.allCases, id: \.self) { behavior in
                        Text(label(for: behavior)).tag(behavior)
                    }
                }
                .pickerStyle(.menu)

                VStack(alignment: .leading, spacing: 8) {
                    LabeledContent("Repetition Penalty") {
                        Text(state.repetitionPenalty, format: .number.precision(.fractionLength(2)))
                            .monospacedDigit()
                    }
                    Slider(value: $state.repetitionPenalty, in: 1...2, step: 0.05)
                }

                Stepper(value: $state.maxTokens, in: 256...8192, step: 256) {
                    LabeledContent("Max Tokens") {
                        Text("\(state.maxTokens)")
                            .monospacedDigit()
                    }
                }
            }

            Section("History") {
                if state.drafts.isEmpty {
                    Text("No drafts yet")
                        .foregroundStyle(.secondary)
                } else {
                    ForEach(state.drafts) { draft in
                        DraftHistoryRow(draft: draft) {
                            pendingDraftRestore = draft
                        }
                    }
                }
            }
        }
        .formStyle(.grouped)
        .confirmationDialog(
            "Restore Draft?",
            isPresented: Binding(
                get: { pendingDraftRestore != nil },
                set: { isPresented in
                    if !isPresented {
                        pendingDraftRestore = nil
                    }
                }
            ),
            titleVisibility: .visible
        ) {
            Button("Restore Draft") {
                guard let pendingDraftRestore else { return }
                state.restoreDraft(pendingDraftRestore)
                self.pendingDraftRestore = nil
            }
            Button("Cancel", role: .cancel) {
                pendingDraftRestore = nil
            }
        } message: {
            Text("This will replace your current text with the saved draft.")
        }
    }

    private func label(for behavior: SpeechSynthesisVoiceBehavior) -> String {
        switch behavior {
        case .auto:
            return "Auto"
        case .expressive:
            return "Expressive"
        case .stableNarrator:
            return "Stable Narrator"
        }
    }
}
