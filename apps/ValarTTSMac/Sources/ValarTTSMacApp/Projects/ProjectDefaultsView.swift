import SwiftUI
import ValarModelKit

struct ProjectDefaultsView: View {
    @Bindable var state: ProjectWorkspaceState

    private let languages = ["auto", "en", "zh", "ja", "ko", "fr", "de", "es"]
    private let voiceBehaviors = SpeechSynthesisVoiceBehavior.allCases

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("Project Defaults", systemImage: "gearshape")
                .font(.headline)

            VStack(alignment: .leading, spacing: 10) {
                if state.availableRenderModels.isEmpty {
                    ContentUnavailableView(
                        "No TTS models installed",
                        systemImage: "waveform.badge.xmark",
                        description: Text("Install a speech synthesis model before rendering a project.")
                    )
                } else {
                    Picker(
                        "Render model",
                        selection: Binding(
                            get: { state.selectedRenderModelID?.rawValue ?? "" },
                            set: { state.selectedRenderModelID = $0.isEmpty ? nil : ModelIdentifier($0) }
                        )
                    ) {
                        ForEach(state.availableRenderModels) { model in
                            Text(model.displayName).tag(model.id.rawValue)
                        }
                    }
                    .pickerStyle(.menu)

                    Button {
                        Task { await state.startRender() }
                    } label: {
                        Label("Render Project", systemImage: "sparkles.rectangle.stack")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(!state.canStartRender)

                    if let progress = state.overallRenderProgress {
                        VStack(alignment: .leading, spacing: 6) {
                            ProgressView(value: progress)
                                .progressViewStyle(.linear)
                            Text(state.renderSummary)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }
                }

                Picker("Language", selection: $state.language) {
                    ForEach(languages, id: \.self) { lang in
                        Text(lang == "auto" ? "Auto-detect" : lang.uppercased()).tag(lang)
                    }
                }
                .pickerStyle(.menu)

                Picker("Voice behavior", selection: $state.voiceBehavior) {
                    ForEach(voiceBehaviors, id: \.self) { behavior in
                        Text(label(for: behavior)).tag(behavior)
                    }
                }
                .pickerStyle(.menu)

                VStack(alignment: .leading, spacing: 8) {
                    LabeledContent("Temperature") {
                        Text(state.temperature, format: .number.precision(.fractionLength(2)))
                            .monospacedDigit()
                    }
                    Slider(value: $state.temperature, in: 0...1.5, step: 0.05)
                }

                VStack(alignment: .leading, spacing: 8) {
                    LabeledContent("Top-p") {
                        Text(state.topP, format: .number.precision(.fractionLength(2)))
                            .monospacedDigit()
                    }
                    Slider(value: $state.topP, in: 0.1...1, step: 0.05)
                }

                VStack(alignment: .leading, spacing: 8) {
                    LabeledContent("Repetition penalty") {
                        Text(state.repetitionPenalty, format: .number.precision(.fractionLength(2)))
                            .monospacedDigit()
                    }
                    Slider(value: $state.repetitionPenalty, in: 1...2, step: 0.05)
                }

                Stepper(value: $state.maxTokens, in: 256...8192, step: 256) {
                    LabeledContent("Max tokens") {
                        Text("\(state.maxTokens)")
                            .monospacedDigit()
                    }
                }
            }
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
