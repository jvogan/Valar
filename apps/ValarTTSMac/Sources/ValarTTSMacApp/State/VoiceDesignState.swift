import SwiftUI
import ValarPersistence

@Observable
@MainActor
final class VoiceDesignState {
    var voiceLabel = ""
    var instruction = ""
    var previewScript = "Hello, this is a preview of the voice you designed for Valar."
    var isGeneratingPreview = false
    var hasPreview = false
    var previewStatus = "Describe a voice, then generate a preview."

    var canGeneratePreview: Bool {
        !normalizedInstruction.isEmpty && !normalizedPreviewScript.isEmpty && !isGeneratingPreview
    }

    var canAccept: Bool {
        !normalizedLabel.isEmpty
            && hasPreview
            && normalizedInstruction == lastPreviewedInstruction
            && !isGeneratingPreview
    }

    private let services: ValarServiceHub
    private var lastPreviewedInstruction = ""

    init(services: ValarServiceHub) {
        self.services = services
    }

    func generatePreview() async {
        guard canGeneratePreview else { return }

        isGeneratingPreview = true
        previewStatus = "Generating preview..."
        defer { isGeneratingPreview = false }

        do {
            await services.audioPlayer.stop()
            let previewVoice = VoiceLibraryRecord(
                label: normalizedLabel.isEmpty ? "Designed Voice" : normalizedLabel,
                modelID: ValarServiceHub.qwenVoiceDesignModelID.rawValue,
                voicePrompt: normalizedInstruction
            )
            let buffer = try await services.synthesizePreview(
                text: normalizedPreviewScript,
                modelID: ValarServiceHub.qwenVoiceDesignModelID,
                voiceRecord: previewVoice
            )
            guard buffer.frameCount > 0 else {
                hasPreview = false
                previewStatus = "Preview generation returned no audio."
                return
            }

            try await services.audioPlayer.play(buffer)
            hasPreview = true
            lastPreviewedInstruction = normalizedInstruction
            previewStatus = "Preview ready. Listen, revise the description, or save it."
        } catch {
            hasPreview = false
            previewStatus = "Preview failed: \(PathRedaction.redactMessage(error.localizedDescription))"
        }
    }

    private var normalizedLabel: String {
        voiceLabel.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private var normalizedInstruction: String {
        instruction.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private var normalizedPreviewScript: String {
        previewScript.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
