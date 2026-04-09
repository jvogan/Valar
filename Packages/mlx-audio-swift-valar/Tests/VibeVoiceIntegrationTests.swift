import Foundation
import Testing
import MLX
import MLXLMCommon

@testable import MLXAudioTTS

@Suite("VibeVoice Integration", .serialized)
struct VibeVoiceIntegrationTests {

    @Test("Installed VibeVoice preset voice renders non-silent audio")
    func installedPresetVoiceRendersAudibleWaveform() async throws {
        guard mlxRuntimeReadyForCurrentProcess() else {
            return
        }
        let modelDir = URL(fileURLWithPath: NSHomeDirectory())
            .appendingPathComponent("Library/Application Support/ValarTTS/ModelPacks/vibevoice-realtime-tts/mlx-community-vibevoice-realtime-0-5b-4bit")

        guard FileManager.default.fileExists(atPath: modelDir.path) else {
            return
        }

        let voicesDir = modelDir.appendingPathComponent("voices")
        let voiceName = try FileManager.default.contentsOfDirectory(
            at: voicesDir,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        )
        .filter { $0.pathExtension == "safetensors" }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }
        .first?
        .deletingPathExtension()
        .lastPathComponent

        guard let voiceName else {
            return
        }

        let model = try await VibeVoiceTTSModel.fromDirectory(modelDir)
        let audio = try await model.generate(
            text: "Hello from VibeVoice. This is a stabilization smoke test.",
            voice: voiceName,
            refAudio: nil,
            refText: nil,
            language: nil,
            generationParameters: GenerateParameters(
                maxTokens: 24,
                temperature: 1.0,
                topP: 1.0,
                repetitionPenalty: 1.0
            )
        )

        MLX.eval(audio)

        let absolute = MLX.abs(audio)
        let peak = MLX.max(absolute).item(Float.self)
        let meanAbs = MLX.mean(absolute).item(Float.self)

        #expect(audio.shape[0] > 1_000)
        #expect(peak > 0.01)
        #expect(meanAbs > 0.0005)
    }
}
