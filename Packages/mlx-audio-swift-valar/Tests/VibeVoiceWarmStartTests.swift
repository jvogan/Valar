import Foundation
import Testing
import MLX

@testable import MLXAudioTTS

@Suite("VibeVoice Warm-Start & Latency")
struct VibeVoiceWarmStartTests {

    // MARK: - LatencyBreakdown struct

    @Test("LatencyBreakdown computes RTF correctly")
    func latencyBreakdownRTF() {
        let breakdown = VibeVoiceTTSModel.LatencyBreakdown(
            voiceLoadSeconds: 0.020,
            textPrefillSeconds: 0.050,
            diffusionSamplingSeconds: 0.800,
            acousticDecodeSeconds: 0.030,
            firstLatentSeconds: 0.100,
            totalWallSeconds: 1.0,
            speechLatentCount: 50,
            audioDurationSeconds: 3.0,
            promptTokenCount: 20
        )

        // RTF = audio_duration / wall_time = 3.0 / 1.0 = 3.0
        #expect(breakdown.realtimeFactor == 3.0)
        // latents/sec = 50 / 1.0 = 50.0
        #expect(breakdown.latentsPerSecond == 50.0)
    }

    @Test("LatencyBreakdown handles zero wall time gracefully")
    func latencyBreakdownZeroWall() {
        let breakdown = VibeVoiceTTSModel.LatencyBreakdown(
            voiceLoadSeconds: 0,
            textPrefillSeconds: 0,
            diffusionSamplingSeconds: 0,
            acousticDecodeSeconds: 0,
            firstLatentSeconds: nil,
            totalWallSeconds: 0,
            speechLatentCount: 0,
            audioDurationSeconds: 0,
            promptTokenCount: 0
        )

        #expect(breakdown.realtimeFactor == 0)
        #expect(breakdown.latentsPerSecond == 0)
    }

    @Test("LatencyBreakdown first latent can be nil")
    func latencyBreakdownNilFirstLatent() {
        let breakdown = VibeVoiceTTSModel.LatencyBreakdown(
            voiceLoadSeconds: 0,
            textPrefillSeconds: 0,
            diffusionSamplingSeconds: 0,
            acousticDecodeSeconds: 0,
            firstLatentSeconds: nil,
            totalWallSeconds: 0.5,
            speechLatentCount: 0,
            audioDurationSeconds: 0,
            promptTokenCount: 10
        )

        #expect(breakdown.firstLatentSeconds == nil)
        #expect(breakdown.promptTokenCount == 10)
    }

    // MARK: - Voice pre-load cache

    @Test("Voice prewarm cache is initially empty")
    func prewarmedVoicesInitiallyEmpty() {
        let config = VibeVoiceModelConfig()
        let model = VibeVoiceTTSModel(config: config)
        #expect(model.prewarmedVoiceNames.isEmpty)
    }

    @Test("prewarmVoice with no model directory throws")
    func prewarmWithoutModelDir() {
        let config = VibeVoiceModelConfig()
        let model = VibeVoiceTTSModel(config: config)

        #expect(throws: (any Error).self) {
            try model.prewarmVoice("en-Emma_woman")
        }
    }

    @Test("prewarmDefaultVoice returns nil when no model directory")
    func prewarmDefaultNoDir() throws {
        let config = VibeVoiceModelConfig()
        let model = VibeVoiceTTSModel(config: config)

        let result = try model.prewarmDefaultVoice()
        #expect(result == nil)
    }

    @Test("prewarmDefaultVoice returns nil when voices directory is missing")
    func prewarmDefaultMissingVoicesDir() throws {
        let config = VibeVoiceModelConfig()
        let model = VibeVoiceTTSModel(config: config)
        let tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("vv-test-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        model.modelDirectory = tmpDir
        let result = try model.prewarmDefaultVoice()
        #expect(result == nil)
    }

    @Test("prewarmDefaultVoice returns nil for empty voices directory")
    func prewarmDefaultEmptyVoicesDir() throws {
        let config = VibeVoiceModelConfig()
        let model = VibeVoiceTTSModel(config: config)
        let tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("vv-test-\(UUID().uuidString)")
        let voicesDir = tmpDir.appendingPathComponent("voices")
        try FileManager.default.createDirectory(at: voicesDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        model.modelDirectory = tmpDir
        let result = try model.prewarmDefaultVoice()
        #expect(result == nil)
    }

    @Test("evictPrewarmedVoices clears cache")
    func evictClearsCache() {
        let config = VibeVoiceModelConfig()
        let model = VibeVoiceTTSModel(config: config)

        // Manually inject a dummy snapshot to verify eviction
        let dummyArray = MLXArray.zeros([1, 2, 10, 64])
        let dummyHidden = MLXArray.zeros([1, 10, 896])
        let pair: (key: MLXArray, value: MLXArray) = (key: dummyArray, value: dummyArray)
        let snapshot = VibeVoiceKVSnapshot(
            lmHidden: dummyHidden,
            lmCache: Array(repeating: pair, count: 4),
            ttsLmHidden: dummyHidden,
            ttsLmCache: Array(repeating: pair, count: 20),
            negTtsLmHidden: dummyHidden,
            negTtsLmCache: Array(repeating: pair, count: 20),
            negLmCache: nil
        )

        // Force-set via the model's internal API
        model.prewarmedVoices["test_voice"] = snapshot
        #expect(model.prewarmedVoiceNames == ["test_voice"])

        model.evictPrewarmedVoices()
        #expect(model.prewarmedVoiceNames.isEmpty)
    }

    @Test("lastLatencyBreakdown is nil before generation")
    func lastBreakdownInitiallyNil() {
        let config = VibeVoiceModelConfig()
        let model = VibeVoiceTTSModel(config: config)
        #expect(model.lastLatencyBreakdown == nil)
    }
}
