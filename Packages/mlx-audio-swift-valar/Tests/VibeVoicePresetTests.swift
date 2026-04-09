import Foundation
import Testing

@testable import MLXAudioTTS

/// Tests for VibeVoice preset voice loading, enumeration, and graceful error handling.
///
/// Tests that require Metal (model instantiation, MLXArray operations) are excluded
/// here — they run in integration testing with the metallib built. These tests cover
/// the metadata, catalog logic, and graceful filesystem-level loading behavior.
@Suite("VibeVoice Preset Voice Loading")
struct VibeVoicePresetTests {

    // MARK: - Preset voice catalog

    @Test("presetVoiceCount is 25")
    func presetVoiceCountIs25() {
        #expect(VibeVoiceTTSModel.presetVoiceCount == 25)
    }

    // MARK: - Graceful loading (filesystem-level, no Metal)

    // Note: "skips corrupt files" test requires Metal (MLX.loadArrays triggers
    // Metal init). Covered in integration tests with metallib built.

    @Test("loadPresetVoicesGracefully handles missing voices directory")
    func gracefulLoadMissingDir() {
        let config = VibeVoiceModelConfig()
        let bogusDir = URL(fileURLWithPath: "/nonexistent/model-\(UUID().uuidString)")

        let result = loadPresetVoicesGracefully(from: bogusDir, config: config)
        #expect(result.voices.isEmpty)
        #expect(result.skipped.isEmpty)
    }

    @Test("loadPresetVoicesGracefully handles empty voices directory")
    func gracefulLoadEmptyDir() throws {
        let config = VibeVoiceModelConfig()
        let tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("vv-preset-test-\(UUID().uuidString)")
        let voicesDir = tmpDir.appendingPathComponent("voices")
        try FileManager.default.createDirectory(at: voicesDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        let result = loadPresetVoicesGracefully(from: tmpDir, config: config)
        #expect(result.voices.isEmpty)
        #expect(result.skipped.isEmpty)
    }

    @Test("loadPresetVoicesGracefully skips non-safetensors files")
    func gracefulLoadIgnoresNonSafetensors() throws {
        let config = VibeVoiceModelConfig()
        let tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("vv-preset-test-\(UUID().uuidString)")
        let voicesDir = tmpDir.appendingPathComponent("voices")
        try FileManager.default.createDirectory(at: voicesDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        // Write a non-safetensors file — should be ignored entirely
        let txtURL = voicesDir.appendingPathComponent("readme.txt")
        try Data("readme".utf8).write(to: txtURL)

        let result = loadPresetVoicesGracefully(from: tmpDir, config: config)
        #expect(result.voices.isEmpty)
        #expect(result.skipped.isEmpty)  // Not skipped — just ignored
    }

    @Test("VibeVoiceBulkLoadResult totalEstimatedBytes is zero for empty set")
    func bulkLoadEmptyBytes() {
        let result = VibeVoiceBulkLoadResult(voices: [:], skipped: [])
        #expect(result.totalEstimatedBytes == 0)
    }

    // MARK: - VibeVoicePresetVoiceInfo struct

    @Test("VibeVoicePresetVoiceInfo correctly exposes fields")
    func presetVoiceInfoFields() {
        let info = VibeVoicePresetVoiceInfo(
            name: "en-Emma_woman",
            displayName: "Emma",
            languageCode: "en",
            isLoaded: false,
            estimatedBytes: nil
        )

        #expect(info.name == "en-Emma_woman")
        #expect(info.displayName == "Emma")
        #expect(info.languageCode == "en")
        #expect(!info.isLoaded)
        #expect(info.estimatedBytes == nil)
    }

    @Test("VibeVoicePresetVoiceInfo loaded state with bytes")
    func presetVoiceInfoLoaded() {
        let info = VibeVoicePresetVoiceInfo(
            name: "en-Carter_man",
            displayName: "Carter",
            languageCode: "en",
            isLoaded: true,
            estimatedBytes: 1_048_576
        )

        #expect(info.isLoaded)
        #expect(info.estimatedBytes == 1_048_576)
    }

    // MARK: - Voice cache error types

    @Test("VibeVoiceVoiceCacheError descriptions are informative")
    func cacheErrorDescriptions() {
        let dirErr = VibeVoiceVoiceCacheError.voiceDirectoryNotFound(
            URL(fileURLWithPath: "/test/voices")
        )
        #expect(dirErr.description.contains("/test/voices"))

        let fileErr = VibeVoiceVoiceCacheError.voiceFileNotFound(
            "en-Emma_woman", URL(fileURLWithPath: "/test/voices/en-Emma_woman.safetensors")
        )
        #expect(fileErr.description.contains("en-Emma_woman"))

        let tensorErr = VibeVoiceVoiceCacheError.missingTensor(
            "lm_hidden", voiceName: "en-Emma_woman"
        )
        #expect(tensorErr.description.contains("lm_hidden"))
        #expect(tensorErr.description.contains("en-Emma_woman"))
    }
}
