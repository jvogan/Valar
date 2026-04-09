import AVFoundation
import Foundation
import XCTest
@testable import ValarAudio

/// End-to-end integration tests: text → audio → verify → save.
///
/// These tests exercise the complete audio pipeline from a text input through
/// audio generation (stub synthesis), disk persistence, and decoded verification.
/// Real MLX inference is not exercised here — the synthesis step is replaced by a
/// deterministic tone generator that produces a valid mono PCM buffer. This keeps
/// the tests hermetic and runnable without a downloaded model while still exercising
/// every downstream stage of the hosting pipeline:
///
///   1. Text input — a known string drives the generation parameters.
///   2. Audio generation — a sine-wave tone stands in for model output.
///   3. Streaming write — chunks are fed through `StreamingWAVWriter`.
///   4. Verify — the saved file is decoded and its properties are asserted.
///   5. Save — the exported file's metadata and on-disk presence are confirmed.
final class EndToEndIntegrationTests: XCTestCase {

    // MARK: - Main pipeline

    /// Drives the full audio pipeline from a text constant through streaming WAV
    /// generation, disk persistence, and decoded format verification.
    ///
    /// This is the primary gate for W8-T7: every stage of the pipeline must
    /// succeed end-to-end before the test passes.
    func testTextToAudioToVerifyToSave() async throws {
        // 1. Text input — the driver for audio generation.
        let inputText = "Hello from ValarTTS."

        // 2. Stub synthesis — generate a deterministic mono tone that stands in
        //    for real TTS model output.  The frequency is derived from the first
        //    character of the input text so the audio is reproducibly linked to
        //    the specific string without requiring a real model.
        let sampleRate = 24_000.0
        let durationSeconds = 0.5
        let frequency = 440.0 + Double(inputText.unicodeScalars.first?.value ?? 0)
        let samples = Self.generateTone(
            frequency: frequency,
            duration: durationSeconds,
            sampleRate: sampleRate
        )

        // 3. Streaming write — feed samples to StreamingWAVWriter in 20 ms chunks,
        //    mirroring the real streaming TTS path where audio arrives incrementally.
        let outputURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("wav")
        defer { try? FileManager.default.removeItem(at: outputURL) }

        let chunkFrames = 480  // 20 ms at 24 kHz — canonical streaming chunk size
        let writer = try StreamingWAVWriter(url: outputURL, sampleRate: sampleRate, channelCount: 1)
        var offset = 0
        while offset < samples.count {
            let end = min(offset + chunkFrames, samples.count)
            try await writer.append(Array(samples[offset..<end]))
            offset = end
        }
        let exportedFile = await writer.finalize()

        // 4. Verify file exists and has a valid WAV header.
        XCTAssertTrue(
            FileManager.default.fileExists(atPath: exportedFile.url.path),
            "Exported WAV file must exist on disk"
        )
        let savedData = try Data(contentsOf: exportedFile.url)
        XCTAssertFalse(savedData.isEmpty, "Exported WAV file must not be empty")
        let riffMarker = String(decoding: savedData.prefix(4), as: UTF8.self)
        let waveMarker = String(decoding: savedData.dropFirst(8).prefix(4), as: UTF8.self)
        XCTAssertEqual(riffMarker, "RIFF", "WAV file must begin with RIFF header")
        XCTAssertEqual(waveMarker, "WAVE", "WAV file must contain WAVE marker")

        // 5. Decode and verify audio properties.
        let decoder = AVFoundationAudioDecoder()
        let decoded = try await decoder.decode(savedData, hint: "wav")
        let expectedFrameCount = Int(durationSeconds * sampleRate)
        XCTAssertGreaterThan(decoded.frameCount, 0, "Decoded audio must have at least one frame")
        XCTAssertEqual(
            decoded.frameCount, expectedFrameCount,
            "Decoded frame count must exactly match the synthesized duration"
        )
        XCTAssertEqual(
            decoded.format.sampleRate, sampleRate,
            "Decoded sample rate must match the export rate"
        )
        XCTAssertEqual(decoded.format.channelCount, 1, "Decoded audio must be mono")

        // 6. Confirm exported file metadata matches the write parameters.
        XCTAssertEqual(exportedFile.format.sampleRate, sampleRate)
        XCTAssertEqual(exportedFile.format.channelCount, 1)
        XCTAssertEqual(exportedFile.format.container, "wav")
        let framesWritten = await writer.framesWritten
        XCTAssertEqual(
            framesWritten, expectedFrameCount,
            "Writer frame count must equal the number of samples appended"
        )
    }

    // MARK: - Reproducibility

    /// Confirms the pipeline produces consistent WAV output for the same input.
    ///
    /// Runs the full pipeline twice and asserts the exported frame counts match,
    /// confirming stub synthesis is deterministic and the writer is stable.
    func testPipelineIsReproducibleForFixedText() async throws {
        let inputText = "Reproducibility check."
        let sampleRate = 24_000.0
        let duration = 0.25
        let samples = Self.generateTone(frequency: 440, duration: duration, sampleRate: sampleRate)

        func runPipeline() async throws -> Int {
            let url = FileManager.default.temporaryDirectory
                .appendingPathComponent(UUID().uuidString)
                .appendingPathExtension("wav")
            defer { try? FileManager.default.removeItem(at: url) }
            let writer = try StreamingWAVWriter(url: url, sampleRate: sampleRate, channelCount: 1)
            try await writer.append(samples)
            let exported = await writer.finalize()
            let data = try Data(contentsOf: exported.url)
            let decoder = AVFoundationAudioDecoder()
            let buffer = try await decoder.decode(data, hint: "wav")
            return buffer.frameCount
        }

        let first = try await runPipeline()
        let second = try await runPipeline()
        _ = inputText  // surfaced in test name / failure messages
        XCTAssertEqual(first, second, "Pipeline must produce identical frame counts for the same input")
        XCTAssertGreaterThan(first, 0, "Frame count must be positive")
    }

    // MARK: - Multi-segment render

    /// Exercises the pipeline with multiple sequential text segments, simulating a
    /// multi-chapter render where each segment produces its own WAV file.
    func testMultiSegmentPipelineSavesEachSegment() async throws {
        let segments = [
            "Chapter one.",
            "Chapter two.",
            "Chapter three.",
        ]
        let sampleRate = 24_000.0
        let duration = 0.1
        var savedURLs: [URL] = []
        defer { savedURLs.forEach { try? FileManager.default.removeItem(at: $0) } }

        for (index, segment) in segments.enumerated() {
            let frequency = 220.0 + Double(index) * 110
            let samples = Self.generateTone(frequency: frequency, duration: duration, sampleRate: sampleRate)
            let url = FileManager.default.temporaryDirectory
                .appendingPathComponent("segment-\(index)-\(UUID().uuidString)")
                .appendingPathExtension("wav")
            savedURLs.append(url)

            let writer = try StreamingWAVWriter(url: url, sampleRate: sampleRate, channelCount: 1)
            try await writer.append(samples)
            _ = await writer.finalize()

            XCTAssertTrue(
                FileManager.default.fileExists(atPath: url.path),
                "Segment \(index) (\(segment)) must be saved to disk"
            )
        }

        XCTAssertEqual(savedURLs.count, segments.count, "One output file per text segment")
    }

    // MARK: - Waveform analysis

    /// Validates waveform analysis on generated audio, confirming peak/RMS properties.
    func testWaveformAnalysisOnGeneratedAudio() async throws {
        let sampleRate = 24_000.0
        let duration = 0.5
        let samples = Self.generateTone(frequency: 440, duration: duration, sampleRate: sampleRate)
        let buffer = AudioPCMBuffer(mono: samples, sampleRate: sampleRate)
        let pipeline = AudioPipeline()
        let waveform = await pipeline.waveform(for: buffer, bucketCount: 16)

        XCTAssertEqual(waveform.frameCount, samples.count)
        XCTAssertGreaterThan(waveform.peak, 0, "Non-silent audio must have positive peak")
        XCTAssertGreaterThan(waveform.rms, 0, "Non-silent audio must have positive RMS")
        XCTAssertLessThanOrEqual(waveform.peak, 1.0, "Peak must not exceed unity")
        XCTAssertEqual(waveform.bucketCount, 16)
    }

    // MARK: - Export via AudioExporter (non-streaming path)

    /// Verifies the non-streaming export path using `AVFoundationAudioExporter`.
    func testNonStreamingExportProducesValidWAV() async throws {
        let sampleRate = 24_000.0
        let samples = Self.generateTone(frequency: 523.25, duration: 0.3, sampleRate: sampleRate)
        let buffer = AudioPCMBuffer(mono: samples, sampleRate: sampleRate)

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("wav")
        defer { try? FileManager.default.removeItem(at: url) }

        let exporter = AVFoundationAudioExporter()
        let format = AudioFormatDescriptor(sampleRate: sampleRate, channelCount: 1, container: "wav")
        let exported = try await exporter.export(buffer, as: format, to: url, chapterMarkers: [])

        XCTAssertTrue(
            FileManager.default.fileExists(atPath: exported.url.path),
            "Non-streaming export must produce a file on disk"
        )
        let data = try Data(contentsOf: exported.url)
        let riffMarker = String(decoding: data.prefix(4), as: UTF8.self)
        XCTAssertEqual(riffMarker, "RIFF", "Non-streaming export must write a valid RIFF header")

        let decoder = AVFoundationAudioDecoder()
        let decoded = try await decoder.decode(data, hint: "wav")
        XCTAssertEqual(decoded.format.sampleRate, sampleRate)
        XCTAssertGreaterThan(decoded.frameCount, 0)
    }

    // MARK: - Playback (best-effort: skipped when no audio hardware is available)

    /// Confirms that generated audio can be fed to `AudioEnginePlayer` without error.
    /// Skipped on CI hosts without audio output hardware.
    func testPlaybackOfGeneratedAudio() async throws {
        let sampleRate = 24_000.0
        let samples = Self.generateTone(frequency: 440, duration: 0.1, sampleRate: sampleRate)
        let player = AudioEnginePlayer()

        do {
            try await player.feedSamples(samples, sampleRate: sampleRate)
        } catch {
            throw XCTSkip(
                "AVAudioEngine could not start (no audio hardware?): \(error.localizedDescription)"
            )
        }

        let snapshot = await player.playbackSnapshot()
        XCTAssertGreaterThan(
            snapshot.position + snapshot.queuedDuration, 0,
            "Player must have queued at least one frame after feedSamples"
        )
        await player.stop()
    }

    // MARK: - Helpers

    /// Generates a mono sine wave at `frequency` Hz for `duration` seconds.
    private static func generateTone(frequency: Double, duration: Double, sampleRate: Double) -> [Float] {
        let frameCount = Int(duration * sampleRate)
        return (0..<frameCount).map { frame in
            Float(sin(2.0 * Double.pi * frequency * Double(frame) / sampleRate))
        }
    }
}
