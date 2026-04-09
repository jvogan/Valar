import AVFoundation
import Foundation
import XCTest
@testable import ValarAudio

/// Unit tests for `StreamingWAVWriter`.
///
/// Covers initialization validation, incremental append (both raw-samples and
/// `AudioPCMBuffer` variants), finalization idempotence, and WAV header validity.
/// The E2E integration test in `EndToEndIntegrationTests` exercises the full
/// text→audio→persist pipeline; this suite targets the writer in isolation.
final class StreamingWAVWriterTests: XCTestCase {

    // MARK: - Helpers

    private func makeTemporaryURL(extension ext: String = "wav") -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension(ext)
    }

    private func riffHeader(of url: URL) throws -> (riff: String, wave: String) {
        let data = try Data(contentsOf: url)
        let riff = String(decoding: data.prefix(4), as: UTF8.self)
        let wave = String(decoding: data.dropFirst(8).prefix(4), as: UTF8.self)
        return (riff, wave)
    }

    // MARK: - Initializer validation

    func testInitThrowsForZeroChannelCount() {
        let url = makeTemporaryURL()
        defer { try? FileManager.default.removeItem(at: url) }
        XCTAssertThrowsError(
            try StreamingWAVWriter(url: url, sampleRate: 24_000, channelCount: 0),
            "Expected exportFailed for channelCount=0"
        )
    }

    func testInitThrowsForZeroSampleRate() {
        let url = makeTemporaryURL()
        defer { try? FileManager.default.removeItem(at: url) }
        XCTAssertThrowsError(
            try StreamingWAVWriter(url: url, sampleRate: 0, channelCount: 1),
            "Expected exportFailed for sampleRate=0"
        )
    }

    func testInitThrowsForNegativeSampleRate() {
        let url = makeTemporaryURL()
        defer { try? FileManager.default.removeItem(at: url) }
        XCTAssertThrowsError(
            try StreamingWAVWriter(url: url, sampleRate: -24_000, channelCount: 1),
            "Expected exportFailed for negative sampleRate"
        )
    }

    func testInitCreatesFileOnDisk() async throws {
        let url = makeTemporaryURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let writer = try StreamingWAVWriter(url: url, sampleRate: 24_000, channelCount: 1)
        _ = await writer.finalize()
        XCTAssertTrue(FileManager.default.fileExists(atPath: url.path))
    }

    func testInitOverwritesExistingFile() async throws {
        let url = makeTemporaryURL()
        defer { try? FileManager.default.removeItem(at: url) }

        // Write some arbitrary content to simulate a stale file.
        try Data("old-contents".utf8).write(to: url)

        let writer = try StreamingWAVWriter(url: url, sampleRate: 24_000, channelCount: 1)
        _ = await writer.finalize()

        let data = try Data(contentsOf: url)
        // The file must now be a WAV, not "old-contents".
        XCTAssertEqual(String(decoding: data.prefix(4), as: UTF8.self), "RIFF")
    }

    // MARK: - Append raw samples

    func testAppendEmptySamplesIsNoOp() async throws {
        let url = makeTemporaryURL()
        defer { try? FileManager.default.removeItem(at: url) }

        let writer = try StreamingWAVWriter(url: url, sampleRate: 24_000, channelCount: 1)
        try await writer.append([])
        let framesWritten = await writer.framesWritten
        XCTAssertEqual(framesWritten, 0)
    }

    func testAppendSingleChunkAccumulatesFrameCount() async throws {
        let url = makeTemporaryURL()
        defer { try? FileManager.default.removeItem(at: url) }

        let writer = try StreamingWAVWriter(url: url, sampleRate: 24_000, channelCount: 1)
        try await writer.append([Float](repeating: 0.5, count: 480))
        let framesWritten = await writer.framesWritten
        XCTAssertEqual(framesWritten, 480)
    }

    func testAppendMultipleChunksAccumulatesFrameCount() async throws {
        let url = makeTemporaryURL()
        defer { try? FileManager.default.removeItem(at: url) }

        let writer = try StreamingWAVWriter(url: url, sampleRate: 24_000, channelCount: 1)
        try await writer.append([Float](repeating: 0.1, count: 240))
        try await writer.append([Float](repeating: 0.2, count: 240))
        try await writer.append([Float](repeating: 0.3, count: 120))

        let framesWritten = await writer.framesWritten
        XCTAssertEqual(framesWritten, 600)
    }

    func testAppendProducesValidWAVHeader() async throws {
        let url = makeTemporaryURL()
        defer { try? FileManager.default.removeItem(at: url) }

        let writer = try StreamingWAVWriter(url: url, sampleRate: 24_000, channelCount: 1)
        try await writer.append([Float](repeating: 0.5, count: 1_000))
        _ = await writer.finalize()

        let header = try riffHeader(of: url)
        XCTAssertEqual(header.riff, "RIFF")
        XCTAssertEqual(header.wave, "WAVE")
    }

    func testAppendedAudioIsDecodable() async throws {
        let url = makeTemporaryURL()
        defer { try? FileManager.default.removeItem(at: url) }

        let sampleRate = 24_000.0
        let samples = (0..<2_400).map { i in Float(sin(Double(i) / sampleRate * 2 * .pi * 440)) }

        let writer = try StreamingWAVWriter(url: url, sampleRate: sampleRate, channelCount: 1)
        try await writer.append(samples)
        _ = await writer.finalize()

        let data = try Data(contentsOf: url)
        let decoder = AVFoundationAudioDecoder()
        let decoded = try await decoder.decode(data, hint: "wav")

        XCTAssertEqual(decoded.frameCount, samples.count)
        XCTAssertEqual(decoded.format.sampleRate, sampleRate)
        XCTAssertEqual(decoded.format.channelCount, 1)
    }

    // MARK: - Append AudioPCMBuffer variant

    func testAppendAudioPCMBufferAccumulatesFrameCount() async throws {
        let url = makeTemporaryURL()
        defer { try? FileManager.default.removeItem(at: url) }

        let writer = try StreamingWAVWriter(url: url, sampleRate: 24_000, channelCount: 1)
        let buffer = AudioPCMBuffer(mono: [Float](repeating: 0.3, count: 960), sampleRate: 24_000)
        try await writer.append(buffer)

        let framesWritten = await writer.framesWritten
        XCTAssertEqual(framesWritten, 960)
    }

    func testAppendEmptyAudioPCMBufferIsNoOp() async throws {
        let url = makeTemporaryURL()
        defer { try? FileManager.default.removeItem(at: url) }

        let writer = try StreamingWAVWriter(url: url, sampleRate: 24_000, channelCount: 1)
        let empty = AudioPCMBuffer(mono: [], sampleRate: 24_000)
        try await writer.append(empty)

        let framesWritten = await writer.framesWritten
        XCTAssertEqual(framesWritten, 0)
    }

    // MARK: - Finalization

    func testFinalizeReturnsCorrectFormat() async throws {
        let url = makeTemporaryURL()
        defer { try? FileManager.default.removeItem(at: url) }

        let writer = try StreamingWAVWriter(url: url, sampleRate: 48_000, channelCount: 2)
        let exported = await writer.finalize()

        XCTAssertEqual(exported.format.sampleRate, 48_000)
        XCTAssertEqual(exported.format.channelCount, 2)
        XCTAssertEqual(exported.format.container, "wav")
        XCTAssertEqual(exported.url, url)
    }

    func testFinalizeIsIdempotent() async throws {
        let url = makeTemporaryURL()
        defer { try? FileManager.default.removeItem(at: url) }

        let writer = try StreamingWAVWriter(url: url, sampleRate: 24_000, channelCount: 1)
        try await writer.append([0.1, 0.2, 0.3])
        let first = await writer.finalize()
        let second = await writer.finalize()

        XCTAssertEqual(first.url, second.url)
        XCTAssertEqual(first.format.sampleRate, second.format.sampleRate)
    }

    func testAppendAfterFinalizeThrows() async throws {
        let url = makeTemporaryURL()
        defer { try? FileManager.default.removeItem(at: url) }

        let writer = try StreamingWAVWriter(url: url, sampleRate: 24_000, channelCount: 1)
        _ = await writer.finalize()

        do {
            try await writer.append([0.5, 0.5])
            XCTFail("Expected exportFailed error when appending to finalized writer")
        } catch AudioPipelineError.exportFailed {
            // expected
        }
    }

    func testFormatPropertyMatchesInitParameters() async throws {
        let url = makeTemporaryURL()
        defer { try? FileManager.default.removeItem(at: url) }

        let writer = try StreamingWAVWriter(url: url, sampleRate: 16_000, channelCount: 2)
        let format = await writer.format

        XCTAssertEqual(format.sampleRate, 16_000)
        XCTAssertEqual(format.channelCount, 2)
        XCTAssertEqual(format.container, "wav")
    }

    // MARK: - Stereo deinterleaving

    func testStereoAppendDeinterleavesCorrectly() async throws {
        let url = makeTemporaryURL()
        defer { try? FileManager.default.removeItem(at: url) }

        // Interleaved stereo: [L0, R0, L1, R1, L2, R2]
        // L channel: [1, 2, 3], R channel: [10, 20, 30]
        let interleaved: [Float] = [1, 10, 2, 20, 3, 30]

        let writer = try StreamingWAVWriter(url: url, sampleRate: 44_100, channelCount: 2)
        try await writer.append(interleaved)
        _ = await writer.finalize()

        let framesWritten = await writer.framesWritten
        XCTAssertEqual(framesWritten, 3)

        // Decode and confirm correct frame count.
        let data = try Data(contentsOf: url)
        let decoder = AVFoundationAudioDecoder()
        let decoded = try await decoder.decode(data, hint: "wav")
        XCTAssertEqual(decoded.frameCount, 3)
        XCTAssertEqual(decoded.format.channelCount, 2)
    }

    // MARK: - Incomplete trailing frames

    func testIncompleteTrailingFramesAreSilentlyDropped() async throws {
        let url = makeTemporaryURL()
        defer { try? FileManager.default.removeItem(at: url) }

        // 5 samples for a stereo writer — 2 complete frames (4 samples) + 1 leftover
        let samples: [Float] = [1, 10, 2, 20, 3]

        let writer = try StreamingWAVWriter(url: url, sampleRate: 24_000, channelCount: 2)
        try await writer.append(samples)

        let framesWritten = await writer.framesWritten
        // Only 2 complete stereo frames (4 samples / 2 channels) should be written.
        XCTAssertEqual(framesWritten, 2)
    }
}
