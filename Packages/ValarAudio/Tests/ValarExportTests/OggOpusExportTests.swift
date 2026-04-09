import Foundation
import SwiftOGG
import XCTest
@testable import ValarAudio
@testable import ValarExport

/// Tests for `ChannelAudioExporter` — OGG/Opus channel export.
///
/// Each test operates entirely in memory: no audio hardware, AVAudioEngine,
/// CADisplayLink, or AVAssetWriter are involved, so these are safe to run
/// in headless / CI environments.
final class OggOpusExportTests: XCTestCase {

    // MARK: - Encoding: mono

    func testEncodeMono24kHzProducesOGGBitstream() async throws {
        let samples = sineWave(frequency: 440, sampleRate: 24_000, duration: 0.5)
        let buffer = AudioPCMBuffer(mono: samples, sampleRate: 24_000)

        let exporter = ChannelAudioExporter()
        let oggData = try await exporter.encode(buffer)

        XCTAssertFalse(oggData.isEmpty)
        assertOGGMagic(oggData)
    }

    func testEncodeMono48kHzProducesOGGBitstream() async throws {
        let samples = sineWave(frequency: 440, sampleRate: 48_000, duration: 0.25)
        let buffer = AudioPCMBuffer(mono: samples, sampleRate: 48_000)

        let exporter = ChannelAudioExporter()
        let oggData = try await exporter.encode(buffer)

        XCTAssertFalse(oggData.isEmpty)
        assertOGGMagic(oggData)
    }

    func testEncodeEmptyBufferReturnsHeaders() async throws {
        let buffer = AudioPCMBuffer(mono: [], sampleRate: 24_000)

        let exporter = ChannelAudioExporter()
        let oggData = try await exporter.encode(buffer)

        // Even an empty encode produces OGG headers from OGGEncoder.init.
        assertOGGMagic(oggData)
    }

    // MARK: - Encoding: stereo

    func testEncodeStereo24kHzProducesOGGBitstream() async throws {
        let ch0 = sineWave(frequency: 440, sampleRate: 24_000, duration: 0.5)
        let ch1 = sineWave(frequency: 880, sampleRate: 24_000, duration: 0.5)
        let format = AudioFormatDescriptor(sampleRate: 24_000, channelCount: 2)
        let buffer = AudioPCMBuffer(channels: [ch0, ch1], format: format)

        let exporter = ChannelAudioExporter()
        let oggData = try await exporter.encode(buffer)

        XCTAssertFalse(oggData.isEmpty)
        assertOGGMagic(oggData)
    }

    // MARK: - Input clipping

    func testClippedSamplesEncodeWithoutError() async throws {
        // Samples well outside [-1, 1] — must not produce Int16 overflow.
        let samples = [Float](repeating: 5.0, count: 2_400)
        let buffer = AudioPCMBuffer(mono: samples, sampleRate: 24_000)

        let exporter = ChannelAudioExporter()
        let oggData = try await exporter.encode(buffer)

        XCTAssertFalse(oggData.isEmpty)
        assertOGGMagic(oggData)
    }

    // MARK: - Validation errors

    func testEncodeThrowsForUnsupportedSampleRate() async throws {
        let buffer = AudioPCMBuffer(mono: [0.0], sampleRate: 22_050)

        let exporter = ChannelAudioExporter()
        do {
            _ = try await exporter.encode(buffer)
            XCTFail("Expected ExportError.unsupportedSampleRate")
        } catch ChannelAudioExporter.ExportError.unsupportedSampleRate(let rate) {
            XCTAssertEqual(rate, 22_050)
        }
    }

    func testEncodeThrowsForUnsupportedChannelCount() async throws {
        let format = AudioFormatDescriptor(sampleRate: 24_000, channelCount: 3)
        let buffer = AudioPCMBuffer(channels: [[0.0], [0.0], [0.0]], format: format)

        let exporter = ChannelAudioExporter()
        do {
            _ = try await exporter.encode(buffer)
            XCTFail("Expected ExportError.unsupportedChannelCount")
        } catch ChannelAudioExporter.ExportError.unsupportedChannelCount(let count) {
            XCTAssertEqual(count, 3)
        }
    }

    // MARK: - Round-trip (encode → decode)

    func testRoundTripMono24kHzIsDecodable() async throws {
        let samples = sineWave(frequency: 440, sampleRate: 24_000, duration: 0.5)
        let buffer = AudioPCMBuffer(mono: samples, sampleRate: 24_000)

        let exporter = ChannelAudioExporter()
        let oggData = try await exporter.encode(buffer)

        // OGGDecoder decodes synchronously at 48 kHz; pcmData stores raw Float32.
        // For 0.5 s at 24 kHz → 24 000 Float32 samples at 48 kHz → 96 000 bytes.
        let decoder = try OGGDecoder(audioData: oggData)
        XCTAssertGreaterThan(decoder.pcmData.count, 0, "Decoded PCM should be non-empty")

        let decodedSamples = decoder.pcmData.count / MemoryLayout<Float>.size
        // Allow loose tolerance: Opus latency/preskip may trim a few frames.
        let expectedMinSamples = Int(24_000 * 0.5 * 0.8)
        XCTAssertGreaterThan(decodedSamples, expectedMinSamples,
            "Expected at least \(expectedMinSamples) decoded samples, got \(decodedSamples)")
    }

    // MARK: - Non-frame-aligned input

    func testNonFrameAlignedInputEncodesCorrectly() async throws {
        // 501 samples — not a multiple of the 480-sample frame size at 24 kHz.
        let samples = sineWave(frequency: 440, sampleRate: 24_000, duration: 501.0 / 24_000)
        let buffer = AudioPCMBuffer(mono: samples, sampleRate: 24_000)

        let exporter = ChannelAudioExporter()
        let oggData = try await exporter.encode(buffer)

        XCTAssertFalse(oggData.isEmpty)
        assertOGGMagic(oggData)
    }

    // MARK: - Helpers

    private func sineWave(frequency: Double, sampleRate: Double, duration: Double) -> [Float] {
        let count = Int(sampleRate * duration)
        return (0..<count).map { i in
            Float(sin(2 * Double.pi * frequency * Double(i) / sampleRate))
        }
    }

    private func assertOGGMagic(_ data: Data, file: StaticString = #file, line: UInt = #line) {
        // Every OGG page starts with the capture pattern "OggS" (0x4F 0x67 0x67 0x53).
        let expected: [UInt8] = [0x4F, 0x67, 0x67, 0x53]
        let actual = Array(data.prefix(4))
        XCTAssertEqual(actual, expected,
            "Expected OGG capture pattern 'OggS', got \(actual.map { String(format: "%02x", $0) })",
            file: file, line: line)
    }
}
