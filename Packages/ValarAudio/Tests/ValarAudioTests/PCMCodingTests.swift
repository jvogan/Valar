import AVFoundation
import XCTest
@testable import ValarAudio

final class PCMCodingTests: XCTestCase {

    // MARK: - Helpers

    private func makeFloatBuffer(channels: Int, frameCount: Int, sampleRate: Double = 24_000) -> AVAudioPCMBuffer {
        let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: sampleRate,
            channels: AVAudioChannelCount(channels),
            interleaved: false
        )!
        let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(frameCount))!
        buffer.frameLength = AVAudioFrameCount(frameCount)
        return buffer
    }

    private func setChannelData(_ buffer: AVAudioPCMBuffer, channelIndex: Int, values: [Float]) {
        guard let data = buffer.floatChannelData else { return }
        for (index, value) in values.enumerated() {
            data[channelIndex][index] = value
        }
    }

    // MARK: - channels(from:) tests

    func testChannelsFromMonoBuffer() {
        let buffer = makeFloatBuffer(channels: 1, frameCount: 4)
        setChannelData(buffer, channelIndex: 0, values: [1, 2, 3, 4])

        let channels = PCMCoding.channels(from: buffer)
        XCTAssertEqual(channels.count, 1)
        XCTAssertEqual(channels[0], [1, 2, 3, 4])
    }

    func testChannelsFromStereoBuffer() {
        let buffer = makeFloatBuffer(channels: 2, frameCount: 3)
        setChannelData(buffer, channelIndex: 0, values: [1, 2, 3])
        setChannelData(buffer, channelIndex: 1, values: [4, 5, 6])

        let channels = PCMCoding.channels(from: buffer)
        XCTAssertEqual(channels.count, 2)
        XCTAssertEqual(channels[0], [1, 2, 3])
        XCTAssertEqual(channels[1], [4, 5, 6])
    }

    func testChannelsFromZeroLengthBufferReturnsEmptyChannels() {
        let buffer = makeFloatBuffer(channels: 2, frameCount: 0)
        let channels = PCMCoding.channels(from: buffer)
        XCTAssertEqual(channels.count, 2)
        XCTAssertTrue(channels[0].isEmpty)
        XCTAssertTrue(channels[1].isEmpty)
    }

    func testChannelsPreservesNegativeAndFractionalValues() {
        let buffer = makeFloatBuffer(channels: 1, frameCount: 4)
        setChannelData(buffer, channelIndex: 0, values: [-1.0, -0.5, 0.5, 1.0])

        let channels = PCMCoding.channels(from: buffer)
        XCTAssertEqual(channels[0][0], -1.0, accuracy: 1e-7)
        XCTAssertEqual(channels[0][1], -0.5, accuracy: 1e-7)
        XCTAssertEqual(channels[0][2],  0.5, accuracy: 1e-7)
        XCTAssertEqual(channels[0][3],  1.0, accuracy: 1e-7)
    }

    // MARK: - fill(_:from:frameCount:) tests

    func testFillMonoBuffer() {
        let buffer = makeFloatBuffer(channels: 1, frameCount: 4)
        PCMCoding.fill(buffer, from: [[10, 20, 30, 40]], frameCount: 4)

        let channels = PCMCoding.channels(from: buffer)
        XCTAssertEqual(channels[0], [10, 20, 30, 40])
    }

    func testFillStereoBuffer() {
        let buffer = makeFloatBuffer(channels: 2, frameCount: 3)
        PCMCoding.fill(buffer, from: [[1, 2, 3], [4, 5, 6]], frameCount: 3)

        let channels = PCMCoding.channels(from: buffer)
        XCTAssertEqual(channels[0], [1, 2, 3])
        XCTAssertEqual(channels[1], [4, 5, 6])
    }

    func testFillWithShorterSourceChannelZeroPadsTail() {
        let buffer = makeFloatBuffer(channels: 1, frameCount: 4)
        // Provide only 2 samples for a 4-frame buffer — tail should be zero-filled
        PCMCoding.fill(buffer, from: [[1, 2]], frameCount: 4)

        let channels = PCMCoding.channels(from: buffer)
        XCTAssertEqual(channels[0][0], 1)
        XCTAssertEqual(channels[0][1], 2)
        XCTAssertEqual(channels[0][2], 0, "Tail should be zero-padded")
        XCTAssertEqual(channels[0][3], 0, "Tail should be zero-padded")
    }

    func testFillWithMissingChannelZeroFillsEntireChannel() {
        let buffer = makeFloatBuffer(channels: 2, frameCount: 3)
        // Set existing values to non-zero so we can verify overwrite
        setChannelData(buffer, channelIndex: 1, values: [9, 9, 9])

        // Only provide channel 0 — channel 1 must be zeroed
        PCMCoding.fill(buffer, from: [[1, 2, 3]], frameCount: 3)

        let channels = PCMCoding.channels(from: buffer)
        XCTAssertEqual(channels[0], [1, 2, 3])
        XCTAssertEqual(channels[1], [0, 0, 0], "Missing channel must be zero-filled")
    }

    func testFillWithEmptyChannelListZeroFillsAllChannels() {
        let buffer = makeFloatBuffer(channels: 2, frameCount: 2)
        setChannelData(buffer, channelIndex: 0, values: [5, 5])
        setChannelData(buffer, channelIndex: 1, values: [5, 5])

        PCMCoding.fill(buffer, from: [], frameCount: 2)

        let channels = PCMCoding.channels(from: buffer)
        XCTAssertEqual(channels[0], [0, 0])
        XCTAssertEqual(channels[1], [0, 0])
    }

    // MARK: - Round-trip

    func testFillThenChannelsRoundTripPreservesData() {
        let original: [[Float]] = [[0.1, 0.2, 0.3, 0.4], [-0.1, -0.2, -0.3, -0.4]]
        let buffer = makeFloatBuffer(channels: 2, frameCount: 4)
        PCMCoding.fill(buffer, from: original, frameCount: 4)

        let recovered = PCMCoding.channels(from: buffer)
        XCTAssertEqual(recovered.count, 2)
        for (index, expectedChannel) in original.enumerated() {
            assertFloatArrayEqual(recovered[index], expectedChannel, accuracy: 1e-7,
                                  "Channel \(index) round-trip mismatch")
        }
    }
}

// MARK: - Array comparison helper

/// Compares two `[Float]` arrays element-wise within the given accuracy.
///
/// Named distinctly from `XCTAssertEqual` to avoid shadowing the global XCTest
/// function (which in Swift 6 causes member-lookup ambiguity for all callers in
/// the same module that subclass `XCTestCase`).
func assertFloatArrayEqual(_ lhs: [Float], _ rhs: [Float], accuracy: Float, _ message: String = "", file: StaticString = #filePath, line: UInt = #line) {
    guard lhs.count == rhs.count else {
        XCTFail("Array length mismatch: \(lhs.count) vs \(rhs.count). \(message)", file: file, line: line)
        return
    }
    for (index, (l, r)) in zip(lhs, rhs).enumerated() {
        XCTAssertEqual(l, r, accuracy: accuracy, "Index \(index): \(message)", file: file, line: line)
    }
}
