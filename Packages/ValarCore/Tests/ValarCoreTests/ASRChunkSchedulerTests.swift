import Foundation
import XCTest
@testable import ValarCore

final class ASRChunkSchedulerTests: XCTestCase {

    // MARK: - Helpers

    private func silent(_ count: Int) -> [Float] {
        Array(repeating: 0.0, count: count)
    }

    // MARK: - Empty / edge cases

    func testEmptyAudioReturnsEmpty() {
        let scheduler = ASRChunkScheduler()
        let result = scheduler.schedule(
            audio: [],
            speechProbabilities: [0.8],
            sampleRate: 16_000,
            vadChunkSize: 4096
        )
        XCTAssertTrue(result.isEmpty)
    }

    func testEmptyProbabilitiesReturnsEmpty() {
        let scheduler = ASRChunkScheduler()
        let result = scheduler.schedule(
            audio: silent(4096),
            speechProbabilities: [],
            sampleRate: 16_000,
            vadChunkSize: 4096
        )
        XCTAssertTrue(result.isEmpty)
    }

    func testAllSilenceReturnsEmpty() {
        let scheduler = ASRChunkScheduler(
            config: ASRChunkSchedulerConfig(
                minSpeechDuration: 0.0,
                silenceThreshold: 0.5
            )
        )
        let result = scheduler.schedule(
            audio: silent(8192),
            speechProbabilities: [0.1, 0.2],
            sampleRate: 16_000,
            vadChunkSize: 4096
        )
        XCTAssertTrue(result.isEmpty)
    }

    // MARK: - Chunk boundaries

    func testSingleSpeechFrameProducesOneChunk() {
        let vadChunkSize = 4096
        let scheduler = ASRChunkScheduler(
            config: ASRChunkSchedulerConfig(
                targetChunkDuration: 10.0,
                overlapDuration: 0.0,
                minSpeechDuration: 0.0,
                silenceThreshold: 0.5
            )
        )
        let result = scheduler.schedule(
            audio: silent(vadChunkSize),
            speechProbabilities: [0.9],
            sampleRate: 16_000,
            vadChunkSize: vadChunkSize
        )

        XCTAssertEqual(result.count, 1)
        XCTAssertEqual(result[0].index, 0)
        XCTAssertEqual(result[0].contentStartSample, 0)
        XCTAssertEqual(result[0].contentEndSample, vadChunkSize)
        XCTAssertEqual(result[0].overlapStartSample, 0)
        XCTAssertEqual(result[0].samples.count, vadChunkSize)
    }

    func testSpeechBoundaryMatchesVADChunkBoundary() {
        // Frames: [silence, speech, silence] → one chunk covering frame 1.
        let vadChunkSize = 4096
        let scheduler = ASRChunkScheduler(
            config: ASRChunkSchedulerConfig(
                overlapDuration: 0.0,
                minSpeechDuration: 0.0,
                silenceThreshold: 0.5
            )
        )
        let result = scheduler.schedule(
            audio: silent(vadChunkSize * 3),
            speechProbabilities: [0.1, 0.9, 0.1],
            sampleRate: 16_000,
            vadChunkSize: vadChunkSize
        )

        XCTAssertEqual(result.count, 1)
        XCTAssertEqual(result[0].contentStartSample, vadChunkSize)
        XCTAssertEqual(result[0].contentEndSample, vadChunkSize * 2)
    }

    func testContiguousSpeechFramesProduceOneChunk() {
        let vadChunkSize = 4096
        let totalSamples = vadChunkSize * 5
        let scheduler = ASRChunkScheduler(
            config: ASRChunkSchedulerConfig(
                targetChunkDuration: 10.0,   // 160 000 samples — all 5 frames fit
                overlapDuration: 0.0,
                minSpeechDuration: 0.0,
                silenceThreshold: 0.5
            )
        )
        let result = scheduler.schedule(
            audio: silent(totalSamples),
            speechProbabilities: [0.9, 0.85, 0.8, 0.9, 0.7],
            sampleRate: 16_000,
            vadChunkSize: vadChunkSize
        )

        XCTAssertEqual(result.count, 1)
        XCTAssertEqual(result[0].contentStartSample, 0)
        XCTAssertEqual(result[0].contentEndSample, totalSamples)
    }

    // MARK: - Silence skipping

    func testNonContiguousSpeechFramesProduceSeparateChunks() {
        // Frames: [silence, speech, silence, speech]
        let vadChunkSize = 4096
        let scheduler = ASRChunkScheduler(
            config: ASRChunkSchedulerConfig(
                overlapDuration: 0.0,
                minSpeechDuration: 0.0,
                silenceThreshold: 0.5
            )
        )
        let result = scheduler.schedule(
            audio: silent(vadChunkSize * 4),
            speechProbabilities: [0.1, 0.9, 0.2, 0.8],
            sampleRate: 16_000,
            vadChunkSize: vadChunkSize
        )

        XCTAssertEqual(result.count, 2)
        XCTAssertEqual(result[0].contentStartSample, vadChunkSize)
        XCTAssertEqual(result[0].contentEndSample, vadChunkSize * 2)
        XCTAssertEqual(result[1].contentStartSample, vadChunkSize * 3)
        XCTAssertEqual(result[1].contentEndSample, vadChunkSize * 4)
    }

    // MARK: - Overlap regions

    func testOverlapPrecedesContentStart() {
        // Frame 0: silence, Frame 1: speech. Overlap = 4000 samples.
        let vadChunkSize = 4096
        let sampleRate = 16_000
        let overlapSamples = 4000
        let scheduler = ASRChunkScheduler(
            config: ASRChunkSchedulerConfig(
                targetChunkDuration: 10.0,
                overlapDuration: Double(overlapSamples) / Double(sampleRate),
                minSpeechDuration: 0.0,
                silenceThreshold: 0.5
            )
        )
        let result = scheduler.schedule(
            audio: silent(vadChunkSize * 2),
            speechProbabilities: [0.1, 0.9],
            sampleRate: sampleRate,
            vadChunkSize: vadChunkSize
        )

        XCTAssertEqual(result.count, 1)
        let chunk = result[0]
        XCTAssertEqual(chunk.contentStartSample, vadChunkSize)
        XCTAssertEqual(chunk.overlapStartSample, vadChunkSize - overlapSamples)
        XCTAssertEqual(chunk.samples.count, overlapSamples + vadChunkSize)
    }

    func testOverlapClampedAtBufferStart() {
        // Speech starts at frame 0; overlap would precede the buffer → clamped to 0.
        let vadChunkSize = 4096
        let scheduler = ASRChunkScheduler(
            config: ASRChunkSchedulerConfig(
                targetChunkDuration: 10.0,
                overlapDuration: 1.0,   // 16 000 samples, but buffer starts at 0
                minSpeechDuration: 0.0,
                silenceThreshold: 0.5
            )
        )
        let result = scheduler.schedule(
            audio: silent(vadChunkSize),
            speechProbabilities: [0.9],
            sampleRate: 16_000,
            vadChunkSize: vadChunkSize
        )

        XCTAssertEqual(result.count, 1)
        XCTAssertEqual(result[0].overlapStartSample, 0)
        XCTAssertEqual(result[0].contentStartSample, 0)
        XCTAssertEqual(result[0].samples.count, vadChunkSize)
    }

    func testSubsequentSubChunksIncludeOverlapFromPrecedingContent() {
        // Two speech frames at 1 s/frame; target = 1 s → two sub-chunks.
        // Overlap = 4000 samples. Second chunk's overlap is drawn from the first chunk's tail.
        let vadChunkSize = 16_000
        let sampleRate = 16_000
        let overlapSamples = 4_000
        let scheduler = ASRChunkScheduler(
            config: ASRChunkSchedulerConfig(
                targetChunkDuration: 1.0,
                overlapDuration: Double(overlapSamples) / Double(sampleRate),
                minSpeechDuration: 0.0,
                silenceThreshold: 0.5
            )
        )
        let result = scheduler.schedule(
            audio: silent(vadChunkSize * 2),
            speechProbabilities: [0.9, 0.9],
            sampleRate: sampleRate,
            vadChunkSize: vadChunkSize
        )

        XCTAssertEqual(result.count, 2)

        // Chunk 0: content [0, 16000), overlap also starts at 0 (clamped).
        XCTAssertEqual(result[0].overlapStartSample, 0)
        XCTAssertEqual(result[0].contentStartSample, 0)
        XCTAssertEqual(result[0].contentEndSample, vadChunkSize)

        // Chunk 1: content [16000, 32000), overlap starts at 12000.
        XCTAssertEqual(result[1].overlapStartSample, vadChunkSize - overlapSamples)  // 12000
        XCTAssertEqual(result[1].contentStartSample, vadChunkSize)
        XCTAssertEqual(result[1].contentEndSample, vadChunkSize * 2)
        XCTAssertEqual(result[1].samples.count, overlapSamples + vadChunkSize)
    }

    // MARK: - Minimum speech duration

    func testRegionShorterThanMinSpeechDurationIsDropped() {
        // 1 frame of speech at 16 kHz, 4096 samples ≈ 256 ms.
        // minSpeechDuration = 0.5 s → region should be dropped.
        let vadChunkSize = 4096
        let scheduler = ASRChunkScheduler(
            config: ASRChunkSchedulerConfig(
                targetChunkDuration: 10.0,
                overlapDuration: 0.0,
                minSpeechDuration: 0.5,
                silenceThreshold: 0.5
            )
        )
        let result = scheduler.schedule(
            audio: silent(vadChunkSize),
            speechProbabilities: [0.9],
            sampleRate: 16_000,
            vadChunkSize: vadChunkSize
        )

        XCTAssertTrue(result.isEmpty, "Region shorter than minSpeechDuration must be excluded")
    }

    func testRegionMeetingExactMinimumIsIncluded() {
        // vadChunkSize = 8000 samples at 16 kHz = 0.5 s exactly → meets the minimum.
        let vadChunkSize = 8_000
        let scheduler = ASRChunkScheduler(
            config: ASRChunkSchedulerConfig(
                targetChunkDuration: 10.0,
                overlapDuration: 0.0,
                minSpeechDuration: 0.5,
                silenceThreshold: 0.5
            )
        )
        let result = scheduler.schedule(
            audio: silent(vadChunkSize),
            speechProbabilities: [0.9],
            sampleRate: 16_000,
            vadChunkSize: vadChunkSize
        )

        XCTAssertEqual(result.count, 1)
    }

    // MARK: - Long region splitting

    func testLongRegionIsSplitAtTargetDuration() {
        // 3 speech frames of 1 s each; targetChunkDuration = 1 s → 3 chunks.
        let vadChunkSize = 16_000
        let sampleRate = 16_000
        let scheduler = ASRChunkScheduler(
            config: ASRChunkSchedulerConfig(
                targetChunkDuration: 1.0,
                overlapDuration: 0.0,
                minSpeechDuration: 0.0,
                silenceThreshold: 0.5
            )
        )
        let result = scheduler.schedule(
            audio: silent(vadChunkSize * 3),
            speechProbabilities: [0.9, 0.9, 0.9],
            sampleRate: sampleRate,
            vadChunkSize: vadChunkSize
        )

        XCTAssertEqual(result.count, 3)
        XCTAssertEqual(result[0].contentStartSample, 0)
        XCTAssertEqual(result[0].contentEndSample, vadChunkSize)
        XCTAssertEqual(result[1].contentStartSample, vadChunkSize)
        XCTAssertEqual(result[1].contentEndSample, vadChunkSize * 2)
        XCTAssertEqual(result[2].contentStartSample, vadChunkSize * 2)
        XCTAssertEqual(result[2].contentEndSample, vadChunkSize * 3)
    }

    func testChunksAreIndexedSequentially() {
        let vadChunkSize = 16_000
        let scheduler = ASRChunkScheduler(
            config: ASRChunkSchedulerConfig(
                targetChunkDuration: 1.0,
                overlapDuration: 0.0,
                minSpeechDuration: 0.0,
                silenceThreshold: 0.5
            )
        )
        let result = scheduler.schedule(
            audio: silent(vadChunkSize * 4),
            speechProbabilities: [0.9, 0.9, 0.9, 0.9],
            sampleRate: 16_000,
            vadChunkSize: vadChunkSize
        )

        for (expected, chunk) in result.enumerated() {
            XCTAssertEqual(chunk.index, expected, "Chunk at position \(expected) must have index \(expected)")
        }
    }

    // MARK: - Silence threshold boundary

    func testProbabilityExactlyAtThresholdIsTreatedAsSpeech() {
        let vadChunkSize = 4096
        let threshold: Float = 0.5
        let scheduler = ASRChunkScheduler(
            config: ASRChunkSchedulerConfig(
                targetChunkDuration: 10.0,
                overlapDuration: 0.0,
                minSpeechDuration: 0.0,
                silenceThreshold: threshold
            )
        )
        let result = scheduler.schedule(
            audio: silent(vadChunkSize),
            speechProbabilities: [threshold],
            sampleRate: 16_000,
            vadChunkSize: vadChunkSize
        )

        XCTAssertEqual(result.count, 1, "Probability == threshold must be treated as speech")
    }

    func testProbabilityBelowThresholdIsTreatedAsSilence() {
        let vadChunkSize = 4096
        let threshold: Float = 0.5
        let scheduler = ASRChunkScheduler(
            config: ASRChunkSchedulerConfig(
                targetChunkDuration: 10.0,
                overlapDuration: 0.0,
                minSpeechDuration: 0.0,
                silenceThreshold: threshold
            )
        )
        let result = scheduler.schedule(
            audio: silent(vadChunkSize),
            speechProbabilities: [threshold - 0.001],
            sampleRate: 16_000,
            vadChunkSize: vadChunkSize
        )

        XCTAssertTrue(result.isEmpty, "Probability below threshold must be treated as silence")
    }

    // MARK: - Default configuration

    func testDefaultConfigHasExpectedValues() {
        let config = ASRChunkSchedulerConfig()
        XCTAssertEqual(config.targetChunkDuration, 10.0)
        XCTAssertEqual(config.overlapDuration, 1.0)
        XCTAssertEqual(config.minSpeechDuration, 0.5)
        XCTAssertEqual(config.silenceThreshold, 0.5)
    }
}
