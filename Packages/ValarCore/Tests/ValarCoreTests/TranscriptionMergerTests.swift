import Foundation
import XCTest
@testable import ValarCore
import ValarModelKit

// MARK: - Helpers

private let stubMetadata = BackendMetadata(modelId: "stub-asr", backendKind: .mock)

private func makeASRChunk(
    index: Int = 0,
    sampleCount: Int = 16_000,
    overlapStartSample: Int = 0,
    contentStartSample: Int = 0,
    contentEndSample: Int? = nil
) -> ASRChunk {
    ASRChunk(
        index: index,
        samples: Array(repeating: 0.0, count: sampleCount),
        overlapStartSample: overlapStartSample,
        contentStartSample: contentStartSample,
        contentEndSample: contentEndSample ?? sampleCount
    )
}

private func makeSegment(
    text: String,
    startTime: Double? = nil,
    endTime: Double? = nil,
    confidence: Float? = nil,
    isFinal: Bool = true
) -> TranscriptionSegment {
    TranscriptionSegment(
        text: text,
        startTime: startTime,
        endTime: endTime,
        confidence: confidence,
        isFinal: isFinal
    )
}

private func makeResult(
    segments: [TranscriptionSegment],
    text: String? = nil
) -> RichTranscriptionResult {
    let t = text ?? segments.map(\.text).joined(separator: " ")
    return RichTranscriptionResult(
        text: t,
        segments: segments,
        backendMetadata: stubMetadata
    )
}

// MARK: - TranscriptionMergerTests

final class TranscriptionMergerTests: XCTestCase {

    // MARK: - Empty / edge cases

    func testEmptyResultProducesNoEvents() async {
        let merger = TranscriptionMerger()
        let chunk = makeASRChunk()
        let events = await merger.merge(
            chunk: chunk,
            result: makeResult(segments: []),
            sampleRate: 16_000
        )
        XCTAssertTrue(events.isEmpty, "No segments → no events (no text to partial)")
    }

    func testZeroSampleRateProducesNoEvents() async {
        let merger = TranscriptionMerger()
        let chunk = makeASRChunk(sampleCount: 16_000, contentEndSample: 16_000)
        let result = makeResult(segments: [makeSegment(text: "hello", startTime: 0, endTime: 1)])
        let events = await merger.merge(chunk: chunk, result: result, sampleRate: 0)
        XCTAssertTrue(events.isEmpty)
    }

    // MARK: - Single-chunk, no overlap

    func testSingleChunkAlwaysEmitsPartial() async {
        let merger = TranscriptionMerger()
        let chunk = makeASRChunk(sampleCount: 16_000, contentEndSample: 16_000)
        let seg = makeSegment(text: "Hello world", startTime: 0.0, endTime: 1.0, confidence: 0.5)
        let events = await merger.merge(chunk: chunk, result: makeResult(segments: [seg]), sampleRate: 16_000)

        let partials = events.compactMap { if case .partial(let s) = $0 { return s } else { return nil } }
        XCTAssertEqual(partials.count, 1)
        XCTAssertEqual(partials.first?.text, "Hello world")
        XCTAssertFalse(partials.first?.isFinal ?? true)
    }

    func testSingleChunkBelowThresholdEmitsNoFinalSegment() async {
        let merger = TranscriptionMerger(config: .init(finalConfidenceThreshold: 0.8))
        let chunk = makeASRChunk(sampleCount: 16_000, contentEndSample: 16_000)
        let seg = makeSegment(text: "Hello", startTime: 0.0, endTime: 1.0, confidence: 0.7)
        let events = await merger.merge(chunk: chunk, result: makeResult(segments: [seg]), sampleRate: 16_000)

        let finals = events.compactMap { if case .finalSegment(let s) = $0 { return s } else { return nil } }
        XCTAssertTrue(finals.isEmpty, "Confidence 0.7 < threshold 0.8 → no finalSegment")
    }

    func testSingleChunkAtOrAboveThresholdEmitsFinalSegment() async {
        let merger = TranscriptionMerger(config: .init(finalConfidenceThreshold: 0.8))
        let chunk = makeASRChunk(sampleCount: 16_000, contentEndSample: 16_000)
        let seg = makeSegment(text: "Hello", startTime: 0.0, endTime: 1.0, confidence: 0.8)
        let events = await merger.merge(chunk: chunk, result: makeResult(segments: [seg]), sampleRate: 16_000)

        let finals = events.compactMap { if case .finalSegment(let s) = $0 { return s } else { return nil } }
        XCTAssertEqual(finals.count, 1)
        XCTAssertEqual(finals.first?.text, "Hello")
    }

    func testSingleChunkNoConfidenceMetadataEmitsNoFinalSegment() async {
        // Segments with no confidence value default to 0.0, which is below any positive threshold.
        let merger = TranscriptionMerger(config: .init(finalConfidenceThreshold: 0.8))
        let chunk = makeASRChunk(sampleCount: 16_000, contentEndSample: 16_000)
        let seg = makeSegment(text: "Hello", startTime: 0.0, endTime: 1.0, confidence: nil)
        let events = await merger.merge(chunk: chunk, result: makeResult(segments: [seg]), sampleRate: 16_000)

        let finals = events.compactMap { if case .finalSegment(let s) = $0 { return s } else { return nil } }
        XCTAssertTrue(finals.isEmpty, "nil confidence → treated as 0.0 → no finalSegment")
    }

    // MARK: - Multi-chunk, no overlap

    func testTwoNonOverlappingChunksMergeText() async {
        // Chunk 0: samples [0, 16000), content [0, 16000), no overlap.
        // Chunk 1: samples [16000, 32000), content [16000, 32000), no overlap.
        let merger = TranscriptionMerger(config: .init(finalConfidenceThreshold: 0.9))
        let sampleRate = 16_000

        let chunk0 = makeASRChunk(
            index: 0, sampleCount: 16_000,
            overlapStartSample: 0, contentStartSample: 0, contentEndSample: 16_000
        )
        let result0 = makeResult(segments: [
            makeSegment(text: "Hello", startTime: 0.0, endTime: 0.5, confidence: 0.95),
        ])

        let chunk1 = makeASRChunk(
            index: 1, sampleCount: 16_000,
            overlapStartSample: 16_000, contentStartSample: 16_000, contentEndSample: 32_000
        )
        let result1 = makeResult(segments: [
            makeSegment(text: "world", startTime: 0.0, endTime: 0.5, confidence: 0.95),
        ])

        let events0 = await merger.merge(chunk: chunk0, result: result0, sampleRate: sampleRate)
        let events1 = await merger.merge(chunk: chunk1, result: result1, sampleRate: sampleRate)

        let partials1 = events1.compactMap { if case .partial(let s) = $0 { return s } else { return nil } }
        XCTAssertEqual(partials1.first?.text, "Hello world")

        let finals0 = events0.compactMap { if case .finalSegment(let s) = $0 { return s } else { return nil } }
        let finals1 = events1.compactMap { if case .finalSegment(let s) = $0 { return s } else { return nil } }
        XCTAssertEqual(finals0.count, 1)
        XCTAssertEqual(finals1.count, 1)
    }

    // MARK: - Overlap zone: last-wins

    func testLastWinsReplacesOverlapText() async {
        // Chunk 0: content [0, 16000) → commits "Hello".
        // Chunk 1: overlap [12000, 16000), content [16000, 32000).
        //   ASR for chunk 1 transcribes the overlap region as "Hi" (replaces "Hello"
        //   in that time range) and the content region as "there".
        let sampleRate = 16_000
        let merger = TranscriptionMerger(config: .init(
            finalConfidenceThreshold: 0.9,
            overlapResolution: .lastWins
        ))

        let chunk0 = makeASRChunk(
            index: 0, sampleCount: 16_000,
            overlapStartSample: 0, contentStartSample: 0, contentEndSample: 16_000
        )
        // "Hello" spans [0.0, 0.8s] absolute.
        let result0 = makeResult(segments: [
            makeSegment(text: "Hello", startTime: 0.0, endTime: 0.8),
        ])
        _ = await merger.merge(chunk: chunk0, result: result0, sampleRate: sampleRate)

        // Chunk 1: overlapStartSample = 12000 → chunkOriginTime = 0.75s.
        //          contentStartSample = 16000 → overlapEndTime = 1.0s.
        // Segment "Hi" at chunk-relative t=0.0 → absolute t=0.75s (< 1.0s → overlap zone).
        // Segment "there" at chunk-relative t=0.25 → absolute t=1.0s (== overlapEndTime → content).
        let chunk1 = makeASRChunk(
            index: 1, sampleCount: 20_000,
            overlapStartSample: 12_000, contentStartSample: 16_000, contentEndSample: 32_000
        )
        let result1 = makeResult(segments: [
            makeSegment(text: "Hi",    startTime: 0.0,  endTime: 0.25),
            makeSegment(text: "there", startTime: 0.25, endTime: 1.25),
        ])
        let events1 = await merger.merge(chunk: chunk1, result: result1, sampleRate: sampleRate)

        let partials = events1.compactMap { if case .partial(let s) = $0 { return s } else { return nil } }
        // The overlap "Hello" at [0.75, 0.8] should be replaced by "Hi" at [0.75, 1.0].
        // The transcript should be "Hi there", not "Hello there".
        XCTAssertEqual(partials.first?.text, "Hi there")
    }

    func testLastWinsOnlyReplacesSegmentsWithinOverlapWindow() async {
        // Chunk 0 commits two segments. Chunk 1's overlap only covers the second one.
        let sampleRate = 16_000
        let merger = TranscriptionMerger(config: .init(overlapResolution: .lastWins))

        let chunk0 = makeASRChunk(
            index: 0, sampleCount: 32_000,
            overlapStartSample: 0, contentStartSample: 0, contentEndSample: 32_000
        )
        // "Alpha" at [0.0, 0.5s], "Beta" at [1.5, 2.0s].
        let result0 = makeResult(segments: [
            makeSegment(text: "Alpha", startTime: 0.0, endTime: 0.5),
            makeSegment(text: "Beta",  startTime: 1.5, endTime: 2.0),
        ])
        _ = await merger.merge(chunk: chunk0, result: result0, sampleRate: sampleRate)

        // Chunk 1: overlapStart = 24000 (1.5s), contentStart = 32000 (2.0s).
        // Overlap window covers [1.5, 2.0s). Only "Beta" (at 1.5s) is in the overlap zone.
        let chunk1 = makeASRChunk(
            index: 1, sampleCount: 24_000,
            overlapStartSample: 24_000, contentStartSample: 32_000, contentEndSample: 48_000
        )
        // "Beta2" replaces "Beta" in overlap zone; "Gamma" is new content.
        let result1 = makeResult(segments: [
            makeSegment(text: "Beta2", startTime: 0.0, endTime: 0.5),   // abs: 1.5–2.0s → overlap
            makeSegment(text: "Gamma", startTime: 0.5, endTime: 1.0),   // abs: 2.0–2.5s → content
        ])
        let events1 = await merger.merge(chunk: chunk1, result: result1, sampleRate: sampleRate)

        let partials = events1.compactMap { if case .partial(let s) = $0 { return s } else { return nil } }
        XCTAssertEqual(partials.first?.text, "Alpha Beta2 Gamma",
                       "Only 'Beta' should be replaced; 'Alpha' must be untouched")
    }

    // MARK: - Overlap zone: confidence-wins

    func testConfidenceWinsKeepsHigherConfidenceSegment() async {
        // Chunk 0 commits "Hello" with confidence 0.9.
        // Chunk 1's overlap zone has "Hi" with confidence 0.7 — should NOT replace.
        let sampleRate = 16_000
        let merger = TranscriptionMerger(config: .init(overlapResolution: .confidenceWins))

        let chunk0 = makeASRChunk(
            index: 0, sampleCount: 16_000,
            overlapStartSample: 0, contentStartSample: 0, contentEndSample: 16_000
        )
        _ = await merger.merge(
            chunk: chunk0,
            result: makeResult(segments: [makeSegment(text: "Hello", startTime: 0.0, endTime: 0.8, confidence: 0.9)]),
            sampleRate: sampleRate
        )

        let chunk1 = makeASRChunk(
            index: 1, sampleCount: 20_000,
            overlapStartSample: 12_000, contentStartSample: 16_000, contentEndSample: 32_000
        )
        let events1 = await merger.merge(
            chunk: chunk1,
            result: makeResult(segments: [
                makeSegment(text: "Hi",    startTime: 0.0,  endTime: 0.25, confidence: 0.7),  // overlap, loses
                makeSegment(text: "there", startTime: 0.25, endTime: 1.25, confidence: 0.85), // content
            ]),
            sampleRate: sampleRate
        )

        let partials = events1.compactMap { if case .partial(let s) = $0 { return s } else { return nil } }
        XCTAssertEqual(partials.first?.text, "Hello there",
                       "Existing 'Hello' (0.9) beats new 'Hi' (0.7)")
    }

    func testConfidenceWinsReplacesLowerConfidenceExisting() async {
        // Chunk 0 commits "Hello" with confidence 0.5.
        // Chunk 1's overlap zone has "Hi" with confidence 0.9 — should replace.
        let sampleRate = 16_000
        let merger = TranscriptionMerger(config: .init(overlapResolution: .confidenceWins))

        let chunk0 = makeASRChunk(
            index: 0, sampleCount: 16_000,
            overlapStartSample: 0, contentStartSample: 0, contentEndSample: 16_000
        )
        _ = await merger.merge(
            chunk: chunk0,
            result: makeResult(segments: [makeSegment(text: "Hello", startTime: 0.0, endTime: 0.8, confidence: 0.5)]),
            sampleRate: sampleRate
        )

        let chunk1 = makeASRChunk(
            index: 1, sampleCount: 20_000,
            overlapStartSample: 12_000, contentStartSample: 16_000, contentEndSample: 32_000
        )
        let events1 = await merger.merge(
            chunk: chunk1,
            result: makeResult(segments: [
                makeSegment(text: "Hi",    startTime: 0.0,  endTime: 0.25, confidence: 0.9),  // overlap, wins
                makeSegment(text: "there", startTime: 0.25, endTime: 1.25),
            ]),
            sampleRate: sampleRate
        )

        let partials = events1.compactMap { if case .partial(let s) = $0 { return s } else { return nil } }
        XCTAssertEqual(partials.first?.text, "Hi there",
                       "New 'Hi' (0.9) should replace existing 'Hello' (0.5)")
    }

    // MARK: - All segments in overlap zone

    func testAllSegmentsInOverlapZoneEmitsOnlyPartialWithPreviousText() async {
        // Chunk 0 commits "Alpha".
        // Chunk 1's result only has overlap-zone segments (no content zone).
        // Partial should reflect the resolved overlap text.
        let sampleRate = 16_000
        let merger = TranscriptionMerger(config: .init(overlapResolution: .lastWins))

        let chunk0 = makeASRChunk(
            index: 0, sampleCount: 16_000,
            overlapStartSample: 0, contentStartSample: 0, contentEndSample: 16_000
        )
        _ = await merger.merge(
            chunk: chunk0,
            result: makeResult(segments: [makeSegment(text: "Alpha", startTime: 0.0, endTime: 1.0)]),
            sampleRate: sampleRate
        )

        // Chunk 1: entire chunk is overlap (contentStart == contentEnd == 16000).
        let chunk1 = makeASRChunk(
            index: 1, sampleCount: 16_000,
            overlapStartSample: 0, contentStartSample: 16_000, contentEndSample: 16_000
        )
        let events1 = await merger.merge(
            chunk: chunk1,
            result: makeResult(segments: [makeSegment(text: "Beta", startTime: 0.0, endTime: 1.0)]),
            sampleRate: sampleRate
        )

        // No finalSegment events (content zone is empty).
        let finals = events1.compactMap { if case .finalSegment(let s) = $0 { return s } else { return nil } }
        XCTAssertTrue(finals.isEmpty)

        // Partial reflects last-wins replacement of "Alpha" → "Beta".
        let partials = events1.compactMap { if case .partial(let s) = $0 { return s } else { return nil } }
        XCTAssertEqual(partials.first?.text, "Beta")
    }

    // MARK: - finalize

    func testFinalizeEmitsCompletedEvent() async {
        let merger = TranscriptionMerger()
        let chunk = makeASRChunk(sampleCount: 16_000, contentEndSample: 16_000)
        _ = await merger.merge(
            chunk: chunk,
            result: makeResult(segments: [makeSegment(text: "Done", startTime: 0.0, endTime: 1.0)]),
            sampleRate: 16_000
        )

        let event = await merger.finalize(language: "en", backendMetadata: stubMetadata)
        if case .completed(let result) = event {
            XCTAssertEqual(result.text, "Done")
            XCTAssertEqual(result.language, "en")
            XCTAssertEqual(result.segments.count, 1)
            XCTAssertEqual(result.segments.first?.text, "Done")
        } else {
            XCTFail("Expected .completed event")
        }
    }

    func testFinalizeIncludesAllCommittedSegmentsInOrder() async {
        let sampleRate = 16_000
        let merger = TranscriptionMerger(config: .init(finalConfidenceThreshold: 1.1)) // never fires finalSegment

        for i in 0..<3 {
            let startSample = i * 16_000
            let chunk = makeASRChunk(
                index: i, sampleCount: 16_000,
                overlapStartSample: startSample,
                contentStartSample: startSample,
                contentEndSample: startSample + 16_000
            )
            let t = Double(startSample) / Double(sampleRate)
            let seg = makeSegment(text: "Seg\(i)", startTime: t, endTime: t + 1.0)
            _ = await merger.merge(chunk: chunk, result: makeResult(segments: [seg]), sampleRate: sampleRate)
        }

        let event = await merger.finalize(backendMetadata: stubMetadata)
        if case .completed(let result) = event {
            XCTAssertEqual(result.segments.count, 3)
            XCTAssertEqual(result.segments.map(\.text), ["Seg0", "Seg1", "Seg2"])
            XCTAssertEqual(result.text, "Seg0 Seg1 Seg2")
        } else {
            XCTFail("Expected .completed event")
        }
    }

    func testFinalizeWithNoChunksProducesEmptyResult() async {
        let merger = TranscriptionMerger()
        let event = await merger.finalize(backendMetadata: stubMetadata)
        if case .completed(let result) = event {
            XCTAssertEqual(result.text, "")
            XCTAssertTrue(result.segments.isEmpty)
        } else {
            XCTFail("Expected .completed event")
        }
    }

    func testFinalizeAppliesPostProcessorWhenSet() async {
        let processor = TranscriptionPostProcessor(
            config: .init(confidenceThreshold: nil,
                          enableRepetitionRemoval: false,
                          enablePunctuationNormalization: true)
        )
        let merger = TranscriptionMerger(postProcessor: processor)
        let chunk = makeASRChunk(sampleCount: 16_000, contentEndSample: 16_000)
        _ = await merger.merge(
            chunk: chunk,
            result: makeResult(segments: [makeSegment(text: "hello world", startTime: 0, endTime: 1)]),
            sampleRate: 16_000
        )
        let event = await merger.finalize(backendMetadata: stubMetadata)
        if case .completed(let result) = event {
            // Punctuation normalization should capitalize and add period.
            XCTAssertEqual(result.segments.first?.text, "Hello world.")
        } else {
            XCTFail("Expected .completed event")
        }
    }

    // MARK: - Event ordering

    func testFinalSegmentEventsPrecedePartialInOutput() async {
        let merger = TranscriptionMerger(config: .init(finalConfidenceThreshold: 0.8))
        let chunk = makeASRChunk(sampleCount: 32_000, contentEndSample: 32_000)
        let result = makeResult(segments: [
            makeSegment(text: "A", startTime: 0.0, endTime: 0.5, confidence: 0.9),  // → finalSegment
            makeSegment(text: "B", startTime: 0.5, endTime: 1.0, confidence: 0.5),  // → partial only
        ])
        let events = await merger.merge(chunk: chunk, result: result, sampleRate: 16_000)

        // Expect: [.finalSegment("A"), .partial("A B")]
        XCTAssertEqual(events.count, 2)
        if case .finalSegment(let s) = events[0] {
            XCTAssertEqual(s.text, "A")
        } else {
            XCTFail("First event should be .finalSegment")
        }
        if case .partial(let s) = events[1] {
            XCTAssertEqual(s.text, "A B")
        } else {
            XCTFail("Last event should be .partial")
        }
    }

    func testPartialSegmentIsFinalFalse() async {
        let merger = TranscriptionMerger()
        let chunk = makeASRChunk(sampleCount: 16_000, contentEndSample: 16_000)
        let events = await merger.merge(
            chunk: chunk,
            result: makeResult(segments: [makeSegment(text: "test", startTime: 0, endTime: 1)]),
            sampleRate: 16_000
        )
        let partial = events.compactMap { if case .partial(let s) = $0 { return s } else { return nil } }.first
        XCTAssertEqual(partial?.isFinal, false)
    }

    // MARK: - Absolute timing in committed segments

    func testAbsoluteTimingIsPreservedAfterFinalize() async {
        // Chunk 0 starts at sample 0; chunk 1 starts at sample 16000.
        let sampleRate = 16_000
        let merger = TranscriptionMerger()

        let chunk0 = makeASRChunk(
            index: 0, sampleCount: 16_000,
            overlapStartSample: 0, contentStartSample: 0, contentEndSample: 16_000
        )
        let chunk1 = makeASRChunk(
            index: 1, sampleCount: 16_000,
            overlapStartSample: 16_000, contentStartSample: 16_000, contentEndSample: 32_000
        )

        _ = await merger.merge(
            chunk: chunk0,
            result: makeResult(segments: [makeSegment(text: "A", startTime: 0.0, endTime: 0.5)]),
            sampleRate: sampleRate
        )
        _ = await merger.merge(
            chunk: chunk1,
            result: makeResult(segments: [makeSegment(text: "B", startTime: 0.0, endTime: 0.5)]),
            sampleRate: sampleRate
        )

        let event = await merger.finalize(backendMetadata: stubMetadata)
        if case .completed(let result) = event {
            let times = result.segments.compactMap(\.startTime)
            XCTAssertEqual(times.count, 2)
            XCTAssertEqual(times[0], 0.0, accuracy: 0.001)
            XCTAssertEqual(times[1], 1.0, accuracy: 0.001)
        } else {
            XCTFail("Expected .completed event")
        }
    }
}
