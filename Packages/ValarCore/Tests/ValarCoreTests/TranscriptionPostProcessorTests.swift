import Foundation
import XCTest
@testable import ValarCore
import ValarModelKit

// MARK: - Helpers

private func makeSegment(
    text: String,
    confidence: Float? = nil,
    startTime: Double? = nil,
    endTime: Double? = nil,
    isFinal: Bool = true,
    chunkIndex: Int? = nil
) -> TranscriptionSegment {
    TranscriptionSegment(
        text: text,
        startTime: startTime,
        endTime: endTime,
        confidence: confidence,
        isFinal: isFinal,
        chunkIndex: chunkIndex
    )
}

private func makeResult(
    text: String = "",
    segments: [TranscriptionSegment] = []
) -> RichTranscriptionResult {
    RichTranscriptionResult(
        text: text,
        language: "en",
        durationSeconds: nil,
        segments: segments,
        words: nil,
        alignmentReference: nil,
        backendMetadata: BackendMetadata(modelId: "stub", backendKind: .mlx)
    )
}

// MARK: - PostProcessConfidenceFilter tests

final class PostProcessConfidenceFilterTests: XCTestCase {

    func testSegmentsAboveThresholdAreKept() {
        let processor = TranscriptionPostProcessor(
            config: .init(confidenceThreshold: 0.5,
                          enableRepetitionRemoval: false,
                          enablePunctuationNormalization: false)
        )
        let segments = [
            makeSegment(text: "Keep me", confidence: 0.9),
            makeSegment(text: "Also keep", confidence: 0.5),
        ]
        let result = processor.process(makeResult(segments: segments))
        XCTAssertEqual(result.segments.map(\.text), ["Keep me", "Also keep"])
    }

    func testSegmentsBelowThresholdAreDropped() {
        let processor = TranscriptionPostProcessor(
            config: .init(confidenceThreshold: 0.5,
                          enableRepetitionRemoval: false,
                          enablePunctuationNormalization: false)
        )
        let segments = [
            makeSegment(text: "Keep me", confidence: 0.9),
            makeSegment(text: "Drop me", confidence: 0.3),
        ]
        let result = processor.process(makeResult(segments: segments))
        XCTAssertEqual(result.segments.map(\.text), ["Keep me"])
    }

    func testSegmentsWithNoConfidenceAreAlwaysKept() {
        let processor = TranscriptionPostProcessor(
            config: .init(confidenceThreshold: 0.8,
                          enableRepetitionRemoval: false,
                          enablePunctuationNormalization: false)
        )
        let segments = [
            makeSegment(text: "No confidence metadata", confidence: nil),
            makeSegment(text: "Low confidence", confidence: 0.1),
        ]
        let result = processor.process(makeResult(segments: segments))
        XCTAssertEqual(result.segments.map(\.text), ["No confidence metadata"])
    }

    func testNilThresholdKeepsAllSegments() {
        let processor = TranscriptionPostProcessor(
            config: .init(confidenceThreshold: nil,
                          enableRepetitionRemoval: false,
                          enablePunctuationNormalization: false)
        )
        let segments = [
            makeSegment(text: "A", confidence: 0.1),
            makeSegment(text: "B", confidence: 0.9),
        ]
        let result = processor.process(makeResult(segments: segments))
        XCTAssertEqual(result.segments.count, 2)
    }

    func testTopLevelTextReflectsFilteredSegments() {
        let processor = TranscriptionPostProcessor(
            config: .init(confidenceThreshold: 0.5,
                          enableRepetitionRemoval: false,
                          enablePunctuationNormalization: false)
        )
        let segments = [
            makeSegment(text: "Hello", confidence: 0.9),
            makeSegment(text: "garbage", confidence: 0.1),
            makeSegment(text: "world", confidence: 0.8),
        ]
        let result = processor.process(makeResult(segments: segments))
        XCTAssertEqual(result.text, "Hello world")
    }
}

// MARK: - PostProcessRepetitionRemoval tests

final class PostProcessRepetitionRemovalTests: XCTestCase {

    func testConsecutiveDuplicateSegmentsCollapsedToOne() {
        let processor = TranscriptionPostProcessor(
            config: .init(confidenceThreshold: nil,
                          enableRepetitionRemoval: true,
                          enablePunctuationNormalization: false)
        )
        let segments = [
            makeSegment(text: "Hello world"),
            makeSegment(text: "Hello world"),
            makeSegment(text: "Goodbye"),
        ]
        let result = processor.process(makeResult(segments: segments))
        XCTAssertEqual(result.segments.count, 2)
        XCTAssertEqual(result.segments.map(\.text), ["Hello world", "Goodbye"])
    }

    func testNonConsecutiveDuplicatesAreKept() {
        let processor = TranscriptionPostProcessor(
            config: .init(confidenceThreshold: nil,
                          enableRepetitionRemoval: true,
                          enablePunctuationNormalization: false)
        )
        let segments = [
            makeSegment(text: "Hello"),
            makeSegment(text: "World"),
            makeSegment(text: "Hello"),
        ]
        let result = processor.process(makeResult(segments: segments))
        XCTAssertEqual(result.segments.count, 3)
    }

    func testWithinSegmentWordStutterCollapsed() {
        let processor = TranscriptionPostProcessor(
            config: .init(confidenceThreshold: nil,
                          enableRepetitionRemoval: true,
                          enablePunctuationNormalization: false)
        )
        let segments = [makeSegment(text: "I I I went to the the store")]
        let result = processor.process(makeResult(segments: segments))
        XCTAssertEqual(result.segments.first?.text, "I went to the store")
    }

    func testWordStutterCollapseIsCaseInsensitive() {
        let processor = TranscriptionPostProcessor(
            config: .init(confidenceThreshold: nil,
                          enableRepetitionRemoval: true,
                          enablePunctuationNormalization: false)
        )
        let segments = [makeSegment(text: "The the The dog")]
        let result = processor.process(makeResult(segments: segments))
        XCTAssertEqual(result.segments.first?.text, "The dog")
    }

    func testNoRepetitionLeavesSegmentsUnchanged() {
        let processor = TranscriptionPostProcessor(
            config: .init(confidenceThreshold: nil,
                          enableRepetitionRemoval: true,
                          enablePunctuationNormalization: false)
        )
        let segments = [
            makeSegment(text: "Alpha"),
            makeSegment(text: "Beta"),
            makeSegment(text: "Gamma"),
        ]
        let result = processor.process(makeResult(segments: segments))
        XCTAssertEqual(result.segments.map(\.text), ["Alpha", "Beta", "Gamma"])
    }

    func testTimingMetadataPreservedAfterRepetitionRemoval() {
        let processor = TranscriptionPostProcessor(
            config: .init(confidenceThreshold: nil,
                          enableRepetitionRemoval: true,
                          enablePunctuationNormalization: false)
        )
        let segment = makeSegment(
            text: "ping ping me",
            confidence: 0.95,
            startTime: 1.0,
            endTime: 2.5,
            chunkIndex: 3
        )
        let result = processor.process(makeResult(segments: [segment]))
        let out = result.segments.first!
        XCTAssertEqual(out.text, "ping me")
        XCTAssertEqual(out.startTime, 1.0)
        XCTAssertEqual(out.endTime, 2.5)
        XCTAssertEqual(out.confidence, 0.95)
        XCTAssertEqual(out.chunkIndex, 3)
    }
}

// MARK: - PostProcessPunctuationNormalization tests

final class PostProcessPunctuationNormalizationTests: XCTestCase {

    func testLowercaseFirstCharacterIsCapitalized() {
        let processor = TranscriptionPostProcessor(
            config: .init(confidenceThreshold: nil,
                          enableRepetitionRemoval: false,
                          enablePunctuationNormalization: true)
        )
        let segments = [makeSegment(text: "hello world")]
        let result = processor.process(makeResult(segments: segments))
        XCTAssertTrue(result.segments.first?.text.hasPrefix("H") == true)
    }

    func testPeriodAppendedWhenNoTerminalPunctuation() {
        let processor = TranscriptionPostProcessor(
            config: .init(confidenceThreshold: nil,
                          enableRepetitionRemoval: false,
                          enablePunctuationNormalization: true)
        )
        let segments = [makeSegment(text: "It was a bright day")]
        let result = processor.process(makeResult(segments: segments))
        XCTAssertTrue(result.segments.first?.text.hasSuffix(".") == true)
    }

    func testExistingPeriodNotDoubled() {
        let processor = TranscriptionPostProcessor(
            config: .init(confidenceThreshold: nil,
                          enableRepetitionRemoval: false,
                          enablePunctuationNormalization: true)
        )
        let segments = [makeSegment(text: "Already done.")]
        let result = processor.process(makeResult(segments: segments))
        XCTAssertEqual(result.segments.first?.text, "Already done.")
    }

    func testExistingExclamationNotDoubled() {
        let processor = TranscriptionPostProcessor(
            config: .init(confidenceThreshold: nil,
                          enableRepetitionRemoval: false,
                          enablePunctuationNormalization: true)
        )
        let segments = [makeSegment(text: "Wow!")]
        let result = processor.process(makeResult(segments: segments))
        XCTAssertEqual(result.segments.first?.text, "Wow!")
    }

    func testExistingQuestionMarkNotDoubled() {
        let processor = TranscriptionPostProcessor(
            config: .init(confidenceThreshold: nil,
                          enableRepetitionRemoval: false,
                          enablePunctuationNormalization: true)
        )
        let segments = [makeSegment(text: "Are you sure?")]
        let result = processor.process(makeResult(segments: segments))
        XCTAssertEqual(result.segments.first?.text, "Are you sure?")
    }

    func testQuestionStarterReceivesQuestionMark() {
        let processor = TranscriptionPostProcessor(
            config: .init(confidenceThreshold: nil,
                          enableRepetitionRemoval: false,
                          enablePunctuationNormalization: true)
        )
        let starters = ["What", "Where", "Who", "When", "Why", "How",
                        "Is", "Are", "Was", "Were", "Do", "Does", "Did",
                        "Can", "Could", "Should", "Would"]
        for starter in starters {
            let segments = [makeSegment(text: "\(starter) is this")]
            let result = processor.process(makeResult(segments: segments))
            XCTAssertTrue(
                result.segments.first?.text.hasSuffix("?") == true,
                "Expected '?' for starter '\(starter)', got: \(result.segments.first?.text ?? "nil")"
            )
        }
    }

    func testNonQuestionStarterReceivesPeriod() {
        let processor = TranscriptionPostProcessor(
            config: .init(confidenceThreshold: nil,
                          enableRepetitionRemoval: false,
                          enablePunctuationNormalization: true)
        )
        let segments = [makeSegment(text: "the sky is blue")]
        let result = processor.process(makeResult(segments: segments))
        XCTAssertTrue(result.segments.first?.text.hasSuffix(".") == true)
    }

    func testEmptySegmentsAreRemovedDuringNormalization() {
        let processor = TranscriptionPostProcessor(
            config: .init(confidenceThreshold: nil,
                          enableRepetitionRemoval: false,
                          enablePunctuationNormalization: true)
        )
        let segments = [
            makeSegment(text: "Hello"),
            makeSegment(text: "   "),
            makeSegment(text: "World"),
        ]
        let result = processor.process(makeResult(segments: segments))
        XCTAssertEqual(result.segments.count, 2)
    }
}

// MARK: - PostProcessPipelineIntegration tests

final class PostProcessPipelineIntegrationTests: XCTestCase {

    func testFullPipelineWithDefaults() {
        let processor = TranscriptionPostProcessor()
        let segments = [
            makeSegment(text: "hello hello world", confidence: 0.9),
            makeSegment(text: "hello hello world", confidence: 0.9),
            makeSegment(text: "what time is it", confidence: 0.7),
        ]
        let raw = makeResult(text: "hello hello world hello hello world what time is it", segments: segments)
        let result = processor.process(raw)

        // Consecutive duplicate collapsed to one.
        XCTAssertEqual(result.segments.count, 2)
        // Stutter removed and capitalized with period.
        XCTAssertEqual(result.segments[0].text, "Hello world.")
        // Question gets '?'.
        XCTAssertEqual(result.segments[1].text, "What time is it?")
    }

    func testPassthroughConfigChangesNothing() {
        let processor = TranscriptionPostProcessor(config: .passthrough)
        let segments = [makeSegment(text: "unchanged text")]
        let raw = makeResult(text: "unchanged text", segments: segments)
        let result = processor.process(raw)
        XCTAssertEqual(result.segments.first?.text, "unchanged text")
    }

    func testConfidenceFilterRunsBeforeRepetitionRemoval() {
        // If a low-confidence duplicate is dropped first, repetition removal
        // should not see it and should leave the remaining segment intact.
        let processor = TranscriptionPostProcessor(
            config: .init(confidenceThreshold: 0.5,
                          enableRepetitionRemoval: true,
                          enablePunctuationNormalization: false)
        )
        let segments = [
            makeSegment(text: "Alpha", confidence: 0.9),
            makeSegment(text: "Alpha", confidence: 0.1), // dropped by confidence filter
        ]
        let result = processor.process(makeResult(segments: segments))
        // Low-confidence duplicate is dropped; only one "Alpha" remains.
        XCTAssertEqual(result.segments.count, 1)
        XCTAssertEqual(result.segments.first?.text, "Alpha")
    }
}

// MARK: - PostProcessTranscriptionMerger tests

final class PostProcessTranscriptionMergerTests: XCTestCase {

    func testMergerConcatenatesChunkSegments() async {
        let merger = TranscriptionMerger()
        let chunk1 = makeResult(text: "Hello", segments: [makeSegment(text: "Hello")])
        let chunk2 = makeResult(text: "world", segments: [makeSegment(text: "world")])
        let result = await merger.merge(chunks: [chunk1, chunk2])
        XCTAssertEqual(result?.segments.count, 2)
        XCTAssertEqual(result?.text, "Hello world")
    }

    func testMergerReturnsNilForEmptyInput() async {
        let merger = TranscriptionMerger()
        let result = await merger.merge(chunks: [])
        XCTAssertNil(result)
    }

    func testMergerWithPostProcessorAppliesCleanup() async {
        let processor = TranscriptionPostProcessor(
            config: .init(confidenceThreshold: nil,
                          enableRepetitionRemoval: false,
                          enablePunctuationNormalization: true)
        )
        let merger = TranscriptionMerger(postProcessor: processor)
        let chunks = [
            makeResult(text: "hello", segments: [makeSegment(text: "hello")]),
        ]
        let result = await merger.merge(chunks: chunks)
        XCTAssertEqual(result?.segments.first?.text, "Hello.")
    }

    func testMergerWithoutPostProcessorLeavesTextRaw() async {
        let merger = TranscriptionMerger(postProcessor: nil)
        let chunks = [
            makeResult(text: "raw text", segments: [makeSegment(text: "raw text")]),
        ]
        let result = await merger.merge(chunks: chunks)
        XCTAssertEqual(result?.segments.first?.text, "raw text")
    }

    func testMergerSumsDurationSeconds() async {
        let merger = TranscriptionMerger()
        let chunk1 = RichTranscriptionResult(
            text: "A",
            language: "en",
            durationSeconds: 3.0,
            segments: [makeSegment(text: "A")],
            words: nil,
            alignmentReference: nil,
            backendMetadata: BackendMetadata(modelId: "stub", backendKind: .mlx)
        )
        let chunk2 = RichTranscriptionResult(
            text: "B",
            language: "en",
            durationSeconds: 5.0,
            segments: [makeSegment(text: "B")],
            words: nil,
            alignmentReference: nil,
            backendMetadata: BackendMetadata(modelId: "stub", backendKind: .mlx)
        )
        let result = await merger.merge(chunks: [chunk1, chunk2])
        XCTAssertEqual(result?.durationSeconds, 8.0)
    }
}
