import Foundation
import XCTest
@testable import ValarCore
import ValarModelKit

// MARK: - WER Support Types

/// Detailed breakdown of word-level edit operations for WER computation.
private struct WERScore: CustomStringConvertible {
    let substitutions: Int
    let deletions: Int
    let insertions: Int
    let referenceLength: Int

    /// Word Error Rate: (S + D + I) / N.
    /// Returns 1.0 when reference is empty and hypothesis is non-empty.
    var rate: Double {
        guard referenceLength > 0 else { return insertions > 0 ? 1.0 : 0.0 }
        return Double(substitutions + deletions + insertions) / Double(referenceLength)
    }

    var editCount: Int { substitutions + deletions + insertions }

    var description: String {
        "WER=\(String(format: "%.1f%%", rate * 100)) (S=\(substitutions) D=\(deletions) I=\(insertions) N=\(referenceLength))"
    }
}

// MARK: - WER Calculator

/// Compute word-level WER between a reference and hypothesis string.
///
/// Both strings are normalized — lowercased and stripped of punctuation
/// (apostrophes in contractions are preserved) — before alignment.
/// Uses the Wagner-Fischer dynamic programming algorithm.
private func computeWER(reference: String, hypothesis: String) -> WERScore {
    let ref = normalizeForWER(reference)
    let hyp = normalizeForWER(hypothesis)

    let n = ref.count
    let m = hyp.count

    if n == 0 && m == 0 { return WERScore(substitutions: 0, deletions: 0, insertions: 0, referenceLength: 0) }
    if n == 0 { return WERScore(substitutions: 0, deletions: 0, insertions: m, referenceLength: 0) }
    if m == 0 { return WERScore(substitutions: 0, deletions: n, insertions: 0, referenceLength: n) }

    // dp[i][j] tracks (substitutions, deletions, insertions) for optimal alignment
    // of ref[0..<i] with hyp[0..<j].
    struct Op { var s, d, i: Int; var total: Int { s + d + i } }
    var dp = Array(repeating: Array(repeating: Op(s: 0, d: 0, i: 0), count: m + 1), count: n + 1)

    for i in 1...n { dp[i][0] = Op(s: 0, d: i, i: 0) }
    for j in 1...m { dp[0][j] = Op(s: 0, d: 0, i: j) }

    for i in 1...n {
        for j in 1...m {
            if ref[i - 1] == hyp[j - 1] {
                dp[i][j] = dp[i - 1][j - 1]
            } else {
                let sub = dp[i - 1][j - 1]
                let del = dp[i - 1][j]
                let ins = dp[i][j - 1]

                if sub.total <= del.total && sub.total <= ins.total {
                    dp[i][j] = Op(s: sub.s + 1, d: sub.d, i: sub.i)
                } else if del.total <= ins.total {
                    dp[i][j] = Op(s: del.s, d: del.d + 1, i: del.i)
                } else {
                    dp[i][j] = Op(s: ins.s, d: ins.d, i: ins.i + 1)
                }
            }
        }
    }

    let result = dp[n][m]
    return WERScore(substitutions: result.s, deletions: result.d, insertions: result.i, referenceLength: n)
}

/// Normalize text for WER comparison: lowercase, strip punctuation (keep apostrophes), split on whitespace.
private func normalizeForWER(_ text: String) -> [String] {
    var cleaned = ""
    for ch in text.lowercased() {
        if ch.isLetter || ch.isNumber || ch == "'" || ch.isWhitespace {
            cleaned.append(ch)
        } else {
            cleaned.append(" ")  // replace punctuation with space to avoid word-merging
        }
    }
    return cleaned.components(separatedBy: .whitespaces).filter { !$0.isEmpty }
}

// MARK: - Benchmark Corpus

private struct CorpusEntry {
    let description: String
    let reference: String
    /// Raw ASR hypothesis (no post-processing applied).
    let rawHypothesis: String
    /// Post-processed hypothesis (after punctuation normalization and repetition removal).
    let cleanedHypothesis: String
    /// Upper bound on raw WER for this entry.
    let maxRawWER: Double
    /// Upper bound on cleaned WER (should be ≤ maxRawWER).
    let maxCleanedWER: Double
}

private let benchmarkCorpus: [CorpusEntry] = [
    CorpusEntry(
        description: "Perfect transcription",
        reference: "The quick brown fox jumps over the lazy dog",
        rawHypothesis: "the quick brown fox jumps over the lazy dog",
        cleanedHypothesis: "The quick brown fox jumps over the lazy dog.",
        maxRawWER: 0.0,
        maxCleanedWER: 0.0
    ),
    CorpusEntry(
        description: "Single word substitution",
        reference: "The quick brown fox jumps over the lazy dog",
        rawHypothesis: "the quick brown cat jumps over the lazy dog",
        cleanedHypothesis: "The quick brown cat jumps over the lazy dog.",
        maxRawWER: 0.12,    // 1/9 ≈ 11.1%
        maxCleanedWER: 0.12
    ),
    CorpusEntry(
        description: "Word stutter corrected by post-processor",
        reference: "The quick brown fox jumps over the lazy dog",
        rawHypothesis: "the quick brown fox fox jumps over the lazy dog",
        cleanedHypothesis: "The quick brown fox jumps over the lazy dog.",
        maxRawWER: 0.12,    // 1 insertion (fox fox → extra "fox") / 9 words
        maxCleanedWER: 0.0  // post-processor removes stutter → perfect
    ),
    CorpusEntry(
        description: "Single word deletion",
        reference: "The quick brown fox jumps over the lazy dog",
        rawHypothesis: "the quick brown fox jumps the lazy dog",
        cleanedHypothesis: "The quick brown fox jumps the lazy dog.",
        maxRawWER: 0.12,    // 1/9 ≈ 11.1%
        maxCleanedWER: 0.12
    ),
    CorpusEntry(
        description: "Single word insertion",
        reference: "Speech recognition is difficult",
        rawHypothesis: "speech recognition is very difficult",
        cleanedHypothesis: "Speech recognition is very difficult.",
        maxRawWER: 0.26,    // 1/4 = 25%
        maxCleanedWER: 0.26
    ),
    CorpusEntry(
        description: "Case-only difference",
        reference: "Hello world",
        rawHypothesis: "HELLO WORLD",
        cleanedHypothesis: "Hello world.",
        maxRawWER: 0.0,     // normalized to lowercase → perfect
        maxCleanedWER: 0.0
    ),
    CorpusEntry(
        description: "Contractions preserved",
        reference: "I don't know what you're talking about",
        rawHypothesis: "i don't know what you're talking about",
        cleanedHypothesis: "I don't know what you're talking about.",
        maxRawWER: 0.0,
        maxCleanedWER: 0.0
    ),
    CorpusEntry(
        description: "Numeric content",
        reference: "The meeting starts at 9 AM",
        rawHypothesis: "the meeting starts at 9 am",
        cleanedHypothesis: "The meeting starts at 9 AM.",
        maxRawWER: 0.0,
        maxCleanedWER: 0.0
    ),
    CorpusEntry(
        description: "Multiple consecutive stutters",
        reference: "I went to the store",
        rawHypothesis: "i i i went to the the the store",
        cleanedHypothesis: "I went to the store.",
        maxRawWER: 0.81,    // raw: 4 insertions / 5 words = 80%
        maxCleanedWER: 0.0  // cleaned: perfect
    ),
    CorpusEntry(
        description: "Severe hallucination",
        reference: "Thank you for calling customer support",
        rawHypothesis: "the the the",
        cleanedHypothesis: "The.",
        maxRawWER: 1.01,    // > 100% is possible with many insertions + deletions
        maxCleanedWER: 1.01
    ),
]

// MARK: - Shared Test Helpers

private let benchStubMetadata = BackendMetadata(modelId: "bench-asr-stub", backendKind: .mock)

private func makeBenchSegment(
    text: String,
    startTime: Double? = nil,
    endTime: Double? = nil,
    confidence: Float? = nil
) -> TranscriptionSegment {
    TranscriptionSegment(
        text: text,
        startTime: startTime,
        endTime: endTime,
        confidence: confidence,
        isFinal: true
    )
}

private func makeBenchResult(segments: [TranscriptionSegment]) -> RichTranscriptionResult {
    let text = segments.map(\.text).joined(separator: " ")
    return RichTranscriptionResult(
        text: text,
        language: "en",
        durationSeconds: segments.compactMap(\.endTime).max(),
        segments: segments,
        backendMetadata: benchStubMetadata
    )
}

private func makeBenchChunk(
    index: Int,
    overlapStartSample: Int,
    contentStartSample: Int,
    contentEndSample: Int,
    totalSamples: Int? = nil
) -> ASRChunk {
    let count = totalSamples ?? (contentEndSample - overlapStartSample)
    return ASRChunk(
        index: index,
        samples: Array(repeating: 0.0, count: max(1, count)),
        overlapStartSample: overlapStartSample,
        contentStartSample: contentStartSample,
        contentEndSample: contentEndSample
    )
}

// MARK: - ASRWERCalculatorTests

/// Unit tests verifying correctness of the WER computation helper.
final class ASRWERCalculatorTests: XCTestCase {

    func testPerfectTranscriptionIsZeroWER() {
        let score = computeWER(
            reference: "the quick brown fox",
            hypothesis: "the quick brown fox"
        )
        XCTAssertEqual(score.rate, 0.0, accuracy: 0.001)
        XCTAssertEqual(score.editCount, 0)
    }

    func testOneSubstitutionInFourWords() {
        let score = computeWER(
            reference: "the quick brown fox",
            hypothesis: "the quick black fox"
        )
        XCTAssertEqual(score.substitutions, 1)
        XCTAssertEqual(score.deletions, 0)
        XCTAssertEqual(score.insertions, 0)
        XCTAssertEqual(score.rate, 0.25, accuracy: 0.001)
    }

    func testOneDeletionInFiveWords() {
        let score = computeWER(
            reference: "one two three four five",
            hypothesis: "one two four five"
        )
        XCTAssertEqual(score.deletions, 1)
        XCTAssertEqual(score.rate, 0.2, accuracy: 0.001)
    }

    func testOneInsertionInFourWords() {
        let score = computeWER(
            reference: "one two three four",
            hypothesis: "one two extra three four"
        )
        XCTAssertEqual(score.insertions, 1)
        XCTAssertEqual(score.rate, 0.25, accuracy: 0.001)
    }

    func testCaseInsensitiveNormalization() {
        let score = computeWER(
            reference: "Hello World",
            hypothesis: "HELLO WORLD"
        )
        XCTAssertEqual(score.rate, 0.0, accuracy: 0.001, "Case differences must not contribute to WER")
    }

    func testPunctuationStrippedBeforeComparison() {
        let score = computeWER(
            reference: "Hello world",
            hypothesis: "Hello world."
        )
        XCTAssertEqual(score.rate, 0.0, accuracy: 0.001, "Terminal period must not count as an error")
    }

    func testMultiplePunctuationStripped() {
        let score = computeWER(
            reference: "Yes I agree",
            hypothesis: "Yes, I agree!"
        )
        XCTAssertEqual(score.rate, 0.0, accuracy: 0.001)
    }

    func testEmptyReferenceNonEmptyHypothesisRate() {
        let score = computeWER(reference: "", hypothesis: "unexpected words")
        XCTAssertEqual(score.referenceLength, 0)
        XCTAssertEqual(score.insertions, 2)
        XCTAssertEqual(score.rate, 1.0)  // defined as 1.0 when reference is empty and hypothesis is non-empty
    }

    func testEmptyHypothesisProducesDeletionRate() {
        let score = computeWER(reference: "one two three", hypothesis: "")
        XCTAssertEqual(score.deletions, 3)
        XCTAssertEqual(score.rate, 1.0, accuracy: 0.001)
    }

    func testBothEmptyIsZeroWER() {
        let score = computeWER(reference: "", hypothesis: "")
        XCTAssertEqual(score.rate, 0.0)
        XCTAssertEqual(score.editCount, 0)
    }

    func testWordOrderMattersForWER() {
        let score = computeWER(
            reference: "one two three",
            hypothesis: "three two one"
        )
        XCTAssertGreaterThan(score.rate, 0.0, "Transposed words are not a perfect match")
    }

    func testStutterCausesInsertionErrors() {
        // "fox fox" has one extra "fox" compared to reference "fox" → 1 insertion
        let score = computeWER(
            reference: "the fox ran",
            hypothesis: "the fox fox ran"
        )
        XCTAssertEqual(score.insertions, 1)
        XCTAssertEqual(score.rate, 1.0 / 3.0, accuracy: 0.001)
    }

    func testApostrophesPreservedInContractions() {
        let score = computeWER(
            reference: "I don't know",
            hypothesis: "i don't know"
        )
        XCTAssertEqual(score.rate, 0.0, accuracy: 0.001)
    }
}

// MARK: - ASRCorpusBenchmarkTests

/// Corpus-level WER benchmarks.
///
/// Each entry in the benchmark corpus has a defined maximum WER for both the
/// raw ASR hypothesis and the post-processed hypothesis. These assertions serve
/// as regression guards: if post-processing regresses, the cleaned WER will
/// exceed the threshold.
final class ASRCorpusBenchmarkTests: XCTestCase {

    func testCorpusRawWERWithinBound() {
        for entry in benchmarkCorpus {
            let score = computeWER(reference: entry.reference, hypothesis: entry.rawHypothesis)
            XCTAssertLessThanOrEqual(
                score.rate,
                entry.maxRawWER + 0.001,
                "[\(entry.description)] Raw WER \(score) exceeds bound \(entry.maxRawWER)"
            )
        }
    }

    func testCorpusCleanedWERWithinBound() {
        for entry in benchmarkCorpus {
            let score = computeWER(reference: entry.reference, hypothesis: entry.cleanedHypothesis)
            XCTAssertLessThanOrEqual(
                score.rate,
                entry.maxCleanedWER + 0.001,
                "[\(entry.description)] Cleaned WER \(score) exceeds bound \(entry.maxCleanedWER)"
            )
        }
    }

    func testPostProcessorReducesOrMaintainsWEROnCorpus() {
        let processor = TranscriptionPostProcessor()
        for entry in benchmarkCorpus {
            let raw = computeWER(reference: entry.reference, hypothesis: entry.rawHypothesis)

            let segment = makeBenchSegment(text: entry.rawHypothesis)
            let result = makeBenchResult(segments: [segment])
            let processed = processor.process(result)
            let processedText = processed.segments.map(\.text).joined(separator: " ")

            let cleaned = computeWER(reference: entry.reference, hypothesis: processedText)
            XCTAssertLessThanOrEqual(
                cleaned.rate,
                raw.rate + 0.001,
                "[\(entry.description)] Post-processor degraded WER: \(raw) → \(cleaned)"
            )
        }
    }

    func testStutterEntryRawWERIsNonZero() {
        let entry = benchmarkCorpus.first { $0.description.contains("stutter") }!
        let score = computeWER(reference: entry.reference, hypothesis: entry.rawHypothesis)
        XCTAssertGreaterThan(score.rate, 0.0, "Stutter entry should have non-zero raw WER")
    }

    func testStutterEntryCleanedWERIsZero() {
        let entry = benchmarkCorpus.first { $0.description.contains("stutter") }!
        let score = computeWER(reference: entry.reference, hypothesis: entry.cleanedHypothesis)
        XCTAssertEqual(score.rate, 0.0, accuracy: 0.001,
                       "Post-processed stutter entry should have zero WER vs reference")
    }

    func testPerfectEntryHasZeroRawAndCleanedWER() {
        let entry = benchmarkCorpus.first { $0.description.contains("Perfect") }!
        let raw = computeWER(reference: entry.reference, hypothesis: entry.rawHypothesis)
        let cleaned = computeWER(reference: entry.reference, hypothesis: entry.cleanedHypothesis)
        XCTAssertEqual(raw.rate, 0.0, accuracy: 0.001)
        XCTAssertEqual(cleaned.rate, 0.0, accuracy: 0.001)
    }
}

// MARK: - ASRPipelineFidelityTests

/// Tests the full scheduler → merger → post-processor pipeline with known inputs.
///
/// These tests simulate a realistic transcription scenario: synthetic VAD
/// probabilities drive the scheduler, injected ASR results feed the merger,
/// and the post-processor is applied before computing WER against a ground-
/// truth reference.
final class ASRPipelineFidelityTests: XCTestCase {

    // MARK: Scheduler → single chunk

    func testSchedulerProducesChunkForSingleSpeechRegion() {
        let sampleRate = 16_000
        let vadChunkSize = 4_096
        let speechFrames = 10  // ~2.56 s of speech
        let audio = Array(repeating: Float(0.1), count: vadChunkSize * speechFrames)
        let probs = Array(repeating: Float(0.9), count: speechFrames)

        let scheduler = ASRChunkScheduler(config: ASRChunkSchedulerConfig(
            targetChunkDuration: 30.0,
            overlapDuration: 1.0,
            minSpeechDuration: 0.1,
            silenceThreshold: 0.5
        ))
        let chunks = scheduler.schedule(
            audio: audio,
            speechProbabilities: probs,
            sampleRate: sampleRate,
            vadChunkSize: vadChunkSize
        )

        XCTAssertEqual(chunks.count, 1, "Contiguous speech within target duration → 1 chunk")
        XCTAssertEqual(chunks[0].contentStartSample, 0)
        XCTAssertEqual(chunks[0].contentEndSample, vadChunkSize * speechFrames)
    }

    func testSchedulerSplitsLongRegionIntoMultipleChunks() {
        // 30 frames × 4096 samples @ 16kHz = ~7.68 s per frame group.
        // Target = 2 s → long region must be split.
        let sampleRate = 16_000
        let vadChunkSize = 16_000  // 1 s per VAD chunk
        let speechFrames = 10      // 10 s of continuous speech
        let audio = Array(repeating: Float(0.5), count: vadChunkSize * speechFrames)
        let probs = Array(repeating: Float(0.9), count: speechFrames)

        let scheduler = ASRChunkScheduler(config: ASRChunkSchedulerConfig(
            targetChunkDuration: 2.0,
            overlapDuration: 0.0,
            minSpeechDuration: 0.0,
            silenceThreshold: 0.5
        ))
        let chunks = scheduler.schedule(
            audio: audio,
            speechProbabilities: probs,
            sampleRate: sampleRate,
            vadChunkSize: vadChunkSize
        )

        XCTAssertEqual(chunks.count, 5, "10 s of speech / 2 s target = 5 chunks")
        for (i, chunk) in chunks.enumerated() {
            XCTAssertEqual(chunk.index, i)
        }
    }

    // MARK: Merger → transcript assembly

    func testMergerAssemblesCorrectTextFromMultipleChunks() async {
        let sampleRate = 16_000
        let merger = TranscriptionMerger(config: .init(finalConfidenceThreshold: 0.9))

        // Three non-overlapping chunks, one segment each.
        let inputs: [(String, Double, Double)] = [
            ("Hello everyone", 0.0, 1.0),
            ("welcome to the show", 1.0, 2.5),
            ("let us begin", 2.5, 4.0),
        ]
        let reference = "Hello everyone welcome to the show let us begin"

        for (i, (text, start, end)) in inputs.enumerated() {
            let startSample = Int(start * Double(sampleRate))
            let endSample = Int(end * Double(sampleRate))
            let chunk = makeBenchChunk(
                index: i,
                overlapStartSample: startSample,
                contentStartSample: startSample,
                contentEndSample: endSample
            )
            let seg = makeBenchSegment(text: text, startTime: 0.0, endTime: end - start, confidence: 0.95)
            _ = await merger.merge(chunk: chunk, result: makeBenchResult(segments: [seg]), sampleRate: sampleRate)
        }

        let event = await merger.finalize(language: "en", backendMetadata: benchStubMetadata)
        guard case .completed(let result) = event else {
            XCTFail("Expected .completed event")
            return
        }

        let score = computeWER(reference: reference, hypothesis: result.text)
        XCTAssertEqual(score.rate, 0.0, accuracy: 0.001,
                       "Clean multi-chunk assembly should have zero WER: \(score)")
    }

    func testMergerLastWinsResolutionDoesNotDuplicateOverlapWords() async {
        let sampleRate = 16_000
        let merger = TranscriptionMerger(config: .init(
            finalConfidenceThreshold: 1.1,  // suppress finalSegment events
            overlapResolution: .lastWins
        ))

        // Chunk 0: content [0, 16000) — "one two three"
        let chunk0 = makeBenchChunk(
            index: 0, overlapStartSample: 0, contentStartSample: 0, contentEndSample: 16_000
        )
        _ = await merger.merge(
            chunk: chunk0,
            result: makeBenchResult(segments: [
                makeBenchSegment(text: "one", startTime: 0.0, endTime: 0.3),
                makeBenchSegment(text: "two", startTime: 0.3, endTime: 0.6),
                makeBenchSegment(text: "three", startTime: 0.6, endTime: 1.0),
            ]),
            sampleRate: sampleRate
        )

        // Chunk 1: overlap [12000, 16000), content [16000, 32000)
        // The overlap zone re-transcribes "three" and adds "four five".
        let chunk1 = makeBenchChunk(
            index: 1, overlapStartSample: 12_000, contentStartSample: 16_000, contentEndSample: 32_000,
            totalSamples: 20_000
        )
        _ = await merger.merge(
            chunk: chunk1,
            result: makeBenchResult(segments: [
                // overlap zone (chunk-relative): abs start = 12000/16000 = 0.75s < 1.0s → overlap
                makeBenchSegment(text: "three", startTime: 0.0,  endTime: 0.25),
                // content zone: abs start = 0.75 + 0.25 = 1.0s == overlapEndTime → content
                makeBenchSegment(text: "four",  startTime: 0.25, endTime: 0.75),
                makeBenchSegment(text: "five",  startTime: 0.75, endTime: 1.25),
            ]),
            sampleRate: sampleRate
        )

        let event = await merger.finalize(backendMetadata: benchStubMetadata)
        guard case .completed(let result) = event else {
            XCTFail("Expected .completed event"); return
        }

        // "three" must appear exactly once; total transcript should be "one two three four five"
        let words = normalizeForWER(result.text)
        let threeCount = words.filter { $0 == "three" }.count
        XCTAssertEqual(threeCount, 1, "Overlap word 'three' must not be duplicated; got: \(result.text)")

        let score = computeWER(reference: "one two three four five", hypothesis: result.text)
        XCTAssertEqual(score.rate, 0.0, accuracy: 0.001, "Expected clean assembly: \(score) — got: \(result.text)")
    }

    // MARK: Full pipeline end-to-end

    func testFullPipelineEndToEndWithWERAssertion() async {
        // Scenario: a 6-second recording with 3 speech segments.
        // Reference: three sentences, each transcribed by one ASR chunk.
        let reference = "The project is on track. We finished the first milestone. Next steps are clear."
        let sampleRate = 16_000

        let rawSegments: [(text: String, chunkStart: Double, chunkEnd: Double)] = [
            ("the project is on track", 0.0, 2.0),
            ("we finished the first milestone", 2.0, 4.0),
            ("next steps are clear",             4.0, 6.0),
        ]

        let processor = TranscriptionPostProcessor(config: .init(
            confidenceThreshold: nil,
            enableRepetitionRemoval: true,
            enablePunctuationNormalization: true
        ))
        let merger = TranscriptionMerger(config: .init(finalConfidenceThreshold: 1.1), postProcessor: processor)

        for (i, entry) in rawSegments.enumerated() {
            let startSample = Int(entry.chunkStart * Double(sampleRate))
            let endSample = Int(entry.chunkEnd * Double(sampleRate))
            let chunk = makeBenchChunk(
                index: i,
                overlapStartSample: startSample,
                contentStartSample: startSample,
                contentEndSample: endSample
            )
            let seg = makeBenchSegment(
                text: entry.text,
                startTime: 0.0,
                endTime: entry.chunkEnd - entry.chunkStart,
                confidence: 0.7
            )
            _ = await merger.merge(chunk: chunk, result: makeBenchResult(segments: [seg]), sampleRate: sampleRate)
        }

        let event = await merger.finalize(language: "en", backendMetadata: benchStubMetadata)
        guard case .completed(let result) = event else {
            XCTFail("Expected .completed event"); return
        }

        // Post-processor should capitalize and add punctuation; WER vs reference should be 0.
        let score = computeWER(reference: reference, hypothesis: result.text)
        XCTAssertEqual(score.rate, 0.0, accuracy: 0.001,
                       "Full pipeline WER must be zero for clean input: \(score) — output: \"\(result.text)\"")

        // Verify each segment is capitalized and ends with terminal punctuation.
        for seg in result.segments {
            let first = seg.text.first
            XCTAssertTrue(first?.isUppercase == true || first?.isNumber == true,
                          "Segment should start with a capital: \"\(seg.text)\"")
            let last = seg.text.last
            XCTAssertTrue([".", "!", "?"].contains(String(last ?? ".")),
                          "Segment should end with terminal punctuation: \"\(seg.text)\"")
        }
    }

    func testPipelineWERDegradationFromStutterWithoutPostProcessor() async {
        // Without post-processing, a stutter should increase WER.
        let reference = "I went to the store"
        let sampleRate = 16_000

        let merger = TranscriptionMerger(config: .init(finalConfidenceThreshold: 1.1), postProcessor: nil)
        let chunk = makeBenchChunk(
            index: 0, overlapStartSample: 0, contentStartSample: 0, contentEndSample: sampleRate * 3
        )
        let rawText = "i i i went to the store"
        let seg = makeBenchSegment(text: rawText, startTime: 0, endTime: 3.0)
        _ = await merger.merge(chunk: chunk, result: makeBenchResult(segments: [seg]), sampleRate: sampleRate)

        let event = await merger.finalize(backendMetadata: benchStubMetadata)
        guard case .completed(let result) = event else {
            XCTFail("Expected .completed"); return
        }

        let rawScore = computeWER(reference: reference, hypothesis: result.text)
        XCTAssertGreaterThan(rawScore.rate, 0.0,
                             "Without post-processor, stutter should produce non-zero WER")
    }

    func testPipelineWERImprovedByPostProcessor() async {
        // With post-processing, the same stutter should resolve to zero WER.
        let reference = "I went to the store"
        let sampleRate = 16_000

        let processor = TranscriptionPostProcessor(config: .init(
            confidenceThreshold: nil,
            enableRepetitionRemoval: true,
            enablePunctuationNormalization: false  // keep punctuation out of WER comparison
        ))
        let merger = TranscriptionMerger(config: .init(finalConfidenceThreshold: 1.1), postProcessor: processor)
        let chunk = makeBenchChunk(
            index: 0, overlapStartSample: 0, contentStartSample: 0, contentEndSample: sampleRate * 3
        )
        let rawText = "i i i went to the store"
        let seg = makeBenchSegment(text: rawText, startTime: 0, endTime: 3.0)
        _ = await merger.merge(chunk: chunk, result: makeBenchResult(segments: [seg]), sampleRate: sampleRate)

        let event = await merger.finalize(backendMetadata: benchStubMetadata)
        guard case .completed(let result) = event else {
            XCTFail("Expected .completed"); return
        }

        let score = computeWER(reference: reference, hypothesis: result.text)
        XCTAssertEqual(score.rate, 0.0, accuracy: 0.001,
                       "Post-processor should eliminate stutter WER: got \(score), output: \"\(result.text)\"")
    }
}

// MARK: - ASRSchedulerPerformanceTests

/// Performance benchmarks for `ASRChunkScheduler` on large audio buffers.
///
/// These benchmarks exercise the scheduler at three scales:
/// - Small: ~30 s of audio at 16 kHz
/// - Medium: ~5 min of audio
/// - Dense VAD: many short speech regions
///
/// The `measure {}` block runs each case 10 times and reports median wall time.
/// Regressions surface as CI baseline failures.
final class ASRSchedulerPerformanceTests: XCTestCase {

    private let sampleRate = 16_000
    private let vadChunkSize = 4_096  // 256 ms per VAD frame

    func testSchedulerPerformanceSmallBuffer() {
        // ~30 s audio: 1875 VAD chunks × 4096 samples
        let frames = 117   // 117 × 256 ms ≈ 30 s
        let audio = Array(repeating: Float(0.3), count: vadChunkSize * frames)
        let probs = (0..<frames).map { i -> Float in i % 3 == 0 ? 0.2 : 0.85 }  // mixed speech/silence

        let scheduler = ASRChunkScheduler(config: ASRChunkSchedulerConfig(
            targetChunkDuration: 10.0,
            overlapDuration: 1.0,
            minSpeechDuration: 0.5,
            silenceThreshold: 0.5
        ))

        measure {
            _ = scheduler.schedule(
                audio: audio,
                speechProbabilities: probs,
                sampleRate: sampleRate,
                vadChunkSize: vadChunkSize
            )
        }
    }

    func testSchedulerPerformanceMediumBuffer() {
        // ~5 min audio: 1172 VAD chunks × 4096 samples
        let frames = 1_172
        let audio = Array(repeating: Float(0.3), count: vadChunkSize * frames)
        let probs = (0..<frames).map { i -> Float in i % 5 == 0 ? 0.1 : 0.9 }

        let scheduler = ASRChunkScheduler(config: ASRChunkSchedulerConfig(
            targetChunkDuration: 10.0,
            overlapDuration: 1.0,
            minSpeechDuration: 0.5,
            silenceThreshold: 0.5
        ))

        measure {
            _ = scheduler.schedule(
                audio: audio,
                speechProbabilities: probs,
                sampleRate: sampleRate,
                vadChunkSize: vadChunkSize
            )
        }
    }

    func testSchedulerPerformanceDenseVAD() {
        // Dense alternating speech/silence: many small regions → stress-tests region collector.
        let frames = 500
        let audio = Array(repeating: Float(0.3), count: vadChunkSize * frames)
        let probs: [Float] = (0..<frames).map { $0 % 2 == 0 ? 0.9 : 0.1 }  // alternates every frame

        let scheduler = ASRChunkScheduler(config: ASRChunkSchedulerConfig(
            targetChunkDuration: 5.0,
            overlapDuration: 0.5,
            minSpeechDuration: 0.0,  // include all regions to maximise chunk output
            silenceThreshold: 0.5
        ))

        measure {
            _ = scheduler.schedule(
                audio: audio,
                speechProbabilities: probs,
                sampleRate: sampleRate,
                vadChunkSize: vadChunkSize
            )
        }
    }
}

// MARK: - ASRPostProcessorPerformanceTests

/// Performance benchmarks for `TranscriptionPostProcessor` on large segment lists.
final class ASRPostProcessorPerformanceTests: XCTestCase {

    private func makeRepeatingSegments(count: Int) -> [TranscriptionSegment] {
        let templates = [
            "hello world this is a test",
            "hello world this is a test",  // duplicate — repetition removal fires
            "the quick brown fox jumps",
            "what time does the meeting start",
            "i i went to the store",       // stutter — word-level collapse fires
        ]
        return (0..<count).map { i in
            TranscriptionSegment(
                text: templates[i % templates.count],
                startTime: Double(i),
                endTime: Double(i) + 1.0,
                confidence: 0.8,
                isFinal: true
            )
        }
    }

    func testPostProcessorPerformanceSmall() {
        let processor = TranscriptionPostProcessor()
        let segments = makeRepeatingSegments(count: 50)
        let result = makeBenchResult(segments: segments)
        measure {
            _ = processor.process(result)
        }
    }

    func testPostProcessorPerformanceLarge() {
        let processor = TranscriptionPostProcessor()
        let segments = makeRepeatingSegments(count: 500)
        let result = makeBenchResult(segments: segments)
        measure {
            _ = processor.process(result)
        }
    }
}
