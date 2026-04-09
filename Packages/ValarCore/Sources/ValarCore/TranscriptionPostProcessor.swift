import Foundation
import ValarModelKit

// MARK: - TranscriptionPostProcessorConfig

/// Configuration for the `TranscriptionPostProcessor` pipeline.
///
/// All stages are opt-in. The pipeline runs stages in this fixed order:
/// 1. Confidence threshold filter
/// 2. Repetition cleanup
/// 3. Punctuation normalization
public struct TranscriptionPostProcessorConfig: Sendable, Equatable {

    /// Drop segments whose `confidence` is below this value.
    /// When `nil`, no confidence-based filtering is applied.
    /// Segments with no confidence metadata are always retained.
    public var confidenceThreshold: Float?

    /// Remove consecutive duplicate segments and within-segment word-level stutters.
    public var enableRepetitionRemoval: Bool

    /// Capitalize the first word of each segment and ensure every segment
    /// ends with terminal punctuation (period, !, or ?).
    public var enablePunctuationNormalization: Bool

    public init(
        confidenceThreshold: Float? = nil,
        enableRepetitionRemoval: Bool = true,
        enablePunctuationNormalization: Bool = true
    ) {
        self.confidenceThreshold = confidenceThreshold
        self.enableRepetitionRemoval = enableRepetitionRemoval
        self.enablePunctuationNormalization = enablePunctuationNormalization
    }

    /// Reasonable defaults: repetition removal + punctuation normalization, no confidence gate.
    public static let `default` = TranscriptionPostProcessorConfig()

    /// Pass-through: no stages run.
    public static let passthrough = TranscriptionPostProcessorConfig(
        confidenceThreshold: nil,
        enableRepetitionRemoval: false,
        enablePunctuationNormalization: false
    )
}

// MARK: - TranscriptionPostProcessor

/// Applies a configurable post-processing pipeline to a `RichTranscriptionResult`.
///
/// Each pipeline stage is independently togglable. Stages run in a fixed order
/// (confidence filter → repetition cleanup → punctuation normalization) so that
/// each stage operates on already-cleaned input.
///
/// **Typical use:**
/// ```swift
/// let processor = TranscriptionPostProcessor()
/// let cleaned = processor.process(rawResult)
/// ```
///
/// Wire into `TranscriptionMerger` as an optional post-pass:
/// ```swift
/// let merger = TranscriptionMerger(postProcessor: TranscriptionPostProcessor())
/// let completed = await merger.finalize(backendMetadata: metadata)
/// ```
public struct TranscriptionPostProcessor: Sendable {

    public let config: TranscriptionPostProcessorConfig

    public init(config: TranscriptionPostProcessorConfig = .default) {
        self.config = config
    }

    // MARK: - Public API

    /// Run the configured pipeline stages and return a cleaned `RichTranscriptionResult`.
    ///
    /// The top-level `text` field of the returned result is re-derived from the
    /// cleaned segments so that it stays consistent.
    public func process(_ result: RichTranscriptionResult) -> RichTranscriptionResult {
        var segments = result.segments

        if let threshold = config.confidenceThreshold {
            segments = filterLowConfidence(segments, threshold: threshold)
        }

        if config.enableRepetitionRemoval {
            segments = removeRepetitions(segments)
        }

        if config.enablePunctuationNormalization {
            segments = normalizePunctuation(segments)
        }

        let cleanedText = segments.map(\.text).joined(separator: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)

        return RichTranscriptionResult(
            text: cleanedText,
            language: result.language,
            durationSeconds: result.durationSeconds,
            segments: segments,
            words: result.words,
            alignmentReference: result.alignmentReference,
            backendMetadata: result.backendMetadata
        )
    }

    // MARK: - Stage 1: Confidence threshold filter

    /// Drop segments whose reported confidence is below `threshold`.
    ///
    /// Segments that carry no confidence value (i.e., `confidence == nil`) are
    /// always kept — absence of metadata is not evidence of low quality.
    func filterLowConfidence(
        _ segments: [TranscriptionSegment],
        threshold: Float
    ) -> [TranscriptionSegment] {
        segments.filter { segment in
            guard let confidence = segment.confidence else { return true }
            return confidence >= threshold
        }
    }

    // MARK: - Stage 2: Repetition cleanup

    /// Remove repetitions that are common ASR hallucination artefacts.
    ///
    /// Two kinds of repetition are addressed:
    ///
    /// 1. **Consecutive duplicate segments** — adjacent segments whose trimmed
    ///    text is identical are collapsed to one copy.
    ///
    /// 2. **Within-segment word stutters** — runs of the same word token are
    ///    collapsed (e.g., "I I I went" → "I went"). Comparison is
    ///    case-insensitive so "The the the" is also reduced.
    func removeRepetitions(
        _ segments: [TranscriptionSegment]
    ) -> [TranscriptionSegment] {
        // Step 1: collapse consecutive identical segments.
        var deduped: [TranscriptionSegment] = []
        for segment in segments {
            let trimmed = segment.text.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed != deduped.last?.text.trimmingCharacters(in: .whitespacesAndNewlines) {
                deduped.append(segment)
            }
        }

        // Step 2: collapse within-segment word stutters.
        return deduped.map { segment in
            let deStuttered = collapseWordStutters(in: segment.text)
            guard deStuttered != segment.text else { return segment }
            return TranscriptionSegment(
                text: deStuttered,
                startTime: segment.startTime,
                endTime: segment.endTime,
                confidence: segment.confidence,
                isFinal: segment.isFinal,
                chunkIndex: segment.chunkIndex
            )
        }
    }

    /// Collapse consecutive identical word tokens in `text`.
    ///
    /// Tokens are separated by whitespace. Comparison is case-insensitive;
    /// the first occurrence of each run is preserved verbatim.
    private func collapseWordStutters(in text: String) -> String {
        let words = text.components(separatedBy: .whitespaces).filter { !$0.isEmpty }
        var result: [String] = []
        for word in words {
            if word.lowercased() != result.last?.lowercased() {
                result.append(word)
            }
        }
        return result.joined(separator: " ")
    }

    // MARK: - Stage 3: Punctuation normalization

    /// Ensure each segment starts with a capital letter and ends with terminal punctuation.
    ///
    /// Rules applied per segment:
    /// - Leading whitespace is stripped.
    /// - The first character is uppercased.
    /// - If the segment does not already end with `.`, `!`, or `?`, a period
    ///   is appended — unless the text appears to be a question (see below),
    ///   in which case `?` is appended instead.
    ///
    /// Question heuristic: a segment is treated as a question when its first
    /// word (case-insensitive) is one of the common English interrogative or
    /// auxiliary-question starters (who, what, where, when, why, how, is,
    /// are, was, were, do, does, did, can, could, should, would, have, has,
    /// had, may, might, shall, will).
    func normalizePunctuation(
        _ segments: [TranscriptionSegment]
    ) -> [TranscriptionSegment] {
        segments.compactMap { segment in
            let trimmed = segment.text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { return nil }

            let normalized = applyPunctuationRules(to: trimmed)
            guard normalized != segment.text else { return segment }

            return TranscriptionSegment(
                text: normalized,
                startTime: segment.startTime,
                endTime: segment.endTime,
                confidence: segment.confidence,
                isFinal: segment.isFinal,
                chunkIndex: segment.chunkIndex
            )
        }
    }

    private static let questionStarters: Set<String> = [
        "who", "what", "where", "when", "why", "how",
        "is", "are", "was", "were",
        "do", "does", "did",
        "can", "could", "should", "would",
        "have", "has", "had",
        "may", "might", "shall", "will"
    ]

    private static let terminalPunctuation: Set<Character> = [".", "!", "?"]

    private func applyPunctuationRules(to text: String) -> String {
        var result = text

        // Capitalize first character.
        if let first = result.unicodeScalars.first, CharacterSet.lowercaseLetters.contains(first) {
            result = result.prefix(1).uppercased() + result.dropFirst()
        }

        // Append terminal punctuation if absent.
        if let last = result.last, !Self.terminalPunctuation.contains(last) {
            let firstWord = result
                .components(separatedBy: .whitespaces)
                .first { !$0.isEmpty }?
                .lowercased()
                .trimmingCharacters(in: CharacterSet.letters.inverted) ?? ""

            let punctuation: Character = Self.questionStarters.contains(firstWord) ? "?" : "."
            result.append(punctuation)
        }

        return result
    }
}

