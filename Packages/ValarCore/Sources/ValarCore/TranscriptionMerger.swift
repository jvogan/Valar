import Foundation
import ValarModelKit

// MARK: - TranscriptionMerger

/// Merges per-chunk ASR results into a unified, de-duplicated event stream.
///
/// ### Streaming usage
///
/// Feed each `(ASRChunk, RichTranscriptionResult)` pair into
/// ``merge(chunk:result:sampleRate:)`` as each ASR chunk completes.
/// Call ``finalize(language:backendMetadata:)`` once all chunks are done
/// to receive the terminal `.completed` event.
///
/// ```swift
/// let merger = TranscriptionMerger()
/// for chunk in asrChunks {
///     let result = try await asrModel.transcribe(chunk.samples)
///     let events = await merger.merge(chunk: chunk, result: result, sampleRate: 16_000)
///     for event in events { eventStream.yield(event) }
/// }
/// let completed = await merger.finalize(backendMetadata: metadata)
/// eventStream.yield(completed)
/// ```
///
/// ### Batch usage
///
/// For one-shot scenarios where all chunk results are already available,
/// use ``merge(chunks:)`` for a simple concatenation without overlap resolution.
///
/// ### Overlap de-duplication
///
/// `ASRChunkScheduler` prepends an overlap region to each chunk so the ASR
/// decoder has context from the preceding audio. `TranscriptionMerger` detects
/// segments that fall in the overlap zone and resolves conflicts using the
/// configured ``OverlapResolution`` policy:
///
/// - `.lastWins` – the later chunk's text always replaces the earlier version.
/// - `.confidenceWins` – the segment with the higher confidence score is kept;
///   ties go to the later chunk.
public actor TranscriptionMerger {

    // MARK: - OverlapResolution

    /// Determines how conflicting text in the overlap zone between consecutive chunks is resolved.
    public enum OverlapResolution: Sendable, Equatable {
        /// The later chunk's text always wins; previously committed overlap text is replaced.
        case lastWins
        /// The segment with the higher confidence score is kept; ties go to the later chunk.
        case confidenceWins
    }

    // MARK: - Config

    /// Configuration for `TranscriptionMerger`.
    public struct Config: Sendable {
        /// Confidence threshold (0–1). Segments at or above this value are emitted as
        /// `.finalSegment`; below it they are included only in the running `.partial` event.
        /// Default: `0.8`.
        public let finalConfidenceThreshold: Float

        /// Policy for resolving text conflicts in the overlap zone. Default: `.lastWins`.
        public let overlapResolution: OverlapResolution

        public init(
            finalConfidenceThreshold: Float = 0.8,
            overlapResolution: OverlapResolution = .lastWins
        ) {
            self.finalConfidenceThreshold = finalConfidenceThreshold
            self.overlapResolution = overlapResolution
        }
    }

    // MARK: - State

    private let config: Config
    /// Optional post-processor applied during ``finalize(language:backendMetadata:)``
    /// and ``merge(chunks:)``.
    private let postProcessor: TranscriptionPostProcessor?
    /// All committed segments, kept sorted by absolute start time.
    private var committedSegments: [TranscriptionSegment] = []

    // MARK: - Init

    public init(config: Config = .init(), postProcessor: TranscriptionPostProcessor? = nil) {
        self.config = config
        self.postProcessor = postProcessor
    }

    // MARK: - Streaming API

    /// Process one chunk's ASR result and return the events to forward to consumers.
    ///
    /// Segments whose absolute start time falls within the overlap region
    /// (i.e., before `chunk.contentStartSample`) are subject to overlap resolution.
    /// Segments in the content region are committed immediately; high-confidence
    /// ones also generate a `.finalSegment` event.
    ///
    /// - Parameters:
    ///   - chunk: The `ASRChunk` that was passed to the ASR model.
    ///   - result: The `RichTranscriptionResult` returned by the model for that chunk.
    ///   - sampleRate: Sample rate of the audio buffer in Hz (e.g. 16 000).
    /// - Returns: Ordered `SpeechRecognitionEvent` values to emit. Zero or more
    ///   `.finalSegment` events for high-confidence content-zone segments, followed
    ///   by a single `.partial` event carrying the full running transcript.
    public func merge(
        chunk: ASRChunk,
        result: RichTranscriptionResult,
        sampleRate: Int
    ) -> [SpeechRecognitionEvent] {
        guard sampleRate > 0 else { return [] }

        guard !result.segments.isEmpty else {
            return partialEventIfNeeded(chunkIndex: chunk.index)
        }

        let sampleRateDouble = Double(sampleRate)
        let chunkOriginTime  = Double(chunk.overlapStartSample) / sampleRateDouble
        let overlapEndTime   = Double(chunk.contentStartSample) / sampleRateDouble

        // Remap each segment from chunk-relative timing to absolute timing.
        var overlapZone: [TranscriptionSegment] = []
        var contentZone: [TranscriptionSegment] = []

        for seg in result.segments {
            let absStart = chunkOriginTime + (seg.startTime ?? 0.0)
            let absEnd   = chunkOriginTime + (seg.endTime   ?? (seg.startTime ?? 0.0) + 0.001)

            let absolute = TranscriptionSegment(
                text:       seg.text,
                startTime:  absStart,
                endTime:    absEnd,
                confidence: seg.confidence,
                isFinal:    seg.isFinal,
                chunkIndex: chunk.index
            )

            if absStart < overlapEndTime {
                overlapZone.append(absolute)
            } else {
                contentZone.append(absolute)
            }
        }

        // Resolve the overlap zone before appending new content.
        applyOverlapResolution(
            overlapSegments: overlapZone,
            chunkOriginTime: chunkOriginTime,
            overlapEndTime:  overlapEndTime
        )

        // Accumulate content-zone segments and build events.
        var events: [SpeechRecognitionEvent] = []
        for seg in contentZone {
            committedSegments.append(seg)
            let confidence = seg.confidence ?? 0.0
            if confidence >= config.finalConfidenceThreshold {
                events.append(.finalSegment(seg))
            }
        }

        // Append a .partial with the current running transcript.
        events.append(contentsOf: partialEventIfNeeded(chunkIndex: chunk.index))
        return events
    }

    /// Build the terminal `.completed` event from all accumulated segments.
    ///
    /// Call this once every chunk has been processed. The `postProcessor`, if set,
    /// is applied to the assembled result before the event is created.
    ///
    /// - Parameters:
    ///   - language: Optional BCP-47 language tag to attach to the result.
    ///   - backendMetadata: Provenance information for the ASR backend.
    public func finalize(
        language: String? = nil,
        backendMetadata: BackendMetadata
    ) -> SpeechRecognitionEvent {
        let text     = buildTranscriptText()
        let duration = committedSegments.last?.endTime

        var assembled = RichTranscriptionResult(
            text:            text,
            language:        language,
            durationSeconds: duration,
            segments:        committedSegments,
            backendMetadata: backendMetadata
        )

        if let processor = postProcessor {
            assembled = processor.process(assembled)
        }

        return .completed(assembled)
    }

    // MARK: - Batch API

    /// Merge an ordered sequence of per-chunk results into a single `RichTranscriptionResult`.
    ///
    /// This is a one-shot batch alternative to the streaming API. No overlap resolution is
    /// performed — segments from each chunk are concatenated in order. The `postProcessor`,
    /// if set, is applied to the assembled result.
    ///
    /// - Parameter chunks: Ordered array of results, one per ASR chunk.
    /// - Returns: The merged result, or `nil` when `chunks` is empty.
    public func merge(chunks: [RichTranscriptionResult]) -> RichTranscriptionResult? {
        guard let first = chunks.first else { return nil }

        let allSegments = chunks.flatMap(\.segments)
        let mergedText  = chunks
            .map { $0.text.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            .joined(separator: " ")

        let totalDuration = chunks.compactMap(\.durationSeconds).reduce(0, +)

        var merged = RichTranscriptionResult(
            text:            mergedText,
            language:        first.language,
            durationSeconds: totalDuration == 0 ? nil : totalDuration,
            segments:        allSegments,
            words:           nil,
            alignmentReference: nil,
            backendMetadata: first.backendMetadata
        )

        if let processor = postProcessor {
            merged = processor.process(merged)
        }

        return merged
    }

    // MARK: - Private helpers

    /// Apply the configured overlap-resolution policy to segments in the overlap zone.
    private func applyOverlapResolution(
        overlapSegments: [TranscriptionSegment],
        chunkOriginTime: Double,
        overlapEndTime:  Double
    ) {
        guard !overlapSegments.isEmpty else { return }

        switch config.overlapResolution {
        case .lastWins:
            // Remove any previously committed segment that intersects the overlap window
            // [chunkOriginTime, overlapEndTime). Using interval intersection rather than
            // a simple start-time check handles segments that begin before the overlap
            // window but whose end time reaches into it (e.g. a segment at [0.0, 0.8s]
            // when the overlap window starts at 0.75s).
            committedSegments.removeAll { seg in
                let segStart = seg.startTime ?? 0.0
                let segEnd   = seg.endTime   ?? segStart
                return segStart < overlapEndTime && segEnd > chunkOriginTime
            }
            committedSegments.append(contentsOf: overlapSegments)
            committedSegments.sort { ($0.startTime ?? 0) < ($1.startTime ?? 0) }

        case .confidenceWins:
            for newSeg in overlapSegments {
                let newStart = newSeg.startTime ?? 0.0
                let newEnd   = newSeg.endTime   ?? newStart
                let newConf  = newSeg.confidence ?? 0.0

                // Find all previously committed segments whose time range intersects newSeg.
                let conflicts = committedSegments.filter { existing in
                    let exStart = existing.startTime ?? 0.0
                    let exEnd   = existing.endTime   ?? exStart
                    return exStart < newEnd && exEnd > newStart
                }

                let existingMaxConf = conflicts.compactMap(\.confidence).max() ?? 0.0
                guard newConf >= existingMaxConf else { continue }

                // New segment wins: remove conflicting segments and insert newSeg.
                committedSegments.removeAll { existing in
                    let exStart = existing.startTime ?? 0.0
                    let exEnd   = existing.endTime   ?? exStart
                    return exStart < newEnd && exEnd > newStart
                }
                committedSegments.append(newSeg)
                committedSegments.sort { ($0.startTime ?? 0) < ($1.startTime ?? 0) }
            }
        }
    }

    /// Return a `.partial` event if there is any committed text, otherwise an empty array.
    private func partialEventIfNeeded(chunkIndex: Int) -> [SpeechRecognitionEvent] {
        let text = buildTranscriptText()
        guard !text.isEmpty else { return [] }
        let seg = TranscriptionSegment(
            text:       text,
            startTime:  committedSegments.first?.startTime,
            endTime:    committedSegments.last?.endTime,
            confidence: nil,
            isFinal:    false,
            chunkIndex: chunkIndex
        )
        return [.partial(seg)]
    }

    /// Join committed segment texts into a single space-separated string.
    private func buildTranscriptText() -> String {
        committedSegments
            .map { $0.text.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
            .joined(separator: " ")
    }
}
