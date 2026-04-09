import Foundation

// MARK: - ASRChunkSchedulerConfig

/// Configuration for `ASRChunkScheduler`.
public struct ASRChunkSchedulerConfig: Sendable, Equatable {
    /// Target duration (seconds) for each output ASR chunk.
    /// Chunks are split if the speech region exceeds this. Default: 10 s.
    public let targetChunkDuration: Double

    /// Duration (seconds) of audio from before the speech content to prepend
    /// to each chunk for decoder context continuity. Default: 1 s.
    public let overlapDuration: Double

    /// Minimum speech duration (seconds) for a contiguous speech region to be
    /// included in the output. Regions shorter than this are dropped. Default: 0.5 s.
    public let minSpeechDuration: Double

    /// Speech probability below which a VAD chunk is treated as silence.
    /// Applied to the per-chunk speech probability values passed to `schedule`.
    /// Default: 0.5.
    public let silenceThreshold: Float

    public init(
        targetChunkDuration: Double = 10.0,
        overlapDuration: Double = 1.0,
        minSpeechDuration: Double = 0.5,
        silenceThreshold: Float = 0.5
    ) {
        self.targetChunkDuration = targetChunkDuration
        self.overlapDuration = overlapDuration
        self.minSpeechDuration = minSpeechDuration
        self.silenceThreshold = silenceThreshold
    }
}

// MARK: - ASRChunk

/// A single audio segment produced by `ASRChunkScheduler`, ready for one-shot ASR.
public struct ASRChunk: Sendable, Equatable {
    /// Zero-based position of this chunk in the output sequence.
    public let index: Int

    /// Mono Float32 samples for this chunk, including context overlap at the start.
    public let samples: [Float]

    /// Start of the context-overlap region in the original audio buffer.
    /// Equal to `contentStartSample` when there is no overlap (e.g. first chunk or no overlap configured).
    public let overlapStartSample: Int

    /// First sample of speech content in the original buffer (end of overlap region).
    public let contentStartSample: Int

    /// One past the last sample of speech content in the original buffer.
    public let contentEndSample: Int

    public init(
        index: Int,
        samples: [Float],
        overlapStartSample: Int,
        contentStartSample: Int,
        contentEndSample: Int
    ) {
        self.index = index
        self.samples = samples
        self.overlapStartSample = overlapStartSample
        self.contentStartSample = contentStartSample
        self.contentEndSample = contentEndSample
    }
}

// MARK: - ASRChunkScheduler

/// Schedules audio chunks for one-shot upstream ASR transcription calls.
///
/// Given a full audio buffer and per-chunk speech probabilities from a voice activity
/// detector, `ASRChunkScheduler` produces an ordered sequence of `ASRChunk` values
/// ready to pass directly to an ASR model. Each chunk:
///
/// * Contains at most `config.targetChunkDuration` seconds of speech content.
/// * Prepends up to `config.overlapDuration` seconds of audio before the speech
///   boundary for decoder context continuity.
/// * Excludes VAD chunks whose speech probability is below `config.silenceThreshold`.
/// * Excludes contiguous speech runs shorter than `config.minSpeechDuration`.
///
/// **Typical use:**
/// ```swift
/// let scheduler = ASRChunkScheduler()
/// let chunks = scheduler.schedule(
///     audio: audioBuffer,
///     speechProbabilities: vadResults.map(\.speechProbability),
///     sampleRate: 16_000,
///     vadChunkSize: 4096
/// )
/// for chunk in chunks {
///     let transcript = try await asrModel.transcribe(samples: chunk.samples, sampleRate: 16_000)
/// }
/// ```
public struct ASRChunkScheduler: Sendable {

    public let config: ASRChunkSchedulerConfig

    public init(config: ASRChunkSchedulerConfig = .init()) {
        self.config = config
    }

    // MARK: - Public API

    /// Produce ordered ASR chunks from a full audio buffer and per-VAD-chunk speech probabilities.
    ///
    /// - Parameters:
    ///   - audio: Full mono Float32 audio at `sampleRate` Hz.
    ///   - speechProbabilities: Per-VAD-chunk speech probability (0–1), in temporal order.
    ///     Typically `vadResults.map(\.speechProbability)`.
    ///   - sampleRate: Sample rate of `audio` in Hz (e.g. 16 000).
    ///   - vadChunkSize: Number of `audio` samples per VAD chunk (e.g. 4096 at 16 kHz → 256 ms).
    /// - Returns: Ordered, non-overlapping speech chunks ready for ASR.
    ///   Empty if all audio is silent or all speech regions are shorter than
    ///   `config.minSpeechDuration`.
    public func schedule(
        audio: [Float],
        speechProbabilities: [Float],
        sampleRate: Int,
        vadChunkSize: Int
    ) -> [ASRChunk] {
        guard !audio.isEmpty, !speechProbabilities.isEmpty, sampleRate > 0, vadChunkSize > 0 else {
            return []
        }

        let targetSamples = max(1, Int((config.targetChunkDuration * Double(sampleRate)).rounded()))
        let overlapSamples = max(0, Int((config.overlapDuration * Double(sampleRate)).rounded()))
        let minSpeechSamples = max(0, Int((config.minSpeechDuration * Double(sampleRate)).rounded()))

        let speechRegions = collectSpeechRegions(
            speechProbabilities: speechProbabilities,
            vadChunkSize: vadChunkSize,
            audioLength: audio.count,
            minSpeechSamples: minSpeechSamples
        )

        return buildChunks(
            audio: audio,
            speechRegions: speechRegions,
            targetSamples: targetSamples,
            overlapSamples: overlapSamples
        )
    }

    // MARK: - Private helpers

    /// Identify contiguous speech regions from per-frame probabilities, filtered by minimum length.
    ///
    /// A "speech region" is a maximal run of consecutive VAD frames whose `speechProbability`
    /// is at or above `config.silenceThreshold`. Each region is expressed as a half-open
    /// sample range `[start, end)` in the original audio buffer.
    private func collectSpeechRegions(
        speechProbabilities: [Float],
        vadChunkSize: Int,
        audioLength: Int,
        minSpeechSamples: Int
    ) -> [(start: Int, end: Int)] {
        var regions: [(start: Int, end: Int)] = []
        var regionStart: Int? = nil

        for (i, prob) in speechProbabilities.enumerated() {
            let isSpeech = prob >= config.silenceThreshold
            if isSpeech {
                if regionStart == nil {
                    regionStart = i * vadChunkSize
                }
            } else {
                if let start = regionStart {
                    let end = min(i * vadChunkSize, audioLength)
                    if end - start >= minSpeechSamples {
                        regions.append((start: start, end: end))
                    }
                    regionStart = nil
                }
            }
        }

        // Close any region still open at the end of the probabilities array.
        if let start = regionStart {
            let end = audioLength
            if end - start >= minSpeechSamples {
                regions.append((start: start, end: end))
            }
        }

        return regions
    }

    /// Slice speech regions into target-sized chunks, prepending overlap from preceding audio.
    ///
    /// Long regions are split at `targetSamples` boundaries. Each resulting chunk's
    /// sample array starts at `max(0, contentStart - overlapSamples)` to include
    /// decoder context from the audio immediately preceding the speech content.
    private func buildChunks(
        audio: [Float],
        speechRegions: [(start: Int, end: Int)],
        targetSamples: Int,
        overlapSamples: Int
    ) -> [ASRChunk] {
        var chunks: [ASRChunk] = []

        for region in speechRegions {
            var chunkStart = region.start
            while chunkStart < region.end {
                let chunkEnd = min(chunkStart + targetSamples, region.end)
                let overlapStart = max(0, chunkStart - overlapSamples)
                let samples = Array(audio[overlapStart..<chunkEnd])

                chunks.append(
                    ASRChunk(
                        index: chunks.count,
                        samples: samples,
                        overlapStartSample: overlapStart,
                        contentStartSample: chunkStart,
                        contentEndSample: chunkEnd
                    )
                )

                chunkStart = chunkEnd
            }
        }

        return chunks
    }
}
