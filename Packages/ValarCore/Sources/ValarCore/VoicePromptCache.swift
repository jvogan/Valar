import Foundation

/// In-memory FIFO cache for decoded voice reference-audio payloads.
///
/// Stores the result of decoding a voice's reference audio asset so that repeated
/// TTS synthesis requests for the same voice skip the expensive decode step.
/// The cache is keyed by voice UUID.
///
/// - Concurrency: `VoicePromptCache` is an actor — all reads and writes are serialised.
/// - Eviction: When `maxEntries` is reached the oldest entry (by insertion order) is evicted.
///   Re-storing an already-cached entry refreshes its payload without changing its position.
/// - Invalidation: Call `invalidate(voiceID:)` whenever a `VoiceLibraryRecord` is saved or
///   deleted so callers always observe fresh data.
public actor VoicePromptCache {

    // MARK: - Payload

    /// Cached data for one voice reference clip.
    public struct Payload: Sendable, Equatable {
        /// Decoded mono PCM samples ready for inference.
        public let monoSamples: [Float]
        /// Sample rate of `monoSamples` in Hz.
        public let sampleRate: Double
        /// Trimmed reference transcript for the voice clip.
        public let referenceTranscript: String
        /// Trimmed `voicePrompt` text, or `nil` if absent.
        public let normalizedVoicePrompt: String?

        public init(
            monoSamples: [Float],
            sampleRate: Double,
            referenceTranscript: String,
            normalizedVoicePrompt: String? = nil
        ) {
            self.monoSamples = monoSamples
            self.sampleRate = sampleRate
            self.referenceTranscript = referenceTranscript
            self.normalizedVoicePrompt = normalizedVoicePrompt
        }
    }

    // MARK: - Storage

    private var store: [UUID: Payload] = [:]
    private var insertionOrder: [UUID] = []

    /// Maximum number of entries before the oldest entry is evicted.
    public let maxEntries: Int

    public init(maxEntries: Int = 64) {
        self.maxEntries = maxEntries
    }

    // MARK: - Read

    /// Returns the cached `Payload` for `voiceID`, or `nil` on a cache miss.
    public func payload(for voiceID: UUID) -> Payload? {
        store[voiceID]
    }

    /// The number of entries currently held in the cache.
    public var count: Int { store.count }

    // MARK: - Write

    /// Stores `payload` for `voiceID`.
    ///
    /// If an entry for `voiceID` already exists its payload is refreshed in-place without
    /// changing the insertion order. If the cache is at capacity the oldest entry is evicted
    /// before the new entry is inserted.
    public func store(_ payload: Payload, for voiceID: UUID) {
        if store[voiceID] != nil {
            store[voiceID] = payload
            return
        }
        if store.count >= maxEntries, let oldest = insertionOrder.first {
            store[oldest] = nil
            insertionOrder.removeFirst()
        }
        store[voiceID] = payload
        insertionOrder.append(voiceID)
    }

    // MARK: - Invalidation

    /// Removes the cached entry for `voiceID`. A no-op if no entry exists.
    public func invalidate(voiceID: UUID) {
        guard store[voiceID] != nil else { return }
        store[voiceID] = nil
        insertionOrder.removeAll { $0 == voiceID }
    }

    /// Removes all cached entries.
    public func invalidateAll() {
        store.removeAll()
        insertionOrder.removeAll()
    }
}
