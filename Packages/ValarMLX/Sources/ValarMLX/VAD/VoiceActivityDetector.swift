import Foundation

// MARK: - VADTransitionType

/// Describes the state transition observed at the end of a processed chunk.
public enum VADTransitionType: Sendable, Equatable {
    /// Silence → speech boundary: the model crossed the speech threshold this chunk.
    case onset
    /// Speech → silence boundary: the model crossed the silence threshold this chunk.
    case offset
    /// No transition; speech continues.
    case sustained
    /// No transition; silence continues.
    case silence
}

// MARK: - VADResult

/// The result of processing a single audio chunk through the voice activity detector.
public struct VADResult: Sendable {
    /// Raw speech probability from the model (0–1).
    public let speechProbability: Float
    /// Whether the model considers this chunk to contain speech, after threshold application.
    public let isSpeech: Bool
    /// State transition observed relative to the previous chunk.
    public let transitionType: VADTransitionType

    public init(
        speechProbability: Float,
        isSpeech: Bool,
        transitionType: VADTransitionType
    ) {
        self.speechProbability = speechProbability
        self.isSpeech = isSpeech
        self.transitionType = transitionType
    }
}

// MARK: - VoiceActivityDetector

/// A stateful VAD that processes a stream of fixed-size audio chunks.
///
/// Implementations maintain internal RNN state across calls so that `process(chunk:)` is
/// a simple push-and-read interface. The caller is responsible for splitting audio into
/// chunks of exactly `chunkSize` samples at `sampleRate` Hz.
public protocol VoiceActivityDetector: Actor {
    /// Number of samples expected per chunk (at `sampleRate` Hz).
    var chunkSize: Int { get }
    /// Sample rate the model expects (typically 16 000 Hz).
    var sampleRate: Int { get }
    /// Current speech-detection threshold (0–1).
    var speechThreshold: Float { get }

    /// Process one chunk of mono Float32 audio and return the VAD result.
    ///
    /// - Parameter chunk: Exactly `chunkSize` samples at `sampleRate` Hz, mono Float32.
    /// - Returns: Speech probability, boolean gate, and transition annotation.
    func process(chunk: [Float]) async throws -> VADResult

    /// Reset the internal RNN state to the initial zero state.
    func reset() async
}
