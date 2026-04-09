import FluidAudio
import Foundation

// MARK: - SileroVADSession

/// A voice activity detector backed by Silero VAD v5 via FluidAudio's CoreML model.
///
/// The underlying model runs on Apple's Neural Engine — it has no Metal/GPU dependency
/// and works in CLI, daemon, and app contexts. Model weights (~2 MB) are downloaded from
/// HuggingFace on first use and cached in Application Support.
///
/// **Typical use:**
/// ```swift
/// let vad = try await SileroVADSession()
/// for chunk in audioChunks {
///     let result = try await vad.process(chunk: chunk)
///     if result.transitionType == .onset { handleSpeechStart() }
/// }
/// ```
///
/// **Thresholds:**
/// The default `speechThreshold` (0.60) is tuned for voice-message / batch ASR pre-segmentation
/// where false positives are cheaper than false negatives. Live microphone endpoint detection
/// should use a higher threshold (0.75–0.85) to avoid premature speech-end detection.
///
/// The silence-release threshold is derived automatically via Silero's standard hysteresis
/// offset: `silenceThreshold = speechThreshold − 0.15`.
public actor SileroVADSession: VoiceActivityDetector {

    // MARK: VoiceActivityDetector

    /// The number of audio samples the model processes per call (256 ms at 16 kHz).
    public let chunkSize: Int = VadManager.chunkSize

    /// Sample rate expected by the Silero model.
    public let sampleRate: Int = VadManager.sampleRate

    /// Speech probability threshold for the `isSpeech` gate (onset).
    public let speechThreshold: Float

    // MARK: - Internal state

    private let manager: VadManager
    private let segConfig: VadSegmentationConfig
    private var streamState: VadStreamState = .initial()

    // MARK: - Initializers

    /// Load and initialise the Silero VAD model, downloading weights if needed.
    ///
    /// - Parameters:
    ///   - speechThreshold: Probability threshold for the speech onset gate.
    ///     Defaults to 0.60, tuned for voice-message batch pre-segmentation.
    ///     The silence-release threshold is `speechThreshold − 0.15` (Silero hysteresis default).
    ///   - progressHandler: Optional download-progress callback forwarded to FluidAudio.
    public init(
        speechThreshold: Float = 0.60,
        progressHandler: DownloadUtils.ProgressHandler? = nil
    ) async throws {
        self.speechThreshold = speechThreshold
        let config = VadConfig(
            defaultThreshold: speechThreshold,
            debugMode: false,
            computeUnits: .cpuAndNeuralEngine
        )
        self.manager = try await VadManager(config: config, progressHandler: progressHandler)
        // Segmentation config: use defaults except silence gap tuned for voice messages
        // (shorter pause tolerance → tighter segments, better for ASR pre-processing)
        self.segConfig = VadSegmentationConfig(
            minSpeechDuration: 0.15,
            minSilenceDuration: 0.60,
            maxSpeechDuration: 14.0,
            speechPadding: 0.10
        )
    }

    /// Initialise from a pre-loaded `VadManager` (for dependency injection / testing).
    public init(manager: VadManager, speechThreshold: Float = 0.60) {
        self.manager = manager
        self.speechThreshold = speechThreshold
        self.segConfig = VadSegmentationConfig(
            minSpeechDuration: 0.15,
            minSilenceDuration: 0.60,
            maxSpeechDuration: 14.0,
            speechPadding: 0.10
        )
    }

    // MARK: - VoiceActivityDetector

    /// Process one chunk of mono 16 kHz Float32 audio and return a `VADResult`.
    ///
    /// State is preserved across calls via the internal `VadStreamState`. Pass chunks
    /// in sequential order. If the chunk is shorter than `chunkSize`, FluidAudio pads
    /// it with the last sample value.
    ///
    /// The `isSpeech` gate uses Silero-style hysteresis to suppress short silence gaps
    /// within utterances. The `transitionType` reflects the gate edge, not the raw
    /// probability threshold.
    public func process(chunk: [Float]) async throws -> VADResult {
        let streamResult = try await manager.processStreamingChunk(
            chunk,
            state: streamState,
            config: segConfig
        )
        let previousTriggered = streamState.triggered
        streamState = streamResult.state

        let isSpeech = streamResult.state.triggered
        let transition = transitionType(
            isSpeech: isSpeech,
            event: streamResult.event,
            previousTriggered: previousTriggered
        )

        return VADResult(
            speechProbability: streamResult.probability,
            isSpeech: isSpeech,
            transitionType: transition
        )
    }

    /// Reset the internal RNN and streaming state to initial zeros.
    public func reset() {
        streamState = .initial()
    }

    // MARK: - Private helpers

    /// Map FluidAudio's event/triggered combination onto our `VADTransitionType`.
    private func transitionType(
        isSpeech: Bool,
        event: VadStreamEvent?,
        previousTriggered: Bool
    ) -> VADTransitionType {
        if let event {
            return event.isStart ? .onset : .offset
        }
        switch (previousTriggered, isSpeech) {
        case (false, true):  return .onset      // fallback if event was nil
        case (true, false):  return .offset     // fallback if event was nil
        case (true, true):   return .sustained
        case (false, false): return .silence
        }
    }
}
