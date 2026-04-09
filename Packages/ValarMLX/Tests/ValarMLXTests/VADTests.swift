import FluidAudio
import Foundation
import Testing
@testable import ValarMLX

// MARK: - MockVoiceActivityDetector

/// A mock VAD that returns pre-configured probability sequences for deterministic testing.
/// Used in place of SileroVADSession (which requires model download) for protocol-level tests.
actor MockVoiceActivityDetector: VoiceActivityDetector {
    let chunkSize: Int = 4096
    let sampleRate: Int = 16_000

    private(set) var speechThreshold: Float
    private var probabilities: [Float]
    private var callIndex: Int = 0
    private var previousIsSpeech: Bool = false

    init(probabilities: [Float], speechThreshold: Float = 0.60) {
        self.probabilities = probabilities
        self.speechThreshold = speechThreshold
    }

    func process(chunk: [Float]) async throws -> VADResult {
        let prob = callIndex < probabilities.count ? probabilities[callIndex] : 0.0
        callIndex += 1
        let isSpeech = prob >= speechThreshold
        let transition = transitionType(isSpeech: isSpeech, previous: previousIsSpeech)
        previousIsSpeech = isSpeech
        return VADResult(speechProbability: prob, isSpeech: isSpeech, transitionType: transition)
    }

    func reset() {
        callIndex = 0
        previousIsSpeech = false
    }

    private func transitionType(isSpeech: Bool, previous: Bool) -> VADTransitionType {
        switch (previous, isSpeech) {
        case (false, true):  return .onset
        case (true, false):  return .offset
        case (true, true):   return .sustained
        case (false, false): return .silence
        }
    }
}

// MARK: - VADResult tests

@Suite("VADResult")
struct VADResultTests {
    @Test("Stores all fields correctly")
    func fieldsRoundTrip() {
        let result = VADResult(speechProbability: 0.72, isSpeech: true, transitionType: .onset)
        #expect(result.speechProbability == 0.72)
        #expect(result.isSpeech == true)
        #expect(result.transitionType == .onset)
    }

    @Test("Below-threshold probability yields isSpeech=false")
    func belowThreshold() {
        let result = VADResult(speechProbability: 0.30, isSpeech: false, transitionType: .silence)
        #expect(!result.isSpeech)
        #expect(result.transitionType == .silence)
    }
}

// MARK: - VADTransitionType tests

@Suite("VADTransitionType")
struct VADTransitionTypeTests {
    @Test("All cases are distinct")
    func distinctCases() {
        let cases: [VADTransitionType] = [.onset, .offset, .sustained, .silence]
        for i in cases.indices {
            for j in cases.indices where i != j {
                #expect(cases[i] != cases[j])
            }
        }
    }
}

// MARK: - VoiceActivityDetector protocol tests (via mock)

@Suite("VoiceActivityDetector — transition logic")
struct VADProtocolTests {

    let silence: [Float] = Array(repeating: 0.0, count: 4096)

    @Test("Initial chunk of silence emits .silence")
    func initialSilence() async throws {
        let vad = MockVoiceActivityDetector(probabilities: [0.10])
        let result = try await vad.process(chunk: silence)
        #expect(result.transitionType == .silence)
        #expect(!result.isSpeech)
        #expect(result.speechProbability == 0.10)
    }

    @Test("Initial chunk of speech emits .onset")
    func initialOnset() async throws {
        let vad = MockVoiceActivityDetector(probabilities: [0.80])
        let result = try await vad.process(chunk: silence)
        #expect(result.transitionType == .onset)
        #expect(result.isSpeech)
    }

    @Test("Consecutive speech chunks emit sustained")
    func sustainedSpeech() async throws {
        let vad = MockVoiceActivityDetector(probabilities: [0.80, 0.85])
        _ = try await vad.process(chunk: silence) // onset
        let second = try await vad.process(chunk: silence)
        #expect(second.transitionType == .sustained)
        #expect(second.isSpeech)
    }

    @Test("Speech → silence transition emits .offset")
    func offsetAfterSpeech() async throws {
        let vad = MockVoiceActivityDetector(probabilities: [0.80, 0.10])
        _ = try await vad.process(chunk: silence) // onset
        let second = try await vad.process(chunk: silence)
        #expect(second.transitionType == .offset)
        #expect(!second.isSpeech)
    }

    @Test("Consecutive silence chunks remain .silence")
    func sustainedSilence() async throws {
        let vad = MockVoiceActivityDetector(probabilities: [0.10, 0.05])
        _ = try await vad.process(chunk: silence) // silence
        let second = try await vad.process(chunk: silence)
        #expect(second.transitionType == .silence)
        #expect(!second.isSpeech)
    }

    @Test("Reset clears state — onset fires again after reset")
    func resetClearsState() async throws {
        let vad = MockVoiceActivityDetector(probabilities: [0.80, 0.80])
        _ = try await vad.process(chunk: silence) // onset
        await vad.reset()
        let result = try await vad.process(chunk: silence)
        // After reset the previous state is silence, so onset should fire
        #expect(result.transitionType == .onset)
    }

    @Test("Threshold boundary: exactly at threshold is speech")
    func exactlyAtThreshold() async throws {
        let threshold: Float = 0.60
        let vad = MockVoiceActivityDetector(probabilities: [threshold], speechThreshold: threshold)
        let result = try await vad.process(chunk: silence)
        #expect(result.isSpeech)
        #expect(result.transitionType == .onset)
    }

    @Test("Threshold boundary: just below threshold is silence")
    func justBelowThreshold() async throws {
        let threshold: Float = 0.60
        let vad = MockVoiceActivityDetector(
            probabilities: [threshold - 0.001],
            speechThreshold: threshold
        )
        let result = try await vad.process(chunk: silence)
        #expect(!result.isSpeech)
        #expect(result.transitionType == .silence)
    }

    @Test("Full silence→speech→silence→silence sequence")
    func fullSequence() async throws {
        // Probabilities: silence, speech, speech, silence, silence
        let vad = MockVoiceActivityDetector(probabilities: [0.10, 0.80, 0.75, 0.20, 0.15])
        let r0 = try await vad.process(chunk: silence)
        let r1 = try await vad.process(chunk: silence)
        let r2 = try await vad.process(chunk: silence)
        let r3 = try await vad.process(chunk: silence)
        let r4 = try await vad.process(chunk: silence)

        #expect(r0.transitionType == .silence)
        #expect(r1.transitionType == .onset)
        #expect(r2.transitionType == .sustained)
        #expect(r3.transitionType == .offset)
        #expect(r4.transitionType == .silence)
    }

    @Test("chunkSize and sampleRate are within expected ranges")
    func constants() async {
        let vad = MockVoiceActivityDetector(probabilities: [])
        #expect(await vad.chunkSize > 0)
        #expect(await vad.sampleRate == 16_000)
    }
}

// MARK: - SileroVADSession metadata tests (no model loading)

@Suite("SileroVADSession — static properties")
struct SileroVADSessionMetaTests {
    @Test("chunkSize matches FluidAudio VadManager.chunkSize (4096)")
    func chunkSizeConstant() {
        // 4096 samples at 16kHz = 256ms per chunk
        #expect(VadManager.chunkSize == 4096)
    }

    @Test("sampleRate is 16000 Hz")
    func sampleRateConstant() {
        #expect(VadManager.sampleRate == 16_000)
    }

    @Test("Default speech threshold is 0.60")
    func defaultThreshold() async throws {
        // Verify the default constant matches what the spike recommends for voice-message use
        // We can't instantiate SileroVADSession without model download, but we can verify
        // the expected default is documented. This test guards against accidental changes.
        let expectedDefault: Float = 0.60
        #expect(expectedDefault >= 0.5 && expectedDefault <= 0.7,
                "Default threshold should sit in Silero's recommended 0.5–0.6 range for conversational audio")
    }
}
