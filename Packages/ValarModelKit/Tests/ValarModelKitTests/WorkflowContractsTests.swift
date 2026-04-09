import Foundation
import XCTest
@testable import ValarModelKit

final class WorkflowContractsTests: XCTestCase {

    // MARK: - SpeechSynthesisRequest

    func testSpeechSynthesisRequestDefaultsAndRoundTrip() throws {
        let request = SpeechSynthesisRequest(
            model: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
            text: "Hello, world!"
        )
        XCTAssertEqual(request.text, "Hello, world!")
        XCTAssertEqual(request.sampleRate, 24_000)
        XCTAssertEqual(request.responseFormat, "wav")
        XCTAssertNil(request.voice)
        XCTAssertNil(request.language)
        XCTAssertNil(request.referenceAudioAssetName)
        XCTAssertNil(request.instruct)
        XCTAssertNil(request.exaggeration)
        XCTAssertNil(request.cfgWeight)

        let data = try JSONEncoder().encode(request)
        let decoded = try JSONDecoder().decode(SpeechSynthesisRequest.self, from: data)
        XCTAssertEqual(decoded.model, request.model)
        XCTAssertEqual(decoded.text, request.text)
        XCTAssertEqual(decoded.sampleRate, request.sampleRate)
        XCTAssertEqual(decoded.responseFormat, request.responseFormat)
    }

    func testSpeechSynthesisRequestWithAllFields() throws {
        let voice = VoiceProfile(
            label: "Narrator",
            backendVoiceID: "neutral_female",
            sourceModel: "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16",
            localeIdentifier: "en-US",
            runtimeModel: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
            referenceAudioAssetName: "narrator.wav",
            referenceTranscript: "The quick brown fox."
        )
        let request = SpeechSynthesisRequest(
            model: "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16",
            text: "Test narration.",
            voice: voice,
            language: "en",
            referenceAudioAssetName: "narrator.wav",
            referenceAudioSampleRate: 24_000,
            referenceTranscript: "The quick brown fox.",
            instruct: "Warm, crisp studio delivery.",
            exaggeration: 0.6,
            cfgWeight: 0.4,
            sampleRate: 16_000,
            responseFormat: "m4a"
        )

        let data = try JSONEncoder().encode(request)
        let decoded = try JSONDecoder().decode(SpeechSynthesisRequest.self, from: data)

        XCTAssertEqual(decoded.voice?.label, "Narrator")
        XCTAssertEqual(decoded.voice?.backendVoiceID, "neutral_female")
        XCTAssertEqual(decoded.language, "en")
        XCTAssertEqual(decoded.referenceTranscript, "The quick brown fox.")
        XCTAssertEqual(decoded.instruct, "Warm, crisp studio delivery.")
        XCTAssertEqual(decoded.exaggeration, 0.6)
        XCTAssertEqual(decoded.cfgWeight, 0.4)
        XCTAssertEqual(decoded.sampleRate, 16_000)
        XCTAssertEqual(decoded.responseFormat, "m4a")
        XCTAssertEqual(decoded.referenceAudioSampleRate, 24_000)
    }

    // MARK: - VoiceProfile

    func testVoiceProfileRoundTrip() throws {
        let id = UUID()
        let profile = VoiceProfile(
            id: id,
            label: "Studio Clone",
            backendVoiceID: "neutral_female",
            sourceModel: "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16",
            localeIdentifier: "en-GB",
            runtimeModel: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
            referenceAudioAssetName: "clone.wav",
            referenceTranscript: "This is a reference.",
            speakerEmbedding: Data([0x01, 0x02, 0x03])
        )

        let data = try JSONEncoder().encode(profile)
        let decoded = try JSONDecoder().decode(VoiceProfile.self, from: data)

        XCTAssertEqual(decoded.id, id)
        XCTAssertEqual(decoded.label, "Studio Clone")
        XCTAssertEqual(decoded.backendVoiceID, "neutral_female")
        XCTAssertEqual(decoded.sourceModel.rawValue, "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16")
        XCTAssertEqual(decoded.localeIdentifier, "en-GB")
        XCTAssertEqual(decoded.runtimeModel?.rawValue, "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16")
        XCTAssertEqual(decoded.referenceAudioAssetName, "clone.wav")
        XCTAssertEqual(decoded.referenceTranscript, "This is a reference.")
        XCTAssertEqual(decoded.speakerEmbedding, Data([0x01, 0x02, 0x03]))
        XCTAssertEqual(decoded.voiceSelector, "neutral_female")
    }

    func testVoiceProfileDefaultsAreNil() {
        let profile = VoiceProfile(label: "Plain", sourceModel: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16")
        XCTAssertNil(profile.localeIdentifier)
        XCTAssertNil(profile.runtimeModel)
        XCTAssertNil(profile.referenceAudioAssetName)
        XCTAssertNil(profile.referenceTranscript)
        XCTAssertNil(profile.speakerEmbedding)
        XCTAssertEqual(profile.voiceSelector, "Plain")
    }

    func testVoiceProfileCompatibilityValidationForVoxtralPresets() {
        let preset = VoiceProfile(
            label: "Neutral Female",
            backendVoiceID: "neutral_female",
            sourceModel: "mistralai/Voxtral-4B-TTS-2603"
        )

        XCTAssertThrowsError(
            try preset.validateCompatibility(
                with: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
                familyID: .qwen3TTS
            )
        )

        XCTAssertNoThrow(
            try preset.validateCompatibility(
                with: "mistralai/Voxtral-4B-TTS-2603",
                familyID: .voxtralTTS
            )
        )
    }

    func testVoiceProfileCompatibilityValidationRejectsCustomVoicesForVoxtral() {
        let custom = VoiceProfile(
            label: "Warm narrator",
            sourceModel: "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16"
        )

        XCTAssertThrowsError(
            try custom.validateCompatibility(
                with: "mistralai/Voxtral-4B-TTS-2603",
                familyID: .voxtralTTS
            )
        )
    }

    // MARK: - AudioChunk

    func testAudioChunkSamplesProtocolAndRoundTrip() throws {
        let chunk = AudioChunk(samples: [0.1, 0.2, -0.3, 0.5], sampleRate: 24_000)
        XCTAssertEqual(chunk.samples, [0.1, 0.2, -0.3, 0.5])
        XCTAssertEqual(chunk.sampleRate, 24_000)

        let data = try JSONEncoder().encode(chunk)
        let decoded = try JSONDecoder().decode(AudioChunk.self, from: data)
        XCTAssertEqual(decoded.samples, chunk.samples)
        XCTAssertEqual(decoded.sampleRate, chunk.sampleRate)
    }

    func testAudioChunkConformsToAudioSampleBufferProtocol() {
        let chunk = AudioChunk(samples: [1.0, -1.0], sampleRate: 16_000)
        let protocol_value: any AudioSampleBuffer = chunk
        XCTAssertEqual(protocol_value.samples, [1.0, -1.0])
        XCTAssertEqual(protocol_value.sampleRate, 16_000)
    }

    // MARK: - SpeechRecognitionRequest

    func testSpeechRecognitionRequestRoundTrip() throws {
        let request = SpeechRecognitionRequest(
            model: "mlx-community/Qwen3-ASR-0.6B-8bit",
            audioAssetName: "clip.wav",
            languageHint: "en",
            sampleRate: 16_000
        )

        let data = try JSONEncoder().encode(request)
        let decoded = try JSONDecoder().decode(SpeechRecognitionRequest.self, from: data)

        XCTAssertEqual(decoded.model.rawValue, "mlx-community/Qwen3-ASR-0.6B-8bit")
        XCTAssertEqual(decoded.audioAssetName, "clip.wav")
        XCTAssertNil(decoded.audioChunk)
        XCTAssertEqual(decoded.languageHint, "en")
        XCTAssertEqual(decoded.sampleRate, 16_000)
    }

    func testSpeechRecognitionRequestInlineAudioRoundTrip() throws {
        let request = SpeechRecognitionRequest(
            model: "mlx-community/Qwen3-ASR-0.6B-8bit",
            audio: AudioChunk(samples: [0.1, -0.2, 0.3], sampleRate: 16_000),
            languageHint: "English"
        )

        let data = try JSONEncoder().encode(request)
        let decoded = try JSONDecoder().decode(SpeechRecognitionRequest.self, from: data)

        XCTAssertEqual(decoded.model.rawValue, "mlx-community/Qwen3-ASR-0.6B-8bit")
        XCTAssertEqual(decoded.audioAssetName, "__inline_audio_chunk__")
        XCTAssertEqual(decoded.audioChunk, AudioChunk(samples: [0.1, -0.2, 0.3], sampleRate: 16_000))
        XCTAssertEqual(decoded.languageHint, "English")
        XCTAssertEqual(decoded.sampleRate, 16_000)
    }

    // MARK: - TranscriptionResult

    func testTranscriptionResultRoundTrip() throws {
        let result = TranscriptionResult(
            text: "Hello world, how are you?",
            segments: ["Hello world,", " how are you?"]
        )
        XCTAssertEqual(result.text, "Hello world, how are you?")
        XCTAssertEqual(result.segments, ["Hello world,", " how are you?"])

        let data = try JSONEncoder().encode(result)
        let decoded = try JSONDecoder().decode(TranscriptionResult.self, from: data)
        XCTAssertEqual(decoded.text, result.text)
        XCTAssertEqual(decoded.segments, result.segments)
    }

    func testTranscriptionResultDefaultSegmentsIsEmpty() {
        let result = TranscriptionResult(text: "Plain text")
        XCTAssertEqual(result.segments, [])
    }

    func testSpeechToTextAliasesPreserveUnderlyingTypes() {
        let request: SpeechToTextRequest = SpeechRecognitionRequest(
            model: "mlx-community/Qwen3-ASR-0.6B-8bit",
            audio: AudioChunk(samples: [0.0], sampleRate: 16_000)
        )
        let response: SpeechToTextResponse = RichTranscriptionResult(
            text: "Hello",
            backendMetadata: BackendMetadata(modelId: "test-model", backendKind: .mlx)
        )

        XCTAssertEqual(request.audioChunk?.sampleRate, 16_000)
        XCTAssertEqual(response.text, "Hello")
    }

    // MARK: - ForcedAlignmentResult and AlignmentToken

    func testAlignmentTokenRoundTrip() throws {
        let token = AlignmentToken(text: "Hello", startTime: 0.0, endTime: 0.45, confidence: 0.98)
        XCTAssertEqual(token.text, "Hello")
        XCTAssertEqual(token.startTime, 0.0)
        XCTAssertEqual(token.endTime, 0.45)
        XCTAssertEqual(token.confidence, 0.98)

        let data = try JSONEncoder().encode(token)
        let decoded = try JSONDecoder().decode(AlignmentToken.self, from: data)
        XCTAssertEqual(decoded, token)
    }

    func testAlignmentTokenDefaultConfidenceIsNil() {
        let token = AlignmentToken(text: "word", startTime: 1.0, endTime: 1.3)
        XCTAssertNil(token.confidence)
    }

    func testForcedAlignmentResultRoundTrip() throws {
        let result = ForcedAlignmentResult(
            transcript: "Hello world",
            tokens: [
                AlignmentToken(text: "Hello", startTime: 0.0, endTime: 0.45, confidence: 0.99),
                AlignmentToken(text: "world", startTime: 0.5, endTime: 0.9, confidence: 0.95),
            ]
        )

        let data = try JSONEncoder().encode(result)
        let decoded = try JSONDecoder().decode(ForcedAlignmentResult.self, from: data)

        XCTAssertEqual(decoded.transcript, "Hello world")
        XCTAssertEqual(decoded.tokens.count, 2)
        XCTAssertEqual(decoded.tokens[0].text, "Hello")
        XCTAssertEqual(decoded.tokens[1].startTime, 0.5)
        XCTAssertEqual(decoded.tokens[1].confidence, 0.95)
    }

    // MARK: - TokenizationRequest / TokenizationResult

    func testTokenizationRequestRoundTrip() throws {
        let request = TokenizationRequest(
            model: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
            text: "The quick brown fox jumps."
        )

        let data = try JSONEncoder().encode(request)
        let decoded = try JSONDecoder().decode(TokenizationRequest.self, from: data)

        XCTAssertEqual(decoded.model, request.model)
        XCTAssertEqual(decoded.text, request.text)
    }

    func testTokenizationResultRoundTrip() throws {
        let result = TokenizationResult(
            model: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
            tokenCount: 12,
            chunkCount: 1
        )

        let data = try JSONEncoder().encode(result)
        let decoded = try JSONDecoder().decode(TokenizationResult.self, from: data)

        XCTAssertEqual(decoded.tokenCount, 12)
        XCTAssertEqual(decoded.chunkCount, 1)
    }

    // MARK: - SpeechToSpeechResult

    func testSpeechToSpeechResultRoundTrip() throws {
        let result = SpeechToSpeechResult(
            translation: "Hola mundo",
            audio: AudioChunk(samples: [0.5, -0.5], sampleRate: 24_000),
            transcription: TranscriptionResult(text: "Hello world")
        )

        let data = try JSONEncoder().encode(result)
        let decoded = try JSONDecoder().decode(SpeechToSpeechResult.self, from: data)

        XCTAssertEqual(decoded.translation, "Hola mundo")
        XCTAssertEqual(decoded.audio.sampleRate, 24_000)
        XCTAssertEqual(decoded.transcription?.text, "Hello world")
    }

    func testSpeechToSpeechResultNilTranscription() throws {
        let result = SpeechToSpeechResult(
            translation: "Bonjour",
            audio: AudioChunk(samples: [0.1], sampleRate: 16_000)
        )
        XCTAssertNil(result.transcription)

        let data = try JSONEncoder().encode(result)
        let decoded = try JSONDecoder().decode(SpeechToSpeechResult.self, from: data)
        XCTAssertNil(decoded.transcription)
    }

    // MARK: - TranslationRequest

    func testTranslationRequestRoundTrip() throws {
        let request = TranslationRequest(sourceLanguage: "en", targetLanguage: "ja", text: "Hello")

        let data = try JSONEncoder().encode(request)
        let decoded = try JSONDecoder().decode(TranslationRequest.self, from: data)

        XCTAssertEqual(decoded.sourceLanguage, "en")
        XCTAssertEqual(decoded.targetLanguage, "ja")
        XCTAssertEqual(decoded.text, "Hello")
    }

    // MARK: - ForcedAlignmentRequest

    func testForcedAlignmentRequestRoundTrip() throws {
        let request = ForcedAlignmentRequest(
            model: "mlx-community/Qwen3-ForcedAligner-0.6B-8bit",
            audioAssetName: "speech.wav",
            transcript: "Hello world",
            languageHint: "en"
        )

        let data = try JSONEncoder().encode(request)
        let decoded = try JSONDecoder().decode(ForcedAlignmentRequest.self, from: data)

        XCTAssertEqual(decoded.model.rawValue, "mlx-community/Qwen3-ForcedAligner-0.6B-8bit")
        XCTAssertEqual(decoded.audioAssetName, "speech.wav")
        XCTAssertEqual(decoded.transcript, "Hello world")
        XCTAssertEqual(decoded.languageHint, "en")
    }

    func testForcedAlignmentRequestDefaultLanguageHintIsNil() {
        let request = ForcedAlignmentRequest(
            model: "mlx-community/Qwen3-ForcedAligner-0.6B-8bit",
            audioAssetName: "clip.wav",
            transcript: "some text"
        )
        XCTAssertNil(request.languageHint)
    }

    // MARK: - TranscriptionSegment

    func testTranscriptionSegmentRoundTrip() throws {
        let segment = TranscriptionSegment(
            text: "Hello",
            startTime: 0.0,
            endTime: 0.5,
            confidence: 0.97,
            isFinal: true,
            chunkIndex: 0
        )

        let data = try JSONEncoder().encode(segment)
        let decoded = try JSONDecoder().decode(TranscriptionSegment.self, from: data)

        XCTAssertEqual(decoded.text, "Hello")
        XCTAssertEqual(decoded.startTime, 0.0)
        XCTAssertEqual(decoded.endTime, 0.5)
        XCTAssertEqual(decoded.confidence ?? 0, 0.97, accuracy: 0.0001)
        XCTAssertTrue(decoded.isFinal)
        XCTAssertEqual(decoded.chunkIndex, 0)
    }

    func testTranscriptionSegmentDefaults() {
        let segment = TranscriptionSegment(text: "world")
        XCTAssertNil(segment.startTime)
        XCTAssertNil(segment.endTime)
        XCTAssertNil(segment.confidence)
        XCTAssertTrue(segment.isFinal)
        XCTAssertNil(segment.chunkIndex)
    }

    // MARK: - BackendMetadata

    func testBackendMetadataRoundTrip() throws {
        let meta = BackendMetadata(
            modelId: "mlx-community/Qwen3-ASR-0.6B-8bit",
            backendKind: .mlx,
            inferenceTimeSeconds: 1.23
        )

        let data = try JSONEncoder().encode(meta)
        let decoded = try JSONDecoder().decode(BackendMetadata.self, from: data)

        XCTAssertEqual(decoded.modelId, "mlx-community/Qwen3-ASR-0.6B-8bit")
        XCTAssertEqual(decoded.backendKind, .mlx)
        XCTAssertEqual(decoded.inferenceTimeSeconds, 1.23)
    }

    func testBackendMetadataInferenceTimeNil() throws {
        let meta = BackendMetadata(modelId: "some-model", backendKind: .coreml)
        XCTAssertNil(meta.inferenceTimeSeconds)

        let data = try JSONEncoder().encode(meta)
        let decoded = try JSONDecoder().decode(BackendMetadata.self, from: data)
        XCTAssertNil(decoded.inferenceTimeSeconds)
    }

    // MARK: - RecognitionMetrics

    func testRecognitionMetricsRoundTrip() throws {
        let metrics = RecognitionMetrics(
            audioDurationSeconds: 5.0,
            inferenceTimeSeconds: 1.0,
            rtf: 0.2,
            segmentsProcessed: 3
        )

        let data = try JSONEncoder().encode(metrics)
        let decoded = try JSONDecoder().decode(RecognitionMetrics.self, from: data)

        XCTAssertEqual(decoded.audioDurationSeconds, 5.0)
        XCTAssertEqual(decoded.inferenceTimeSeconds, 1.0)
        XCTAssertEqual(decoded.rtf, 0.2)
        XCTAssertEqual(decoded.segmentsProcessed, 3)
    }

    func testRecognitionMetricsDefaults() {
        let metrics = RecognitionMetrics()
        XCTAssertNil(metrics.audioDurationSeconds)
        XCTAssertNil(metrics.inferenceTimeSeconds)
        XCTAssertNil(metrics.rtf)
        XCTAssertEqual(metrics.segmentsProcessed, 0)
    }

    // MARK: - RichTranscriptionResult

    func testRichTranscriptionResultRoundTrip() throws {
        let segment = TranscriptionSegment(text: "Hello world", startTime: 0.0, endTime: 1.2, confidence: 0.95, isFinal: true, chunkIndex: 0)
        let word = TranscriptionSegment(text: "Hello", startTime: 0.0, endTime: 0.5, isFinal: true)
        let meta = BackendMetadata(modelId: "mlx-community/Qwen3-ASR-0.6B-8bit", backendKind: .mlx, inferenceTimeSeconds: 0.8)

        let result = RichTranscriptionResult(
            text: "Hello world",
            language: "en",
            durationSeconds: 1.2,
            segments: [segment],
            words: [word],
            alignmentReference: nil,
            backendMetadata: meta
        )

        let data = try JSONEncoder().encode(result)
        let decoded = try JSONDecoder().decode(RichTranscriptionResult.self, from: data)

        XCTAssertEqual(decoded.text, "Hello world")
        XCTAssertEqual(decoded.language, "en")
        XCTAssertEqual(decoded.durationSeconds, 1.2)
        XCTAssertEqual(decoded.segments.count, 1)
        XCTAssertEqual(decoded.segments[0].text, "Hello world")
        XCTAssertEqual(decoded.words?.count, 1)
        XCTAssertNil(decoded.alignmentReference)
        XCTAssertEqual(decoded.backendMetadata.modelId, "mlx-community/Qwen3-ASR-0.6B-8bit")
        XCTAssertEqual(decoded.backendMetadata.backendKind, .mlx)
    }

    func testRichTranscriptionResultWithAlignmentReference() throws {
        let alignmentToken = AlignmentToken(text: "Hello", startTime: 0.0, endTime: 0.45, confidence: 0.99)
        let alignmentResult = ForcedAlignmentResult(transcript: "Hello world", tokens: [alignmentToken])
        let meta = BackendMetadata(modelId: "model", backendKind: .coreml)

        let result = RichTranscriptionResult(text: "Hello world", backendMetadata: meta)
        XCTAssertNil(result.alignmentReference)

        let resultWithAlignment = RichTranscriptionResult(
            text: "Hello world",
            alignmentReference: alignmentResult,
            backendMetadata: meta
        )

        let data = try JSONEncoder().encode(resultWithAlignment)
        let decoded = try JSONDecoder().decode(RichTranscriptionResult.self, from: data)

        XCTAssertEqual(decoded.alignmentReference?.transcript, "Hello world")
        XCTAssertEqual(decoded.alignmentReference?.tokens.count, 1)
    }

    // MARK: - TranscriptionResult convenience init from RichTranscriptionResult

    func testTranscriptionResultConvenienceInitFromRich() {
        let segments = [
            TranscriptionSegment(text: "Hello,", startTime: 0.0, endTime: 0.4, isFinal: true),
            TranscriptionSegment(text: "world!", startTime: 0.5, endTime: 0.9, isFinal: true),
        ]
        let meta = BackendMetadata(modelId: "model", backendKind: .mlx)
        let rich = RichTranscriptionResult(text: "Hello, world!", segments: segments, backendMetadata: meta)

        let flat = TranscriptionResult(rich)

        XCTAssertEqual(flat.text, "Hello, world!")
        XCTAssertEqual(flat.segments, ["Hello,", "world!"])
    }

    func testTranscriptionResultConvenienceInitEmptySegments() {
        let meta = BackendMetadata(modelId: "model", backendKind: .mlx)
        let rich = RichTranscriptionResult(text: "Plain text", backendMetadata: meta)
        let flat = TranscriptionResult(rich)
        XCTAssertEqual(flat.text, "Plain text")
        XCTAssertEqual(flat.segments, [])
    }

    // MARK: - SpeechRecognitionEvent

    func testSpeechRecognitionEventPartialRoundTrip() throws {
        let segment = TranscriptionSegment(text: "partial...", isFinal: false, chunkIndex: 2)
        let event = SpeechRecognitionEvent.partial(segment)

        let data = try JSONEncoder().encode(event)
        let decoded = try JSONDecoder().decode(SpeechRecognitionEvent.self, from: data)

        guard case .partial(let s) = decoded else {
            XCTFail("Expected .partial"); return
        }
        XCTAssertEqual(s.text, "partial...")
        XCTAssertFalse(s.isFinal)
        XCTAssertEqual(s.chunkIndex, 2)
    }

    func testSpeechRecognitionEventFinalSegmentRoundTrip() throws {
        let segment = TranscriptionSegment(text: "done segment", startTime: 1.0, endTime: 2.0, isFinal: true)
        let event = SpeechRecognitionEvent.finalSegment(segment)

        let data = try JSONEncoder().encode(event)
        let decoded = try JSONDecoder().decode(SpeechRecognitionEvent.self, from: data)

        guard case .finalSegment(let s) = decoded else {
            XCTFail("Expected .finalSegment"); return
        }
        XCTAssertEqual(s.text, "done segment")
        XCTAssertEqual(s.startTime, 1.0)
    }

    func testSpeechRecognitionEventCompletedRoundTrip() throws {
        let meta = BackendMetadata(modelId: "model", backendKind: .mlx, inferenceTimeSeconds: 0.5)
        let rich = RichTranscriptionResult(text: "Full transcript.", language: "en", backendMetadata: meta)
        let event = SpeechRecognitionEvent.completed(rich)

        let data = try JSONEncoder().encode(event)
        let decoded = try JSONDecoder().decode(SpeechRecognitionEvent.self, from: data)

        guard case .completed(let r) = decoded else {
            XCTFail("Expected .completed"); return
        }
        XCTAssertEqual(r.text, "Full transcript.")
        XCTAssertEqual(r.language, "en")
        XCTAssertEqual(r.backendMetadata.inferenceTimeSeconds, 0.5)
    }

    func testSpeechRecognitionEventMetricsRoundTrip() throws {
        let m = RecognitionMetrics(audioDurationSeconds: 10.0, inferenceTimeSeconds: 2.0, rtf: 0.2, segmentsProcessed: 5)
        let event = SpeechRecognitionEvent.metrics(m)

        let data = try JSONEncoder().encode(event)
        let decoded = try JSONDecoder().decode(SpeechRecognitionEvent.self, from: data)

        guard case .metrics(let metrics) = decoded else {
            XCTFail("Expected .metrics"); return
        }
        XCTAssertEqual(metrics.rtf, 0.2)
        XCTAssertEqual(metrics.segmentsProcessed, 5)
    }

    func testSpeechRecognitionEventWarningRoundTrip() throws {
        let event = SpeechRecognitionEvent.warning("Language fallback to English.")

        let data = try JSONEncoder().encode(event)
        let decoded = try JSONDecoder().decode(SpeechRecognitionEvent.self, from: data)

        guard case .warning(let msg) = decoded else {
            XCTFail("Expected .warning"); return
        }
        XCTAssertEqual(msg, "Language fallback to English.")
    }

    // MARK: - ChunkPolicy

    func testChunkPolicyDefaults() {
        let policy = ChunkPolicy()
        XCTAssertEqual(policy.targetChunkDuration, 30.0)
        XCTAssertEqual(policy.overlapDuration, 1.0)
        XCTAssertEqual(policy.minSpeechDuration, 0.25)
        XCTAssertEqual(policy.silenceThreshold, 0.01)
    }

    func testChunkPolicyRoundTrip() throws {
        let policy = ChunkPolicy(
            targetChunkDuration: 15.0,
            overlapDuration: 0.5,
            minSpeechDuration: 0.1,
            silenceThreshold: 0.02
        )

        let data = try JSONEncoder().encode(policy)
        let decoded = try JSONDecoder().decode(ChunkPolicy.self, from: data)

        XCTAssertEqual(decoded.targetChunkDuration, 15.0)
        XCTAssertEqual(decoded.overlapDuration, 0.5)
        XCTAssertEqual(decoded.minSpeechDuration, 0.1)
        XCTAssertEqual(decoded.silenceThreshold, 0.02)
    }

    // MARK: - VADModelKind

    func testVADModelKindRoundTrip() throws {
        for kind in VADModelKind.allCases {
            let data = try JSONEncoder().encode(kind)
            let decoded = try JSONDecoder().decode(VADModelKind.self, from: data)
            XCTAssertEqual(decoded, kind)
        }
    }

    func testVADModelKindCasesExist() {
        XCTAssertEqual(VADModelKind.sileroV5.rawValue, "sileroV5")
        XCTAssertEqual(VADModelKind.energyBased.rawValue, "energyBased")
        XCTAssertEqual(VADModelKind.allCases.count, 2)
    }

    // MARK: - VADPolicy

    func testVADPolicyDefaults() {
        let policy = VADPolicy()
        XCTAssertFalse(policy.enabled)
        XCTAssertEqual(policy.model, .energyBased)
        XCTAssertEqual(policy.onsetThreshold, 0.5, accuracy: 0.0001)
        XCTAssertEqual(policy.offsetThreshold, 0.35, accuracy: 0.0001)
        XCTAssertEqual(policy.minSpeechMs, 250)
        XCTAssertEqual(policy.minSilenceMs, 300)
    }

    func testVADPolicyRoundTrip() throws {
        let policy = VADPolicy(
            enabled: true,
            model: .sileroV5,
            onsetThreshold: 0.6,
            offsetThreshold: 0.4,
            minSpeechMs: 200,
            minSilenceMs: 400
        )

        let data = try JSONEncoder().encode(policy)
        let decoded = try JSONDecoder().decode(VADPolicy.self, from: data)

        XCTAssertTrue(decoded.enabled)
        XCTAssertEqual(decoded.model, .sileroV5)
        XCTAssertEqual(decoded.onsetThreshold, 0.6, accuracy: 0.0001)
        XCTAssertEqual(decoded.offsetThreshold, 0.4, accuracy: 0.0001)
        XCTAssertEqual(decoded.minSpeechMs, 200)
        XCTAssertEqual(decoded.minSilenceMs, 400)
    }

    // MARK: - ContextCarryOverPolicy

    func testContextCarryOverPolicyDefaults() {
        let policy = ContextCarryOverPolicy()
        XCTAssertTrue(policy.enabled)
        XCTAssertNil(policy.maxTokens)
    }

    func testContextCarryOverPolicyRoundTrip() throws {
        let policy = ContextCarryOverPolicy(enabled: true, maxTokens: 64)

        let data = try JSONEncoder().encode(policy)
        let decoded = try JSONDecoder().decode(ContextCarryOverPolicy.self, from: data)

        XCTAssertTrue(decoded.enabled)
        XCTAssertEqual(decoded.maxTokens, 64)
    }

    func testContextCarryOverPolicyNilMaxTokens() throws {
        let policy = ContextCarryOverPolicy(enabled: false, maxTokens: nil)

        let data = try JSONEncoder().encode(policy)
        let decoded = try JSONDecoder().decode(ContextCarryOverPolicy.self, from: data)

        XCTAssertFalse(decoded.enabled)
        XCTAssertNil(decoded.maxTokens)
    }

    // MARK: - StreamingSpeechRecognitionRequest

    func testStreamingSpeechRecognitionRequestDefaults() {
        let request = StreamingSpeechRecognitionRequest(
            model: "mlx-community/Qwen3-ASR-0.6B-8bit"
        )
        XCTAssertEqual(request.model.rawValue, "mlx-community/Qwen3-ASR-0.6B-8bit")
        XCTAssertNil(request.languageHint)
        XCTAssertEqual(request.chunkPolicy.targetChunkDuration, 30.0)
        XCTAssertFalse(request.vadPolicy.enabled)
        XCTAssertEqual(request.vadPolicy.model, .energyBased)
        XCTAssertTrue(request.contextCarryOver.enabled)
        XCTAssertNil(request.contextCarryOver.maxTokens)
    }

    func testStreamingSpeechRecognitionRequestSessionIdIsUnique() {
        let a = StreamingSpeechRecognitionRequest(model: "test/model")
        let b = StreamingSpeechRecognitionRequest(model: "test/model")
        XCTAssertNotEqual(a.sessionId, b.sessionId)
    }

    func testStreamingSpeechRecognitionRequestRoundTrip() throws {
        let sessionId = UUID()
        let request = StreamingSpeechRecognitionRequest(
            model: "mlx-community/Qwen3-ASR-0.6B-8bit",
            sessionId: sessionId,
            languageHint: "en",
            chunkPolicy: ChunkPolicy(targetChunkDuration: 20.0, overlapDuration: 0.5, minSpeechDuration: 0.2, silenceThreshold: 0.02),
            vadPolicy: VADPolicy(enabled: true, model: .sileroV5, onsetThreshold: 0.6, offsetThreshold: 0.3, minSpeechMs: 200, minSilenceMs: 350),
            contextCarryOver: ContextCarryOverPolicy(enabled: true, maxTokens: 128)
        )

        let data = try JSONEncoder().encode(request)
        let decoded = try JSONDecoder().decode(StreamingSpeechRecognitionRequest.self, from: data)

        XCTAssertEqual(decoded.model.rawValue, "mlx-community/Qwen3-ASR-0.6B-8bit")
        XCTAssertEqual(decoded.sessionId, sessionId)
        XCTAssertEqual(decoded.languageHint, "en")
        XCTAssertEqual(decoded.chunkPolicy.targetChunkDuration, 20.0)
        XCTAssertEqual(decoded.chunkPolicy.overlapDuration, 0.5)
        XCTAssertTrue(decoded.vadPolicy.enabled)
        XCTAssertEqual(decoded.vadPolicy.model, .sileroV5)
        XCTAssertEqual(decoded.vadPolicy.minSpeechMs, 200)
        XCTAssertTrue(decoded.contextCarryOver.enabled)
        XCTAssertEqual(decoded.contextCarryOver.maxTokens, 128)
    }

    func testStreamingSpeechRecognitionRequestNilLanguageHint() throws {
        let request = StreamingSpeechRecognitionRequest(model: "test/model")

        let data = try JSONEncoder().encode(request)
        let decoded = try JSONDecoder().decode(StreamingSpeechRecognitionRequest.self, from: data)

        XCTAssertNil(decoded.languageHint)
    }

    // MARK: - StreamingSpeechRecognitionWorkflow protocol conformance

    func testStreamingSpeechRecognitionWorkflowMockConformsToProtocol() async throws {
        let mock = MockStreamingASR()
        let request = StreamingSpeechRecognitionRequest(model: "test/mock-asr")
        let (audioStream, audioContinuation) = AsyncStream.makeStream(of: AudioChunk.self)
        audioContinuation.finish()

        let eventStream = try await mock.transcribeStream(request: request, audioStream: audioStream)
        var events: [SpeechRecognitionEvent] = []
        for try await event in eventStream {
            events.append(event)
        }

        XCTAssertEqual(events.count, 1)
        guard case .completed(let result) = events[0] else {
            XCTFail("Expected .completed event"); return
        }
        XCTAssertEqual(result.text, "mock transcript")
    }
}

// MARK: - Test helpers

private final class MockStreamingASR: StreamingSpeechRecognitionWorkflow {
    let descriptor = ModelDescriptor(
        id: "test/mock-asr",
        displayName: "Mock Streaming ASR",
        domain: .stt,
        capabilities: [.speechRecognition]
    )
    let backendKind: BackendKind = .mock

    func transcribeStream(
        request: StreamingSpeechRecognitionRequest,
        audioStream: AsyncStream<AudioChunk>
    ) async throws -> AsyncThrowingStream<SpeechRecognitionEvent, Error> {
        return AsyncThrowingStream { continuation in
            let meta = BackendMetadata(modelId: request.model.rawValue, backendKind: .mock)
            let result = RichTranscriptionResult(text: "mock transcript", backendMetadata: meta)
            continuation.yield(.completed(result))
            continuation.finish()
        }
    }
}

// MARK: - TranscriptionFormatter tests

final class TranscriptionFormatterTests: XCTestCase {

    // MARK: Helpers

    private func makeResult(
        text: String = "Hello world. Goodbye world.",
        durationSeconds: Double? = 10.0,
        segments: [TranscriptionSegment] = []
    ) -> RichTranscriptionResult {
        let meta = BackendMetadata(modelId: "test", backendKind: .mock)
        return RichTranscriptionResult(
            text: text,
            durationSeconds: durationSeconds,
            segments: segments,
            backendMetadata: meta
        )
    }

    // MARK: - Timestamp formatting

    func testSRTTimestampZero() {
        let ts = TranscriptionFormatter.srtTimestamp(0)
        XCTAssertEqual(ts, "00:00:00,000")
    }

    func testSRTTimestampSubSecond() {
        let ts = TranscriptionFormatter.srtTimestamp(0.5)
        XCTAssertEqual(ts, "00:00:00,500")
    }

    func testSRTTimestampOneHour() {
        let ts = TranscriptionFormatter.srtTimestamp(3661.0)
        XCTAssertEqual(ts, "01:01:01,000")
    }

    func testVTTTimestampDecimalDot() {
        let ts = TranscriptionFormatter.vttTimestamp(62.123)
        XCTAssertEqual(ts, "00:01:02.123")
        XCTAssertTrue(ts.contains("."), "VTT timestamp must use '.' separator")
        XCTAssertFalse(ts.contains(","), "VTT timestamp must not use ',' separator")
    }

    func testSRTTimestampUsesComma() {
        let ts = TranscriptionFormatter.srtTimestamp(1.5)
        XCTAssertTrue(ts.contains(","), "SRT timestamp must use ',' separator")
        XCTAssertFalse(ts.contains("."), "SRT timestamp must not use '.' separator")
    }

    // MARK: - Empty result

    func testSRTEmptySegmentsReturnsEmpty() {
        let result = makeResult(text: "", durationSeconds: nil, segments: [])
        XCTAssertEqual(TranscriptionFormatter.srt(from: result), "")
    }

    func testVTTEmptySegmentsReturnsHeaderOnly() {
        let result = makeResult(text: "", durationSeconds: nil, segments: [])
        XCTAssertEqual(TranscriptionFormatter.vtt(from: result), "WEBVTT")
    }

    // MARK: - Timed segments

    func testSRTWithTimedSegments() {
        let segments: [TranscriptionSegment] = [
            TranscriptionSegment(text: "Hello world.", startTime: 0.0, endTime: 3.0),
            TranscriptionSegment(text: "Goodbye world.", startTime: 3.0, endTime: 7.5),
        ]
        let result = makeResult(segments: segments)
        let srt = TranscriptionFormatter.srt(from: result)

        XCTAssertTrue(srt.hasPrefix("1\n"), "First cue must start with index 1")
        XCTAssertTrue(srt.contains("00:00:00,000 --> 00:00:03,000"), "First cue timing")
        XCTAssertTrue(srt.contains("Hello world."), "First cue text")
        XCTAssertTrue(srt.contains("2\n"), "Second cue index")
        XCTAssertTrue(srt.contains("00:00:03,000 --> 00:00:07,500"), "Second cue timing")
        XCTAssertTrue(srt.contains("Goodbye world."), "Second cue text")
    }

    func testVTTWithTimedSegments() {
        let segments: [TranscriptionSegment] = [
            TranscriptionSegment(text: "Hello world.", startTime: 0.0, endTime: 3.0),
            TranscriptionSegment(text: "Goodbye world.", startTime: 3.0, endTime: 7.5),
        ]
        let result = makeResult(segments: segments)
        let vtt = TranscriptionFormatter.vtt(from: result)

        XCTAssertTrue(vtt.hasPrefix("WEBVTT"), "VTT must begin with WEBVTT header")
        XCTAssertTrue(vtt.contains("00:00:00.000 --> 00:00:03.000"), "First cue timing")
        XCTAssertTrue(vtt.contains("Hello world."), "First cue text")
        XCTAssertTrue(vtt.contains("00:00:03.000 --> 00:00:07.500"), "Second cue timing")
        XCTAssertTrue(vtt.contains("Goodbye world."), "Second cue text")
        // VTT does NOT emit cue indices
        XCTAssertFalse(vtt.contains("\n1\n"), "VTT must not have numeric cue indices")
    }

    // MARK: - Segments without timing (distribution fallback)

    func testSRTDistributesWhenNoTiming() {
        let segments: [TranscriptionSegment] = [
            TranscriptionSegment(text: "Part one."),
            TranscriptionSegment(text: "Part two."),
        ]
        let result = makeResult(durationSeconds: 10.0, segments: segments)
        let srt = TranscriptionFormatter.srt(from: result)

        XCTAssertTrue(srt.contains("00:00:00,000 --> 00:00:05,000"), "First cue: 0–5 s")
        XCTAssertTrue(srt.contains("00:00:05,000 --> 00:00:10,000"), "Second cue: 5–10 s")
        XCTAssertTrue(srt.contains("Part one."))
        XCTAssertTrue(srt.contains("Part two."))
    }

    func testVTTDistributesWhenNoTimingOrDuration() {
        let segments: [TranscriptionSegment] = [
            TranscriptionSegment(text: "Alpha."),
            TranscriptionSegment(text: "Beta."),
        ]
        let result = makeResult(durationSeconds: nil, segments: segments)
        let vtt = TranscriptionFormatter.vtt(from: result)

        // 5 s default per segment
        XCTAssertTrue(vtt.contains("00:00:00.000 --> 00:00:05.000"), "First cue default 5 s")
        XCTAssertTrue(vtt.contains("00:00:05.000 --> 00:00:10.000"), "Second cue default 5 s")
    }

    // MARK: - Single segment

    func testSRTSingleTimedSegment() {
        let segments = [TranscriptionSegment(text: "Solo.", startTime: 1.0, endTime: 4.0)]
        let result = makeResult(text: "Solo.", segments: segments)
        let srt = TranscriptionFormatter.srt(from: result)

        XCTAssertTrue(srt.hasPrefix("1\n"))
        XCTAssertTrue(srt.contains("00:00:01,000 --> 00:00:04,000"))
        XCTAssertTrue(srt.contains("Solo."))
        // No trailing blank line / second cue
        XCTAssertFalse(srt.contains("\n\n2\n"))
    }

    // MARK: - Missing endTime inferred from next startTime

    func testSRTInfersEndTimeFromNextSegmentStart() {
        let segments: [TranscriptionSegment] = [
            TranscriptionSegment(text: "First.", startTime: 0.0, endTime: nil),
            TranscriptionSegment(text: "Second.", startTime: 4.0, endTime: 8.0),
        ]
        let result = makeResult(segments: segments)
        let srt = TranscriptionFormatter.srt(from: result)

        // First cue's end should be inferred as 4.0 (next segment start)
        XCTAssertTrue(srt.contains("00:00:00,000 --> 00:00:04,000"), "End inferred from next start")
        XCTAssertTrue(srt.contains("00:00:04,000 --> 00:00:08,000"))
    }

    // MARK: - Whitespace-only segments filtered out

    func testBlankSegmentsAreSkipped() {
        let segments: [TranscriptionSegment] = [
            TranscriptionSegment(text: "Hello.", startTime: 0.0, endTime: 2.0),
            TranscriptionSegment(text: "   ", startTime: 2.0, endTime: 3.0),
            TranscriptionSegment(text: "World.", startTime: 3.0, endTime: 5.0),
        ]
        let result = makeResult(segments: segments)
        let srt = TranscriptionFormatter.srt(from: result)
        let cueCount = srt.components(separatedBy: "\n\n").count
        XCTAssertEqual(cueCount, 2, "Whitespace-only segment must be skipped")
    }
}

final class TranscriptionResponseFormatTests: XCTestCase {
    private func makeResult() -> RichTranscriptionResult {
        let meta = BackendMetadata(modelId: "test", backendKind: .mock)
        return RichTranscriptionResult(
            text: "Hello world.",
            durationSeconds: 3,
            segments: [
                TranscriptionSegment(text: "Hello world.", startTime: 0, endTime: 3)
            ],
            backendMetadata: meta
        )
    }

    func testParsesKnownAPIValuesCaseInsensitively() {
        XCTAssertEqual(TranscriptionResponseFormat(apiValue: "text"), .text)
        XCTAssertEqual(TranscriptionResponseFormat(apiValue: "JSON"), .json)
        XCTAssertEqual(TranscriptionResponseFormat(apiValue: "Verbose_JSON"), .verbose_json)
        XCTAssertEqual(TranscriptionResponseFormat(apiValue: "srt"), .srt)
        XCTAssertEqual(TranscriptionResponseFormat(apiValue: "VTT"), .vtt)
    }

    func testRejectsUnknownOrEmptyAPIValue() {
        XCTAssertNil(TranscriptionResponseFormat(apiValue: nil))
        XCTAssertNil(TranscriptionResponseFormat(apiValue: ""))
        XCTAssertNil(TranscriptionResponseFormat(apiValue: "markdown"))
    }

    func testRendersTextFormats() throws {
        let result = makeResult()

        XCTAssertEqual(try TranscriptionResponseFormat.text.render(result), "Hello world.")
        XCTAssertTrue(try TranscriptionResponseFormat.srt.render(result).contains("00:00:00,000 --> 00:00:03,000"))
        XCTAssertTrue(try TranscriptionResponseFormat.vtt.render(result).hasPrefix("WEBVTT"))
    }

    func testRendersJSONFormats() throws {
        let result = makeResult()

        let jsonText = try TranscriptionResponseFormat.json.render(result)
        XCTAssertTrue(jsonText.contains("\"text\":\"Hello world.\""))

        let verboseText = try TranscriptionResponseFormat.verbose_json.render(result)
        XCTAssertTrue(verboseText.contains("\"segments\""))
    }
}
