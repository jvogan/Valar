import Foundation
import MLX
import MLXAudioCore
import MLXAudioSTT
@testable import MLXAudioTTS
import MLXLMCommon
import Testing
@testable import ValarMLX
import ValarModelKit

@Suite("MLX Backend")
struct MLXBackendTests {
    @Test("Reports correct backend kind and capabilities")
    func backendCapabilities() async {
        let backend = MLXInferenceBackend()
        #expect(backend.backendKind == .mlx)
        let caps = backend.runtimeCapabilities
        #expect(caps.features.contains(.streamingSynthesis))
        #expect(caps.features.contains(.streamingRecognition))
        #expect(caps.supportedFamilies.contains(.qwen3TTS))
        #expect(caps.supportedFamilies.contains(.qwen3ASR))
        #expect(caps.supportedFamilies.contains(.qwen3ForcedAligner))
        #expect(caps.supportedFamilies.contains(.soprano))
        #expect(caps.supportedFamilies.contains(.voxtralTTS))
    }

    @Test("Load and unload model")
    func loadUnload() async throws {
        let backend = MLXInferenceBackend(
            modelDirectoryResolver: { _ in nil },
            warningHandler: { _ in },
            qwenModelLoader: { descriptor in
                MLXModelHandle(descriptor: descriptor)
            }
        )
        let descriptor = ModelDescriptor(
            id: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
            familyID: .qwen3TTS,
            displayName: "Test Model",
            domain: .tts,
            capabilities: [.speechSynthesis]
        )
        let config = ModelRuntimeConfiguration(backendKind: .mlx)

        let model = try await backend.loadModel(descriptor: descriptor, configuration: config)
        #expect(model.descriptor.id == descriptor.id)
        #expect(model.backendKind == .mlx)

        let count = await backend.loadedModelCount
        #expect(count == 1)

        try await backend.unloadModel(model)
        let countAfter = await backend.loadedModelCount
        #expect(countAfter == 0)
    }

    @Test("Returns cached model on duplicate load")
    func cacheHit() async throws {
        let backend = MLXInferenceBackend(
            modelDirectoryResolver: { _ in nil },
            warningHandler: { _ in },
            qwenModelLoader: { descriptor in
                MLXModelHandle(descriptor: descriptor)
            }
        )
        let descriptor = ModelDescriptor(
            id: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
            familyID: .qwen3TTS,
            displayName: "Cached",
            domain: .tts,
            capabilities: [.speechSynthesis]
        )
        let config = ModelRuntimeConfiguration(backendKind: .mlx)

        let first = try await backend.loadModel(descriptor: descriptor, configuration: config)
        let second = try await backend.loadModel(descriptor: descriptor, configuration: config)
        #expect(first.descriptor.id == second.descriptor.id)

        let count = await backend.loadedModelCount
        #expect(count == 1)
    }

    @Test("Load and unload second TTS family through the shared loader dispatch")
    func loadUnloadSecondTTsFamily() async throws {
        let backend = MLXInferenceBackend(
            modelDirectoryResolver: { _ in nil },
            warningHandler: { _ in },
            qwenModelLoader: { descriptor in
                MLXModelHandle(descriptor: descriptor)
            }
        )
        let descriptor = ModelDescriptor(
            id: "mlx-community/Soprano-1.1-80M-bf16",
            familyID: .soprano,
            displayName: "Soprano 1.1 80M",
            domain: .tts,
            capabilities: [.speechSynthesis, .tokenization]
        )
        let config = ModelRuntimeConfiguration(backendKind: .mlx)

        let model = try await backend.loadModel(descriptor: descriptor, configuration: config)
        #expect(model.descriptor.familyID == .soprano)

        let count = await backend.loadedModelCount
        #expect(count == 1)

        try await backend.unloadModel(model)
        let countAfter = await backend.loadedModelCount
        #expect(countAfter == 0)
    }

    @Test("Load Qwen3 ASR through the backend dispatch")
    func loadASRFamily() async throws {
        let backend = MLXInferenceBackend(
            modelDirectoryResolver: { _ in nil },
            warningHandler: { _ in },
            qwenModelLoader: { descriptor in
                MLXModelHandle(descriptor: descriptor)
            },
            qwenASRModelLoader: { descriptor in
                MLXASRModelHandle(descriptor: descriptor, mlxSTTModel: StubSTTModel())
            }
        )
        let descriptor = ModelDescriptor(
            id: "mlx-community/Qwen3-ASR-0.6B-8bit",
            familyID: .qwen3ASR,
            displayName: "Qwen3 ASR 0.6B",
            domain: .stt,
            capabilities: [.speechRecognition, .translation]
        )
        let config = ModelRuntimeConfiguration(backendKind: .mlx)

        let model = try await backend.loadModel(descriptor: descriptor, configuration: config)
        #expect(model.descriptor.familyID == .qwen3ASR)
        #expect(model is any SpeechToTextWorkflow)
        #expect(model is MLXASRModelHandle)

        try await backend.unloadModel(model)
    }

    @Test("Load Qwen3 forced aligner through the backend dispatch")
    func loadForcedAlignerFamily() async throws {
        let alignerLoaderCalls = LockedCounter()
        let backend = MLXInferenceBackend(
            modelDirectoryResolver: { _ in nil },
            warningHandler: { _ in },
            qwenModelLoader: { descriptor in
                MLXModelHandle(descriptor: descriptor)
            },
            qwenASRModelLoader: { descriptor in
                MLXASRModelHandle(descriptor: descriptor, mlxSTTModel: StubSTTModel())
            },
            qwenAlignerModelLoader: { descriptor in
                alignerLoaderCalls.increment()
                return MLXAlignerModelHandle(descriptor: descriptor)
            }
        )
        let descriptor = ModelDescriptor(
            id: "mlx-community/Qwen3-ForcedAligner-0.6B-8bit",
            familyID: .qwen3ForcedAligner,
            displayName: "Qwen3 ForcedAligner 0.6B",
            domain: .stt,
            capabilities: [.forcedAlignment, .tokenization]
        )
        let config = ModelRuntimeConfiguration(backendKind: .mlx)

        let model = try await backend.loadModel(descriptor: descriptor, configuration: config)
        #expect(model.descriptor.familyID == .qwen3ForcedAligner)
        #expect(model is MLXAlignerModelHandle)
        #expect(alignerLoaderCalls.value == 1)

        try await backend.unloadModel(model)
    }

    @Test("Load Voxtral TTS through the dedicated backend dispatch")
    func loadVoxtralFamily() async throws {
        let voxtralLoaderCalls = LockedCounter()
        let backend = MLXInferenceBackend(
            modelDirectoryResolver: { _ in nil },
            warningHandler: { _ in },
            qwenModelLoader: { descriptor in
                MLXModelHandle(descriptor: descriptor)
            },
            voxtralTTSModelLoader: { descriptor in
                voxtralLoaderCalls.increment()
                return MLXModelHandle(descriptor: descriptor, mlxModel: RecordingSpeechModel())
            }
        )
        let descriptor = ModelDescriptor(
            id: "mistralai/Voxtral-4B-TTS-2603",
            familyID: .voxtralTTS,
            displayName: "Voxtral 4B TTS 2603",
            domain: .tts,
            capabilities: [.speechSynthesis, .multilingual, .presetVoices, .streaming]
        )
        let config = ModelRuntimeConfiguration(backendKind: .mlx)

        let model = try await backend.loadModel(descriptor: descriptor, configuration: config)
        #expect(model.descriptor.familyID == .voxtralTTS)
        #expect(model is any TextToSpeechWorkflow)
        #expect(voxtralLoaderCalls.value == 1)

        try await backend.unloadModel(model)
    }

}

@Suite("Qwen long-form cache policy")
struct QwenLongFormCachePolicyTests {
    @Test("does not trim cache on medium segmented runs")
    func avoidsMediumRunTrim() {
        #expect(
            MLXModelHandle.shouldTrimQwenCache(
                afterCompletedSegment: 4,
                totalSegments: 6,
                detectedMemoryGrowth: true,
                isUnderMemoryPressure: false
            ) == false
        )
        #expect(
            MLXModelHandle.shouldTrimQwenCache(
                afterCompletedSegment: 8,
                totalSegments: 10,
                detectedMemoryGrowth: true,
                isUnderMemoryPressure: true
            ) == false
        )
    }

    @Test("trims cache only on long runs with real pressure signals")
    func trimsOnlyUnderPressureSignals() {
        #expect(
            MLXModelHandle.shouldTrimQwenCache(
                afterCompletedSegment: 4,
                totalSegments: 12,
                detectedMemoryGrowth: false,
                isUnderMemoryPressure: false
            ) == false
        )
        #expect(
            MLXModelHandle.shouldTrimQwenCache(
                afterCompletedSegment: 8,
                totalSegments: 12,
                detectedMemoryGrowth: true,
                isUnderMemoryPressure: false
            ) == true
        )
        #expect(
            MLXModelHandle.shouldTrimQwenCache(
                afterCompletedSegment: 5,
                totalSegments: 20,
                detectedMemoryGrowth: false,
                isUnderMemoryPressure: true
            ) == true
        )
    }

    @Test("detects meaningful long-run memory growth")
    func detectsMeaningfulLongRunMemoryGrowth() {
        #expect(
            MLXModelHandle.detectedQwenLongFormMemoryGrowth(
                previousPeakBytes: 4 * 1_024 * 1_024 * 1_024,
                currentPeakBytes: 4 * 1_024 * 1_024 * 1_024 + 256 * 1_024 * 1_024
            ) == false
        )
        #expect(
            MLXModelHandle.detectedQwenLongFormMemoryGrowth(
                previousPeakBytes: 4 * 1_024 * 1_024 * 1_024,
                currentPeakBytes: 4 * 1_024 * 1_024 * 1_024 + 640 * 1_024 * 1_024
            ) == true
        )
    }
}

@Suite("MLX ASR Transcription")
struct MLXASRTranscriptionTests {

    @Test("transcribe returns RichTranscriptionResult with text from STTOutput")
    func transcribeReturnsRichResult() async throws {
        let modelID = ModelIdentifier("mlx-community/Qwen3-ASR-0.6B-8bit")
        let descriptor = ModelDescriptor(
            id: modelID,
            familyID: .qwen3ASR,
            displayName: "Qwen3 ASR 0.6B",
            domain: .stt,
            capabilities: [.speechRecognition]
        )
        let handle = MLXASRModelHandle(descriptor: descriptor, mlxSTTModel: StubSTTModel())
        let session = ModelRuntimeSession(
            descriptor: descriptor,
            backendKind: .mlx,
            configuration: ModelRuntimeConfiguration(backendKind: .mlx),
            state: .resident
        )
        let request = SpeechRecognitionRequest(
            model: modelID,
            audio: AudioChunk(samples: [0.1, -0.2, 0.3], sampleRate: 16_000)
        )

        let result = try await handle.transcribe(request: request, in: session)

        #expect(result.text == "stub transcript")
        #expect(result.backendMetadata.backendKind == BackendKind.mlx)
        #expect(result.backendMetadata.modelId == modelID.rawValue)
        #expect(result.segments.count == 1)
        #expect(result.segments[0].text == "stub segment")
    }

    @Test("richTranscriptionResult maps STTOutput fields including language and timing")
    func richTranscriptionResultMapsAllFields() {
        let output = STTOutput(
            text: "hello world",
            segments: [
                ["text": "hello", "start": 0.0, "end": 0.5],
                ["text": "world", "start": 0.6, "end": 1.1],
            ],
            language: "English",
            totalTime: 1.23
        )
        let modelId = ModelIdentifier("test/model")
        let result = MLXModelHandle.richTranscriptionResult(from: output, modelId: modelId)

        #expect(result.text == "hello world")
        #expect(result.language == "English")
        #expect(result.durationSeconds == 1.23)
        #expect(result.segments.count == 2)
        #expect(result.segments[0].text == "hello")
        #expect(result.segments[0].startTime == 0.0)
        #expect(result.segments[0].endTime == 0.5)
        #expect(result.segments[1].text == "world")
        #expect(result.backendMetadata.backendKind == .mlx)
        #expect(result.backendMetadata.modelId == "test/model")
        #expect(result.backendMetadata.inferenceTimeSeconds == 1.23)
    }

    @Test("richTranscriptionResult with zero totalTime produces nil timing fields")
    func richTranscriptionResultZeroTimingIsNil() {
        let output = STTOutput(text: "text", totalTime: 0.0)
        let result = MLXModelHandle.richTranscriptionResult(from: output, modelId: ModelIdentifier("m"))

        #expect(result.durationSeconds == nil)
        #expect(result.backendMetadata.inferenceTimeSeconds == nil)
    }

    @Test("richTranscriptionResult with missing text in segment skips that segment")
    func richTranscriptionResultSkipsMalformedSegments() {
        let output = STTOutput(
            text: "ok",
            segments: [
                ["notext": "ignored"],
                ["text": "ok"],
            ]
        )
        let result = MLXModelHandle.richTranscriptionResult(from: output, modelId: ModelIdentifier("m"))

        #expect(result.segments.count == 1)
        #expect(result.segments[0].text == "ok")
    }
}

private final class StubSTTModel: STTGenerationModel {
    var defaultGenerationParameters: STTGenerateParameters {
        STTGenerateParameters(language: "English")
    }

    func generate(audio: MLXArray, generationParameters: STTGenerateParameters) -> STTOutput {
        #expect(audio.asArray(Float.self) == [0.1, -0.2, 0.3])
        #expect(generationParameters.language == "English")
        return STTOutput(
            text: "stub transcript",
            segments: [["text": "stub segment"]]
        )
    }

    func generateStream(
        audio: MLXArray,
        generationParameters: STTGenerateParameters
    ) -> AsyncThrowingStream<STTGeneration, Error> {
        AsyncThrowingStream { continuation in
            continuation.yield(.result(generate(audio: audio, generationParameters: generationParameters)))
            continuation.finish()
        }
    }
}

private final class RecordingSpeechModel: SpeechGenerationModel {
    let sampleRate: Int = 24_000
    var defaultGenerationParameters: GenerateParameters { GenerateParameters() }

    private(set) var generatedLanguages: [String?] = []
    private(set) var streamedLanguages: [String?] = []

    func generate(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) async throws -> MLXArray {
        _ = text
        _ = voice
        _ = refAudio
        _ = refText
        _ = generationParameters
        generatedLanguages.append(language)
        return MLXArray([0.1 as Float, -0.1 as Float, 0.05 as Float])
    }

    func generateStream(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        _ = text
        _ = voice
        _ = refAudio
        _ = refText
        _ = generationParameters
        streamedLanguages.append(language)
        return AsyncThrowingStream { continuation in
            continuation.yield(.audio(MLXArray([0.25 as Float, -0.25 as Float])))
            continuation.finish()
        }
    }

    func generateStream(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters,
        streamingInterval: Double
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        _ = streamingInterval
        return generateStream(
            text: text,
            voice: voice,
            refAudio: refAudio,
            refText: refText,
            language: language,
            generationParameters: generationParameters
        )
    }
}

private final class LockedCounter: @unchecked Sendable {
    private let lock = NSLock()
    private var storage: Int = 0

    var value: Int {
        lock.withLock { storage }
    }

    func increment() {
        lock.withLock { storage += 1 }
    }
}

@Suite("MLX Model Directory Security")
struct MLXModelDirectorySecurityTests {
    @Test("Unsafe weight formats are rejected before loading")
    func rejectsUnsafeWeightFormats() async throws {
        for fileExtension in ["bin", "pkl", "pt"] {
            let directory = try makeTemporaryModelDirectory()
            defer { try? FileManager.default.removeItem(at: directory) }

            try Data().write(to: directory.appendingPathComponent("weights.\(fileExtension)"))

            let backend = MLXInferenceBackend(
                modelDirectoryResolver: { _ in directory },
                warningHandler: { _ in },
                qwenModelLoader: { descriptor in
                    MLXModelHandle(descriptor: descriptor)
                }
            )

            do {
                _ = try await backend.loadModel(
                    descriptor: securityTestDescriptor(),
                    configuration: ModelRuntimeConfiguration(backendKind: .mlx)
                )
                Issue.record("Expected unsafe .\(fileExtension) file to be rejected")
            } catch let error as MLXBackendError {
                guard case .rejectedUnsafeWeightFile(_, let rejectedExtension) = error else {
                    Issue.record("Expected rejectedUnsafeWeightFile, got \(error)")
                    continue
                }
                #expect(rejectedExtension == fileExtension)
                #expect(error.errorDescription?.contains("Only .safetensors weights are allowed.") == true)
            } catch {
                Issue.record("Expected MLXBackendError, got \(error)")
            }
        }
    }

    @Test("Safetensors files are validated by header bytes before loading")
    func rejectsInvalidSafeTensorsHeader() async throws {
        let directory = try makeTemporaryModelDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        try Data("not-safetensors".utf8).write(to: directory.appendingPathComponent("model.safetensors"))

        let backend = MLXInferenceBackend(
            modelDirectoryResolver: { _ in directory },
            warningHandler: { _ in },
            qwenModelLoader: { descriptor in
                MLXModelHandle(descriptor: descriptor)
            }
        )

        do {
            _ = try await backend.loadModel(
                descriptor: securityTestDescriptor(),
                configuration: ModelRuntimeConfiguration(backendKind: .mlx)
            )
            Issue.record("Expected invalid safetensors header to be rejected")
        } catch let error as MLXBackendError {
            guard case .invalidSafeTensorsHeader = error else {
                Issue.record("Expected invalidSafeTensorsHeader, got \(error)")
                return
            }
            #expect(error.errorDescription?.contains("magic bytes") == true)
        } catch {
            Issue.record("Expected MLXBackendError, got \(error)")
        }
    }

    @Test("Unexpected files are logged as warnings")
    func warnsOnUnexpectedFiles() async throws {
        let directory = try makeTemporaryModelDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        try validSafeTensorsFixture().write(to: directory.appendingPathComponent("model.safetensors"))
        try Data("debug".utf8).write(to: directory.appendingPathComponent("notes.md"))

        let warnings = WarningRecorder()
        let backend = MLXInferenceBackend(
            modelDirectoryResolver: { _ in directory },
            warningHandler: { message in
                warnings.append(message)
            },
            qwenModelLoader: { descriptor in
                MLXModelHandle(descriptor: descriptor)
            }
        )

        _ = try await backend.loadModel(
            descriptor: securityTestDescriptor(),
            configuration: ModelRuntimeConfiguration(backendKind: .mlx)
        )

        #expect(warnings.messages.count == 1)
        #expect(warnings.messages[0].contains("Unexpected file in model directory"))
        #expect(warnings.messages[0].contains("notes.md"))
    }

    @Test("Voxtral preset voice assets are allowed during directory validation")
    func allowsVoxtralPresetVoiceAssets() async throws {
        let directory = try makeTemporaryModelDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        try validSafeTensorsFixture().write(to: directory.appendingPathComponent("consolidated.safetensors"))
        try Data("{}".utf8).write(to: directory.appendingPathComponent("params.json"))
        try Data("{}".utf8).write(to: directory.appendingPathComponent("tekken.json"))

        let voiceDirectory = directory.appendingPathComponent("voice_embedding_safe", isDirectory: true)
        try FileManager.default.createDirectory(at: voiceDirectory, withIntermediateDirectories: true)
        try Data([0x01, 0x02, 0x03]).write(to: voiceDirectory.appendingPathComponent("neutral_female.bin"))
        try Data("{}".utf8).write(to: voiceDirectory.appendingPathComponent("index.json"))

        let warnings = WarningRecorder()
        let backend = MLXInferenceBackend(
            modelDirectoryResolver: { _ in directory },
            warningHandler: { message in
                warnings.append(message)
            },
            qwenModelLoader: { descriptor in
                MLXModelHandle(descriptor: descriptor)
            },
            voxtralTTSModelLoader: { descriptor in
                MLXModelHandle(descriptor: descriptor, mlxModel: RecordingSpeechModel())
            }
        )
        let descriptor = ModelDescriptor(
            id: "mistralai/Voxtral-4B-TTS-2603",
            familyID: .voxtralTTS,
            displayName: "Voxtral 4B TTS 2603",
            domain: .tts,
            capabilities: [.speechSynthesis, .multilingual, .presetVoices, .streaming]
        )

        let model = try await backend.loadModel(
            descriptor: descriptor,
            configuration: ModelRuntimeConfiguration(backendKind: .mlx)
        )

        #expect(model.descriptor.id == descriptor.id)
        #expect(warnings.messages.isEmpty)
    }

    @Test("Nested safetensors do not satisfy the root weight requirement")
    func nestedSafeTensorsStillRequireRootWeights() async throws {
        let directory = try makeTemporaryModelDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let nestedDirectory = directory.appendingPathComponent("speech_tokenizer", isDirectory: true)
        try FileManager.default.createDirectory(at: nestedDirectory, withIntermediateDirectories: true)
        try validSafeTensorsFixture().write(to: nestedDirectory.appendingPathComponent("decoder.safetensors"))

        let backend = MLXInferenceBackend(
            modelDirectoryResolver: { _ in directory },
            warningHandler: { _ in },
            qwenModelLoader: { descriptor in
                MLXModelHandle(descriptor: descriptor)
            }
        )

        do {
            _ = try await backend.loadModel(
                descriptor: securityTestDescriptor(),
                configuration: ModelRuntimeConfiguration(backendKind: .mlx)
            )
            Issue.record("Expected missing root safetensors weights to be rejected")
        } catch let error as MLXBackendError {
            guard case .missingSafeTensorsWeights(let path) = error else {
                Issue.record("Expected missingSafeTensorsWeights, got \(error)")
                return
            }
            #expect(path == directory.path)
        } catch {
            Issue.record("Expected MLXBackendError, got \(error)")
        }
    }

    @Test("Expected speech tokenizer assets do not warn")
    func expectedSpeechTokenizerAssetsDoNotWarn() async throws {
        let directory = try makeTemporaryModelDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        try validSafeTensorsFixture().write(to: directory.appendingPathComponent("model.safetensors"))

        let nestedDirectory = directory.appendingPathComponent("speech_tokenizer", isDirectory: true)
        try FileManager.default.createDirectory(at: nestedDirectory, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: nestedDirectory.appendingPathComponent("config.json"))
        try validSafeTensorsFixture().write(to: nestedDirectory.appendingPathComponent("decoder.safetensors"))

        let warnings = WarningRecorder()
        let backend = MLXInferenceBackend(
            modelDirectoryResolver: { _ in directory },
            warningHandler: { message in
                warnings.append(message)
            },
            qwenModelLoader: { descriptor in
                MLXModelHandle(descriptor: descriptor)
            }
        )

        _ = try await backend.loadModel(
            descriptor: securityTestDescriptor(),
            configuration: ModelRuntimeConfiguration(backendKind: .mlx)
        )

        #expect(warnings.messages.isEmpty)
    }

    @Test("Voxtral normalized voice embedding bins are allowed")
    func expectedVoxtralVoiceEmbeddingAssetsDoNotWarn() async throws {
        let directory = try makeTemporaryModelDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        try validSafeTensorsFixture().write(to: directory.appendingPathComponent("model.safetensors"))
        try Data("{}".utf8).write(to: directory.appendingPathComponent("params.json"))
        try Data("{}".utf8).write(to: directory.appendingPathComponent("tekken.json"))

        let safeDirectory = directory.appendingPathComponent("voice_embedding_safe", isDirectory: true)
        try FileManager.default.createDirectory(at: safeDirectory, withIntermediateDirectories: true)
        try Data([0, 0, 0, 0]).write(to: safeDirectory.appendingPathComponent("casual_female.bin"))
        try Data(#"{"voices":[]}"#.utf8).write(to: safeDirectory.appendingPathComponent("index.json"))

        let warnings = WarningRecorder()
        let backend = MLXInferenceBackend(
            modelDirectoryResolver: { _ in directory },
            warningHandler: { message in warnings.append(message) },
            qwenModelLoader: { descriptor in MLXModelHandle(descriptor: descriptor) }
        )

        _ = try await backend.loadModel(
            descriptor: securityTestDescriptor(),
            configuration: ModelRuntimeConfiguration(backendKind: .mlx)
        )

        #expect(warnings.messages.isEmpty)
    }

    @Test("Voxtral MLX community voice embedding safetensors are allowed")
    func expectedVoxtralVoiceEmbeddingSafeTensorsDoNotWarn() async throws {
        let directory = try makeTemporaryModelDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        try validSafeTensorsFixture().write(to: directory.appendingPathComponent("model.safetensors"))
        try Data("{}".utf8).write(to: directory.appendingPathComponent("params.json"))
        try Data("{}".utf8).write(to: directory.appendingPathComponent("tekken.json"))

        let voiceDirectory = directory.appendingPathComponent("voice_embedding", isDirectory: true)
        try FileManager.default.createDirectory(at: voiceDirectory, withIntermediateDirectories: true)
        try validSafeTensorsFixture().write(to: voiceDirectory.appendingPathComponent("neutral_female.safetensors"))

        let warnings = WarningRecorder()
        try await MLXInferenceBackend.validateModelDirectoryForTests(
            directory,
            familyID: .voxtralTTS,
            warningHandler: { message in warnings.append(message) }
        )

        #expect(warnings.messages.isEmpty)
    }

    @Test("Expected root metadata files do not warn")
    func expectedRootMetadataFilesDoNotWarn() async throws {
        let directory = try makeTemporaryModelDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        try validSafeTensorsFixture().write(to: directory.appendingPathComponent("model.safetensors"))
        try Data("{}".utf8).write(to: directory.appendingPathComponent("config.json"))
        try Data("{}".utf8).write(to: directory.appendingPathComponent("tokenizer.json"))
        try Data("{}".utf8).write(to: directory.appendingPathComponent("tokenizer_config.json"))
        try Data("{}".utf8).write(to: directory.appendingPathComponent("vocab.json"))
        try Data("merge".utf8).write(to: directory.appendingPathComponent("merges.txt"))

        let warnings = WarningRecorder()
        let backend = MLXInferenceBackend(
            modelDirectoryResolver: { _ in directory },
            warningHandler: { message in
                warnings.append(message)
            },
            qwenModelLoader: { descriptor in
                MLXModelHandle(descriptor: descriptor)
            }
        )

        _ = try await backend.loadModel(
            descriptor: securityTestDescriptor(),
            configuration: ModelRuntimeConfiguration(backendKind: .mlx)
        )

        #expect(warnings.messages.isEmpty)
    }

    @Test("Unexpected root JSON files are logged as warnings")
    func unexpectedRootJSONFilesWarn() async throws {
        let directory = try makeTemporaryModelDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        try validSafeTensorsFixture().write(to: directory.appendingPathComponent("model.safetensors"))
        try Data("{}".utf8).write(to: directory.appendingPathComponent("special_tokens_map.json"))

        let warnings = WarningRecorder()
        let backend = MLXInferenceBackend(
            modelDirectoryResolver: { _ in directory },
            warningHandler: { message in
                warnings.append(message)
            },
            qwenModelLoader: { descriptor in
                MLXModelHandle(descriptor: descriptor)
            }
        )

        _ = try await backend.loadModel(
            descriptor: securityTestDescriptor(),
            configuration: ModelRuntimeConfiguration(backendKind: .mlx)
        )

        #expect(warnings.count(matching: "special_tokens_map.json") == 1)
    }

    @Test("Multiple valid safetensors shards all pass parallel header validation")
    func multipleValidShardsPassValidation() async throws {
        let directory = try makeTemporaryModelDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        // Write three valid sharded weight files — simulates a multi-shard model.
        for name in ["model-00001-of-00003.safetensors",
                     "model-00002-of-00003.safetensors",
                     "model-00003-of-00003.safetensors"] {
            try validSafeTensorsFixture().write(to: directory.appendingPathComponent(name))
        }

        let backend = MLXInferenceBackend(
            modelDirectoryResolver: { _ in directory },
            warningHandler: { _ in },
            qwenModelLoader: { descriptor in MLXModelHandle(descriptor: descriptor) }
        )

        // All three shards have valid headers — load must succeed.
        let model = try await backend.loadModel(
            descriptor: securityTestDescriptor(),
            configuration: ModelRuntimeConfiguration(backendKind: .mlx)
        )
        #expect(model.backendKind == .mlx)
    }

    @Test("One invalid shard among multiple causes parallel validation to fail")
    func oneInvalidShardFailsParallelValidation() async throws {
        let directory = try makeTemporaryModelDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        try validSafeTensorsFixture().write(to: directory.appendingPathComponent("model-00001-of-00002.safetensors"))
        // Second shard has a corrupted header.
        try Data("corrupted".utf8).write(to: directory.appendingPathComponent("model-00002-of-00002.safetensors"))

        let backend = MLXInferenceBackend(
            modelDirectoryResolver: { _ in directory },
            warningHandler: { _ in },
            qwenModelLoader: { descriptor in MLXModelHandle(descriptor: descriptor) }
        )

        do {
            _ = try await backend.loadModel(
                descriptor: securityTestDescriptor(),
                configuration: ModelRuntimeConfiguration(backendKind: .mlx)
            )
            Issue.record("Expected corrupted shard to be rejected")
        } catch let error as MLXBackendError {
            guard case .invalidSafeTensorsHeader = error else {
                Issue.record("Expected invalidSafeTensorsHeader, got \(error)")
                return
            }
            #expect(error.errorDescription?.contains("magic bytes") == true)
        } catch {
            Issue.record("Expected MLXBackendError, got \(error)")
        }
    }

    @Test("Safetensors symlink escaping the model directory is rejected")
    func rejectsSafetensorsSymlinkEscapingModelDirectory() async throws {
        let baseDir = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: baseDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: baseDir) }

        // A valid safetensors file that lives OUTSIDE the model directory.
        let externalTarget = baseDir.appendingPathComponent("external.safetensors")
        try validSafeTensorsFixture().write(to: externalTarget)

        // The model directory contains only a symlink pointing to the outside file.
        let modelDir = baseDir.appendingPathComponent("model", isDirectory: true)
        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)
        let symlink = modelDir.appendingPathComponent("model.safetensors")
        try FileManager.default.createSymbolicLink(at: symlink, withDestinationURL: externalTarget)

        let backend = MLXInferenceBackend(
            modelDirectoryResolver: { _ in modelDir },
            warningHandler: { _ in },
            qwenModelLoader: { descriptor in MLXModelHandle(descriptor: descriptor) }
        )

        do {
            _ = try await backend.loadModel(
                descriptor: securityTestDescriptor(),
                configuration: ModelRuntimeConfiguration(backendKind: .mlx)
            )
            // If isRegularFile filters the symlink, we get missingSafeTensorsWeights.
            // If it doesn't, we must get pathTraversalDetected.
            // Both outcomes are correct security behavior — the load must not succeed.
            Issue.record("Expected a security error but load succeeded")
        } catch let error as MLXBackendError {
            switch error {
            case .pathTraversalDetected(let path):
                #expect(path.contains("model.safetensors"))
                #expect(error.errorDescription?.contains("model directory") == true)
            case .missingSafeTensorsWeights:
                // Symlink was filtered by isRegularFile before the containment check.
                // The safeguard held — load correctly rejected.
                break
            default:
                Issue.record("Expected pathTraversalDetected or missingSafeTensorsWeights, got \(error)")
            }
        } catch {
            Issue.record("Expected MLXBackendError, got \(error)")
        }
    }

    @Test("Directory validation is skipped in extractSpeakerEmbedding after prior loadModel")
    func validationSkippedInEmbeddingExtractionAfterLoad() async throws {
        let directory = try makeTemporaryModelDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        try validSafeTensorsFixture().write(to: directory.appendingPathComponent("model.safetensors"))

        let backend = MLXInferenceBackend(
            modelDirectoryResolver: { _ in directory },
            warningHandler: { _ in },
            qwenModelLoader: { descriptor in MLXModelHandle(descriptor: descriptor) },
            speakerEmbeddingExtractor: { _, _ in [Float(0.5)] }
        )

        // First load validates and caches the directory.
        _ = try await backend.loadModel(
            descriptor: securityTestDescriptor(),
            configuration: ModelRuntimeConfiguration(backendKind: .mlx)
        )

        // Remove the safetensors file — a fresh validation pass would now throw
        // missingSafeTensorsWeights. Extraction must succeed via the cache.
        try FileManager.default.removeItem(at: directory.appendingPathComponent("model.safetensors"))

        let data = try await backend.extractSpeakerEmbedding(
            descriptor: securityTestDescriptor(),
            monoReferenceSamples: [0.1]
        )
        #expect(!data.isEmpty)
    }

    @Test("Directory validation is skipped on repeated extractSpeakerEmbedding calls")
    func validationSkippedOnRepeatedEmbeddingExtraction() async throws {
        let directory = try makeTemporaryModelDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        try validSafeTensorsFixture().write(to: directory.appendingPathComponent("model.safetensors"))

        let backend = MLXInferenceBackend(
            modelDirectoryResolver: { _ in directory },
            warningHandler: { _ in },
            qwenModelLoader: { descriptor in MLXModelHandle(descriptor: descriptor) },
            speakerEmbeddingExtractor: { _, _ in [Float(0.5)] }
        )

        // First extraction validates and caches the directory.
        _ = try await backend.extractSpeakerEmbedding(
            descriptor: securityTestDescriptor(),
            monoReferenceSamples: [0.1]
        )

        // Remove the safetensors file — a fresh validation pass would now throw.
        try FileManager.default.removeItem(at: directory.appendingPathComponent("model.safetensors"))

        // Second extraction must succeed via the cache.
        let data = try await backend.extractSpeakerEmbedding(
            descriptor: securityTestDescriptor(),
            monoReferenceSamples: [0.1]
        )
        #expect(!data.isEmpty)
    }

    @Test("Speaker embedding extraction returns PCM-encoded float data")
    func speakerEmbeddingExtraction() async throws {
        let directory = try makeTemporaryModelDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        try validSafeTensorsFixture().write(to: directory.appendingPathComponent("model.safetensors"))

        let backend = MLXInferenceBackend(
            modelDirectoryResolver: { _ in directory },
            warningHandler: { _ in },
            qwenModelLoader: { descriptor in
                MLXModelHandle(descriptor: descriptor)
            },
            speakerEmbeddingExtractor: { resolvedDirectory, monoSamples in
                #expect(resolvedDirectory == directory)
                #expect(monoSamples == [Float(0.1), -0.2, 0.3])
                return [Float(1.25), -2.5]
            }
        )

        let data = try await backend.extractSpeakerEmbedding(
            descriptor: securityTestDescriptor(),
            monoReferenceSamples: [0.1, -0.2, 0.3]
        )

        #expect(decodePCMFloat32LE(data) == [Float(1.25), -2.5])
    }
}

@Suite("MLX TTS Synthesis")
struct MLXTTSWorkflowTests {
    @Test("synthesize and synthesizeStream pass language through to the speech model")
    func passesLanguageToSpeechGenerationModel() async throws {
        let descriptor = ModelDescriptor(
            id: "mistralai/Voxtral-4B-TTS-2603",
            familyID: .voxtralTTS,
            displayName: "Voxtral 4B TTS 2603",
            domain: .tts,
            capabilities: [.speechSynthesis, .multilingual, .presetVoices, .streaming]
        )
        let recordingModel = RecordingSpeechModel()
        let handle = MLXModelHandle(descriptor: descriptor, mlxModel: recordingModel)
        let session = ModelRuntimeSession(
            descriptor: descriptor,
            backendKind: .mlx,
            configuration: ModelRuntimeConfiguration(backendKind: .mlx),
            state: .resident
        )
        let request = SpeechSynthesisRequest(
            model: descriptor.id,
            text: "bonjour",
            voice: nil,
            language: "fr"
        )

        let chunk = try await handle.synthesize(request: request, in: session)
        #expect(chunk.sampleRate == 24_000)
        #expect(recordingModel.generatedLanguages == ["fr"])

        let stream = try await handle.synthesizeStream(request: request)
        var streamedChunks: [AudioChunk] = []
        for try await streamedChunk in stream {
            streamedChunks.append(streamedChunk)
        }

        #expect(streamedChunks.count == 1)
        #expect(streamedChunks[0].sampleRate == 24_000)
        #expect(recordingModel.streamedLanguages == ["fr"])
    }

    @Test("voice-design requests route instruct through the voice selector")
    func voiceDesignUsesInstructAsVoiceSelector() {
        let descriptor = ModelDescriptor(
            id: "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16",
            familyID: .qwen3TTS,
            displayName: "Qwen3 VoiceDesign",
            domain: .tts,
            capabilities: [.speechSynthesis, .voiceDesign]
        )
        let request = SpeechSynthesisRequest(
            model: descriptor.id,
            text: "hello",
            voice: VoiceProfile(label: "ignored", sourceModel: descriptor.id),
            instruct: "Low-register cinematic trailer voice."
        )

        #expect(MLXModelHandle.resolvedVoiceSelector(from: request, descriptor: descriptor) == "Low-register cinematic trailer voice.")
    }

    @Test("chatterbox request overrides are applied and restored")
    func chatterboxOverridesAreScopedToRequest() {
        let model = ChatterboxModel()
        model.cfgWeightOverride = 0.2
        model.emotionAdvOverride = 0.3

        let request = SpeechSynthesisRequest(
            model: "mlx-community/Chatterbox-TTS-fp16",
            text: "hello",
            exaggeration: 0.8,
            cfgWeight: 0.6
        )

        let restore = MLXModelHandle.prepareSpeechModel(model, for: request)
        #expect(model.cfgWeightOverride == 0.6)
        #expect(model.emotionAdvOverride == 0.8)

        restore?()
        #expect(model.cfgWeightOverride == 0.2)
        #expect(model.emotionAdvOverride == 0.3)
    }
}

@Suite("Voxtral Stabilization")
struct VoxtralStabilizationTests {
    @Test("Voxtral preset voice routing is deterministic")
    func presetVoiceRoutingIsDeterministic() {
        #expect(VoxtralTTSModel.resolvedPresetVoiceName(nil) == "neutral_female")
        #expect(VoxtralTTSModel.resolvedPresetVoiceName("NEUTRAL_FEMALE") == "neutral_female")
        #expect(VoxtralTTSModel.resolvedPresetVoiceName("Emma") == "neutral_female")
    }

    @Test("Voxtral missing parameter audit reports missing keys")
    func missingParameterAuditReportsMissingKeys() {
        let missing = VoxtralTTSModel.missingParameterKeys(
            expected: Set([
                "backbone.tok_embeddings.weight",
                "backbone.norm.weight",
                "acoustic_transformer.semantic_codebook_output.weight",
            ]),
            provided: Set([
                "backbone.tok_embeddings.weight",
                "acoustic_transformer.semantic_codebook_output.weight",
            ])
        )

        #expect(missing == ["backbone.norm.weight"])
    }

    @Test("Voxtral audio embedding layout detects flattened checkpoints")
    func audioEmbeddingLayoutDetectsFlattenedCheckpoints() {
        let config = VoxtralTTSConfig()
        let layout = VoxtralTTSModel.resolvedAudioEmbeddingLayout(
            from: [
                "audio_codebook_embeddings.embeddings.weight": MLXArray.zeros([123, 32], dtype: .float32)
            ],
            config: config
        )

        switch layout {
        case let .flattened(totalRows):
            #expect(totalRows == 123)
        case .codebookwise:
            Issue.record("Expected flattened layout for 2D checkpoint tensor")
        }
    }

    @Test("Voxtral model accepts flattened audio embedding shape")
    func modelAcceptsFlattenedAudioEmbeddingShape() {
        let config = VoxtralTTSConfig(
            backbone: VoxtralTTSBackboneConfig(dim: 16),
            audioModelArgs: VoxtralTTSAudioModelConfig(
                semanticCodebookSize: 8,
                acousticCodebookSize: 3,
                nAcousticCodebook: 2,
                audioEncodingArgs: VoxtralTTSAudioEncodingConfig(numCodebooks: 3),
                acousticTransformerArgs: VoxtralTTSAcousticTransformerConfig(
                    inputDim: 16,
                    dim: 16,
                    nLayers: 1,
                    headDim: 8,
                    hiddenDim: 32,
                    nHeads: 2,
                    nKvHeads: 1
                )
            ),
            audioTokenizerArgs: VoxtralTTSCodecConfig(
                semanticCodebookSize: 8,
                semanticDim: 4,
                acousticCodebookSize: 3,
                acousticDim: 2,
                dim: 16,
                hiddenDim: 32,
                headDim: 8,
                nHeads: 2,
                nKvHeads: 1
            )
        )
        let model = VoxtralTTSModel(
            config: config,
            codecVariant: .legacy,
            audioEmbeddingLayout: .flattened(totalRows: 31)
        )

        #expect(model.audioTokenEmbedding.embeddings.weight.shape == [31, 16])
        #expect(model.audioTokenEmbedding.embeddings.bias.shape == [3, 16])
    }
}

@Suite("Model Adapters")
struct AdapterTests {
    @Test("Qwen3TTS adapter validates matching family")
    func qwen3TTSValid() throws {
        let adapter = Qwen3TTSAdapter()
        #expect(adapter.familyID == .qwen3TTS)

        let manifest = ModelPackManifest(
            id: "test/qwen-tts",
            familyID: .qwen3TTS,
            displayName: "Test Qwen TTS",
            domain: .tts,
            capabilities: [.speechSynthesis],
            supportedBackends: [BackendRequirement(backendKind: .mlx)],
            artifacts: [ArtifactSpec(id: "weights", role: .weights, relativePath: "model.safetensors")]
        )

        try adapter.validate(manifest: manifest)
        let descriptor = try adapter.makeDescriptor(from: manifest)
        #expect(descriptor.id == manifest.id)
    }

    @Test("Qwen3TTS adapter rejects mismatched family")
    func qwen3TTSRejectsMismatch() {
        let adapter = Qwen3TTSAdapter()
        let wrongManifest = ModelPackManifest(
            id: "test/whisper",
            familyID: .whisper,
            displayName: "Whisper",
            domain: .stt,
            capabilities: [.speechRecognition],
            supportedBackends: [BackendRequirement(backendKind: .mlx)],
            artifacts: [ArtifactSpec(id: "weights", role: .weights, relativePath: "model.safetensors")]
        )

        #expect(throws: AdapterError.self) {
            try adapter.validate(manifest: wrongManifest)
        }
    }

    @Test("Qwen3TTS adapter rejects empty artifacts")
    func qwen3TTSRejectsEmpty() {
        let adapter = Qwen3TTSAdapter()
        let manifest = ModelPackManifest(
            id: "test/qwen-tts",
            familyID: .qwen3TTS,
            displayName: "Qwen TTS",
            domain: .tts,
            capabilities: [.speechSynthesis],
            supportedBackends: [BackendRequirement(backendKind: .mlx)],
            artifacts: []
        )

        #expect(throws: AdapterError.self) {
            try adapter.validate(manifest: manifest)
        }
    }

    @Test("Qwen3ASR adapter validates family")
    func qwen3ASRValid() throws {
        let adapter = Qwen3ASRAdapter()
        #expect(adapter.familyID == .qwen3ASR)

        let manifest = ModelPackManifest(
            id: "test/qwen-asr",
            familyID: .qwen3ASR,
            displayName: "Test ASR",
            domain: .stt,
            capabilities: [.speechRecognition],
            supportedBackends: [BackendRequirement(backendKind: .mlx)],
            artifacts: [ArtifactSpec(id: "weights", role: .weights, relativePath: "model.safetensors")]
        )

        try adapter.validate(manifest: manifest)
    }

    @Test("Qwen3Aligner adapter validates family")
    func qwen3AlignerValid() throws {
        let adapter = Qwen3AlignerAdapter()
        #expect(adapter.familyID == .qwen3ForcedAligner)

        let manifest = ModelPackManifest(
            id: "test/aligner",
            familyID: .qwen3ForcedAligner,
            displayName: "Test Aligner",
            domain: .stt,
            capabilities: [.forcedAlignment],
            supportedBackends: [BackendRequirement(backendKind: .mlx)],
            artifacts: [ArtifactSpec(id: "weights", role: .weights, relativePath: "model.safetensors")]
        )

        try adapter.validate(manifest: manifest)
    }

    @Test("Generic adapter accepts any family")
    func genericAdapter() throws {
        let adapter = GenericTTSAdapter(familyID: ModelFamilyID("soprano"))
        #expect(adapter.familyID.rawValue == "soprano")

        let manifest = ModelPackManifest(
            id: "test/soprano",
            familyID: ModelFamilyID("soprano"),
            displayName: "Soprano",
            domain: .tts,
            capabilities: [.speechSynthesis],
            supportedBackends: [BackendRequirement(backendKind: .mlx)],
            artifacts: [ArtifactSpec(id: "weights", role: .weights, relativePath: "model.safetensors")]
        )

        try adapter.validate(manifest: manifest)
    }
}

@Suite("PCM Encoding")
struct PCMEncodingTests {
    @Test("Encode-decode round-trip with 48000 samples produces identical output")
    func testPCMEncodeDecodeRoundTripLargeBuffer() {
        let count = 48_000
        var samples: [Float] = []
        samples.reserveCapacity(count)
        for i in 0..<count {
            // Use a sine wave to exercise a range of positive, negative, and fractional values
            let angle = Float(i) * Float.pi * 2.0 / 480.0
            samples.append(sin(angle))
        }

        let data = MLXModelHandle.pcmFloat32LEData(from: samples)
        #expect(data.count == count * MemoryLayout<Float>.size)

        let decoded = decodePCMFloat32LE(data)
        #expect(decoded == samples)
    }
}

@Suite("Stream Bridge")
struct StreamBridgeTests {
    @Test("Audio chunk conversion returns typed float samples at the model sample rate")
    func audioChunkConversionReturnsPCMFloatSamples() {
        let chunk = MLXModelHandle.audioChunk(
            from: [Float(0.0), 0.25, -0.5],
            sampleRate: 24_000
        )

        #expect(chunk.sampleRate == 24_000.0)
        #expect(chunk.samples == [Float(0.0), 0.25, -0.5])
    }

    @Test("Sample stream bridge yields audio chunks as they arrive")
    func sampleStreamYieldsChunks() async throws {
        let stream = MLXStreamBridge.stream(
            from: AsyncThrowingStream { continuation in
                continuation.yield([Float(0.1), 0.2])
                continuation.yield([Float(-0.3)])
                continuation.finish()
            },
            sampleRate: 24_000
        )

        var chunks: [AudioChunk] = []
        for try await chunk in stream {
            chunks.append(chunk)
        }

        #expect(chunks.count == 2)
        #expect(chunks.map(\.sampleRate) == [24_000.0, 24_000.0])
        #expect(chunks[0].samples == [Float(0.1), 0.2])
        #expect(chunks[1].samples == [Float(-0.3)])
    }

    @Test("Placeholder stream finishes immediately")
    func placeholderStreamCompletes() async throws {
        let stream = MLXStreamBridge.placeholderStream()
        var chunks: [AudioChunk] = []
        for try await chunk in stream {
            chunks.append(chunk)
        }
        #expect(chunks.isEmpty)
    }
}

@Suite("Weight Loader Path Security")
struct SpeechTokenizerWeightLoaderPathSecurityTests {

    @Test("discoverSafetensors rejects a symlink that escapes the directory")
    func discoverSafetensorsRejectsEscapingSymlink() throws {
        let baseDir = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: baseDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: baseDir) }

        // Create a real .safetensors file OUTSIDE the search directory.
        let targetFile = baseDir.appendingPathComponent("outside.safetensors")
        try Data("payload".utf8).write(to: targetFile)

        // Create the search directory containing only a symlink to the outside file.
        let searchDir = baseDir.appendingPathComponent("weights", isDirectory: true)
        try FileManager.default.createDirectory(at: searchDir, withIntermediateDirectories: true)
        let symlink = searchDir.appendingPathComponent("model.safetensors")
        try FileManager.default.createSymbolicLink(at: symlink, withDestinationURL: targetFile)

        do {
            _ = try SpeechTokenizerWeightLoader.discoverSafetensors(in: searchDir)
            Issue.record("Expected pathTraversalDetected but no error was thrown")
        } catch let error as SpeechTokenizerWeightLoaderError {
            guard case .pathTraversalDetected = error else {
                Issue.record("Expected pathTraversalDetected, got \(error)")
                return
            }
            #expect(error.errorDescription?.contains("Path traversal detected") == true)
        } catch {
            Issue.record("Expected SpeechTokenizerWeightLoaderError, got \(error)")
        }
    }

    @Test("discoverSafetensors accepts a symlink that stays within the directory")
    func discoverSafetensorsAcceptsInternalSymlink() throws {
        let searchDir = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: searchDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: searchDir) }

        // Real file inside the directory.
        let realFile = searchDir.appendingPathComponent("real.safetensors")
        try Data("payload".utf8).write(to: realFile)

        // Symlink that points to another file inside the same directory.
        let symlink = searchDir.appendingPathComponent("alias.safetensors")
        try FileManager.default.createSymbolicLink(at: symlink, withDestinationURL: realFile)

        // Both the real file and the internal symlink should be discoverable (no throw).
        let found = try SpeechTokenizerWeightLoader.discoverSafetensors(in: searchDir)
        #expect(found.count >= 1)
    }

    @Test("discoverSafetensors canonicalizes a directory URL containing .. components")
    func discoverSafetensorsCanonicalizesDotDotDirectory() throws {
        let searchDir = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: searchDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: searchDir) }

        let realFile = searchDir.appendingPathComponent("model.safetensors")
        try Data("payload".utf8).write(to: realFile)

        // Construct a URL that reaches the same directory via a `..` component.
        let subdirName = UUID().uuidString
        let subdir = searchDir.appendingPathComponent(subdirName, isDirectory: true)
        try FileManager.default.createDirectory(at: subdir, withIntermediateDirectories: true)
        let dotDotURL = subdir.appendingPathComponent("..") // resolves back to searchDir

        // Should enumerate searchDir correctly rather than throwing.
        let found = try SpeechTokenizerWeightLoader.discoverSafetensors(in: dotDotURL)
        #expect(found.count == 1)
    }
}

@Suite("Qwen3 Speaker Encoder Config Validation")
struct Qwen3SpeakerEncoderConfigValidationTests {
    @Test("Malformed speaker encoder config arrays throw descriptive errors")
    func malformedConfigArraysThrow() throws {
        struct Scenario {
            let name: String
            let encChannels: [Int]
            let encKernelSizes: [Int]
            let encDilations: [Int]
            let expectedMessageFragment: String
        }

        let scenarios = [
            Scenario(
                name: "empty channels",
                encChannels: [],
                encKernelSizes: [],
                encDilations: [],
                expectedMessageFragment: "enc_channels must contain at least 2 values"
            ),
            Scenario(
                name: "short kernel sizes",
                encChannels: [512, 512, 1536],
                encKernelSizes: [5],
                encDilations: [1, 2],
                expectedMessageFragment: "enc_kernel_sizes must contain at least 2 values"
            ),
            Scenario(
                name: "short dilations",
                encChannels: [512, 512, 1536],
                encKernelSizes: [5, 3],
                encDilations: [1],
                expectedMessageFragment: "enc_dilations must contain at least 2 values"
            ),
        ]

        for scenario in scenarios {
            let directory = try makeTemporaryModelDirectory()
            defer { try? FileManager.default.removeItem(at: directory) }

            let configData = try speakerEncoderConfigData(
                encChannels: scenario.encChannels,
                encKernelSizes: scenario.encKernelSizes,
                encDilations: scenario.encDilations
            )
            try configData.write(to: directory.appendingPathComponent("config.json"))

            do {
                _ = try Qwen3SpeakerEmbeddingExtractor.extract(from: directory, monoSamples: [Float(0.0), 0.1])
                Issue.record("Expected malformed config scenario '\(scenario.name)' to throw")
            } catch {
                #expect(error.localizedDescription.contains("Invalid Qwen3 speaker encoder config"))
                #expect(error.localizedDescription.contains(scenario.expectedMessageFragment))
            }
        }
    }
}

private func decodePCMFloat32LE(_ data: Data) -> [Float] {
    precondition(data.count.isMultiple(of: MemoryLayout<Float>.size))

    var samples: [Float] = []
    samples.reserveCapacity(data.count / MemoryLayout<Float>.size)

    for offset in stride(from: 0, to: data.count, by: MemoryLayout<Float>.size) {
        var bits: UInt32 = 0
        _ = withUnsafeMutableBytes(of: &bits) { rawBuffer in
            data.copyBytes(to: rawBuffer, from: offset ..< (offset + MemoryLayout<Float>.size))
        }
        samples.append(Float(bitPattern: UInt32(littleEndian: bits)))
    }

    return samples
}

private func securityTestDescriptor() -> ModelDescriptor {
    ModelDescriptor(
        id: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
        familyID: .qwen3TTS,
        displayName: "Secure Test Model",
        domain: .tts,
        capabilities: [.speechSynthesis]
    )
}

private func makeTemporaryModelDirectory() throws -> URL {
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    return directory
}

private func validSafeTensorsFixture() -> Data {
    var data = Data()
    var headerLength = UInt64(2).littleEndian
    withUnsafeBytes(of: &headerLength) { data.append(contentsOf: $0) }
    data.append(contentsOf: [0x7B, 0x7D])
    return data
}

private func speakerEncoderConfigData(
    encChannels: [Int],
    encKernelSizes: [Int],
    encDilations: [Int]
) throws -> Data {
    let config: [String: Any] = [
        "speaker_encoder_config": [
            "mel_dim": 128,
            "enc_dim": 192,
            "enc_channels": encChannels,
            "enc_kernel_sizes": encKernelSizes,
            "enc_dilations": encDilations,
            "enc_attention_channels": 128,
            "enc_res2net_scale": 8,
            "enc_se_channels": 128,
            "sample_rate": 24_000,
        ]
    ]

    return try JSONSerialization.data(withJSONObject: config)
}

private final class WarningRecorder: Sendable {
    // warningHandler closures are always invoked synchronously on the MLXInferenceBackend
    // actor before loadModel returns, so reads and writes to `messages` never overlap.
    // The unsafe annotation opts out of the automatic Sendable check while the sequential
    // access pattern is manually guaranteed by the test structure.
    nonisolated(unsafe) private(set) var messages: [String] = []

    func append(_ message: String) {
        messages.append(message)
    }

    func count(matching substring: String) -> Int {
        messages.filter { $0.contains(substring) }.count
    }
}
