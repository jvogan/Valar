//  Run the TTS suites in this file:
//    xcodebuild test \
//      -scheme MLXAudio-Package \
//      -destination 'platform=macOS' \
//      -parallel-testing-enabled NO \
//      -only-testing:MLXAudioTests/SopranoTextCleaningTests \
//      CODE_SIGNING_ALLOWED=NO
//
//  Run a single category:
//    -only-testing:'MLXAudioTests/SopranoTextCleaningTests'
//
//  Run a single test (note the trailing parentheses for Swift Testing):
//    -only-testing:'MLXAudioTests/SopranoTextCleaningTests/testTextCleaning()'
//
//  Filter test results:
//    2>&1 | grep --color=never -E '(Suite.*started|Test test.*started|passed after|failed after|TEST SUCCEEDED|TEST FAILED|Suite.*passed|Test run)'

import Testing
import MLX
import MLXLMCommon
import Foundation
import HuggingFace

@testable import MLXAudioCore
@testable import MLXAudioTTS
@testable import MLXAudioCodecs

private func loadTTSNetworkFixture(sampleRate: Int, maxSamples: Int) throws -> MLXArray {
    let audioURL = Bundle.module.url(
        forResource: "intention",
        withExtension: "wav",
        subdirectory: "media"
    )!
    let (_, audio) = try loadAudioArray(from: audioURL, sampleRate: sampleRate)
    let sampleCount = min(audio.shape[0], maxSamples)
    return audio[0..<sampleCount]
}


// MARK: - Text Cleaning Unit Tests

struct SopranoTextCleaningTests {

    @Test func testTextCleaning() {
        // Test number normalization
        let text1 = "I have $100 and 50 cents."
        let cleaned1 = cleanTextForSoprano(text1)
        #expect(cleaned1.contains("one hundred dollars"), "Should expand dollar amounts")

        // Test abbreviations
        let text2 = "Dr. Smith went to the API conference."
        let cleaned2 = cleanTextForSoprano(text2)
        #expect(cleaned2.contains("doctor"), "Should expand Dr. to doctor")
        #expect(cleaned2.contains("a p i"), "Should expand API")

        // Test ordinals
        let text3 = "This is the 1st and 2nd test."
        let cleaned3 = cleanTextForSoprano(text3)
        #expect(cleaned3.contains("first"), "Should expand 1st to first")
        #expect(cleaned3.contains("second"), "Should expand 2nd to second")

        print("\u{001B}[32mText cleaning tests passed!\u{001B}[0m")
    }
}

struct TTSLoaderResolutionTests {

    @Test func testModelDirectoryKindClassifiesManagedValarPack() {
        let path = URL(fileURLWithPath: "/tmp/mock-home/Library/Application Support/ValarTTS/ModelPacks/qwen3-tts/model")
        #expect(ModelUtils.modelDirectoryKind(path) == .valarManagedPack)
        #expect(ModelUtils.shouldAutoDeleteCorruptedModelDirectory(path) == false)
    }

    @Test func testModelDirectoryKindClassifiesHuggingFaceCache() {
        let path = URL(fileURLWithPath: "/tmp/mock-home/.cache/huggingface/hub/models--mlx-community--Qwen3/snapshots/abc123")
        #expect(ModelUtils.modelDirectoryKind(path) == .huggingFaceCache)
        #expect(ModelUtils.shouldAutoDeleteCorruptedModelDirectory(path) == true)
    }

    @Test func testModelDirectoryKindClassifiesLegacyMLXAudioCache() {
        let path = URL(fileURLWithPath: "/tmp/mock-home/.cache/huggingface/hub/mlx-audio/mlx-community_Qwen3-TTS-12Hz-1.7B-Base-bf16")
        #expect(ModelUtils.modelDirectoryKind(path) == .legacyMLXAudioCache)
        #expect(ModelUtils.shouldAutoDeleteCorruptedModelDirectory(path) == true)
    }

    @Test func testModelDirectoryKindPreservesUnknownDirectories() {
        let path = URL(fileURLWithPath: "/tmp/valar-qwen-fixture")
        #expect(ModelUtils.modelDirectoryKind(path) == .other)
        #expect(ModelUtils.shouldAutoDeleteCorruptedModelDirectory(path) == false)
    }

    @Test func testResolveModelTypePrefersConfigOverParams() throws {
        let modelDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: modelDir) }

        try Data(#"{"model_type":"echo_tts"}"#.utf8)
            .write(to: modelDir.appendingPathComponent("config.json"))
        try Data(#"{"model_type":"voxtral_tts"}"#.utf8)
            .write(to: modelDir.appendingPathComponent("params.json"))

        let resolved = try ModelUtils.resolveModelType(modelDirectory: modelDir)
        #expect(resolved == "echo_tts")
    }

    @Test func testResolveModelTypeFallsBackToParamsAndTTSFromDirectoryUsesIt() async throws {
        let modelDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: modelDir) }

        try Data(#"{"model_type":"voxtral_tts"}"#.utf8)
            .write(to: modelDir.appendingPathComponent("params.json"))
        try Data([0x01]).write(to: modelDir.appendingPathComponent("model.safetensors"))

        let resolved = try ModelUtils.resolveModelType(modelDirectory: modelDir)
        #expect(resolved == "voxtral_tts")

        do {
            _ = try await TTS.fromDirectory(modelDir)
            Issue.record("Expected voxtral_tts local loader path to hit the explicit stub")
        } catch is TTSModelError {
            Issue.record("Expected voxtral_tts local loader dispatch to continue past model-type resolution")
        } catch {
        }
    }

    @Test func testResolveModelTypeSopranoAndTTSFromDirectoryUsesIt() async throws {
        let modelDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: modelDir) }

        try Data(#"{"model_type":"soprano"}"#.utf8)
            .write(to: modelDir.appendingPathComponent("config.json"))
        try Data([0x01]).write(to: modelDir.appendingPathComponent("model.safetensors"))

        let resolved = try ModelUtils.resolveModelType(modelDirectory: modelDir)
        #expect(resolved == "soprano")

        do {
            _ = try await TTS.fromDirectory(modelDir)
            Issue.record("Expected soprano local loader path to continue into model loading")
        } catch is TTSModelError {
            Issue.record("Expected soprano local loader dispatch to continue past model-type resolution")
        } catch {
        }
    }

    @Test func testResolveModelTypeFindsNestedTadaConfig() throws {
        let modelDir = try makeTadaFixture()
        defer { try? FileManager.default.removeItem(at: modelDir) }

        let resolved = try ModelUtils.resolveModelType(modelDirectory: modelDir)
        #expect(resolved == "tada")
    }

    @Test func testResolveModelTypeRecognizesRealTadaConfigWithLlamaModelType() throws {
        let modelDir = try makeTadaFixture()
        defer { try? FileManager.default.removeItem(at: modelDir) }

        let json = """
        {
          "model_type": "llama",
          "architectures": ["TadaForCausalLM"],
          "diffusion_head_type": "vibevoice",
          "acoustic_dim": 512
        }
        """
        try Data(json.utf8)
            .write(to: modelDir.appendingPathComponent("model/config.json"))

        let resolved = try ModelUtils.resolveModelType(modelDirectory: modelDir)
        #expect(resolved == "tada")
    }

    @Test func testInlineTadaConfigDecodesPublicHumeCheckpointShape() throws {
        let json = """
        {
          "model_type": "llama",
          "architectures": ["TadaForCausalLM"],
          "hidden_size": 2048,
          "num_hidden_layers": 16,
          "num_attention_heads": 32,
          "num_key_value_heads": 8,
          "head_dim": 64,
          "acoustic_dim": 512,
          "num_time_classes": 256,
          "shift_acoustic": 5,
          "head_layers": 6,
          "head_ffn_ratio": 4.0,
          "bos_token_id": 128000,
          "eos_token_id": 128001
        }
        """

        let config = try JSONDecoder().decode(TADAConfig.self, from: Data(json.utf8))
        #expect(config.hiddenSize == 2048)
        #expect(config.eosTokenId == [128001])
        #expect(config.eotId == 128001)
    }

    @Test func testTadaFromDirectoryRequiresSelfContainedOfflinePack() async throws {
        let modelDir = try makeTadaFixture(includeTokenizer: false)
        defer { try? FileManager.default.removeItem(at: modelDir) }

        do {
            _ = try await TTS.fromDirectory(modelDir)
            Issue.record("Expected incomplete TADA pack to fail validation")
        } catch let error as ModelUtilsError {
            #expect(error.errorDescription?.contains("Incomplete TADA pack") == true)
            #expect(error.errorDescription?.contains("tokenizer.json") == true)
        }
    }

    @Test func testCachedTadaRepoDispatchesToPackContract() async throws {
        let cacheRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("tada-hf-cache-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: cacheRoot, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: cacheRoot) }

        let cache = HubCache(cacheDirectory: cacheRoot)
        let modelDir = cache.cacheDirectory
            .appendingPathComponent("mlx-audio")
            .appendingPathComponent("HumeAI_mlx-tada-3b", isDirectory: true)
        try writeTadaFixture(to: modelDir, includeTokenizer: true)

        do {
            _ = try await TTS.loadModel(modelRepo: "HumeAI/mlx-tada-3b", cache: cache)
            Issue.record("Expected cached TADA repo load to stop at the runtime contract")
        } catch let error as AudioGenerationError {
            #expect(error.errorDescription?.contains("runtime loading is not implemented") == true)
        }
    }

    @Test func testCachedTadaRepoFailsEarlyWithoutMetaTokenizerAccess() async throws {
        let cacheRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("tada-hf-cache-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: cacheRoot, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: cacheRoot) }

        let cache = HubCache(cacheDirectory: cacheRoot)
        let modelDir = cache.cacheDirectory
            .appendingPathComponent("mlx-audio")
            .appendingPathComponent("HumeAI_mlx-tada-1b", isDirectory: true)
        try writeTadaFixture(to: modelDir, includeTokenizer: false)

        do {
            _ = try await TTS.loadModel(modelRepo: "HumeAI/mlx-tada-1b", cache: cache)
            Issue.record("Expected TADA install to fail early without Meta tokenizer access")
        } catch let error as ModelUtilsError {
            #expect(error.errorDescription?.contains("Meta Llama 3.2 license") == true)
            #expect(error.errorDescription?.contains("meta-llama/Llama-3.2-1B") == true)
        }
    }
}

private func makeTadaFixture(includeTokenizer: Bool = true) throws -> URL {
    let modelDir = FileManager.default.temporaryDirectory
        .appendingPathComponent("tada-fixture-\(UUID().uuidString)", isDirectory: true)
    try writeTadaFixture(to: modelDir, includeTokenizer: includeTokenizer)
    return modelDir
}

private func writeTadaFixture(to modelDir: URL, includeTokenizer: Bool) throws {
    try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)
    let modelSubdir = modelDir.appendingPathComponent("model", isDirectory: true)
    let encoderSubdir = modelDir.appendingPathComponent("encoder", isDirectory: true)
    let decoderSubdir = modelDir.appendingPathComponent("decoder", isDirectory: true)
    let alignerSubdir = modelDir.appendingPathComponent("aligner", isDirectory: true)
    try FileManager.default.createDirectory(at: modelSubdir, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: encoderSubdir, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: decoderSubdir, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: alignerSubdir, withIntermediateDirectories: true)

    try Data(#"{"model_type":"tada"}"#.utf8)
        .write(to: modelSubdir.appendingPathComponent("config.json"))
    try Data([0x01]).write(to: modelSubdir.appendingPathComponent("weights.safetensors"))
    try Data([0x01]).write(to: encoderSubdir.appendingPathComponent("weights.safetensors"))
    try Data([0x01]).write(to: decoderSubdir.appendingPathComponent("weights.safetensors"))
    try Data([0x01]).write(to: alignerSubdir.appendingPathComponent("weights.safetensors"))

    if includeTokenizer {
        try Data(#"{"version":"1.0"}"#.utf8)
            .write(to: modelDir.appendingPathComponent("tokenizer.json"))
    }
}

struct VoxtralTTSTokenizerTests {

    @Test func testTekkenEncodeUsesMergeRanks() throws {
        let modelDir = try makeVoxtralTekkenFixture()
        defer { try? FileManager.default.removeItem(at: modelDir) }

        let tokenizer = try VoxtralTTSTokenizer.fromModelDirectory(modelDir)
        #expect(tokenizer.encode(text: "abc") == [37])
        #expect(tokenizer.decode(tokenIds: [37]) == "abc")
    }

    @Test func testPackSpeechRequestMatchesPromptLayout() throws {
        let modelDir = try makeVoxtralTekkenFixture()
        defer { try? FileManager.default.removeItem(at: modelDir) }

        let tokenizer = try VoxtralTTSTokenizer.fromModelDirectory(modelDir)
        let packed = tokenizer.packSpeechRequest(text: "abc", voiceFrameCount: 3)

        #expect(packed == [1, 25, 24, 24, 24, 26, 37, 27, 25])
    }
}

struct VoxtralTTSWeightMappingTests {

    @Test func testCommunityCodecWeightsMapToAlternateDecoderLayout() {
        let rawWeights: [String: MLXArray] = [
            "audio_tokenizer.decoder_blocks.0.conv.parametrizations.weight.original0": MLXArray.ones([1024, 1, 1], dtype: .float32),
            "audio_tokenizer.decoder_blocks.0.conv.parametrizations.weight.original1": MLXArray.ones([1024, 292, 3], dtype: .float32),
            "audio_tokenizer.decoder_blocks.1.layers.0.attention.q_norm.weight": MLXArray.ones([1024], dtype: .float32),
            "audio_tokenizer.decoder_blocks.1.layers.0.attention.wq.weight": MLXArray.ones([1024, 1024], dtype: .float32),
            "audio_tokenizer.quantizer.semantic_codebook.embedding_sum": MLXArray.ones([8192, 256], dtype: .float32),
            "audio_tokenizer.quantizer.semantic_codebook.cluster_usage": MLXArray.ones([8192], dtype: .float32),
        ]

        #expect(VoxtralTTSModel.inferredCodecVariant(from: rawWeights) == .community)

        let sanitized = VoxtralTTSModel.sanitize(weights: rawWeights, codecVariant: .community)

        #expect(sanitized["audio_tokenizer.communityDecoder.inputConv.conv.weight"]?.shape == [1024, 3, 292])
        #expect(sanitized["audio_tokenizer.communityDecoder.stage0.layers.0.attention.q_norm.weight"]?.shape == [1024])
        #expect(sanitized["audio_tokenizer.communityDecoder.stage0.layers.0.attention.wq.weight"]?.shape == [1024, 1024])
        #expect(sanitized["audio_tokenizer.communityQuantizer.semanticCodebook.weight"]?.shape == [8192, 256])
    }
}

struct EchoTTSTests {

    @Test func testTextNormalization() {
        let normalized = echoTtsNormalizeTextPrompt("Hello: world\nnew line")
        #expect(normalized.hasPrefix("[S1] "))
        #expect(normalized.contains(","))
        #expect(!normalized.contains("\n"))
    }

    @Test func testTokenizerEncode() {
        let tokens = echoTtsTokenizerEncode("hello", appendBOS: true, normalize: false)
        #expect(tokens.shape == [6])
        #expect(tokens[0].item(Int32.self) == 0)
    }

    @Test func testTextInputIDsAndMask() {
        let result = echoTtsTextInputIDsAndMask(
            ["hello", "world"],
            maxLength: 10,
            normalize: true,
            padToMax: true
        )
        #expect(result.inputIDs.shape == [2, 10])
        #expect(result.mask.shape == [2, 10])
        #expect(result.normalizedTexts.count == 2)
    }

    @Test func testEchoDiTForwardShapes() {
        let config = EchoDiTConfig(
            latentSize: 8,
            modelSize: 32,
            numLayers: 2,
            numHeads: 4,
            intermediateSize: 64,
            normEps: 1e-5,
            textVocabSize: 256,
            textModelSize: 32,
            textNumLayers: 1,
            textNumHeads: 4,
            textIntermediateSize: 64,
            speakerPatchSize: 2,
            speakerModelSize: 32,
            speakerNumLayers: 1,
            speakerNumHeads: 4,
            speakerIntermediateSize: 64,
            timestepEmbedSize: 16,
            adalnRank: 8
        )
        let model = EchoDiT(
            latentSize: config.latentSize,
            modelSize: config.modelSize,
            numLayers: config.numLayers,
            numHeads: config.numHeads,
            intermediateSize: config.intermediateSize,
            normEps: config.normEps,
            textVocabSize: config.textVocabSize,
            textModelSize: config.textModelSize,
            textNumLayers: config.textNumLayers,
            textNumHeads: config.textNumHeads,
            textIntermediateSize: config.textIntermediateSize,
            speakerPatchSize: config.speakerPatchSize,
            speakerModelSize: config.speakerModelSize,
            speakerNumLayers: config.speakerNumLayers,
            speakerNumHeads: config.speakerNumHeads,
            speakerIntermediateSize: config.speakerIntermediateSize,
            timestepEmbedSize: config.timestepEmbedSize,
            adalnRank: config.adalnRank
        )

        let x = MLXRandom.normal([1, 6, config.latentSize])
        let t = MLXArray([Float(0.7)])
        let textInputIDs = MLXArray([Int32(0), 1, 2, 3, 4]).reshaped([1, 5])
        let textMask = MLXArray([true, true, true, true, true]).reshaped([1, 5])
        let speakerLatent = MLXRandom.normal([1, 8, config.latentSize])
        let speakerMask = MLXArray.ones([1, 8], dtype: .bool)

        let kvText = model.getKVCacheText(textInputIDs, textMask: textMask)
        let kvSpeaker = model.getKVCacheSpeaker(speakerLatent)
        let output = model(
            x: x,
            t: t,
            textMask: textMask,
            speakerMask: speakerMask,
            kvCacheText: kvText,
            kvCacheSpeaker: kvSpeaker
        )

        #expect(output.shape == [1, 6, config.latentSize])
    }

    @Test func testSanitizeAndGenerateSmoke() throws {
        final class FakeFishAE: EchoTTSAudioCodec {
            func encodeZQ(_ audioData: MLXArray) -> MLXArray {
                MLXArray.zeros([audioData.shape[0], 8, max(audioData.shape[2] / 2_048, 1)], dtype: .float32)
            }

            func decodeZQ(_ zQ: MLXArray) -> MLXArray {
                MLXArray.zeros([zQ.shape[0], 1, zQ.shape[2] * 2_048], dtype: .float32)
            }
        }

        let config = EchoTTSConfig(
            dit: EchoDiTConfig(
                latentSize: 8,
                modelSize: 32,
                numLayers: 2,
                numHeads: 4,
                intermediateSize: 64,
                normEps: 1e-5,
                textVocabSize: 256,
                textModelSize: 32,
                textNumLayers: 1,
                textNumHeads: 4,
                textIntermediateSize: 64,
                speakerPatchSize: 2,
                speakerModelSize: 32,
                speakerNumLayers: 1,
                speakerNumHeads: 4,
                speakerIntermediateSize: 64,
                timestepEmbedSize: 16,
                adalnRank: 8
            ),
            sampler: EchoTTSSamplerConfig(
                numSteps: 1,
                cfgScaleText: 1,
                cfgScaleSpeaker: 1,
                sequenceLength: 4
            )
        )
        let model = EchoTTSModel(
            config: config,
            fishAE: FakeFishAE(),
            pcaState: EchoTTSPCAState(
                pcaComponents: MLXArray.eye(8, dtype: .float32),
                pcaMean: MLXArray.zeros([8], dtype: .float32),
                latentScale: 1
            )
        )

        let sanitized = model.sanitize(weights: [
            "cond_module.0.weight": MLXArray.zeros([1, 1], dtype: .float32),
            "pca_components": MLXArray.zeros([1], dtype: .float32),
        ])
        #expect(sanitized["model.condModule.layers.0.weight"] != nil)
        #expect(sanitized["model.pca_components"] == nil)

        let result = try model.generateDetailed(
            text: "hi",
            refAudio: nil,
            rngSeed: 0,
            numSteps: 1,
            sequenceLength: 4
        )
        #expect(model.sampleRate == 44_100)
        #expect(result.audio.shape[0] > 0)
    }

    @Test func testDeleteBlockwiseModules() throws {
        let config = EchoTTSConfig(
            deleteBlockwiseModules: true,
            dit: EchoDiTConfig(
                latentSize: 8,
                modelSize: 32,
                numLayers: 2,
                numHeads: 4,
                intermediateSize: 64,
                normEps: 1e-5,
                textVocabSize: 256,
                textModelSize: 32,
                textNumLayers: 1,
                textNumHeads: 4,
                textIntermediateSize: 64,
                speakerPatchSize: 2,
                speakerModelSize: 32,
                speakerNumLayers: 1,
                speakerNumHeads: 4,
                speakerIntermediateSize: 64,
                timestepEmbedSize: 16,
                adalnRank: 8
            ),
            sampler: EchoTTSSamplerConfig(numSteps: 1, sequenceLength: 4)
        )
        let model = EchoTTSModel(config: config)

        let sanitized = model.sanitize(weights: [
            "latent_encoder.in_proj.weight": MLXArray.zeros([1, 1], dtype: .float32),
            "blocks.0.attention.wk_latent.weight": MLXArray.zeros([1, 1], dtype: .float32),
            "blocks.0.attention.wv_latent.weight": MLXArray.zeros([1, 1], dtype: .float32),
            "out_proj.weight": MLXArray.zeros([8, 32], dtype: .float32),
        ])
        #expect(sanitized["model.outProj.weight"] != nil)
        #expect(!sanitized.keys.contains(where: { $0.contains("latent_encoder") }))
        #expect(!sanitized.keys.contains(where: { $0.contains("wk_latent") }))
        #expect(!sanitized.keys.contains(where: { $0.contains("wv_latent") }))

        #expect(throws: AudioGenerationError.self) {
            try model.model.getKVCacheLatent(MLXArray.zeros([1, 0, 8], dtype: .float32))
        }

        #expect(throws: AudioGenerationError.self) {
            try model.generateLatents(text: "hi", blockSizes: [2], numSteps: 1, sequenceLength: 4)
        }
    }
}


struct TADATTSConfigTests {

    @Test func testDecodeSpecConfigAndBridgeToLlamaConfig() throws {
        let json = """
        {
          "vocab_size": 128256,
          "hidden_size": 3072,
          "intermediate_size": 8192,
          "num_hidden_layers": 28,
          "num_attention_heads": 24,
          "num_key_value_heads": 8,
          "head_dim": 128,
          "rms_norm_eps": 1e-5,
          "rope_theta": 500000.0,
          "rope_scaling": {
            "factor": 32.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3"
          },
          "tie_word_embeddings": true,
          "max_position_embeddings": 131072,
          "acoustic_dim": 512,
          "num_time_classes": 256,
          "shift_acoustic": 5,
          "head_layers": 6,
          "head_ffn_ratio": 4.0,
          "bottleneck_dim": null,
          "acoustic_mean": 0.0,
          "acoustic_std": 1.5,
          "bos_token_id": 128000,
          "eos_token_id": [128001, 128008, 128009],
          "eot_id": 128009
        }
        """

        let config = try JSONDecoder().decode(TADATTSConfig.self, from: Data(json.utf8))
        let llama = config.llamaConfiguration

        #expect(config.hiddenSize == 3072)
        #expect(config.acousticDim == 512)
        #expect(config.eosTokenIDs == [128001, 128008, 128009])
        #expect(llama.hiddenSize == 3072)
        #expect(llama.kvHeads == 8)
        #expect(llama.headDimensions == 128)
        #expect(llama.ropeTheta == 500000.0)
    }
}

struct TADATTSBackboneTests {

    @Test func testBackboneRunsThroughSharedLlamaInner() {
        let config = TADATTSConfig(
            vocabSize: 64,
            hiddenSize: 64,
            intermediateSize: 128,
            numHiddenLayers: 2,
            numAttentionHeads: 4,
            numKeyValueHeads: 2,
            headDim: 16,
            maxPositionEmbeddings: 512,
            acousticDim: 8,
            numTimeClasses: 16
        )
        let backbone = TADATTSBackbone(config: config)

        let inputIDs = MLXArray([Int32(1), 2, 3]).reshaped([1, 3])
        let acoustic = MLXArray.zeros([1, 3, 8], dtype: .float32)
        let acousticMask = MLXArray([Int32(1), 1, 1]).reshaped([1, 3])
        let timeBefore = MLXArray([Int32(0), 1, 2]).reshaped([1, 3])
        let timeAfter = MLXArray([Int32(2), 1, 0]).reshaped([1, 3])

        let hiddenStates = backbone(
            inputIDs: inputIDs,
            acousticFeatures: acoustic,
            acousticMask: acousticMask,
            timeBefore: timeBefore,
            timeAfter: timeAfter
        )

        #expect(hiddenStates.shape == [1, 3, 64])
    }
}

struct TADATTSAlignerTests {

    @Test func testSanitizeMapsAlignerKeysAndFusesPositionalConvWeights() {
        let weights: [String: MLXArray] = [
            "aligner.feature_extractor.conv_layers.0.conv.weight": MLXArray.zeros([2, 3, 4], dtype: .float32),
            "aligner.encoder.pos_conv_embed.conv.weight_g": MLXArray.ones([1, 1, 4], dtype: .float32),
            "aligner.encoder.pos_conv_embed.conv.weight_v": MLXArray.ones([2, 1, 4], dtype: .float32),
            "aligner.lm_head.bias": MLXArray.zeros([4], dtype: .float32),
        ]

        let sanitized = TADATTSAligner.sanitize(weights: weights)

        #expect(sanitized["feature_extractor.conv_layers.0.conv.weight"]?.shape == [2, 4, 3])
        #expect(sanitized["encoder.pos_conv_embed.conv.weight"] != nil)
        #expect(sanitized["lm_head.bias"] != nil)
        #expect(sanitized["aligner.lm_head.bias"] == nil)
    }

    @Test func testViterbiAlignmentReturnsOneIndexedTokenPositions() {
        let logProbs = MLXArray([
            Float(0.0), Float(-2.0), Float(-4.0),
            Float(-3.0), Float(0.0), Float(-4.0),
            Float(-3.0), Float(-4.0), Float(0.0),
        ]).reshaped([3, 3])

        let aligned = TADATTSAligner.viterbiAlignment(
            logProbs: logProbs,
            tokenIDs: [1, 2],
            blankTokenID: 0
        )

        #expect(aligned.positions == [2, 3])
        #expect(aligned.mask == [0, 1, 1])
    }
}

struct TADATTSModelSanitizeTests {

    @Test func testSanitizeDropsUnsupportedAcousticProjectionBias() {
        let weights: [String: MLXArray] = [
            "acoustic_proj.weight": MLXArray.zeros([2048, 512], dtype: .float32),
            "acoustic_proj.bias": MLXArray.zeros([2048], dtype: .float32),
            "prediction_head.t_embedder.mlp.0.weight": MLXArray.zeros([16, 16], dtype: .float32),
        ]

        let sanitized = TADATTSModel.sanitize(weights: weights)

        #expect(sanitized["acoustic_proj.weight"] != nil)
        #expect(sanitized["acoustic_proj.bias"] == nil)
        #expect(sanitized["prediction_head.t_embedder.mlp.0.weight"] != nil)
    }
}

private func makeVoxtralTekkenFixture() throws -> URL {
    let modelDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)

    let tekken = """
    {
      "config": {
        "pattern": "\\\\s+|\\\\S+",
        "default_vocab_size": 38,
        "default_num_special_tokens": 32,
        "version": "v7"
      },
      "special_tokens": [
        { "rank": 1, "token_str": "~~", "is_control": true },
        { "rank": 24, "token_str": "[AUDIO]", "is_control": true },
        { "rank": 25, "token_str": "[BEGIN_AUDIO]", "is_control": true },
        { "rank": 26, "token_str": "[NEXT_AUDIO_TEXT]", "is_control": true },
        { "rank": 27, "token_str": "[REPEAT_AUDIO_TEXT]", "is_control": true }
      ],
      "vocab": [
        { "rank": 0, "token_bytes": "YQ==", "token_str": "a" },
        { "rank": 1, "token_bytes": "Yg==", "token_str": "b" },
        { "rank": 2, "token_bytes": "Yw==", "token_str": "c" },
        { "rank": 3, "token_bytes": "YWI=", "token_str": "ab" },
        { "rank": 4, "token_bytes": "YmM=", "token_str": "bc" },
        { "rank": 5, "token_bytes": "YWJj", "token_str": "abc" }
      ]
    }
    """

    try Data(tekken.utf8).write(to: modelDir.appendingPathComponent("tekken.json"))
    return modelDir
}

@Suite("Echo TTS Network Tests", .serialized)
struct EchoTTSNetworkTests {

    @Test func echoTTSBaseLoadsConfiguredCodecAndGeneratesAudio() async throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXAUDIO_ENABLE_NETWORK_TESTS"] == "1" else {
            print("Skipping network Echo TTS test. Set MLXAUDIO_ENABLE_NETWORK_TESTS=1 to enable.")
            return
        }

        let repo = env["MLXAUDIO_ECHO_TTS_REPO"] ?? "mlx-community/echo-tts-base"
        let model = try await EchoTTSModel.fromPretrained(repo)
        let refAudio = try loadTTSNetworkFixture(sampleRate: model.sampleRate, maxSamples: model.sampleRate / 4)

        if repo == "mlx-community/echo-tts-base" {
            #expect(model.config.fishCodecRepo == "jordand/fish-s1-dac-min")
        }

        let result = try model.generateDetailed(
            text: "hello",
            refAudio: refAudio,
            rngSeed: 0,
            numSteps: 1,
            sequenceLength: 8
        )

        #expect(result.audio.shape[0] > 0)
        #expect(result.info.generationTokenCount == 8)
        #expect(model.fishAE != nil)
        #expect(model.pcaState != nil)
    }
}
