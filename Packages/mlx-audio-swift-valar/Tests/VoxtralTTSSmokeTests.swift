import Foundation
import HuggingFace
@testable import MLXAudioCore
@testable import MLXAudioTTS
import Testing

@Suite("Voxtral TTS Smoke Tests", .serialized)
struct VoxtralTTSSmokeTests {

    @Test func resolvesVoxtralTypeFromConfigAndFailsCleanlyWithoutParams() async throws {
        let modelDir = try makeVoxtralFixture(includeConfig: true, includeParams: false)
        defer { try? FileManager.default.removeItem(at: modelDir) }

        let resolved = try ModelUtils.resolveModelType(modelDirectory: modelDir)
        #expect(resolved == "voxtral_tts")

        do {
            _ = try await TTS.fromDirectory(modelDir)
            Issue.record("Expected config-backed Voxtral load to fail without params.json")
        } catch let error as AudioGenerationError {
            #expect(error.errorDescription?.contains("Missing Voxtral params.json") == true)
        }
    }

    @Test func resolvesVoxtralTypeFromParamsAndDispatchesLocalLoader() async throws {
        let modelDir = try makeVoxtralFixture(includeTekken: true)
        defer { try? FileManager.default.removeItem(at: modelDir) }

        let resolved = try ModelUtils.resolveModelType(modelDirectory: modelDir)
        #expect(resolved == "voxtral_tts")

        do {
            _ = try await TTS.fromDirectory(modelDir)
            Issue.record("Expected Voxtral local load to fail cleanly without normalized voice assets")
        } catch let error as AudioGenerationError {
            #expect(error.errorDescription?.contains("voice_embedding_safe") == true)
        }
    }

    @Test func hfRepoStringDispatchesToVoxtralLoaderUsingCacheFixture() async throws {
        let cacheRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("voxtral-hf-cache-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: cacheRoot, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: cacheRoot) }

        let cache = HubCache(cacheDirectory: cacheRoot)
        let modelDir = cache.cacheDirectory
            .appendingPathComponent("mlx-audio")
            .appendingPathComponent("mistralai_Voxtral-4B-TTS-2603", isDirectory: true)
        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)

        try writeVoxtralParams(to: modelDir)
        try writeTekkenFixture(to: modelDir)
        try Data([0x01]).write(to: modelDir.appendingPathComponent("model.safetensors"))

        do {
            _ = try await TTS.loadModel(modelRepo: "mistralai/Voxtral-4B-TTS-2603", cache: cache)
            Issue.record("Expected cached Voxtral repo load to fail cleanly without normalized voice assets")
        } catch let error as AudioGenerationError {
            #expect(error.errorDescription?.contains("voice_embedding_safe") == true)
        }
    }

    @Test func missingTekkenGetsCleanError() async throws {
        let modelDir = try makeVoxtralFixture()
        defer { try? FileManager.default.removeItem(at: modelDir) }

        do {
            _ = try await TTS.fromDirectory(modelDir)
            Issue.record("Expected missing tekken.json to fail cleanly")
        } catch let error as VoxtralTTSTokenizerError {
            #expect(error.errorDescription?.contains("tekken.json not found") == true)
        }
    }

    @Test func malformedParamsGetsCleanError() async throws {
        let modelDir = try makeVoxtralFixture(includeConfig: true, malformedParams: true)
        defer { try? FileManager.default.removeItem(at: modelDir) }

        do {
            _ = try await TTS.fromDirectory(modelDir)
            Issue.record("Expected malformed params.json to fail cleanly")
        } catch let error as AudioGenerationError {
            #expect(error.errorDescription?.contains("Invalid Voxtral params.json") == true)
        }
    }

    @Test func synthSignatureFailsCleanlyUntilTokenizerIsLoaded() async throws {
        let model = VoxtralTTSModel(config: VoxtralTTSConfig())

        do {
            _ = try await model.generate(
                text: "hello",
                voice: nil,
                refAudio: nil,
                refText: nil,
                language: nil,
                generationParameters: model.defaultGenerationParameters
            )
            Issue.record("Expected unloaded Voxtral model generate() to fail cleanly")
        } catch let error as AudioGenerationError {
            #expect(error.errorDescription?.contains("Tokenizer not loaded") == true)
        }
    }

    @Test func streamSignatureFailsCleanlyUntilTokenizerIsLoaded() async throws {
        let model = VoxtralTTSModel(config: VoxtralTTSConfig())
        let stream = model.generateStream(
            text: "hello",
            voice: nil,
            refAudio: nil,
            refText: nil,
            language: nil,
            generationParameters: model.defaultGenerationParameters
        )

        do {
            for try await _ in stream {}
            Issue.record("Expected unloaded Voxtral model generateStream() to fail cleanly")
        } catch let error as AudioGenerationError {
            #expect(error.errorDescription?.contains("Tokenizer not loaded") == true)
        }
    }
}

private func makeVoxtralFixture(
    includeConfig: Bool = false,
    includeParams: Bool = true,
    malformedParams: Bool = false,
    includeTekken: Bool = false
) throws -> URL {
    let modelDir = FileManager.default.temporaryDirectory
        .appendingPathComponent("voxtral-fixture-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)

    if includeConfig {
        try Data(#"{"model_type":"voxtral_tts"}"#.utf8)
            .write(to: modelDir.appendingPathComponent("config.json"))
    }

    if includeParams {
        if malformedParams {
            try Data(#"{"model_type":"voxtral_tts""#.utf8)
                .write(to: modelDir.appendingPathComponent("params.json"))
        } else {
            try writeVoxtralParams(to: modelDir)
        }
    }

    if includeTekken {
        try writeTekkenFixture(to: modelDir)
    }

    return modelDir
}

private func writeVoxtralParams(to modelDir: URL) throws {
    let params = """
    {
      "model_type": "voxtral_tts",
      "dim": 3072,
      "n_layers": 26,
      "head_dim": 128,
      "hidden_dim": 9216,
      "n_heads": 32,
      "n_kv_heads": 8,
      "vocab_size": 131072,
      "rope_theta": 1000000.0,
      "norm_eps": 1e-05,
      "tied_embeddings": true,
      "causal": true,
      "max_seq_len": 65536,
      "audio_model_args": {
        "semantic_codebook_size": 8192,
        "acoustic_codebook_size": 21,
        "n_acoustic_codebook": 36,
        "audio_encoding_args": {
          "codebook_pattern": "parallel",
          "num_codebooks": 37,
          "sampling_rate": 24000,
          "frame_rate": 12.5
        },
        "acoustic_transformer_args": {
          "input_dim": 3072,
          "dim": 768,
          "n_layers": 3,
          "head_dim": 128,
          "hidden_dim": 2048,
          "n_heads": 6,
          "n_kv_heads": 2,
          "use_biases": false,
          "rope_theta": 10000.0,
          "sigma": 1e-05,
          "sigma_max": 1.0
        }
      },
      "audio_tokenizer_args": {
        "channels": 1,
        "sampling_rate": 24000,
        "pretransform_patch_size": 240,
        "patch_proj_kernel_size": 7,
        "semantic_codebook_size": 8192,
        "semantic_dim": 256,
        "acoustic_codebook_size": 21,
        "acoustic_dim": 36,
        "conv_weight_norm": true,
        "causal": true,
        "attn_sliding_window_size": 16,
        "half_attn_window_upon_downsampling": true,
        "dim": 1024,
        "hidden_dim": 4096,
        "head_dim": 128,
        "n_heads": 8,
        "n_kv_heads": 8,
        "qk_norm_eps": 1e-06,
        "qk_norm": true,
        "use_biases": false,
        "norm_eps": 0.01,
        "layer_scale": true,
        "layer_scale_init": 0.01,
        "decoder_transformer_lengths_str": "2,2,2,2",
        "decoder_convs_kernels_str": "3,4,4,4",
        "decoder_convs_strides_str": "1,2,2,2"
      }
    }
    """

    try Data(params.utf8).write(to: modelDir.appendingPathComponent("params.json"))
}

private func writeTekkenFixture(to modelDir: URL) throws {
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
}
