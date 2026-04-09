import Foundation
import Testing
import MLX
import MLXLMCommon

@testable import MLXAudioTTS

@Suite("VibeVoice Config Tests")
struct VibeVoiceConfigTests {

    // MARK: - Full config.json deserialization

    private static let fullConfigJSON = """
    {
      "model_type": "vibevoice_streaming",
      "sample_rate": 24000,
      "acoustic_vae_dim": 64,
      "tts_backbone_num_hidden_layers": 20,
      "decoder_config": {
        "model_type": "qwen2",
        "hidden_size": 896,
        "intermediate_size": 4864,
        "num_attention_heads": 14,
        "num_key_value_heads": 2,
        "num_hidden_layers": 24,
        "rms_norm_eps": 1e-6,
        "vocab_size": 151936,
        "max_position_embeddings": 8192,
        "rope_theta": 1000000.0,
        "tie_word_embeddings": false,
        "attention_dropout": 0.0,
        "hidden_act": "silu",
        "max_window_layers": 24,
        "use_cache": true,
        "use_sliding_window": false
      },
      "diffusion_head_config": {
        "model_type": "vibevoice_diffusion_head",
        "hidden_size": 896,
        "head_layers": 4,
        "head_ffn_ratio": 3.0,
        "rms_norm_eps": 1e-5,
        "latent_size": 64,
        "speech_vae_dim": 64,
        "prediction_type": "v_prediction",
        "diffusion_type": "ddpm",
        "ddpm_num_steps": 1000,
        "ddpm_num_inference_steps": 20,
        "ddpm_beta_schedule": "cosine",
        "ddpm_batch_mul": 4
      },
      "acoustic_tokenizer_config": {
        "model_type": "vibevoice_acoustic_tokenizer",
        "channels": 1,
        "vae_dim": 64,
        "causal": true,
        "fix_std": 0.5,
        "encoder_n_filters": 32,
        "encoder_ratios": [8, 5, 5, 4, 2, 2],
        "encoder_depths": "3-3-3-3-3-3-8",
        "mixer_layer": "depthwise_conv",
        "conv_bias": true,
        "layer_scale_init_value": 1e-6,
        "layernorm": "RMSNorm",
        "layernorm_eps": 1e-5
      }
    }
    """

    @Test("Full config.json decodes all top-level fields")
    func fullConfigDecode() throws {
        let config = try JSONDecoder().decode(
            VibeVoiceModelConfig.self,
            from: Data(Self.fullConfigJSON.utf8)
        )

        #expect(config.modelType == "vibevoice_streaming")
        #expect(config.sampleRate == 24_000)
        #expect(config.acousticVaeDim == 64)
        #expect(config.ttsBackboneNumHiddenLayers == 20)
    }

    @Test("Decoder config matches Qwen2-0.5B architecture")
    func decoderConfigValues() throws {
        let config = try JSONDecoder().decode(
            VibeVoiceModelConfig.self,
            from: Data(Self.fullConfigJSON.utf8)
        )
        let dc = config.decoderConfig

        #expect(dc.modelType == "qwen2")
        #expect(dc.hiddenSize == 896)
        #expect(dc.intermediateSize == 4_864)
        #expect(dc.numAttentionHeads == 14)
        #expect(dc.numKeyValueHeads == 2)
        #expect(dc.numHiddenLayers == 24)
        #expect(dc.rmsNormEps == 1e-6)
        #expect(dc.vocabSize == 151_936)
        #expect(dc.maxPositionEmbeddings == 8_192)
        #expect(dc.ropeTheta == 1_000_000.0)
        #expect(dc.tieWordEmbeddings == false)
    }

    @Test("Head dim derived correctly: hidden_size / num_attention_heads = 64")
    func effectiveHeadDim() throws {
        let config = try JSONDecoder().decode(
            VibeVoiceModelConfig.self,
            from: Data(Self.fullConfigJSON.utf8)
        )

        #expect(config.headDim == 64)
        #expect(config.decoderConfig.effectiveHeadDim == 64)
    }

    @Test("Layer split: 4 LM + 20 TTS = 24 total")
    func layerSplit() throws {
        let config = try JSONDecoder().decode(
            VibeVoiceModelConfig.self,
            from: Data(Self.fullConfigJSON.utf8)
        )

        #expect(config.languageModelLayers == 4)
        #expect(config.ttsLanguageModelLayers == 20)
        #expect(config.languageModelLayers + config.ttsLanguageModelLayers == config.decoderConfig.numHiddenLayers)
    }

    @Test("Diffusion head config matches reference values")
    func diffusionHeadConfig() throws {
        let config = try JSONDecoder().decode(
            VibeVoiceModelConfig.self,
            from: Data(Self.fullConfigJSON.utf8)
        )
        let dh = config.diffusionHeadConfig

        #expect(dh.modelType == "vibevoice_diffusion_head")
        #expect(dh.hiddenSize == 896)
        #expect(dh.headLayers == 4)
        #expect(dh.headFfnRatio == 3.0)
        #expect(dh.rmsNormEps == 1e-5)
        #expect(dh.latentSize == 64)
        #expect(dh.predictionType == "v_prediction")
        #expect(dh.ddpmNumSteps == 1_000)
        #expect(dh.ddpmNumInferenceSteps == 20)
        #expect(dh.ddpmBetaSchedule == "cosine")
        #expect(dh.ffnDim == 2_688)
    }

    @Test("Acoustic tokenizer config matches reference values")
    func acousticTokenizerConfig() throws {
        let config = try JSONDecoder().decode(
            VibeVoiceModelConfig.self,
            from: Data(Self.fullConfigJSON.utf8)
        )
        let at = config.acousticTokenizerConfig

        #expect(at.modelType == "vibevoice_acoustic_tokenizer")
        #expect(at.channels == 1)
        #expect(at.vaeDim == 64)
        #expect(at.causal == true)
        #expect(at.fixStd == 0.5)
        #expect(at.encoderNFilters == 32)
        #expect(at.encoderRatios == [8, 5, 5, 4, 2, 2])
        #expect(at.encoderDepths == "3-3-3-3-3-3-8")
        #expect(at.mixerLayer == "depthwise_conv")
        #expect(at.convBias == true)
    }

    @Test("Hop length = product of encoder_ratios = 3200")
    func hopLength() throws {
        let config = try JSONDecoder().decode(
            VibeVoiceModelConfig.self,
            from: Data(Self.fullConfigJSON.utf8)
        )

        #expect(config.hopLength == 3_200)
        #expect(config.acousticTokenizerConfig.hopLength == 3_200)
    }

    @Test("Parsed encoder depths from hyphen-separated string")
    func parsedDepths() throws {
        let config = try JSONDecoder().decode(
            VibeVoiceModelConfig.self,
            from: Data(Self.fullConfigJSON.utf8)
        )
        let at = config.acousticTokenizerConfig

        #expect(at.parsedEncoderDepths == [3, 3, 3, 3, 3, 3, 8])
        #expect(at.parsedDecoderDepths == [8, 3, 3, 3, 3, 3, 3])
    }

    // MARK: - Default value fallback

    @Test("Minimal JSON uses all defaults correctly")
    func defaultsFallback() throws {
        let minimal = """
        {
          "model_type": "vibevoice_streaming"
        }
        """
        let config = try JSONDecoder().decode(
            VibeVoiceModelConfig.self,
            from: Data(minimal.utf8)
        )

        #expect(config.sampleRate == 24_000)
        #expect(config.acousticVaeDim == 64)
        #expect(config.ttsBackboneNumHiddenLayers == 20)
        #expect(config.decoderConfig.hiddenSize == 896)
        #expect(config.diffusionHeadConfig.ddpmNumInferenceSteps == 20)
        #expect(config.acousticTokenizerConfig.vaeDim == 64)
    }

    @Test("Empty JSON object produces valid default config")
    func emptyJSONDefaults() throws {
        let config = try JSONDecoder().decode(
            VibeVoiceModelConfig.self,
            from: Data("{}".utf8)
        )

        #expect(config.modelType == "vibevoice_streaming")
        #expect(config.languageModelLayers == 4)
        #expect(config.ttsLanguageModelLayers == 20)
    }

    // MARK: - Quantization config

    @Test("quantization_config key is accepted")
    func quantizationConfigKey() throws {
        let json = """
        {
          "model_type": "vibevoice_streaming",
          "quantization_config": {
            "group_size": 64,
            "bits": 4
          }
        }
        """
        let config = try JSONDecoder().decode(
            VibeVoiceModelConfig.self,
            from: Data(json.utf8)
        )

        #expect(config.quantization != nil)
        #expect(config.quantization?.groupSize == 64)
        #expect(config.quantization?.bits == 4)
    }

    // MARK: - Sub-config isolation

    @Test("Qwen2DecoderConfig decodes independently")
    func qwen2DecoderConfigStandalone() throws {
        let json = """
        {
          "model_type": "qwen2",
          "hidden_size": 896,
          "num_attention_heads": 14,
          "num_key_value_heads": 2,
          "num_hidden_layers": 24,
          "head_dim": 64
        }
        """
        let config = try JSONDecoder().decode(
            VibeVoiceQwen2DecoderConfig.self,
            from: Data(json.utf8)
        )

        #expect(config.effectiveHeadDim == 64)
        #expect(config.languageModelLayerCount(ttsBackboneNumHiddenLayers: 20) == 4)
    }

    @Test("DiffusionHeadConfig decodes independently")
    func diffusionHeadConfigStandalone() throws {
        let json = """
        {
          "hidden_size": 896,
          "head_ffn_ratio": 3.0,
          "latent_size": 64
        }
        """
        let config = try JSONDecoder().decode(
            VibeVoiceDiffusionHeadConfig.self,
            from: Data(json.utf8)
        )

        #expect(config.ffnDim == 2_688)
        #expect(config.predictionType == "v_prediction")
    }

    // MARK: - Codable round-trip

    @Test("Config survives encode-decode round-trip")
    func codableRoundTrip() throws {
        let original = try JSONDecoder().decode(
            VibeVoiceModelConfig.self,
            from: Data(Self.fullConfigJSON.utf8)
        )

        let encoded = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(VibeVoiceModelConfig.self, from: encoded)

        #expect(original == decoded)
    }

    // MARK: - Generation constants

    @Test("Static generation constants match spec")
    func generationConstants() {
        #expect(VibeVoiceModelConfig.ttsTextWindowSize == 5)
        #expect(VibeVoiceModelConfig.ttsSpeechWindowSize == 6)
        #expect(VibeVoiceModelConfig.streamingSpeechWindowSize == 4)
        #expect(VibeVoiceModelConfig.streamingSpeechFrameRateHz == 12.5)
        #expect(VibeVoiceModelConfig.defaultCFGScale == 1.5)
        #expect(VibeVoiceModelConfig.defaultEOSThreshold == 0.5)
        #expect(VibeVoiceModelConfig.defaultMaxTokens == 512)
    }

    @Test("Streaming chunk size resolution respects requested interval and clamps safely")
    func streamingChunkSizeResolution() {
        #expect(VibeVoiceTTSModel.resolvedStreamingChunkSize(for: 0.32) == 4)
        #expect(VibeVoiceTTSModel.resolvedStreamingChunkSize(for: 0.08) == 1)
        #expect(VibeVoiceTTSModel.resolvedStreamingChunkSize(for: 10.0) == 6)
        #expect(VibeVoiceTTSModel.resolvedStreamingChunkSize(for: 0.0) == 4)
        #expect(VibeVoiceTTSModel.resolvedStreamingChunkSize(for: -0.1) == 4)
    }

    @Test("Deterministic generation seed is stable per text and voice")
    func deterministicGenerationSeed() {
        let first = VibeVoiceTTSModel.deterministicGenerationSeed(
            text: "hello\n",
            voiceName: "en-Emma_woman"
        )
        let second = VibeVoiceTTSModel.deterministicGenerationSeed(
            text: "hello\n",
            voiceName: "en-Emma_woman"
        )
        let differentVoice = VibeVoiceTTSModel.deterministicGenerationSeed(
            text: "hello\n",
            voiceName: "en-Carter_man"
        )

        #expect(first == second)
        #expect(first != differentVoice)
    }

    @Test("Special token type IDs")
    func specialTokens() {
        #expect(VibeVoiceModelConfig.SpecialTokens.speechTypeID == 0)
        #expect(VibeVoiceModelConfig.SpecialTokens.textTypeID == 1)
    }

    // MARK: - Voice cache loading (mock-based)

    @Test("loadVoiceCache throws on missing file")
    func voiceCacheMissingFile() {
        let config = VibeVoiceModelConfig()
        let bogusURL = URL(fileURLWithPath: "/nonexistent/voice.safetensors")

        #expect(throws: (any Error).self) {
            try loadVoiceCache(from: bogusURL, config: config)
        }
    }

    @Test("loadPresetVoices throws on missing voices directory")
    func presetVoicesMissingDir() {
        let config = VibeVoiceModelConfig()
        let bogusDir = URL(fileURLWithPath: "/nonexistent/model")

        #expect(throws: VibeVoiceVoiceCacheError.self) {
            try loadPresetVoices(from: bogusDir, config: config)
        }
    }

    @Test("loadPresetVoices returns empty dict for empty voices directory")
    func presetVoicesEmptyDir() throws {
        let config = VibeVoiceModelConfig()
        let tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("vibevoice-test-\(UUID().uuidString)")
        let voicesDir = tmpDir.appendingPathComponent("voices")

        try FileManager.default.createDirectory(at: voicesDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        let result = try loadPresetVoices(from: tmpDir, config: config)
        #expect(result.isEmpty)
    }

    @Test("VibeVoiceKVSnapshot stores correct layer counts")
    func kvSnapshotStructure() {
        let dummyArray = MLXArray.zeros([1, 2, 10, 64])
        let dummyHidden = MLXArray.zeros([1, 10, 896])

        let lmPair: (key: MLXArray, value: MLXArray) = (key: dummyArray, value: dummyArray)
        let ttsPair: (key: MLXArray, value: MLXArray) = (key: dummyArray, value: dummyArray)

        let snapshot = VibeVoiceKVSnapshot(
            lmHidden: dummyHidden,
            lmCache: Array(repeating: lmPair, count: 4),
            ttsLmHidden: dummyHidden,
            ttsLmCache: Array(repeating: ttsPair, count: 20),
            negTtsLmHidden: dummyHidden,
            negTtsLmCache: Array(repeating: ttsPair, count: 20),
            negLmCache: Array(repeating: lmPair, count: 4)
        )

        #expect(snapshot.lmCache.count == 4)
        #expect(snapshot.ttsLmCache.count == 20)
        #expect(snapshot.negTtsLmCache.count == 20)
        #expect(snapshot.negLmCache?.count == 4)
        #expect(snapshot.lmHidden.shape == [1, 10, 896])
    }

    @Test("VibeVoiceKVSnapshot allows nil negLmCache")
    func kvSnapshotOptionalNegLm() {
        let dummyArray = MLXArray.zeros([1, 2, 10, 64])
        let dummyHidden = MLXArray.zeros([1, 10, 896])

        let pair: (key: MLXArray, value: MLXArray) = (key: dummyArray, value: dummyArray)

        let snapshot = VibeVoiceKVSnapshot(
            lmHidden: dummyHidden,
            lmCache: Array(repeating: pair, count: 4),
            ttsLmHidden: dummyHidden,
            ttsLmCache: Array(repeating: pair, count: 20),
            negTtsLmHidden: dummyHidden,
            negTtsLmCache: Array(repeating: pair, count: 20),
            negLmCache: nil
        )

        #expect(snapshot.negLmCache == nil)
    }

    @Test("VibeVoice parameter flattening exposes camelCase Swift module paths")
    func parameterFlatteningUsesSwiftModuleNames() {
        let model = VibeVoiceTTSModel(config: VibeVoiceModelConfig())
        let currentWeights = Dictionary(uniqueKeysWithValues: model.parameters().flattened())

        #expect(currentWeights["languageModel.embedTokens.weight"] != nil)
        #expect(currentWeights["languageModel.layers.0.selfAttn.kProj.weight"] != nil)
        #expect(currentWeights["ttsLanguageModel.layers.0.selfAttn.kProj.weight"] != nil)
        #expect(currentWeights["ttsInputTypes.weight"] != nil)
        #expect(currentWeights["acousticConnector.fc1.weight"] != nil)
        #expect(currentWeights["predictionHead.noisyImagesProj.weight"] != nil)
        #expect(currentWeights["predictionHead.tEmbedder.mlp.inputProjection.weight"] != nil)
        #expect(currentWeights["predictionHead.tEmbedder.mlp.outputProjection.weight"] != nil)
        #expect(currentWeights["predictionHead.layers.0.adaLNModulation.projection.weight"] != nil)
        #expect(currentWeights["ttsEosClassifier.fc1.weight"] != nil)
        #expect(currentWeights["acousticTokenizer.decoder.stem.conv.conv.weight"] != nil)
        #expect(currentWeights["acousticTokenizer.decoder.upsamplers.0.convtr.convtr.weight"] != nil)
        #expect(currentWeights["acousticTokenizer.decoder.stages.0.layers.0.norm.weight"] != nil)
        #expect(currentWeights["acousticTokenizer.decoder.stages.0.layers.0.ffnNorm.weight"] != nil)
        #expect(currentWeights["acousticTokenizer.decoder.stages.0.layers.0.gamma"] != nil)
        #expect(currentWeights["acousticTokenizer.decoder.stages.0.layers.0.ffnGamma"] != nil)
        #expect(currentWeights["acousticTokenizer.decoder.head.conv.conv.weight"] != nil)
    }

    @Test("VibeVoice sanitize maps snake_case HF keys and transposes decoder weights")
    func sanitizeMapsSnakeCaseHFWeights() throws {
        let model = VibeVoiceTTSModel(config: VibeVoiceModelConfig())
        let currentWeights = Dictionary(uniqueKeysWithValues: model.parameters().flattened())

        func shape(for key: String) throws -> [Int] {
            guard let value = currentWeights[key] else {
                throw VibeVoiceTestError("missing current parameter '\(key)'")
            }
            return value.shape
        }

        let languageModelKProjKey = "languageModel.layers.0.selfAttn.kProj.weight"
        let speechConnectorKey = "acousticConnector.fc1.weight"
        let stemConvKey = "acousticTokenizer.decoder.stem.conv.conv.weight"
        let upsampleConvTrKey = "acousticTokenizer.decoder.upsamplers.0.convtr.convtr.weight"
        let normWeightKey = "acousticTokenizer.decoder.stages.0.layers.0.norm.weight"
        let gammaKey = "acousticTokenizer.decoder.stages.0.layers.0.gamma"
        let ffnGammaKey = "acousticTokenizer.decoder.stages.0.layers.0.ffnGamma"
        let ttsInputTypesKey = "ttsInputTypes.weight"

        let languageModelKProjShape = try shape(for: languageModelKProjKey)
        let speechConnectorShape = try shape(for: speechConnectorKey)
        let stemConvShape = try shape(for: stemConvKey)
        let upsampleConvTrShape = try shape(for: upsampleConvTrKey)
        let normWeightShape = try shape(for: normWeightKey)
        let gammaShape = try shape(for: gammaKey)
        let ffnGammaShape = try shape(for: ffnGammaKey)
        let ttsInputTypesShape = try shape(for: ttsInputTypesKey)

        let sanitized = model.sanitize(weights: [
            "model.language_model.layers.0.self_attn.k_proj.weight":
                MLXArray.zeros([languageModelKProjShape[1], languageModelKProjShape[0]], dtype: .float32),
            "model.language_model.layers.0.self_attn.k_proj.scales":
                MLXArray.zeros([languageModelKProjShape[0], 2], dtype: .float32),
            "model.language_model.layers.0.self_attn.k_proj.biases":
                MLXArray.zeros([languageModelKProjShape[0], 2], dtype: .float32),
            "model.acoustic_connector.fc1.weight":
                MLXArray.zeros([speechConnectorShape[1], speechConnectorShape[0]], dtype: .float32),
            "model.acoustic_tokenizer.decoder.upsample_layers.0.0.conv.conv.weight":
                MLXArray.zeros([stemConvShape[0], stemConvShape[2], stemConvShape[1]], dtype: .float32),
            "model.acoustic_tokenizer.decoder.upsample_layers.1.0.convtr.convtr.weight":
                MLXArray.zeros([upsampleConvTrShape[2], upsampleConvTrShape[0], upsampleConvTrShape[1]], dtype: .float32),
            "model.acoustic_tokenizer.decoder.stages.0.0.norm.weight":
                MLXArray.zeros(normWeightShape, dtype: .float32),
            "model.acoustic_tokenizer.decoder.stages.0.0.gamma":
                MLXArray.zeros(gammaShape, dtype: .float32),
            "model.acoustic_tokenizer.decoder.stages.0.0.ffn_gamma":
                MLXArray.zeros(ffnGammaShape, dtype: .float32),
            "model.tts_input_types.weight":
                MLXArray.zeros(ttsInputTypesShape, dtype: .float32),
            "model.speech_scaling_factor": MLXArray(Float(0.5)),
            "model.speech_bias_factor": MLXArray(Float(-0.25)),
        ])

        #expect(sanitized[languageModelKProjKey]?.shape == languageModelKProjShape)
        #expect(sanitized["languageModel.layers.0.selfAttn.kProj.scales"] != nil)
        #expect(sanitized["languageModel.layers.0.selfAttn.kProj.biases"] != nil)
        #expect(sanitized[speechConnectorKey]?.shape == speechConnectorShape)
        #expect(sanitized[stemConvKey]?.shape == stemConvShape)
        #expect(sanitized[upsampleConvTrKey]?.shape == upsampleConvTrShape)
        #expect(sanitized[normWeightKey]?.shape == normWeightShape)
        #expect(sanitized[gammaKey]?.shape == gammaShape)
        #expect(sanitized[ffnGammaKey]?.shape == ffnGammaShape)
        #expect(sanitized[ttsInputTypesKey]?.shape == ttsInputTypesShape)
        #expect(sanitized["speechScalingFactor"] != nil)
        #expect(sanitized["speechBiasFactor"] != nil)
    }
}

private struct VibeVoiceTestError: Error, CustomStringConvertible {
    let message: String

    init(_ message: String) {
        self.message = message
    }

    var description: String { message }
}
