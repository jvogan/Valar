import Foundation
import Testing
@preconcurrency import MLX
import MLXNN
@testable import ValarMLX

@Suite("CodePredictor")
struct CodePredictorTests {

    // MARK: - Config

    @Test("Default config matches Qwen3-TTS code predictor spec")
    func defaultConfig() {
        let config = CodePredictorConfig.default
        #expect(config.vocabSize == 2048)
        #expect(config.hiddenSize == 1024)
        #expect(config.intermediateSize == 3072)
        #expect(config.numHiddenLayers == 5)
        #expect(config.numAttentionHeads == 16)
        #expect(config.numKeyValueHeads == 8)
        #expect(config.headDim == 128)
        #expect(config.rmsNormEps == 1e-6)
        #expect(config.ropeTheta == 1_000_000.0)
        #expect(config.attentionBias == false)
        #expect(config.numCodeGroups == 16)
        #expect(config.numPredictedGroups == 15)
    }

    // MARK: - KV Cache

    @Test("KVCache starts empty with zero offset")
    func kvCacheInitial() {
        let cache = CodePredictorKVCache()
        #expect(cache.keys == nil)
        #expect(cache.values == nil)
        #expect(cache.offset == 0)
    }

    @Test("KVCache accumulates keys and values across steps")
    func kvCacheAccumulation() {
        let cache = CodePredictorKVCache()
        let k1 = MLXRandom.normal([1, 2, 1, 8])
        let v1 = MLXRandom.normal([1, 2, 1, 8])
        let (keys1, values1) = cache.updateAndFetch(keys: k1, values: v1)
        #expect(keys1.shape == [1, 2, 1, 8])
        #expect(values1.shape == [1, 2, 1, 8])
        #expect(cache.offset == 1)

        let k2 = MLXRandom.normal([1, 2, 1, 8])
        let v2 = MLXRandom.normal([1, 2, 1, 8])
        let (keys2, values2) = cache.updateAndFetch(keys: k2, values: v2)
        #expect(keys2.shape == [1, 2, 2, 8])
        #expect(values2.shape == [1, 2, 2, 8])
        #expect(cache.offset == 2)
    }

    // MARK: - Sublayer shapes

    @Test("CodePredictorRMSNorm preserves shape")
    func rmsNormShape() {
        let norm = CodePredictorRMSNorm(dimensions: 16)
        let x = MLXRandom.normal([2, 5, 16])
        let y = norm(x)
        #expect(y.shape == [2, 5, 16])
    }

    @Test("CodePredictorMLP produces correct output shape")
    func mlpShape() {
        let config = smallConfig()
        let mlp = CodePredictorMLP(config: config)
        let x = MLXRandom.normal([2, 5, config.hiddenSize])
        let y = mlp(x)
        #expect(y.shape == [2, 5, config.hiddenSize])
    }

    @Test("CodePredictorAttention produces correct output shape")
    func attentionShape() {
        let config = smallConfig()
        let attn = CodePredictorAttention(config: config)
        let x = MLXRandom.normal([2, 5, config.hiddenSize])
        let y = attn(x, mask: nil, cache: nil, offset: 0)
        #expect(y.shape == [2, 5, config.hiddenSize])
    }

    @Test("CodePredictorAttention works with GQA (fewer KV heads than query heads)")
    func attentionGQA() {
        // Explicitly test GQA: 4 query heads, 2 KV heads
        let config = CodePredictorConfig(
            vocabSize: 64,
            hiddenSize: 32,
            intermediateSize: 64,
            numHiddenLayers: 1,
            numAttentionHeads: 4,
            numKeyValueHeads: 2,
            headDim: 8,
            rmsNormEps: 1e-6,
            ropeTheta: 10000.0,
            attentionBias: false,
            numCodeGroups: 4
        )
        let attn = CodePredictorAttention(config: config)
        let x = MLXRandom.normal([1, 3, config.hiddenSize])
        let y = attn(x, mask: nil, cache: nil, offset: 0)
        #expect(y.shape == [1, 3, config.hiddenSize])
    }

    @Test("CodePredictorDecoderLayer produces correct output shape")
    func layerShape() {
        let config = smallConfig()
        let layer = CodePredictorDecoderLayer(config: config)
        let x = MLXRandom.normal([2, 5, config.hiddenSize])
        let y = layer(x, mask: nil, cache: nil, offset: 0)
        #expect(y.shape == [2, 5, config.hiddenSize])
    }

    // MARK: - CodePredictorModel

    @Test("CodePredictorModel forward pass shape")
    func modelForwardShape() {
        let config = smallConfig()
        let model = CodePredictorModel(config: config, talkerHiddenSize: config.hiddenSize)
        let x = MLXRandom.normal([1, 4, config.hiddenSize])
        let y = model(x)
        #expect(y.shape == [1, 4, config.hiddenSize])
    }

    @Test("CodePredictorModel has correct layer count")
    func modelLayerCount() {
        let config = smallConfig()
        let model = CodePredictorModel(config: config, talkerHiddenSize: config.hiddenSize)
        #expect(model.layers.count == config.numHiddenLayers)
    }

    @Test("CodePredictorModel has correct number of codec embeddings")
    func modelEmbeddingCount() {
        let config = smallConfig()
        let model = CodePredictorModel(config: config, talkerHiddenSize: config.hiddenSize)
        #expect(model.codecEmbedding.count == config.numPredictedGroups)
    }

    @Test("CodePredictorModel single token skips mask")
    func modelSingleToken() {
        let config = smallConfig()
        let model = CodePredictorModel(config: config, talkerHiddenSize: config.hiddenSize)
        let x = MLXRandom.normal([1, 1, config.hiddenSize])
        let y = model(x)
        #expect(y.shape == [1, 1, config.hiddenSize])
    }

    @Test("CodePredictorModel makeCache returns correct count")
    func modelMakeCache() {
        let config = smallConfig()
        let model = CodePredictorModel(config: config, talkerHiddenSize: config.hiddenSize)
        let cache = model.makeCache()
        #expect(cache.count == config.numHiddenLayers)
    }

    // MARK: - Full CodePredictor

    @Test("CodePredictor forward pass produces correct logits shape")
    func predictorForwardShape() throws {
        let config = smallConfig()
        let predictor = CodePredictor(config: config, talkerHiddenSize: config.hiddenSize)
        let x = MLXRandom.normal([1, 4, config.hiddenSize])
        let (logits, _, nextStep) = try predictor(inputsEmbeds: x, generationStep: 0)
        #expect(logits.shape == [1, 4, config.vocabSize])
        #expect(nextStep == 1)
    }

    @Test("CodePredictor has 5 layers with default config")
    func predictorDefaultLayerCount() {
        let predictor = CodePredictor(config: .default, talkerHiddenSize: 1024)
        #expect(predictor.model.layers.count == 5)
    }

    @Test("CodePredictor has 15 LM heads")
    func predictorHeadCount() {
        let config = smallConfig()
        let predictor = CodePredictor(config: config, talkerHiddenSize: config.hiddenSize)
        #expect(predictor.lmHead.count == config.numPredictedGroups)
    }

    @Test("CodePredictor has 15 codec embeddings")
    func predictorEmbeddingCount() {
        let config = smallConfig()
        let predictor = CodePredictor(config: config, talkerHiddenSize: config.hiddenSize)
        #expect(predictor.codecEmbedding.count == config.numPredictedGroups)
    }

    @Test("CodePredictor without projection when sizes match")
    func predictorNoProjection() {
        let config = smallConfig()
        let predictor = CodePredictor(config: config, talkerHiddenSize: config.hiddenSize)
        #expect(predictor.projection == nil)
    }

    @Test("CodePredictor with projection when sizes differ")
    func predictorWithProjection() throws {
        let config = smallConfig()
        let talkerSize = config.hiddenSize * 2
        let predictor = CodePredictor(config: config, talkerHiddenSize: talkerSize)
        #expect(predictor.projection != nil)

        let x = MLXRandom.normal([1, 3, talkerSize])
        let (logits, _, _) = try predictor(inputsEmbeds: x)
        #expect(logits.shape == [1, 3, config.vocabSize])
    }

    @Test("CodePredictor generation step selects correct head")
    func predictorStepProgression() throws {
        let config = smallConfig()
        let predictor = CodePredictor(config: config, talkerHiddenSize: config.hiddenSize)
        let x = MLXRandom.normal([1, 2, config.hiddenSize])

        var step = 0
        for expected in 0 ..< config.numPredictedGroups {
            let (logits, _, nextStep) = try predictor(inputsEmbeds: x, generationStep: step)
            #expect(logits.shape == [1, 2, config.vocabSize])
            #expect(nextStep == expected + 1)
            step = nextStep
        }
    }

    @Test("CodePredictor throws when generation step is out of range")
    func predictorInvalidGenerationStep() {
        let config = smallConfig()

        do {
            _ = try CodePredictor.validateGenerationStep(
                config.numPredictedGroups,
                validRange: 0 ..< config.numPredictedGroups
            )
            Issue.record("Expected invalid generation step to throw")
        } catch let error as CodePredictorError {
            #expect(
                error == .invalidGenerationStep(
                    requested: config.numPredictedGroups,
                    validRange: 0 ..< config.numPredictedGroups
                )
            )
            #expect(error.errorDescription?.contains("\(config.numPredictedGroups)") == true)
        } catch {
            Issue.record("Expected CodePredictorError, got \(error)")
        }
    }

    @Test("CodePredictor makeCache returns correct count")
    func predictorMakeCache() {
        let config = smallConfig()
        let predictor = CodePredictor(config: config, talkerHiddenSize: config.hiddenSize)
        let cache = predictor.makeCache()
        #expect(cache.count == config.numHiddenLayers)
    }

    // MARK: - Weight Key Structure

    @Test("Module parameter keys match expected weight structure")
    func weightKeyStructure() {
        let config = smallConfig()
        let predictor = CodePredictor(config: config, talkerHiddenSize: config.hiddenSize)

        let params = predictor.parameters()
        let keys = Set(params.flattened().map(\.0))

        // LM heads
        #expect(keys.contains("lm_head.0.weight"))

        // Model norm
        #expect(keys.contains("model.norm.weight"))

        // Codec embeddings
        #expect(keys.contains("model.codec_embedding.0.weight"))

        // Layer 0 attention projections
        #expect(keys.contains("model.layers.0.self_attn.q_proj.weight"))
        #expect(keys.contains("model.layers.0.self_attn.k_proj.weight"))
        #expect(keys.contains("model.layers.0.self_attn.v_proj.weight"))
        #expect(keys.contains("model.layers.0.self_attn.o_proj.weight"))

        // Layer 0 QK norms
        #expect(keys.contains("model.layers.0.self_attn.q_norm.weight"))
        #expect(keys.contains("model.layers.0.self_attn.k_norm.weight"))

        // Layer 0 layernorms
        #expect(keys.contains("model.layers.0.input_layernorm.weight"))
        #expect(keys.contains("model.layers.0.post_attention_layernorm.weight"))

        // Layer 0 MLP
        #expect(keys.contains("model.layers.0.mlp.gate_proj.weight"))
        #expect(keys.contains("model.layers.0.mlp.up_proj.weight"))
        #expect(keys.contains("model.layers.0.mlp.down_proj.weight"))
    }

    @Test("Projection weight key present when sizes differ")
    func projectionWeightKey() {
        let config = smallConfig()
        let predictor = CodePredictor(config: config, talkerHiddenSize: config.hiddenSize * 2)

        let params = predictor.parameters()
        let keys = Set(params.flattened().map(\.0))

        #expect(keys.contains("small_to_mtp_projection.weight"))
        #expect(keys.contains("small_to_mtp_projection.bias"))
    }

    // MARK: - Helpers

    /// Small config for fast tests. Uses 2 layers with tiny dimensions.
    private func smallConfig() -> CodePredictorConfig {
        CodePredictorConfig(
            vocabSize: 64,
            hiddenSize: 32,
            intermediateSize: 64,
            numHiddenLayers: 2,
            numAttentionHeads: 4,
            numKeyValueHeads: 2,
            headDim: 8,
            rmsNormEps: 1e-6,
            ropeTheta: 10000.0,
            attentionBias: false,
            numCodeGroups: 4
        )
    }

}
