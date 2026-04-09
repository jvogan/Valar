import Foundation
import Testing
@preconcurrency import MLX
import MLXNN
@testable import ValarMLX

@Suite("DecoderTransformer")
struct DecoderTransformerTests {

    // MARK: - Config

    @Test("Default config matches Qwen3-TTS decoder spec")
    func defaultConfig() {
        let config = DecoderTransformerConfig.default
        #expect(config.hiddenSize == 512)
        #expect(config.intermediateSize == 1024)
        #expect(config.numHiddenLayers == 8)
        #expect(config.numAttentionHeads == 16)
        #expect(config.numKeyValueHeads == 16)
        #expect(config.headDim == 64)
        #expect(config.slidingWindow == 72)
        #expect(config.latentDim == 1024)
        #expect(config.attentionBias == false)
    }

    // MARK: - Sliding Window Causal Mask

    @Test("Causal mask blocks future positions")
    func causalMaskBlocksFuture() {
        let mask = DecoderTransformer.createSlidingWindowCausalMask(
            seqLen: 4, windowSize: 100
        )
        #expect(mask.shape == [4, 4])

        // Force evaluation
        MLX.eval(mask)

        // Upper triangle should be -1e9 (masked), lower triangle + diagonal should be 0
        let values = mask.asArray(Float.self)

        // Diagonal: all zeros
        for i in 0 ..< 4 {
            #expect(values[i * 4 + i] == 0.0)
        }
        // Below diagonal: zeros (can attend)
        #expect(values[1 * 4 + 0] == 0.0) // row 1, col 0
        #expect(values[2 * 4 + 1] == 0.0) // row 2, col 1
        #expect(values[3 * 4 + 2] == 0.0) // row 3, col 2

        // Above diagonal: large negative (masked)
        #expect(values[0 * 4 + 1] < -1e8) // row 0, col 1
        #expect(values[0 * 4 + 3] < -1e8) // row 0, col 3
        #expect(values[1 * 4 + 2] < -1e8) // row 1, col 2
    }

    @Test("Sliding window mask limits attention span")
    func slidingWindowMask() {
        // Window size 2: each position can attend to itself and one previous position
        let mask = DecoderTransformer.createSlidingWindowCausalMask(
            seqLen: 5, windowSize: 2
        )
        #expect(mask.shape == [5, 5])
        MLX.eval(mask)

        let values = mask.asArray(Float.self)

        // Row 0: can attend to col 0 only (diagonal)
        #expect(values[0 * 5 + 0] == 0.0)
        #expect(values[0 * 5 + 1] < -1e8) // future: masked

        // Row 1: can attend to cols 0,1 (within window of 2)
        #expect(values[1 * 5 + 0] == 0.0)
        #expect(values[1 * 5 + 1] == 0.0)
        #expect(values[1 * 5 + 2] < -1e8) // future: masked

        // Row 2: can attend to cols 1,2 (window=2, so col 0 is too far back)
        #expect(values[2 * 5 + 0] < -1e8) // outside window: masked
        #expect(values[2 * 5 + 1] == 0.0)
        #expect(values[2 * 5 + 2] == 0.0)
        #expect(values[2 * 5 + 3] < -1e8) // future: masked

        // Row 3: can attend to cols 2,3
        #expect(values[3 * 5 + 0] < -1e8) // outside window
        #expect(values[3 * 5 + 1] < -1e8) // outside window
        #expect(values[3 * 5 + 2] == 0.0)
        #expect(values[3 * 5 + 3] == 0.0)
        #expect(values[3 * 5 + 4] < -1e8) // future: masked

        // Row 4: can attend to cols 3,4
        #expect(values[4 * 5 + 2] < -1e8) // outside window
        #expect(values[4 * 5 + 3] == 0.0)
        #expect(values[4 * 5 + 4] == 0.0)
    }

    @Test("Window size 1 means self-attention only")
    func windowSizeOne() {
        let mask = DecoderTransformer.createSlidingWindowCausalMask(
            seqLen: 3, windowSize: 1
        )
        MLX.eval(mask)
        let values = mask.asArray(Float.self)

        // Only diagonal should be 0
        #expect(values[0 * 3 + 0] == 0.0)
        #expect(values[1 * 3 + 1] == 0.0)
        #expect(values[2 * 3 + 2] == 0.0)

        // Everything else should be masked
        #expect(values[1 * 3 + 0] < -1e8) // row 1, col 0: delta=1 >= window=1
        #expect(values[2 * 3 + 0] < -1e8)
        #expect(values[2 * 3 + 1] < -1e8)
    }

    // MARK: - Sublayer shapes

    @Test("DecoderRMSNorm preserves shape")
    func rmsNormShape() {
        let norm = DecoderRMSNorm(hiddenSize: 16)
        let x = MLXRandom.normal([2, 5, 16])
        let y = norm(x)
        #expect(y.shape == [2, 5, 16])
    }

    @Test("LayerScale preserves shape")
    func layerScaleShape() {
        let ls = LayerScale(channels: 16, initialScale: 0.01)
        let x = MLXRandom.normal([2, 5, 16])
        let y = ls(x)
        #expect(y.shape == [2, 5, 16])
    }

    @Test("DecoderMLP produces correct output shape")
    func mlpShape() {
        let config = smallConfig()
        let mlp = DecoderMLP(config: config)
        let x = MLXRandom.normal([2, 5, config.hiddenSize])
        let y = mlp(x)
        #expect(y.shape == [2, 5, config.hiddenSize])
    }

    @Test("DecoderAttention produces correct output shape")
    func attentionShape() {
        let config = smallConfig()
        let attn = DecoderAttention(config: config)
        let x = MLXRandom.normal([2, 5, config.hiddenSize])
        let y = attn(x, mask: nil, offset: 0)
        #expect(y.shape == [2, 5, config.hiddenSize])
    }

    @Test("DecoderTransformerLayer produces correct output shape")
    func layerShape() {
        let config = smallConfig()
        let layer = DecoderTransformerLayer(config: config)
        let x = MLXRandom.normal([2, 5, config.hiddenSize])
        let y = layer(x, mask: nil, offset: 0)
        #expect(y.shape == [2, 5, config.hiddenSize])
    }

    // MARK: - Full Transformer

    @Test("DecoderTransformer forward pass with correct input/output shapes")
    func forwardPassShape() {
        let config = smallConfig()
        let transformer = DecoderTransformer(config: config)
        let x = MLXRandom.normal([1, 10, config.latentDim])
        let y = transformer(x)
        #expect(y.shape == [1, 10, config.latentDim])
    }

    @Test("DecoderTransformer has 8 layers with default config")
    func layerCount() {
        let transformer = DecoderTransformer(config: .default)
        #expect(transformer.layers.count == 8)
    }

    @Test("DecoderTransformer with sequence length 1 skips mask")
    func singleTokenNoMask() {
        let config = smallConfig()
        let transformer = DecoderTransformer(config: config)
        let x = MLXRandom.normal([1, 1, config.latentDim])
        let y = transformer(x)
        #expect(y.shape == [1, 1, config.latentDim])
    }

    @Test("DecoderTransformer with explicit mask uses provided mask")
    func explicitMask() {
        let config = smallConfig()
        let transformer = DecoderTransformer(config: config)
        let seqLen = 4
        let x = MLXRandom.normal([1, seqLen, config.latentDim])
        // Standard causal mask (no sliding window)
        let mask = MultiHeadAttention.createAdditiveCausalMask(seqLen)
        let y = transformer(x, mask: mask)
        #expect(y.shape == [1, seqLen, config.latentDim])
    }

    // MARK: - Weight Key Structure

    @Test("Module parameter keys match expected weight structure")
    func weightKeyStructure() {
        let config = smallConfig()
        let transformer = DecoderTransformer(config: config)

        let params = transformer.parameters()
        let keys = Set(params.flattened().map(\.0))

        // Input/output projections (these match post-remapKey names)
        #expect(keys.contains("inputProj.weight"))
        #expect(keys.contains("inputProj.bias"))
        #expect(keys.contains("outputProj.weight"))
        #expect(keys.contains("outputProj.bias"))

        // Final norm
        #expect(keys.contains("norm.weight"))

        // Layer 0 attention (key: annotations match Python snake_case)
        #expect(keys.contains("layers.0.self_attn.q_proj.weight"))
        #expect(keys.contains("layers.0.self_attn.k_proj.weight"))
        #expect(keys.contains("layers.0.self_attn.v_proj.weight"))
        #expect(keys.contains("layers.0.self_attn.o_proj.weight"))

        // Layer 0 norms
        #expect(keys.contains("layers.0.input_layernorm.weight"))
        #expect(keys.contains("layers.0.post_attention_layernorm.weight"))

        // Layer 0 layer scales
        #expect(keys.contains("layers.0.self_attn_layer_scale.scale"))
        #expect(keys.contains("layers.0.mlp_layer_scale.scale"))

        // Layer 0 MLP
        #expect(keys.contains("layers.0.mlp.gate_proj.weight"))
        #expect(keys.contains("layers.0.mlp.up_proj.weight"))
        #expect(keys.contains("layers.0.mlp.down_proj.weight"))
    }

    // MARK: - Helpers

    /// Small config for fast tests. Uses 2 layers with tiny dimensions.
    private func smallConfig() -> DecoderTransformerConfig {
        DecoderTransformerConfig(
            hiddenSize: 32,
            intermediateSize: 64,
            numHiddenLayers: 2,
            numAttentionHeads: 4,
            numKeyValueHeads: 4,
            headDim: 8,
            rmsNormEps: 1e-5,
            ropeTheta: 10000.0,
            slidingWindow: 8,
            layerScaleInitialScale: 0.01,
            latentDim: 16,
            attentionBias: false
        )
    }

    /// Recursively collect all parameter key paths from a nested dictionary.
}
