import Foundation
import Testing
@preconcurrency import MLX
import MLXNN
@testable import ValarMLX

@Suite("SpeechTokenizerDecoder")
struct SpeechTokenizerDecoderTests {

    // MARK: - Test Helpers

    /// Small config for fast shape-verification tests.
    ///
    /// decoderDim=48, upsampleRates=[2,3], upsamplingRatios=[2]
    /// → totalUpsample=12, outputDim=12, decoder has 5 elements.
    static func smallConfig() -> SpeechTokenizerDecoderConfig {
        SpeechTokenizerDecoderConfig(
            codebookDim: 16,
            codebookSize: 32,
            numQuantizers: 2,
            numSemanticQuantizers: 1,
            latentDim: 32,
            decoderDim: 48,
            upsampleRates: [2, 3],
            upsamplingRatios: [2],
            hiddenSize: 16,
            intermediateSize: 32,
            numHiddenLayers: 1,
            numAttentionHeads: 2,
            numKeyValueHeads: 2,
            headDim: 8,
            rmsNormEps: 1e-5,
            ropeTheta: 10000.0,
            slidingWindow: 8,
            layerScaleInitialScale: 0.01,
            attentionBias: false
        )
    }

    // MARK: - Config Tests

    @Test("Default config matches Qwen3-TTS speech tokenizer decoder spec")
    func defaultConfig() {
        let config = SpeechTokenizerDecoderConfig.default
        #expect(config.codebookDim == 512)
        #expect(config.codebookSize == 2048)
        #expect(config.numQuantizers == 16)
        #expect(config.numSemanticQuantizers == 1)
        #expect(config.latentDim == 1024)
        #expect(config.decoderDim == 1536)
        #expect(config.upsampleRates == [8, 5, 4, 3])
        #expect(config.upsamplingRatios == [2, 2])
        #expect(config.hiddenSize == 512)
        #expect(config.intermediateSize == 1024)
        #expect(config.numHiddenLayers == 8)
        #expect(config.numAttentionHeads == 16)
        #expect(config.numKeyValueHeads == 16)
        #expect(config.headDim == 64)
        #expect(config.slidingWindow == 72)
        #expect(config.attentionBias == false)
    }

    @Test("totalUpsample computes product of all rates")
    func totalUpsample() {
        let config = SpeechTokenizerDecoderConfig.default
        // upsampleRates=[8,5,4,3] → 480, upsamplingRatios=[2,2] → 4, total=1920
        #expect(config.totalUpsample == 1920)
    }

    @Test("outputDim halves channels per decoder block")
    func outputDim() {
        let config = SpeechTokenizerDecoderConfig.default
        // 1536 / 2^4 = 96
        #expect(config.outputDim == 96)
    }

    @Test("Small config derived properties are consistent")
    func smallConfigProperties() {
        let config = Self.smallConfig()
        // upsampleRates=[2,3], upsamplingRatios=[2] → 2*3*2 = 12
        #expect(config.totalUpsample == 12)
        // 48 / 2^2 = 12
        #expect(config.outputDim == 12)
    }

    @Test("Transformer config derives correctly from decoder config")
    func transformerConfigDerivation() {
        let config = SpeechTokenizerDecoderConfig.default
        let tc = config.transformerConfig
        #expect(tc.hiddenSize == 512)
        #expect(tc.intermediateSize == 1024)
        #expect(tc.numHiddenLayers == 8)
        #expect(tc.latentDim == 1024)
        #expect(tc.slidingWindow == 72)
    }

    @Test("Decoder block config derives correctly from decoder config")
    func decoderBlockConfigDerivation() {
        let config = SpeechTokenizerDecoderConfig.default
        let bc = config.decoderBlockConfig
        #expect(bc.decoderDim == 1536)
        #expect(bc.upsampleRates == [8, 5, 4, 3])
    }

    // MARK: - Decoder Array Structure

    @Test("Decoder array has correct element count and types")
    func decoderArrayStructure() {
        let config = Self.smallConfig()
        let decoder = SpeechTokenizerDecoder(config: config)
        // [initialConv, block0, block1, outputSnake, outputConv]
        let numBlocks = config.upsampleRates.count
        let expectedCount = 1 + numBlocks + 1 + 1 // initial + blocks + snake + output
        #expect(decoder.decoder.count == expectedCount)
        #expect(decoder.decoder[0] is CausalConv1d)
        for i in 1 ... numBlocks {
            #expect(decoder.decoder[i] is DecoderBlock)
        }
        #expect(decoder.decoder[numBlocks + 1] is SnakeBeta)
        #expect(decoder.decoder[numBlocks + 2] is CausalConv1d)
    }

    @Test("Default decoder array has 7 elements matching Python structure")
    func defaultDecoderArrayStructure() {
        let config = SpeechTokenizerDecoderConfig.default
        let decoder = SpeechTokenizerDecoder(config: config)
        // [initial, block0, block1, block2, block3, snake, output] = 7
        #expect(decoder.decoder.count == 7)
        #expect(decoder.decoder[0] is CausalConv1d)
        #expect(decoder.decoder[1] is DecoderBlock)
        #expect(decoder.decoder[2] is DecoderBlock)
        #expect(decoder.decoder[3] is DecoderBlock)
        #expect(decoder.decoder[4] is DecoderBlock)
        #expect(decoder.decoder[5] is SnakeBeta)
        #expect(decoder.decoder[6] is CausalConv1d)
    }

    // MARK: - Forward Pass Shape

    @Test("Full forward pass produces correct output shape")
    func forwardPassShape() {
        let config = Self.smallConfig()
        let decoder = SpeechTokenizerDecoder(config: config)
        // codes: [batch=1, numQuantizers=2, time=4]
        let codes = MLXArray(Array(repeating: Int32(0), count: 1 * 2 * 4)).reshaped(1, 2, 4)
        let output = decoder(codes)
        MLX.eval(output)
        // time=4, totalUpsample=12 → 48 samples
        #expect(output.shape == [1, 1, 48])
    }

    @Test("Output is clipped to [-1, 1]")
    func outputClipping() {
        let config = Self.smallConfig()
        let decoder = SpeechTokenizerDecoder(config: config)
        let codes = MLXArray(Array(repeating: Int32(0), count: 1 * 2 * 4)).reshaped(1, 2, 4)
        let output = decoder(codes)
        MLX.eval(output)
        let maxVal = output.max().item(Float.self)
        let minVal = output.min().item(Float.self)
        #expect(maxVal <= 1.0)
        #expect(minVal >= -1.0)
    }

    @Test("Batch dimension is preserved")
    func batchDimensionPreserved() {
        let config = Self.smallConfig()
        let decoder = SpeechTokenizerDecoder(config: config)
        let codes = MLXArray(Array(repeating: Int32(0), count: 2 * 2 * 4)).reshaped(2, 2, 4)
        let output = decoder(codes)
        MLX.eval(output)
        #expect(output.shape == [2, 1, 48])
    }

    // MARK: - Chunked Decode

    @Test("Chunked decode produces correct output length")
    func chunkedDecodeShape() {
        let config = Self.smallConfig()
        let decoder = SpeechTokenizerDecoder(config: config)
        let totalFrames = 8
        let codes = MLXArray(Array(repeating: Int32(0), count: 1 * 2 * totalFrames)).reshaped(1, 2, totalFrames)
        let output = decoder.chunkedDecode(codes, chunkSize: 4, leftContextSize: 2)
        MLX.eval(output)
        // totalFrames=8, totalUpsample=12 → 96 samples
        #expect(output.shape == [1, 1, totalFrames * config.totalUpsample])
    }

    // MARK: - Weight Key Structure

    @Test("Module tree generates expected weight key prefixes")
    func weightKeyPrefixes() {
        let config = Self.smallConfig()
        let decoder = SpeechTokenizerDecoder(config: config)
        let params = decoder.parameters()
        let keys = collectLeafKeys(from: params)

        // Quantizer keys
        #expect(keys.contains { $0.hasPrefix("quantizer.") })
        // Pre-conv keys
        #expect(keys.contains { $0.hasPrefix("preConv.") })
        // Pre-transformer keys
        #expect(keys.contains { $0.hasPrefix("preTransformer.") })
        // Upsample keys
        #expect(keys.contains { $0.hasPrefix("upsample.") })
        // Decoder array keys
        #expect(keys.contains { $0.hasPrefix("decoder.0.") })
        #expect(keys.contains { $0.hasPrefix("decoder.1.") })
    }

    @Test("Initial conv weight key matches Python decoder.0.conv.weight")
    func initialConvWeightKey() {
        let config = Self.smallConfig()
        let decoder = SpeechTokenizerDecoder(config: config)
        let keys = collectLeafKeys(from: decoder.parameters())
        #expect(keys.contains("decoder.0.conv.weight"))
        #expect(keys.contains("decoder.0.conv.bias"))
    }

    @Test("Output snake weight keys match Python decoder.N+1.alpha/beta")
    func outputSnakeWeightKeys() {
        let config = Self.smallConfig()
        let decoder = SpeechTokenizerDecoder(config: config)
        let keys = collectLeafKeys(from: decoder.parameters())
        let snakeIndex = config.upsampleRates.count + 1 // after initial conv + blocks
        #expect(keys.contains("decoder.\(snakeIndex).alpha"))
        #expect(keys.contains("decoder.\(snakeIndex).beta"))
    }

    @Test("Output conv weight keys match Python decoder.N+2.conv.weight")
    func outputConvWeightKeys() {
        let config = Self.smallConfig()
        let decoder = SpeechTokenizerDecoder(config: config)
        let keys = collectLeafKeys(from: decoder.parameters())
        let convIndex = config.upsampleRates.count + 2
        #expect(keys.contains("decoder.\(convIndex).conv.weight"))
        #expect(keys.contains("decoder.\(convIndex).conv.bias"))
    }

    // MARK: - Weight Loader Integration

    @Test("decoderWeights strips decoder. prefix")
    func decoderWeightsPrefixStripping() {
        let input: [String: MLXArray] = [
            "decoder.quantizer.rvqFirst.vq.layers.0.codebook.embed.weight": MLXArray.zeros([4]),
            "decoder.preConv.conv.weight": MLXArray.zeros([4]),
            "decoder.decoder.0.conv.weight": MLXArray.zeros([4]),
            "encoder.something": MLXArray.zeros([4]),
        ]
        let result = SpeechTokenizerWeightLoader.decoderWeights(from: input)
        #expect(result.count == 3)
        #expect(result["quantizer.rvqFirst.vq.layers.0.codebook.embed.weight"] != nil)
        #expect(result["preConv.conv.weight"] != nil)
        #expect(result["decoder.0.conv.weight"] != nil)
        #expect(result["encoder.something"] == nil)
    }

    // MARK: - Helpers

    /// Collect all leaf key paths from module parameters using the built-in flattened() method.
    private func collectLeafKeys(from params: ModuleParameters) -> Set<String> {
        Set(params.flattened().map(\.0))
    }
}
