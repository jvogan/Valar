import Foundation
@preconcurrency import MLX
import MLXNN
import Testing
@testable import ValarMLX

// MARK: - Golden Corpus End-to-End Parity Tests
//
// Exercises the full ValarMLX speech pipeline against golden fixture data
// to verify parity with the Python Qwen3-TTS reference implementation.
//
// Coverage:
//   1. Chat template formatting and token structure
//   2. SpeechTokenizerDecoderConfig defaults vs Python spec
//   3. SpeechTokenizerDecoder forward pass (shape, clipping, determinism)
//   4. Chunked vs unchunked decode equivalence
//   5. Weight key sanitization (Python → Swift remapping)
//   6. Special token ID parity

@Suite("Golden Corpus E2E Parity")
struct GoldenCorpusE2EParityTests {

    // MARK: - Fixture Types

    struct GoldenFixture: Codable {
        let version: Int
        let chat_template_cases: [ChatTemplateCase]
        let decoder_config_parity: DecoderConfigParity
        let weight_key_remap_cases: [WeightKeyRemapCase]
        let small_decoder_cases: [SmallDecoderCase]
        let special_tokens_parity: SpecialTokensParity
    }

    struct ChatTemplateCase: Codable {
        let tag: String
        let input_text: String
        let expected_format: String
    }

    struct DecoderConfigParity: Codable {
        let codebook_dim: Int
        let codebook_size: Int
        let num_quantizers: Int
        let num_semantic_quantizers: Int
        let latent_dim: Int
        let decoder_dim: Int
        let upsample_rates: [Int]
        let upsampling_ratios: [Int]
        let hidden_size: Int
        let intermediate_size: Int
        let num_hidden_layers: Int
        let num_attention_heads: Int
        let num_key_value_heads: Int
        let head_dim: Int
        let rms_norm_eps: Float
        let rope_theta: Float
        let sliding_window: Int
        let layer_scale_initial_scale: Float
        let attention_bias: Bool
        let total_upsample: Int
        let output_dim: Int
    }

    struct WeightKeyRemapCase: Codable {
        let tag: String
        let python_key: String
        let swift_key: String
    }

    struct SmallDecoderCase: Codable {
        let tag: String
        let num_quantizers: Int
        let time_steps: Int
        let batch_size: Int
        let expected_output_channels: Int
        let expected_temporal_upsample: Int
    }

    struct SpecialTokensParity: Codable {
        let im_start: Int
        let im_end: Int
        let tts_pad: Int
        let tts_bos: Int
        let tts_eos: Int
    }

    // MARK: - Fixture Loading

    static func loadFixture() throws -> GoldenFixture {
        let fixtureURL = Bundle.module.url(
            forResource: "e2e_parity_golden",
            withExtension: "json"
        )!
        let data = try Data(contentsOf: fixtureURL)
        return try JSONDecoder().decode(GoldenFixture.self, from: data)
    }

    // MARK: - Small Decoder Factory

    /// Creates a small SpeechTokenizerDecoder for fast testing.
    ///
    /// Config: codebookDim=16, codebookSize=32, numQuantizers=2,
    /// latentDim=32, decoderDim=48, upsampleRates=[2,3],
    /// upsamplingRatios=[2] → totalUpsample=12, outputDim=12.
    static func makeSmallDecoder() -> SpeechTokenizerDecoder {
        let config = SpeechTokenizerDecoderConfig(
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
        return SpeechTokenizerDecoder(config: config)
    }

    /// Creates deterministic input codes for the small decoder.
    static func makeCodes(batch: Int, quantizers: Int, time: Int) -> MLXArray {
        let values = (0 ..< batch * quantizers * time).map { Int32($0 % 16) }
        return MLXArray(values).reshaped(batch, quantizers, time)
    }

    // MARK: - 1. Chat Template Parity

    @Test("Chat template format matches golden fixture for all cases")
    func chatTemplateFormat() throws {
        let fixture = try Self.loadFixture()
        let template = Qwen3ChatTemplate()

        for golden in fixture.chat_template_cases {
            let actual = template.format(golden.input_text)
            #expect(
                actual == golden.expected_format,
                "[\(golden.tag)] format mismatch"
            )
        }
    }

    @Test("Chat template contains exactly two im_start markers")
    func chatTemplateStructure() {
        let template = Qwen3ChatTemplate()
        let formatted = template.format("test text")

        let imStartCount = formatted.components(separatedBy: "<|im_start|>").count - 1
        let imEndCount = formatted.components(separatedBy: "<|im_end|>").count - 1

        #expect(imStartCount == 2, "Expected 2 <|im_start|> markers")
        #expect(imEndCount == 1, "Expected 1 <|im_end|> marker")
    }

    @Test("Chat template encode produces im_start as first token")
    func chatTemplateEncodeStructure() throws {
        let template = Qwen3ChatTemplate()
        let vocab = GoldenTokenizerTests.goldenVocab
        var encoder = BPEEncoder(vocabulary: vocab)

        let ids = try template.encode("hello", using: &encoder)

        #expect(!ids.isEmpty, "Token IDs should not be empty")
        #expect(ids.first == BPESpecialTokens.Qwen3TTS.imStart,
                "First token should be im_start (151644)")

        // im_end must appear exactly once
        let imEndCount = ids.filter { $0 == BPESpecialTokens.Qwen3TTS.imEnd }.count
        #expect(imEndCount == 1, "Expected exactly 1 im_end token")

        // im_start must appear exactly twice
        let imStartCount = ids.filter { $0 == BPESpecialTokens.Qwen3TTS.imStart }.count
        #expect(imStartCount == 2, "Expected exactly 2 im_start tokens")
    }

    // MARK: - 2. Decoder Config Parity

    @Test("Default SpeechTokenizerDecoderConfig matches Python Qwen3TTSTokenizerDecoderConfig")
    func decoderConfigParity() throws {
        let fixture = try Self.loadFixture()
        let golden = fixture.decoder_config_parity
        let config = SpeechTokenizerDecoderConfig.default

        #expect(config.codebookDim == golden.codebook_dim)
        #expect(config.codebookSize == golden.codebook_size)
        #expect(config.numQuantizers == golden.num_quantizers)
        #expect(config.numSemanticQuantizers == golden.num_semantic_quantizers)
        #expect(config.latentDim == golden.latent_dim)
        #expect(config.decoderDim == golden.decoder_dim)
        #expect(config.upsampleRates == golden.upsample_rates)
        #expect(config.upsamplingRatios == golden.upsampling_ratios)
        #expect(config.hiddenSize == golden.hidden_size)
        #expect(config.intermediateSize == golden.intermediate_size)
        #expect(config.numHiddenLayers == golden.num_hidden_layers)
        #expect(config.numAttentionHeads == golden.num_attention_heads)
        #expect(config.numKeyValueHeads == golden.num_key_value_heads)
        #expect(config.headDim == golden.head_dim)
        #expect(config.rmsNormEps == golden.rms_norm_eps)
        #expect(config.ropeTheta == golden.rope_theta)
        #expect(config.slidingWindow == golden.sliding_window)
        #expect(config.layerScaleInitialScale == golden.layer_scale_initial_scale)
        #expect(config.attentionBias == golden.attention_bias)
        #expect(config.totalUpsample == golden.total_upsample)
        #expect(config.outputDim == golden.output_dim)
    }

    // MARK: - 3. Decoder Pipeline Shape Parity

    @Test("SpeechTokenizerDecoder output shape matches golden cases")
    func decoderOutputShape() throws {
        let fixture = try Self.loadFixture()
        let decoder = Self.makeSmallDecoder()

        for golden in fixture.small_decoder_cases {
            let codes = Self.makeCodes(
                batch: golden.batch_size,
                quantizers: golden.num_quantizers,
                time: golden.time_steps
            )
            let output = decoder(codes)
            MLX.eval(output)

            let expectedSamples = golden.time_steps * golden.expected_temporal_upsample
            #expect(
                output.shape == [golden.batch_size, golden.expected_output_channels, expectedSamples],
                "[\(golden.tag)] shape mismatch: got \(output.shape)"
            )
        }
    }

    @Test("SpeechTokenizerDecoder output is clipped to [-1, 1]")
    func decoderOutputClipping() {
        let decoder = Self.makeSmallDecoder()
        let codes = Self.makeCodes(batch: 1, quantizers: 2, time: 4)

        let output = decoder(codes)
        MLX.eval(output)

        let flat = output.reshaped(-1).asArray(Float.self)
        let maxVal = flat.max() ?? 0
        let minVal = flat.min() ?? 0
        #expect(maxVal <= 1.0, "Output exceeds +1.0: \(maxVal)")
        #expect(minVal >= -1.0, "Output exceeds -1.0: \(minVal)")
    }

    @Test("SpeechTokenizerDecoder is deterministic")
    func decoderDeterminism() {
        let decoder = Self.makeSmallDecoder()
        let codes = Self.makeCodes(batch: 1, quantizers: 2, time: 4)

        let out1 = decoder(codes).reshaped(-1).asArray(Float.self)
        let out2 = decoder(codes).reshaped(-1).asArray(Float.self)

        #expect(out1 == out2, "Same input should produce identical output")
    }

    @Test("SpeechTokenizerDecoder output contains no NaN or Inf")
    func decoderOutputFinite() {
        let decoder = Self.makeSmallDecoder()
        let codes = Self.makeCodes(batch: 1, quantizers: 2, time: 4)

        let output = decoder(codes)
        MLX.eval(output)
        let flat = output.reshaped(-1).asArray(Float.self)

        #expect(!flat.contains { $0.isNaN }, "Output contains NaN")
        #expect(!flat.contains { $0.isInfinite }, "Output contains Inf")
    }

    @Test("SpeechTokenizerDecoder produces non-trivial output")
    func decoderOutputNonTrivial() {
        let decoder = Self.makeSmallDecoder()
        let codes = Self.makeCodes(batch: 1, quantizers: 2, time: 4)

        let output = decoder(codes)
        MLX.eval(output)
        let flat = output.reshaped(-1).asArray(Float.self)

        let hasNonZero = flat.contains { $0 != 0.0 }
        #expect(hasNonZero, "Output is all zeros")
    }

    // MARK: - 4. Chunked Decode Parity

    @Test("Chunked decode produces same total sample count as unchunked")
    func chunkedDecodeSampleCount() {
        let decoder = Self.makeSmallDecoder()
        let timeSteps = 8
        let codes = Self.makeCodes(batch: 1, quantizers: 2, time: timeSteps)

        let unchunked = decoder(codes)
        MLX.eval(unchunked)

        let chunked = decoder.chunkedDecode(codes, chunkSize: 4, leftContextSize: 1)
        MLX.eval(chunked)

        #expect(
            unchunked.dim(2) == chunked.dim(2),
            "Sample count mismatch: unchunked=\(unchunked.dim(2)), chunked=\(chunked.dim(2))"
        )
    }

    @Test("Chunked decode output shape matches [batch, 1, samples]")
    func chunkedDecodeShape() {
        let decoder = Self.makeSmallDecoder()
        let codes = Self.makeCodes(batch: 1, quantizers: 2, time: 8)

        let output = decoder.chunkedDecode(codes, chunkSize: 4, leftContextSize: 1)
        MLX.eval(output)

        #expect(output.ndim == 3, "Expected 3D output")
        #expect(output.dim(0) == 1, "Expected batch=1")
        #expect(output.dim(1) == 1, "Expected channels=1")
    }

    @Test("Chunked decode with chunk size >= sequence is equivalent to unchunked")
    func chunkedDecodeFullChunk() {
        let decoder = Self.makeSmallDecoder()
        let codes = Self.makeCodes(batch: 1, quantizers: 2, time: 4)

        let unchunked = decoder(codes).reshaped(-1).asArray(Float.self)
        let chunked = decoder.chunkedDecode(codes, chunkSize: 100, leftContextSize: 1)
            .reshaped(-1).asArray(Float.self)

        #expect(unchunked == chunked,
                "Full-chunk decode should be identical to unchunked")
    }

    // MARK: - 5. Weight Key Sanitization Parity

    @Test("Weight key remapping matches golden fixture for all cases")
    func weightKeySanitization() throws {
        let fixture = try Self.loadFixture()

        for golden in fixture.weight_key_remap_cases {
            let dummyValue = MLXArray.zeros([1])
            let input = [golden.python_key: dummyValue]
            let sanitized = SpeechTokenizerWeightLoader.sanitize(weights: input)

            #expect(
                sanitized.keys.contains(golden.swift_key),
                "[\(golden.tag)] expected key \"\(golden.swift_key)\", got \(Array(sanitized.keys))"
            )
        }
    }

    @Test("Sanitize is idempotent: applying twice produces same keys")
    func sanitizeIdempotent() {
        let dummyValue = MLXArray.zeros([1])
        let pythonKeys = [
            "decoder.quantizer.rvq_first.layers.0.codebook.embed.weight",
            "decoder.pre_conv.conv.weight",
            "decoder.pre_transformer.layers.0.self_attn.q_proj.weight",
        ]

        let input = Dictionary(uniqueKeysWithValues: pythonKeys.map { ($0, dummyValue) })
        let firstPass = SpeechTokenizerWeightLoader.sanitize(weights: input)
        let secondPass = SpeechTokenizerWeightLoader.sanitize(weights: firstPass)

        let firstKeys = Set(firstPass.keys).sorted()
        let secondKeys = Set(secondPass.keys).sorted()
        #expect(firstKeys == secondKeys, "Double sanitize changed keys")
    }

    @Test("decoderWeights strips decoder. prefix from all keys")
    func decoderWeightsPrefixStrip() {
        let dummyValue = MLXArray.zeros([1])
        let sanitized = [
            "decoder.quantizer.rvqFirst.layers.0.codebook.embed.weight": dummyValue,
            "decoder.preConv.conv.weight": dummyValue,
            "encoder.something.weight": dummyValue,
        ]

        let decoderOnly = SpeechTokenizerWeightLoader.decoderWeights(from: sanitized)

        #expect(decoderOnly.count == 2, "Expected 2 decoder keys")
        #expect(decoderOnly.keys.contains("quantizer.rvqFirst.layers.0.codebook.embed.weight"))
        #expect(decoderOnly.keys.contains("preConv.conv.weight"))
        #expect(!decoderOnly.keys.contains { $0.hasPrefix("decoder.") },
                "No key should retain decoder. prefix")
    }

    // MARK: - 6. Special Token Parity

    @Test("Special token IDs match Python Qwen3-TTS reference values")
    func specialTokenParity() throws {
        let fixture = try Self.loadFixture()
        let golden = fixture.special_tokens_parity

        #expect(BPESpecialTokens.Qwen3TTS.imStart == golden.im_start)
        #expect(BPESpecialTokens.Qwen3TTS.imEnd == golden.im_end)
        #expect(BPESpecialTokens.Qwen3TTS.ttsPad == golden.tts_pad)
        #expect(BPESpecialTokens.Qwen3TTS.ttsBos == golden.tts_bos)
        #expect(BPESpecialTokens.Qwen3TTS.ttsEos == golden.tts_eos)
    }

    @Test("Qwen3ChatTemplate exposes correct TTS control token IDs")
    func chatTemplateControlTokens() {
        let template = Qwen3ChatTemplate()

        #expect(template.ttsBosTokenID == 151672)
        #expect(template.ttsEosTokenID == 151673)
        #expect(template.ttsPadTokenID == 151671)
    }

    // MARK: - 7. Full Pipeline Smoke Test

    @Test("Full e2e pipeline: tokenize → format → decode → valid audio shape")
    func fullPipelineSmoke() throws {
        // Stage 1: Tokenization
        let template = Qwen3ChatTemplate()
        let vocab = GoldenTokenizerTests.goldenVocab
        var encoder = BPEEncoder(vocabulary: vocab)

        let tokenIDs = try template.encode("hello world", using: &encoder)
        #expect(!tokenIDs.isEmpty, "Tokenization produced empty output")

        // Verify structural integrity of token sequence
        #expect(tokenIDs.first == BPESpecialTokens.Qwen3TTS.imStart)
        let imEndIndex = tokenIDs.firstIndex(of: BPESpecialTokens.Qwen3TTS.imEnd)
        #expect(imEndIndex != nil, "Token sequence must contain im_end")

        // Stage 2: Decoder (with small config — simulates the code-predictor
        // output being fed to the speech tokenizer decoder)
        let decoder = Self.makeSmallDecoder()
        let timeSteps = 4
        let codes = Self.makeCodes(batch: 1, quantizers: 2, time: timeSteps)
        let audio = decoder(codes)
        MLX.eval(audio)

        // Stage 3: Output validation
        let expectedSamples = timeSteps * 12  // totalUpsample for small config
        #expect(audio.shape == [1, 1, expectedSamples])

        let samples = audio.reshaped(-1).asArray(Float.self)
        #expect(!samples.isEmpty)
        #expect(samples.allSatisfy { $0 >= -1.0 && $0 <= 1.0 },
                "All audio samples must be in [-1, 1]")
        #expect(!samples.allSatisfy { $0 == 0.0 },
                "Audio output should not be all zeros")
    }

    // MARK: - 8. Config Derived Properties

    @Test("Small config derived properties are self-consistent")
    func smallConfigConsistency() {
        let decoder = Self.makeSmallDecoder()
        let config = decoder.config

        // totalUpsample = product(upsampleRates) * product(upsamplingRatios)
        let expectedTotal = config.upsampleRates.reduce(1, *) * config.upsamplingRatios.reduce(1, *)
        #expect(config.totalUpsample == expectedTotal)

        // outputDim = decoderDim / 2^(numBlocks)
        let expectedOutputDim = config.decoderDim / (1 << config.upsampleRates.count)
        #expect(config.outputDim == expectedOutputDim)

        // Transformer config matches parent
        let tConfig = config.transformerConfig
        #expect(tConfig.latentDim == config.latentDim)
        #expect(tConfig.hiddenSize == config.hiddenSize)
        #expect(tConfig.numHiddenLayers == config.numHiddenLayers)
    }

    @Test("Default config totalUpsample matches Qwen3-TTS 24kHz sample rate")
    func defaultConfigSampleRate() {
        let config = SpeechTokenizerDecoderConfig.default
        // Qwen3-TTS operates at 12.5 Hz token rate:
        //   24000 Hz / 12.5 Hz = 1920 samples per token frame
        #expect(config.totalUpsample == 1920,
                "totalUpsample must be 1920 for 24kHz output at 12.5Hz token rate")
    }

    // MARK: - 9. Batch Invariance

    @Test("Single-item batch output matches first element of multi-item batch")
    func batchInvariance() {
        let decoder = Self.makeSmallDecoder()
        let time = 4

        // Single item
        let singleCodes = Self.makeCodes(batch: 1, quantizers: 2, time: time)
        let singleOut = decoder(singleCodes)
        MLX.eval(singleOut)

        // Same item repeated in batch of 2
        let batchCodes = concatenated([singleCodes, singleCodes], axis: 0)
        let batchOut = decoder(batchCodes)
        MLX.eval(batchOut)

        let singleSamples = singleOut.reshaped(-1).asArray(Float.self)
        let firstBatchSamples = batchOut[0].reshaped(-1).asArray(Float.self)

        #expect(singleSamples == firstBatchSamples,
                "Single-item output should match first element of batch output")
    }
}
