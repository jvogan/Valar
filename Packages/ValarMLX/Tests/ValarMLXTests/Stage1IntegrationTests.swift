import Foundation
@preconcurrency import MLX
import MLXNN
import Testing
@testable import ValarMLX

// MARK: - Stage 1 Integration Tests
//
// Stage 1 of the speech tokenizer decoder:
//   codes → VQ decode → pre_conv → transformer → upsample stages
//
// The CausalConvUpsampler wraps pre_conv + upsample, but the transformer
// sits between them in the actual pipeline, so these tests exercise the
// subcomponents in pipeline order.

@Suite("Stage 1 Integration")
struct Stage1IntegrationTests {

    // MARK: - Full Pipeline

    @Test("Full Stage 1 pipeline: VQ → preConv → transformer → upsample")
    func fullPipelineShapeAndDataFlow() {
        let (vq, upsampler, transformer) = makeStage1Components()
        let batch = 1
        let numQuantizers = 3
        let time = 8

        let codes = MLXArray(
            (0 ..< batch * numQuantizers * time).map { _ in Int32.random(in: 0 ..< 16) }
        ).reshaped(batch, numQuantizers, time)

        // Step 1: VQ decode — codes to continuous embedding
        let vqOut = vq.decode(codes)
        MLX.eval(vqOut)
        #expect(vqOut.shape == [batch, 8, time]) // [B, outputDim=8, T]

        // Step 2: pre_conv — project to latent dim
        let preConvOut = upsampler.preConv(vqOut)
        MLX.eval(preConvOut)
        #expect(preConvOut.shape == [batch, 16, time]) // [B, latentDim=16, T]

        // Step 3: Transpose → transformer → transpose
        var hidden = preConvOut.transposed(0, 2, 1) // NCL → NLC: [B, T, latentDim]
        #expect(hidden.shape == [batch, time, 16])

        hidden = transformer(hidden)
        MLX.eval(hidden)
        #expect(hidden.shape == [batch, time, 16])

        var convHidden = hidden.transposed(0, 2, 1) // NLC → NCL: [B, latentDim, T]
        #expect(convHidden.shape == [batch, 16, time])

        // Step 4: Upsample stages — 2×2 = 4× temporal upsampling
        for stage in upsampler.upsample {
            convHidden = stage(convHidden)
        }
        MLX.eval(convHidden)
        #expect(convHidden.shape == [batch, 16, time * 4]) // [B, latentDim, 4T]
    }

    @Test("Pipeline output is non-trivial (not all zeros)")
    func pipelineOutputNonTrivial() {
        let (vq, upsampler, transformer) = makeStage1Components()
        let codes = MLXArray(
            (0 ..< 24).map { Int32($0 % 16) }
        ).reshaped(1, 3, 8)

        let output = runStage1Pipeline(codes, vq: vq, upsampler: upsampler, transformer: transformer)
        let flat = output.reshaped(-1).asArray(Float.self)
        let hasNonZero = flat.contains { $0 != 0.0 }
        #expect(hasNonZero)
    }

    @Test("Pipeline is deterministic: same input produces same output")
    func pipelineDeterminism() {
        let (vq, upsampler, transformer) = makeStage1Components()
        let codes = MLXArray(
            (0 ..< 24).map { Int32($0 % 16) }
        ).reshaped(1, 3, 8)

        let result1 = runStage1Pipeline(codes, vq: vq, upsampler: upsampler, transformer: transformer)
            .reshaped(-1).asArray(Float.self)
        let result2 = runStage1Pipeline(codes, vq: vq, upsampler: upsampler, transformer: transformer)
            .reshaped(-1).asArray(Float.self)
        #expect(result1 == result2)
    }

    @Test("Pipeline handles batch size > 1")
    func pipelineBatched() {
        let (vq, upsampler, transformer) = makeStage1Components()
        let batch = 3
        let time = 6

        let codes = MLXArray(
            (0 ..< batch * 3 * time).map { _ in Int32.random(in: 0 ..< 16) }
        ).reshaped(batch, 3, time)

        let output = runStage1Pipeline(codes, vq: vq, upsampler: upsampler, transformer: transformer)
        #expect(output.shape == [batch, 16, time * 4])
    }

    // MARK: - Dimension Compatibility

    @Test("VQ output dimension matches upsampler input dimension")
    func vqUpsamplerDimensionMatch() {
        let (vq, upsampler, _) = makeStage1Components()

        let codes = MLXArray(Array(repeating: Int32(0), count: 24)).reshaped(1, 3, 8)
        let vqOut = vq.decode(codes)
        MLX.eval(vqOut)

        // VQ output channels must equal upsampler inputDim
        #expect(vqOut.dim(1) == upsampler.config.inputDim)

        // Verify pre_conv accepts the VQ output
        let preConvOut = upsampler.preConv(vqOut)
        MLX.eval(preConvOut)
        #expect(preConvOut.dim(1) == upsampler.config.latentDim)
    }

    @Test("Upsampler latent dimension matches transformer latent dimension")
    func upsamplerTransformerDimensionMatch() {
        let (_, upsampler, transformer) = makeStage1Components()
        #expect(upsampler.config.latentDim == transformer.config.latentDim)
    }

    // MARK: - Temporal Upsampling Factor

    @Test("Total temporal upsampling factor is product of ratios")
    func temporalUpsamplingFactor() {
        let (vq, upsampler, transformer) = makeStage1Components()
        let time = 10

        let codes = MLXArray(Array(repeating: Int32(0), count: 3 * time)).reshaped(1, 3, time)
        let output = runStage1Pipeline(codes, vq: vq, upsampler: upsampler, transformer: transformer)

        let expectedFactor = upsampler.config.upsamplingRatios.reduce(1, *)
        #expect(output.dim(2) == time * expectedFactor)
    }

    // MARK: - Semantic-Only Codes Path

    @Test("VQ decode works with semantic-only codes (nQSemantic quantizers)")
    func semanticOnlyCodes() {
        let vq = SplitResidualVectorQuantizer(
            nQ: 3, nQSemantic: 1,
            dimension: 4, inputDimension: 8,
            outputDimension: 8, bins: 16
        )

        // Only 1 quantizer (semantic only) — should still produce valid output
        let codes = MLXArray(Array(repeating: Int32(0), count: 8)).reshaped(1, 1, 8)
        let result = vq.decode(codes)
        MLX.eval(result)
        #expect(result.shape == [1, 8, 8])
    }

    // MARK: - Causal Property Through Pipeline

    @Test("Pipeline preserves causality: future codes don't affect past output")
    func pipelineCausality() {
        let (vq, upsampler, transformer) = makeStage1Components()

        // Shared prefix codes for first 4 time steps
        let sharedCodes: [Int32] = (0 ..< 12).map { Int32($0 % 16) }
        // Different suffix codes for last 4 time steps
        let suffix1: [Int32] = Array(repeating: Int32(1), count: 12)
        let suffix2: [Int32] = Array(repeating: Int32(15), count: 12)

        let codes1 = MLXArray(sharedCodes + suffix1).reshaped(1, 3, 8)
        let codes2 = MLXArray(sharedCodes + suffix2).reshaped(1, 3, 8)

        let out1 = runStage1Pipeline(codes1, vq: vq, upsampler: upsampler, transformer: transformer)
        let out2 = runStage1Pipeline(codes2, vq: vq, upsampler: upsampler, transformer: transformer)

        // Causal boundary: pre_conv kernel=3 means shared output at times 0..3.
        // After 2x2 upsample, shared region covers output times 0..15.
        // The transformer's sliding window only looks backward, preserving causality.
        // Check the first few output time steps are identical.
        let earlySteps = 8 // conservative: first 2 input steps x 4x upsample
        let early1 = out1[0..., 0..., ..<earlySteps].reshaped(-1).asArray(Float.self)
        let early2 = out2[0..., 0..., ..<earlySteps].reshaped(-1).asArray(Float.self)
        #expect(early1 == early2)

        // The later steps must differ since codes differ
        let late1 = out1[0..., 0..., (earlySteps)...].reshaped(-1).asArray(Float.self)
        let late2 = out2[0..., 0..., (earlySteps)...].reshaped(-1).asArray(Float.self)
        #expect(late1 != late2)
    }

    // MARK: - No NaN/Inf

    @Test("Pipeline output contains no NaN or Inf values")
    func noNanOrInf() {
        let (vq, upsampler, transformer) = makeStage1Components()
        let codes = MLXArray(
            (0 ..< 24).map { Int32($0 % 16) }
        ).reshaped(1, 3, 8)

        let output = runStage1Pipeline(codes, vq: vq, upsampler: upsampler, transformer: transformer)
        let flat = output.reshaped(-1).asArray(Float.self)
        let hasNan = flat.contains { $0.isNaN }
        let hasInf = flat.contains { $0.isInfinite }
        #expect(!hasNan, "Pipeline output contains NaN values")
        #expect(!hasInf, "Pipeline output contains Inf values")
    }

    // MARK: - Helpers

    /// Run the full Stage 1 pipeline: VQ → preConv → transformer → upsample.
    private func runStage1Pipeline(
        _ codes: MLXArray,
        vq: SplitResidualVectorQuantizer,
        upsampler: CausalConvUpsampler,
        transformer: DecoderTransformer
    ) -> MLXArray {
        // VQ decode
        let vqOut = vq.decode(codes)
        // pre_conv
        let preConvOut = upsampler.preConv(vqOut)
        // Transpose → transformer → transpose
        var hidden = transformer(preConvOut.transposed(0, 2, 1))
        hidden = hidden.transposed(0, 2, 1)
        // Upsample stages
        for stage in upsampler.upsample {
            hidden = stage(hidden)
        }
        MLX.eval(hidden)
        return hidden
    }

    /// Create matched Stage 1 components with small dimensions for fast testing.
    ///
    /// Dimensions: VQ outputDim=8 -> upsampler inputDim=8, latentDim=16 ->
    /// transformer latentDim=16 with hiddenSize=8.
    private func makeStage1Components() -> (
        SplitResidualVectorQuantizer,
        CausalConvUpsampler,
        DecoderTransformer
    ) {
        let vq = SplitResidualVectorQuantizer(
            nQ: 3, nQSemantic: 1,
            dimension: 4, inputDimension: 8,
            outputDimension: 8, bins: 16
        )

        let upsamplerConfig = CausalConvUpsamplerConfig(
            inputDim: 8, latentDim: 16,
            upsamplingRatios: [2, 2],
            convNeXtKernelSize: 3
        )
        let upsampler = CausalConvUpsampler(config: upsamplerConfig)

        let transformerConfig = DecoderTransformerConfig(
            hiddenSize: 8,
            intermediateSize: 16,
            numHiddenLayers: 2,
            numAttentionHeads: 2,
            numKeyValueHeads: 2,
            headDim: 4,
            rmsNormEps: 1e-5,
            ropeTheta: 10000.0,
            slidingWindow: 8,
            layerScaleInitialScale: 0.01,
            latentDim: 16,
            attentionBias: false
        )
        let transformer = DecoderTransformer(config: transformerConfig)

        return (vq, upsampler, transformer)
    }
}
