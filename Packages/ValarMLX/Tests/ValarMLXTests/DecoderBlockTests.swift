import Foundation
@preconcurrency import MLX
import MLXNN
import Testing
@testable import ValarMLX

@Suite("DecoderBlock")
struct DecoderBlockTests {

    // MARK: - DecoderResidualUnit

    @Test("DecoderResidualUnit preserves shape with dilation=1")
    func residualUnitPreservesShapeDilation1() {
        let unit = DecoderResidualUnit(dim: 16, dilation: 1)
        let input = MLXArray.ones([1, 16, 32]) // [B, C, T]
        let output = unit(input)
        MLX.eval(output)
        #expect(output.shape == [1, 16, 32])
    }

    @Test("DecoderResidualUnit preserves shape with dilation=3")
    func residualUnitPreservesShapeDilation3() {
        let unit = DecoderResidualUnit(dim: 8, dilation: 3)
        let input = MLXArray.ones([1, 8, 20])
        let output = unit(input)
        MLX.eval(output)
        #expect(output.shape == [1, 8, 20])
    }

    @Test("DecoderResidualUnit preserves shape with dilation=9")
    func residualUnitPreservesShapeDilation9() {
        let unit = DecoderResidualUnit(dim: 8, dilation: 9)
        let input = MLXArray.ones([1, 8, 20])
        let output = unit(input)
        MLX.eval(output)
        #expect(output.shape == [1, 8, 20])
    }

    @Test("DecoderResidualUnit residual connection: output is finite for zero input")
    func residualUnitPassthrough() {
        let unit = DecoderResidualUnit(dim: 4, dilation: 1)
        let input = MLXArray.zeros([1, 4, 8])
        let output = unit(input)
        MLX.eval(output)
        let flat = output.reshaped(-1).asArray(Float.self)
        for val in flat {
            #expect(val.isFinite)
        }
    }

    // MARK: - DecoderBlockUpsample

    @Test("DecoderBlockUpsample with rate=2 doubles time, halves channels")
    func upsampleRate2() {
        let up = DecoderBlockUpsample(inDim: 16, outDim: 8, upsampleRate: 2)
        let input = MLXArray.ones([1, 16, 10])
        let output = up(input)
        MLX.eval(output)
        #expect(output.shape == [1, 8, 20])
    }

    @Test("DecoderBlockUpsample with rate=4 quadruples time")
    func upsampleRate4() {
        let up = DecoderBlockUpsample(inDim: 32, outDim: 16, upsampleRate: 4)
        let input = MLXArray.ones([1, 32, 8])
        let output = up(input)
        MLX.eval(output)
        #expect(output.shape == [1, 16, 32])
    }

    @Test("DecoderBlockUpsample with rate=8 upsamples 8x")
    func upsampleRate8() {
        let up = DecoderBlockUpsample(inDim: 64, outDim: 32, upsampleRate: 8)
        let input = MLXArray.ones([1, 64, 4])
        let output = up(input)
        MLX.eval(output)
        #expect(output.shape == [1, 32, 32])
    }

    // MARK: - DecoderBlock (full)

    @Test("DecoderBlock output shape with rate=2")
    func decoderBlockShapeRate2() {
        let block = DecoderBlock(inDim: 16, outDim: 8, upsampleRate: 2)
        let input = MLXArray.zeros([1, 16, 10])
        let output = block(input)
        MLX.eval(output)
        #expect(output.shape == [1, 8, 20])
    }

    @Test("DecoderBlock output shape with rate=8")
    func decoderBlockShapeRate8() {
        let block = DecoderBlock(inDim: 32, outDim: 16, upsampleRate: 8)
        let input = MLXArray.zeros([1, 32, 4])
        let output = block(input)
        MLX.eval(output)
        #expect(output.shape == [1, 16, 32])
    }

    @Test("DecoderBlock convenience init from config matches manual init")
    func decoderBlockFromConfig() throws {
        let config = DecoderBlockConfig(decoderDim: 32, upsampleRates: [2, 2])
        // Layer 0: inDim=32, outDim=16, rate=2
        let block = try DecoderBlock(config: config, layerIndex: 0)
        let input = MLXArray.zeros([1, 32, 8])
        let output = block(input)
        MLX.eval(output)
        #expect(output.shape == [1, 16, 16])
    }

    @Test("DecoderBlock chain produces expected shape for 2-layer stack")
    func decoderBlockChain() throws {
        let config = DecoderBlockConfig(decoderDim: 32, upsampleRates: [2, 4])
        let block0 = try DecoderBlock(config: config, layerIndex: 0)
        let block1 = try DecoderBlock(config: config, layerIndex: 1)

        let input = MLXArray.zeros([1, 32, 4])
        var h = block0(input)
        MLX.eval(h)
        #expect(h.shape == [1, 16, 8]) // 32->16 channels, 4->8 time

        h = block1(h)
        MLX.eval(h)
        #expect(h.shape == [1, 8, 32]) // 16->8 channels, 8->32 time
    }

    @Test("DecoderBlock convenience init throws for invalid layer index")
    func decoderBlockInvalidLayerIndex() {
        let config = DecoderBlockConfig(decoderDim: 32, upsampleRates: [2, 4])

        do {
            _ = try DecoderBlock(config: config, layerIndex: 2)
            Issue.record("Expected invalid layer index to throw")
        } catch let error as DecoderBlockConfigError {
            #expect(
                error == .invalidLayerIndex(
                    layerIndex: 2,
                    validIndices: config.upsampleRates.indices
                )
            )
        } catch {
            Issue.record("Unexpected error: \(error)")
        }
    }

    @Test("DecoderBlock output is finite for random input")
    func decoderBlockFiniteOutput() {
        let block = DecoderBlock(inDim: 8, outDim: 4, upsampleRate: 2)
        let input = MLXRandom.normal([2, 8, 6]) * 0.1
        let output = block(input)
        MLX.eval(output)
        let flat = output.reshaped(-1).asArray(Float.self)
        for val in flat {
            #expect(val.isFinite)
        }
    }

    @Test("DecoderBlock is causal: future input changes don't affect past outputs")
    func causalProperty() {
        let block = DecoderBlock(inDim: 4, outDim: 2, upsampleRate: 2)

        // Two inputs sharing the first 4 time steps, differing at steps 4-7
        let shared = MLXArray(Array(stride(from: Float(0.1), through: Float(1.6), by: Float(0.1))))
            .reshaped(1, 4, 4)
        let tail1 = MLXArray(Array(stride(from: Float(0.2), through: Float(3.2), by: Float(0.2))))
            .reshaped(1, 4, 4)
        let tail2 = MLXArray(Array(stride(from: Float(5.0), through: Float(80.0), by: Float(5.0))))
            .reshaped(1, 4, 4)

        let input1 = concatenated([shared, tail1], axis: 2) // [1, 4, 8]
        let input2 = concatenated([shared, tail2], axis: 2) // [1, 4, 8]
        MLX.eval(input1, input2)

        let output1 = block(input1) // [1, 2, 16]
        let output2 = block(input2)
        MLX.eval(output1, output2)

        #expect(output1.shape == [1, 2, 16])

        // Due to causal convolutions, early time steps should be identical
        // After rate-2 upsample, shared 4 time steps -> 8 output steps
        // Residual units with k=7 left-pad still only see past
        let early1 = output1[0..., 0..., 0 ..< 8].reshaped(-1).asArray(Float.self)
        let early2 = output2[0..., 0..., 0 ..< 8].reshaped(-1).asArray(Float.self)
        #expect(early1 == early2)

        // Later steps should differ
        let late1 = output1[0..., 0..., 8...].reshaped(-1).asArray(Float.self)
        let late2 = output2[0..., 0..., 8...].reshaped(-1).asArray(Float.self)
        #expect(late1 != late2)
    }
}
