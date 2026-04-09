import Foundation
@preconcurrency import MLX
import MLXNN
import Testing
@testable import ValarMLX

@Suite("CausalConvUpsampler")
struct CausalConvUpsamplerTests {

    @Test("CausalTransposeConv1d doubles temporal dimension with stride-2")
    func transposeConvDoubles() {
        let conv = CausalTransposeConv1d(
            inChannels: 4, outChannels: 4, kernelSize: 2, stride: 2
        )
        let input = MLXArray.ones([1, 4, 8]) // [B=1, C=4, T=8]
        let output = conv(input)
        MLX.eval(output)
        #expect(output.shape == [1, 4, 16]) // T doubled
    }

    @Test("CausalTransposeConv1d trims right when kernel > stride")
    func transposeConvTrims() {
        let conv = CausalTransposeConv1d(
            inChannels: 4, outChannels: 4, kernelSize: 4, stride: 2
        )
        let input = MLXArray.ones([1, 4, 8])
        let output = conv(input)
        MLX.eval(output)
        // Raw output: (8-1)*2+4 = 18, trim 2 -> 16
        #expect(output.shape == [1, 4, 16])
    }

    @Test("CausalConv1d preserves temporal dimension with stride-1")
    func causalConvPreservesTime() {
        let conv = CausalConv1d(inChannels: 4, outChannels: 8, kernelSize: 3)
        let input = MLXArray.ones([1, 4, 10])
        let output = conv(input)
        MLX.eval(output)
        #expect(output.shape == [1, 8, 10])
    }

    @Test("CausalConv1d depthwise preserves shape")
    func causalConvDepthwise() {
        let conv = CausalConv1d(
            inChannels: 8, outChannels: 8, kernelSize: 7, groups: 8
        )
        let input = MLXArray.ones([1, 8, 16])
        let output = conv(input)
        MLX.eval(output)
        #expect(output.shape == [1, 8, 16])
    }

    @Test("ConvNeXtBlock preserves shape")
    func convNextPreservesShape() {
        let block = ConvNeXtBlock(dim: 8, kernelSize: 3)
        let input = MLXArray.ones([1, 8, 16])
        let output = block(input)
        MLX.eval(output)
        #expect(output.shape == [1, 8, 16])
    }

    @Test("CausalConvUpsampler produces [B,1024,4T] from [B,512,T]")
    func upsamplerOutputShape() {
        let upsampler = CausalConvUpsampler()
        let input = MLXArray.zeros([1, 512, 8])
        let output = upsampler(input)
        MLX.eval(output)
        #expect(output.shape == [1, 1024, 32])
    }

    @Test("Upsampler is causal: future input changes don't affect past outputs")
    func causalProperty() {
        // Use small dimensions for speed
        let config = CausalConvUpsamplerConfig(
            inputDim: 4, latentDim: 8,
            upsamplingRatios: [2, 2],
            convNeXtKernelSize: 3
        )
        let upsampler = CausalConvUpsampler(config: config)

        // Two inputs sharing the first 5 time steps, differing at steps 5-9
        let sharedValues = (0 ..< 20).map { Float($0 + 1) * 0.1 }
        let shared = MLXArray(sharedValues).reshaped(1, 4, 5)

        let tailValues1 = (0 ..< 20).map { Float($0 + 1) * 0.2 }
        let tail1 = MLXArray(tailValues1).reshaped(1, 4, 5)

        let tailValues2 = (0 ..< 20).map { Float($0 + 1) * 5.0 }
        let tail2 = MLXArray(tailValues2).reshaped(1, 4, 5)

        let input1 = concatenated([shared, tail1], axis: 2) // [1, 4, 10]
        let input2 = concatenated([shared, tail2], axis: 2) // [1, 4, 10]
        MLX.eval(input1, input2)

        let output1 = upsampler(input1)
        let output2 = upsampler(input2)
        MLX.eval(output1, output2)

        // Output shape: [1, 8, 40] (10 * 4x upsampling)
        #expect(output1.shape == [1, 8, 40])

        // Causal boundary analysis:
        // - pre_conv (kernel=3, stride=1): shared output at times 0-4
        // - CausalTransposeConv1d (stride=2): shared output at times 0-9
        // - ConvNeXtBlock (kernel=3): shared output at times 0-9
        // - CausalTransposeConv1d (stride=2): shared output at times 0-19
        // - ConvNeXtBlock (kernel=3): shared output at times 0-19
        // First 20 output time steps should be identical.
        let parts1 = split(output1, indices: [20], axis: 2)
        let parts2 = split(output2, indices: [20], axis: 2)

        // Early outputs must be identical (causal property)
        let earlyFlat1 = parts1[0].reshaped(-1).asArray(Float.self)
        let earlyFlat2 = parts2[0].reshaped(-1).asArray(Float.self)
        #expect(earlyFlat1 == earlyFlat2)

        // Later outputs must differ (verify the modification has effect)
        let lateFlat1 = parts1[1].reshaped(-1).asArray(Float.self)
        let lateFlat2 = parts2[1].reshaped(-1).asArray(Float.self)
        #expect(lateFlat1 != lateFlat2)
    }
}
