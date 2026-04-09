import Foundation
@preconcurrency import MLX
import MLXNN
import MLXAudioCodecs

public struct TADAExpandedTokenFrames {
    public let values: MLXArray
    public let frameMask: MLXArray

    public init(values: MLXArray, frameMask: MLXArray) {
        self.values = values
        self.frameMask = frameMask
    }
}

public final class TADADACDecoder: Module {
    @ModuleInfo(key: "conv") public var conv: DescriptWNConv1d
    @ModuleInfo(key: "blocks") public var blocks: [DescriptDecoderBlock]
    @ModuleInfo(key: "final_snake") public var finalSnake: DescriptSnake1d
    @ModuleInfo(key: "out") public var out: DescriptWNConv1d

    public let inputChannels: Int
    public let channels: Int
    public let rates: [Int]

    public init(
        inputChannels: Int = 1024,
        channels: Int = 1536,
        rates: [Int] = [4, 4, 5, 6]
    ) {
        self.inputChannels = inputChannels
        self.channels = channels
        self.rates = rates

        _conv.wrappedValue = DescriptWNConv1d(
            inChannels: inputChannels,
            outChannels: channels,
            kernelSize: 7,
            padding: 3
        )

        var decoderBlocks: [DescriptDecoderBlock] = []
        for (index, stride) in rates.enumerated() {
            let inputDim = channels / Int(pow(2.0, Double(index)))
            let outputDim = channels / Int(pow(2.0, Double(index + 1)))
            decoderBlocks.append(
                DescriptDecoderBlock(
                    inputDim: inputDim,
                    outputDim: outputDim,
                    stride: stride
                )
            )
        }
        _blocks.wrappedValue = decoderBlocks

        let finalChannels = channels / Int(pow(2.0, Double(rates.count)))
        _finalSnake.wrappedValue = DescriptSnake1d(channels: finalChannels)
        _out.wrappedValue = DescriptWNConv1d(
            inChannels: finalChannels,
            outChannels: 1,
            kernelSize: 7,
            padding: 3
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var hidden = x.transposed(0, 2, 1)
        hidden = conv(hidden)
        for block in blocks {
            hidden = block(hidden)
        }
        hidden = finalSnake(hidden)
        hidden = out(hidden)
        return tanh(hidden).transposed(0, 2, 1)
    }
}

public final class TADATTSDecoder: Module {
    @ModuleInfo(key: "decoder_proj") public var decoderProj: Linear
    // key matches Python: decoder/weights.safetensors → local_attention_decoder.*
    @ModuleInfo(key: "local_attention_decoder") public var localAttentionDecoder: TADAv2AttentionStack
    @ModuleInfo(key: "wav_decoder") public var wavDecoder: TADADACDecoder

    public init(
        acousticDim: Int = 512,
        hiddenDim: Int = 1024,
        numLayers: Int = 6,
        numHeads: Int = 8,
        feedForwardSize: Int = 4096,
        decoderChannels: Int = 1536,
        strides: [Int] = [4, 4, 5, 6]
    ) {
        _decoderProj.wrappedValue = Linear(acousticDim, hiddenDim, bias: false)
        _localAttentionDecoder.wrappedValue = TADAv2AttentionStack(
            hiddenSize: hiddenDim,
            numLayers: numLayers,
            numHeads: numHeads,
            feedForwardSize: feedForwardSize
        )
        _wavDecoder.wrappedValue = TADADACDecoder(
            inputChannels: hiddenDim,
            channels: decoderChannels,
            rates: strides
        )
    }

    public func decodeTokenFeatures(
        _ tokenValues: MLXArray,
        tokenMask: MLXArray? = nil,
        attentionMask: MLXArray? = nil
    ) -> MLXArray {
        let projected = decoderProj(tokenValues)
        let mask = tokenMask.map { tadaDecoderSegmentMask(tokenMask: $0) }
        return localAttentionDecoder(projected, mask: mask)
    }

    public func expandTokenFeatures(
        _ tokenValues: MLXArray,
        timeBefore: MLXArray,
        timeAfter: MLXArray? = nil
    ) -> TADAExpandedTokenFrames {
        let batch = tokenValues.dim(0)
        let acousticDim = tokenValues.dim(2)
        let beforeClipped = clip(timeBefore, min: 0, max: 255).asType(.int32)
        let afterClipped = timeAfter.map { clip($0, min: 0, max: 255).asType(.int32) }

        var batchFrames: [MLXArray] = []
        var batchMasks: [MLXArray] = []
        var maxFrames = 0

        for batchIndex in 0..<batch {
            let beforeValues = beforeClipped[batchIndex].asArray(Int32.self).map(Int.init)
            let afterValues = afterClipped?[batchIndex].asArray(Int32.self).map(Int.init)
                ?? Array(repeating: 0, count: beforeValues.count)

            var pieces: [MLXArray] = []
            var masks: [MLXArray] = []

            for tokenIndex in 0..<beforeValues.count {
                let leadingZeros = max(beforeValues[tokenIndex] - 1, 0)
                if leadingZeros > 0 {
                    pieces.append(MLXArray.zeros([leadingZeros, acousticDim], dtype: tokenValues.dtype))
                    masks.append(MLXArray.zeros([leadingZeros], dtype: .float32))
                }

                pieces.append(tokenValues[batchIndex, tokenIndex, 0...].expandedDimensions(axis: 0))
                masks.append(MLXArray.ones([1], dtype: .float32))

                let trailingZeros = max(afterValues[tokenIndex], 0)
                if trailingZeros > 0 {
                    pieces.append(MLXArray.zeros([trailingZeros, acousticDim], dtype: tokenValues.dtype))
                    masks.append(MLXArray.zeros([trailingZeros], dtype: .float32))
                }
            }

            let expandedValues = pieces.isEmpty
                ? MLXArray.zeros([1, acousticDim], dtype: tokenValues.dtype)
                : concatenated(pieces, axis: 0)
            let expandedMask = masks.isEmpty
                ? MLXArray.zeros([1], dtype: .float32)
                : concatenated(masks, axis: 0)

            maxFrames = max(maxFrames, expandedValues.dim(0))
            batchFrames.append(expandedValues)
            batchMasks.append(expandedMask)
        }

        var paddedValues: [MLXArray] = []
        var paddedMasks: [MLXArray] = []

        for index in 0..<batchFrames.count {
            let values = batchFrames[index]
            let mask = batchMasks[index]
            let padding = maxFrames - values.dim(0)
            if padding > 0 {
                paddedValues.append(
                    MLX.padded(
                        values,
                        widths: [IntOrPair((0, padding)), IntOrPair(0)]
                    )
                )
                paddedMasks.append(
                    MLX.padded(mask, widths: [IntOrPair((0, padding))])
                )
            } else {
                paddedValues.append(values)
                paddedMasks.append(mask)
            }
        }

        return TADAExpandedTokenFrames(
            values: stacked(paddedValues, axis: 0),
            frameMask: stacked(paddedMasks, axis: 0)
        )
    }

    public func callAsFunction(
        _ tokenValues: MLXArray,
        timeBefore: MLXArray,
        timeAfter: MLXArray? = nil,
        tokenMask: MLXArray? = nil,
        attentionMask: MLXArray? = nil
    ) -> MLXArray {
        let decodedTokens = decodeTokenFeatures(
            tokenValues,
            tokenMask: tokenMask,
            attentionMask: attentionMask
        )
        let expanded = expandTokenFeatures(
            decodedTokens,
            timeBefore: timeBefore,
            timeAfter: timeAfter
        )
        return wavDecoder(expanded.values)
    }

    public static func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        tadaSanitizeModuleWeights(weights, prefix: "decoder")
    }
}
