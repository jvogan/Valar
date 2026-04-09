import Foundation
@preconcurrency import MLX
import MLXNN

public enum TADAGrayCodeDurationCodec {
    public static let numTimeClasses = 256
    public static let numTimeBits = 8

    public static func encode(_ values: MLXArray, numBits: Int = numTimeBits) -> MLXArray {
        let clipped = clip(values, min: 0, max: (1 << numBits) - 1).asType(.int32)
        let flatValues = clipped.asArray(Int32.self)
        var encoded: [Float] = []
        encoded.reserveCapacity(flatValues.count * numBits)

        for value in flatValues {
            let gray = Int(value) ^ (Int(value) >> 1)
            for bit in stride(from: numBits - 1, through: 0, by: -1) {
                encoded.append(Float((gray >> bit) & 1))
            }
        }

        return MLXArray(encoded).reshaped(clipped.shape + [numBits])
    }

    public static func decode(_ bits: MLXArray) -> MLXArray {
        let hardBits = (bits .> MLXArray(Float(0))).asType(.int32)
        let bitCount = hardBits.dim(-1)
        let valueCount = Array(hardBits.shape.dropLast()).reduce(1, *)
        let flatBits = hardBits.reshaped([valueCount, bitCount]).asArray(Int32.self)
        var decoded: [Int32] = []
        decoded.reserveCapacity(valueCount)

        for valueIndex in 0..<valueCount {
            var integer = 0
            var previousBit = 0
            for bitIndex in 0..<bitCount {
                let grayBit = Int(flatBits[(valueIndex * bitCount) + bitIndex])
                if bitIndex == 0 {
                    previousBit = grayBit
                } else {
                    previousBit ^= grayBit
                }
                integer = (integer << 1) | previousBit
            }
            decoded.append(Int32(integer))
        }

        return MLXArray(decoded).reshaped(Array(hardBits.shape.dropLast()))
    }
}

public struct TADAVibeVoiceSample {
    public let acoustic: MLXArray
    public let timeBefore: MLXArray
    public let timeAfter: MLXArray
    public let latent: MLXArray

    public init(acoustic: MLXArray, timeBefore: MLXArray, timeAfter: MLXArray, latent: MLXArray) {
        self.acoustic = acoustic
        self.timeBefore = timeBefore
        self.timeAfter = timeAfter
        self.latent = latent
    }
}

public enum TADACFGSchedule: String, Sendable {
    case constant
    case cosine
}

public final class TADATimestepEmbedder: Module {
    @ModuleInfo(key: "mlp") public var mlp: [Module]
    public let hiddenSize: Int
    public let frequencySize: Int

    public init(hiddenSize: Int, frequencySize: Int = 256) {
        self.hiddenSize = hiddenSize
        self.frequencySize = frequencySize
        _mlp.wrappedValue = [
            Linear(frequencySize, hiddenSize),
            SiLU(),
            Linear(hiddenSize, hiddenSize)
        ]
    }

    public func callAsFunction(_ timesteps: MLXArray) -> MLXArray {
        let embedding = sinusoidalEmbedding(timesteps, dimension: frequencySize)
        var hidden = embedding
        for layer in mlp {
            hidden = (layer as! UnaryLayer).callAsFunction(hidden)
        }
        return hidden
    }

    private func sinusoidalEmbedding(_ timesteps: MLXArray, dimension: Int) -> MLXArray {
        var t = timesteps
        if t.ndim == 0 {
            t = t.expandedDimensions(axis: 0)
        }
        if t.ndim == 1 {
            t = t.expandedDimensions(axis: -1)
        }

        let halfDim = dimension / 2
        let exponent = -log(Float(10_000)) / Float(max(halfDim - 1, 1))
        let frequencies = MLX.exp(MLXArray(0..<halfDim).asType(.float32) * MLXArray(exponent))
        let args = t.asType(.float32) * frequencies.expandedDimensions(axis: 0)
        return concatenated([MLX.sin(args), MLX.cos(args)], axis: -1)
    }
}

public final class TADAVibeVoiceHeadLayer: Module {
    @ModuleInfo(key: "norm") public var norm: RMSNorm
    @ModuleInfo(key: "adaLN_modulation") public var adaLNModulation: Linear
    @ModuleInfo(key: "ffn") public var ffn: TADALocalAttentionFeedForward

    public init(hiddenSize: Int, feedForwardSize: Int) {
        _norm.wrappedValue = RMSNorm(dimensions: hiddenSize, eps: 1e-5)
        _adaLNModulation.wrappedValue = Linear(hiddenSize, 3 * hiddenSize)
        _ffn.wrappedValue = TADALocalAttentionFeedForward(
            hiddenSize: hiddenSize,
            intermediateSize: feedForwardSize
        )
    }

    public func callAsFunction(_ x: MLXArray, conditioning: MLXArray) -> MLXArray {
        let modulation = adaLNModulation(conditioning)
        let parts = split(modulation, indices: [x.dim(-1), 2 * x.dim(-1)], axis: -1)
        let shift = parts[0]
        let scale = parts[1]
        let gate = parts[2]

        let normalized = norm(x)
        let modulated = normalized * (MLXArray(Float(1)) + scale) + shift
        return x + gate * ffn(modulated)
    }
}

public final class TADAVibeVoiceFinalLayer: Module {
    @ModuleInfo(key: "norm_final") public var normFinal: RMSNorm
    @ModuleInfo(key: "adaLN_modulation") public var adaLNModulation: Linear
    @ModuleInfo(key: "linear") public var linear: Linear

    public init(hiddenSize: Int, outputSize: Int) {
        _normFinal.wrappedValue = RMSNorm(dimensions: hiddenSize, eps: 1e-5)
        _adaLNModulation.wrappedValue = Linear(hiddenSize, 2 * hiddenSize)
        _linear.wrappedValue = Linear(hiddenSize, outputSize)
    }

    public func callAsFunction(_ x: MLXArray, conditioning: MLXArray) -> MLXArray {
        let modulation = adaLNModulation(conditioning)
        let parts = split(modulation, indices: [x.dim(-1)], axis: -1)
        let shift = parts[0]
        let scale = parts[1]
        let modulated = normFinal(x) * (MLXArray(Float(1)) + scale) + shift
        return linear(modulated)
    }
}

public final class TADAVibeVoice: Module {
    public static let defaultFlowSteps = 20
    public static let defaultNoiseTemperature: Float = 0.9
    public static let defaultAcousticCFG: Float = 1.6
    public static let defaultDurationCFG: Float = 1.0

    public let acousticDim: Int
    public let numTimeBits: Int
    public let latentSize: Int
    public let acousticMean: Float
    public let acousticStd: Float

    @ModuleInfo(key: "noisy_images_proj") public var noisyImagesProj: Linear
    @ModuleInfo(key: "cond_proj") public var condProj: Linear
    @ModuleInfo(key: "t_embedder") public var tEmbedder: TADATimestepEmbedder
    @ModuleInfo(key: "layers") public var layers: [TADAVibeVoiceHeadLayer]
    @ModuleInfo(key: "final_layer") public var finalLayer: TADAVibeVoiceFinalLayer

    public init(
        hiddenSize: Int = 3072,
        acousticDim: Int = 512,
        numTimeBits: Int = TADAGrayCodeDurationCodec.numTimeBits,
        headLayers: Int = 6,
        headFFNRatio: Float = 4.0,
        acousticMean: Float = 0,
        acousticStd: Float = 1.5
    ) {
        self.acousticDim = acousticDim
        self.numTimeBits = numTimeBits
        self.latentSize = acousticDim + (2 * numTimeBits)
        self.acousticMean = acousticMean
        self.acousticStd = acousticStd

        let feedForwardSize = Int(Float(hiddenSize) * headFFNRatio)

        _noisyImagesProj.wrappedValue = Linear(latentSize, hiddenSize)
        _condProj.wrappedValue = Linear(hiddenSize, hiddenSize)
        _tEmbedder.wrappedValue = TADATimestepEmbedder(hiddenSize: hiddenSize)
        _layers.wrappedValue = (0..<headLayers).map { _ in
            TADAVibeVoiceHeadLayer(hiddenSize: hiddenSize, feedForwardSize: feedForwardSize)
        }
        _finalLayer.wrappedValue = TADAVibeVoiceFinalLayer(hiddenSize: hiddenSize, outputSize: latentSize)
    }

    public func callAsFunction(
        _ noisyLatents: MLXArray,
        timestep: MLXArray,
        conditioning: MLXArray
    ) -> MLXArray {
        let projectedNoise = noisyImagesProj(noisyLatents)
        let projectedConditioning = condProj(conditioning)
        let timestepEmbedding = broadcast(
            tEmbedder(timestep).expandedDimensions(axis: 1),
            to: projectedConditioning.shape
        )
        let modulation = projectedConditioning + timestepEmbedding

        var hidden = projectedNoise + projectedConditioning
        for layer in layers {
            hidden = layer(hidden, conditioning: modulation)
        }
        return finalLayer(hidden, conditioning: modulation)
    }

    public func computeVelocity(
        _ latents: MLXArray,
        timestep: Float,
        conditioning: MLXArray,
        negativeConditioning: MLXArray? = nil,
        acousticCFGScale: Float = TADAVibeVoice.defaultAcousticCFG,
        durationCFGScale: Float = TADAVibeVoice.defaultDurationCFG,
        guidanceScale: Float = 1
    ) -> MLXArray {
        let batch = latents.dim(0)
        let negative = negativeConditioning
            ?? MLXArray.zeros(conditioning.shape, dtype: conditioning.dtype)

        let timestepVector = MLXArray(Array(repeating: timestep, count: batch)).expandedDimensions(axis: -1)
        let duplicatedLatents = concatenated([latents, latents], axis: 0)
        let duplicatedTimestep = concatenated([timestepVector, timestepVector], axis: 0)
        let duplicatedConditioning = concatenated([conditioning, negative], axis: 0)

        let allVelocity = self(
            duplicatedLatents,
            timestep: duplicatedTimestep,
            conditioning: duplicatedConditioning
        )

        let positive = allVelocity[0..<batch, 0..., 0...]
        let negativeVelocity = allVelocity[batch..<(2 * batch), 0..., 0...]

        let acousticPositive = positive[0..., 0..., 0..<acousticDim]
        let acousticNegative = negativeVelocity[0..., 0..., 0..<acousticDim]
        let durationPositive = positive[0..., 0..., acousticDim...]
        let durationNegative = negativeVelocity[0..., 0..., acousticDim...]

        let acoustic = acousticNegative
            + MLXArray(acousticCFGScale * guidanceScale) * (acousticPositive - acousticNegative)
        let duration = durationNegative
            + MLXArray(durationCFGScale * guidanceScale) * (durationPositive - durationNegative)

        return concatenated([acoustic, duration], axis: -1)
    }

    public func solve(
        noise: MLXArray,
        conditioning: MLXArray,
        negativeConditioning: MLXArray? = nil,
        numSteps: Int = TADAVibeVoice.defaultFlowSteps,
        noiseTemperature: Float = TADAVibeVoice.defaultNoiseTemperature,
        acousticCFGScale: Float = TADAVibeVoice.defaultAcousticCFG,
        durationCFGScale: Float = TADAVibeVoice.defaultDurationCFG,
        cfgSchedule: TADACFGSchedule = .cosine
    ) -> TADAVibeVoiceSample {
        let schedule = logSNRSchedule(numSteps: numSteps)
        var speech = noise * noiseTemperature

        for step in 0..<numSteps {
            let tPrev = schedule[step]
            let tCurr = schedule[step + 1]
            let guidanceScale = cfgScheduleScale(step: step, totalSteps: numSteps, schedule: cfgSchedule)
            let velocity = computeVelocity(
                speech,
                timestep: tPrev,
                conditioning: conditioning,
                negativeConditioning: negativeConditioning,
                acousticCFGScale: acousticCFGScale,
                durationCFGScale: durationCFGScale,
                guidanceScale: guidanceScale
            )
            speech = speech + MLXArray(tCurr - tPrev) * velocity
        }

        return decodeSample(from: speech)
    }

    public func decodeSample(from latents: MLXArray) -> TADAVibeVoiceSample {
        let acoustic = latents[0..., 0..., 0..<acousticDim] * MLXArray(acousticStd) + MLXArray(acousticMean)
        let timeBits = latents[0..., 0..., acousticDim...]
        let beforeBits = timeBits[0..., 0..., 0..<numTimeBits]
        let afterBits = timeBits[0..., 0..., numTimeBits...]

        return TADAVibeVoiceSample(
            acoustic: acoustic,
            timeBefore: TADAGrayCodeDurationCodec.decode(beforeBits),
            timeAfter: TADAGrayCodeDurationCodec.decode(afterBits),
            latent: latents
        )
    }

    public static func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        tadaSanitizeModuleWeights(weights, prefix: "prediction_head")
    }

    public func logSNRSchedule(numSteps: Int) -> [Float] {
        guard numSteps > 0 else { return [0, 1] }
        return (0...numSteps).map { step in
            let linear = Float(5) - (Float(10) * Float(step) / Float(numSteps))
            return 1 / (1 + Foundation.exp(linear / 2))
        }
    }

    private func cfgScheduleScale(
        step: Int,
        totalSteps: Int,
        schedule: TADACFGSchedule
    ) -> Float {
        guard totalSteps > 1 else { return 1 }
        switch schedule {
        case .constant:
            return 1
        case .cosine:
            let progress = Float(step) / Float(totalSteps - 1)
            return 0.5 - 0.5 * Foundation.cos(progress * Float.pi)
        }
    }
}
