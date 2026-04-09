// Copyright (c) 2025, Prince Canuma and contributors
// Swift port for ValarTTS

import Foundation
@preconcurrency import MLX

// MARK: - Scheduler Output

/// Output from a single DPM-Solver step.
struct VibeVoiceDPMSchedulerOutput {
    let prevSample: MLXArray
    let x0Pred: MLXArray?
}

// MARK: - Beta Schedule

/// Create a beta schedule based on alpha_bar function (cosine schedule).
private func betasForAlphaBar(
    numDiffusionTimesteps: Int,
    maxBeta: Float = 0.999,
    alphaTransformType: String = "cosine"
) -> [Float] {
    var betas: [Float] = []
    betas.reserveCapacity(numDiffusionTimesteps)

    for i in 0..<numDiffusionTimesteps {
        let t1 = Float(i) / Float(numDiffusionTimesteps)
        let t2 = Float(i + 1) / Float(numDiffusionTimesteps)

        let alphaBar1: Float
        let alphaBar2: Float
        if alphaTransformType == "cosine" {
            alphaBar1 = pow(cos((t1 + 0.008) / 1.008 * .pi / 2), 2)
            alphaBar2 = pow(cos((t2 + 0.008) / 1.008 * .pi / 2), 2)
        } else if alphaTransformType == "exp" {
            alphaBar1 = exp(t1 * -12.0)
            alphaBar2 = exp(t2 * -12.0)
        } else {
            fatalError("Unsupported alpha_transform_type: \(alphaTransformType)")
        }
        betas.append(min(1 - alphaBar2 / alphaBar1, maxBeta))
    }
    return betas
}

// MARK: - DPM-Solver Multistep Scheduler

/// DPM-Solver multistep scheduler for fast diffusion sampling.
///
/// Implements DPM-Solver++ algorithm with first and second order updates,
/// using a cosine beta schedule and v_prediction mode.
final class VibeVoiceDPMSolverMultistepScheduler {
    let numTrainTimesteps: Int
    let predictionType: String
    let solverOrder: Int
    let lowerOrderFinal: Bool
    let finalSigmasType: String

    // Precomputed schedule values
    private let alphaT: [Float]
    private let sigmaT: [Float]

    // State
    var numInferenceSteps: Int = 0
    var timesteps: [Int] = []
    private var cachedAlphaT: [Float] = []
    private var cachedSigmaT: [Float] = []
    private var cachedLambda: [Float] = []
    private var modelOutputs: [MLXArray?]
    private var lowerOrderNums: Int = 0
    private var stepIndex: Int?

    init(
        numTrainTimesteps: Int = 1000,
        betaSchedule: String = "cosine",
        predictionType: String = "v_prediction",
        solverOrder: Int = 2,
        lowerOrderFinal: Bool = true,
        finalSigmasType: String = "zero"
    ) {
        self.numTrainTimesteps = numTrainTimesteps
        self.predictionType = predictionType
        self.solverOrder = solverOrder
        self.lowerOrderFinal = lowerOrderFinal
        self.finalSigmasType = finalSigmasType

        // Create beta schedule (always cosine for VibeVoice)
        let betas = betasForAlphaBar(
            numDiffusionTimesteps: numTrainTimesteps,
            alphaTransformType: "cosine"
        )

        // Compute cumulative alpha products
        var alphasCumprod: [Float] = []
        alphasCumprod.reserveCapacity(numTrainTimesteps)
        var cumulativeProduct: Float = 1.0
        for beta in betas {
            cumulativeProduct *= (1.0 - beta)
            alphasCumprod.append(cumulativeProduct)
        }

        // DPM-Solver parameters
        self.alphaT = alphasCumprod.map { sqrtf($0) }
        self.sigmaT = alphasCumprod.map { sqrtf(1 - $0) }

        self.modelOutputs = Array(repeating: nil, count: solverOrder)
    }

    /// Set the number of inference steps and precompute timestep values.
    func setTimesteps(_ numInferenceSteps: Int) {
        self.numInferenceSteps = numInferenceSteps

        // Create timesteps — linspace from max to 0
        var timestepValues: [Int] = []
        timestepValues.reserveCapacity(numInferenceSteps)
        for i in 0..<numInferenceSteps {
            let t = Float(numTrainTimesteps - 1) * (1.0 - Float(i) / Float(numInferenceSteps))
            timestepValues.append(Int(t.rounded()))
        }
        self.timesteps = timestepValues

        // Precompute values for each inference timestep
        cachedAlphaT = []
        cachedSigmaT = []
        cachedLambda = []
        cachedAlphaT.reserveCapacity(numInferenceSteps + 1)
        cachedSigmaT.reserveCapacity(numInferenceSteps + 1)
        cachedLambda.reserveCapacity(numInferenceSteps + 1)

        for t in timestepValues {
            let sigma = sqrtf((1 - alphaT[t] * alphaT[t]) / (alphaT[t] * alphaT[t]))
            let alphaVal = 1.0 / sqrtf(sigma * sigma + 1.0)
            let sigmaVal = sigma * alphaVal
            let lambdaVal = logf(alphaVal) - logf(sigmaVal)

            cachedAlphaT.append(alphaVal)
            cachedSigmaT.append(sigmaVal)
            cachedLambda.append(lambdaVal)
        }

        // Add final step values
        cachedAlphaT.append(1.0)
        cachedSigmaT.append(0.0)
        cachedLambda.append(Float.infinity)

        // Reset state
        modelOutputs = Array(repeating: nil, count: solverOrder)
        lowerOrderNums = 0
        stepIndex = nil
    }

    /// Convert model output to x0 prediction based on prediction type.
    private func convertModelOutput(
        _ modelOutput: MLXArray,
        sample: MLXArray,
        stepIdx: Int
    ) -> MLXArray {
        let alpha = cachedAlphaT[stepIdx]
        let sigma = cachedSigmaT[stepIdx]

        switch predictionType {
        case "epsilon":
            return (sample - sigma * modelOutput) / alpha
        case "v_prediction":
            return alpha * sample - sigma * modelOutput
        case "sample":
            return modelOutput
        default:
            fatalError("Unknown prediction_type: \(predictionType)")
        }
    }

    /// First order DPM-Solver++ update.
    private func firstOrderUpdate(
        x0Pred: MLXArray,
        sample: MLXArray,
        stepIdx: Int
    ) -> MLXArray {
        let alphaNext = cachedAlphaT[stepIdx + 1]
        let sigmaNext = cachedSigmaT[stepIdx + 1]
        let sigmaCurr = cachedSigmaT[stepIdx]
        let lambdaNext = cachedLambda[stepIdx + 1]
        let lambdaCurr = cachedLambda[stepIdx]
        let h = lambdaNext - lambdaCurr

        let sigmaRatio: Float = sigmaCurr > 0 ? sigmaNext / sigmaCurr : 0.0
        let expNegH = expf(-h)

        return sigmaRatio * sample - alphaNext * (expNegH - 1.0) * x0Pred
    }

    /// Second order DPM-Solver++ update.
    private func secondOrderUpdate(
        x0Pred: MLXArray,
        prevX0: MLXArray,
        sample: MLXArray,
        stepIdx: Int
    ) -> MLXArray {
        let alphaNext = cachedAlphaT[stepIdx + 1]
        let sigmaNext = cachedSigmaT[stepIdx + 1]
        let sigmaCurr = cachedSigmaT[stepIdx]
        let lambdaNext = cachedLambda[stepIdx + 1]
        let lambdaCurr = cachedLambda[stepIdx]
        let lambdaPrev = stepIdx > 0 ? cachedLambda[stepIdx - 1] : lambdaCurr

        let h = lambdaNext - lambdaCurr
        let h0 = lambdaCurr - lambdaPrev
        let r0: Float = h != 0 ? h0 / h : 1.0

        let D0 = x0Pred
        let D1: MLXArray
        if r0 != 0 {
            D1 = (1.0 / r0) * (x0Pred - prevX0)
        } else {
            D1 = MLXArray.zeros(like: x0Pred)
        }

        let sigmaRatio: Float = sigmaCurr > 0 ? sigmaNext / sigmaCurr : 0.0
        let expNegH = expf(-h)

        return sigmaRatio * sample
            - alphaNext * (expNegH - 1.0) * D0
            - 0.5 * alphaNext * (expNegH - 1.0) * D1
    }

    /// Perform one step of the DPM-Solver.
    func step(
        modelOutput: MLXArray,
        timestep: Int,
        sample: MLXArray,
        prevX0: MLXArray? = nil
    ) -> VibeVoiceDPMSchedulerOutput {
        if stepIndex == nil {
            stepIndex = 0
        }
        let idx = stepIndex!

        // Convert to x0 prediction
        let x0Pred = convertModelOutput(modelOutput, sample: sample, stepIdx: idx)

        // Shift model outputs for multi-order
        for i in stride(from: solverOrder - 1, through: 1, by: -1) {
            modelOutputs[i] = modelOutputs[i - 1]
        }
        modelOutputs[0] = x0Pred

        // Determine order for this step
        let lowerOrderFinalFlag = (idx == numInferenceSteps - 1) && (
            (lowerOrderFinal && numInferenceSteps < 15) || finalSigmasType == "zero"
        )

        let order: Int
        if lowerOrderNums < 1 || lowerOrderFinalFlag {
            order = 1
        } else if solverOrder == 2 || lowerOrderNums < 2 {
            order = 2
        } else {
            order = solverOrder
        }

        // Compute previous sample
        let prevSample: MLXArray
        if order == 1 {
            prevSample = firstOrderUpdate(x0Pred: x0Pred, sample: sample, stepIdx: idx)
        } else if order == 2 {
            if let prevX0 {
                prevSample = secondOrderUpdate(x0Pred: x0Pred, prevX0: prevX0, sample: sample, stepIdx: idx)
            } else if let stored = modelOutputs[1] {
                prevSample = secondOrderUpdate(x0Pred: x0Pred, prevX0: stored, sample: sample, stepIdx: idx)
            } else {
                prevSample = firstOrderUpdate(x0Pred: x0Pred, sample: sample, stepIdx: idx)
            }
        } else {
            // Higher orders: fall back to second order
            if let prevX0 {
                prevSample = secondOrderUpdate(x0Pred: x0Pred, prevX0: prevX0, sample: sample, stepIdx: idx)
            } else {
                prevSample = firstOrderUpdate(x0Pred: x0Pred, sample: sample, stepIdx: idx)
            }
        }

        // Update lower order count
        if lowerOrderNums < solverOrder - 1 {
            lowerOrderNums += 1
        }

        stepIndex = idx + 1

        return VibeVoiceDPMSchedulerOutput(prevSample: prevSample, x0Pred: x0Pred)
    }

    /// Reset scheduler state for new generation.
    func reset() {
        modelOutputs = Array(repeating: nil, count: solverOrder)
        lowerOrderNums = 0
        stepIndex = nil
    }
}
