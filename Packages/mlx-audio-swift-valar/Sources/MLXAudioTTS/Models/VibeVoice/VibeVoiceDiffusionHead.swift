// Copyright (c) 2025, Prince Canuma and contributors
// Swift port for ValarTTS

import Foundation
@preconcurrency import MLX
import MLXNN

// MARK: - Diffusion RMSNorm

/// RMSNorm for the diffusion head with optional element-wise affine.
final class VibeVoiceDiffusionRMSNorm: Module {
    @ParameterInfo var weight: MLXArray?
    let eps: Float
    let elementwiseAffine: Bool

    init(dim: Int, eps: Float = 1e-6, elementwiseAffine: Bool = true) {
        self.eps = eps
        self.elementwiseAffine = elementwiseAffine
        if elementwiseAffine {
            _weight.wrappedValue = MLXArray.ones([dim])
        } else {
            _weight.wrappedValue = nil
        }
        super.init()
    }

    private func norm(_ x: MLXArray) -> MLXArray {
        x * MLX.rsqrt(MLX.mean(x * x, axis: -1, keepDims: true) + eps)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let output = norm(x.asType(.float32)).asType(x.dtype)
        if let weight {
            return output * weight
        }
        return output
    }
}

// MARK: - AdaLN Modulation

/// Adaptive layer normalization modulation: `x * (1 + scale) + shift`.
private func modulate(_ x: MLXArray, shift: MLXArray, scale: MLXArray) -> MLXArray {
    x * (1 + scale) + shift
}

// MARK: - Timestep Embedder

/// Minimal MLP wrapper that preserves the original numbered Python weight keys
/// without routing through `Sequential.layers`, which MLXNN update struggles with
/// for sparse parameterized indices (0 and 2 with SiLU in the middle).
final class VibeVoiceTimestepMLP: Module {
    @ModuleInfo var inputProjection: Linear
    let activation: SiLU
    @ModuleInfo var outputProjection: Linear

    init(hiddenSize: Int, frequencyEmbeddingSize: Int) {
        _inputProjection.wrappedValue = Linear(frequencyEmbeddingSize, hiddenSize, bias: false)
        self.activation = SiLU()
        _outputProjection.wrappedValue = Linear(hiddenSize, hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        outputProjection(activation(inputProjection(x)))
    }
}

/// Embeds scalar diffusion timesteps into vector representations via sinusoidal encoding + MLP.
final class VibeVoiceTimestepEmbedder: Module {
    let frequencyEmbeddingSize: Int
    @ModuleInfo var mlp: VibeVoiceTimestepMLP

    init(hiddenSize: Int, frequencyEmbeddingSize: Int = 256) {
        self.frequencyEmbeddingSize = frequencyEmbeddingSize
        _mlp.wrappedValue = VibeVoiceTimestepMLP(
            hiddenSize: hiddenSize,
            frequencyEmbeddingSize: frequencyEmbeddingSize
        )
        super.init()
    }

    /// Create sinusoidal timestep embeddings.
    static func timestepEmbedding(_ t: MLXArray, dim: Int, maxPeriod: Int = 10000) -> MLXArray {
        let half = dim / 2
        let freqs = MLX.exp(
            -log(Float(maxPeriod)) * MLX.arange(0, half).asType(.float32) / Float(half)
        )
        let args = t.expandedDimensions(axis: 1).asType(.float32) * freqs.expandedDimensions(axis: 0)
        var embedding = MLX.concatenated([MLX.cos(args), MLX.sin(args)], axis: -1)
        if dim % 2 != 0 {
            embedding = MLX.concatenated([embedding, MLXArray.zeros(like: embedding[0..., ..<1])], axis: -1)
        }
        return embedding
    }

    func callAsFunction(_ t: MLXArray) -> MLXArray {
        let tFreq = Self.timestepEmbedding(t, dim: frequencyEmbeddingSize)
        return mlp(tFreq)
    }
}

// MARK: - Feed Forward (SwiGLU)

/// SwiGLU feed-forward network for diffusion head layers.
final class VibeVoiceDiffusionFFN: Module {
    @ModuleInfo var gateProj: Linear
    @ModuleInfo var upProj: Linear
    @ModuleInfo var downProj: Linear

    init(embedDim: Int, ffnDim: Int) {
        _gateProj.wrappedValue = Linear(embedDim, ffnDim, bias: false)
        _upProj.wrappedValue = Linear(embedDim, ffnDim, bias: false)
        _downProj.wrappedValue = Linear(ffnDim, embedDim, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

/// AdaLN modulation wrapper preserving the original numbered Python linear key
/// (`*.adaLN_modulation.1.weight`) without going through `Sequential.layers`.
final class VibeVoiceAdaLNModulation: Module {
    let activation: SiLU
    @ModuleInfo var projection: Linear

    init(condDim: Int, outputDim: Int) {
        self.activation = SiLU()
        _projection.wrappedValue = Linear(condDim, outputDim, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        projection(activation(x))
    }
}

// MARK: - Head Layer

/// A layer in the diffusion head with adaptive layer normalization (AdaLN).
///
/// Modulation produces shift, scale, gate vectors from the conditioning signal.
/// Formula: `x + gate * ffn(norm(x) * (1 + scale) + shift)`
final class VibeVoiceHeadLayer: Module {
    @ModuleInfo var ffn: VibeVoiceDiffusionFFN
    @ModuleInfo var norm: VibeVoiceDiffusionRMSNorm
    @ModuleInfo var adaLNModulation: VibeVoiceAdaLNModulation

    init(embedDim: Int, ffnDim: Int, condDim: Int, normEps: Float = 1e-5) {
        _ffn.wrappedValue = VibeVoiceDiffusionFFN(embedDim: embedDim, ffnDim: ffnDim)
        _norm.wrappedValue = VibeVoiceDiffusionRMSNorm(dim: embedDim, eps: normEps)
        _adaLNModulation.wrappedValue = VibeVoiceAdaLNModulation(
            condDim: condDim,
            outputDim: 3 * embedDim
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray, c: MLXArray) -> MLXArray {
        let modulation = adaLNModulation(c)
        let parts = MLX.split(modulation, parts: 3, axis: -1)
        let shift = parts[0]
        let scale = parts[1]
        let gate = parts[2]

        return x + gate * ffn(modulate(norm(x), shift: shift, scale: scale))
    }
}

// MARK: - Final Layer

/// Final layer in the diffusion head — AdaLN with shift+scale (no gate) followed by linear projection.
final class VibeVoiceFinalLayer: Module {
    @ModuleInfo var normFinal: VibeVoiceDiffusionRMSNorm
    @ModuleInfo var linear: Linear
    @ModuleInfo var adaLNModulation: VibeVoiceAdaLNModulation

    init(hiddenSize: Int, outputSize: Int, condSize: Int, normEps: Float = 1e-5) {
        _normFinal.wrappedValue = VibeVoiceDiffusionRMSNorm(dim: hiddenSize, eps: normEps, elementwiseAffine: false)
        _linear.wrappedValue = Linear(hiddenSize, outputSize, bias: false)
        _adaLNModulation.wrappedValue = VibeVoiceAdaLNModulation(
            condDim: condSize,
            outputDim: 2 * hiddenSize
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray, c: MLXArray) -> MLXArray {
        let modulation = adaLNModulation(c)
        let parts = MLX.split(modulation, parts: 2, axis: -1)
        let shift = parts[0]
        let scale = parts[1]
        return linear(modulate(normFinal(x), shift: shift, scale: scale))
    }
}

// MARK: - Diffusion Head

/// Diffusion prediction head for VibeVoice.
///
/// Predicts noise/velocity for the DPM-Solver diffusion process.
/// Architecture: noisy_images_proj → (cond_proj + t_embedder) → N HeadLayers → FinalLayer.
final class VibeVoiceDiffusionHead: Module {
    let condDim: Int

    @ModuleInfo var noisyImagesProj: Linear
    @ModuleInfo var condProj: Linear
    @ModuleInfo var tEmbedder: VibeVoiceTimestepEmbedder
    @ModuleInfo(key: "layers") var layers: [VibeVoiceHeadLayer]
    @ModuleInfo var finalLayer: VibeVoiceFinalLayer

    init(config: VibeVoiceDiffusionHeadConfig) {
        let condDim = config.hiddenSize
        self.condDim = condDim
        let latentSize = config.latentSize
        let ffnDim = config.ffnDim

        _noisyImagesProj.wrappedValue = Linear(latentSize, config.hiddenSize, bias: false)
        _condProj.wrappedValue = Linear(config.hiddenSize, condDim, bias: false)
        _tEmbedder.wrappedValue = VibeVoiceTimestepEmbedder(hiddenSize: condDim)

        _layers.wrappedValue = (0..<config.headLayers).map { _ in
            VibeVoiceHeadLayer(
                embedDim: config.hiddenSize,
                ffnDim: ffnDim,
                condDim: condDim,
                normEps: config.rmsNormEps
            )
        }

        _finalLayer.wrappedValue = VibeVoiceFinalLayer(
            hiddenSize: config.hiddenSize,
            outputSize: latentSize,
            condSize: condDim,
            normEps: config.rmsNormEps
        )

        super.init()
    }

    /// Forward pass of the prediction head.
    ///
    /// - Parameters:
    ///   - noisyImages: Noisy latents to denoise, shape (B, latentSize)
    ///   - timesteps: Diffusion timesteps, shape (B,)
    ///   - condition: Conditioning information, shape (B, hiddenSize)
    /// - Returns: Predicted noise/velocity, shape (B, latentSize)
    func callAsFunction(
        _ noisyImages: MLXArray,
        timesteps: MLXArray,
        condition: MLXArray
    ) -> MLXArray {
        var x = noisyImagesProj(noisyImages)
        let t = tEmbedder(timesteps)
        let c = condProj(condition) + t

        for layer in layers {
            x = layer(x, c: c)
        }

        return finalLayer(x, c: c)
    }
}
