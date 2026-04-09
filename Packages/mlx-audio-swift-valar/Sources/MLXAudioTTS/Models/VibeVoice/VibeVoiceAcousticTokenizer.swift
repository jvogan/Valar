// Copyright (c) 2025, Prince Canuma and contributors
// Swift port for ValarTTS

import Foundation
@preconcurrency import MLX
import MLXNN

// MARK: - ConvRMSNorm

/// RMSNorm for convolutional features in (B, C, T) format.
/// Transposes to (B, T, C) for normalization, then transposes back.
final class VibeVoiceConvRMSNorm: Module {
    @ParameterInfo var weight: MLXArray?
    let eps: Float

    init(dim: Int, eps: Float = 1e-5, elementwiseAffine: Bool = true) {
        self.eps = eps
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
        // x: (B, C, T) -> (B, T, C)
        var t = x.transposed(0, 2, 1)
        t = norm(t.asType(.float32)).asType(t.dtype)
        if let weight {
            t = t * weight
        }
        // (B, T, C) -> (B, C, T)
        return t.transposed(0, 2, 1)
    }
}

// MARK: - CausalConv1d

/// Causal 1D convolution with left-padding.
///
/// Operates in (B, C, T) format externally but transposes to (B, T, C) for MLX Conv1d.
final class VibeVoiceCausalConv1d: Module {
    let padding: Int
    @ModuleInfo var conv: Conv1d

    init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        dilation: Int = 1,
        groups: Int = 1,
        bias: Bool = true
    ) {
        self.padding = (kernelSize - 1) * dilation
        _conv.wrappedValue = Conv1d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: kernelSize,
            stride: stride,
            padding: 0,
            dilation: dilation,
            groups: groups,
            bias: bias
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // (B, C, T) -> (B, T, C)
        var t = x.transposed(0, 2, 1)

        // Left-pad for causal
        if padding > 0 {
            t = MLX.padded(t, widths: [IntOrPair((0, 0)), IntOrPair((padding, 0)), IntOrPair((0, 0))])
        }

        // Conv and back
        t = conv(t)
        return t.transposed(0, 2, 1)
    }
}

// MARK: - CausalConvTranspose1d

/// Causal transposed 1D convolution for upsampling.
///
/// After the transposed convolution, trims padding to maintain causality.
final class VibeVoiceCausalConvTranspose1d: Module {
    let paddingTotal: Int
    let trimRightRatio: Float

    @ModuleInfo var convtr: ConvTransposed1d

    init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        bias: Bool = true,
        trimRightRatio: Float = 1.0
    ) {
        self.paddingTotal = kernelSize - stride
        self.trimRightRatio = trimRightRatio

        _convtr.wrappedValue = ConvTransposed1d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: kernelSize,
            stride: stride,
            padding: 0,
            bias: bias
        )

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // (B, C, T) -> (B, T, C)
        var t = x.transposed(0, 2, 1)
        t = convtr(t)
        // (B, T, C) -> (B, C, T)
        t = t.transposed(0, 2, 1)

        // Trim padding for causality
        let paddingRight = Int(ceilf(Float(paddingTotal) * trimRightRatio))
        let paddingLeft = paddingTotal - paddingRight

        if paddingLeft > 0 {
            t = t[0..., 0..., paddingLeft...]
        }
        if paddingRight > 0 {
            t = t[0..., 0..., ..<(-paddingRight)]
        }
        return t
    }
}

// MARK: - Depthwise Conv

/// Depthwise separable convolution wrapped to match HF structure (mixer.conv.conv.conv).
final class VibeVoiceDepthwiseConv: Module {
    @ModuleInfo var conv: VibeVoiceCausalConv1d

    init(dim: Int, kernelSize: Int = 7, bias: Bool = true) {
        _conv.wrappedValue = VibeVoiceCausalConv1d(
            inChannels: dim, outChannels: dim,
            kernelSize: kernelSize, groups: dim, bias: bias
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { conv(x) }
}

// MARK: - Mixer

/// Mixer module wrapping depthwise conv.
final class VibeVoiceMixer: Module {
    @ModuleInfo var conv: VibeVoiceDepthwiseConv

    init(dim: Int, kernelSize: Int = 7, bias: Bool = true) {
        _conv.wrappedValue = VibeVoiceDepthwiseConv(dim: dim, kernelSize: kernelSize, bias: bias)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { conv(x) }
}

// MARK: - FeedForward

/// Feed-forward network with GELU activation for the acoustic tokenizer.
final class VibeVoiceAcousticFeedForward: Module {
    @ModuleInfo var linear1: Linear
    @ModuleInfo var linear2: Linear

    init(dim: Int, mult: Float = 4.0, bias: Bool = true) {
        let hiddenDim = Int(Float(dim) * mult)
        _linear1.wrappedValue = Linear(dim, hiddenDim, bias: bias)
        _linear2.wrappedValue = Linear(hiddenDim, dim, bias: bias)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear2(gelu(linear1(x)))
    }
}

// MARK: - Block1D

/// 1D convolutional block with depthwise conv and FFN, using layer-scale gamma.
///
/// Architecture: norm → mixer → gamma scale + residual → norm → FFN → gamma scale + residual.
final class VibeVoiceBlock1D: Module {
    @ModuleInfo var norm: VibeVoiceConvRMSNorm
    @ModuleInfo var ffnNorm: VibeVoiceConvRMSNorm
    @ModuleInfo var mixer: VibeVoiceMixer
    @ModuleInfo var ffn: VibeVoiceAcousticFeedForward
    @ParameterInfo var gamma: MLXArray?
    @ParameterInfo var ffnGamma: MLXArray?

    init(dim: Int, eps: Float = 1e-6, bias: Bool = true, layerScaleInitValue: Float = 1e-6) {
        _norm.wrappedValue = VibeVoiceConvRMSNorm(dim: dim, eps: eps)
        _ffnNorm.wrappedValue = VibeVoiceConvRMSNorm(dim: dim, eps: eps)
        _mixer.wrappedValue = VibeVoiceMixer(dim: dim, kernelSize: 7, bias: bias)
        _ffn.wrappedValue = VibeVoiceAcousticFeedForward(dim: dim, mult: 4.0, bias: bias)

        if layerScaleInitValue > 0 {
            _gamma.wrappedValue = MLXArray.ones([dim]) * layerScaleInitValue
            _ffnGamma.wrappedValue = MLXArray.ones([dim]) * layerScaleInitValue
        } else {
            _gamma.wrappedValue = nil
            _ffnGamma.wrappedValue = nil
        }

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: (B, C, T)

        // Mixer path
        var out = x
        var residual = out
        out = norm(out)
        out = mixer(out)
        if let gamma {
            out = out * gamma.expandedDimensions(axes: [0, 2])
        }
        out = residual + out

        // FFN path
        residual = out
        out = ffnNorm(out)
        // (B, C, T) -> (B, T, C) for FFN
        out = out.transposed(0, 2, 1)
        out = ffn(out)
        // (B, T, C) -> (B, C, T)
        out = out.transposed(0, 2, 1)
        if let ffnGamma {
            out = out * ffnGamma.expandedDimensions(axes: [0, 2])
        }
        out = residual + out

        return out
    }
}

// MARK: - Stem Conv

/// Stem convolution layer, matching HF structure (upsample_layers.0.0.conv).
final class VibeVoiceStemConv: Module {
    @ModuleInfo var conv: VibeVoiceCausalConv1d

    init(inChannels: Int, outChannels: Int, kernelSize: Int = 7, bias: Bool = true) {
        _conv.wrappedValue = VibeVoiceCausalConv1d(
            inChannels: inChannels, outChannels: outChannels,
            kernelSize: kernelSize, bias: bias
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { conv(x) }
}

// MARK: - Upsample Layer

/// Upsample layer with transposed convolution.
final class VibeVoiceUpsampleLayer: Module {
    @ModuleInfo var convtr: VibeVoiceCausalConvTranspose1d

    init(inChannels: Int, outChannels: Int, kernelSize: Int, stride: Int, bias: Bool = true) {
        _convtr.wrappedValue = VibeVoiceCausalConvTranspose1d(
            inChannels: inChannels, outChannels: outChannels,
            kernelSize: kernelSize, stride: stride, bias: bias
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { convtr(x) }
}

// MARK: - Head Conv

/// Output head convolution.
final class VibeVoiceHeadConv: Module {
    @ModuleInfo var conv: VibeVoiceCausalConv1d

    init(inChannels: Int, outChannels: Int, kernelSize: Int = 7, bias: Bool = true) {
        _conv.wrappedValue = VibeVoiceCausalConv1d(
            inChannels: inChannels, outChannels: outChannels,
            kernelSize: kernelSize, bias: bias
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { conv(x) }
}

// MARK: - Tokenizer Decoder

/// A stage of Block1D modules. Wrapping the array in its own module keeps
/// MLXNN parameter updates deterministic and future-model-friendly.
final class VibeVoiceBlockStage: Module {
    @ModuleInfo(key: "layers") var layers: [VibeVoiceBlock1D]

    init(layers: [VibeVoiceBlock1D]) {
        _layers.wrappedValue = layers
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = x
        for layer in layers {
            out = layer(out)
        }
        return out
    }
}

/// Decoder that converts latent representations back to audio waveforms.
///
/// Architecture (7-stage):
/// - `stem`: initial causal conv
/// - `upsamplers[0..5]`: transposed convolutions for upsampling
/// - `stages[0..6]`: Block1D stage wrappers
/// - `head`: output convolution to mono audio
final class VibeVoiceTokenizerDecoder: Module {
    let dimension: Int
    let nStages: Int

    @ModuleInfo var stem: VibeVoiceStemConv
    @ModuleInfo(key: "upsamplers") var upsamplers: [VibeVoiceUpsampleLayer]
    @ModuleInfo(key: "stages") var stages: [VibeVoiceBlockStage]
    @ModuleInfo var head: VibeVoiceHeadConv

    init(config: VibeVoiceAcousticTokenizerConfig) {
        self.dimension = config.vaeDim
        let nFilters = config.decoderNFilters > 0 ? config.decoderNFilters : config.encoderNFilters
        let ratios = config.effectiveDecoderRatios
        let depths = config.parsedDecoderDepths
        self.nStages = depths.count

        // First: stem conv
        let stemOutCh = nFilters * (1 << (nStages - 1))
        _stem.wrappedValue = VibeVoiceStemConv(
            inChannels: config.vaeDim, outChannels: stemOutCh,
            kernelSize: 7, bias: config.convBias
        )

        // Remaining: transposed convolutions
        var upsamplerModules: [VibeVoiceUpsampleLayer] = []
        for i in 0..<ratios.count {
            let inCh = nFilters * (1 << (nStages - 1 - i))
            let outCh: Int
            if i < ratios.count - 1 {
                outCh = nFilters * (1 << (nStages - 2 - i))
            } else {
                outCh = nFilters
            }
            upsamplerModules.append(VibeVoiceUpsampleLayer(
                inChannels: inCh, outChannels: outCh,
                kernelSize: ratios[i] * 2, stride: ratios[i],
                bias: config.convBias
            ))
        }
        _upsamplers.wrappedValue = upsamplerModules

        // Build stages
        var stageModules: [VibeVoiceBlockStage] = []
        for i in 0..<nStages {
            let inCh = nFilters * (1 << (nStages - 1 - i))
            let blocks = (0..<depths[i]).map { _ in
                VibeVoiceBlock1D(
                    dim: inCh, eps: config.layernormEps,
                    bias: config.convBias,
                    layerScaleInitValue: config.layerScaleInitValue
                )
            }
            stageModules.append(VibeVoiceBlockStage(layers: blocks))
        }
        _stages.wrappedValue = stageModules

        // Output head
        _head.wrappedValue = VibeVoiceHeadConv(
            inChannels: nFilters, outChannels: config.channels,
            kernelSize: 7, bias: config.convBias
        )

        super.init()
    }

    /// Decode latents to audio.
    /// - Parameter x: Latent tensor (B, T, D) or (B, D, T)
    /// - Returns: Audio tensor (B, 1, T')
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = x
        // Ensure (B, D, T) format
        if out.dim(1) != dimension {
            out = out.transposed(0, 2, 1)
        }

        // Apply stem
        out = stem(out)

        // Process through stages and upsampling
        for i in 0..<nStages {
            out = stages[i](out)
            if i < upsamplers.count {
                out = upsamplers[i](out)
            }
        }

        // Output head
        out = head(out)
        return out
    }
}

// MARK: - Acoustic Tokenizer

/// VibeVoice acoustic tokenizer (decoder-only for inference).
///
/// Converts speech latent sequences to 24kHz mono audio waveforms.
final class VibeVoiceAcousticTokenizer: Module {
    @ModuleInfo var decoder: VibeVoiceTokenizerDecoder

    init(config: VibeVoiceAcousticTokenizerConfig) {
        _decoder.wrappedValue = VibeVoiceTokenizerDecoder(config: config)
        super.init()
    }

    /// Decode latent representations to audio.
    /// - Parameter latents: Shape (B, T, D) where D = vaeDim
    /// - Returns: Audio tensor (B, 1, T')
    func decode(_ latents: MLXArray) -> MLXArray {
        decoder(latents)
    }
}
