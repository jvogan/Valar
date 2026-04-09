import Foundation
@preconcurrency import MLX
import MLXNN

// MARK: - EuclideanCodebook

/// Euclidean codebook for vector quantization using an embedding table.
///
/// Stores `codebookSize` centroid vectors of dimension `dim`. The `decode`
/// method maps integer codes to their corresponding centroid vectors.
///
/// Reference: `speech_tokenizer.py` class `EuclideanCodebook`.
final class EuclideanCodebook: Module {
    let dim: Int
    @ModuleInfo var embed: Embedding

    init(dim: Int, codebookSize: Int) {
        self.dim = dim
        _embed.wrappedValue = Embedding(embeddingCount: codebookSize, dimensions: dim)
    }

    func decode(_ codes: MLXArray) -> MLXArray {
        embed(codes)
    }
}

// MARK: - VectorQuantization

/// Single-codebook vector quantization layer.
///
/// Decode path: embedding lookup then transpose to NCL format.
/// The Python reference supports an optional `project_out` linear projection
/// when `codebook_dim != dim`, but in the decoder config these are always
/// equal so the projection is omitted here.
///
/// Reference: `speech_tokenizer.py` class `VectorQuantization`.
final class VectorQuantization: Module {
    @ModuleInfo var codebook: EuclideanCodebook

    init(dim: Int, codebookSize: Int) {
        _codebook.wrappedValue = EuclideanCodebook(dim: dim, codebookSize: codebookSize)
    }

    func decode(_ codes: MLXArray) -> MLXArray {
        // codes: [batch, time]
        let quantized = codebook.decode(codes) // [batch, time, dim]
        return quantized.transposed(0, 2, 1)   // [batch, dim, time]
    }
}

// MARK: - ResidualVectorQuantization

/// Multi-layer residual vector quantization. Sums the decoded outputs of all
/// quantizer layers to reconstruct the continuous embedding.
///
/// Reference: `speech_tokenizer.py` class `ResidualVectorQuantization`.
final class ResidualVectorQuantization: Module {
    @ModuleInfo var layers: [VectorQuantization]

    init(numQuantizers: Int, dim: Int, codebookSize: Int) {
        _layers.wrappedValue = (0 ..< numQuantizers).map { _ in
            VectorQuantization(dim: dim, codebookSize: codebookSize)
        }
    }

    func decode(_ codes: MLXArray) -> MLXArray {
        // codes: [numQuantizers, batch, time]
        let batch = codes.dim(1)
        let time = codes.dim(2)
        let dim = layers.first?.codebook.dim ?? 0
        guard codes.dim(0) <= layers.count else {
            return MLXArray.zeros([batch, dim, time])
        }
        var quantized = MLXArray.zeros([batch, dim, time])
        for idx in 0 ..< codes.dim(0) {
            quantized = quantized + layers[idx].decode(codes[idx])
        }
        return quantized
    }
}

// MARK: - ResidualVectorQuantizer

/// Residual vector quantizer with input/output Conv1d projections.
///
/// Used as a sub-component of ``SplitResidualVectorQuantizer``. The
/// projections bridge between the model's latent dimension and the codebook
/// dimension. In the decode path only the output projection is applied;
/// `inputProj` is present for weight-loading compatibility.
///
/// Reference: `speech_tokenizer.py` class `ResidualVectorQuantizer`.
final class ResidualVectorQuantizer: Module {
    let dimension: Int
    @ModuleInfo var inputProj: Conv1d
    @ModuleInfo var outputProj: Conv1d
    @ModuleInfo var vq: ResidualVectorQuantization

    init(dimension: Int, inputDimension: Int, outputDimension: Int, nQ: Int, bins: Int) {
        self.dimension = dimension
        _inputProj.wrappedValue = Conv1d(
            inputChannels: inputDimension, outputChannels: dimension,
            kernelSize: 1, bias: false
        )
        _outputProj.wrappedValue = Conv1d(
            inputChannels: dimension, outputChannels: outputDimension,
            kernelSize: 1, bias: false
        )
        _vq.wrappedValue = ResidualVectorQuantization(
            numQuantizers: nQ, dim: dimension, codebookSize: bins
        )
    }

    func decode(_ codes: MLXArray) -> MLXArray {
        // codes: [batch, numQuantizers, time]
        let transposed = codes.transposed(1, 0, 2) // [numQuantizers, batch, time]
        var quantized = vq.decode(transposed)       // [batch, dim, time]
        // Output projection: NCL → NLC → Conv1d → NLC → NCL
        quantized = quantized.transposed(0, 2, 1)   // [batch, time, dim]
        quantized = outputProj(quantized)            // [batch, time, outputDim]
        quantized = quantized.transposed(0, 2, 1)   // [batch, outputDim, time]
        return quantized
    }
}

// MARK: - SplitResidualVectorQuantizer

/// Split residual vector quantizer with separate semantic and acoustic paths.
///
/// The first `nQSemantic` quantizer layers are decoded through `rvqFirst`
/// (semantic codebooks); remaining layers go through `rvqRest` (acoustic
/// codebooks). The two decoded results are summed to produce the final
/// continuous representation.
///
/// ## Weight Loading
///
/// Call ``sanitize(weights:)`` on raw PyTorch-keyed weight dictionaries
/// before passing them to `update(parameters:verify:)`. The sanitize
/// function remaps snake_case Python keys to camelCase Swift keys and
/// computes embedding matrices from `cluster_usage` / `embedding_sum`
/// pairs.
///
/// Reference: `speech_tokenizer.py` class `SplitResidualVectorQuantizer`.
final class SplitResidualVectorQuantizer: Module {
    let nQSemantic: Int
    @ModuleInfo var rvqFirst: ResidualVectorQuantizer
    @ModuleInfo var rvqRest: ResidualVectorQuantizer

    init(
        nQ: Int,
        nQSemantic: Int,
        dimension: Int,
        inputDimension: Int,
        outputDimension: Int,
        bins: Int
    ) {
        self.nQSemantic = nQSemantic
        _rvqFirst.wrappedValue = ResidualVectorQuantizer(
            dimension: dimension, inputDimension: inputDimension,
            outputDimension: outputDimension, nQ: nQSemantic, bins: bins
        )
        _rvqRest.wrappedValue = ResidualVectorQuantizer(
            dimension: dimension, inputDimension: inputDimension,
            outputDimension: outputDimension, nQ: nQ - nQSemantic, bins: bins
        )
    }

    func decode(_ codes: MLXArray) -> MLXArray {
        // codes: [batch, numQuantizers, time]
        var quantized = rvqFirst.decode(codes[0..., ..<nQSemantic, 0...])
        if codes.dim(1) > nQSemantic {
            quantized = quantized + rvqRest.decode(codes[0..., nQSemantic..., 0...])
        }
        return quantized
    }

    /// Remaps PyTorch / Python-style weight keys to the Swift module tree.
    ///
    /// Handles two transformations:
    /// 1. Snake-case → camelCase key renaming (`rvq_first` → `rvqFirst`, etc.)
    /// 2. Codebook computation: `_codebook.cluster_usage` + `_codebook.embedding_sum`
    ///    → `codebook.embed.weight` via normalized centroid calculation.
    static func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]
        var codebookData: [String: [String: MLXArray]] = [:]

        for (key, value) in weights {
            if key.contains("._codebook.cluster_usage") || key.contains("._codebook.embedding_sum") {
                guard let range = key.range(of: "._codebook.") else { continue }
                let basePath = String(key[key.startIndex ..< range.lowerBound])
                if codebookData[basePath] == nil { codebookData[basePath] = [:] }
                if key.hasSuffix("cluster_usage") {
                    codebookData[basePath]!["cluster_usage"] = value
                } else {
                    codebookData[basePath]!["embedding_sum"] = value
                }
                continue
            }

            result[remapKey(key)] = value
        }

        // Compute embedding matrices from cluster_usage and embedding_sum.
        let eps: Float = 1e-5
        for (basePath, data) in codebookData {
            guard let clusterUsage = data["cluster_usage"],
                  let embeddingSum = data["embedding_sum"] else { continue }
            let embedding = embeddingSum / clip(clusterUsage.reshaped(-1, 1), min: eps)
            result[remapKey("\(basePath).codebook.embed.weight")] = embedding
        }

        return result
    }

    private static func remapKey(_ key: String) -> String {
        key.replacingOccurrences(of: "rvq_first", with: "rvqFirst")
            .replacingOccurrences(of: "rvq_rest", with: "rvqRest")
            .replacingOccurrences(of: "input_proj", with: "inputProj")
            .replacingOccurrences(of: "output_proj", with: "outputProj")
    }
}
