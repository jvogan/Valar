import Foundation
@preconcurrency import MLX
import MLXNN

// MARK: - Weight Load Result

/// Result of loading and sanitizing speech tokenizer weights.
///
/// Contains the sanitized weight dictionary alongside any diagnostic information
/// about missing or unexpected keys. Consumers should inspect ``missingKeys``
/// and ``unexpectedKeys`` to detect weight-file mismatches before calling
/// `Module.update(parameters:verify:)`.
struct SpeechTokenizerWeightLoadResult {
    let weights: [String: MLXArray]
    let missingKeys: Set<String>
    let unexpectedKeys: Set<String>
}

// MARK: - Errors

enum SpeechTokenizerWeightLoaderError: Error, LocalizedError {
    case directoryNotFound(String)
    case noSafetensorsFiles(String)
    case pathTraversalDetected(String)
    case shapeMismatch(key: String, expected: [Int], actual: [Int])

    var errorDescription: String? {
        switch self {
        case .directoryNotFound(let path):
            return "Speech tokenizer directory not found: \(path)"
        case .noSafetensorsFiles(let path):
            return "No .safetensors files found in: \(path)"
        case .pathTraversalDetected(let path):
            return "Path traversal detected in weight loader: \(path). Only files within the declared directory are permitted."
        case .shapeMismatch(let key, let expected, let actual):
            return "Shape mismatch for '\(key)': expected \(expected), got \(actual)"
        }
    }
}

// MARK: - Weight Loader

/// Loads and sanitizes weights from the `speech_tokenizer/` subdirectory.
///
/// The speech tokenizer ships its weights in one or more `.safetensors` files
/// inside a `speech_tokenizer/` subdirectory of the model directory. The raw
/// weight keys use Python-style naming conventions (snake_case, PyTorch tensor
/// layout). This loader:
///
/// 1. Discovers all `.safetensors` files in the directory.
/// 2. Loads and merges weight dictionaries.
/// 3. Remaps Python keys to Swift module paths (camelCase).
/// 4. Transposes Conv1d / ConvTranspose1d weights from PyTorch to MLX layout.
/// 5. Computes codebook embeddings from `cluster_usage` / `embedding_sum` pairs.
/// 6. Validates that all expected keys are present and reports discrepancies.
///
/// Reference: `speech_tokenizer.py` class `Qwen3TTSSpeechTokenizer.sanitize`.
enum SpeechTokenizerWeightLoader {

    // MARK: - Public API

    /// Discover all `.safetensors` files in the given directory, sorted by name.
    ///
    /// The directory URL is canonicalized (standardized + symlinks resolved) before
    /// enumeration. Each discovered file is then verified to resolve within the
    /// canonicalized directory so that symlinks pointing outside the directory
    /// cannot silently redirect weight loading to arbitrary filesystem locations.
    static func discoverSafetensors(in directory: URL) throws -> [URL] {
        // Canonicalize the directory to remove `.`/`..` components and resolve symlinks.
        // This ensures the containment check below is performed against the real path.
        let resolvedDirectory = directory.standardized.resolvingSymlinksInPath()

        let fm = FileManager.default
        guard fm.fileExists(atPath: resolvedDirectory.path) else {
            throw SpeechTokenizerWeightLoaderError.directoryNotFound(directory.path)
        }
        let contents = try fm.contentsOfDirectory(
            at: resolvedDirectory,
            includingPropertiesForKeys: nil
        )
        // Build the canonical root prefix once for O(1) containment checks.
        let rootPrefix = resolvedDirectory.path.hasSuffix("/")
            ? resolvedDirectory.path
            : "\(resolvedDirectory.path)/"

        var safetensors: [URL] = []
        for fileURL in contents where fileURL.pathExtension == "safetensors" {
            // Resolve the file URL so symlinks pointing outside the directory are caught.
            let resolvedFile = fileURL.resolvingSymlinksInPath()
            guard resolvedFile.path.hasPrefix(rootPrefix) else {
                throw SpeechTokenizerWeightLoaderError.pathTraversalDetected(fileURL.path)
            }
            safetensors.append(fileURL)
        }
        safetensors.sort { $0.lastPathComponent < $1.lastPathComponent }

        guard !safetensors.isEmpty else {
            throw SpeechTokenizerWeightLoaderError.noSafetensorsFiles(resolvedDirectory.path)
        }
        return safetensors
    }

    /// Load and merge all safetensors files from the directory.
    static func loadRawWeights(from directory: URL) throws -> [String: MLXArray] {
        let files = try discoverSafetensors(in: directory)
        var merged: [String: MLXArray] = [:]
        for file in files {
            let weights = try MLX.loadArrays(url: file)
            merged.merge(weights) { _, newest in newest }
        }
        return merged
    }

    /// Sanitize raw Python-keyed weights to Swift module paths.
    ///
    /// Handles three transformations:
    /// 1. **Key remapping**: snake_case Python keys → camelCase Swift module paths
    ///    (`rvq_first` → `rvqFirst`, `pre_conv` → `preConv`, etc.)
    /// 2. **Conv1d transpose**: PyTorch `[out, in, kernel]` → MLX `[out, kernel, in]`
    ///    and ConvTranspose1d: PyTorch `[in, out, kernel]` → MLX `[out, kernel, in]`
    /// 3. **Codebook computation**: `_codebook.cluster_usage` + `_codebook.embedding_sum`
    ///    → `codebook.embed.weight` via normalized centroid calculation.
    static func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        var codebookData: [String: [String: MLXArray]] = [:]

        for (key, originalValue) in weights {
            // Collect codebook data for deferred computation.
            if key.contains("._codebook.cluster_usage") || key.contains("._codebook.embedding_sum") {
                guard let range = key.range(of: "._codebook.") else { continue }
                let basePath = String(key[key.startIndex ..< range.lowerBound])
                if codebookData[basePath] == nil { codebookData[basePath] = [:] }
                if key.hasSuffix("cluster_usage") {
                    codebookData[basePath]!["cluster_usage"] = originalValue
                } else {
                    codebookData[basePath]!["embedding_sum"] = originalValue
                }
                continue
            }

            var value = originalValue

            // Transpose 3D weight tensors from PyTorch layout to MLX layout.
            if key.hasSuffix(".weight"), value.ndim == 3, !isMLXLayout(value) {
                if isTransposeConvWeight(key: key) {
                    // ConvTranspose1d: PyTorch [in, out, kernel] → MLX [out, kernel, in]
                    value = value.transposed(1, 2, 0)
                } else if key.contains(".conv.weight") || key.contains("_proj.weight") {
                    // Conv1d / projection: PyTorch [out, in, kernel] → MLX [out, kernel, in]
                    value = value.transposed(0, 2, 1)
                }
            }

            sanitized[remapKey(key)] = value
        }

        // Compute codebook embeddings from cluster_usage and embedding_sum.
        let eps: Float = 1e-5
        for (basePath, data) in codebookData {
            guard let clusterUsage = data["cluster_usage"],
                  let embeddingSum = data["embedding_sum"] else { continue }
            let embedding = embeddingSum / clip(clusterUsage.reshaped(-1, 1), min: eps)
            sanitized[remapKey("\(basePath).codebook.embed.weight")] = embedding
        }

        return sanitized
    }

    /// Validate sanitized weights against a set of expected keys.
    ///
    /// Returns a result containing the weights and any diagnostic information
    /// about missing or unexpected keys. An empty `expectedKeys` set skips
    /// validation and returns empty diagnostic sets.
    static func validate(
        sanitized: [String: MLXArray],
        expectedKeys: Set<String>
    ) -> SpeechTokenizerWeightLoadResult {
        let loadedKeys = Set(sanitized.keys)
        let missing = expectedKeys.subtracting(loadedKeys)
        let unexpected = loadedKeys.subtracting(expectedKeys)
        return SpeechTokenizerWeightLoadResult(
            weights: sanitized,
            missingKeys: missing,
            unexpectedKeys: unexpected
        )
    }

    /// Full pipeline: discover, load, sanitize, and optionally validate.
    ///
    /// When `expectedKeys` is non-empty, the result includes diagnostic
    /// information about missing and unexpected keys. When empty, validation
    /// is skipped.
    static func load(
        from directory: URL,
        expectedKeys: Set<String> = []
    ) throws -> SpeechTokenizerWeightLoadResult {
        let raw = try loadRawWeights(from: directory)
        let sanitized = sanitize(weights: raw)
        if expectedKeys.isEmpty {
            return SpeechTokenizerWeightLoadResult(
                weights: sanitized,
                missingKeys: [],
                unexpectedKeys: []
            )
        }
        return validate(sanitized: sanitized, expectedKeys: expectedKeys)
    }

    /// Filter sanitized weights to decoder-only keys and strip the `decoder.` prefix.
    ///
    /// Raw safetensors keys for the decoder start with `decoder.` (e.g.,
    /// `decoder.quantizer.rvq_first...`). After sanitization they become
    /// `decoder.quantizer.rvqFirst...`. This method strips the leading
    /// `decoder.` so keys match the ``SpeechTokenizerDecoder`` module tree.
    static func decoderWeights(from sanitized: [String: MLXArray]) -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]
        let prefix = "decoder."
        for (key, value) in sanitized {
            if key.hasPrefix(prefix) {
                result[String(key.dropFirst(prefix.count))] = value
            }
        }
        return result
    }

    // MARK: - Shape Validation

    /// Validate expected tensor shapes in decoder weights against the model config.
    ///
    /// Checks a representative set of tensors whose shapes are fully determined
    /// by the config:
    /// - Codebook embedding weights for all `rvqFirst` and `rvqRest` layers.
    /// - `preConv.conv.weight` — 3D in MLX format `[latentDim, kernelSize, codebookDim]`.
    ///   Conv1d weight transpose is only applied when the raw tensor is 3D; this
    ///   check verifies the post-sanitize result matches the expected layout.
    /// - `preTransformer.norm.weight` — 1D `[hiddenSize]`.
    ///
    /// Only tensors that are **present** in `decoderWeights` are checked — missing
    /// keys are handled separately by ``validate(sanitized:expectedKeys:)``.
    ///
    /// - Parameters:
    ///   - decoderWeights: Decoder-only weights with the `decoder.` prefix stripped
    ///     (output of ``decoderWeights(from:)``).
    ///   - config: Decoder configuration supplying expected dimensions.
    /// - Throws: ``SpeechTokenizerWeightLoaderError/shapeMismatch`` if any present
    ///   tensor has an unexpected shape.
    static func validateShapes(
        decoderWeights: [String: MLXArray],
        config: SpeechTokenizerDecoderConfig
    ) throws {
        // codebookDim / 2 is the `dimension` passed to EuclideanCodebook.
        let codebookDim = config.codebookDim / 2
        let expectedCodebookShape = [config.codebookSize, codebookDim]

        // rvqFirst codebook layers (numSemanticQuantizers total)
        for i in 0 ..< config.numSemanticQuantizers {
            let key = "quantizer.rvqFirst.vq.layers.\(i).codebook.embed.weight"
            try checkShape(in: decoderWeights, key: key, expected: expectedCodebookShape)
        }

        // rvqRest codebook layers (numQuantizers - numSemanticQuantizers total)
        let numRest = config.numQuantizers - config.numSemanticQuantizers
        for i in 0 ..< numRest {
            let key = "quantizer.rvqRest.vq.layers.\(i).codebook.embed.weight"
            try checkShape(in: decoderWeights, key: key, expected: expectedCodebookShape)
        }

        // preConv weight — MLX Conv1d format [outChannels, kernelSize, inChannels]
        // (kernelSize = 3 as set in SpeechTokenizerDecoder.init)
        try checkShape(
            in: decoderWeights,
            key: "preConv.conv.weight",
            expected: [config.latentDim, 3, config.codebookDim]
        )

        // Transformer norm weight — 1D [hiddenSize]
        try checkShape(
            in: decoderWeights,
            key: "preTransformer.norm.weight",
            expected: [config.hiddenSize]
        )
    }

    /// Validates that a weight tensor, if present, has the expected shape.
    /// Missing keys are silently skipped (they may be optional or handled elsewhere).
    private static func checkShape(
        in weights: [String: MLXArray],
        key: String,
        expected: [Int]
    ) throws {
        guard let tensor = weights[key] else { return }
        let actual = tensor.shape.map { $0 }
        guard actual == expected else {
            throw SpeechTokenizerWeightLoaderError.shapeMismatch(
                key: key,
                expected: expected,
                actual: actual
            )
        }
    }

    // MARK: - Private Helpers

    /// Check if a 3D weight tensor is already in MLX layout.
    ///
    /// MLX Conv1d format: `[out_channels, kernel_size, in_channels]`
    /// PyTorch Conv1d format: `[out_channels, in_channels, kernel_size]`
    ///
    /// Uses a shape heuristic: kernel_size is typically much smaller than
    /// in_channels, so if `dim(1) < dim(2)` the weight is likely already in
    /// MLX layout (kernel is in the middle position).
    ///
    /// Reference: `qwen3_tts.py` function `check_array_shape_qwen3`.
    private static func isMLXLayout(_ array: MLXArray) -> Bool {
        guard array.ndim == 3 else { return false }
        let dim2 = array.dim(1)
        let dim3 = array.dim(2)

        if dim2 == 1 { return dim3 > 64 }
        if dim3 == 1 { return dim2 <= 64 }
        return dim2 < dim3
    }

    /// Determine if a weight key corresponds to a ConvTranspose1d weight.
    ///
    /// ConvTranspose1d weights are identified by their position in the module
    /// tree:
    /// - Upsample blocks: `upsample.{i}.0.conv.weight` (first layer is CausalTransposeConv1d)
    /// - Decoder blocks: `decoder.decoder.{i}.block.1.conv.weight` (transposed conv within blocks)
    private static func isTransposeConvWeight(key: String) -> Bool {
        (key.contains("upsample") && key.contains(".0.conv.weight")) ||
        (key.contains("decoder.decoder") && key.contains("block.1.conv.weight"))
    }

    /// Remap Python snake_case weight keys to Swift camelCase module paths.
    private static func remapKey(_ key: String) -> String {
        key.replacingOccurrences(of: "rvq_first", with: "rvqFirst")
            .replacingOccurrences(of: "rvq_rest", with: "rvqRest")
            .replacingOccurrences(of: "input_proj", with: "inputProj")
            .replacingOccurrences(of: "output_proj", with: "outputProj")
            .replacingOccurrences(of: "pre_transformer", with: "preTransformer")
            .replacingOccurrences(of: "pre_conv", with: "preConv")
            .replacingOccurrences(of: "encoder_model", with: "encoderModel")
    }
}
