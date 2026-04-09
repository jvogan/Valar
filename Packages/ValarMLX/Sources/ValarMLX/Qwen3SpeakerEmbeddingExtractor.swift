import Foundation
@preconcurrency import MLX
import MLXAudioCore
import MLXNN
import os

// Adapted from the MIT-licensed Qwen3 speaker encoder implementation in
// `Blaizzy/mlx-audio-swift` so Valar can persist speaker embeddings without
// patching the dependency checkout in `.build/checkouts`.

private struct LocalQwen3SpeakerEncoderModelConfig: Decodable, Sendable {
    let speakerEncoderConfig: LocalQwen3SpeakerEncoderConfig?

    private enum CodingKeys: String, CodingKey {
        case speakerEncoderConfig = "speaker_encoder_config"
    }
}

private struct LocalQwen3SpeakerEncoderConfig: Codable, Sendable {
    let melDim: Int
    let encDim: Int
    let encChannels: [Int]
    let encKernelSizes: [Int]
    let encDilations: [Int]
    let encAttentionChannels: Int
    let encRes2netScale: Int
    let encSeChannels: Int
    let sampleRate: Int

    private enum CodingKeys: String, CodingKey {
        case melDim = "mel_dim"
        case encDim = "enc_dim"
        case encChannels = "enc_channels"
        case encKernelSizes = "enc_kernel_sizes"
        case encDilations = "enc_dilations"
        case encAttentionChannels = "enc_attention_channels"
        case encRes2netScale = "enc_res2net_scale"
        case encSeChannels = "enc_se_channels"
        case sampleRate = "sample_rate"
    }

    func validate() throws {
        guard encChannels.count >= 2 else {
            throw SpeakerEncoderError.invalidConfiguration(
                "enc_channels must contain at least 2 values; found \(encChannels.count)."
            )
        }

        let requiredEncoderStageCount = encChannels.count - 1
        guard encKernelSizes.count >= requiredEncoderStageCount else {
            throw SpeakerEncoderError.invalidConfiguration(
                "enc_kernel_sizes must contain at least \(requiredEncoderStageCount) values for \(encChannels.count) enc_channels entries; found \(encKernelSizes.count)."
            )
        }

        guard encDilations.count >= requiredEncoderStageCount else {
            throw SpeakerEncoderError.invalidConfiguration(
                "enc_dilations must contain at least \(requiredEncoderStageCount) values for \(encChannels.count) enc_channels entries; found \(encDilations.count)."
            )
        }
    }
}

private func localReflectPad1D(_ x: MLXArray, pad: Int) -> MLXArray {
    guard pad > 0 else { return x }
    let timeLength = x.dim(1)
    guard timeLength > 1 else { return x }
    let clampedPad = Swift.min(pad, Swift.max(timeLength - 1, 0))
    guard clampedPad > 0 else { return x }

    let left = x[0..., 1 ..< (clampedPad + 1), 0...][0..., .stride(by: -1), 0...]
    let right = x[0..., (-(clampedPad + 1)) ..< (-1), 0...][0..., .stride(by: -1), 0...]
    return concatenated([left, x, right], axis: 1)
}

private func checkLocalQwen3ArrayShape(_ array: MLXArray) -> Bool {
    guard array.ndim == 3 else { return false }
    let dim2 = array.dim(1)
    let dim3 = array.dim(2)

    if dim2 == 1 {
        return dim3 > 64
    }
    if dim3 == 1 {
        return dim2 <= 64
    }
    return dim2 < dim3
}

private final class LocalTimeDelayNetBlock: Module {
    let pad: Int
    @ModuleInfo var conv: Conv1d

    init(inChannels: Int, outChannels: Int, kernelSize: Int, dilation: Int) {
        self.pad = (kernelSize - 1) * dilation / 2
        _conv.wrappedValue = Conv1d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: kernelSize,
            stride: 1,
            padding: 0,
            dilation: dilation
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var output = x.transposed(0, 2, 1)
        output = localReflectPad1D(output, pad: pad)
        output = conv(output)
        return relu(output.transposed(0, 2, 1))
    }
}

private final class LocalRes2NetBlock: Module {
    let scale: Int
    let inChannel: Int
    @ModuleInfo var blocks: [LocalTimeDelayNetBlock]

    init(inChannels: Int, outChannels: Int, scale: Int = 8, kernelSize: Int = 3, dilation: Int = 1) {
        self.scale = scale
        self.inChannel = inChannels / scale

        var builtBlocks: [LocalTimeDelayNetBlock] = []
        builtBlocks.reserveCapacity(max(0, scale - 1))
        for _ in 0 ..< max(0, scale - 1) {
            builtBlocks.append(
                LocalTimeDelayNetBlock(
                    inChannels: self.inChannel,
                    outChannels: outChannels / scale,
                    kernelSize: kernelSize,
                    dilation: dilation
                )
            )
        }
        _blocks.wrappedValue = builtBlocks
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let chunks = split(x, parts: scale, axis: 1)
        var outputs: [MLXArray] = []
        outputs.reserveCapacity(scale)

        var partial: MLXArray?
        for index in 0 ..< scale {
            if index == 0 {
                partial = chunks[index]
            } else if index == 1 {
                partial = blocks[index - 1](chunks[index])
            } else if let previous = partial {
                partial = blocks[index - 1](chunks[index] + previous)
            }

            if let partial {
                outputs.append(partial)
            }
        }

        return concatenated(outputs, axis: 1)
    }
}

private final class LocalSqueezeExcitationBlock: Module {
    @ModuleInfo var conv1: Conv1d
    @ModuleInfo var conv2: Conv1d

    init(inChannels: Int, seChannels: Int, outChannels: Int) {
        _conv1.wrappedValue = Conv1d(inputChannels: inChannels, outputChannels: seChannels, kernelSize: 1, stride: 1, padding: 0)
        _conv2.wrappedValue = Conv1d(inputChannels: seChannels, outputChannels: outChannels, kernelSize: 1, stride: 1, padding: 0)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var squeezed = mean(x, axis: 2, keepDims: true)
        squeezed = squeezed.transposed(0, 2, 1)
        squeezed = relu(conv1(squeezed))
        squeezed = sigmoid(conv2(squeezed))
        squeezed = squeezed.transposed(0, 2, 1)
        return x * squeezed
    }
}

private final class LocalSqueezeExcitationRes2NetBlock: Module {
    @ModuleInfo var tdnn1: LocalTimeDelayNetBlock
    @ModuleInfo var res2netBlock: LocalRes2NetBlock
    @ModuleInfo var tdnn2: LocalTimeDelayNetBlock
    @ModuleInfo var seBlock: LocalSqueezeExcitationBlock

    init(
        inChannels: Int,
        outChannels: Int,
        res2netScale: Int = 8,
        seChannels: Int = 128,
        kernelSize: Int = 3,
        dilation: Int = 1
    ) {
        _tdnn1.wrappedValue = LocalTimeDelayNetBlock(inChannels: inChannels, outChannels: outChannels, kernelSize: 1, dilation: 1)
        _res2netBlock.wrappedValue = LocalRes2NetBlock(inChannels: outChannels, outChannels: outChannels, scale: res2netScale, kernelSize: kernelSize, dilation: dilation)
        _tdnn2.wrappedValue = LocalTimeDelayNetBlock(inChannels: outChannels, outChannels: outChannels, kernelSize: 1, dilation: 1)
        _seBlock.wrappedValue = LocalSqueezeExcitationBlock(inChannels: outChannels, seChannels: seChannels, outChannels: outChannels)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = x
        var output = tdnn1(x)
        output = res2netBlock(output)
        output = tdnn2(output)
        output = seBlock(output)
        return output + residual
    }
}

private final class LocalAttentiveStatisticsPooling: Module {
    let eps: Float = 1e-12
    @ModuleInfo var tdnn: LocalTimeDelayNetBlock
    @ModuleInfo var conv: Conv1d

    init(channels: Int, attentionChannels: Int = 128) {
        _tdnn.wrappedValue = LocalTimeDelayNetBlock(inChannels: channels * 3, outChannels: attentionChannels, kernelSize: 1, dilation: 1)
        _conv.wrappedValue = Conv1d(inputChannels: attentionChannels, outputChannels: channels, kernelSize: 1, stride: 1, padding: 0)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let batch = x.dim(0)
        let channels = x.dim(1)
        let sequenceLength = x.dim(2)

        let meanTensor = mean(x, axis: 2, keepDims: true)
        let centered = x - meanTensor
        let stdTensor = sqrt(mean(centered * centered, axis: 2, keepDims: true) + eps)

        let meanExpanded = broadcast(meanTensor, to: [batch, channels, sequenceLength])
        let stdExpanded = broadcast(stdTensor, to: [batch, channels, sequenceLength])

        var attention = concatenated([x, meanExpanded, stdExpanded], axis: 1)
        attention = tdnn(attention)
        attention = tanh(attention)
        attention = conv(attention.transposed(0, 2, 1)).transposed(0, 2, 1)
        attention = softmax(attention, axis: 2)

        let meanOut = sum(attention * x, axis: 2, keepDims: true)
        let varOut = sum(attention * (x - meanOut) * (x - meanOut), axis: 2, keepDims: true)
        let stdOut = sqrt(clip(varOut, min: eps))
        return concatenated([meanOut, stdOut], axis: 1)
    }
}

private final class LocalQwen3SpeakerEncoder: Module {
    @ModuleInfo var blocks: [Module]
    @ModuleInfo var mfa: LocalTimeDelayNetBlock
    @ModuleInfo var asp: LocalAttentiveStatisticsPooling
    @ModuleInfo var fc: Conv1d

    init(config: LocalQwen3SpeakerEncoderConfig) throws {
        try config.validate()

        var builtBlocks: [Module] = []
        builtBlocks.append(
            LocalTimeDelayNetBlock(
                inChannels: config.melDim,
                outChannels: config.encChannels[0],
                kernelSize: config.encKernelSizes[0],
                dilation: config.encDilations[0]
            )
        )

        if config.encChannels.count > 1 {
            for index in 1 ..< config.encChannels.count - 1 {
                builtBlocks.append(
                    LocalSqueezeExcitationRes2NetBlock(
                        inChannels: config.encChannels[index - 1],
                        outChannels: config.encChannels[index],
                        res2netScale: config.encRes2netScale,
                        seChannels: config.encSeChannels,
                        kernelSize: config.encKernelSizes[index],
                        dilation: config.encDilations[index]
                    )
                )
            }
        }

        _blocks.wrappedValue = builtBlocks
        _mfa.wrappedValue = LocalTimeDelayNetBlock(
            inChannels: config.encChannels.last ?? 1,
            outChannels: config.encChannels.last ?? 1,
            kernelSize: config.encKernelSizes.last ?? 1,
            dilation: config.encDilations.last ?? 1
        )
        _asp.wrappedValue = LocalAttentiveStatisticsPooling(
            channels: config.encChannels.last ?? 1,
            attentionChannels: config.encAttentionChannels
        )
        _fc.wrappedValue = Conv1d(
            inputChannels: (config.encChannels.last ?? 1) * 2,
            outputChannels: config.encDim,
            kernelSize: 1,
            stride: 1,
            padding: 0
        )
    }

    func callAsFunction(_ x: MLXArray) throws -> MLXArray {
        var states = x.transposed(0, 2, 1)
        var hiddenStates: [MLXArray] = []

        for block in blocks {
            if let tdnn = block as? LocalTimeDelayNetBlock {
                states = tdnn(states)
            } else if let seRes2Net = block as? LocalSqueezeExcitationRes2NetBlock {
                states = seRes2Net(states)
            } else {
                throw SpeakerEncoderError.unsupportedBlockType(String(describing: type(of: block)))
            }
            hiddenStates.append(states)
        }

        if hiddenStates.count >= 2 {
            states = concatenated(Array(hiddenStates[1...]), axis: 1)
        }

        states = mfa(states)
        states = asp(states)
        states = fc(states.transposed(0, 2, 1)).transposed(0, 2, 1)
        return states.squeezed(axis: -1)
    }

    static func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]

        for (key, originalValue) in weights {
            guard let sanitizedKey = stripSpeakerEncoderPrefix(from: key), !sanitizedKey.isEmpty else {
                continue
            }

            var value = originalValue
            if sanitizedKey.hasSuffix(".weight"), value.ndim == 3, !checkLocalQwen3ArrayShape(value) {
                value = value.transposed(0, 2, 1)
            }
            sanitized[sanitizedKey] = value
        }

        return sanitized
    }

    private static func stripSpeakerEncoderPrefix(from key: String) -> String? {
        let parts = key.split(separator: ".")
        guard let markerIndex = parts.firstIndex(of: "speaker_encoder") else { return nil }
        let suffixParts = parts[(markerIndex + 1)...]
        guard !suffixParts.isEmpty else { return nil }
        return suffixParts.joined(separator: ".")
    }
}

private enum SpeakerEncoderError: LocalizedError {
    case unsupportedBlockType(String)
    case invalidConfiguration(String)

    var errorDescription: String? {
        switch self {
        case .unsupportedBlockType(let typeName):
            return "Unsupported speaker encoder block type: \(typeName)."
        case .invalidConfiguration(let reason):
            return "Invalid Qwen3 speaker encoder config: \(reason)"
        }
    }
}

// Process-local encoder cache. Keyed by standardized model directory URL so that
// repeated extractions from the same model skip weight loading entirely.
// OSAllocatedUnfairLock is declared @unchecked Sendable by the OS framework,
// satisfying Swift 6's requirement that static-let storage be Sendable.
private struct CachedSpeakerEncoder: Sendable {
    // LocalQwen3SpeakerEncoder (an MLXNN Module subclass) is not declared Sendable, but
    // every CachedSpeakerEncoder value is immutable after construction and all access is
    // serialized through OSAllocatedUnfairLock. The unsafe annotation opts out of the
    // automatic Sendable check for this one property while the struct's other fields are
    // statically verified.
    nonisolated(unsafe) let encoder: LocalQwen3SpeakerEncoder
    let sampleRate: Int
}

private let encoderCache = OSAllocatedUnfairLock(initialState: [URL: CachedSpeakerEncoder]())

enum Qwen3SpeakerEmbeddingExtractor {
    static func extract(from modelDirectory: URL, monoSamples: [Float]) throws -> [Float] {
        let cacheKey = modelDirectory.standardizedFileURL

        if let cached = encoderCache.withLock({ $0[cacheKey] }) {
            return try runInference(cached, monoSamples: monoSamples)
        }

        let cached = try loadEncoder(from: modelDirectory)
        encoderCache.withLock { $0[cacheKey] = cached }
        return try runInference(cached, monoSamples: monoSamples)
    }

    /// Removes the cached encoder for the given model directory.
    /// Called by `MLXInferenceBackend` when a model is unloaded.
    static func evictEncoder(at modelDirectory: URL) {
        let cacheKey = modelDirectory.standardizedFileURL
        _ = encoderCache.withLock { $0.removeValue(forKey: cacheKey) }
    }

    /// Number of currently cached encoders. Exposed for testing.
    static var _encoderCacheCount: Int {
        encoderCache.withLock { $0.count }
    }
}

private extension Qwen3SpeakerEmbeddingExtractor {
    static func loadEncoder(from modelDirectory: URL) throws -> CachedSpeakerEncoder {
        let resolvedModelDirectory = modelDirectory.standardized.resolvingSymlinksInPath()
        let configData = try Data(contentsOf: resolvedModelDirectory.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(LocalQwen3SpeakerEncoderModelConfig.self, from: configData)
        guard let speakerEncoderConfig = config.speakerEncoderConfig else {
            throw MLXBackendError.inferenceError(
                "Model at \(modelDirectory.lastPathComponent) does not contain a speaker encoder. "
                + "Use the Base model for voice cloning."
            )
        }
        let encoder = try LocalQwen3SpeakerEncoder(config: speakerEncoderConfig)

        let fileManager = FileManager.default
        let modelFiles = try fileManager.contentsOfDirectory(at: resolvedModelDirectory, includingPropertiesForKeys: nil)
        var allWeights: [String: MLXArray] = [:]
        let rootPrefix = resolvedModelDirectory.path.hasSuffix("/")
            ? resolvedModelDirectory.path
            : "\(resolvedModelDirectory.path)/"
        for file in modelFiles where file.pathExtension == "safetensors" {
            let resolvedFile = try resolveWeightFile(file, rootPrefix: rootPrefix)
            let weights = try MLX.loadArrays(url: resolvedFile)
            allWeights.merge(weights) { _, newest in newest }
        }

        let speakerWeights = LocalQwen3SpeakerEncoder.sanitize(weights: allWeights)
        guard !speakerWeights.isEmpty else {
            throw MLXBackendError.inferenceError("Speaker encoder weights were not found in \(resolvedModelDirectory.path).")
        }

        let speakerPairs = speakerWeights.map { ($0.key, $0.value) }
        try encoder.update(parameters: ModuleParameters.unflattened(speakerPairs), verify: .all)
        // Note: eval() here is MLX's native array evaluation function, not JavaScript eval
        eval(encoder.parameters())

        return CachedSpeakerEncoder(encoder: encoder, sampleRate: speakerEncoderConfig.sampleRate)
    }

    static func resolveWeightFile(_ file: URL, rootPrefix: String) throws -> URL {
        // Resolve symlinks before loading so a malicious bundle cannot redirect a
        // `.safetensors` entry to an arbitrary path outside the model directory.
        let resolvedFile = file.resolvingSymlinksInPath()
        guard resolvedFile.path.hasPrefix(rootPrefix) else {
            throw MLXBackendError.pathTraversalDetected(file.path)
        }
        return resolvedFile
    }

    static func runInference(_ cached: CachedSpeakerEncoder, monoSamples: [Float]) throws -> [Float] {
        let waveform = MLXArray(monoSamples)
        let mel = computeMelSpectrogram(
            audio: waveform,
            sampleRate: cached.sampleRate,
            nFft: 1024,
            hopLength: 256,
            nMels: 128
        )
        let embedding = try cached.encoder(stacked([mel], axis: 0))
        eval(embedding)
        return embedding.squeezed().asArray(Float.self)
    }
}
