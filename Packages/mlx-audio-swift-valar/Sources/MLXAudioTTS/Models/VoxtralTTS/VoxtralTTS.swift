import Foundation
import HuggingFace
@preconcurrency import MLX
import MLXAudioCore
@preconcurrency import MLXLMCommon
import MLXNN

private struct VoxtralTTSVoiceManifest: Decodable {
    struct Entry: Decodable {
        let voiceName: String
        let file: String
        let shape: [Int]
        let frameCount: Int

        enum CodingKeys: String, CodingKey {
            case voiceName = "voice_name"
            case file
            case shape
            case frameCount = "frame_count"
        }
    }

    let format: String
    let expectedDim: Int
    let voiceCount: Int
    let voices: [Entry]

    enum CodingKeys: String, CodingKey {
        case format
        case expectedDim = "expected_dim"
        case voiceCount = "voice_count"
        case voices
    }
}

private enum VoxtralTTSStreamingStrategy {
    static let frameRate: Double = 12.5
    static let firstChunkFrames = 5
    static let leftContextFrames = 25
    static let samplesPerFrame = 1920

    static func resolveInterval(
        from generationParameters: GenerateParameters,
        fallback: Double
    ) -> Double {
        let reflectedValue = reflectedNumericValue(
            in: generationParameters,
            matching: ["streamingInterval", "audioStreamingInterval", "ttsStreamingInterval"]
        )
        let interval = reflectedValue ?? fallback
        return max(1.0 / frameRate, interval)
    }

    static func regularChunkFrames(for streamingInterval: Double) -> Int {
        max(1, Int(streamingInterval * frameRate))
    }

    private static func reflectedNumericValue(
        in value: Any,
        matching labels: Set<String>,
        depth: Int = 0
    ) -> Double? {
        guard depth <= 3 else { return nil }

        let mirror = Mirror(reflecting: value)
        if mirror.displayStyle == .optional {
            guard let child = mirror.children.first else { return nil }
            return reflectedNumericValue(in: child.value, matching: labels, depth: depth)
        }

        for child in mirror.children {
            if let label = child.label, labels.contains(label), let numericValue = numericValue(from: child.value) {
                return numericValue
            }
            if let nestedValue = reflectedNumericValue(in: child.value, matching: labels, depth: depth + 1) {
                return nestedValue
            }
        }

        return nil
    }

    private static func numericValue(from value: Any) -> Double? {
        switch value {
        case let value as Double:
            return value
        case let value as Float:
            return Double(value)
        case let value as Int:
            return Double(value)
        case let value as NSNumber:
            return value.doubleValue
        default:
            return nil
        }
    }
}

struct VoxtralTTSCheckpointSnapshot {
    let promptEmbeddings: MLXArray
    let prefillHidden: MLXArray
    let initialAudioHidden: MLXArray
    let semanticLogits: MLXArray
}

final class VoxtralTTSAudioCodebookEmbeddings: Module {
    enum Layout: Sendable {
        case flattened(totalRows: Int)
        case codebookwise(maxCodebookSize: Int)
    }

    var weight: MLXArray
    var bias: MLXArray
    let semanticVocabularySize: Int
    let acousticVocabularySize: Int

    init(
        numCodebooks: Int,
        semanticVocabularySize: Int,
        acousticVocabularySize: Int,
        maxCodebookSize: Int,
        layout: Layout,
        dim: Int
    ) {
        switch layout {
        case let .flattened(totalRows):
            weight = MLXArray.zeros([totalRows, dim], dtype: .float32)
        case let .codebookwise(maxCodebookSize):
            weight = MLXArray.zeros([numCodebooks, maxCodebookSize, dim], dtype: .float32)
        }
        bias = MLXArray.zeros([numCodebooks, dim], dtype: .float32)
        self.semanticVocabularySize = semanticVocabularySize
        self.acousticVocabularySize = acousticVocabularySize
    }

    func callAsFunction(_ codes: MLXArray) -> MLXArray {
        let codeMatrix = codes.ndim == 1 ? codes.expandedDimensions(axis: 0) : codes
        let batch = codeMatrix.dim(0)
        let codebookCount = min(codeMatrix.dim(1), bias.dim(0))

        if weight.ndim == 2 {
            var output = MLXArray.zeros([batch, weight.dim(1)], dtype: weight.dtype)
            var offset = 0

            for codebookIndex in 0..<codebookCount {
                let vocabularySize = codebookIndex == 0 ? semanticVocabularySize : acousticVocabularySize
                let ids = clip(
                    codeMatrix[0..., codebookIndex],
                    min: 0,
                    max: vocabularySize - 1
                ).asType(.int32) + MLXArray(Int32(offset))
                let emb = take(weight, ids, axis: 0)
                output = output + emb
                offset += vocabularySize
            }

            return output
        }

        let maxCode = weight.dim(1) - 1
        var output = MLXArray.zeros([batch, weight.dim(2)], dtype: weight.dtype)

        for codebookIndex in 0..<codebookCount {
            let table = weight[codebookIndex]
            let ids = clip(codeMatrix[0..., codebookIndex], min: 0, max: maxCode).asType(.int32)
            let emb = take(table, ids, axis: 0) + bias[codebookIndex].reshaped(1, -1)
            output = output + emb
        }

        return output
    }
}

final class VoxtralTTSAudioTokenEmbedding: Module {
    @ModuleInfo(key: "embeddings") var embeddings: VoxtralTTSAudioCodebookEmbeddings

    init(
        numCodebooks: Int,
        semanticVocabularySize: Int,
        acousticVocabularySize: Int,
        maxCodebookSize: Int,
        layout: VoxtralTTSAudioCodebookEmbeddings.Layout,
        dim: Int
    ) {
        _embeddings.wrappedValue = VoxtralTTSAudioCodebookEmbeddings(
            numCodebooks: numCodebooks,
            semanticVocabularySize: semanticVocabularySize,
            acousticVocabularySize: acousticVocabularySize,
            maxCodebookSize: maxCodebookSize,
            layout: layout,
            dim: dim
        )
    }

    func callAsFunction(_ codes: MLXArray) -> MLXArray {
        embeddings(codes)
    }
}

public final class VoxtralTTSModel: Module, SpeechGenerationModel, @unchecked Sendable {
    public let config: VoxtralTTSConfig
    let codecVariant: VoxtralTTSCodecVariant

    @ModuleInfo(key: "backbone") var backbone: VoxtralTTSBackbone
    @ModuleInfo(key: "audio_token_embedding") var audioTokenEmbedding: VoxtralTTSAudioTokenEmbedding
    @ModuleInfo(key: "acoustic_transformer") var acousticTransformer: VoxtralTTSAcousticTransformer
    @ModuleInfo(key: "audio_tokenizer") var audioTokenizer: VoxtralTTSCodecDecoder

    var tokenizer: VoxtralTTSTokenizer?
    var presetVoices: [String: MLXArray] = [:]

    public var sampleRate: Int { config.sampleRate }

    public var defaultGenerationParameters: GenerateParameters {
        GenerateParameters(
            maxTokens: 8192,
            temperature: 0.8,
            topP: 0.95,
            repetitionPenalty: nil
        )
    }

    public convenience init(config: VoxtralTTSConfig, codecVariant: VoxtralTTSCodecVariant = .legacy) {
        self.init(
            config: config,
            codecVariant: codecVariant,
            audioEmbeddingLayout: .flattened(
                totalRows: max(
                    config.audioModelArgs.semanticCodebookSize + 2
                        + (config.audioModelArgs.nAcousticCodebook * (config.audioModelArgs.acousticCodebookSize + 2)),
                    config.totalCodebooks
                )
            )
        )
    }

    init(
        config: VoxtralTTSConfig,
        codecVariant: VoxtralTTSCodecVariant,
        audioEmbeddingLayout: VoxtralTTSAudioCodebookEmbeddings.Layout
    ) {
        self.config = config
        self.codecVariant = codecVariant

        _backbone.wrappedValue = VoxtralTTSBackbone(config.backbone)
        _audioTokenEmbedding.wrappedValue = VoxtralTTSAudioTokenEmbedding(
            numCodebooks: config.totalCodebooks,
            semanticVocabularySize: config.audioModelArgs.semanticCodebookSize + 2,
            acousticVocabularySize: config.audioModelArgs.acousticCodebookSize + 2,
            maxCodebookSize: max(
                config.audioModelArgs.semanticCodebookSize + 2,
                config.audioModelArgs.acousticCodebookSize + 2
            ),
            layout: audioEmbeddingLayout,
            dim: config.backbone.dim
        )
        _acousticTransformer.wrappedValue = VoxtralTTSAcousticTransformer(
            config: config.audioModelArgs.acousticTransformerArgs,
            semanticCodebookSize: config.audioModelArgs.semanticCodebookSize,
            acousticCodebookSize: config.audioModelArgs.acousticCodebookSize,
            nAcousticCodebook: config.audioModelArgs.nAcousticCodebook
        )
        _audioTokenizer.wrappedValue = VoxtralTTSCodecDecoder(
            config: config.audioTokenizerArgs,
            variant: codecVariant
        )
    }

    public func generate(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) async throws -> MLXArray {
        _ = language
        let frames = try generateFrames(
            text: text,
            voice: voice,
            refAudio: refAudio,
            refText: refText,
            generationParameters: generationParameters
        )
        return decodeFrames(frames)
    }

    func debugCheckpointSnapshot(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?
    ) throws -> VoxtralTTSCheckpointSnapshot {
        guard refAudio == nil, refText == nil else {
            throw AudioGenerationError.invalidInput("VoxtralTTS currently supports preset voices only")
        }
        guard let tokenizer else {
            throw AudioGenerationError.modelNotInitialized("Tokenizer not loaded")
        }

        let resolvedVoiceName = Self.resolvedPresetVoiceName(voice)
        guard let voiceEmbedding = presetVoices[resolvedVoiceName] else {
            throw AudioGenerationError.invalidInput("Unknown Voxtral preset voice: \(resolvedVoiceName)")
        }

        let promptEmbeddings = try preparePromptEmbeddings(
            text: text,
            tokenizer: tokenizer,
            voiceEmbedding: voiceEmbedding
        )
        let promptLen = promptEmbeddings.dim(0)

        var currentCache: [VoxtralTTSBackboneKVCache?]? = nil
        var prefillHidden = MLXArray.zeros([1, config.backbone.dim], dtype: .float32)
        (prefillHidden, currentCache) = backbone(promptEmbeddings, startPos: 0, cache: currentCache)

        let audioTokenIds = MLXArray([Int32(config.audioTokenId)])
        let audioTokenInput = backbone.embedTokens(audioTokenIds)
        var initialAudioHidden = MLXArray.zeros([1, config.backbone.dim], dtype: .float32)
        (initialAudioHidden, currentCache) = backbone(audioTokenInput, startPos: promptLen, cache: currentCache)

        let lastHidden = initialAudioHidden[(initialAudioHidden.dim(0) - 1)...]
        let semanticLogits = acousticTransformer.semanticCodebookOutput(lastHidden)

        return VoxtralTTSCheckpointSnapshot(
            promptEmbeddings: promptEmbeddings,
            prefillHidden: prefillHidden,
            initialAudioHidden: initialAudioHidden,
            semanticLogits: semanticLogits
        )
    }

    public func generateStream(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        generateStream(
            text: text,
            voice: voice,
            refAudio: refAudio,
            refText: refText,
            language: language,
            generationParameters: generationParameters,
            streamingInterval: VoxtralTTSStreamingStrategy.resolveInterval(
                from: generationParameters,
                fallback: 2.0
            )
        )
    }

    public func generateStream(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters,
        streamingInterval: Double
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        let (stream, continuation) = AsyncThrowingStream<AudioGeneration, Error>.makeStream()
        let text = text
        let voice = voice
        let refText = refText
        let generationParameters = generationParameters
        let resolvedStreamingInterval = VoxtralTTSStreamingStrategy.resolveInterval(
            from: generationParameters,
            fallback: streamingInterval
        )
        let task = Task { @Sendable [weak self, continuation] in
            guard let self else {
                continuation.finish()
                return
            }
            do {
                let regularChunkFrames = VoxtralTTSStreamingStrategy.regularChunkFrames(
                    for: resolvedStreamingInterval
                )
                var streamedFrames: [MLXArray] = []
                streamedFrames.reserveCapacity(generationParameters.maxTokens ?? regularChunkFrames)
                var emittedFrameCount = 0
                var nextChunkTarget = VoxtralTTSStreamingStrategy.firstChunkFrames

                func emitPendingChunk(upTo frameCount: Int) {
                    guard frameCount > emittedFrameCount else { return }
                    let chunk = self.decodeStreamingChunk(
                        from: streamedFrames,
                        emittedRange: emittedFrameCount ..< frameCount,
                        leftContextFrames: VoxtralTTSStreamingStrategy.leftContextFrames
                    )
                    emittedFrameCount = frameCount
                    nextChunkTarget = regularChunkFrames
                    continuation.yield(.audio(chunk))
                }

                _ = try self.generateFrames(
                    text: text,
                    voice: voice,
                    refAudio: refAudio,
                    refText: refText,
                    generationParameters: generationParameters
                ) { frame in
                    streamedFrames.append(frame)
                    eval(frame) // materialize MLX lazy graph to prevent unbounded accumulation
                    continuation.yield(.codes(frame))
                    if streamedFrames.count - emittedFrameCount >= nextChunkTarget {
                        emitPendingChunk(upTo: streamedFrames.count)
                    }
                }

                if streamedFrames.isEmpty {
                    continuation.yield(.audio(MLXArray.zeros([0], dtype: .float32)))
                } else if streamedFrames.count > emittedFrameCount {
                    emitPendingChunk(upTo: streamedFrames.count)
                }
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }
        continuation.onTermination = { @Sendable _ in task.cancel() }
        return stream
    }

    public func generateCodeStream(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) -> AsyncThrowingStream<MLXArray, Error> {
        _ = language
        let (stream, continuation) = AsyncThrowingStream<MLXArray, Error>.makeStream()
        let text = text
        let voice = voice
        let refText = refText
        let generationParameters = generationParameters
        let codeTask = Task { @Sendable [weak self, continuation] in
            guard let self else {
                continuation.finish()
                return
            }
            do {
                _ = try self.generateFrames(
                    text: text,
                    voice: voice,
                    refAudio: refAudio,
                    refText: refText,
                    generationParameters: generationParameters
                ) { frame in
                    continuation.yield(frame)
                }
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }
        continuation.onTermination = { @Sendable _ in codeTask.cancel() }
        return stream
    }

    func generateFrames(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        generationParameters: GenerateParameters,
        onFrame: ((MLXArray) -> Void)? = nil
    ) throws -> [MLXArray] {
        guard refAudio == nil, refText == nil else {
            throw AudioGenerationError.invalidInput("VoxtralTTS currently supports preset voices only")
        }
        guard let tokenizer else {
            throw AudioGenerationError.modelNotInitialized("Tokenizer not loaded")
        }

        let resolvedVoiceName = Self.resolvedPresetVoiceName(voice)
        guard let voiceEmbedding = presetVoices[resolvedVoiceName] else {
            throw AudioGenerationError.invalidInput("Unknown Voxtral preset voice: \(resolvedVoiceName)")
        }

        let promptEmbeddings = try preparePromptEmbeddings(
            text: text,
            tokenizer: tokenizer,
            voiceEmbedding: voiceEmbedding
        )
        let promptLen = promptEmbeddings.dim(0)

        // Prefill: run full prompt through backbone with KV cache
        var currentCache: [VoxtralTTSBackboneKVCache?]? = nil
        var hidden = MLXArray.zeros([1, config.backbone.dim], dtype: .float32)
        (hidden, currentCache) = backbone(promptEmbeddings, startPos: 0, cache: currentCache)

        // Initial decode step: feed a plain [AUDIO] token embedding through the backbone.
        // This matches the Python reference and transitions from prompt mode to generation mode,
        // preserving voice conditioning in the hidden state.
        let audioTokenIds = MLXArray([Int32(config.audioTokenId)])
        let audioTokenInput = backbone.embedTokens(audioTokenIds) // [1, dim]
        (hidden, currentCache) = backbone(audioTokenInput, startPos: promptLen, cache: currentCache)

        let maxFrames = generationParameters.maxTokens ?? 256
        var frames: [MLXArray] = []
        frames.reserveCapacity(maxFrames)

        // AR steps start at promptLen + 1 (after the initial decode step above)
        for step in 0..<maxFrames {
            try Task.checkCancellation()
            let lastHidden = hidden[(hidden.dim(0) - 1)...]
            let framePrediction = acousticTransformer.predictFrame(from: lastHidden)
            if framePrediction.isEOS {
                break
            }

            let frame = framePrediction.codes.asType(.int32)
            frames.append(frame)
            onFrame?(frame)

            let nextEmbedding = audioTokenEmbedding(frame)
            let next = backbone(
                nextEmbedding,
                startPos: promptLen + 1 + step,
                cache: currentCache
            )
            hidden = next.0
            currentCache = next.1
        }

        return frames
    }

    func preparePromptEmbeddings(
        text: String,
        tokenizer: VoxtralTTSTokenizer,
        voiceEmbedding: MLXArray
    ) throws -> MLXArray {
        let frameCount = voiceEmbedding.dim(0)
        let promptIds = tokenizer.packSpeechRequest(text: text, voiceFrameCount: frameCount)
        guard promptIds.count >= 2 + frameCount else {
            throw AudioGenerationError.invalidInput("Malformed Voxtral prompt packing")
        }

        let prefixIds = Array(promptIds.prefix(2))
        let suffixIds = Array(promptIds.dropFirst(2 + frameCount))

        let prefix = backbone.embedTokens(MLXArray(prefixIds.map(Int32.init)))
        let suffix = suffixIds.isEmpty
            ? MLXArray.zeros([0, config.backbone.dim], dtype: prefix.dtype)
            : backbone.embedTokens(MLXArray(suffixIds.map(Int32.init)))
        return concatenated([prefix, voiceEmbedding.asType(prefix.dtype), suffix], axis: 0)
    }

    func decodeFrames(_ frames: [MLXArray]) -> MLXArray {
        guard !frames.isEmpty else {
            return MLXArray.zeros([0], dtype: .float32)
        }

        let stackedFrames = stacked(frames.map { $0.squeezed(axis: 0) }, axis: 0)
        let codes = stackedFrames.transposed(1, 0).expandedDimensions(axis: 0)
        let audio = audioTokenizer.decode(codes).squeezed()
        eval(audio)
        return audio
    }

    func decodeStreamingChunk(
        from frames: [MLXArray],
        emittedRange: Range<Int>,
        leftContextFrames: Int
    ) -> MLXArray {
        guard !emittedRange.isEmpty else {
            return MLXArray.zeros([0], dtype: .float32)
        }

        let decodeStart = max(0, emittedRange.lowerBound - leftContextFrames)
        let decoded = decodeFrames(Array(frames[decodeStart ..< emittedRange.upperBound]))
        let contextFrameCount = emittedRange.lowerBound - decodeStart
        let contextSamples = contextFrameCount * VoxtralTTSStreamingStrategy.samplesPerFrame

        guard contextSamples > 0 else {
            return decoded
        }
        guard contextSamples < decoded.dim(0) else {
            return MLXArray.zeros([0], dtype: .float32)
        }

        let trimmed = decoded[contextSamples...]
        eval(trimmed)
        return trimmed
    }

    public static func fromPretrained(
        _ modelRepo: String,
        cache: HubCache = .default
    ) async throws -> VoxtralTTSModel {
        let fileManager = FileManager.default
        if fileManager.fileExists(atPath: modelRepo) {
            return try fromDirectory(URL(fileURLWithPath: modelRepo))
        }

        guard let repoID = Repo.ID(rawValue: modelRepo) else {
            throw AudioGenerationError.modelNotInitialized("Invalid model repository ID: \(modelRepo)")
        }

        let modelDir = try await ModelUtils.resolveOrDownloadModel(
            repoID: repoID,
            requiredExtension: "safetensors",
            additionalMatchingPatterns: [
                "params.json",
                "tekken.json",
                "voice_embedding_safe/*.bin",
                "voice_embedding_safe/*.json",
                "voice_embedding/*.safetensors",
            ],
            cache: cache
        )
        return try fromDirectory(modelDir)
    }

    public static func fromDirectory(_ modelDir: URL) throws -> VoxtralTTSModel {
        let paramsURL = modelDir.appendingPathComponent("params.json")
        guard FileManager.default.fileExists(atPath: paramsURL.path) else {
            throw AudioGenerationError.modelNotInitialized("Missing Voxtral params.json at \(paramsURL.path)")
        }

        let configData: Data
        do {
            configData = try Data(contentsOf: paramsURL)
        } catch {
            throw AudioGenerationError.modelNotInitialized(
                "Unable to read Voxtral params.json at \(paramsURL.path): \(error.localizedDescription)"
            )
        }

        let config: VoxtralTTSConfig
        do {
            config = try JSONDecoder().decode(VoxtralTTSConfig.self, from: configData)
        } catch {
            throw AudioGenerationError.modelNotInitialized(
                "Invalid Voxtral params.json at \(paramsURL.path): \(error.localizedDescription)"
            )
        }
        var weights: [String: MLXArray] = [:]
        let files = try FileManager.default.contentsOfDirectory(at: modelDir, includingPropertiesForKeys: nil)
        for file in files where file.pathExtension == "safetensors" {
            let shard = try MLX.loadArrays(url: file)
            weights.merge(shard) { _, new in new }
        }

        let codecVariant = inferredCodecVariant(from: weights)
        let audioEmbeddingLayout = resolvedAudioEmbeddingLayout(from: weights, config: config)
        let model = VoxtralTTSModel(
            config: config,
            codecVariant: codecVariant,
            audioEmbeddingLayout: audioEmbeddingLayout
        )
        model.tokenizer = try VoxtralTTSTokenizer.fromModelDirectory(
            modelDir,
            audioTokenId: config.audioTokenId,
            beginAudioTokenId: config.beginAudioTokenId
        )

        var sanitized = sanitize(weights: weights, codecVariant: codecVariant)
        if sanitized["audio_token_embedding.embeddings.weight"] != nil,
           sanitized["audio_token_embedding.embeddings.bias"] == nil {
            sanitized["audio_token_embedding.embeddings.bias"] = MLXArray.zeros(
                [config.totalCodebooks, config.backbone.dim],
                dtype: .float32
            )
        }
        if let quantization = inferredQuantization(from: modelDir, weights: sanitized) {
            quantize(model: model, groupSize: quantization.groupSize, bits: quantization.bits) { path, _ in
                sanitized["\(path).scales"] != nil
            }
        }
        let generatedParameters = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        for (key, value) in generatedParameters where sanitized[key] == nil && key.hasSuffix(".alibiSlopes") {
            sanitized[key] = value
        }
        let expectedKeys = Set(model.parameters().flattened().map(\.0))
        let missingKeys = VoxtralTTSModel.missingParameterKeys(
            expected: expectedKeys,
            provided: Set(sanitized.keys)
        )
        guard missingKeys.isEmpty else {
            throw AudioGenerationError.modelNotInitialized(
                "Voxtral loader missing required model parameters: \(missingKeys.joined(separator: ", "))"
            )
        }

        try model.update(parameters: ModuleParameters.unflattened(sanitized), verify: .all)
        eval(model.parameters())
        model.presetVoices = try loadPresetVoices(from: modelDir, expectedDim: config.backbone.dim)
        return model
    }

    static func resolvedPresetVoiceName(_ voice: String?) -> String {
        let normalized = (voice ?? "neutral_female")
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        let aliases: [String: String] = [
            "emma": "neutral_female",
            "default_f": "neutral_female",
            "alex": "neutral_male",
            "default_m": "neutral_male",
            "lily": "cheerful_female",
            "happy_f": "cheerful_female",
            "pedro": "pt_male",
            "claire": "fr_female",
            "carlos": "es_male",
            "lena": "de_female",
            "priya": "hi_female",
        ]
        return aliases[normalized] ?? normalized
    }

    internal static func missingParameterKeys(
        expected expectedKeys: Set<String>,
        provided providedKeys: Set<String>
    ) -> [String] {
        expectedKeys
            .subtracting(providedKeys)
            .filter { $0.hasSuffix(".alibiSlopes") == false }
            .sorted()
    }

    static func inferredCodecVariant(from weights: [String: MLXArray]) -> VoxtralTTSCodecVariant {
        if weights.keys.contains(where: {
            $0.hasPrefix("audio_tokenizer.decoder_blocks.")
                || $0 == "audio_tokenizer.quantizer.semantic_codebook.embedding_sum"
                || $0 == "audio_tokenizer.quantizer.semantic_codebook.cluster_usage"
        }) {
            return .community
        }
        return .legacy
    }

    static func resolvedAudioEmbeddingLayout(
        from weights: [String: MLXArray],
        config: VoxtralTTSConfig
    ) -> VoxtralTTSAudioCodebookEmbeddings.Layout {
        let candidateKeys = [
            "audio_codebook_embeddings.embeddings.weight",
            "mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight",
            "audio_token_embedding.embeddings.weight"
        ]
        for key in candidateKeys {
            guard let weight = weights[key] else { continue }
            if weight.ndim == 2 {
                return .flattened(totalRows: weight.dim(0))
            }
            if weight.ndim == 3 {
                return .codebookwise(maxCodebookSize: weight.dim(1))
            }
        }

        let fallbackRows = max(
            config.audioModelArgs.semanticCodebookSize + 2
                + (config.audioModelArgs.nAcousticCodebook * (config.audioModelArgs.acousticCodebookSize + 2)),
            config.totalCodebooks
        )
        return .flattened(totalRows: fallbackRows)
    }

    static func sanitize(
        weights: [String: MLXArray],
        codecVariant: VoxtralTTSCodecVariant = .legacy
    ) -> [String: MLXArray] {
        var remapped: [String: MLXArray] = [:]
        var weightNormG: [String: MLXArray] = [:]
        var weightNormV: [String: MLXArray] = [:]

        for (key, value) in weights {
            var mappedKey: String?
            var mappedValue = value

            if key == "mm_audio_embeddings.tok_embeddings.weight" || key == "tok_embeddings.weight" {
                mappedKey = "backbone.tok_embeddings.weight"
            } else if let languageModelKey = mapLanguageModelBackboneKey(key) {
                mappedKey = languageModelKey
            } else if key == "mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight"
                || key == "audio_codebook_embeddings.embeddings.weight" {
                mappedKey = "audio_token_embedding.embeddings.weight"
            } else if key == "mm_audio_embeddings.audio_codebook_embeddings.embeddings.bias"
                || key == "audio_codebook_embeddings.embeddings.bias" {
                mappedKey = "audio_token_embedding.embeddings.bias"
            } else if key == "norm.weight" {
                mappedKey = "backbone.norm.weight"
            } else if key.hasPrefix("layers.") {
                mappedKey = "backbone.\(key)"
            } else if key.hasPrefix("acoustic_transformer.") {
                mappedKey = key
            } else if key.hasPrefix("audio_tokenizer.") {
                switch codecVariant {
                case .legacy:
                    mappedKey = key
                case .community:
                    mappedKey = mapCommunityCodecKey(key)
                }
            }

            guard var finalKey = mappedKey else { continue }
            if finalKey.contains(".parametrizations.weight.original0") {
                finalKey = finalKey.replacingOccurrences(
                    of: ".parametrizations.weight.original0",
                    with: ".weight_g"
                )
            } else if finalKey.contains(".parametrizations.weight.original1") {
                finalKey = finalKey.replacingOccurrences(
                    of: ".parametrizations.weight.original1",
                    with: ".weight_v"
                )
            }
            if finalKey == "audio_tokenizer.quantizer.semantic_quantizer.codebook" {
                finalKey += ".weight"
            }

            if finalKey.hasSuffix(".weight_g") {
                weightNormG[String(finalKey.dropLast(".weight_g".count))] = mappedValue
                continue
            }
            if finalKey.hasSuffix(".weight_v") {
                weightNormV[String(finalKey.dropLast(".weight_v".count))] = mappedValue
                continue
            }
            if finalKey.hasPrefix("audio_tokenizer.") && finalKey.contains(".conv.weight") && mappedValue.ndim == 3 {
                if finalKey.contains("upsample") || finalKey.contains("convtr") {
                    // ConvTranspose1d: PyTorch [inC, outC, kW] → MLX [outC, kW, inC]
                    mappedValue = mappedValue.transposed(1, 2, 0)
                } else {
                    // Conv1d: PyTorch [outC, inC, kW] → MLX [outC, kW, inC]
                    mappedValue = mappedValue.transposed(0, 2, 1)
                }
            }
            remapped[finalKey] = mappedValue
        }

        for baseKey in Set(weightNormG.keys).union(weightNormV.keys) {
            guard let weightG = weightNormG[baseKey], let weightV = weightNormV[baseKey] else { continue }
            var fused = fuseWeightNorm(weightG: weightG, weightV: weightV)
            let finalKey = "\(baseKey).weight"
            if finalKey.hasPrefix("audio_tokenizer.") && finalKey.contains(".conv.weight") && fused.ndim == 3 {
                if finalKey.contains("upsample") || finalKey.contains("convtr") {
                    // ConvTranspose1d: PyTorch [inC, outC, kW] → MLX [outC, kW, inC]
                    fused = fused.transposed(1, 2, 0)
                } else {
                    // Conv1d: PyTorch [outC, inC, kW] → MLX [outC, kW, inC]
                    fused = fused.transposed(0, 2, 1)
                }
            }
            remapped[finalKey] = fused
        }

        if let semanticCodebook = remapped["audio_tokenizer.quantizer.semantic_quantizer.codebook"] {
            remapped["audio_tokenizer.quantizer.semantic_quantizer.codebook.weight"] = semanticCodebook
            remapped["audio_tokenizer.quantizer.semantic_quantizer.codebook"] = nil
        }

        if let embeddingSum = remapped["audio_tokenizer.communityQuantizer.semanticCodebook.embedding_sum"],
           let clusterUsage = remapped["audio_tokenizer.communityQuantizer.semanticCodebook.cluster_usage"] {
            let safeUsage = MLX.maximum(clusterUsage.asType(.float32), MLXArray(Float(1e-12)))
                .reshaped(clusterUsage.dim(0), 1)
            remapped["audio_tokenizer.communityQuantizer.semanticCodebook.weight"] =
                embeddingSum.asType(.float32) / safeUsage
            remapped["audio_tokenizer.communityQuantizer.semanticCodebook.embedding_sum"] = nil
            remapped["audio_tokenizer.communityQuantizer.semanticCodebook.cluster_usage"] = nil
        }

        return remapped
    }

    private static func mapCommunityCodecKey(_ key: String) -> String? {
        let decoderMappings: [(String, String)] = [
            ("audio_tokenizer.decoder_blocks.0.conv.", "audio_tokenizer.communityDecoder.inputConv.conv."),
            ("audio_tokenizer.decoder_blocks.1.layers.", "audio_tokenizer.communityDecoder.stage0.layers."),
            ("audio_tokenizer.decoder_blocks.2.conv.", "audio_tokenizer.communityDecoder.upsample1.conv."),
            ("audio_tokenizer.decoder_blocks.3.layers.", "audio_tokenizer.communityDecoder.stage1.layers."),
            ("audio_tokenizer.decoder_blocks.4.conv.", "audio_tokenizer.communityDecoder.upsample2.conv."),
            ("audio_tokenizer.decoder_blocks.5.layers.", "audio_tokenizer.communityDecoder.stage2.layers."),
            ("audio_tokenizer.decoder_blocks.6.conv.", "audio_tokenizer.communityDecoder.upsample3.conv."),
            ("audio_tokenizer.decoder_blocks.7.layers.", "audio_tokenizer.communityDecoder.stage3.layers."),
            ("audio_tokenizer.output_proj.conv.", "audio_tokenizer.communityDecoder.outputProj.conv."),
            (
                "audio_tokenizer.quantizer.semantic_codebook.embedding_sum",
                "audio_tokenizer.communityQuantizer.semanticCodebook.embedding_sum"
            ),
            (
                "audio_tokenizer.quantizer.semantic_codebook.cluster_usage",
                "audio_tokenizer.communityQuantizer.semanticCodebook.cluster_usage"
            ),
        ]

        for (source, target) in decoderMappings {
            if key == source {
                return target
            }
            if key.hasPrefix(source) {
                return target + key.dropFirst(source.count)
            }
        }

        return nil
    }

    private static func mapLanguageModelBackboneKey(_ key: String) -> String? {
        let prefix = "language_model.model.model."
        guard key.hasPrefix(prefix) else {
            return nil
        }

        let suffix = String(key.dropFirst(prefix.count))
        if suffix.hasPrefix("embed_tokens.") {
            return "backbone.tok_embeddings.\(suffix.dropFirst("embed_tokens.".count))"
        }
        if suffix == "norm.weight" {
            return "backbone.norm.weight"
        }
        guard suffix.hasPrefix("layers.") else {
            return nil
        }

        var mapped = "backbone.\(suffix)"
        let replacements: [(String, String)] = [
            (".input_layernorm.", ".attention_norm."),
            (".post_attention_layernorm.", ".ffn_norm."),
            (".self_attn.q_proj.", ".attention.wq."),
            (".self_attn.k_proj.", ".attention.wk."),
            (".self_attn.v_proj.", ".attention.wv."),
            (".self_attn.o_proj.", ".attention.wo."),
            (".mlp.gate_proj.", ".feed_forward.w1."),
            (".mlp.down_proj.", ".feed_forward.w2."),
            (".mlp.up_proj.", ".feed_forward.w3."),
        ]
        for (source, target) in replacements {
            mapped = mapped.replacingOccurrences(of: source, with: target)
        }
        return mapped
    }

    private static func inferredQuantization(
        from modelDir: URL,
        weights: [String: MLXArray]
    ) -> (groupSize: Int, bits: Int)? {
        guard weights.keys.contains(where: { $0.hasSuffix(".scales") }) else {
            return nil
        }

        let lowercasedDirectoryName = modelDir.lastPathComponent.lowercased()
        if lowercasedDirectoryName.contains("4bit") {
            return (64, 4)
        }
        if lowercasedDirectoryName.contains("6bit") {
            return (64, 6)
        }
        return (64, 4)
    }

    static func fuseWeightNorm(weightG: MLXArray, weightV: MLXArray) -> MLXArray {
        let norm = sqrt(sum(weightV * weightV, axes: [1, 2], keepDims: true) + MLXArray(1e-12))
        return weightG * (weightV / norm)
    }

    static func loadPresetVoices(from modelDir: URL, expectedDim: Int) throws -> [String: MLXArray] {
        let safeDir = modelDir.appendingPathComponent("voice_embedding_safe")
        var isDirectory: ObjCBool = false
        if FileManager.default.fileExists(atPath: safeDir.path, isDirectory: &isDirectory), isDirectory.boolValue {
            let indexURL = safeDir.appendingPathComponent("index.json")
            guard FileManager.default.fileExists(atPath: indexURL.path) else {
                throw AudioGenerationError.modelNotInitialized(
                    "Missing Voxtral voice manifest at \(indexURL.path). " +
                        "Run 'bash scripts/voxtral/bootstrap_env.sh' from the repository root, then reinstall or re-normalize the model pack."
                )
            }

            let indexData = try Data(contentsOf: indexURL)
            let manifest = try JSONDecoder().decode(VoxtralTTSVoiceManifest.self, from: indexData)
            guard manifest.expectedDim == expectedDim else {
                throw AudioGenerationError.modelNotInitialized(
                    "Voice embedding dim mismatch: expected \(expectedDim), found \(manifest.expectedDim)"
                )
            }

            var voices: [String: MLXArray] = [:]
            for entry in manifest.voices {
                guard entry.shape.count == 2, entry.shape[1] == expectedDim else {
                    throw AudioGenerationError.modelNotInitialized("Invalid voice embedding shape for \(entry.voiceName)")
                }
                guard !entry.file.contains("/"), !entry.file.contains("\\"), !entry.file.contains(".."), !entry.file.isEmpty else {
                    throw AudioGenerationError.modelNotInitialized("Invalid voice embedding filename: \(entry.file)")
                }
                let fileURL = safeDir.appendingPathComponent(entry.file)
                let data = try Data(contentsOf: fileURL)
                guard data.count % MemoryLayout<Float>.stride == 0 else {
                    throw AudioGenerationError.modelNotInitialized("Invalid voice embedding byte count for \(entry.voiceName)")
                }

                let values: [Float] = data.withUnsafeBytes { rawBuffer in
                    Array(rawBuffer.bindMemory(to: Float.self))
                }
                let expectedCount = entry.shape[0] * entry.shape[1]
                guard values.count == expectedCount, entry.frameCount == entry.shape[0] else {
                    throw AudioGenerationError.modelNotInitialized("Voice embedding size mismatch for \(entry.voiceName)")
                }

                voices[entry.voiceName.lowercased()] = MLXArray(values).reshaped(entry.shape[0], entry.shape[1])
            }

            return voices
        }

        let safetensorsVoiceDir = modelDir.appendingPathComponent("voice_embedding")
        var isSafetensorsDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: safetensorsVoiceDir.path, isDirectory: &isSafetensorsDirectory),
              isSafetensorsDirectory.boolValue else {
            throw AudioGenerationError.modelNotInitialized(
                "Missing Voxtral preset voice assets at \(safeDir.path) or \(safetensorsVoiceDir.path). " +
                    "Official packs require the managed normalization step; MLX community packs must include voice_embedding/*.safetensors."
            )
        }

        var voices: [String: MLXArray] = [:]
        let voiceFiles = try FileManager.default.contentsOfDirectory(
            at: safetensorsVoiceDir,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        ).filter { $0.pathExtension.lowercased() == "safetensors" }

        guard !voiceFiles.isEmpty else {
            throw AudioGenerationError.modelNotInitialized(
                "Missing Voxtral preset voice safetensors in \(safetensorsVoiceDir.path)."
            )
        }

        for fileURL in voiceFiles {
            let voiceName = fileURL.deletingPathExtension().lastPathComponent.lowercased()
            let arrays = try MLX.loadArrays(url: fileURL)
            guard let tensor = arrays.values.first, arrays.count == 1 else {
                throw AudioGenerationError.modelNotInitialized("Invalid voice embedding payload for \(voiceName)")
            }
            guard tensor.ndim == 2, tensor.dim(1) == expectedDim else {
                throw AudioGenerationError.modelNotInitialized("Invalid voice embedding shape for \(voiceName)")
            }
            voices[voiceName] = tensor
        }

        return voices
    }
}
