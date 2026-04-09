import Foundation
import HuggingFace
@preconcurrency import MLX
import MLXAudioCore
@preconcurrency import MLXLMCommon
import MLXNN
import Tokenizers

// MARK: - Qwen3TTS Model

public final class Qwen3TTSModel: Module, ConditionedSpeechGenerationModel, @unchecked Sendable {
    private final class TTSSpecialEmbeddingsBox: NSObject {
        let bos: MLXArray
        let eos: MLXArray
        let pad: MLXArray

        init(bos: MLXArray, eos: MLXArray, pad: MLXArray) {
            self.bos = bos
            self.eos = eos
            self.pad = pad
        }
    }

    private final class ResolvedVoiceClonePromptBox: NSObject {
        let prompt: ResolvedVoiceClonePrompt

        init(_ prompt: ResolvedVoiceClonePrompt) {
            self.prompt = prompt
        }
    }

    private final class MLXArrayBox: NSObject {
        let array: MLXArray

        init(_ array: MLXArray) {
            self.array = array
        }
    }

    private final class PreparedVoiceClonePromptContextBox: NSObject {
        let context: PreparedVoiceClonePromptContext

        init(_ context: PreparedVoiceClonePromptContext) {
            self.context = context
        }
    }

    struct VoiceClonePromptPayloadV1: Codable, Sendable, Equatable {
        let version: Int
        let refSpeakerEmbedding: Data?
        let refCode: Data?
        let refText: String?
        let numCodeGroups: Int?
        let frameCount: Int?
        let xVectorOnlyMode: Bool
        let iclMode: Bool

        init(
            refSpeakerEmbedding: Data?,
            refCode: Data?,
            refText: String?,
            numCodeGroups: Int?,
            frameCount: Int?,
            xVectorOnlyMode: Bool,
            iclMode: Bool
        ) {
            self.version = 1
            self.refSpeakerEmbedding = refSpeakerEmbedding
            self.refCode = refCode
            self.refText = refText
            self.numCodeGroups = numCodeGroups
            self.frameCount = frameCount
            self.xVectorOnlyMode = xVectorOnlyMode
            self.iclMode = iclMode
        }
    }

    struct ResolvedVoiceClonePrompt {
        let speakerEmbedding: MLXArray?
        let refCodes: MLXArray?
        let refText: String?
        let xVectorOnlyMode: Bool
        let iclMode: Bool
    }

    public struct PreparedVoiceClonePromptContext {
        let speakerEmbedding: MLXArray?
        let refCodes: MLXArray
        let refTextEmbedding: MLXArray
        let codecContextEmbedding: MLXArray
    }

    public enum PreparedTalkerPromptPhase {
        case anchor
        case continuation
    }

    public struct PreparedTalkerPrefixCache {
        let phase: PreparedTalkerPromptPhase
        let prefixEmbeddings: MLXArray
        let cacheSnapshot: [KVCacheSnapshot]
        let prefixTokenCount: Int
    }

    public struct PreparedStableNarratorPromptPhase {
        let prefixCache: PreparedTalkerPrefixCache
        let speakerEmbedding: MLXArray?
        let preparedClonePrompt: PreparedVoiceClonePromptContext?
        let language: String
    }

    public struct PreparedStableNarratorPromptPhases {
        public let anchor: PreparedStableNarratorPromptPhase
        public let continuation: PreparedStableNarratorPromptPhase?
    }

    private struct PreparedPromptExecutionInputs {
        let promptEmbeddings: MLXArray
        let trailingTextHidden: MLXArray
        let ttsPadEmbed: MLXArray
        let refCodes: MLXArray?
        let prefixCache: PreparedTalkerPrefixCache?

        var prefillTokenCount: Int {
            (prefixCache?.prefixTokenCount ?? 0) + promptEmbeddings.dim(1)
        }
    }

    let config: Qwen3TTSModelConfig
    let talker: Qwen3TTSTalkerForConditionalGeneration
    var speakerEncoder: Qwen3TTSSpeakerEncoder?
    var speechTokenizer: Qwen3TTSSpeechTokenizer?
    var tokenizer: Tokenizer?
    private let resolvedClonePromptCache = NSCache<NSData, ResolvedVoiceClonePromptBox>()
    private let resolvedSpeakerEmbeddingCache = NSCache<NSData, MLXArrayBox>()
    private let preparedClonePromptContextCache = NSCache<NSData, PreparedVoiceClonePromptContextBox>()
    private let ttsSpecialEmbeddingsCache = NSCache<NSString, TTSSpecialEmbeddingsBox>()
    private let inputEmbeddingCache = NSCache<NSString, MLXArrayBox>()

    public var sampleRate: Int { config.sampleRate }

    public var defaultGenerationParameters: GenerateParameters {
        GenerateParameters(
            maxTokens: 8192,
            temperature: 0.9,
            topP: 1.0,
            repetitionPenalty: 1.05
        )
    }

    init(config: Qwen3TTSModelConfig) {
        let talkerConfig = config.talkerConfig ?? {
            let json = "{}".data(using: .utf8)!
            return try! JSONDecoder().decode(Qwen3TTSTalkerConfig.self, from: json)
        }()
        self.config = config
        self.talker = Qwen3TTSTalkerForConditionalGeneration(config: talkerConfig)
        self.speakerEncoder = config.ttsModelType == "base"
            ? Qwen3TTSSpeakerEncoder(config: config.speakerEncoderConfig)
            : nil
        self.resolvedClonePromptCache.countLimit = 16
        self.resolvedSpeakerEmbeddingCache.countLimit = 32
        self.preparedClonePromptContextCache.countLimit = 16
        self.ttsSpecialEmbeddingsCache.countLimit = 1
        self.inputEmbeddingCache.countLimit = 16
    }

    private func normalized(_ value: String?) -> String? {
        guard let trimmed = value?.trimmingCharacters(in: .whitespacesAndNewlines),
              !trimmed.isEmpty else {
            return nil
        }
        return trimmed
    }

    private func cachedTTSSpecialEmbeddings() -> TTSSpecialEmbeddingsBox {
        let cacheKey = "default"
        if let cached = ttsSpecialEmbeddingsCache.object(forKey: cacheKey as NSString) {
            return cached
        }

        let ttsTokens = MLXArray(
            [Int32(config.ttsBosTokenId), Int32(config.ttsEosTokenId), Int32(config.ttsPadTokenId)]
        ).reshaped(1, 3)
        let ttsEmbeds = talker.textProjection(talker.getTextEmbeddings()(ttsTokens))
        let prepared = TTSSpecialEmbeddingsBox(
            bos: ttsEmbeds[0..., 0 ..< 1, 0...],
            eos: ttsEmbeds[0..., 1 ..< 2, 0...],
            pad: ttsEmbeds[0..., 2 ..< 3, 0...]
        )
        ttsSpecialEmbeddingsCache.setObject(prepared, forKey: cacheKey as NSString)
        return prepared
    }

    private func cachedInputEmbedding(tokens: [Int32], key: String) -> MLXArray {
        if let cached = inputEmbeddingCache.object(forKey: key as NSString) {
            return cached.array
        }

        let embedding = talker.getInputEmbeddings()(MLXArray(tokens).reshaped(1, -1))
        inputEmbeddingCache.setObject(MLXArrayBox(embedding), forKey: key as NSString)
        return embedding
    }

    private func cachedCodecPadEmbedding(talkerConfig: Qwen3TTSTalkerConfig) -> MLXArray {
        cachedInputEmbedding(
            tokens: [Int32(talkerConfig.codecPadId)],
            key: "codec-pad"
        )
    }

    private func cachedCodecBosEmbedding(talkerConfig: Qwen3TTSTalkerConfig) -> MLXArray {
        cachedInputEmbedding(
            tokens: [Int32(talkerConfig.codecBosId)],
            key: "codec-bos"
        )
    }

    private func cachedCodecPrefixSuffixEmbedding(talkerConfig: Qwen3TTSTalkerConfig) -> MLXArray {
        cachedInputEmbedding(
            tokens: [Int32(talkerConfig.codecPadId), Int32(talkerConfig.codecBosId)],
            key: "codec-prefix-suffix"
        )
    }

    private func cachedCodecPrefillEmbedding(
        language: String,
        talkerConfig: Qwen3TTSTalkerConfig
    ) -> MLXArray {
        let normalizedLanguage = language.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let languageKey = normalizedLanguage.isEmpty ? "auto" : normalizedLanguage
        let key = "codec-prefill:\(languageKey)"
        if let cached = inputEmbeddingCache.object(forKey: key as NSString) {
            return cached.array
        }

        let languageId = resolvedCodecLanguageID(language: language, talkerConfig: talkerConfig)
        let codecPrefill: [Int32] = if let langId = languageId {
            [
                Int32(talkerConfig.codecThinkId),
                Int32(talkerConfig.codecThinkBosId),
                Int32(langId),
                Int32(talkerConfig.codecThinkEosId)
            ]
        } else {
            [
                Int32(talkerConfig.codecNothinkId),
                Int32(talkerConfig.codecThinkBosId),
                Int32(talkerConfig.codecThinkEosId)
            ]
        }

        let embedding = talker.getInputEmbeddings()(MLXArray(codecPrefill).reshaped(1, -1))
        inputEmbeddingCache.setObject(MLXArrayBox(embedding), forKey: key as NSString)
        return embedding
    }

    private func resolvedCodecLanguageID(
        language: String,
        talkerConfig: Qwen3TTSTalkerConfig
    ) -> Int? {
        let normalizedLanguage = language.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard normalizedLanguage != "auto",
              !normalizedLanguage.isEmpty,
              let langMap = talkerConfig.codecLanguageId else {
            return nil
        }
        return langMap[normalizedLanguage]
    }

    static func updateRepetitionWindow(
        tokenId: Int,
        repetitionContextSize: Int,
        recentTokens: inout [Int],
        recentUniqueTokens: inout [Int],
        recentUniqueTokenSet: inout Set<Int>
    ) {
        guard repetitionContextSize > 0 else {
            recentTokens.removeAll(keepingCapacity: true)
            recentUniqueTokens.removeAll(keepingCapacity: true)
            recentUniqueTokenSet.removeAll(keepingCapacity: true)
            return
        }

        recentTokens.append(tokenId)
        if recentUniqueTokenSet.insert(tokenId).inserted {
            recentUniqueTokens.append(tokenId)
        }

        let overflow = recentTokens.count - repetitionContextSize
        guard overflow > 0 else {
            return
        }

        let removedTokens = recentTokens.prefix(overflow)
        recentTokens.removeFirst(overflow)

        guard !removedTokens.isEmpty else {
            return
        }

        let removedSet = Set(removedTokens)
        for removedToken in removedSet {
            guard !recentTokens.contains(removedToken) else {
                continue
            }
            recentUniqueTokenSet.remove(removedToken)
            recentUniqueTokens.removeAll { $0 == removedToken }
        }
    }

    private func resolvedVoiceInputs(for voice: String?) -> (speaker: String?, instruct: String?) {
        let normalizedVoice = normalized(voice)
        switch config.ttsModelType {
        case "voice_design":
            return (speaker: nil, instruct: normalizedVoice)
        case "custom_voice", "base":
            return (speaker: normalizedVoice, instruct: nil)
        default:
            return (speaker: normalizedVoice, instruct: nil)
        }
    }

    // MARK: - SpeechGenerationModel protocol

    public func generate(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) async throws -> MLXArray {
        let routed = resolvedVoiceInputs(for: voice)
        return try await generate(
            text: text,
            speaker: routed.speaker,
            instruct: routed.instruct,
            conditioning: nil,
            refAudio: refAudio,
            refText: refText,
            language: language,
            generationParameters: generationParameters
        )
    }

    public func generate(
        text: String,
        voice: String?,
        conditioning: SpeechConditioning?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) async throws -> MLXArray {
        let routed = resolvedVoiceInputs(for: voice)
        return try await generate(
            text: text,
            speaker: routed.speaker,
            instruct: routed.instruct,
            conditioning: conditioning,
            refAudio: refAudio,
            refText: refText,
            language: language,
            generationParameters: generationParameters
        )
    }

    public func generate(
        text: String,
        speaker: String?,
        instruct: String?,
        conditioning: SpeechConditioning?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) async throws -> MLXArray {
        guard speechTokenizer != nil else {
            throw AudioGenerationError.modelNotInitialized("Speech tokenizer not loaded")
        }
        guard tokenizer != nil else {
            throw AudioGenerationError.modelNotInitialized("Text tokenizer not loaded")
        }

        return try await generatePrepared(
            text: text,
            speaker: speaker,
            instruct: instruct,
            conditioning: conditioning,
            refAudio: refAudio,
            refText: refText,
            language: language,
            generationParameters: generationParameters,
            preparedStablePromptPhase: nil
        )
    }

    public func generate(
        text: String,
        speaker: String?,
        instruct: String?,
        conditioning: SpeechConditioning?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters,
        preparedStablePromptPhase: PreparedStableNarratorPromptPhase?,
        onInfo: ((AudioGenerationInfo) -> Void)? = nil
    ) async throws -> MLXArray {
        guard speechTokenizer != nil else {
            throw AudioGenerationError.modelNotInitialized("Speech tokenizer not loaded")
        }
        guard tokenizer != nil else {
            throw AudioGenerationError.modelNotInitialized("Text tokenizer not loaded")
        }

        return try await generatePrepared(
            text: text,
            speaker: speaker,
            instruct: instruct,
            conditioning: conditioning,
            refAudio: refAudio,
            refText: refText,
            language: language,
            generationParameters: generationParameters,
            preparedStablePromptPhase: preparedStablePromptPhase,
            onInfo: onInfo
        )
    }

    func generatePrepared(
        text: String,
        speaker: String?,
        instruct: String?,
        conditioning: SpeechConditioning?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters,
        preparedStablePromptPhase: PreparedStableNarratorPromptPhase?,
        onInfo: ((AudioGenerationInfo) -> Void)? = nil
    ) async throws -> MLXArray {
        guard speechTokenizer != nil else {
            throw AudioGenerationError.modelNotInitialized("Speech tokenizer not loaded")
        }
        guard tokenizer != nil else {
            throw AudioGenerationError.modelNotInitialized("Text tokenizer not loaded")
        }

        let clonePrompt = try resolvedVoiceClonePrompt(from: conditioning)
        let preparedClonePrompt = try preparedVoiceClonePromptContext(
            from: conditioning,
            resolvedPrompt: clonePrompt
        )
        let speakerEmbedding: MLXArray? = if let cloneSpeakerEmbedding = clonePrompt?.speakerEmbedding {
            cloneSpeakerEmbedding
        } else {
            try resolvedSpeakerEmbedding(from: conditioning)
        }

        let audio = try generateSpeech(
            text: text,
            speaker: speaker,
            instruct: instruct,
            language: language ?? "auto",
            speakerEmbedding: speakerEmbedding,
            voiceClonePrompt: clonePrompt,
            preparedClonePrompt: preparedClonePrompt,
            preparedStablePromptPhase: preparedStablePromptPhase,
            refAudio: refAudio,
            refText: refText,
            temperature: generationParameters.temperature,
            topK: 50,
            topP: generationParameters.topP,
            repetitionPenalty: generationParameters.repetitionPenalty ?? 1.05,
            repetitionContextSize: generationParameters.repetitionContextSize,
            minP: 0.0,
            maxTokens: generationParameters.maxTokens ?? 8192,
            onInfo: onInfo
        )
        return audio
    }

    public func generateStream(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        let routed = resolvedVoiceInputs(for: voice)
        return generateStream(
            text: text,
            speaker: routed.speaker,
            instruct: routed.instruct,
            conditioning: nil,
            refAudio: refAudio,
            refText: refText,
            language: language,
            generationParameters: generationParameters,
            streamingInterval: 2.0
        )
    }

    public func generateStream(
        text: String,
        voice: String?,
        conditioning: SpeechConditioning?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        let routed = resolvedVoiceInputs(for: voice)
        return generateStream(
            text: text,
            speaker: routed.speaker,
            instruct: routed.instruct,
            conditioning: conditioning,
            refAudio: refAudio,
            refText: refText,
            language: language,
            generationParameters: generationParameters,
            streamingInterval: 2.0
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
        let routed = resolvedVoiceInputs(for: voice)
        return generateStream(
            text: text,
            speaker: routed.speaker,
            instruct: routed.instruct,
            conditioning: nil,
            refAudio: refAudio,
            refText: refText,
            language: language,
            generationParameters: generationParameters,
            streamingInterval: streamingInterval
        )
    }

    public func generateStream(
        text: String,
        voice: String?,
        conditioning: SpeechConditioning?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters,
        streamingInterval: Double
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        let routed = resolvedVoiceInputs(for: voice)
        return generateStream(
            text: text,
            speaker: routed.speaker,
            instruct: routed.instruct,
            conditioning: conditioning,
            refAudio: refAudio,
            refText: refText,
            language: language,
            generationParameters: generationParameters,
            streamingInterval: streamingInterval
        )
    }

    public func generateStream(
        text: String,
        speaker: String?,
        instruct: String?,
        conditioning: SpeechConditioning?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters,
        streamingInterval: Double = 2.0
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        let (stream, continuation) = AsyncThrowingStream<AudioGeneration, Error>.makeStream()
        let inheritedHeartbeatHandler = AudioGenerationObserverContext.heartbeatHandler
        let task = Task { @Sendable [weak self] in
            guard let self else { return }
            do {
                try AudioGenerationObserverContext.$heartbeatHandler.withValue(inheritedHeartbeatHandler) {
                    guard speechTokenizer != nil else {
                        throw AudioGenerationError.modelNotInitialized("Speech tokenizer not loaded")
                    }
                    guard tokenizer != nil else {
                        throw AudioGenerationError.modelNotInitialized("Text tokenizer not loaded")
                    }

                    let clonePrompt = try self.resolvedVoiceClonePrompt(from: conditioning)
                    let preparedClonePrompt = try self.preparedVoiceClonePromptContext(
                        from: conditioning,
                        resolvedPrompt: clonePrompt
                    )
                    let speakerEmbedding: MLXArray? = if let cloneSpeakerEmbedding = clonePrompt?.speakerEmbedding {
                        cloneSpeakerEmbedding
                    } else {
                        try self.resolvedSpeakerEmbedding(from: conditioning)
                    }
                    let lang = language ?? "auto"
                    let temp = generationParameters.temperature
                    let topP = generationParameters.topP
                    let repPenalty = generationParameters.repetitionPenalty ?? 1.05
                    let repetitionContextSize = generationParameters.repetitionContextSize
                    let maxTokens = generationParameters.maxTokens ?? 8192

                    _ = try generateSpeech(
                        text: text,
                        speaker: speaker,
                        instruct: instruct,
                        language: lang,
                        speakerEmbedding: speakerEmbedding,
                        voiceClonePrompt: clonePrompt,
                        preparedClonePrompt: preparedClonePrompt,
                        refAudio: refAudio,
                        refText: refText,
                        temperature: temp,
                        topK: 50,
                        topP: topP,
                        repetitionPenalty: repPenalty,
                        repetitionContextSize: repetitionContextSize,
                        minP: 0.0,
                        maxTokens: maxTokens,
                        streamingInterval: streamingInterval,
                        onToken: { tokenId in
                            continuation.yield(.token(tokenId))
                        },
                        onInfo: { info in
                            continuation.yield(.info(info))
                        },
                        onAudioChunk: { chunk in
                            continuation.yield(.audio(chunk))
                        }
                    )
                }
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }
        continuation.onTermination = { @Sendable _ in task.cancel() }
        return stream
    }

    // MARK: - Code stream (Valar native decoder path)

    /// Stream raw codec frames for Valar's native `SpeechTokenizerDecoder`.
    /// Each yielded `MLXArray` has shape `[1, numCodeGroups]` (one per generation step).
    public func generateCodeStream(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) -> AsyncThrowingStream<MLXArray, Error> {
        let routed = resolvedVoiceInputs(for: voice)
        return generateCodeStream(
            text: text,
            speaker: routed.speaker,
            instruct: routed.instruct,
            conditioning: nil,
            refAudio: refAudio,
            refText: refText,
            language: language,
            generationParameters: generationParameters
        )
    }

    public func generateCodeStream(
        text: String,
        voice: String?,
        conditioning: SpeechConditioning?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) -> AsyncThrowingStream<MLXArray, Error> {
        let routed = resolvedVoiceInputs(for: voice)
        return generateCodeStream(
            text: text,
            speaker: routed.speaker,
            instruct: routed.instruct,
            conditioning: conditioning,
            refAudio: refAudio,
            refText: refText,
            language: language,
            generationParameters: generationParameters
        )
    }

    public func generateCodeStream(
        text: String,
        speaker: String?,
        instruct: String?,
        conditioning: SpeechConditioning?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) -> AsyncThrowingStream<MLXArray, Error> {
        let stream = AsyncThrowingStream<MLXArray, Error> { continuation in
            let inheritedHeartbeatHandler = AudioGenerationObserverContext.heartbeatHandler
            let task = Task { @Sendable [weak self] in
                guard let self else {
                    continuation.finish()
                    return
                }
                do {
                    try AudioGenerationObserverContext.$heartbeatHandler.withValue(inheritedHeartbeatHandler) {
                        guard self.speechTokenizer != nil else {
                            throw AudioGenerationError.modelNotInitialized("Speech tokenizer not loaded")
                        }
                        guard self.tokenizer != nil else {
                            throw AudioGenerationError.modelNotInitialized("Text tokenizer not loaded")
                        }

                        let clonePrompt = try self.resolvedVoiceClonePrompt(from: conditioning)
                        let preparedClonePrompt = try self.preparedVoiceClonePromptContext(
                            from: conditioning,
                            resolvedPrompt: clonePrompt
                        )
                        let speakerEmbedding: MLXArray? = if let cloneSpeakerEmbedding = clonePrompt?.speakerEmbedding {
                            cloneSpeakerEmbedding
                        } else {
                            try self.resolvedSpeakerEmbedding(from: conditioning)
                        }
                        let lang = language ?? "auto"
                        let temp = generationParameters.temperature
                        let topP = generationParameters.topP
                        let repPenalty = generationParameters.repetitionPenalty ?? 1.05
                        let repetitionContextSize = generationParameters.repetitionContextSize
                        let maxTokens = generationParameters.maxTokens ?? 8192

                        _ = try self.generateSpeech(
                            text: text,
                            speaker: speaker,
                            instruct: instruct,
                            language: lang,
                            speakerEmbedding: speakerEmbedding,
                            voiceClonePrompt: clonePrompt,
                            preparedClonePrompt: preparedClonePrompt,
                            refAudio: refAudio,
                            refText: refText,
                            temperature: temp,
                            topK: 50,
                            topP: topP,
                            repetitionPenalty: repPenalty,
                            repetitionContextSize: repetitionContextSize,
                            minP: 0.0,
                            maxTokens: maxTokens,
                            onCodeFrame: { codeFrame in
                                continuation.yield(codeFrame)
                            }
                        )
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
        return stream
    }

    // MARK: - Decode chunk helper

    /// Decode a chunk of codec codes to audio waveform.
    /// - Parameters:
    ///   - codes: Codec codes [1, time, numCodeGroups]
    ///   - chunkTokens: Tokens per decode chunk (controls decode granularity)
    /// - Returns: Decoded audio waveform (1D)
    private func decodeChunk(_ codes: MLXArray, chunkTokens: Int = 300) -> MLXArray {
        guard let speechTokenizer else { return MLXArray.zeros([1]) }

        var audioChunks = [MLXArray]()
        for chunk in speechTokenizer.streamingDecode(codes, chunkTokens: chunkTokens) {
            audioChunks.append(chunk)
        }
        var audio = concatenated(audioChunks, axis: -1)[0]

        let validLen = Int((codes[0..., 0..., 0] .> 0).sum().item(Int32.self))
            * speechTokenizer.decodeUpsampleRate
        if validLen > 0, validLen < audio.dim(0) {
            audio = audio[..<validLen]
        }

        eval(audio)
        return audio
    }

    // MARK: - Qwen generation

    func generateSpeech(
        text: String,
        speaker: String?,
        instruct: String?,
        language: String,
        speakerEmbedding: MLXArray? = nil,
        voiceClonePrompt: ResolvedVoiceClonePrompt? = nil,
        preparedClonePrompt: PreparedVoiceClonePromptContext? = nil,
        preparedStablePromptPhase: PreparedStableNarratorPromptPhase? = nil,
        refAudio: MLXArray?,
        refText: String?,
        temperature: Float,
        topK: Int,
        topP: Float,
        repetitionPenalty: Float,
        repetitionContextSize: Int,
        minP: Float,
        maxTokens: Int,
        streamingInterval: Double = 2.0,
        onToken: ((Int) -> Void)? = nil,
        onInfo: ((AudioGenerationInfo) -> Void)? = nil,
        onAudioChunk: ((MLXArray) -> Void)? = nil,
        onCodeFrame: ((MLXArray) -> Void)? = nil
    ) throws -> MLXArray {
        guard let speechTokenizer, tokenizer != nil else {
            return MLXArray.zeros([1])
        }

        guard let talkerConfig = config.talkerConfig else {
            return MLXArray.zeros([1])
        }

        let executionInputs = try prepareSegmentExecutionInputs(
            text: text,
            language: language,
            speaker: speaker,
            instruct: instruct,
            speakerEmbedding: speakerEmbedding,
            voiceClonePrompt: voiceClonePrompt,
            preparedClonePrompt: preparedClonePrompt,
            refAudio: refAudio,
            refText: refText,
            preparedStablePromptPhase: preparedStablePromptPhase
        )
        guard executionInputs.promptEmbeddings.size > 0 else {
            return MLXArray.zeros([1])
        }

        let trailingTextHidden = executionInputs.trailingTextHidden
        let ttsPadEmbed = executionInputs.ttsPadEmbed
        let refCodes = executionInputs.refCodes

        // Initialize cache and timing
        let startTime = Date()
        let prefillStartedAt = Date()
        let cache = if let prefixCache = executionInputs.prefixCache {
            cloneKVCaches(from: prefixCache.cacheSnapshot, using: talker.makeCache)
        } else {
            talker.makeCache()
        }
        let (initialLogits, initialHidden) = talker(executionInputs.promptEmbeddings, cache: cache)
        // Start the prefill work immediately so the first token sample does not absorb the full lazy-eval cost.
        asyncEval(initialLogits, initialHidden)
        let prefillTime = Date().timeIntervalSince(prefillStartedAt)
        let prefillTokenCount = executionInputs.prefillTokenCount
        var generatedCodes = [MLXArray]()
        generatedCodes.reserveCapacity(maxTokens)
        let boundedRepetitionContextSize = max(0, repetitionContextSize)
        var recentGeneratedCodebookTokens = [Int]()
        var recentUniqueGeneratedCodebookTokens = [Int]()
        if boundedRepetitionContextSize > 0 {
            recentGeneratedCodebookTokens.reserveCapacity(boundedRepetitionContextSize)
            recentUniqueGeneratedCodebookTokens.reserveCapacity(boundedRepetitionContextSize)
        }
        var recentUniqueGeneratedCodebookTokenSet = Set<Int>()
        let eosTokenId = talkerConfig.codecEosTokenId

        // Suppress special tokens
        let suppressStart = max(0, talkerConfig.vocabSize - 1024)
        let suppressTokens = (suppressStart ..< talkerConfig.vocabSize)
            .filter { $0 != eosTokenId }

        // Streaming decode state
        let codecTokenRateHz = 12.5
        let streamingChunkSize = max(1, Int(streamingInterval * codecTokenRateHz))
        var decodedTokens = 0

        var trailingIdx = 0
        var inputEmbeds = executionInputs.promptEmbeddings
        var wasCancelled = false
        let talkerInputEmbeddings = talker.getInputEmbeddings()
        let codePredictor = talker.codePredictor
        let codePredictorCodecEmbeddings = codePredictor.codecEmbedding
        var codeCache = codePredictor.makeCache()
        var lastHeartbeatTokens = 0
        var lastHeartbeatTime = startTime

        if onAudioChunk != nil {
            speechTokenizer.decoder.resetStreamingState()
        }
        defer {
            if onAudioChunk != nil {
                speechTokenizer.decoder.resetStreamingState()
            }
        }

        var currentLogits = initialLogits
        var currentLastHidden = initialHidden[0..., (-1)..., 0...]
        let generateStartedAt = Date()
        var samplingTime: TimeInterval = 0
        var evalTime: TimeInterval = 0
        var tokenMaterializationTime: TimeInterval = 0
        var embeddingAssemblyTime: TimeInterval = 0
        var talkerForwardTime: TimeInterval = 0
        var codePredictorTime: TimeInterval = 0
        let preparedTalkerGenerationPositions = talker.prepareGenerationPositions(
            startOffset: cache.first?.offset ?? 0,
            count: maxTokens,
            dtype: inputEmbeds.dtype
        )

        for step in 0 ..< maxTokens {
            if Task.isCancelled {
                wasCancelled = true
                break
            }

            // Sample first codebook token
            let primarySamplingStartedAt = Date()
            let nextToken = sampleToken(
                currentLogits,
                temperature: temperature,
                topP: topP,
                topK: topK,
                repetitionPenalty: repetitionPenalty,
                generatedTokens: recentGeneratedCodebookTokens,
                uniqueGeneratedTokens: recentUniqueGeneratedCodebookTokens,
                suppressTokens: suppressTokens,
                eosTokenId: eosTokenId,
                minP: minP
            )
            samplingTime += Date().timeIntervalSince(primarySamplingStartedAt)

            let tokenMaterializationStartedAt = Date()
            let tokenId = Int(nextToken[0, 0].item(Int32.self))
            let isEOS = tokenId == eosTokenId
            tokenMaterializationTime += Date().timeIntervalSince(tokenMaterializationStartedAt)
            onToken?(tokenId)
            if isEOS {
                break
            }

            // Generate remaining codebook tokens with code predictor
            var codeTokens = [nextToken]
            let codeHidden = currentLastHidden
            let embeddingAssemblyStartedAt = Date()
            let nextTokenEmbedding = talkerInputEmbeddings(nextToken)
            var codecEmbed = nextTokenEmbedding
            var predictorInputEmbedding: MLXArray?
            resetKVCachesForReuse(&codeCache)

            for codeIdx in 0 ..< talkerConfig.numCodeGroups - 1 {
                if Task.isCancelled {
                    wasCancelled = true
                    break
                }
                let codeInput: MLXArray
                if codeIdx == 0 {
                    codeInput = concatenated([codeHidden, nextTokenEmbedding], axis: 1)
                } else {
                    guard let predictorInputEmbedding else {
                        throw AudioGenerationError.generationFailed("Missing cached code predictor embedding for Qwen code group \(codeIdx)")
                    }
                    codeInput = predictorInputEmbedding
                }

                let codePredictorStartedAt = Date()
                let (codeLogits, _, _) = codePredictor(
                    codeInput, cache: codeCache, generationStep: codeIdx
                )
                codePredictorTime += Date().timeIntervalSince(codePredictorStartedAt)

                let codeSamplingStartedAt = Date()
                let nextCode = sampleToken(
                    codeLogits,
                    temperature: temperature,
                    topP: topP,
                    topK: topK,
                    minP: minP
                )
                samplingTime += Date().timeIntervalSince(codeSamplingStartedAt)
                codeTokens.append(nextCode)
                let nextCodeEmbedding = codePredictorCodecEmbeddings[codeIdx](nextCode)
                codecEmbed = codecEmbed + nextCodeEmbedding
                predictorInputEmbedding = nextCodeEmbedding
            }
            embeddingAssemblyTime += Date().timeIntervalSince(embeddingAssemblyStartedAt)
            if wasCancelled {
                break
            }

            let allCodes = concatenated(codeTokens, axis: 1) // [1, num_code_groups]
            Self.updateRepetitionWindow(
                tokenId: tokenId,
                repetitionContextSize: boundedRepetitionContextSize,
                recentTokens: &recentGeneratedCodebookTokens,
                recentUniqueTokens: &recentUniqueGeneratedCodebookTokens,
                recentUniqueTokenSet: &recentUniqueGeneratedCodebookTokenSet
            )
            generatedCodes.append(allCodes)

            // Prepare next input
            let textEmbed: MLXArray
            if trailingIdx < trailingTextHidden.dim(1) {
                textEmbed = trailingTextHidden[0..., trailingIdx ..< (trailingIdx + 1), 0...]
                trailingIdx += 1
            } else {
                textEmbed = ttsPadEmbed
            }

            let inputAssemblyStartedAt = Date()
            inputEmbeds = textEmbed + codecEmbed
            embeddingAssemblyTime += Date().timeIntervalSince(inputAssemblyStartedAt)

            let now = Date()
            let shouldHeartbeat = generatedCodes.count == 1
                || generatedCodes.count - lastHeartbeatTokens >= 64
                || now.timeIntervalSince(lastHeartbeatTime) >= 2.0
            if shouldHeartbeat {
                AudioGenerationObserverContext.heartbeatHandler?(
                    AudioGenerationHeartbeat(
                        generatedTokenCount: generatedCodes.count,
                        maxTokens: maxTokens,
                        wallTimeSeconds: now.timeIntervalSince(startTime)
                    )
                )
                lastHeartbeatTokens = generatedCodes.count
                lastHeartbeatTime = now
            }

            // Yield raw code frame for native decoder path (Valar fork addition)
            onCodeFrame?(allCodes)

            // Streaming: decode and yield audio chunks during generation
            if let onAudioChunk {
                if Task.isCancelled {
                    wasCancelled = true
                    break
                }
                let newTokens = generatedCodes.count - decodedTokens
                if newTokens >= streamingChunkSize {
                    let codesChunk = stacked(Array(generatedCodes[decodedTokens...]), axis: 1)
                    let codesForDecoder = codesChunk.transposed(0, 2, 1)
                    let decodeEvalStartedAt = Date()
                    eval(codesForDecoder)
                    let decoded = speechTokenizer.decoder.streamingStep(codesForDecoder).squeezed(axis: 1)
                    let audioChunk = decoded[0]
                    eval(audioChunk)
                    evalTime += Date().timeIntervalSince(decodeEvalStartedAt)

                    decodedTokens = generatedCodes.count
                    onAudioChunk(audioChunk)
                }
            }

            guard step + 1 < maxTokens else {
                continue
            }
            let talkerForwardStartedAt = Date()
            let output = talker(
                inputEmbeds,
                positionEmbeddings: preparedTalkerGenerationPositions?[safe: step],
                cache: cache
            )
            // Keep the next-step logits/hidden moving while the loop does host-side bookkeeping.
            asyncEval(output.0, output.1)
            talkerForwardTime += Date().timeIntervalSince(talkerForwardStartedAt)
            currentLogits = output.0
            currentLastHidden = output.1

        }

        guard !generatedCodes.isEmpty else {
            return MLXArray.zeros([1])
        }

        // Emit generation info
        let generateTime = Date().timeIntervalSince(generateStartedAt)
        let tokenCount = generatedCodes.count
        let info = AudioGenerationInfo(
            promptTokenCount: prefillTokenCount,
            generationTokenCount: tokenCount,
            prefillTime: prefillTime,
            generateTime: generateTime,
            samplingTime: samplingTime,
            evalTime: evalTime,
            tokenMaterializationTime: tokenMaterializationTime,
            embeddingAssemblyTime: embeddingAssemblyTime,
            talkerForwardTime: talkerForwardTime,
            codePredictorTime: codePredictorTime,
            tokensPerSecond: Double(tokenCount) / max(generateTime, 0.001),
            peakMemoryUsage: Double(Memory.peakMemory) / 1e9
        )
        onInfo?(info)
        AudioGenerationObserverContext.infoHandler?(info)

        // Streaming path: yield remaining tokens and return early
        if let onAudioChunk, !wasCancelled {
            if generatedCodes.count > decodedTokens {
                let codesChunk = stacked(Array(generatedCodes[decodedTokens...]), axis: 1)
                let codesForDecoder = codesChunk.transposed(0, 2, 1)
                eval(codesForDecoder)
                let decoded = speechTokenizer.decoder.streamingStep(codesForDecoder).squeezed(axis: 1)
                let audioChunk = decoded[0]
                eval(audioChunk)
                onAudioChunk(audioChunk)
            }
            // Streaming chunks already yielded; return empty (caller uses chunks)
            return MLXArray.zeros([1])
        }

        // Non-streaming path: full decode (existing behavior)
        let codes = stacked(generatedCodes, axis: 1) // [1, seq_len, num_code_groups]

        var decodeCodes = codes
        if let refCodes {
            let refCodesT = refCodes.transposed(0, 2, 1)
            decodeCodes = concatenated([refCodesT, codes], axis: 1)
        }

        var audio = decodeChunk(decodeCodes)

        if let refCodes {
            let refLen = refCodes.dim(2)
            let totalLen = decodeCodes.dim(1)
            let cut = Int(Double(refLen) / Double(max(totalLen, 1)) * Double(audio.dim(0)))
            if cut > 0, cut < audio.dim(0) {
                audio = audio[cut...]
            }
        }

        eval(audio)
        return audio
    }

    // MARK: - Prepare generation inputs

    func prepareICLGenerationInputs(
        text: String,
        refAudio: MLXArray,
        refText: String,
        speakerEmbedding: MLXArray?,
        language: String
    ) -> (MLXArray, MLXArray, MLXArray, MLXArray) {
        var refAudioForEncoder = refAudio
        if refAudio.ndim == 1 {
            refAudioForEncoder = refAudio.reshaped(1, 1, refAudio.dim(0))
        } else if refAudio.ndim == 2 {
            refAudioForEncoder = refAudio.reshaped(1, refAudio.dim(0), refAudio.dim(1))
        }

        let refCodes = speechTokenizer?.encode(refAudioForEncoder) ?? MLXArray.zeros([1])
        return prepareICLGenerationInputs(
            text: text,
            refCodes: refCodes,
            refText: refText,
            speakerEmbedding: speakerEmbedding,
            language: language
        )
    }

    func prepareICLGenerationInputs(
        text: String,
        refCodes: MLXArray,
        refText: String,
        speakerEmbedding: MLXArray?,
        language: String
    ) -> (MLXArray, MLXArray, MLXArray, MLXArray) {
        guard let tokenizer, let talkerConfig = config.talkerConfig else {
            // Return zero tensors instead of crashing — caller checks for empty output
            return (MLXArray.zeros([1]), MLXArray.zeros([1]), MLXArray.zeros([1]), MLXArray.zeros([1]))
        }

        // Reference text and target text tokenization
        let refChatText = "<|im_start|>assistant\n\(refText)<|im_end|>\n"
        let refIds = MLXArray(tokenizer.encode(text: refChatText).map { Int32($0) }).reshaped(1, -1)
        let refCount = refIds.dim(1)
        let refStart = min(3, refCount)
        let refEnd = max(refStart, refCount - 2)
        let refTextIds = refIds[0..., refStart ..< refEnd]

        let targetChatText = "<|im_start|>assistant\n\(text)<|im_end|>\n<|im_start|>assistant\n"
        let targetIds = MLXArray(tokenizer.encode(text: targetChatText).map { Int32($0) }).reshaped(1, -1)
        let targetCount = targetIds.dim(1)
        let targetStart = min(3, targetCount)
        let targetEnd = max(targetStart, targetCount - 5)
        let targetTextIds = targetIds[0..., targetStart ..< targetEnd]

        // The reference prompt can come either from freshly encoded audio or a cached clone prompt.
        let normalizedRefCodes: MLXArray
        if refCodes.ndim == 3 {
            normalizedRefCodes = refCodes
        } else if refCodes.ndim == 2 {
            normalizedRefCodes = refCodes.reshaped(1, refCodes.dim(0), refCodes.dim(1))
        } else {
            normalizedRefCodes = MLXArray.zeros([1])
        }

        let ttsEmbeddings = cachedTTSSpecialEmbeddings()
        let ttsBosEmbed = ttsEmbeddings.bos
        let ttsEosEmbed = ttsEmbeddings.eos
        let ttsPadEmbed = ttsEmbeddings.pad

        // Build text embeddings for ref+target
        let combinedTextIds = concatenated([refTextIds, targetTextIds], axis: 1)
        var textEmbed = talker.textProjection(talker.getTextEmbeddings()(combinedTextIds))
        textEmbed = concatenated([textEmbed, ttsEosEmbed], axis: 1)
        let textLen = textEmbed.dim(1)

        // Build codec embeddings from reference codes: codec_bos + sum of all codebook embeddings
        let firstCbCodes = normalizedRefCodes[0..., 0, 0...]
        var refCodecEmbed = talker.getInputEmbeddings()(firstCbCodes)
        if talkerConfig.numCodeGroups > 1 {
            for i in 0 ..< (talkerConfig.numCodeGroups - 1) {
                let codeIdx = i + 1
                if codeIdx >= normalizedRefCodes.dim(1) { break }
                let cbCodes = normalizedRefCodes[0..., codeIdx, 0...]
                refCodecEmbed = refCodecEmbed + talker.codePredictor.codecEmbedding[i](cbCodes)
            }
        }

        let codecBosEmbed = cachedCodecBosEmbedding(talkerConfig: talkerConfig)
        let codecEmbedIcl = concatenated([codecBosEmbed, refCodecEmbed], axis: 1)

        // Non-streaming overlay of text and codec contexts
        let codecPadEmbed = cachedCodecPadEmbedding(talkerConfig: talkerConfig)
        let textWithCodecPad = textEmbed + broadcast(
            codecPadEmbed,
            to: [1, textLen, codecPadEmbed.dim(-1)]
        )
        let codecWithTextPad = codecEmbedIcl + broadcast(
            ttsPadEmbed,
            to: [1, codecEmbedIcl.dim(1), ttsPadEmbed.dim(-1)]
        )

        let iclInputEmbed = concatenated([textWithCodecPad, codecWithTextPad], axis: 1)
        let trailingTextHidden = ttsPadEmbed

        var codecPrefixEmbed = cachedCodecPrefillEmbedding(
            language: language,
            talkerConfig: talkerConfig
        )
        let codecPrefixSuffix = cachedCodecPrefixSuffixEmbedding(talkerConfig: talkerConfig)
        if let speakerEmbedding {
            let speakerEmbed = speakerEmbedding.reshaped(1, 1, -1)
            codecPrefixEmbed = concatenated([codecPrefixEmbed, speakerEmbed, codecPrefixSuffix], axis: 1)
        } else {
            codecPrefixEmbed = concatenated([codecPrefixEmbed, codecPrefixSuffix], axis: 1)
        }

        // Role embedding
        let roleEmbed = talker.textProjection(talker.getTextEmbeddings()(targetIds[0..., 0 ..< 3]))

        // Build prefix: text side overlayed with codec prefix
        let padCount = codecPrefixEmbed.dim(1) - 2
        let padEmbeds = broadcast(ttsPadEmbed, to: [1, padCount, ttsPadEmbed.dim(-1)])
        var combinedPrefix = concatenated([padEmbeds, ttsBosEmbed], axis: 1)
        combinedPrefix = combinedPrefix + codecPrefixEmbed[0..., 0 ..< (codecPrefixEmbed.dim(1) - 1), 0...]

        // Full input embedding
        let inputEmbeds = concatenated([roleEmbed, combinedPrefix, iclInputEmbed], axis: 1)

        return (inputEmbeds, trailingTextHidden, ttsPadEmbed, normalizedRefCodes)
    }

    func prepareICLGenerationInputs(
        text: String,
        preparedClonePrompt: PreparedVoiceClonePromptContext,
        language: String
    ) -> (MLXArray, MLXArray, MLXArray, MLXArray) {
        guard let tokenizer, let talkerConfig = config.talkerConfig else {
            return (MLXArray.zeros([1]), MLXArray.zeros([1]), MLXArray.zeros([1]), MLXArray.zeros([1]))
        }

        let targetChatText = "<|im_start|>assistant\n\(text)<|im_end|>\n<|im_start|>assistant\n"
        let targetIds = MLXArray(tokenizer.encode(text: targetChatText).map { Int32($0) }).reshaped(1, -1)
        let targetCount = targetIds.dim(1)
        let targetStart = min(3, targetCount)
        let targetEnd = max(targetStart, targetCount - 5)
        let targetTextIds = targetIds[0..., targetStart ..< targetEnd]

        let ttsEmbeddings = cachedTTSSpecialEmbeddings()
        let ttsBosEmbed = ttsEmbeddings.bos
        let ttsEosEmbed = ttsEmbeddings.eos
        let ttsPadEmbed = ttsEmbeddings.pad

        let targetTextEmbed = talker.textProjection(talker.getTextEmbeddings()(targetTextIds))
        let textEmbed = concatenated(
            [preparedClonePrompt.refTextEmbedding, targetTextEmbed, ttsEosEmbed],
            axis: 1
        )
        let textLen = textEmbed.dim(1)

        let codecPadEmbed = cachedCodecPadEmbedding(talkerConfig: talkerConfig)
        let textWithCodecPad = textEmbed + broadcast(
            codecPadEmbed,
            to: [1, textLen, codecPadEmbed.dim(-1)]
        )
        let codecWithTextPad = preparedClonePrompt.codecContextEmbedding + broadcast(
            ttsPadEmbed,
            to: [1, preparedClonePrompt.codecContextEmbedding.dim(1), ttsPadEmbed.dim(-1)]
        )

        let iclInputEmbed = concatenated([textWithCodecPad, codecWithTextPad], axis: 1)
        let trailingTextHidden = ttsPadEmbed

        var codecPrefixEmbed = cachedCodecPrefillEmbedding(
            language: language,
            talkerConfig: talkerConfig
        )
        let codecPrefixSuffix = cachedCodecPrefixSuffixEmbedding(talkerConfig: talkerConfig)
        if let speakerEmbedding = preparedClonePrompt.speakerEmbedding {
            codecPrefixEmbed = concatenated(
                [codecPrefixEmbed, speakerEmbedding.reshaped(1, 1, -1), codecPrefixSuffix],
                axis: 1
            )
        } else {
            codecPrefixEmbed = concatenated([codecPrefixEmbed, codecPrefixSuffix], axis: 1)
        }

        let roleEmbed = talker.textProjection(talker.getTextEmbeddings()(targetIds[0..., 0 ..< 3]))
        let padCount = codecPrefixEmbed.dim(1) - 2
        let padEmbeds = broadcast(ttsPadEmbed, to: [1, padCount, ttsPadEmbed.dim(-1)])
        var combinedPrefix = concatenated([padEmbeds, ttsBosEmbed], axis: 1)
        combinedPrefix = combinedPrefix + codecPrefixEmbed[0..., 0 ..< (codecPrefixEmbed.dim(1) - 1), 0...]

        let inputEmbeds = concatenated([roleEmbed, combinedPrefix, iclInputEmbed], axis: 1)
        return (inputEmbeds, trailingTextHidden, ttsPadEmbed, preparedClonePrompt.refCodes)
    }

    func extractSpeakerEmbedding(_ refAudio: MLXArray) -> MLXArray? {
        guard let speakerEncoder else { return nil }

        let rawAudio: MLXArray
        if refAudio.ndim == 1 {
            rawAudio = refAudio.reshaped(1, refAudio.dim(0))
        } else if refAudio.ndim == 2 {
            if refAudio.dim(0) == 1 {
                rawAudio = refAudio
            } else {
                rawAudio = refAudio[0 ..< 1]
            }
        } else if refAudio.ndim == 3, refAudio.dim(1) == 1 {
            let squeezed = refAudio[0..., 0...]
            if squeezed.dim(0) == 1 {
                rawAudio = squeezed
            } else {
                rawAudio = squeezed[0 ..< 1]
            }
        } else {
            return nil
        }

        let batchSize = rawAudio.dim(0)
        var mels = [MLXArray]()
        mels.reserveCapacity(batchSize)

        for batch in 0 ..< batchSize {
            let waveform = rawAudio[batch]
            let mel = computeMelSpectrogram(
                audio: waveform,
                sampleRate: speakerEncoder.config.sampleRate,
                nFft: 1024,
                hopLength: 256,
                nMels: 128
            )
            mels.append(mel)
        }

        let stackedMels = stacked(mels, axis: 0)
        let embedding = speakerEncoder(stackedMels)
        return embedding
    }

    public func createVoiceClonePromptPayload(
        refAudio: MLXArray,
        refText: String?,
        xVectorOnlyMode: Bool = false,
        speakerEmbeddingOverride: MLXArray? = nil
    ) throws -> Data {
        guard config.ttsModelType == "base" else {
            throw AudioGenerationError.invalidInput(
                "Qwen3-TTS clone prompt creation requires the Base model."
            )
        }

        let trimmedRefText = refText?.trimmingCharacters(in: .whitespacesAndNewlines)
        let iclMode = !xVectorOnlyMode
        if iclMode, trimmedRefText?.isEmpty != false {
            throw AudioGenerationError.invalidInput(
                "Reference transcript is required when creating an ICL voice clone prompt."
            )
        }

        let speakerEmbedding = speakerEmbeddingOverride ?? extractSpeakerEmbedding(refAudio)
        let refCodes: MLXArray?
        if iclMode {
            guard let speechTokenizer, speechTokenizer.hasEncoder else {
                throw AudioGenerationError.invalidInput(
                    "Speech tokenizer encoder is unavailable for clone-prompt extraction."
                )
            }
            let normalizedReferenceAudio = normalizedReferenceAudioForTokenizer(refAudio)
            refCodes = speechTokenizer.encode(normalizedReferenceAudio).asType(.int32)
        } else {
            refCodes = nil
        }

        let payload = VoiceClonePromptPayloadV1(
            refSpeakerEmbedding: float32Data(from: speakerEmbedding),
            refCode: int32Data(from: refCodes),
            refText: trimmedRefText,
            numCodeGroups: refCodes?.dim(1),
            frameCount: refCodes?.dim(2),
            xVectorOnlyMode: xVectorOnlyMode,
            iclMode: iclMode
        )
        return try JSONEncoder().encode(payload)
    }

    public func createVoiceClonePromptConditioning(
        referenceAudio: MLXArray,
        referenceTranscript: String?,
        cachedSpeakerEmbedding: Data? = nil
    ) throws -> SpeechConditioning {
        let speakerEmbedding = try float32Array(from: cachedSpeakerEmbedding)
            ?? extractSpeakerEmbedding(referenceAudio)
        let trimmedTranscript = referenceTranscript?.trimmingCharacters(in: .whitespacesAndNewlines)
        let xVectorOnlyMode = trimmedTranscript?.isEmpty != false
        let encoded = try createVoiceClonePromptPayload(
            refAudio: referenceAudio,
            refText: trimmedTranscript,
            xVectorOnlyMode: xVectorOnlyMode,
            speakerEmbeddingOverride: speakerEmbedding
        )
        return SpeechConditioning(format: "qwen.clone_prompt/v1", payload: encoded)
    }

    func prepareGenerationInputs(
        text: String,
        language: String,
        speaker: String?,
        speakerEmbedding: MLXArray?,
        instruct: String?
    ) -> (MLXArray, MLXArray, MLXArray) {
        guard let tokenizer, let talkerConfig = config.talkerConfig else {
            return (MLXArray.zeros([1]), MLXArray.zeros([1]), MLXArray.zeros([1]))
        }

        // Tokenize text with ChatML template
        let chatText = "<|im_start|>assistant\n\(text)<|im_end|>\n<|im_start|>assistant\n"
        let inputIds = MLXArray(tokenizer.encode(text: chatText).map { Int32($0) }).reshaped(1, -1)

        // Get text embeddings
        let textEmbed = talker.textProjection(talker.getTextEmbeddings()(inputIds))

        let ttsEmbeddings = cachedTTSSpecialEmbeddings()
        let ttsBosEmbed = ttsEmbeddings.bos
        let ttsEosEmbed = ttsEmbeddings.eos
        let ttsPadEmbed = ttsEmbeddings.pad

        var codecEmbed = cachedCodecPrefillEmbedding(language: language, talkerConfig: talkerConfig)
        if let resolvedSpeakerEmbedding = resolvedSpeakerEmbedding(
            namedSpeaker: speaker,
            explicitEmbedding: speakerEmbedding,
            talkerConfig: talkerConfig
        ) {
            codecEmbed = concatenated([codecEmbed, resolvedSpeakerEmbedding.reshaped(1, 1, -1)], axis: 1)
        }
        let codecEmbedSuffix = cachedCodecPrefixSuffixEmbedding(talkerConfig: talkerConfig)
        codecEmbed = concatenated([codecEmbed, codecEmbedSuffix], axis: 1)

        // Instruct embedding
        var instructEmbed: MLXArray?
        if let instruct, !instruct.isEmpty {
            let instructText = "<|im_start|>user\n\(instruct)<|im_end|>\n"
            let instructIds = MLXArray(tokenizer.encode(text: instructText).map { Int32($0) }).reshaped(1, -1)
            instructEmbed = talker.textProjection(talker.getTextEmbeddings()(instructIds))
        }

        // Role embedding (first 3 tokens: <|im_start|>assistant\n)
        let roleEmbed = textEmbed[0..., ..<3, 0...]

        // Build pad/bos prefix
        let padCount = codecEmbed.dim(1) - 2
        let padEmbeds = broadcast(ttsPadEmbed, to: [1, padCount, ttsPadEmbed.dim(-1)])
        var combinedEmbed = concatenated([padEmbeds, ttsBosEmbed], axis: 1)
        combinedEmbed = combinedEmbed + codecEmbed[0..., ..<(-1), 0...]

        // Full input embedding
        var inputEmbeds: MLXArray = if let instructEmbed {
            concatenated([instructEmbed, roleEmbed, combinedEmbed], axis: 1)
        } else {
            concatenated([roleEmbed, combinedEmbed], axis: 1)
        }

        // Add first text token (index 3) + last codec embed
        let firstTextEmbed = textEmbed[0..., 3 ..< 4, 0...] + codecEmbed[0..., (-1)..., 0...]
        inputEmbeds = concatenated([inputEmbeds, firstTextEmbed], axis: 1)

        // Trailing text (tokens 4 to -5, plus EOS)
        let trailingTextHidden = concatenated(
            [textEmbed[0..., 4 ..< (textEmbed.dim(1) - 5), 0...], ttsEosEmbed],
            axis: 1
        )

        return (inputEmbeds, trailingTextHidden, ttsPadEmbed)
    }

    private func resolvedSpeakerEmbedding(
        namedSpeaker: String?,
        explicitEmbedding: MLXArray?,
        talkerConfig: Qwen3TTSTalkerConfig
    ) -> MLXArray? {
        if let explicitEmbedding {
            return explicitEmbedding
        }

        guard let normalizedSpeaker = normalized(namedSpeaker),
              let speakerID = talkerConfig.spkId?[normalizedSpeaker]?.values.first else {
            return nil
        }

        let speakerTokens = MLXArray([Int32(speakerID)]).reshaped(1, 1)
        return talker.getInputEmbeddings()(speakerTokens)
    }

    func prepareConditionedGenerationInputs(
        text: String,
        speakerEmbedding: MLXArray,
        language: String,
        instruct: String?
    ) -> (MLXArray, MLXArray, MLXArray) {
        guard let tokenizer, let talkerConfig = config.talkerConfig else {
            return (MLXArray.zeros([1]), MLXArray.zeros([1]), MLXArray.zeros([1]))
        }

        let chatText = "<|im_start|>assistant\n\(text)<|im_end|>\n<|im_start|>assistant\n"
        let inputIds = MLXArray(tokenizer.encode(text: chatText).map { Int32($0) }).reshaped(1, -1)
        let textEmbed = talker.textProjection(talker.getTextEmbeddings()(inputIds))

        let ttsEmbeddings = cachedTTSSpecialEmbeddings()
        let ttsBosEmbed = ttsEmbeddings.bos
        let ttsEosEmbed = ttsEmbeddings.eos
        let ttsPadEmbed = ttsEmbeddings.pad

        var codecEmbed = cachedCodecPrefillEmbedding(language: language, talkerConfig: talkerConfig)
        let speakerEmbed = speakerEmbedding.reshaped(1, 1, -1)
        let codecEmbedSuffix = cachedCodecPrefixSuffixEmbedding(talkerConfig: talkerConfig)
        codecEmbed = concatenated([codecEmbed, speakerEmbed, codecEmbedSuffix], axis: 1)

        var instructEmbed: MLXArray?
        if let instruct, !instruct.isEmpty {
            let instructText = "<|im_start|>user\n\(instruct)<|im_end|>\n"
            let instructIds = MLXArray(tokenizer.encode(text: instructText).map { Int32($0) }).reshaped(1, -1)
            instructEmbed = talker.textProjection(talker.getTextEmbeddings()(instructIds))
        }

        let roleEmbed = textEmbed[0..., ..<3, 0...]
        let padCount = codecEmbed.dim(1) - 2
        let padEmbeds = broadcast(ttsPadEmbed, to: [1, padCount, ttsPadEmbed.dim(-1)])
        var combinedEmbed = concatenated([padEmbeds, ttsBosEmbed], axis: 1)
        combinedEmbed = combinedEmbed + codecEmbed[0..., ..<(-1), 0...]

        var inputEmbeds: MLXArray = if let instructEmbed {
            concatenated([instructEmbed, roleEmbed, combinedEmbed], axis: 1)
        } else {
            concatenated([roleEmbed, combinedEmbed], axis: 1)
        }

        let firstTextEmbed = textEmbed[0..., 3 ..< 4, 0...] + codecEmbed[0..., (-1)..., 0...]
        inputEmbeds = concatenated([inputEmbeds, firstTextEmbed], axis: 1)

        let trailingTextHidden = concatenated(
            [textEmbed[0..., 4 ..< (textEmbed.dim(1) - 5), 0...], ttsEosEmbed],
            axis: 1
        )

        return (inputEmbeds, trailingTextHidden, ttsPadEmbed)
    }

    func resolvedVoiceClonePrompt(from conditioning: SpeechConditioning?) throws -> ResolvedVoiceClonePrompt? {
        guard let conditioning else { return nil }
        switch conditioning.format {
        case "qwen.speaker_embedding/v1":
            return nil
        case "qwen.clone_prompt/v1":
            break
        default:
            throw AudioGenerationError.invalidInput(
                "Qwen3TTS only supports qwen.speaker_embedding/v1 or qwen.clone_prompt/v1 conditioning, got '\(conditioning.format)'."
            )
        }

        guard !conditioning.payload.isEmpty else {
            return nil
        }

        let cacheKey = conditioning.payload as NSData
        if let cached = resolvedClonePromptCache.object(forKey: cacheKey) {
            return cached.prompt
        }

        let resolvedPrompt: ResolvedVoiceClonePrompt
        if let payload = try? JSONDecoder().decode(VoiceClonePromptPayloadV1.self, from: conditioning.payload) {
            resolvedPrompt = ResolvedVoiceClonePrompt(
                speakerEmbedding: try float32Array(from: payload.refSpeakerEmbedding),
                refCodes: try int32CodeArray(
                    from: payload.refCode,
                    numCodeGroups: payload.numCodeGroups,
                    frameCount: payload.frameCount
                ),
                refText: normalized(payload.refText),
                xVectorOnlyMode: payload.xVectorOnlyMode,
                iclMode: payload.iclMode
            )
        } else {
            // Legacy fallback: early recovery builds stored only a raw Float32 speaker embedding
            // while already labeling the payload as qwen.clone_prompt/v1.
            let legacyEmbedding = try float32Array(from: conditioning.payload)
            resolvedPrompt = ResolvedVoiceClonePrompt(
                speakerEmbedding: legacyEmbedding,
                refCodes: nil,
                refText: nil,
                xVectorOnlyMode: true,
                iclMode: false
            )
        }

        resolvedClonePromptCache.setObject(ResolvedVoiceClonePromptBox(resolvedPrompt), forKey: cacheKey)
        return resolvedPrompt
    }

    public func primeReusableConditioningCaches(for conditioning: SpeechConditioning?) throws {
        guard let conditioning else { return }
        let resolvedPrompt = try resolvedVoiceClonePrompt(from: conditioning)
        _ = try preparedVoiceClonePromptContext(
            from: conditioning,
            resolvedPrompt: resolvedPrompt
        )
        _ = try resolvedSpeakerEmbedding(from: conditioning)
    }

    private func preparedVoiceClonePromptContext(
        from conditioning: SpeechConditioning?,
        resolvedPrompt: ResolvedVoiceClonePrompt?
    ) throws -> PreparedVoiceClonePromptContext? {
        guard let conditioning,
              conditioning.format == "qwen.clone_prompt/v1",
              let resolvedPrompt,
              resolvedPrompt.iclMode,
              let refText = normalized(resolvedPrompt.refText),
              let refCodes = resolvedPrompt.refCodes,
              let tokenizer,
              let talkerConfig = config.talkerConfig else {
            return nil
        }

        let cacheKey = conditioning.payload as NSData
        if let cached = preparedClonePromptContextCache.object(forKey: cacheKey) {
            return cached.context
        }

        let refChatText = "<|im_start|>assistant\n\(refText)<|im_end|>\n"
        let refIds = MLXArray(tokenizer.encode(text: refChatText).map { Int32($0) }).reshaped(1, -1)
        let refCount = refIds.dim(1)
        let refStart = min(3, refCount)
        let refEnd = max(refStart, refCount - 2)
        let refTextIds = refIds[0..., refStart ..< refEnd]
        let refTextEmbedding = talker.textProjection(talker.getTextEmbeddings()(refTextIds))

        let firstCbCodes = refCodes[0..., 0, 0...]
        var refCodecEmbed = talker.getInputEmbeddings()(firstCbCodes)
        if talkerConfig.numCodeGroups > 1 {
            for i in 0 ..< (talkerConfig.numCodeGroups - 1) {
                let codeIdx = i + 1
                if codeIdx >= refCodes.dim(1) { break }
                let cbCodes = refCodes[0..., codeIdx, 0...]
                refCodecEmbed = refCodecEmbed + talker.codePredictor.codecEmbedding[i](cbCodes)
            }
        }
        let codecBosEmbed = cachedCodecBosEmbedding(talkerConfig: talkerConfig)
        let codecContextEmbedding = concatenated([codecBosEmbed, refCodecEmbed], axis: 1)

        let prepared = PreparedVoiceClonePromptContext(
            speakerEmbedding: resolvedPrompt.speakerEmbedding,
            refCodes: refCodes,
            refTextEmbedding: refTextEmbedding,
            codecContextEmbedding: codecContextEmbedding
        )
        preparedClonePromptContextCache.setObject(
            PreparedVoiceClonePromptContextBox(prepared),
            forKey: cacheKey
        )
        return prepared
    }

    func resolvedSpeakerEmbedding(from conditioning: SpeechConditioning?) throws -> MLXArray? {
        guard let conditioning else { return nil }
        switch conditioning.format {
        case "qwen.speaker_embedding/v1":
            guard !conditioning.payload.isEmpty else {
                return nil
            }
            let cacheKey = conditioning.payload as NSData
            if let cached = resolvedSpeakerEmbeddingCache.object(forKey: cacheKey) {
                return cached.array
            }
            let resolved = try float32Array(from: conditioning.payload)
            if let resolved {
                resolvedSpeakerEmbeddingCache.setObject(MLXArrayBox(resolved), forKey: cacheKey)
            }
            return resolved
        case "qwen.clone_prompt/v1":
            return nil
        default:
            throw AudioGenerationError.invalidInput(
                "Qwen3TTS only supports qwen.speaker_embedding/v1 or qwen.clone_prompt/v1 conditioning, got '\(conditioning.format)'."
            )
        }
    }

    public func prepareStableNarratorPromptPhases(
        anchorConditioning: SpeechConditioning?,
        continuationConditioning: SpeechConditioning?,
        continuationInstruct: String?,
        language: String
    ) throws -> PreparedStableNarratorPromptPhases? {
        let resolvedAnchorClonePrompt = try resolvedVoiceClonePrompt(from: anchorConditioning)
        guard let preparedAnchorClonePrompt = try preparedVoiceClonePromptContext(
            from: anchorConditioning,
            resolvedPrompt: resolvedAnchorClonePrompt
        ) else {
            return nil
        }

        let anchorPrefix = prepareStableNarratorAnchorPrefix(
            preparedClonePrompt: preparedAnchorClonePrompt,
            language: language
        )
        let anchor = PreparedStableNarratorPromptPhase(
            prefixCache: try prefilledTalkerPrefixCache(
                phase: .anchor,
                prefixEmbeddings: anchorPrefix
            ),
            speakerEmbedding: preparedAnchorClonePrompt.speakerEmbedding,
            preparedClonePrompt: preparedAnchorClonePrompt,
            language: language
        )

        let continuationSpeakerEmbedding = try resolvedSpeakerEmbedding(from: continuationConditioning)
        let continuation: PreparedStableNarratorPromptPhase? = try continuationSpeakerEmbedding.map { embedding in
            let prefix = prepareStableNarratorContinuationPrefix(
                speakerEmbedding: embedding,
                instruct: continuationInstruct,
                language: language
            )
            return PreparedStableNarratorPromptPhase(
                prefixCache: try prefilledTalkerPrefixCache(
                    phase: .continuation,
                    prefixEmbeddings: prefix
                ),
                speakerEmbedding: embedding,
                preparedClonePrompt: nil,
                language: language
            )
        }

        return PreparedStableNarratorPromptPhases(
            anchor: anchor,
            continuation: continuation
        )
    }

    private func prefilledTalkerPrefixCache(
        phase: PreparedTalkerPromptPhase,
        prefixEmbeddings: MLXArray
    ) throws -> PreparedTalkerPrefixCache {
        let talkerCache = talker.makeCache()
        _ = talker(prefixEmbeddings, cache: talkerCache)
        eval(prefixEmbeddings)
        return PreparedTalkerPrefixCache(
            phase: phase,
            prefixEmbeddings: prefixEmbeddings,
            cacheSnapshot: snapshotKVCaches(talkerCache),
            prefixTokenCount: prefixEmbeddings.dim(1)
        )
    }

    private func prepareStableNarratorAnchorPrefix(
        preparedClonePrompt: PreparedVoiceClonePromptContext,
        language: String
    ) -> MLXArray {
        guard let tokenizer, let talkerConfig = config.talkerConfig else {
            return MLXArray.zeros([1])
        }

        let assistantPrefixIDs = MLXArray(
            tokenizer.encode(text: "<|im_start|>assistant\n").map { Int32($0) }
        ).reshaped(1, -1)
        let roleEmbed = talker.textProjection(talker.getTextEmbeddings()(assistantPrefixIDs[0..., 0 ..< 3]))

        let ttsEmbeddings = cachedTTSSpecialEmbeddings()
        let ttsBosEmbed = ttsEmbeddings.bos
        let ttsPadEmbed = ttsEmbeddings.pad
        let codecPadEmbed = cachedCodecPadEmbedding(talkerConfig: talkerConfig)

        var codecPrefixEmbed = cachedCodecPrefillEmbedding(
            language: language,
            talkerConfig: talkerConfig
        )
        let codecPrefixSuffix = cachedCodecPrefixSuffixEmbedding(talkerConfig: talkerConfig)
        if let speakerEmbedding = preparedClonePrompt.speakerEmbedding {
            codecPrefixEmbed = concatenated(
                [codecPrefixEmbed, speakerEmbedding.reshaped(1, 1, -1), codecPrefixSuffix],
                axis: 1
            )
        } else {
            codecPrefixEmbed = concatenated([codecPrefixEmbed, codecPrefixSuffix], axis: 1)
        }

        let padCount = codecPrefixEmbed.dim(1) - 2
        let padEmbeds = broadcast(ttsPadEmbed, to: [1, padCount, ttsPadEmbed.dim(-1)])
        var combinedPrefix = concatenated([padEmbeds, ttsBosEmbed], axis: 1)
        combinedPrefix = combinedPrefix + codecPrefixEmbed[0..., 0 ..< (codecPrefixEmbed.dim(1) - 1), 0...]

        let invariantTextWithCodecPad = preparedClonePrompt.refTextEmbedding + broadcast(
            codecPadEmbed,
            to: [1, preparedClonePrompt.refTextEmbedding.dim(1), codecPadEmbed.dim(-1)]
        )
        let codecWithTextPad = preparedClonePrompt.codecContextEmbedding + broadcast(
            ttsPadEmbed,
            to: [1, preparedClonePrompt.codecContextEmbedding.dim(1), ttsPadEmbed.dim(-1)]
        )

        return concatenated([roleEmbed, combinedPrefix, invariantTextWithCodecPad, codecWithTextPad], axis: 1)
    }

    private func prepareStableNarratorContinuationPrefix(
        speakerEmbedding: MLXArray,
        instruct: String?,
        language: String
    ) -> MLXArray {
        guard let tokenizer, let talkerConfig = config.talkerConfig else {
            return MLXArray.zeros([1])
        }

        let assistantPrefixIDs = MLXArray(
            tokenizer.encode(text: "<|im_start|>assistant\n").map { Int32($0) }
        ).reshaped(1, -1)
        let roleEmbed = talker.textProjection(talker.getTextEmbeddings()(assistantPrefixIDs[0..., 0 ..< 3]))

        let ttsEmbeddings = cachedTTSSpecialEmbeddings()
        let ttsBosEmbed = ttsEmbeddings.bos
        let ttsPadEmbed = ttsEmbeddings.pad
        var codecEmbed = cachedCodecPrefillEmbedding(language: language, talkerConfig: talkerConfig)
        let codecEmbedSuffix = cachedCodecPrefixSuffixEmbedding(talkerConfig: talkerConfig)
        codecEmbed = concatenated([codecEmbed, speakerEmbedding.reshaped(1, 1, -1), codecEmbedSuffix], axis: 1)

        let padCount = codecEmbed.dim(1) - 2
        let padEmbeds = broadcast(ttsPadEmbed, to: [1, padCount, ttsPadEmbed.dim(-1)])
        var combinedEmbed = concatenated([padEmbeds, ttsBosEmbed], axis: 1)
        combinedEmbed = combinedEmbed + codecEmbed[0..., ..<(-1), 0...]

        if let instruct, !instruct.isEmpty {
            let instructText = "<|im_start|>user\n\(instruct)<|im_end|>\n"
            let instructIDs = MLXArray(tokenizer.encode(text: instructText).map { Int32($0) }).reshaped(1, -1)
            let instructEmbed = talker.textProjection(talker.getTextEmbeddings()(instructIDs))
            return concatenated([instructEmbed, roleEmbed, combinedEmbed], axis: 1)
        }

        return concatenated([roleEmbed, combinedEmbed], axis: 1)
    }

    private func prepareSegmentExecutionInputs(
        text: String,
        language: String,
        speaker: String?,
        instruct: String?,
        speakerEmbedding: MLXArray?,
        voiceClonePrompt: ResolvedVoiceClonePrompt?,
        preparedClonePrompt: PreparedVoiceClonePromptContext?,
        refAudio: MLXArray?,
        refText: String?,
        preparedStablePromptPhase: PreparedStableNarratorPromptPhase?
    ) throws -> PreparedPromptExecutionInputs {
        if let preparedStablePromptPhase {
            switch preparedStablePromptPhase.prefixCache.phase {
            case .anchor:
                guard let preparedClonePrompt = preparedStablePromptPhase.preparedClonePrompt else {
                    throw AudioGenerationError.invalidInput(
                        "Stable narrator anchor phase requires prepared clone-prompt conditioning."
                    )
                }
                return prepareStableNarratorAnchorInputs(
                    text: text,
                    preparedClonePrompt: preparedClonePrompt,
                    language: preparedStablePromptPhase.language,
                    prefixCache: preparedStablePromptPhase.prefixCache
                )
            case .continuation:
                guard let speakerEmbedding = preparedStablePromptPhase.speakerEmbedding else {
                    throw AudioGenerationError.invalidInput(
                        "Stable narrator continuation phase requires a speaker embedding."
                    )
                }
                return prepareStableNarratorContinuationInputs(
                    text: text,
                    speakerEmbedding: speakerEmbedding,
                    language: preparedStablePromptPhase.language,
                    instruct: instruct,
                    prefixCache: preparedStablePromptPhase.prefixCache
                )
            }
        }

        let resolvedRefText = normalized(refText) ?? voiceClonePrompt?.refText
        let useCachedClonePrompt = voiceClonePrompt?.iclMode == true
            && voiceClonePrompt?.refCodes != nil
            && resolvedRefText != nil
        let useICL = useCachedClonePrompt || (refAudio != nil && resolvedRefText != nil && speechTokenizer?.hasEncoder == true)

        if useICL, let resolvedRefText {
            let prepared: (MLXArray, MLXArray, MLXArray, MLXArray)
            if let preparedClonePrompt, useCachedClonePrompt {
                prepared = prepareICLGenerationInputs(
                    text: text,
                    preparedClonePrompt: preparedClonePrompt,
                    language: language
                )
            } else if let cachedRefCodes = voiceClonePrompt?.refCodes, voiceClonePrompt?.iclMode == true {
                prepared = prepareICLGenerationInputs(
                    text: text,
                    refCodes: cachedRefCodes,
                    refText: resolvedRefText,
                    speakerEmbedding: speakerEmbedding,
                    language: language
                )
            } else if let refAudio {
                let extractedSpeakerEmbedding = speakerEmbedding ?? extractSpeakerEmbedding(refAudio)
                prepared = prepareICLGenerationInputs(
                    text: text,
                    refAudio: refAudio,
                    refText: resolvedRefText,
                    speakerEmbedding: extractedSpeakerEmbedding,
                    language: language
                )
            } else {
                return PreparedPromptExecutionInputs(
                    promptEmbeddings: MLXArray.zeros([1]),
                    trailingTextHidden: MLXArray.zeros([1]),
                    ttsPadEmbed: MLXArray.zeros([1]),
                    refCodes: nil,
                    prefixCache: nil
                )
            }
            return PreparedPromptExecutionInputs(
                promptEmbeddings: prepared.0,
                trailingTextHidden: prepared.1,
                ttsPadEmbed: prepared.2,
                refCodes: prepared.3,
                prefixCache: nil
            )
        }

        let prepared = prepareGenerationInputs(
            text: text,
            language: language,
            speaker: speaker,
            speakerEmbedding: speakerEmbedding,
            instruct: instruct
        )
        return PreparedPromptExecutionInputs(
            promptEmbeddings: prepared.0,
            trailingTextHidden: prepared.1,
            ttsPadEmbed: prepared.2,
            refCodes: nil,
            prefixCache: nil
        )
    }

    private func prepareStableNarratorAnchorInputs(
        text: String,
        preparedClonePrompt: PreparedVoiceClonePromptContext,
        language: String,
        prefixCache: PreparedTalkerPrefixCache
    ) -> PreparedPromptExecutionInputs {
        guard tokenizer != nil, config.talkerConfig != nil else {
            return PreparedPromptExecutionInputs(
                promptEmbeddings: MLXArray.zeros([1]),
                trailingTextHidden: MLXArray.zeros([1]),
                ttsPadEmbed: MLXArray.zeros([1]),
                refCodes: nil,
                prefixCache: prefixCache
            )
        }
        let prepared = prepareICLGenerationInputs(
            text: text,
            preparedClonePrompt: preparedClonePrompt,
            language: language
        )
        let targetTextWithCodecPad = prepared.0[0..., prefixCache.prefixTokenCount..., 0...]

        return PreparedPromptExecutionInputs(
            promptEmbeddings: targetTextWithCodecPad,
            trailingTextHidden: prepared.1,
            ttsPadEmbed: prepared.2,
            refCodes: preparedClonePrompt.refCodes,
            prefixCache: prefixCache
        )
    }

    private func prepareStableNarratorContinuationInputs(
        text: String,
        speakerEmbedding: MLXArray,
        language: String,
        instruct: String?,
        prefixCache: PreparedTalkerPrefixCache
    ) -> PreparedPromptExecutionInputs {
        guard tokenizer != nil else {
            return PreparedPromptExecutionInputs(
                promptEmbeddings: MLXArray.zeros([1]),
                trailingTextHidden: MLXArray.zeros([1]),
                ttsPadEmbed: MLXArray.zeros([1]),
                refCodes: nil,
                prefixCache: prefixCache
            )
        }
        let prepared = prepareConditionedGenerationInputs(
            text: text,
            speakerEmbedding: speakerEmbedding,
            language: language,
            instruct: instruct
        )
        let promptEmbeddings = prepared.0[0..., prefixCache.prefixTokenCount..., 0...]
        return PreparedPromptExecutionInputs(
            promptEmbeddings: promptEmbeddings,
            trailingTextHidden: prepared.1,
            ttsPadEmbed: prepared.2,
            refCodes: nil,
            prefixCache: prefixCache
        )
    }

    private func normalizedReferenceAudioForTokenizer(_ referenceAudio: MLXArray) -> MLXArray {
        if referenceAudio.ndim == 1 {
            return referenceAudio.reshaped(1, 1, referenceAudio.dim(0))
        }
        if referenceAudio.ndim == 2 {
            return referenceAudio.reshaped(1, referenceAudio.dim(0), referenceAudio.dim(1))
        }
        return referenceAudio
    }

    private func float32Array(from data: Data?) throws -> MLXArray? {
        guard let data, !data.isEmpty else { return nil }
        let stride = MemoryLayout<Float>.size
        guard data.count.isMultiple(of: stride) else {
            throw AudioGenerationError.invalidInput("Speaker embedding payload is not valid Float32 data.")
        }

        let count = data.count / stride
        let floats = data.withUnsafeBytes { raw in
            (0 ..< count).map { index in
                raw.load(fromByteOffset: index * stride, as: Float.self)
            }
        }
        guard !floats.isEmpty else {
            return nil
        }
        return MLXArray(floats)
    }

    private func int32CodeArray(
        from data: Data?,
        numCodeGroups: Int?,
        frameCount: Int?
    ) throws -> MLXArray? {
        guard let data, !data.isEmpty else { return nil }
        guard let numCodeGroups, let frameCount, numCodeGroups > 0, frameCount > 0 else {
            throw AudioGenerationError.invalidInput(
                "Clone prompt payload is missing codec code dimensions."
            )
        }

        let stride = MemoryLayout<Int32>.size
        guard data.count.isMultiple(of: stride) else {
            throw AudioGenerationError.invalidInput("Clone prompt ref_code payload is not valid Int32 data.")
        }

        let count = data.count / stride
        guard count == numCodeGroups * frameCount else {
            throw AudioGenerationError.invalidInput(
                "Clone prompt ref_code payload size does not match its stored dimensions."
            )
        }

        let values = data.withUnsafeBytes { raw in
            (0 ..< count).map { index in
                raw.load(fromByteOffset: index * stride, as: Int32.self)
            }
        }
        return MLXArray(values).reshaped(1, numCodeGroups, frameCount)
    }

    private func float32Data(from array: MLXArray?) -> Data? {
        guard let array else { return nil }
        let values = array.asType(.float32).reshaped(-1).asArray(Float.self)
        guard !values.isEmpty else { return nil }
        return values.withUnsafeBufferPointer { Data(buffer: $0) }
    }

    private func int32Data(from codes: MLXArray?) -> Data? {
        guard let codes else { return nil }
        let normalizedCodes: MLXArray
        if codes.ndim == 3 {
            normalizedCodes = codes
        } else if codes.ndim == 2 {
            normalizedCodes = codes.reshaped(1, codes.dim(0), codes.dim(1))
        } else {
            return nil
        }
        let values = normalizedCodes.asType(.int32).reshaped(-1).asArray(Int32.self)
        guard !values.isEmpty else { return nil }
        return values.withUnsafeBufferPointer { Data(buffer: $0) }
    }

    // MARK: - Token sampling

    func sampleToken(
        _ logits: MLXArray,
        temperature: Float = 0.9,
        topP: Float = 1.0,
        topK: Int = 50,
        repetitionPenalty: Float = 1.0,
        generatedTokens: [Int]? = nil,
        uniqueGeneratedTokens: [Int]? = nil,
        suppressTokens: [Int]? = nil,
        eosTokenId: Int? = nil,
        minP: Float = 0.0
    ) -> MLXArray {
        var logitsSlice = logits[0..., (-1)..., 0...].squeezed(axis: 1) // [batch, vocab_size]
        let vocabSize = logitsSlice.dim(-1)

        // Suppress tokens by setting to -inf
        if let suppress = suppressTokens, !suppress.isEmpty {
            let suppressArr = MLXArray(suppress.map { Int32($0) }).reshaped(1, -1)
            let negInf = MLXArray.full([1, suppress.count], values: MLXArray(-Float.infinity), dtype: logitsSlice.dtype)
            logitsSlice = putAlong(logitsSlice, suppressArr, values: negInf, axis: -1)
        }

        // Repetition penalty
        if let tokens = generatedTokens, !tokens.isEmpty, repetitionPenalty != 1.0 {
            let repetitionTokens = (uniqueGeneratedTokens ?? Array(Set(tokens)))
                .filter { $0 >= 0 && $0 < vocabSize }
            if !repetitionTokens.isEmpty {
                let tokenIds = MLXArray(repetitionTokens.map { Int32($0) }).reshaped(1, -1)
                let selected = takeAlong(logitsSlice, tokenIds, axis: -1)
                let penalized = which(
                    selected .< 0,
                    selected * repetitionPenalty,
                    selected / repetitionPenalty
                )
                logitsSlice = putAlong(logitsSlice, tokenIds, values: penalized, axis: -1)
            }
        }

        // Greedy if temperature 0
        if temperature <= 0 {
            return argMax(logitsSlice, axis: -1, keepDims: true)
        }

        // The code predictor samples 15 times per frame with top-k already clamped to a
        // tiny candidate set. Restricting top-p/min-p to that reduced set avoids repeated
        // full-vocabulary sorts and softmaxes in the hottest decode path.
        if let reducedCandidateToken = Self.reducedCandidateSampleToken(
                logitsSlice,
                temperature: temperature,
                topP: topP,
                topK: topK,
                minP: minP,
                eosTokenId: eosTokenId
           ) {
            return reducedCandidateToken
        }

        // Preserve EOS logit so top-k/top-p/min-p do not permanently suppress it.
        let eosLogit: MLXArray? = if let eosTokenId, eosTokenId >= 0, eosTokenId < logitsSlice.dim(-1) {
            logitsSlice[0..., eosTokenId ..< (eosTokenId + 1)]
        } else {
            nil
        }

        // Narrow candidate filtering to the top-k slice when possible so we do not
        // sort and softmax the full vocabulary on every Qwen talker/codebook step.
        let candidateCount = if topK > 0, topK < vocabSize {
            min(topK, vocabSize)
        } else {
            vocabSize
        }
        let hasCandidateSubset = candidateCount < vocabSize
        let candidateIndices: MLXArray?
        if hasCandidateSubset {
            let kth = max(0, candidateCount - 1)
            candidateIndices = argPartition(-logitsSlice, kth: kth, axis: -1)[0..., ..<candidateCount].asType(.int32)
        } else {
            candidateIndices = nil
        }
        var filteredCandidateLogits = if let candidateIndices {
            takeAlong(logitsSlice, candidateIndices, axis: -1)
        } else {
            logitsSlice
        }

        // Apply top-p (nucleus) sampling
        if topP > 0, topP < 1.0 {
            let probs = softmax(filteredCandidateLogits, axis: -1)

            // Sort in ASCENDING order (like Python)
            let sortedIndices = argSort(filteredCandidateLogits, axis: -1).asType(.int32)
            let sortedProbs = takeAlong(probs, sortedIndices, axis: -1)

            // Cumulative probabilities
            let cumProbs = cumsum(sortedProbs, axis: -1)

            // Rearrange cumulative probs back to original order
            // Create inverse index mapping using putAlong
            let sortedCount = sortedIndices.dim(-1)
            let arangeIndices = MLXArray(0 ..< sortedCount).reshaped(1, -1).asType(Int32.self)
            let zeros = MLXArray.zeros(sortedIndices.shape, type: Int32.self)
            let inverseIndices = putAlong(zeros, sortedIndices, values: arangeIndices, axis: -1)
            let cumProbsOrigOrder = takeAlong(cumProbs, inverseIndices, axis: -1)

            // Mask tokens where cumulative prob > (1 - top_p)
            // Keep tokens that are in the top_p nucleus
            let threshold = 1.0 - topP
            let mask = cumProbsOrigOrder .> threshold
            let negInf = MLXArray.full(
                filteredCandidateLogits.shape,
                values: MLXArray(-Float.infinity),
                dtype: filteredCandidateLogits.dtype
            )
            filteredCandidateLogits = which(mask, filteredCandidateLogits, negInf)
        }

        // Apply min-p sampling behavior (default kept at 0.0 for now)
        if minP > 0.0 {
            let scaledMinP = Float(log(Double(minP)))
            // Indices sorted in descending order (like Python `argsort(-logits)`)
            let sortedIndices = argSort(-filteredCandidateLogits, axis: -1).asType(.int32)
            let sortedLogits = takeAlong(filteredCandidateLogits, sortedIndices, axis: -1)
            let topLogits = sortedLogits[0..., 0 ..< 1]
            let scaledMinPArray = MLXArray.full(
                topLogits.shape,
                values: MLXArray(scaledMinP),
                dtype: sortedLogits.dtype
            ) + topLogits
            let removeMask = sortedLogits .< scaledMinPArray
            let negInf = MLXArray.full(sortedLogits.shape, values: MLXArray(-Float.infinity), dtype: sortedLogits.dtype)
            let filteredSortedLogits = which(removeMask, negInf, sortedLogits)

            let sortedCount = sortedIndices.dim(-1)
            let invArange = MLXArray(0 ..< sortedCount).reshaped(1, -1).asType(Int32.self)
            let inverseIndices = putAlong(MLXArray.zeros(sortedIndices.shape, type: Int32.self), sortedIndices, values: invArange, axis: -1)
            filteredCandidateLogits = takeAlong(filteredSortedLogits, inverseIndices, axis: -1)
        }

        var filteredLogits: MLXArray
        if let candidateIndices {
            let negInf = MLXArray.full(logitsSlice.shape, values: MLXArray(-Float.infinity), dtype: logitsSlice.dtype)
            filteredLogits = putAlong(negInf, candidateIndices, values: filteredCandidateLogits, axis: -1)
        } else {
            filteredLogits = filteredCandidateLogits
        }

        if let eosLogit, let eosTokenId {
            let eosIdx = MLXArray([Int32(eosTokenId)]).reshaped(1, 1)
            filteredLogits = putAlong(filteredLogits, eosIdx, values: eosLogit, axis: -1)
        }

        // Sample with temperature
        let token = categorical(filteredLogits / temperature)
        return token.reshaped(1, 1)
    }

    static func reducedCandidateSampleToken(
        _ logitsSlice: MLXArray,
        temperature: Float,
        topP: Float,
        topK: Int,
        minP: Float,
        eosTokenId: Int? = nil
    ) -> MLXArray? {
        let vocabSize = logitsSlice.dim(-1)
        guard topK > 0, topK < vocabSize else {
            return nil
        }

        let candidateCount = min(topK, vocabSize)
        let kth = candidateCount - 1
        guard kth >= 0 else {
            return nil
        }

        var candidateIndices = argPartition(-logitsSlice, kth: kth, axis: -1)[0..., ..<candidateCount].asType(.int32)
        var candidateLogits = takeAlong(logitsSlice, candidateIndices, axis: -1)

        if let eosTokenId,
           eosTokenId >= 0,
           eosTokenId < vocabSize {
            let eosIndex = MLXArray([Int32(eosTokenId)]).reshaped(1, 1)
            let eosLogit = takeAlong(logitsSlice, eosIndex, axis: -1)
            // Keep EOS eligible without forcing a host-side membership check that would
            // synchronize the hottest decode path. If EOS is already in the top-k slice
            // this duplicates one candidate near the end of generation, which is a much
            // smaller cost than reconstructing full-vocabulary filtered logits every frame.
            candidateIndices = concatenated([candidateIndices, eosIndex], axis: -1)
            candidateLogits = concatenated([candidateLogits, eosLogit], axis: -1)
        }

        let needsSortedCandidates = (topP > 0 && topP < 1.0) || minP > 0.0
        if needsSortedCandidates {
            let sortedOrder = argSort(-candidateLogits, axis: -1)
            candidateIndices = takeAlong(candidateIndices, sortedOrder, axis: -1)
            candidateLogits = takeAlong(candidateLogits, sortedOrder, axis: -1)
        }

        if topP > 0, topP < 1.0 {
            let sortedProbs = softmax(candidateLogits, axis: -1)
            let cumulativeProbs = cumsum(sortedProbs, axis: -1)
            let batchSize = candidateLogits.dim(0)
            let effectiveCandidateCount = candidateLogits.dim(-1)
            let prefixZeros = MLXArray.zeros([batchSize, 1]).asType(cumulativeProbs.dtype)
            let shiftedCumulative: MLXArray
            if effectiveCandidateCount > 1 {
                shiftedCumulative = concatenated(
                    [prefixZeros, cumulativeProbs[0..., 0..<(effectiveCandidateCount - 1)]],
                    axis: -1
                )
            } else {
                shiftedCumulative = prefixZeros
            }

            let negInf = MLXArray.full(
                candidateLogits.shape,
                values: MLXArray(-Float.infinity),
                dtype: candidateLogits.dtype
            )
            candidateLogits = which(shiftedCumulative .< topP, candidateLogits, negInf)
        }

        if minP > 0.0 {
            let scaledMinP = Float(log(Double(minP)))
            let topLogits = candidateLogits[0..., 0 ..< 1]
            let scaledMinPArray = MLXArray.full(
                topLogits.shape,
                values: MLXArray(scaledMinP),
                dtype: candidateLogits.dtype
            ) + topLogits
            let removeMask = candidateLogits .< scaledMinPArray
            let negInf = MLXArray.full(
                candidateLogits.shape,
                values: MLXArray(-Float.infinity),
                dtype: candidateLogits.dtype
            )
            candidateLogits = which(removeMask, negInf, candidateLogits)
        }

        let sampledLocalIndex = categorical(candidateLogits / temperature).reshaped(1, 1)
        let sampledToken = takeAlong(candidateIndices, sampledLocalIndex, axis: -1)
        return sampledToken.reshaped(1, 1)
    }

    // MARK: - fromPretrained

    public static func fromPretrained(
        _ modelRepo: String,
        cache: HubCache = .default
    ) async throws -> Qwen3TTSModel {
        guard let repoID = Repo.ID(rawValue: modelRepo) else {
            throw AudioGenerationError.modelNotInitialized("Invalid model repository ID: \(modelRepo)")
        }
        let modelDir = try await ModelUtils.resolveOrDownloadModel(
            repoID: repoID,
            requiredExtension: "safetensors",
            cache: cache
        )

        return try await fromDirectory(modelDir)
    }

    public static func fromDirectory(_ modelDir: URL) async throws -> Qwen3TTSModel {
        // Load main config
        let configData = try Data(contentsOf: modelDir.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(Qwen3TTSModelConfig.self, from: configData)

        let model = Qwen3TTSModel(config: config)

        // Load talker weights
        var allWeights = [String: MLXArray]()
        let fm = FileManager.default
        for searchDirectory in qwenWeightSearchDirectories(modelDir) where fm.fileExists(atPath: searchDirectory.path) {
            let modelFiles = try fm.contentsOfDirectory(at: searchDirectory, includingPropertiesForKeys: nil)
            for file in modelFiles where file.pathExtension == "safetensors" {
                let weights = try MLX.loadArrays(url: file)
                allWeights.merge(weights) { _, new in new }
            }
        }

        // Sanitize and load talker weights
        let talkerWeights = Qwen3TTSTalkerForConditionalGeneration.sanitize(weights: allWeights)
        let talkerPairs = talkerWeights.map { ($0.key, $0.value) }

        // Quantized checkpoints store packed weights and companion .scales tensors.
        // Convert talker Linear layers before loading those tensors.
        if config.quantization != nil || config.perLayerQuantization != nil {
            quantize(model: model.talker) { path, _ in
                guard talkerWeights["\(path).scales"] != nil else {
                    return nil
                }

                if let perLayerQuant = config.perLayerQuantization,
                   let layerQuant = perLayerQuant.quantization(layer: path) {
                    return layerQuant.asTuple
                }

                return config.quantization?.asTuple
            }
        }

        try model.talker.update(parameters: ModuleParameters.unflattened(talkerPairs), verify: .all)
        eval(model.talker.parameters())

        // Generate tokenizer.json if missing (Qwen3-TTS ships without it)
        let tokenizerJsonPath = try ensureTokenizerJSONPresent(in: modelDir)
        if !fm.fileExists(atPath: tokenizerJsonPath.path) {
            let vocabPath = modelDir.appendingPathComponent("vocab.json")
            let mergesPath = modelDir.appendingPathComponent("merges.txt")
            let hasVocab = fm.fileExists(atPath: vocabPath.path)
            let hasMerges = fm.fileExists(atPath: mergesPath.path)
            if hasVocab, hasMerges {
                do {
                    try generateTokenizerJson(
                        vocabPath: vocabPath,
                        mergesPath: mergesPath,
                        tokenizerConfigPath: modelDir.appendingPathComponent("tokenizer_config.json"),
                        outputPath: tokenizerJsonPath
                    )
                    print("Generated tokenizer.json from vocab.json + merges.txt")
                } catch {
                    print("Warning: Failed to generate tokenizer.json: \(error)")
                }
            } else {
                print("Warning: Cannot generate tokenizer.json — vocab.json: \(hasVocab), merges.txt: \(hasMerges)")
            }
        }

        // Load tokenizer
        let tokenizerFilePresent = fm.fileExists(atPath: tokenizerJsonPath.path)
        do {
            model.tokenizer = try await AutoTokenizer.from(modelFolder: modelDir)
        } catch {
            if tokenizerFilePresent {
                print("Error: tokenizer.json exists but AutoTokenizer failed to parse it: \(error)")
            } else {
                print("Warning: tokenizer.json not found and AutoTokenizer could not load: \(error)")
            }
        }

        // Load speech tokenizer — check that it's a directory, not a stale file
        let speechTokenizerPath = modelDir.appendingPathComponent("speech_tokenizer")
        var isDir: ObjCBool = false
        if fm.fileExists(atPath: speechTokenizerPath.path, isDirectory: &isDir), isDir.boolValue {
            try loadSpeechTokenizer(model: model, path: speechTokenizerPath)
        } else if fm.fileExists(atPath: speechTokenizerPath.path) {
            let directoryKind = ModelUtils.modelDirectoryKind(modelDir)
            if ModelUtils.shouldAutoDeleteCorruptedModelDirectory(modelDir) {
                print("speech_tokenizer is not a directory (stale cache), clearing cached model directory...")
                try? fm.removeItem(at: modelDir)
            }
            let recoveryHint: String = switch directoryKind {
            case .valarManagedPack:
                "Managed Valar ModelPacks are preserved; reinstall the model pack to repair it."
            case .huggingFaceCache, .legacyMLXAudioCache:
                "The corrupted cache was cleared; reload the model to re-download the missing tokenizer directory."
            case .other:
                "Repair or replace the model directory and try again."
            }
            throw AudioGenerationError.modelNotInitialized(
                "Model directory is corrupted: speech_tokenizer exists but is not a directory at \(speechTokenizerPath.path). \(recoveryHint)"
            )
        } else {
            print("Warning: speech_tokenizer directory not found, speech decoding unavailable")
        }

        // Load speaker encoder for base models when available
        if config.ttsModelType == "base" {
            let speakerWeights = Qwen3TTSSpeakerEncoder.sanitize(weights: allWeights)
            if !speakerWeights.isEmpty {
                if let speakerEncoder = model.speakerEncoder {
                    let speakerPairs = speakerWeights.map { ($0.key, $0.value) }
                    try speakerEncoder.update(parameters: ModuleParameters.unflattened(speakerPairs), verify: .all)
                    eval(speakerEncoder.parameters())
                }
            }
            if model.speakerEncoder != nil {
                print("Loaded speaker encoder")
            } else {
                print("Warning: speaker encoder config missing, skipping speaker encoder load")
            }
        }

        print("Loaded Qwen3-TTS model (\(config.ttsModelType))")
        return model
    }

    private static func qwenWeightSearchDirectories(_ modelDir: URL) -> [URL] {
        [
            modelDir,
            modelDir.appendingPathComponent("weights", isDirectory: true),
        ]
    }

    private static func ensureTokenizerJSONPresent(in modelDir: URL) throws -> URL {
        let rootTokenizer = modelDir.appendingPathComponent("tokenizer.json", isDirectory: false)
        if FileManager.default.fileExists(atPath: rootTokenizer.path) {
            return rootTokenizer
        }

        let legacyTokenizer = modelDir
            .appendingPathComponent("tokenizers", isDirectory: true)
            .appendingPathComponent("tokenizer.json", isDirectory: false)
        guard FileManager.default.fileExists(atPath: legacyTokenizer.path) else {
            return rootTokenizer
        }

        do {
            try FileManager.default.linkItem(at: legacyTokenizer, to: rootTokenizer)
        } catch {
            if !FileManager.default.fileExists(atPath: rootTokenizer.path) {
                try FileManager.default.copyItem(at: legacyTokenizer, to: rootTokenizer)
            }
        }
        return rootTokenizer
    }

    private static func loadSpeechTokenizer(model: Qwen3TTSModel, path: URL) throws {
        // Load config — fall back to defaults if config.json is missing
        let tokenizerConfig: Qwen3TTSTokenizerConfig
        let configPath = path.appendingPathComponent("config.json")
        if let configData = try? Data(contentsOf: configPath) {
            tokenizerConfig = try JSONDecoder().decode(Qwen3TTSTokenizerConfig.self, from: configData)
        } else {
            print("Warning: speech_tokenizer/config.json not found, using defaults")
            let defaultJson = "{}".data(using: .utf8)!
            tokenizerConfig = try JSONDecoder().decode(Qwen3TTSTokenizerConfig.self, from: defaultJson)
        }

        let speechTokenizer = Qwen3TTSSpeechTokenizer(config: tokenizerConfig)

        // Load weights
        var tokenizerWeights = [String: MLXArray]()
        let files = try FileManager.default.contentsOfDirectory(at: path, includingPropertiesForKeys: nil)
        for file in files where file.pathExtension == "safetensors" {
            let weights = try MLX.loadArrays(url: file)
            tokenizerWeights.merge(weights) { _, new in new }
        }

        if !tokenizerWeights.isEmpty {
            let sanitized = Qwen3TTSSpeechTokenizer.sanitize(weights: tokenizerWeights)
            let pairs = sanitized.map { ($0.key, $0.value) }
            try speechTokenizer.update(parameters: ModuleParameters.unflattened(pairs), verify: .all)
            eval(speechTokenizer.parameters())
        }

        model.speechTokenizer = speechTokenizer
        print("Loaded speech tokenizer decoder")
    }

    // MARK: - Generate tokenizer.json from vocab.json + merges.txt

    /// Qwen3-TTS repos ship with a slow tokenizer (vocab.json + merges.txt) but
    /// swift-transformers requires tokenizer.json (fast tokenizer format). This
    /// generates the fast tokenizer JSON from the available files.
    private static func generateTokenizerJson(
        vocabPath: URL,
        mergesPath: URL,
        tokenizerConfigPath: URL,
        outputPath: URL
    ) throws {
        // Read vocab
        let vocabData = try Data(contentsOf: vocabPath)
        let vocabDict = try JSONSerialization.jsonObject(with: vocabData) as? [String: Int] ?? [:]

        // Read merges (skip header line "#version: ...")
        let mergesText = try String(contentsOf: mergesPath, encoding: .utf8)
        let mergeLines = mergesText.components(separatedBy: .newlines)
            .filter { !$0.isEmpty && !$0.hasPrefix("#") }

        // Read added_tokens from tokenizer_config.json
        var addedTokens = [[String: Any]]()
        if let configData = try? Data(contentsOf: tokenizerConfigPath),
           let configDict = try? JSONSerialization.jsonObject(with: configData) as? [String: Any],
           let addedTokensDecoder = configDict["added_tokens_decoder"] as? [String: [String: Any]] {
            for (idStr, tokenInfo) in addedTokensDecoder {
                guard let tokenId = Int(idStr),
                      let content = tokenInfo["content"] as? String else { continue }
                let entry: [String: Any] = [
                    "id": tokenId,
                    "content": content,
                    "single_word": tokenInfo["single_word"] as? Bool ?? false,
                    "lstrip": tokenInfo["lstrip"] as? Bool ?? false,
                    "rstrip": tokenInfo["rstrip"] as? Bool ?? false,
                    "normalized": tokenInfo["normalized"] as? Bool ?? false,
                    "special": tokenInfo["special"] as? Bool ?? true
                ]
                addedTokens.append(entry)
            }
            addedTokens.sort { ($0["id"] as? Int ?? 0) < ($1["id"] as? Int ?? 0) }
        }

        // Build tokenizer.json
        // Qwen2 uses ByteLevel BPE with a GPT-2-style regex pre-tokenizer
        let tokenizerJson: [String: Any] = [
            "version": "1.0",
            "truncation": NSNull(),
            "padding": NSNull(),
            "added_tokens": addedTokens,
            "normalizer": NSNull(),
            "pre_tokenizer": [
                "type": "Sequence",
                "pretokenizers": [
                    [
                        "type": "Split",
                        "pattern": [
                            "Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
                        ],
                        "behavior": "Isolated",
                        "invert": false
                    ] as [String: Any],
                    [
                        "type": "ByteLevel",
                        "add_prefix_space": false,
                        "trim_offsets": true,
                        "use_regex": false
                    ] as [String: Any]
                ] as [[String: Any]]
            ] as [String: Any],
            "post_processor": NSNull(),
            "decoder": [
                "type": "ByteLevel",
                "add_prefix_space": true,
                "trim_offsets": true,
                "use_regex": true
            ] as [String: Any],
            "model": [
                "type": "BPE",
                "dropout": NSNull(),
                "unk_token": NSNull(),
                "continuing_subword_prefix": "",
                "end_of_word_suffix": "",
                "fuse_unk": false,
                "byte_fallback": false,
                "ignore_merges": false,
                "vocab": vocabDict,
                "merges": mergeLines
            ] as [String: Any]
        ]

        let jsonData = try JSONSerialization.data(withJSONObject: tokenizerJson, options: [.sortedKeys])
        try jsonData.write(to: outputPath)
    }
}
