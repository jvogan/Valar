// Copyright (c) 2025, Prince Canuma and contributors
// Swift port for ValarTTS

import Foundation
import Hub
import HuggingFace
@preconcurrency import MLX
import MLXAudioCore
@preconcurrency import MLXLMCommon
import MLXNN
import os
import Tokenizers

// MARK: - VibeVoice TTS Model

/// Top-level VibeVoice streaming TTS model.
///
/// Architecture:
/// - `languageModel`: Lower Qwen2 layers (4) for text encoding — no final norm
/// - `ttsLanguageModel`: Upper Qwen2 layers (20) for TTS — with final norm
/// - `ttsInputTypes`: 2-entry embedding (0=speech, 1=text)
/// - `acousticConnector`: Projects speech latents to LM hidden size
/// - `predictionHead`: Diffusion head for speech latent prediction
/// - `ttsEosClassifier`: Binary classifier for end-of-speech detection
/// - `acousticTokenizer`: VAE decoder converting latents to audio
/// - `noiseScheduler`: DPM-Solver++ for diffusion sampling

/// Metadata for a single VibeVoice preset voice, exposed for enumeration.
public struct VibeVoicePresetVoiceInfo: Sendable {
    /// Canonical voice name (e.g. `"en-Emma_woman"`).
    public let name: String
    /// Human-readable display name (e.g. `"Emma"`).
    public let displayName: String
    /// ISO 639-1 language code (e.g. `"en"`).
    public let languageCode: String
    /// Whether this voice is currently loaded in memory.
    public let isLoaded: Bool
    /// Estimated memory footprint in bytes, or nil if not loaded.
    public let estimatedBytes: Int?
}

public final class VibeVoiceTTSModel: Module, SpeechGenerationModel, @unchecked Sendable {

    private static let logger = Logger(subsystem: "com.valar.tts", category: "VibeVoiceModel")

    let config: VibeVoiceModelConfig

    @ModuleInfo var languageModel: VibeVoiceQwen2Model
    @ModuleInfo var ttsLanguageModel: VibeVoiceQwen2Model
    @ModuleInfo var ttsInputTypes: Embedding
    @ModuleInfo var acousticConnector: VibeVoiceSpeechConnector
    @ModuleInfo var predictionHead: VibeVoiceDiffusionHead
    @ModuleInfo var ttsEosClassifier: VibeVoiceBinaryClassifier
    @ModuleInfo var acousticTokenizer: VibeVoiceAcousticTokenizer

    let noiseScheduler: VibeVoiceDPMSolverMultistepScheduler
    let ddpmInferenceSteps: Int

    /// Standalone scalar weights loaded from model.safetensors.
    var speechScalingFactor: MLXArray = MLXArray(Float(1.0))
    var speechBiasFactor: MLXArray = MLXArray(Float(0.0))

    var tokenizer: Tokenizer?
    var modelDirectory: URL?

    /// In-memory cache of pre-loaded voice KV snapshots. Keyed by voice name.
    /// Populated by `prewarmVoice(_:)` or `prewarmDefaultVoice()` to eliminate
    /// disk I/O on first synthesis.
    var prewarmedVoices: [String: VibeVoiceKVSnapshot] = [:]
    /// Single-entry transient cache for the last on-demand voice used in synthesis.
    /// This keeps repeated same-voice requests fast without letting multilingual sweeps
    /// accumulate every encountered preset in memory.
    private var transientVoiceCache: (name: String, snapshot: VibeVoiceKVSnapshot)?

    // MARK: - SpeechGenerationModel

    public var sampleRate: Int { config.sampleRate }

    public var defaultGenerationParameters: GenerateParameters {
        GenerateParameters(
            maxTokens: VibeVoiceModelConfig.defaultMaxTokens,
            temperature: 1.0,
            topP: 1.0,
            repetitionPenalty: 1.0
        )
    }

    static func deterministicGenerationSeed(text: String, voiceName: String?) -> UInt64 {
        var hash: UInt64 = 0xcbf29ce484222325
        let prime: UInt64 = 0x00000100000001B3

        func mix(_ string: String) {
            for byte in string.utf8 {
                hash ^= UInt64(byte)
                hash &*= prime
            }
        }

        mix(text)
        hash ^= 0xff
        hash &*= prime
        mix(voiceName ?? "")
        return hash
    }

    // MARK: - Init

    init(config: VibeVoiceModelConfig) {
        self.config = config

        let decoder = config.decoderConfig
        let lmLayers = config.languageModelLayers
        let ttsLayers = config.ttsLanguageModelLayers

        // Create split configs
        let lmConfig = VibeVoiceQwen2DecoderConfig(
            hiddenSize: decoder.hiddenSize,
            intermediateSize: decoder.intermediateSize,
            numAttentionHeads: decoder.numAttentionHeads,
            numKeyValueHeads: decoder.numKeyValueHeads,
            numHiddenLayers: lmLayers,
            rmsNormEps: decoder.rmsNormEps,
            vocabSize: decoder.vocabSize,
            maxPositionEmbeddings: decoder.maxPositionEmbeddings,
            ropeTheta: decoder.ropeTheta,
            headDim: decoder.headDim
        )
        let ttsConfig = VibeVoiceQwen2DecoderConfig(
            hiddenSize: decoder.hiddenSize,
            intermediateSize: decoder.intermediateSize,
            numAttentionHeads: decoder.numAttentionHeads,
            numKeyValueHeads: decoder.numKeyValueHeads,
            numHiddenLayers: ttsLayers,
            rmsNormEps: decoder.rmsNormEps,
            vocabSize: 0,  // TTS LM doesn't need token embeddings
            maxPositionEmbeddings: decoder.maxPositionEmbeddings,
            ropeTheta: decoder.ropeTheta,
            headDim: decoder.headDim
        )

        _languageModel.wrappedValue = VibeVoiceQwen2Model(config: lmConfig, useNorm: false)
        _ttsLanguageModel.wrappedValue = VibeVoiceQwen2Model(config: ttsConfig, useNorm: true)
        _ttsInputTypes.wrappedValue = Embedding(embeddingCount: 2, dimensions: decoder.hiddenSize)
        _acousticConnector.wrappedValue = VibeVoiceSpeechConnector(
            inputDim: config.acousticVaeDim, outputDim: decoder.hiddenSize
        )
        _predictionHead.wrappedValue = VibeVoiceDiffusionHead(config: config.diffusionHeadConfig)
        _ttsEosClassifier.wrappedValue = VibeVoiceBinaryClassifier(hiddenSize: decoder.hiddenSize)
        _acousticTokenizer.wrappedValue = VibeVoiceAcousticTokenizer(config: config.acousticTokenizerConfig)

        self.ddpmInferenceSteps = config.diffusionHeadConfig.ddpmNumInferenceSteps
        self.noiseScheduler = VibeVoiceDPMSolverMultistepScheduler(
            numTrainTimesteps: config.diffusionHeadConfig.ddpmNumSteps,
            betaSchedule: config.diffusionHeadConfig.ddpmBetaSchedule,
            predictionType: config.diffusionHeadConfig.predictionType
        )

        super.init()
    }

    // MARK: - Voice Loading

    /// Load a voice cache for conditioning. Returns a prewarmed snapshot if available,
    /// otherwise loads from disk.
    func loadVoice(_ voiceName: String) throws -> VibeVoiceKVSnapshot {
        // Check prewarmed cache first (zero-cost warm start)
        if let cached = prewarmedVoices[voiceName] {
            return cached
        }
        if transientVoiceCache?.name == voiceName, let cached = transientVoiceCache?.snapshot {
            return cached
        }

        guard let modelDir = modelDirectory else {
            throw AudioGenerationError.modelNotInitialized("Model directory not set")
        }
        let voicePath = modelDir.appendingPathComponent("voices/\(voiceName).safetensors")
        guard FileManager.default.fileExists(atPath: voicePath.path) else {
            throw AudioGenerationError.invalidInput("Voice not found: \(voiceName)")
        }
        let snapshot = try loadVoiceCache(from: voicePath, config: config)
        if transientVoiceCache?.name != voiceName {
            transientVoiceCache = nil
            Memory.clearCache()
        }
        transientVoiceCache = (voiceName, snapshot)
        return snapshot
    }

    /// Pre-load a specific voice into memory so the first synthesis call avoids disk I/O.
    /// Safe to call multiple times — already-cached voices are skipped.
    public func prewarmVoice(_ voiceName: String) throws {
        guard prewarmedVoices[voiceName] == nil else { return }
        guard let modelDir = modelDirectory else {
            throw AudioGenerationError.modelNotInitialized("Model directory not set")
        }
        let voicePath = modelDir.appendingPathComponent("voices/\(voiceName).safetensors")
        guard FileManager.default.fileExists(atPath: voicePath.path) else {
            return  // Voice not available — skip silently
        }
        prewarmedVoices[voiceName] = try loadVoiceCache(from: voicePath, config: config)
    }

    /// Pre-load the first available preset voice. Called during model warm-start to
    /// ensure the first synthesis has near-zero voice load overhead.
    /// Returns the name of the prewarmed voice, or nil if none was found.
    @discardableResult
    public func prewarmDefaultVoice() throws -> String? {
        guard let modelDir = modelDirectory else { return nil }
        let voicesDir = modelDir.appendingPathComponent("voices")
        guard FileManager.default.fileExists(atPath: voicesDir.path) else { return nil }

        let contents = try FileManager.default.contentsOfDirectory(
            at: voicesDir,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        )

        // Prefer a common English default, then fall back to the first installed voice.
        let candidates = contents
            .filter { $0.pathExtension == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }

        let preferred = candidates.first { $0.lastPathComponent == "en-Emma_woman.safetensors" }
            ?? candidates.first { $0.lastPathComponent == "en-Carter_man.safetensors" }
            ?? candidates.first

        guard let voiceFile = preferred else { return nil }
        let voiceName = voiceFile.deletingPathExtension().lastPathComponent
        try prewarmVoice(voiceName)
        return voiceName
    }

    /// Evict all prewarmed voice caches to free memory.
    public func evictPrewarmedVoices() {
        prewarmedVoices.removeAll()
        transientVoiceCache = nil
        Memory.clearCache()
    }

    /// Names of all currently prewarmed voices.
    var prewarmedVoiceNames: [String] {
        Array(prewarmedVoices.keys).sorted()
    }

    /// Total estimated memory footprint of all prewarmed voices in bytes.
    public var prewarmedVoicesTotalBytes: Int {
        prewarmedVoices.values.reduce(0) { $0 + $1.estimatedBytes }
    }

    public var loadedVoiceCacheBytes: Int {
        let transientBytes = transientVoiceCache.map { cache in
            prewarmedVoices[cache.name] == nil ? cache.snapshot.estimatedBytes : 0
        } ?? 0
        return prewarmedVoicesTotalBytes + transientBytes
    }

    // MARK: - Preset Voice Metadata

    /// Static metadata for the 25 bundled preset voices in the current MLX pack.
    private static let presetVoiceMetadata: [(name: String, displayName: String, languageCode: String)] = [
        ("en-Carter_man", "Carter", "en"),
        ("en-Davis_man", "Davis", "en"),
        ("en-Emma_woman", "Emma", "en"),
        ("en-Frank_man", "Frank", "en"),
        ("en-Grace_woman", "Grace", "en"),
        ("en-Mike_man", "Mike", "en"),
        ("de-Spk0_man", "German Male", "de"),
        ("de-Spk1_woman", "German Female", "de"),
        ("fr-Spk0_man", "French Male", "fr"),
        ("fr-Spk1_woman", "French Female", "fr"),
        ("in-Samuel_man", "Samuel", "hi"),
        ("it-Spk0_woman", "Italian Female", "it"),
        ("it-Spk1_man", "Italian Male", "it"),
        ("jp-Spk0_man", "Japanese Male", "ja"),
        ("jp-Spk1_woman", "Japanese Female", "ja"),
        ("kr-Spk0_woman", "Korean Female", "ko"),
        ("kr-Spk1_man", "Korean Male", "ko"),
        ("nl-Spk0_man", "Dutch Male", "nl"),
        ("nl-Spk1_woman", "Dutch Female", "nl"),
        ("pl-Spk0_man", "Polish Male", "pl"),
        ("pl-Spk1_woman", "Polish Female", "pl"),
        ("pt-Spk0_woman", "Portuguese Female", "pt"),
        ("pt-Spk1_man", "Portuguese Male", "pt"),
        ("sp-Spk0_woman", "Spanish Female", "es"),
        ("sp-Spk1_man", "Spanish Male", "es"),
    ]

    /// Enumerate all 25 preset voices with metadata, including load status.
    public var availablePresetVoices: [VibeVoicePresetVoiceInfo] {
        Self.presetVoiceMetadata.map { meta in
            let snapshot = prewarmedVoices[meta.name]
                ?? (transientVoiceCache?.name == meta.name ? transientVoiceCache?.snapshot : nil)
            return VibeVoicePresetVoiceInfo(
                name: meta.name,
                displayName: meta.displayName,
                languageCode: meta.languageCode,
                isLoaded: snapshot != nil,
                estimatedBytes: snapshot?.estimatedBytes
            )
        }
    }

    /// Number of preset voices defined in the catalog (always 25).
    public static var presetVoiceCount: Int { presetVoiceMetadata.count }

    // MARK: - Eager Voice Loading

    /// Eagerly load all bundled preset voices from the model's `voices/` directory.
    ///
    /// Gracefully skips voices that are missing or corrupt, logging a warning for each.
    @discardableResult
    public func loadAllPresetVoices() -> (loaded: Int, skipped: [String]) {
        guard let modelDir = modelDirectory else {
            Self.logger.warning("Cannot load preset voices: model directory not set")
            return (loaded: 0, skipped: [])
        }

        let result = loadPresetVoicesGracefully(from: modelDir, config: config)

        for (name, snapshot) in result.voices {
            prewarmedVoices[name] = snapshot
        }

        if !result.skipped.isEmpty {
            Self.logger.warning("Skipped \(result.skipped.count) voices: \(result.skipped.joined(separator: ", "))")
        }

        let totalMB = Double(result.totalEstimatedBytes) / (1024 * 1024)
        Self.logger.info("Loaded \(result.voices.count)/\(Self.presetVoiceCount) preset voices (\(String(format: "%.1f", totalMB)) MB)")

        return (loaded: result.voices.count, skipped: result.skipped)
    }

    // MARK: - Diffusion Sampling

    /// Sample speech latents using diffusion with classifier-free guidance.
    ///
    /// - Parameters:
    ///   - condition: Positive conditioning, shape (B, hidden_size)
    ///   - negCondition: Negative conditioning, shape (B, hidden_size)
    ///   - cfgScale: Classifier-free guidance scale
    ///   - ddpmSteps: Override number of diffusion steps
    /// - Returns: Sampled speech latents, shape (B, acoustic_vae_dim)
    func sampleSpeechTokens(
        condition: MLXArray,
        negCondition: MLXArray,
        cfgScale: Float = 3.0,
        ddpmSteps: Int? = nil
    ) -> MLXArray {
        // Use float32 for diffusion math to reduce gritty artifacts
        let cond = condition.asType(.float32)
        let negCond = negCondition.asType(.float32)

        // Reset and configure scheduler
        noiseScheduler.reset()
        noiseScheduler.setTimesteps(ddpmSteps ?? ddpmInferenceSteps)

        // Concatenate conditions for batched prediction: [positive; negative]
        let condCombined = MLX.concatenated([cond, negCond], axis: 0)

        let batchSize = cond.dim(0)
        let latentDim = config.acousticVaeDim

        // Initialize noise
        var speech = MLXRandom.normal([batchSize, latentDim]).asType(.float32)
        var prevX0: MLXArray? = nil

        for tVal in noiseScheduler.timesteps {
            let tFloat = Float(tVal)
            let timesteps = MLXArray([tFloat, tFloat])

            // Duplicate speech for batched CFG prediction
            let combinedSpeech = MLX.concatenated([speech, speech], axis: 0)

            // Predict v/epsilon
            let eps = predictionHead(combinedSpeech, timesteps: timesteps, condition: condCombined)

            // Apply CFG
            let condEps = eps[..<batchSize]
            let uncondEps = eps[batchSize...]
            let guidedEps = uncondEps + MLXArray(cfgScale) * (condEps - uncondEps)

            // Duplicate for scheduler
            let fullEps = MLX.concatenated([guidedEps, guidedEps], axis: 0)
            let fullSpeech = MLX.concatenated([speech, speech], axis: 0)

            // Scheduler step
            let output = noiseScheduler.step(
                modelOutput: fullEps,
                timestep: tVal,
                sample: fullSpeech,
                prevX0: prevX0
            )

            // Extract positive conditioning result
            speech = output.prevSample[..<batchSize]
            if let x0 = output.x0Pred {
                prevX0 = x0[..<batchSize]
            } else {
                prevX0 = nil
            }
        }

        return speech
    }

    // MARK: - Core Generation

    /// Core generation loop for single-speaker synthesis.
    ///
    /// Implements the VibeVoice text-window / speech-window streaming architecture:
    /// 1. Consume up to 5 text tokens per window
    /// 2. Generate up to 6 speech latents per window via diffusion
    /// 3. Feed speech latents back through tts_language_model
    /// 4. Check EOS classifier after each speech latent
    ///
    /// - Parameters:
    ///   - text: Input text to synthesize
    ///   - voiceName: Preset voice name
    ///   - maxTokens: Maximum speech latent steps
    ///   - cfgScale: Classifier-free guidance scale
    ///   - ddpmSteps: Override diffusion inference steps
    /// - Returns: Generated audio waveform as MLXArray, or nil if empty
    /// Per-phase latency breakdown for a single VibeVoice generation.
    struct LatencyBreakdown: Sendable {
        let voiceLoadSeconds: TimeInterval
        let textPrefillSeconds: TimeInterval
        let diffusionSamplingSeconds: TimeInterval
        let acousticDecodeSeconds: TimeInterval
        let firstLatentSeconds: TimeInterval?
        let totalWallSeconds: TimeInterval
        let speechLatentCount: Int
        let audioDurationSeconds: Double
        let promptTokenCount: Int

        /// Real-time factor: audio duration / wall time. Values > 1.0 mean faster than real-time.
        var realtimeFactor: Double {
            guard totalWallSeconds > 0 else { return 0 }
            return audioDurationSeconds / totalWallSeconds
        }

        /// Speech latents generated per second.
        var latentsPerSecond: Double {
            guard totalWallSeconds > 0 else { return 0 }
            return Double(speechLatentCount) / totalWallSeconds
        }
    }

    /// The latency breakdown from the most recent `generateSingle` call, if any.
    private(set) var lastLatencyBreakdown: LatencyBreakdown?

    func generateSingle(
        text: String,
        voiceName: String?,
        maxTokens: Int,
        cfgScale: Float,
        ddpmSteps: Int? = nil
    ) throws -> MLXArray? {
        guard let tokenizer else {
            throw AudioGenerationError.modelNotInitialized("Tokenizer not loaded")
        }

        let generationStart = Date()

        // --- Voice cache load (instrumented) ---
        let voiceLoadStart = Date()
        let voice: VibeVoiceKVSnapshot?
        if let voiceName {
            voice = try loadVoice(voiceName)
        } else {
            voice = nil
        }
        let voiceLoadTime = Date().timeIntervalSince(voiceLoadStart)

        // Tokenize input — text.strip() + "\n", no special tokens
        let cleanText = text.trimmingCharacters(in: .whitespacesAndNewlines) + "\n"
        MLXRandom.seed(Self.deterministicGenerationSeed(text: cleanText, voiceName: voiceName))
        let textTokenIds = tokenizer.encode(text: cleanText)
        let inputIds = MLXArray(textTokenIds.map { Int32($0) }).expandedDimensions(axis: 0)  // (1, seqLen)

        let batchSize = 1
        let seqLen = inputIds.dim(1)
        let hiddenSize = config.decoderConfig.hiddenSize

        // Initialize caches from voice or empty
        var lmCache: [(key: MLXArray, value: MLXArray)?]?
        var ttsCache: [(key: MLXArray, value: MLXArray)?]?
        var negCache: [(key: MLXArray, value: MLXArray)?]?
        var ttsHidden: MLXArray?
        var negHidden: MLXArray?

        if let voice {
            // Transpose KV caches from (B, kv_heads, seq, head_dim) to (B, seq, kv_heads, head_dim)
            func transposeKV(_ pairs: [(key: MLXArray, value: MLXArray)]) -> [(key: MLXArray, value: MLXArray)?] {
                pairs.map { pair in
                    let k = pair.key.ndim == 4 ? pair.key.transposed(0, 2, 1, 3) : pair.key
                    let v = pair.value.ndim == 4 ? pair.value.transposed(0, 2, 1, 3) : pair.value
                    return (key: k, value: v) as (key: MLXArray, value: MLXArray)?
                }
            }

            lmCache = transposeKV(voice.lmCache)
            ttsCache = transposeKV(voice.ttsLmCache)
            negCache = transposeKV(voice.negTtsLmCache)
            ttsHidden = voice.ttsLmHidden
            negHidden = voice.negTtsLmHidden
        }

        var speechLatents: [MLXArray] = []
        var finished = false
        var step = 0
        var textPos = 0

        // Latency instrumentation accumulators
        var textPrefillTime: TimeInterval = 0
        var diffusionSamplingTime: TimeInterval = 0
        var firstLatentTime: TimeInterval? = nil

        while !finished && step < maxTokens {
            // Consume text tokens in windows of TTS_TEXT_WINDOW_SIZE
            if textPos < seqLen {
                let prefillStart = Date()

                let windowEnd = min(seqLen, textPos + VibeVoiceModelConfig.ttsTextWindowSize)
                let curTextIds = inputIds[0..., textPos..<windowEnd]
                let curWindow = curTextIds.dim(1)
                textPos = windowEnd

                // Embed text tokens through language_model
                let textEmbeds = languageModel.embedTokens!(curTextIds)
                let (lmOut, newLmCache) = languageModel(inputsEmbeds: textEmbeds, cache: lmCache)
                lmCache = newLmCache.map { Optional($0) }

                // Add text type embedding (type=1) and feed through tts_language_model
                let textType = MLXArray.ones([batchSize, curWindow], type: Int32.self)
                let typeEmbed = ttsInputTypes(textType)
                let ttsIn = lmOut + typeEmbed
                let (ttsOut, newTtsCache) = ttsLanguageModel(inputsEmbeds: ttsIn, cache: ttsCache)
                ttsCache = newTtsCache.map { Optional($0) }

                if ttsHidden == nil {
                    ttsHidden = ttsOut
                } else {
                    ttsHidden = MLX.concatenated([ttsHidden!, ttsOut], axis: 1)
                }

                // Negative (unconditioned) path: zero embeddings with text type
                // Skip when voice cache provides neg_hidden — running this would corrupt
                // the cached negative conditioning and cause premature EOS.
                if negHidden == nil || voice == nil {
                    let negEmbed = MLXArray.zeros([batchSize, curWindow, hiddenSize])
                    let negTypeEmbed = ttsInputTypes(MLXArray.ones([batchSize, curWindow], type: Int32.self))
                    let negIn = negEmbed + negTypeEmbed
                    let (negOut, newNegCache) = ttsLanguageModel(inputsEmbeds: negIn, cache: negCache)
                    negCache = newNegCache.map { Optional($0) }

                    if negHidden == nil {
                        negHidden = negOut
                    } else {
                        negHidden = MLX.concatenated([negHidden!, negOut], axis: 1)
                    }
                }

                textPrefillTime += Date().timeIntervalSince(prefillStart)
            }

            guard ttsHidden != nil, negHidden != nil else {
                break
            }

            // Generate speech latents (up to TTS_SPEECH_WINDOW_SIZE per text window)
            for _ in 0..<VibeVoiceModelConfig.ttsSpeechWindowSize {
                // Take last hidden state as conditioning
                let positiveCondition = ttsHidden![0..., (-1)..., 0...]
                    .squeezed(axis: 1)
                let negativeCondition = negHidden![0..., (-1)..., 0...]
                    .squeezed(axis: 1)

                // Sample speech latent via diffusion (instrumented)
                let diffusionStart = Date()
                let speechLatent = sampleSpeechTokens(
                    condition: positiveCondition,
                    negCondition: negativeCondition,
                    cfgScale: cfgScale,
                    ddpmSteps: ddpmSteps
                )
                diffusionSamplingTime += Date().timeIntervalSince(diffusionStart)

                let speechLatentExpanded = speechLatent.expandedDimensions(axis: 1)  // (1, 1, vaeDim)
                speechLatents.append(speechLatentExpanded)

                // Track first latent time (approximates first-chunk latency)
                if firstLatentTime == nil {
                    firstLatentTime = Date().timeIntervalSince(generationStart)
                }

                // Project speech latent back into TTS LM space
                let acousticEmbed = acousticConnector(speechLatentExpanded)

                // Feed through tts_language_model with speech type embedding (type=0)
                let speechTypeEmbed = ttsInputTypes(MLXArray.zeros([batchSize, 1], type: Int32.self))
                let ttsInput = acousticEmbed + speechTypeEmbed
                let (ttsOut, newTtsCache) = ttsLanguageModel(inputsEmbeds: ttsInput, cache: ttsCache)
                ttsCache = newTtsCache.map { Optional($0) }
                ttsHidden = MLX.concatenated([ttsHidden!, ttsOut], axis: 1)

                // Negative path
                let negInput = acousticEmbed + speechTypeEmbed
                let (negOut, newNegCache) = ttsLanguageModel(inputsEmbeds: negInput, cache: negCache)
                negCache = newNegCache.map { Optional($0) }
                negHidden = MLX.concatenated([negHidden!, negOut], axis: 1)

                // Keep the port aligned with the upstream realtime loop so short utterances
                // do not spill into long babbling tails.
                let eosInput = ttsOut[0..., (-1)..., 0...].squeezed(axis: 1)
                let eosLogits = sigmoid(ttsEosClassifier(eosInput))
                MLX.eval(eosLogits)
                let eosVal = eosLogits[0].item(Float.self)
                if eosVal > VibeVoiceModelConfig.defaultEOSThreshold {
                    finished = true
                    break
                }

                step += 1
                if step >= maxTokens {
                    finished = true
                    break
                }

                // Emit heartbeat with real wall time
                if let handler = AudioGenerationObserverContext.heartbeatHandler {
                    handler(AudioGenerationHeartbeat(
                        generatedTokenCount: step,
                        maxTokens: maxTokens,
                        wallTimeSeconds: Date().timeIntervalSince(generationStart)
                    ))
                }
            }
        }

        guard !speechLatents.isEmpty else { return nil }

        // --- Acoustic decode (instrumented) ---
        let decodeStart = Date()
        let speechLatentSeq = MLX.concatenated(speechLatents, axis: 1)  // (1, N, vaeDim)
        let scaledLatents = speechLatentSeq / speechScalingFactor - speechBiasFactor
        let audio = acousticTokenizer.decode(scaledLatents)  // (1, 1, T)
        let decodeTime = Date().timeIntervalSince(decodeStart)

        let totalWallTime = Date().timeIntervalSince(generationStart)

        // Compute audio duration for RTF
        let audioSamples = audio.dim(2)
        let audioDurationSeconds = Double(audioSamples) / Double(config.sampleRate)

        // Store latency breakdown for external inspection
        let breakdown = LatencyBreakdown(
            voiceLoadSeconds: voiceLoadTime,
            textPrefillSeconds: textPrefillTime,
            diffusionSamplingSeconds: diffusionSamplingTime,
            acousticDecodeSeconds: decodeTime,
            firstLatentSeconds: firstLatentTime,
            totalWallSeconds: totalWallTime,
            speechLatentCount: step,
            audioDurationSeconds: audioDurationSeconds,
            promptTokenCount: seqLen
        )
        lastLatencyBreakdown = breakdown

        // Emit generation info through observer
        if let infoHandler = AudioGenerationObserverContext.infoHandler {
            infoHandler(AudioGenerationInfo(
                promptTokenCount: seqLen,
                generationTokenCount: step,
                prefillTime: voiceLoadTime + textPrefillTime,
                generateTime: diffusionSamplingTime + decodeTime,
                samplingTime: diffusionSamplingTime,
                evalTime: textPrefillTime,
                tokenMaterializationTime: decodeTime,
                embeddingAssemblyTime: voiceLoadTime,
                tokensPerSecond: breakdown.latentsPerSecond,
                peakMemoryUsage: 0
            ))
        }

        // Final heartbeat with total wall time
        if let handler = AudioGenerationObserverContext.heartbeatHandler {
            handler(AudioGenerationHeartbeat(
                generatedTokenCount: step,
                maxTokens: maxTokens,
                wallTimeSeconds: totalWallTime
            ))
        }

        return audio.squeezed(axis: 1).squeezed(axis: 0)  // (T,)
    }

    // MARK: - SpeechGenerationModel Protocol

    public func generate(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) async throws -> MLXArray {
        let maxTokens = generationParameters.maxTokens ?? VibeVoiceModelConfig.defaultMaxTokens
        let cfgScale = VibeVoiceModelConfig.defaultCFGScale

        guard let audio = try generateSingle(
            text: text,
            voiceName: voice,
            maxTokens: maxTokens,
            cfgScale: cfgScale
        ) else {
            return MLXArray([Float]())
        }
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
        generateStream(
            text: text,
            voice: voice,
            refAudio: refAudio,
            refText: refText,
            language: language,
            generationParameters: generationParameters,
            streamingInterval: Double(VibeVoiceModelConfig.streamingSpeechWindowSize)
                / VibeVoiceModelConfig.streamingSpeechFrameRateHz
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
        AsyncThrowingStream { continuation in
            let task = Task { @Sendable [self] in
                do {
                    try generateStreaming(
                        text: text,
                        voiceName: voice,
                        maxTokens: generationParameters.maxTokens ?? VibeVoiceModelConfig.defaultMaxTokens,
                        cfgScale: VibeVoiceModelConfig.defaultCFGScale,
                        streamingChunkSize: Self.resolvedStreamingChunkSize(for: streamingInterval),
                        continuation: continuation
                    )
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }

    // MARK: - Streaming Generation

    /// Core streaming generation loop that yields decoded audio after each speech window.
    ///
    /// Instead of accumulating all speech latents and decoding at the end (as `generateSingle`
    /// does), this method decodes smaller streaming chunks and yields audio as soon as each
    /// chunk completes.
    static func resolvedStreamingChunkSize(for streamingInterval: Double) -> Int {
        let fallback = VibeVoiceModelConfig.streamingSpeechWindowSize
        guard streamingInterval.isFinite, streamingInterval > 0 else {
            return fallback
        }

        let requestedChunkSize = Int(
            (streamingInterval * VibeVoiceModelConfig.streamingSpeechFrameRateHz)
                .rounded(.toNearestOrAwayFromZero)
        )
        return min(VibeVoiceModelConfig.ttsSpeechWindowSize, max(1, requestedChunkSize))
    }

    func generateStreaming(
        text: String,
        voiceName: String?,
        maxTokens: Int,
        cfgScale: Float,
        ddpmSteps: Int? = nil,
        streamingChunkSize: Int,
        continuation: AsyncThrowingStream<AudioGeneration, Error>.Continuation
    ) throws {
        guard let tokenizer else {
            throw AudioGenerationError.modelNotInitialized("Tokenizer not loaded")
        }

        let resolvedStreamingChunkSize = min(
            VibeVoiceModelConfig.ttsSpeechWindowSize,
            max(1, streamingChunkSize)
        )
        let steadyStateStreamingChunkSize = VibeVoiceModelConfig.ttsSpeechWindowSize

        let voice: VibeVoiceKVSnapshot?
        if let voiceName {
            voice = try loadVoice(voiceName)
        } else {
            voice = nil
        }

        let cleanText = text.trimmingCharacters(in: .whitespacesAndNewlines) + "\n"
        MLXRandom.seed(Self.deterministicGenerationSeed(text: cleanText, voiceName: voiceName))
        let textTokenIds = tokenizer.encode(text: cleanText)
        let inputIds = MLXArray(textTokenIds.map { Int32($0) }).expandedDimensions(axis: 0)

        let batchSize = 1
        let seqLen = inputIds.dim(1)
        let hiddenSize = config.decoderConfig.hiddenSize

        var lmCache: [(key: MLXArray, value: MLXArray)?]?
        var ttsCache: [(key: MLXArray, value: MLXArray)?]?
        var negCache: [(key: MLXArray, value: MLXArray)?]?
        var ttsHidden: MLXArray?
        var negHidden: MLXArray?

        if let voice {
            func transposeKV(_ pairs: [(key: MLXArray, value: MLXArray)]) -> [(key: MLXArray, value: MLXArray)?] {
                pairs.map { pair in
                    let k = pair.key.ndim == 4 ? pair.key.transposed(0, 2, 1, 3) : pair.key
                    let v = pair.value.ndim == 4 ? pair.value.transposed(0, 2, 1, 3) : pair.value
                    return (key: k, value: v) as (key: MLXArray, value: MLXArray)?
                }
            }

            lmCache = transposeKV(voice.lmCache)
            ttsCache = transposeKV(voice.ttsLmCache)
            negCache = transposeKV(voice.negTtsLmCache)
            ttsHidden = voice.ttsLmHidden
            negHidden = voice.negTtsLmHidden
        }

        var finished = false
        var step = 0
        var textPos = 0
        let generationStart = Date()
        var decodedSpeechLatents: [MLXArray] = []
        decodedSpeechLatents.reserveCapacity(maxTokens)
        var emittedSampleCount = 0
        var currentFlushTarget = resolvedStreamingChunkSize

        while !finished && step < maxTokens {
            try Task.checkCancellation()
            if textPos < seqLen {
                let windowEnd = min(seqLen, textPos + VibeVoiceModelConfig.ttsTextWindowSize)
                let curTextIds = inputIds[0..., textPos..<windowEnd]
                let curWindow = curTextIds.dim(1)
                textPos = windowEnd

                let textEmbeds = languageModel.embedTokens!(curTextIds)
                let (lmOut, newLmCache) = languageModel(inputsEmbeds: textEmbeds, cache: lmCache)
                lmCache = newLmCache.map { Optional($0) }

                let textType = MLXArray.ones([batchSize, curWindow], type: Int32.self)
                let typeEmbed = ttsInputTypes(textType)
                let ttsIn = lmOut + typeEmbed
                let (ttsOut, newTtsCache) = ttsLanguageModel(inputsEmbeds: ttsIn, cache: ttsCache)
                ttsCache = newTtsCache.map { Optional($0) }

                if ttsHidden == nil {
                    ttsHidden = ttsOut
                } else {
                    ttsHidden = MLX.concatenated([ttsHidden!, ttsOut], axis: 1)
                }

                // Negative path: skip when voice cache provides neg_hidden
                if negHidden == nil || voice == nil {
                    let negEmbed = MLXArray.zeros([batchSize, curWindow, hiddenSize])
                    let negTypeEmbed = ttsInputTypes(MLXArray.ones([batchSize, curWindow], type: Int32.self))
                    let negIn = negEmbed + negTypeEmbed
                    let (negOut, newNegCache) = ttsLanguageModel(inputsEmbeds: negIn, cache: negCache)
                    negCache = newNegCache.map { Optional($0) }

                    if negHidden == nil {
                        negHidden = negOut
                    } else {
                        negHidden = MLX.concatenated([negHidden!, negOut], axis: 1)
                    }
                }
            }

            guard ttsHidden != nil, negHidden != nil else {
                break
            }

            var windowLatents: [MLXArray] = []
            windowLatents.reserveCapacity(resolvedStreamingChunkSize)

            func flushWindowLatents() {
                guard !windowLatents.isEmpty else { return }

                decodedSpeechLatents.append(contentsOf: windowLatents)

                let latentSeq = MLX.concatenated(decodedSpeechLatents, axis: 1)
                let scaledLatents = latentSeq / speechScalingFactor - speechBiasFactor
                let audio = acousticTokenizer.decode(scaledLatents)
                let samples = audio.squeezed(axis: 1).squeezed(axis: 0)
                let totalSampleCount = samples.dim(0)
                guard totalSampleCount > emittedSampleCount else {
                    windowLatents.removeAll(keepingCapacity: true)
                    return
                }

                let delta = samples[emittedSampleCount..<totalSampleCount]
                MLX.eval(delta)
                continuation.yield(.audio(delta))
                emittedSampleCount = totalSampleCount
                windowLatents.removeAll(keepingCapacity: true)
            }

            for _ in 0..<VibeVoiceModelConfig.ttsSpeechWindowSize {
                let positiveCondition = ttsHidden![0..., (-1)..., 0...]
                    .squeezed(axis: 1)
                let negativeCondition = negHidden![0..., (-1)..., 0...]
                    .squeezed(axis: 1)

                let speechLatent = sampleSpeechTokens(
                    condition: positiveCondition,
                    negCondition: negativeCondition,
                    cfgScale: cfgScale,
                    ddpmSteps: ddpmSteps
                )
                let speechLatentExpanded = speechLatent.expandedDimensions(axis: 1)
                windowLatents.append(speechLatentExpanded)

                let acousticEmbed = acousticConnector(speechLatentExpanded)
                let speechTypeEmbed = ttsInputTypes(MLXArray.zeros([batchSize, 1], type: Int32.self))
                let ttsInput = acousticEmbed + speechTypeEmbed
                let (ttsOut, newTtsCache) = ttsLanguageModel(inputsEmbeds: ttsInput, cache: ttsCache)
                ttsCache = newTtsCache.map { Optional($0) }
                ttsHidden = MLX.concatenated([ttsHidden!, ttsOut], axis: 1)

                let negInput = acousticEmbed + speechTypeEmbed
                let (negOut, newNegCache) = ttsLanguageModel(inputsEmbeds: negInput, cache: negCache)
                negCache = newNegCache.map { Optional($0) }
                negHidden = MLX.concatenated([negHidden!, negOut], axis: 1)

                if windowLatents.count >= currentFlushTarget {
                    flushWindowLatents()
                    currentFlushTarget = steadyStateStreamingChunkSize
                }

                // Match the upstream realtime loop to avoid runaway multilingual tails.
                let eosInput = ttsOut[0..., (-1)..., 0...].squeezed(axis: 1)
                let eosLogits = sigmoid(ttsEosClassifier(eosInput))
                MLX.eval(eosLogits)
                if eosLogits[0].item(Float.self) > VibeVoiceModelConfig.defaultEOSThreshold {
                    finished = true
                    break
                }

                step += 1
                if step >= maxTokens {
                    finished = true
                    break
                }

                if let handler = AudioGenerationObserverContext.heartbeatHandler {
                    handler(AudioGenerationHeartbeat(
                        generatedTokenCount: step,
                        maxTokens: maxTokens,
                        wallTimeSeconds: Date().timeIntervalSince(generationStart)
                    ))
                }
            }

            flushWindowLatents()
            currentFlushTarget = steadyStateStreamingChunkSize
        }
    }

    // MARK: - Weight Loading

    /// Load a VibeVoice model from a local directory.
    ///
    /// Reads `config.json`, merges safetensors shards, sanitizes weight keys,
    /// applies quantization if present, and loads the tokenizer.
    public static func fromDirectory(_ modelDir: URL) async throws -> VibeVoiceTTSModel {
        // Read config
        let configURL = modelDir.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            throw AudioGenerationError.modelNotInitialized("Missing config.json at \(configURL.path)")
        }
        let configData = try Data(contentsOf: configURL)
        let config = try JSONDecoder().decode(VibeVoiceModelConfig.self, from: configData)

        // Load all safetensors
        var weights: [String: MLXArray] = [:]
        let files = try FileManager.default.contentsOfDirectory(at: modelDir, includingPropertiesForKeys: nil)
        for file in files where file.pathExtension == "safetensors" {
            // Skip voice files
            if file.path.contains("/voices/") { continue }
            let shard = try MLX.loadArrays(url: file)
            weights.merge(shard) { _, new in new }
        }

        let model = VibeVoiceTTSModel(config: config)
        model.modelDirectory = modelDir

        // Sanitize weights
        let sanitized = model.sanitize(weights: weights)

        // Apply quantization if present
        if let quantization = config.quantization {
            quantize(
                model: model,
                groupSize: quantization.groupSize,
                bits: quantization.bits
            ) { path, _ in
                sanitized["\(path).scales"] != nil
            }
        }

        // Load weights
        try model.update(parameters: ModuleParameters.unflattened(sanitized), verify: [])

        // Load scalar weights that aren't @ModuleInfo (not picked up by update)
        if let scale = sanitized["speechScalingFactor"] {
            model.speechScalingFactor = scale
            logger.debug("Loaded speechScalingFactor: \(String(describing: scale), privacy: .public)")
        } else {
            logger.warning("speechScalingFactor not found in sanitized weights")
        }
        if let bias = sanitized["speechBiasFactor"] {
            model.speechBiasFactor = bias
            logger.debug("Loaded speechBiasFactor: \(String(describing: bias), privacy: .public)")
        } else {
            logger.warning("speechBiasFactor not found in sanitized weights")
        }

        // Load tokenizer
        model.tokenizer = try await AutoTokenizer.from(modelFolder: modelDir)

        let voicesDir = modelDir.appendingPathComponent("voices")
        if FileManager.default.fileExists(atPath: voicesDir.path) {
            let contents = try FileManager.default.contentsOfDirectory(
                at: voicesDir,
                includingPropertiesForKeys: nil,
                options: [.skipsHiddenFiles]
            )
            let bundledVoiceCount = contents.filter { $0.pathExtension == "safetensors" }.count
            logger.info("Discovered \(bundledVoiceCount, privacy: .public) bundled VibeVoice preset voices; loading remains on demand")
        } else {
            logger.info("No voices directory found — voices will load on demand")
        }

        return model
    }

    /// Load from a HuggingFace repository.
    public static func fromPretrained(
        _ modelRepo: String,
        hfToken: String? = nil
    ) async throws -> VibeVoiceTTSModel {
        let hub = HubApi(hfToken: hfToken)
        let modelDir = try await hub.snapshot(
            from: modelRepo,
            matching: ["*.json", "*.safetensors"]
        )

        return try await fromDirectory(modelDir)
    }

    // MARK: - Weight Sanitization

    /// Sanitize HuggingFace weights to match the Swift model's parameter names.
    ///
    /// Transformations:
    /// - Strip `model.` prefix
    /// - Remap `t_embedder.mlp.0` -> `t_embedder.mlp.layers.0`
    /// - Remap `adaLN_modulation.1` -> `adaLN_modulation.layers.1`
    /// - Transpose Linear weights `(out, in)` -> `(in, out)`
    /// - Transpose Conv1d `(C_out, C_in, K)` -> `(C_out, K, C_in)`
    /// - Transpose ConvTranspose1d `(C_in, C_out, K)` -> `(C_out, K, C_in)`
    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        let currentShapes = Dictionary(
            uniqueKeysWithValues: parameters().flattened().map { ($0.0, $0.1.shape) }
        )

        var newWeights: [String: MLXArray] = [:]

        for (rawKey, value) in weights {
            var key = rawKey

            // Strip "model." prefix
            if key.hasPrefix("model.") {
                key = String(key.dropFirst(6))
            }

            key = Self.normalizeWeightPath(Self.camelizeWeightPath(key))

            // Handle scaling factors as top-level model params
            if key == "speechScalingFactor" {
                newWeights["speechScalingFactor"] = value
                continue
            }
            if key == "speechBiasFactor" {
                newWeights["speechBiasFactor"] = value
                continue
            }

            // Skip rotary embedding inv_freq (computed at init)
            if key.contains("rotaryEmb.invFreq") { continue }

            // Preserve MLX quantization companion tensors after namespace remapping.
            // Quantized checkpoints store packed weight tensors at `<path>.weight` plus
            // `<path>.scales` / `<path>.biases`. The model is quantized after sanitize(),
            // so these companion tensors are not present in the unquantized parameter tree.
            if key.hasSuffix(".scales") || key.hasSuffix(".biases") {
                let suffix = key.hasSuffix(".scales") ? ".scales" : ".biases"
                let basePath = String(key.dropLast(suffix.count))
                if currentShapes["\(basePath).weight"] != nil || currentShapes["\(basePath).bias"] != nil {
                    newWeights[key] = value
                }
                continue
            }

            // Check if key exists in model
            guard let targetShape = currentShapes[key] else { continue }

            if value.shape == targetShape {
                newWeights[key] = value
            } else if value.ndim == 2, value.T.shape == targetShape {
                // Transpose Linear weights
                newWeights[key] = value.T
            } else if value.ndim == 3 {
                let isConvTr = key.contains("convtr")
                if isConvTr {
                    // ConvTranspose1d: PyTorch (C_in, C_out, K) -> MLX (C_out, K, C_in)
                    if value.dim(1) == targetShape[0]
                        && value.dim(2) == targetShape[1]
                        && value.dim(0) == targetShape[2] {
                        newWeights[key] = value.transposed(1, 2, 0)
                    } else {
                        newWeights[key] = value
                    }
                } else {
                    // Conv1d: PyTorch (C_out, C_in, K) -> MLX (C_out, K, C_in)
                    if value.dim(0) == targetShape[0]
                        && value.dim(1) == targetShape[2]
                        && value.dim(2) == targetShape[1] {
                        newWeights[key] = value.transposed(0, 2, 1)
                    } else {
                        newWeights[key] = value
                    }
                }
            } else {
                newWeights[key] = value
            }
        }

        return newWeights
    }

    private static func camelizeWeightPath(_ rawPath: String) -> String {
        rawPath
            .split(separator: ".", omittingEmptySubsequences: false)
            .map { segment in
                camelizeWeightSegment(String(segment))
            }
            .joined(separator: ".")
    }

    private static func camelizeWeightSegment(_ segment: String) -> String {
        guard segment.contains("_") else { return segment }

        let pieces = segment.split(separator: "_", omittingEmptySubsequences: false)
        guard let first = pieces.first else { return segment }

        return String(first) + pieces.dropFirst().map { part in
            guard let firstScalar = part.first else { return "" }
            return String(firstScalar).uppercased() + part.dropFirst()
        }.joined()
    }

    private static func normalizeWeightPath(_ rawPath: String) -> String {
        var key = rawPath

        // VibeVoice's Python checkpoints use numbered child paths for a small
        // MLP/AdaLN stack with parameterless activations between Linear layers.
        // Map those sparse indices onto stable named properties.
        key = key.replacingOccurrences(
            of: ".tEmbedder.mlp.layers.0.",
            with: ".tEmbedder.mlp.inputProjection."
        )
        key = key.replacingOccurrences(
            of: ".tEmbedder.mlp.layers.2.",
            with: ".tEmbedder.mlp.outputProjection."
        )
        key = key.replacingOccurrences(
            of: ".tEmbedder.mlp.0.",
            with: ".tEmbedder.mlp.inputProjection."
        )
        key = key.replacingOccurrences(
            of: ".tEmbedder.mlp.2.",
            with: ".tEmbedder.mlp.outputProjection."
        )
        key = key.replacingOccurrences(
            of: ".adaLNModulation.layers.1.",
            with: ".adaLNModulation.projection."
        )
        key = key.replacingOccurrences(
            of: ".adaLNModulation.1.",
            with: ".adaLNModulation.projection."
        )

        if key.hasPrefix("acousticTokenizer.decoder.upsampleLayers.0.0.") {
            key = key.replacingOccurrences(
                of: "acousticTokenizer.decoder.upsampleLayers.0.0.",
                with: "acousticTokenizer.decoder.stem."
            )
        } else if let upsampleMatch = key.wholeMatch(
            of: /acousticTokenizer\.decoder\.upsampleLayers\.(\d+)\.0\.(.+)/
        ), let sourceIndex = Int(upsampleMatch.1), sourceIndex > 0 {
            key = "acousticTokenizer.decoder.upsamplers.\(sourceIndex - 1).\(upsampleMatch.2)"
        }

        if let stageMatch = key.wholeMatch(
            of: /acousticTokenizer\.decoder\.stages\.(\d+)\.(\d+)\.(.+)/
        ) {
            key = "acousticTokenizer.decoder.stages.\(stageMatch.1).layers.\(stageMatch.2).\(stageMatch.3)"
        }

        return key
    }
}
