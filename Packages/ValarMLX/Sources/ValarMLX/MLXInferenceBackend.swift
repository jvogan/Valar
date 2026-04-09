import Foundation
import HuggingFace
import Metal
@preconcurrency import MLX
import MLXAudioCore
import MLXAudioSTT
import MLXAudioTTS
import os
import ValarModelKit

/// Point-in-time snapshot of a model's current residency in the MLX inference backend.
public struct ResidencyInfo: Sendable {
    public let modelId: ModelIdentifier
    public let residentSince: Date
    public let lastUsedAt: Date
    public let estimatedBytes: Int?
    public let sessionCount: Int
}

/// Configures when the MLX inference backend should proactively evict resident models.
///
/// Both criteria can be set independently. When a model satisfies either criterion and has
/// no active inference session (`activeSessionCount == 0`), `evictIfNeeded()` will remove it.
/// ASR models are always evicted before TTS models when budget-based eviction must choose
/// among several candidates.
public struct EvictionPolicy: Sendable {
    /// Maximum total estimated GPU/RAM bytes allowed across all resident models.
    /// When the sum of `estimatedBytes` for all resident models exceeds this value,
    /// `evictIfNeeded()` removes candidates (ASR first, then LRU) until under budget.
    /// `nil` disables budget-based eviction.
    public let maxResidentBytes: Int?

    /// Number of seconds a model may remain loaded without being used before it is
    /// considered idle. `evictIfNeeded()` removes models whose `lastUsedAt` timestamp
    /// is older than this threshold. `nil` disables idle-timeout eviction.
    public let idleTimeoutSeconds: TimeInterval?

    public init(maxResidentBytes: Int? = nil, idleTimeoutSeconds: TimeInterval? = nil) {
        self.maxResidentBytes = maxResidentBytes
        self.idleTimeoutSeconds = idleTimeoutSeconds
    }

    /// No-op policy: neither budget nor idle-timeout eviction is active.
    public static let `default` = EvictionPolicy()

    /// Recommended policy for local daemon use: evict idle models after 10 minutes,
    /// cap total resident memory at 20 GB. ASR models evict before TTS.
    public static let localDaemon = EvictionPolicy(
        maxResidentBytes: 20 * 1_024 * 1_024 * 1_024,
        idleTimeoutSeconds: 600
    )
}

/// MLX inference backend wrapping mlx-audio-swift.
///
/// When mlx-audio-swift is linked, this actor manages loaded model instances,
/// delegates inference to mlx-audio-swift's loading and generation APIs,
/// and bridges streaming events to ValarTTS AudioChunk streams.
///
/// API map from `Blaizzy/mlx-audio-swift` `main` as of March 18, 2026:
/// - `TTS.loadModel(modelRepo:hfToken:cache:)` returns `any SpeechGenerationModel`.
///   It resolves `config.json` `model_type` first, then dispatches to a concrete
///   `fromPretrained` constructor such as `Qwen3TTSModel.fromPretrained(...)`.
/// - Concrete `fromPretrained()` entry points return the concrete loaded model type,
///   not an erased wrapper. For the family Valar currently targets, the concrete type
///   is `Qwen3TTSModel`; other branches of the switch return `EchoTTSModel`,
///   `Qwen3Model`, `LlamaTTSModel`, `MarvisTTSModel`, `SopranoModel`,
///   `PocketTTSModel`, or `ChatterboxModel`.
/// - `generateStream(...)` returns `AsyncThrowingStream<AudioGeneration, Error>`.
///   `AudioGeneration` is a tagged event enum with `.token(Int)`,
///   `.info(AudioGenerationInfo)`, and `.audio(MLXArray)` cases.
///   `Qwen3TTSModel.generateStream(...)` yields token ids during decoding,
///   zero or more streaming audio chunks as `MLXArray`, and one `.info(...)`
///   metrics event near completion; completion is signaled by stream finish
///   rather than by a dedicated terminal event case.
/// - Upstream model caching is Hugging Face cache-backed, not process-local only.
///   `ModelUtils.resolveOrDownloadModel(...)` prefers a valid standard Hugging Face
///   snapshot and downloads into that standard cache layout. It will still read an
///   older legacy `HubCache.cacheDirectory/mlx-audio/<repo-id-with-slashes-replaced>`
///   directory when one already exists for backward compatibility, but new Valar
///   flows should not create fresh legacy cache trees.
///   It treats cache entries as valid only when required weight files are non-zero
///   and `config.json` parses, and clears stale managed cache roots when they are
///   incomplete or corrupted.
///   `Qwen3TTSModel.fromPretrained(...)` adds one more corruption guard for a stale
///   `speech_tokenizer` file by deleting the model directory and throwing so the next
///   load retries from a clean cache.
public actor MLXInferenceBackend: InferenceBackend {
    private static let logger = Logger(subsystem: "com.valar.tts", category: "MLXInference")
    private nonisolated(unsafe) static let fileManager = FileManager.default

    private typealias ModelDirectoryResolver = @Sendable (ModelDescriptor) async throws -> URL?
    private typealias WarningHandler = @Sendable (String) -> Void
    private typealias ModelLoader = @Sendable (ModelDescriptor) async throws -> MLXModelHandle
    private typealias SpeakerEmbeddingExtractor = @Sendable (URL, [Float]) throws -> [Float]
    private typealias SpeakerEncoderEvictor = @Sendable (URL) -> Void
    /// Returns the Metal device's recommended maximum working set size in bytes,
    /// or `nil` if no Metal device is available on this system.
    private typealias MetalMemoryBudgetProvider = @Sendable () -> UInt64?
    /// Returns the Metal device's current total allocated size in bytes,
    /// or `nil` if no Metal device is available on this system.
    private typealias MetalAllocatedSizeProvider = @Sendable () -> UInt64?

    private struct ResidencyRecord {
        let residentSince: Date
        var lastUsedAt: Date
        var sessionCount: Int
        let baseEstimatedBytes: Int?
        var estimatedBytes: Int?
        /// Number of currently active inference calls on this model.
        /// Incremented by `beginSession(for:)` and decremented by `endSession(for:)`.
        /// Models with `activeSessionCount > 0` are never evicted by `evictIfNeeded()`.
        var activeSessionCount: Int = 0
    }

    public nonisolated let backendKind: BackendKind = .mlx
    public nonisolated let runtimeCapabilities = BackendCapabilities(
        features: [.warmStart, .quantizedWeights, .streamingSynthesis, .streamingRecognition],
        supportedFamilies: [.qwen3TTS, .qwen3ASR, .qwen3ForcedAligner, .soprano, .orpheus, .marvis, .chatterbox, .pocketTTS, .voxtralTTS, .tadaTTS, .vibevoiceRealtimeTTS]
    )

    // Security model:
    // MLX model loading eventually hands files to upstream tensor loaders. Some common
    // checkpoint formats in the wider Python ecosystem (`.pkl`, `.pt`, `.bin`) can carry
    // pickled or otherwise unsafe content, so Valar narrows the trust boundary here before
    // any upstream loader sees the directory. We only accept `.safetensors` weight files,
    // with one explicit exception for Voxtral's normalized `voice_embedding_safe/*.bin`
    // voice assets. Everything else is rejected or surfaced as operator warnings.
    private static let rejectedWeightFileExtensions: Set<String> = ["bin", "pkl", "pt"]
    private static let asciiWhitespace = CharacterSet.whitespacesAndNewlines
    private static let tadaComponentDirectories: Set<String> = ["model", "encoder", "decoder", "aligner"]
    private static let expectedRootModelFiles: Set<String> = [
        "LICENSE",
        "config.json",
        "manifest.json",
        "merges.txt",
        "params.json",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "tekken.json",
        "vocab.json",
        "conds.safetensors",
        "special_tokens_map.json",
        "generation_config.json",
        "model.safetensors.index.json",
    ]
    private static let processEnvironment = ProcessInfo.processInfo.environment

    private var loadedModels: [ModelIdentifier: MLXModelHandle] = [:]
    // Tracks in-flight load tasks — both from explicit prewarm() and from concurrent loadModel()
    // calls. Keyed by model ID. A task in this map is loading but not yet stored in loadedModels.
    // loadModel() awaits an in-flight task instead of starting a duplicate load.
    private var pendingLoadTasks: [ModelIdentifier: Task<MLXModelHandle, Error>] = [:]
    private var validatedModelDirectories: Set<URL> = []
    // Maps model ID → resolved directory URL so the encoder cache can be evicted on unload.
    private var modelDirectories: [ModelIdentifier: URL] = [:]
    // LRU order: oldest-loaded at index 0. Updated on each successful load and on unload.
    private var modelLoadOrder: [ModelIdentifier] = []
    private let adapters: [ModelFamilyID: any ModelAdapter]
    private let loaders: [ModelFamilyID: ModelLoader]
    private let modelDirectoryResolver: ModelDirectoryResolver
    private let warningHandler: WarningHandler
    private let speakerEmbeddingExtractor: SpeakerEmbeddingExtractor
    private let speakerEncoderEvictor: SpeakerEncoderEvictor
    private let metalMemoryBudgetProvider: MetalMemoryBudgetProvider
    private let metalAllocatedSizeProvider: MetalAllocatedSizeProvider
    private let memoryPressureObserver: MLXMemoryPressureObserver
    // Per-model residency metadata, updated on every load/access/eviction.
    private var residencyRecords: [ModelIdentifier: ResidencyRecord] = [:]
    // Transient cache for estimated bytes computed during a load; consumed by recordModelResident.
    private var estimatedBytesCache: [ModelIdentifier: Int] = [:]
    // Stored nonisolated(unsafe) so deinit can cancel without actor isolation.
    nonisolated(unsafe) private var pressureMonitorTask: Task<Void, Never>?

    // Weight cache: persists resolved directory paths + safetensors mtimes across restarts.
    // Loaded from disk lazily on first validation request. A cache hit skips HuggingFace
    // Hub resolution, but still re-scans the directory for rejected extensions, path
    // traversal, and unexpected files before trusting the cached path.
    private var modelWeightCache: MLXModelWeightCache = MLXModelWeightCache()
    private var weightCacheNeedsLoad: Bool = true
    private let weightCacheFileURL: URL?
    private let evictionPolicy: EvictionPolicy

    public init(evictionPolicy: EvictionPolicy = .default) {
        self.adapters = Self.makeAdapters()
        self.loaders = Self.makeLoaders(
            qwenModelLoader: Self.defaultQwenModelLoader,
            qwenASRModelLoader: Self.defaultQwenASRModelLoader,
            qwenAlignerModelLoader: Self.defaultQwenAlignerModelLoader,
            voxtralTTSModelLoader: Self.defaultVoxtralTTSModelLoader,
            tadaTTSModelLoader: Self.defaultTadaTTSModelLoader,
            vibeVoiceTTSModelLoader: Self.defaultVibeVoiceTTSModelLoader
        )
        self.modelDirectoryResolver = Self.defaultModelDirectoryResolver
        self.warningHandler = Self.defaultWarningHandler
        self.speakerEmbeddingExtractor = Qwen3SpeakerEmbeddingExtractor.extract
        self.speakerEncoderEvictor = Qwen3SpeakerEmbeddingExtractor.evictEncoder(at:)
        self.metalMemoryBudgetProvider = Self.defaultMetalMemoryBudgetProvider
        self.metalAllocatedSizeProvider = Self.defaultMetalAllocatedSizeProvider
        self.weightCacheFileURL = Self.defaultWeightCacheFileURL
        self.memoryPressureObserver = MLXMemoryPressureObserver()
        self.evictionPolicy = evictionPolicy
        self.startPressureMonitor()
    }

    init(
        modelDirectoryResolver: @escaping @Sendable (ModelDescriptor) async throws -> URL?,
        warningHandler: @escaping @Sendable (String) -> Void,
        qwenModelLoader: @escaping @Sendable (ModelDescriptor) async throws -> MLXModelHandle,
        qwenASRModelLoader: @escaping @Sendable (ModelDescriptor) async throws -> MLXModelHandle = MLXInferenceBackend.defaultQwenASRModelLoader,
        qwenAlignerModelLoader: @escaping @Sendable (ModelDescriptor) async throws -> MLXModelHandle = MLXInferenceBackend.defaultQwenAlignerModelLoader,
        voxtralTTSModelLoader: @escaping @Sendable (ModelDescriptor) async throws -> MLXModelHandle = MLXInferenceBackend.defaultVoxtralTTSModelLoader,
        tadaTTSModelLoader: @escaping @Sendable (ModelDescriptor) async throws -> MLXModelHandle = MLXInferenceBackend.defaultTadaTTSModelLoader,
        vibeVoiceTTSModelLoader: @escaping @Sendable (ModelDescriptor) async throws -> MLXModelHandle = MLXInferenceBackend.defaultVibeVoiceTTSModelLoader,
        speakerEmbeddingExtractor: @escaping @Sendable (URL, [Float]) throws -> [Float] = Qwen3SpeakerEmbeddingExtractor.extract,
        speakerEncoderEvictor: @escaping @Sendable (URL) -> Void = Qwen3SpeakerEmbeddingExtractor.evictEncoder(at:),
        metalMemoryBudgetProvider: @escaping @Sendable () -> UInt64? = MLXInferenceBackend.defaultMetalMemoryBudgetProvider,
        metalAllocatedSizeProvider: @escaping @Sendable () -> UInt64? = { nil },
        weightCacheFileURL: URL? = nil,
        memoryPressureObserver: MLXMemoryPressureObserver = MLXMemoryPressureObserver(testingMode: true),
        evictionPolicy: EvictionPolicy = .default
    ) {
        self.adapters = Self.makeAdapters()
        self.loaders = Self.makeLoaders(
            qwenModelLoader: qwenModelLoader,
            qwenASRModelLoader: qwenASRModelLoader,
            qwenAlignerModelLoader: qwenAlignerModelLoader,
            voxtralTTSModelLoader: voxtralTTSModelLoader,
            tadaTTSModelLoader: tadaTTSModelLoader,
            vibeVoiceTTSModelLoader: vibeVoiceTTSModelLoader
        )
        self.modelDirectoryResolver = modelDirectoryResolver
        self.warningHandler = warningHandler
        self.speakerEmbeddingExtractor = speakerEmbeddingExtractor
        self.speakerEncoderEvictor = speakerEncoderEvictor
        self.metalMemoryBudgetProvider = metalMemoryBudgetProvider
        self.metalAllocatedSizeProvider = metalAllocatedSizeProvider
        self.weightCacheFileURL = weightCacheFileURL
        self.memoryPressureObserver = memoryPressureObserver
        self.evictionPolicy = evictionPolicy
        self.startPressureMonitor()
    }

    public func validate(requirement: BackendRequirement) async throws {
        guard requirement.backendKind == .mlx else {
            throw MLXBackendError.unsupportedBackend(requirement.backendKind)
        }
        Self.validateMetalBudget(
            minimumMemoryBytes: requirement.minimumMemoryBytes,
            metalMemoryBudgetProvider: metalMemoryBudgetProvider,
            warningHandler: warningHandler
        )
    }

    public func prewarm(
        descriptor: ModelDescriptor,
        configuration: ModelRuntimeConfiguration
    ) async {
        _ = configuration
        guard loadedModels[descriptor.id] == nil,
              pendingLoadTasks[descriptor.id] == nil else { return }
        pendingLoadTasks[descriptor.id] = makeLoadTask(for: descriptor)
    }

    public func loadModel(
        descriptor: ModelDescriptor,
        configuration: ModelRuntimeConfiguration
    ) async throws -> any ValarModel {
        _ = configuration
        if let cached = loadedModels[descriptor.id] {
            // Move to most-recently-used position and increment session count.
            recordModelResident(descriptor.id)
            recordModelAccess(descriptor.id)
            return cached
        }

        // If a prewarm (or a prior loadModel call) already started a load task for this
        // model, await its result rather than kicking off a duplicate load.
        if let pendingTask = pendingLoadTasks[descriptor.id] {
            do {
                let handle = try await pendingTask.value
                loadedModels[descriptor.id] = handle
                pendingLoadTasks[descriptor.id] = nil
                recordModelResident(descriptor.id)
                recordModelAccess(descriptor.id)
                return handle
            } catch {
                pendingLoadTasks[descriptor.id] = nil
                throw error
            }
        }

        // When mlx-audio-swift is linked:
        // `loadedModels` is Valar's actor-local cache keyed by stable `ModelIdentifier`.
        // It sits above mlx-audio-swift's own Hugging Face `HubCache`: the upstream cache
        // avoids re-downloading and re-parsing model snapshots on disk, while this map keeps
        // the already-materialized `SpeechGenerationModel` instance alive for reuse.
        //
        // A likely loading path for Qwen3-TTS is either:
        // `let mlxModel: any SpeechGenerationModel = try await TTS.loadModel(...)`
        // or `let mlxModel = try await Qwen3TTSModel.fromPretrained(...)`
        //
        // In both cases the loaded object is a concrete reference type conforming to
        // `SpeechGenerationModel`, and `generate(...)` returns a full `MLXArray` waveform
        // while `generateStream(...)` yields `AudioGeneration` events that Valar needs to
        // translate into `AudioChunk` values.
        // let mlxModel = try await loadMLXModel(for: descriptor)
        // let handle = MLXModelHandle(descriptor: descriptor, mlxModel: mlxModel)

        let handle = try await loadHandle(for: descriptor)
        loadedModels[descriptor.id] = handle
        recordModelResident(descriptor.id)
        recordModelAccess(descriptor.id)
        return handle
    }

    public func unloadModel(_ model: any ValarModel) async throws {
        guard let handle = model as? MLXModelHandle else { return }
        if let vibeVoiceModel = handle.mlxSpeechModel as? VibeVoiceTTSModel {
            vibeVoiceModel.evictPrewarmedVoices()
        }
        loadedModels.removeValue(forKey: handle.descriptor.id)
        // Cancel and discard any in-flight load so the model doesn't re-appear in loadedModels
        // after the caller has explicitly unloaded it.
        pendingLoadTasks.removeValue(forKey: handle.descriptor.id)?.cancel()
        modelLoadOrder.removeAll { $0 == handle.descriptor.id }
        residencyRecords.removeValue(forKey: handle.descriptor.id)
        if let directory = modelDirectories.removeValue(forKey: handle.descriptor.id) {
            speakerEncoderEvictor(directory)
        }
    }

    public func unloadModel(withID identifier: ModelIdentifier) async {
        performEviction(of: identifier)
    }

    public func extractSpeakerEmbedding(
        descriptor: ModelDescriptor,
        monoReferenceSamples: [Float]
    ) async throws -> Data {
        guard descriptor.familyID == .qwen3TTS else {
            throw MLXBackendError.inferenceError("Speaker embedding extraction only supports Qwen3 TTS models.")
        }
        guard !monoReferenceSamples.isEmpty else {
            throw MLXBackendError.inferenceError("Reference audio is empty.")
        }
        guard let modelDirectory = try await modelDirectoryResolver(descriptor) else {
            throw MLXBackendError.modelNotFound(descriptor.id)
        }

        // Record the directory so it can be evicted from the encoder cache on unload.
        modelDirectories[descriptor.id] = modelDirectory

        let cacheKey = modelDirectory.standardizedFileURL
        if !validatedModelDirectories.contains(cacheKey) {
            try await Self.validateModelDirectory(modelDirectory, familyID: descriptor.familyID, warningHandler: warningHandler)
            validatedModelDirectories.insert(cacheKey)
        }
        let embedding = try speakerEmbeddingExtractor(modelDirectory, monoReferenceSamples)
        return MLXModelHandle.pcmFloat32LEData(from: embedding)
    }

    public func extractVoiceConditioning(
        _ request: VoiceConditioningExtractionRequest
    ) async throws -> VoiceConditioning {
        if request.descriptor.familyID == .tadaTTS {
            return try await extractTADAConditioning(request)
        }
        if request.descriptor.familyID == .qwen3TTS {
            return try await extractQwenConditioning(request)
        }
        let data = try await extractSpeakerEmbedding(
            descriptor: request.descriptor,
            monoReferenceSamples: request.monoReferenceSamples
        )
        return .qwenSpeakerEmbedding(data)
    }

    private func extractQwenConditioning(
        _ request: VoiceConditioningExtractionRequest
    ) async throws -> VoiceConditioning {
        guard !request.monoReferenceSamples.isEmpty else {
            throw MLXBackendError.inferenceError("Reference audio is empty.")
        }

        let handle: MLXModelHandle
        if let cached = loadedModels[request.descriptor.id] {
            handle = cached
            recordModelAccess(request.descriptor.id)
        } else if let pendingTask = pendingLoadTasks[request.descriptor.id] {
            handle = try await pendingTask.value
            loadedModels[request.descriptor.id] = handle
            pendingLoadTasks[request.descriptor.id] = nil
            recordModelResident(request.descriptor.id)
            recordModelAccess(request.descriptor.id)
        } else {
            handle = try await loadHandle(for: request.descriptor)
            loadedModels[request.descriptor.id] = handle
            recordModelResident(request.descriptor.id)
            recordModelAccess(request.descriptor.id)
        }

        guard let qwenModel = handle.mlxSpeechModel as? Qwen3TTSModel else {
            let data = try await extractSpeakerEmbedding(
                descriptor: request.descriptor,
                monoReferenceSamples: request.monoReferenceSamples
            )
            return .qwenSpeakerEmbedding(data)
        }

        let audio = MLXArray(request.monoReferenceSamples).reshaped(1, -1)
        let clonePrompt = try qwenModel.createVoiceClonePromptConditioning(
            referenceAudio: audio,
            referenceTranscript: request.referenceTranscript
        )
        return VoiceConditioning(
            format: clonePrompt.format,
            payload: clonePrompt.payload
        )
    }

    private func extractTADAConditioning(
        _ request: VoiceConditioningExtractionRequest
    ) async throws -> VoiceConditioning {
        guard !request.monoReferenceSamples.isEmpty else {
            throw MLXBackendError.inferenceError("Reference audio is empty.")
        }

        // Load (or reuse) the TADA model handle.
        let handle: MLXModelHandle
        if let cached = loadedModels[request.descriptor.id] {
            handle = cached
            recordModelAccess(request.descriptor.id)
        } else if let pendingTask = pendingLoadTasks[request.descriptor.id] {
            handle = try await pendingTask.value
            loadedModels[request.descriptor.id] = handle
            pendingLoadTasks[request.descriptor.id] = nil
            recordModelResident(request.descriptor.id)
            recordModelAccess(request.descriptor.id)
        } else {
            handle = try await loadHandle(for: request.descriptor)
            loadedModels[request.descriptor.id] = handle
            recordModelResident(request.descriptor.id)
            recordModelAccess(request.descriptor.id)
        }

        guard let tadaModel = handle.mlxSpeechModel as? TADATTSModel else {
            throw MLXBackendError.inferenceError(
                "TADA conditioning extraction requires a loaded TADATTSModel."
            )
        }

        let audioArray = MLXArray(request.monoReferenceSamples).reshaped(1, -1)
        let condData = try tadaModel.extractReferenceConditioning(
            audio: audioArray,
            referenceTranscript: request.referenceTranscript ?? "",
            language: request.language
        )

        // Serialize tokenValues: float32 [1,T,D] → float16 binary.
        let squeezed = condData.tokenValues.squeezed(axis: 0)  // [T, D]
        let tokenCount = squeezed.dim(0)
        let acousticDim = squeezed.dim(1)
        let f16s = squeezed.asType(.float16).asArray(Float16.self)
        let valuesData: Data = f16s.withUnsafeBufferPointer { Data(buffer: $0) }

        // Serialize tokenPositions: [Int] → int32 binary.
        let positionsI32 = condData.tokenPositions.map { Int32($0) }
        let positionsData: Data = positionsI32.withUnsafeBufferPointer { Data(buffer: $0) }

        // Serialize textTokens: [Int32] → int32 binary.
        let textTokensData: Data = condData.textTokens.withUnsafeBufferPointer { Data(buffer: $0) }

        var assetFiles: [VoiceConditioningAssetFile] = [
            VoiceConditioningAssetFile(filename: "token_values.f16", data: valuesData),
            VoiceConditioningAssetFile(filename: "token_positions.i32", data: positionsData),
            VoiceConditioningAssetFile(filename: "text_tokens.i32", data: textTokensData),
        ]
        if !condData.tokenMask.isEmpty {
            assetFiles.append(VoiceConditioningAssetFile(
                filename: "token_masks.u8",
                data: Data(condData.tokenMask)
            ))
        }

        let metadata = VoiceConditioningMetadata(
            modelID: request.descriptor.id.rawValue,
            transcript: request.referenceTranscript ?? "",
            language: request.language ?? "en",
            sampleRate: request.sampleRate,
            tokenCount: tokenCount,
            acousticDimension: acousticDim,
            frameCount: condData.frameCount > 0 ? condData.frameCount : nil
        )

        return .tadaReference(
            assetFiles: assetFiles,
            sourceModel: request.descriptor.id,
            metadata: metadata
        )
    }

    public var loadedModelCount: Int { loadedModels.count }
    public var pendingLoadTaskCount: Int { pendingLoadTasks.count }

    /// The model IDs in LRU order (index 0 = least recently used).
    var modelLoadOrderSnapshot: [ModelIdentifier] { modelLoadOrder }

    deinit {
        pressureMonitorTask?.cancel()
    }
}

extension MLXInferenceBackend {
    // MARK: - Memory pressure monitoring

    /// Starts a long-lived task that reacts to memory pressure events:
    /// `.warning` → log (loads self-pause via `waitUntilClear`),
    /// `.critical` → log + evict the least-recently-used loaded model,
    /// `.normal` → log (gate already resumes suspended loads).
    ///
    /// Called from `init` — must be `nonisolated` because Swift's actor initializer
    /// does not run in the actor's isolated context.
    private nonisolated func startPressureMonitor() {
        // `memoryPressureObserver` is a `let`, so it is accessible nonisolated.
        let capturedObserver = memoryPressureObserver
        pressureMonitorTask = Task { [weak self] in
            for await rawEvent in capturedObserver.pressureEvents {
                let event = DispatchSource.MemoryPressureEvent(rawValue: rawEvent)
                if event.contains(.critical) {
                    Self.logger.warning("Critical memory pressure — evicting LRU model")
                    await self?.evictLRUModel()
                } else if event.contains(.warning) {
                    Self.logger.warning("Memory pressure warning — new model loads will pause until pressure clears")
                } else if event.contains(.normal) {
                    Self.logger.info("Memory pressure cleared — model loads may resume")
                }
            }
        }
    }

    /// Evicts the least-recently-used loaded model that has no active sessions.
    /// No-op when no models are loaded or all models have active sessions.
    private func evictLRUModel() {
        guard let lruID = modelLoadOrder.first(where: {
                  (residencyRecords[$0]?.activeSessionCount ?? 0) == 0
              }),
              loadedModels[lruID] != nil else { return }
        performEviction(of: lruID)
        Self.logger.warning("Evicted LRU model '\(lruID.rawValue, privacy: .public)' under critical memory pressure")
    }

    /// Removes a single model from all in-memory caches.
    ///
    /// This is the single eviction primitive used by both `evictLRUModel()` (pressure-driven)
    /// and `evictIfNeeded()` (policy-driven). Callers are responsible for logging.
    private func performEviction(of modelID: ModelIdentifier) {
        if let vibeVoiceModel = loadedModels[modelID]?.mlxSpeechModel as? VibeVoiceTTSModel {
            vibeVoiceModel.evictPrewarmedVoices()
        }
        loadedModels.removeValue(forKey: modelID)
        pendingLoadTasks.removeValue(forKey: modelID)?.cancel()
        modelLoadOrder.removeAll { $0 == modelID }
        residencyRecords.removeValue(forKey: modelID)
        if let directory = modelDirectories.removeValue(forKey: modelID) {
            speakerEncoderEvictor(directory)
        }
        Memory.clearCache()
    }

    // MARK: - Eviction policy

    /// Applies the configured `EvictionPolicy`, evicting resident models that exceed the
    /// memory budget or have been idle longer than the configured timeout.
    ///
    /// - Models with `activeSessionCount > 0` are never evicted — evicting a model during
    ///   active inference would leave callers with a dangling handle.
    /// - When budget-based eviction must choose among candidates, ASR (`.stt` domain) models
    ///   are evicted before TTS models to keep the TTS model warm for low-latency synthesis.
    /// - Within the same priority group, candidates are ordered LRU-first.
    ///
    /// Call this after a session ends or on a periodic timer to reclaim GPU memory.
    ///
    /// - Returns: The identifiers of all models that were evicted.
    @discardableResult
    public func evictIfNeeded() -> [ModelIdentifier] {
        var evicted: [ModelIdentifier] = []
        let candidates = evictionCandidates()

        // Idle-timeout eviction: remove models whose last use is older than the threshold.
        if let timeout = evictionPolicy.idleTimeoutSeconds {
            let cutoff = Date.now.addingTimeInterval(-timeout)
            for id in candidates where !evicted.contains(id) {
                guard let record = residencyRecords[id], record.lastUsedAt < cutoff else { continue }
                performEviction(of: id)
                evicted.append(id)
                Self.logger.info(
                    "Evicted idle model '\(id.rawValue, privacy: .public)' (idle > \(timeout, format: .fixed(precision: 1))s)"
                )
            }
        }

        // Budget-based eviction: remove candidates until total bytes fits within the budget.
        if let budget = evictionPolicy.maxResidentBytes {
            let remaining = candidates.filter { !evicted.contains($0) }
            for id in remaining {
                guard totalResidentBytes() > budget else { break }
                performEviction(of: id)
                evicted.append(id)
                Self.logger.info(
                    "Evicted model '\(id.rawValue, privacy: .public)' to satisfy \(budget)-byte budget"
                )
            }
        }

        return evicted
    }

    /// Signals that an active inference session has started on `modelID`.
    ///
    /// Models with at least one active session are skipped by `evictIfNeeded()`.
    /// Call this immediately before starting inference; balance with `endSession(for:)`.
    public func beginSession(for modelID: ModelIdentifier) {
        residencyRecords[modelID]?.activeSessionCount += 1
    }

    /// Signals that an active inference session on `modelID` has ended.
    ///
    /// Decrements the active session counter incremented by `beginSession(for:)`.
    /// The counter is clamped to 0 to guard against unbalanced calls.
    public func endSession(for modelID: ModelIdentifier) {
        guard var record = residencyRecords[modelID] else { return }
        record.activeSessionCount = max(0, record.activeSessionCount - 1)
        record.lastUsedAt = Date.now
        residencyRecords[modelID] = record
        refreshDynamicResidencyEstimate(for: modelID)
        if record.activeSessionCount == 0 {
            _ = evictIfNeeded()
        }
    }

    /// Returns candidate model IDs for eviction, ordered by eviction priority.
    ///
    /// Candidates are models with no active inference sessions. Within the candidate set,
    /// ASR models (`.stt` domain) are sorted first — they have higher eviction priority
    /// than TTS models, which are kept warm for low-latency synthesis. Within the same
    /// domain group, models are ordered LRU-first (least-recently-used appears first).
    private func evictionCandidates() -> [ModelIdentifier] {
        let inactive = modelLoadOrder.filter {
            (residencyRecords[$0]?.activeSessionCount ?? 0) == 0
        }
        return inactive.sorted { lhs, rhs in
            let lhsIsASR = loadedModels[lhs]?.descriptor.domain == .stt
            let rhsIsASR = loadedModels[rhs]?.descriptor.domain == .stt
            if lhsIsASR != rhsIsASR { return lhsIsASR } // ASR first
            // Within same domain group, preserve LRU order from modelLoadOrder.
            let lhsIdx = modelLoadOrder.firstIndex(of: lhs) ?? Int.max
            let rhsIdx = modelLoadOrder.firstIndex(of: rhs) ?? Int.max
            return lhsIdx < rhsIdx
        }
    }

    /// Returns the sum of `estimatedBytes` for all currently resident models.
    private func totalResidentBytes() -> Int {
        residencyRecords.values.compactMap(\.estimatedBytes).reduce(0, +)
    }

    /// Moves `modelID` to the most-recently-used position in `modelLoadOrder`
    /// and updates its residency `lastUsedAt` timestamp.
    private func recordModelAccess(_ modelID: ModelIdentifier) {
        modelLoadOrder.removeAll { $0 == modelID }
        modelLoadOrder.append(modelID)
        residencyRecords[modelID]?.lastUsedAt = .now
    }

    /// Records a model as resident (first load) or increments its session count (cache hit).
    /// Consumes any pending `estimatedBytesCache` entry for `modelID`.
    private func recordModelResident(_ modelID: ModelIdentifier) {
        let bytes = estimatedBytesCache.removeValue(forKey: modelID)
        if residencyRecords[modelID] == nil {
            residencyRecords[modelID] = ResidencyRecord(
                residentSince: .now,
                lastUsedAt: .now,
                sessionCount: 1,
                baseEstimatedBytes: bytes,
                estimatedBytes: bytes
            )
        } else {
            residencyRecords[modelID]?.sessionCount += 1
        }
        refreshDynamicResidencyEstimate(for: modelID)
    }

    private func refreshDynamicResidencyEstimate(for modelID: ModelIdentifier) {
        guard var record = residencyRecords[modelID] else { return }
        let extraBytes: Int
        if let handle = loadedModels[modelID],
           let vibeVoiceModel = handle.mlxSpeechModel as? VibeVoiceTTSModel {
            extraBytes = vibeVoiceModel.loadedVoiceCacheBytes
        } else {
            extraBytes = 0
        }

        if let baseEstimatedBytes = record.baseEstimatedBytes {
            record.estimatedBytes = baseEstimatedBytes + extraBytes
        } else if extraBytes > 0 {
            record.estimatedBytes = extraBytes
        } else {
            record.estimatedBytes = nil
        }
        residencyRecords[modelID] = record
    }

    /// Returns the name of the default Metal device, or `nil` if no Metal device is available.
    public nonisolated func metalDeviceName() -> String? {
        MTLCreateSystemDefaultDevice()?.name
    }

    /// Returns a point-in-time snapshot of all currently resident models.
    public func residencySnapshot() -> [ResidencyInfo] {
        residencyRecords.map { id, record in
            ResidencyInfo(
                modelId: id,
                residentSince: record.residentSince,
                lastUsedAt: record.lastUsedAt,
                estimatedBytes: record.estimatedBytes,
                sessionCount: record.sessionCount
            )
        }
    }
}

extension MLXInferenceBackend {
    private func loadHandle(for descriptor: ModelDescriptor) async throws -> MLXModelHandle {
        if let adapter = adapters[descriptor.familyID] {
            // Stub hook for future adapter-specific loading paths.
            _ = adapter
        }

        return try await loadHandle(for: descriptor, using: loaders[descriptor.familyID])
    }

    /// Creates a task that loads the model described by `descriptor` and returns the handle.
    ///
    /// The task runs on the global concurrent executor. Actor-isolated helpers are called via
    /// `await self.*` which hops onto the actor's serialized executor — this is safe because
    /// callers either do not await the task (prewarm) or suspend the actor while waiting for it
    /// (loadModel), leaving the actor's executor free to service the hops.
    private func makeLoadTask(for descriptor: ModelDescriptor) -> Task<MLXModelHandle, Error> {
        // Capture the loader closure before the task starts so it can be called from
        // the global executor without needing to re-enter the actor to read the property.
        let capturedLoader = loaders[descriptor.familyID]
        return Task { [self] in
            try await self.loadHandle(for: descriptor, using: capturedLoader)
        }
    }
}

extension MLXInferenceBackend {
    private func loadHandle(
        for descriptor: ModelDescriptor,
        using loader: ModelLoader?
    ) async throws -> MLXModelHandle {
        guard let loader else {
            throw MLXBackendError.unsupportedFamily(descriptor.familyID)
        }

        // Pause loading while memory pressure is elevated (.warning or .critical).
        // When the kernel clears pressure, waitUntilClear() returns and loading resumes.
        if await memoryPressureObserver.isUnderPressure {
            Self.logger.warning(
                "Memory pressure elevated — model load for '\(descriptor.id.rawValue, privacy: .public)' is paused"
            )
            try await memoryPressureObserver.waitUntilClear()
            Self.logger.info(
                "Memory pressure cleared — resuming model load for '\(descriptor.id.rawValue, privacy: .public)'"
            )
        }

        try await validateModelDirectoryIfPresent(for: descriptor)

        // Load native speech tokenizer decoder only when explicitly enabled via env var.
        // Lazy-loading avoids ~200-400 MB GPU memory cost when the flag is off (the default).
        var nativeDecoder: SpeechTokenizerDecoder? = nil
        if MLXStreamBridge.useNativeDecoderPath, descriptor.familyID == .qwen3TTS, let modelDir = modelDirectories[descriptor.id] {
            let speechTokenizerDir = modelDir.appendingPathComponent("speech_tokenizer")
            if FileManager.default.fileExists(atPath: speechTokenizerDir.path) {
                do {
                    let decoder = SpeechTokenizerDecoder()
                    try decoder.loadWeights(from: speechTokenizerDir)
                    nativeDecoder = decoder
                } catch {
                    warningHandler("Failed to load speech tokenizer decoder: \(error)")
                }
            }
        }

        // Measure Metal allocation delta to estimate per-model GPU memory usage.
        // Falls back to the descriptor's minimumMemoryBytes if Metal is unavailable
        // or the delta is non-positive (concurrent loads can make the delta unreliable).
        let allocBefore = metalAllocatedSizeProvider()
        let handle: MLXModelHandle
        if let modelDirectory = modelDirectories[descriptor.id],
           FileManager.default.fileExists(
               atPath: modelDirectory.appendingPathComponent("manifest.json", isDirectory: false).path
           )
        {
            handle = try await Self.loadInstalledHandle(for: descriptor, from: modelDirectory)
        } else {
            handle = try await loader(descriptor)
        }
        let allocAfter = metalAllocatedSizeProvider()

        if let before = allocBefore, let after = allocAfter, after > before {
            estimatedBytesCache[descriptor.id] = Int(after - before)
        } else if let minBytes = descriptor.supportedBackends
            .first(where: { $0.backendKind == .mlx })?.minimumMemoryBytes {
            estimatedBytesCache[descriptor.id] = minBytes
        }

        if let decoder = nativeDecoder, let mlxModel = handle.mlxSpeechModel {
            return MLXModelHandle(descriptor: descriptor, mlxModel: mlxModel, nativeDecoder: decoder)
        }
        return handle
    }

    private static func loadInstalledHandle(
        for descriptor: ModelDescriptor,
        from modelDirectory: URL
    ) async throws -> MLXModelHandle {
        switch descriptor.familyID {
        case .qwen3TTS:
            let mlxModel = try await TTS.loadModel(modelDirectory: modelDirectory, modelType: "qwen3_tts")
            return MLXModelHandle(descriptor: descriptor, mlxModel: mlxModel)
        case .qwen3ASR:
            let asrModel = try await Qwen3ASRModel.fromDirectory(modelDirectory)
            return MLXModelHandle(descriptor: descriptor, mlxSTTModel: asrModel)
        case .qwen3ForcedAligner:
            let alignerModel = try await Qwen3ForcedAlignerModel.fromDirectory(modelDirectory)
            return MLXModelHandle(descriptor: descriptor, mlxAlignerModel: alignerModel)
        case .soprano:
            let mlxModel = try await SopranoModel.fromDirectory(modelDirectory)
            return MLXModelHandle(descriptor: descriptor, mlxModel: mlxModel)
        case .voxtralTTS, .tadaTTS:
            let mlxModel = try await TTS.fromDirectory(modelDirectory)
            return MLXModelHandle(descriptor: descriptor, mlxModel: mlxModel)
        case .vibevoiceRealtimeTTS:
            let mlxModel = try await VibeVoiceTTSModel.fromDirectory(modelDirectory)
            return MLXModelHandle(descriptor: descriptor, mlxModel: mlxModel)
        default:
            throw MLXBackendError.unsupportedFamily(descriptor.familyID)
        }
    }

    private func validateModelDirectoryIfPresent(for descriptor: ModelDescriptor) async throws {
        if let installedDirectory = Self.installedModelPackDirectory(for: descriptor) {
            let cacheKey = installedDirectory.standardizedFileURL
            modelDirectories[descriptor.id] = installedDirectory
            guard !validatedModelDirectories.contains(cacheKey) else {
                return
            }
            try await Self.validateModelDirectory(
                installedDirectory,
                familyID: descriptor.familyID,
                warningHandler: warningHandler
            )
            validatedModelDirectories.insert(cacheKey)
            return
        }

        // Load the persisted weight cache from disk on first access.
        if weightCacheNeedsLoad {
            if let url = weightCacheFileURL {
                modelWeightCache = (try? MLXModelWeightCache.load(from: url)) ?? MLXModelWeightCache()
            }
            weightCacheNeedsLoad = false
        }

        // Warm restart fast path: if the weight cache has a valid entry for this model
        // (all safetensors mtimes match), re-run the cheap directory scan before
        // trusting the cached path. This keeps cache hits O(n) in file count while
        // still re-checking rejected extensions, path traversal, and unexpected files.
        if let cachedDirectory = modelWeightCache.cachedDirectory(for: descriptor.id.rawValue) {
            let cacheKey = cachedDirectory.standardizedFileURL
            do {
                try Self.validateCachedModelDirectory(
                    cachedDirectory,
                    familyID: descriptor.familyID,
                    warningHandler: warningHandler
                )
                modelDirectories[descriptor.id] = cachedDirectory
                validatedModelDirectories.insert(cacheKey)
                return
            } catch {
                validatedModelDirectories.remove(cacheKey)
                persistWeightCacheEviction(for: descriptor.id)
            }
        }

        // Cache miss: resolve the model directory via HuggingFace Hub.
        guard let modelDirectory = try await modelDirectoryResolver(descriptor) else {
            return
        }
        let cacheKey = modelDirectory.standardizedFileURL
        modelDirectories[descriptor.id] = modelDirectory
        guard !validatedModelDirectories.contains(cacheKey) else {
            return
        }

        // Full filesystem validation (header bytes, rejected extensions, etc.).
        try await Self.validateModelDirectory(
            modelDirectory,
            familyID: descriptor.familyID,
            warningHandler: warningHandler
        )
        validatedModelDirectories.insert(cacheKey)

        // Persist the resolved directory + safetensors mtimes for future warm restarts.
        if let weightCacheFileURL {
            try? modelWeightCache.store(modelID: descriptor.id.rawValue, directory: modelDirectory)
            try? modelWeightCache.save(to: weightCacheFileURL)
        }
    }

    private func persistWeightCacheEviction(for modelID: ModelIdentifier) {
        modelWeightCache.evict(modelID: modelID.rawValue)
        guard let weightCacheFileURL else { return }
        try? modelWeightCache.save(to: weightCacheFileURL)
    }

    private static var defaultWeightCacheFileURL: URL? {
        FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first?
            .appendingPathComponent("ValarTTS/MLXModelWeightCache.json")
    }

    private static func defaultWarningHandler(_ message: String) {
        logger.warning("\(message, privacy: .private)")
    }

    private static func defaultMetalMemoryBudgetProvider() -> UInt64? {
        MTLCreateSystemDefaultDevice()?.recommendedMaxWorkingSetSize
    }

    private static func defaultMetalAllocatedSizeProvider() -> UInt64? {
        guard let device = MTLCreateSystemDefaultDevice() else { return nil }
        return UInt64(device.currentAllocatedSize)
    }

    private static func makeLoaders(
        qwenModelLoader: @escaping ModelLoader,
        qwenASRModelLoader: @escaping ModelLoader,
        qwenAlignerModelLoader: @escaping ModelLoader,
        voxtralTTSModelLoader: @escaping ModelLoader,
        tadaTTSModelLoader: @escaping ModelLoader,
        vibeVoiceTTSModelLoader: @escaping ModelLoader
    ) -> [ModelFamilyID: ModelLoader] {
        [
            .qwen3TTS: qwenModelLoader,
            .qwen3ASR: qwenASRModelLoader,
            .qwen3ForcedAligner: qwenAlignerModelLoader,
            .soprano: qwenModelLoader,
            .orpheus: qwenModelLoader,
            .marvis: qwenModelLoader,
            .chatterbox: qwenModelLoader,
            .pocketTTS: qwenModelLoader,
            .voxtralTTS: voxtralTTSModelLoader,
            .tadaTTS: tadaTTSModelLoader,
            .vibevoiceRealtimeTTS: vibeVoiceTTSModelLoader,
        ]
    }

    private static func makeAdapters() -> [ModelFamilyID: any ModelAdapter] {
        [
            .qwen3ASR: Qwen3ASRAdapter(),
            .qwen3ForcedAligner: Qwen3AlignerAdapter(),
            .soprano: GenericTTSAdapter(familyID: .soprano),
            .orpheus: GenericTTSAdapter(familyID: .orpheus),
            .marvis: GenericTTSAdapter(familyID: .marvis),
            .chatterbox: GenericTTSAdapter(familyID: .chatterbox),
            .pocketTTS: GenericTTSAdapter(familyID: .pocketTTS),
            .voxtralTTS: GenericTTSAdapter(familyID: .voxtralTTS),
            .tadaTTS: TADATTSAdapter(),
            .vibevoiceRealtimeTTS: VibeVoiceTTSAdapter(),
        ]
    }

    /// Checks Metal GPU availability and compares the model's minimum memory requirement
    /// against `recommendedMaxWorkingSetSize`. Issues warnings rather than errors because
    /// `recommendedMaxWorkingSetSize` is advisory and Metal may be absent in CI environments.
    private static func validateMetalBudget(
        minimumMemoryBytes: Int?,
        metalMemoryBudgetProvider: MetalMemoryBudgetProvider,
        warningHandler: WarningHandler
    ) {
        guard let minimumBytes = minimumMemoryBytes else { return }

        guard let budget = metalMemoryBudgetProvider() else {
            warningHandler(
                "Metal GPU is not available on this system. MLX inference requires Metal; " +
                "loading a model that needs \(minimumBytes / (1024 * 1024)) MB will fail at runtime."
            )
            return
        }

        if UInt64(minimumBytes) > budget {
            let minimumMB = minimumBytes / (1024 * 1024)
            let budgetMB = Int(budget) / (1024 * 1024)
            warningHandler(
                "Model requires approximately \(minimumMB) MB of GPU memory but the " +
                "recommended GPU budget is \(budgetMB) MB. " +
                "Loading may fail or cause memory pressure."
            )
        }
    }

    private static func defaultQwenModelLoader(_ descriptor: ModelDescriptor) async throws -> MLXModelHandle {
        guard let resolvedURL = try await defaultModelDirectoryResolver(for: descriptor) else {
            throw MLXBackendError.modelNotFound(descriptor.id)
        }
        let mlxModel = try await TTS.fromDirectory(resolvedURL)
        return MLXModelHandle(descriptor: descriptor, mlxModel: mlxModel)
    }

    private static func defaultQwenASRModelLoader(_ descriptor: ModelDescriptor) async throws -> MLXModelHandle {
        guard let resolvedURL = try await defaultModelDirectoryResolver(for: descriptor) else {
            throw MLXBackendError.modelNotFound(descriptor.id)
        }
        let mlxModel = try await Qwen3ASRModel.fromDirectory(resolvedURL)
        return MLXASRModelHandle(descriptor: descriptor, mlxSTTModel: mlxModel)
    }

    private static func defaultQwenAlignerModelLoader(_ descriptor: ModelDescriptor) async throws -> MLXModelHandle {
        guard let resolvedURL = try await defaultModelDirectoryResolver(for: descriptor) else {
            throw MLXBackendError.modelNotFound(descriptor.id)
        }
        let mlxModel = try await Qwen3ForcedAlignerModel.fromDirectory(resolvedURL)
        return MLXAlignerModelHandle(descriptor: descriptor, mlxAlignerModel: mlxModel)
    }

    private static func defaultVoxtralTTSModelLoader(_ descriptor: ModelDescriptor) async throws -> MLXModelHandle {
        guard let resolvedURL = try await defaultModelDirectoryResolver(for: descriptor) else {
            throw MLXBackendError.modelNotFound(descriptor.id)
        }
        let mlxModel = try VoxtralTTSModel.fromDirectory(resolvedURL)
        return MLXModelHandle(descriptor: descriptor, mlxModel: mlxModel)
    }

    private static func defaultTadaTTSModelLoader(_ descriptor: ModelDescriptor) async throws -> MLXModelHandle {
        guard let resolvedURL = try await defaultModelDirectoryResolver(for: descriptor) else {
            throw MLXBackendError.modelNotFound(descriptor.id)
        }
        let mlxModel = try await TTS.fromDirectory(resolvedURL)
        return MLXModelHandle(descriptor: descriptor, mlxModel: mlxModel)
    }

    private static func defaultVibeVoiceTTSModelLoader(_ descriptor: ModelDescriptor) async throws -> MLXModelHandle {
        guard let resolvedURL = try await defaultModelDirectoryResolver(for: descriptor) else {
            throw MLXBackendError.modelNotFound(descriptor.id)
        }
        let mlxModel = try await VibeVoiceTTSModel.fromDirectory(resolvedURL)
        // Pre-load default voice cache during model warm-up for faster first synthesis
        _ = try? mlxModel.prewarmDefaultVoice()
        return MLXModelHandle(descriptor: descriptor, mlxModel: mlxModel)
    }

    private static func defaultModelDirectoryResolver(for descriptor: ModelDescriptor) async throws -> URL? {
        if let installedDirectory = installedModelPackDirectory(for: descriptor) {
            return installedDirectory
        }

        guard let repoID = Repo.ID(rawValue: descriptor.id.rawValue) else { return nil }

        switch descriptor.familyID {
        case .qwen3TTS, .qwen3ASR, .qwen3ForcedAligner, .soprano, .orpheus, .marvis, .chatterbox, .pocketTTS:
            return try await ModelUtils.resolveOrDownloadModel(
                repoID: repoID,
                requiredExtension: "safetensors"
            )
        case .voxtralTTS:
            return try await ModelUtils.resolveOrDownloadModel(
                repoID: repoID,
                requiredExtension: "safetensors",
                additionalMatchingPatterns: [
                    "params.json",
                    "tekken.json",
                    "voice_embedding/*.safetensors",
                    "voice_embedding_safe/*.bin",
                    "voice_embedding_safe/*.json",
                ]
            )
        case .tadaTTS:
            return try await resolveOrDownloadTadaModel(repoID: repoID)
        case .vibevoiceRealtimeTTS:
            return try await ModelUtils.resolveOrDownloadModel(
                repoID: repoID,
                requiredExtension: "safetensors",
                additionalMatchingPatterns: [
                    "voices/*.safetensors",
                ]
            )
        default:
            return nil
        }
    }

    private static func installedModelPackDirectory(for descriptor: ModelDescriptor) -> URL? {
        guard supportsInstalledPackDirectory(for: descriptor.familyID) else {
            return nil
        }

        let baseDirectory = installedModelPackBaseDirectory()
        let familyDirectory = sanitizePathIdentifier(descriptor.familyID.rawValue)
        let modelDirectory = sanitizePathIdentifier(descriptor.id.rawValue)
        let candidate = baseDirectory
            .appendingPathComponent(familyDirectory, isDirectory: true)
            .appendingPathComponent(modelDirectory, isDirectory: true)
        let manifestURL = candidate.appendingPathComponent("manifest.json", isDirectory: false)
        guard fileManager.fileExists(atPath: manifestURL.path) else {
            return nil
        }
        guard isInstalledPackDirectoryUsable(candidate, familyID: descriptor.familyID) else {
            return nil
        }
        return candidate
    }

    private static func supportsInstalledPackDirectory(for familyID: ModelFamilyID) -> Bool {
        switch familyID {
        case .qwen3TTS, .qwen3ASR, .qwen3ForcedAligner,
             .soprano, .orpheus, .marvis, .chatterbox, .pocketTTS,
             .voxtralTTS, .tadaTTS, .vibevoiceRealtimeTTS:
            return true
        default:
            return false
        }
    }

    private static func isInstalledPackDirectoryUsable(_ directory: URL, familyID: ModelFamilyID) -> Bool {
        switch familyID {
        case .qwen3TTS:
            return hasAnyInstalledPackPath(
                in: directory,
                candidates: ["config.json"]
            ) && hasAnyInstalledPackPath(
                in: directory,
                candidates: ["model.safetensors", "weights/model.safetensors"]
            ) && (hasAnyInstalledPackPath(
                in: directory,
                candidates: ["tokenizer.json", "tokenizers/tokenizer.json"]
            ) || (hasAnyInstalledPackPath(
                in: directory,
                candidates: ["vocab.json"]
            ) && hasAnyInstalledPackPath(
                in: directory,
                candidates: ["merges.txt"]
            ))) && hasAnyInstalledPackPath(
                in: directory,
                candidates: ["speech_tokenizer/model.safetensors"]
            )
        case .qwen3ASR, .qwen3ForcedAligner:
            return hasAnyInstalledPackPath(
                in: directory,
                candidates: ["config.json"]
            ) && hasAnyInstalledPackPath(
                in: directory,
                candidates: ["model.safetensors", "weights/model.safetensors"]
            ) && hasAnyInstalledPackPath(
                in: directory,
                candidates: ["tokenizer.json", "tokenizers/tokenizer.json"]
            )
        case .soprano, .orpheus, .marvis:
            return hasAnyInstalledPackPath(
                in: directory,
                candidates: ["config.json"]
            ) && hasAnyInstalledPackPath(
                in: directory,
                candidates: ["model.safetensors", "weights/model.safetensors"]
            ) && hasAnyInstalledPackPath(
                in: directory,
                candidates: ["tokenizer.json", "tokenizers/tokenizer.json"]
            )
        case .chatterbox:
            return hasAnyInstalledPackPath(
                in: directory,
                candidates: ["config.json"]
            ) && hasAnyInstalledPackPath(
                in: directory,
                candidates: ["model.safetensors", "weights/model.safetensors"]
            ) && (
                hasAnyInstalledPackPath(
                    in: directory,
                    candidates: ["tokenizer.json", "tokenizers/tokenizer.json"]
                ) || hasAnyInstalledPackPath(
                    in: directory,
                    candidates: ["vocab.json"]
                ) && hasAnyInstalledPackPath(
                    in: directory,
                    candidates: ["merges.txt"]
                )
            )
        case .pocketTTS:
            return hasAnyInstalledPackPath(
                in: directory,
                candidates: ["config.json"]
            ) && hasAnyInstalledPackPath(
                in: directory,
                candidates: ["model.safetensors", "weights/model.safetensors"]
            )
        case .voxtralTTS:
            return hasAnyInstalledPackPath(
                in: directory,
                candidates: ["params.json", "config.json"]
            ) && hasAnyInstalledPackPath(
                in: directory,
                candidates: ["voice_embedding_safe", "voice_embedding"]
            )
        case .tadaTTS:
            return isTadaModelDirectoryComplete(directory)
        case .vibevoiceRealtimeTTS:
            return hasAnyInstalledPackPath(
                in: directory,
                candidates: ["config.json"]
            ) && hasAnyInstalledPackPath(
                in: directory,
                candidates: ["model.safetensors"]
            ) && hasVoicesDirectory(in: directory)
        default:
            return false
        }
    }

    private static func hasAnyInstalledPackPath(in directory: URL, candidates: [String]) -> Bool {
        candidates.contains { relativePath in
            let candidate = directory.appendingPathComponent(relativePath, isDirectory: false)
            return fileManager.fileExists(atPath: candidate.path)
        }
    }

    /// Checks that a `voices/` subdirectory exists and contains at least one `.safetensors` file.
    private static func hasVoicesDirectory(in directory: URL) -> Bool {
        let voicesDir = directory.appendingPathComponent("voices", isDirectory: true)
        var isDir: ObjCBool = false
        guard fileManager.fileExists(atPath: voicesDir.path, isDirectory: &isDir), isDir.boolValue else {
            return false
        }
        guard let contents = try? fileManager.contentsOfDirectory(atPath: voicesDir.path) else {
            return false
        }
        return contents.contains { $0.hasSuffix(".safetensors") }
    }

    private static func installedModelPackBaseDirectory() -> URL {
        if let rawOverride = processEnvironment["VALARTTS_CLI_HOME"]?.trimmingCharacters(in: asciiWhitespace),
           !rawOverride.isEmpty {
            return URL(fileURLWithPath: rawOverride, isDirectory: true)
                .standardizedFileURL
                .appendingPathComponent("ModelPacks", isDirectory: true)
        }
        if let rawOverride = processEnvironment["VALARTTS_HOME"]?.trimmingCharacters(in: asciiWhitespace),
           !rawOverride.isEmpty {
            return URL(fileURLWithPath: rawOverride, isDirectory: true)
                .standardizedFileURL
                .appendingPathComponent("ModelPacks", isDirectory: true)
        }

        let applicationSupport = fileManager.urls(
            for: .applicationSupportDirectory,
            in: .userDomainMask
        ).first ?? fileManager.homeDirectoryForCurrentUser
            .appendingPathComponent("Library", isDirectory: true)
            .appendingPathComponent("Application Support", isDirectory: true)

        return applicationSupport
            .appendingPathComponent("ValarTTS", isDirectory: true)
            .appendingPathComponent("ModelPacks", isDirectory: true)
    }

    private static func sanitizePathIdentifier(_ value: String) -> String {
        let allowed = CharacterSet.alphanumerics.union(.whitespaces)
        let collapsed = value
            .unicodeScalars
            .map { allowed.contains($0) ? Character($0) : " " }
            .reduce(into: "") { $0.append($1) }
            .trimmingCharacters(in: asciiWhitespace)
            .lowercased()

        let sanitized = collapsed
            .components(separatedBy: asciiWhitespace)
            .filter { !$0.isEmpty }
            .joined(separator: "-")

        return sanitized.isEmpty ? "model" : sanitized
    }

    private static func validateModelDirectory(
        _ directory: URL,
        familyID: ModelFamilyID,
        warningHandler: @escaping WarningHandler = defaultWarningHandler
    ) async throws {
        // Pass 1 (synchronous): enumerate directory to reject unsafe weight formats,
        // accumulate warnings, and collect safetensors URLs for parallel validation.
        // FileManager.DirectoryEnumerator cannot be iterated in async contexts.
        let (safeTensorsURLs, foundRootSafeTensor, relativePaths) = try enumerateModelDirectory(
            directory, warningHandler: warningHandler
        )

        try validateExpectedModelLayout(
            relativePaths: relativePaths,
            foundRootSafeTensor: foundRootSafeTensor,
            familyID: familyID,
            directory: directory
        )

        // Pass 2: validate safetensors headers in parallel.
        try await withThrowingTaskGroup(of: Void.self) { group in
            for url in safeTensorsURLs {
                group.addTask {
                    try Self.validateSafeTensorsHeader(at: url)
                }
            }
            try await group.waitForAll()
        }
    }

    static func validateModelDirectoryForTests(
        _ directory: URL,
        familyID: ModelFamilyID,
        warningHandler: @escaping @Sendable (String) -> Void
    ) async throws {
        try await validateModelDirectory(
            directory,
            familyID: familyID,
            warningHandler: warningHandler
        )
    }

    private static func validateCachedModelDirectory(
        _ directory: URL,
        familyID: ModelFamilyID,
        warningHandler: @escaping WarningHandler = defaultWarningHandler
    ) throws {
        let (_, foundRootSafeTensor, relativePaths) = try enumerateModelDirectory(
            directory,
            warningHandler: warningHandler
        )

        try validateExpectedModelLayout(
            relativePaths: relativePaths,
            foundRootSafeTensor: foundRootSafeTensor,
            familyID: familyID,
            directory: directory
        )
    }

    private static func enumerateModelDirectory(
        _ directory: URL,
        warningHandler: @escaping WarningHandler = defaultWarningHandler
    ) throws -> (safeTensorsURLs: [URL], foundRootSafeTensor: Bool, relativePaths: Set<String>) {
        let fileManager = FileManager.default
        // Canonicalize the directory to remove `.`/`..` components and resolve symlinks.
        // All containment checks below are performed against this canonical path.
        let resolvedDirectory = directory.standardized.resolvingSymlinksInPath()
        let rootPrefix = resolvedDirectory.path.hasSuffix("/")
            ? resolvedDirectory.path
            : "\(resolvedDirectory.path)/"
        guard let enumerator = fileManager.enumerator(
            at: resolvedDirectory,
            includingPropertiesForKeys: [.isDirectoryKey, .isRegularFileKey, .fileSizeKey],
            options: []
        ) else {
            throw MLXBackendError.inferenceError("Failed to enumerate model directory at \(directory.path)")
        }

        var safeTensorsURLs: [URL] = []
        var foundRootSafeTensor = false
        var relativePaths: Set<String> = []

        for case let fileURL as URL in enumerator {
            let values = try fileURL.resourceValues(forKeys: [.isDirectoryKey, .isRegularFileKey])
            if values.isDirectory == true || values.isRegularFile != true {
                continue
            }

            let relativePath = resolvedRelativePath(for: fileURL, rootDirectory: resolvedDirectory)
            relativePaths.insert(relativePath)
            let fileExtension = fileURL.pathExtension.lowercased()
            if rejectedWeightFileExtensions.contains(fileExtension),
               !isAllowedSerializedModelFile(relativePath: relativePath, fileExtension: fileExtension)
            {
                throw MLXBackendError.rejectedUnsafeWeightFile(
                    path: fileURL.path,
                    fileExtension: fileExtension
                )
            }

            if fileExtension == "safetensors" {
                // Resolve symlinks on the file before containment check so that a
                // symlink pointing outside the model directory is caught and rejected
                // rather than silently loading weights from an arbitrary path.
                let resolvedFile = fileURL.resolvingSymlinksInPath()
                guard resolvedFile.path.hasPrefix(rootPrefix) else {
                    throw MLXBackendError.pathTraversalDetected(fileURL.path)
                }
                if resolvedFile
                    .deletingLastPathComponent()
                    .standardizedFileURL == resolvedDirectory.standardizedFileURL
                {
                    foundRootSafeTensor = true
                }
                safeTensorsURLs.append(fileURL)
                continue
            }
            guard isExpectedModelFile(relativePath: relativePath, fileExtension: fileExtension) else {
                warningHandler("Unexpected file in model directory '\(directory.path)': \(fileURL.path)")
                continue
            }
        }

        return (safeTensorsURLs, foundRootSafeTensor, relativePaths)
    }

    private static func validateSafeTensorsHeader(at fileURL: URL) throws {
        let fileHandle = try FileHandle(forReadingFrom: fileURL)
        defer { try? fileHandle.close() }

        let headerPrefix = try fileHandle.read(upToCount: 32) ?? Data()
        guard headerPrefix.count >= 10 else {
            throw MLXBackendError.invalidSafeTensorsHeader(fileURL.path)
        }

        let fileSize = (try fileURL.resourceValues(forKeys: [.fileSizeKey])).fileSize ?? 0
        guard fileSize >= 10 else {
            throw MLXBackendError.invalidSafeTensorsHeader(fileURL.path)
        }

        let headerLength = readLittleEndianUInt64(from: headerPrefix.prefix(8))
        guard headerLength <= 64 * 1024 * 1024 else {
            throw MLXBackendError.invalidSafeTensorsHeader(fileURL.path)
        }
        let remainingBytes = UInt64(max(0, fileSize - 8))
        guard headerLength > 1, headerLength <= remainingBytes else {
            throw MLXBackendError.invalidSafeTensorsHeader(fileURL.path)
        }

        let jsonPrefix = headerPrefix.dropFirst(8)
        guard let firstNonWhitespace = jsonPrefix.first(where: { byte in
            let scalar = UnicodeScalar(byte)
            return !asciiWhitespace.contains(scalar)
        }),
        firstNonWhitespace == 0x7B
        else {
            throw MLXBackendError.invalidSafeTensorsHeader(fileURL.path)
        }
    }

    private static func readLittleEndianUInt64<S: Sequence>(from bytes: S) -> UInt64 where S.Element == UInt8 {
        bytes.enumerated().reduce(into: UInt64(0)) { partialResult, item in
            partialResult |= UInt64(item.element) << (item.offset * 8)
        }
    }

    private static func isExpectedModelFile(relativePath: String, fileExtension: String) -> Bool {
        let components = relativePath.split(separator: "/", omittingEmptySubsequences: true)
        guard !components.isEmpty else { return false }

        if components.count == 1 {
            return fileExtension == "safetensors"
                || expectedRootModelFiles.contains(relativePath)
        }

        if components.first == "speech_tokenizer" {
            return fileExtension == "safetensors" || fileExtension == "json"
        }

        if components.first == "prompts" {
            return fileExtension == "wav" || fileExtension == "txt" || fileExtension == "safetensors"
        }

        if components.first == "embeddings" {
            return fileExtension == "safetensors"
        }

        if components.first == "voice_embedding_safe" {
            return fileExtension == "bin" || fileExtension == "json"
        }

        if components.first == "voice_embedding" {
            return fileExtension == "safetensors" || fileExtension == "pt"
        }

        if components.first == "voices" {
            return fileExtension == "safetensors"
        }

        if let firstComponent = components.first, tadaComponentDirectories.contains(String(firstComponent)) {
            return fileExtension == "safetensors" || fileExtension == "json"
        }

        return false
    }

    private static func isAllowedSerializedModelFile(relativePath: String, fileExtension: String) -> Bool {
        guard fileExtension == "bin" else { return false }
        let components = relativePath.split(separator: "/", omittingEmptySubsequences: true)
        return components.count == 2 && components.first == "voice_embedding_safe"
    }

    private static func resolvedRelativePath(for fileURL: URL, rootDirectory: URL) -> String {
        let resolvedFilePath = fileURL.resolvingSymlinksInPath().path
        let resolvedRootPath = rootDirectory.path.hasSuffix("/")
            ? rootDirectory.path
            : "\(rootDirectory.path)/"
        return resolvedFilePath.replacingOccurrences(of: resolvedRootPath, with: "")
    }

    private static func validateExpectedModelLayout(
        relativePaths: Set<String>,
        foundRootSafeTensor: Bool,
        familyID: ModelFamilyID,
        directory: URL
    ) throws {
        if familyID == .tadaTTS {
            guard relativePaths.contains("tokenizer.json") else {
                throw MLXBackendError.inferenceError(
                    "TADA install is missing tokenizer.json. Accept the Meta Llama 3.2 license and reinstall the model."
                )
            }
            guard relativePaths.contains("model/config.json") else {
                throw MLXBackendError.inferenceError("TADA install is missing model/config.json.")
            }
            let hasAllComponents = tadaComponentDirectories.allSatisfy { component in
                relativePaths.contains { path in
                    path.hasPrefix("\(component)/") && path.hasSuffix(".safetensors")
                }
            }
            guard hasAllComponents else {
                throw MLXBackendError.missingSafeTensorsWeights(directory.path)
            }
            return
        }

        guard foundRootSafeTensor else {
            throw MLXBackendError.missingSafeTensorsWeights(directory.path)
        }
    }

    private static func resolveOrDownloadTadaModel(
        repoID: Repo.ID,
        hfToken: String? = nil,
        cache: HubCache = .default
    ) async throws -> URL {
        if let hubSnapshot = preferredHubSnapshotDirectory(
            repoID: repoID,
            cache: cache,
            requiredRelativePaths: ["model/config.json", "tokenizer.json"]
        ), isTadaModelDirectoryComplete(hubSnapshot) {
            return hubSnapshot
        }

        let client: HubClient
        if let token = hfToken, token.isEmpty == false {
            client = HubClient(host: HubClient.defaultHost, bearerToken: token, cache: cache)
        } else {
            client = HubClient(cache: cache)
        }
        let resolvedCache = client.cache ?? cache
        let modelSubdir = repoID.description.replacingOccurrences(of: "/", with: "_")
        let modelDir = resolvedCache.cacheDirectory
            .appendingPathComponent("mlx-audio")
            .appendingPathComponent(modelSubdir)

        // Backward compatibility for older mlx-audio cache layouts already present on disk.
        if fileManager.fileExists(atPath: modelDir.path) {
            if isTadaModelDirectoryComplete(modelDir) {
                return modelDir
            }
            clearCaches(modelDir: modelDir, repoID: repoID, hubCache: resolvedCache)
        }

        let matching = [
            "model/*.json",
            "model/*.safetensors",
            "encoder/*.safetensors",
            "decoder/*.safetensors",
            "aligner/*.safetensors",
            "tokenizer.json",
            "LICENSE*",
            "*.txt",
        ]

        let snapshotDir = try await client.downloadSnapshot(
            of: repoID,
            kind: .model,
            revision: "main",
            matching: matching,
            progressHandler: { progress in
                print("\(progress.completedUnitCount)/\(progress.totalUnitCount) files")
            }
        )

        guard isTadaModelDirectoryComplete(snapshotDir) else {
            clearCaches(modelDir: modelDir, repoID: repoID, hubCache: resolvedCache)
            let tokenizerURL = snapshotDir.appendingPathComponent("tokenizer.json")
            if fileManager.fileExists(atPath: tokenizerURL.path) == false {
                throw MLXBackendError.inferenceError(
                    "TADA install is missing tokenizer.json. Accept the Meta Llama 3.2 license and reinstall the model."
                )
            }
            throw ModelUtilsError.incompleteDownload(repoID.description)
        }

        return snapshotDir
    }

    private static func preferredHubSnapshotDirectory(
        repoID: Repo.ID,
        cache: HubCache,
        requiredRelativePaths: [String]
    ) -> URL? {
        let repoDirectory = cache.repoDirectory(repo: repoID, kind: .model)
        let snapshotsDirectory = repoDirectory.appendingPathComponent("snapshots", isDirectory: true)
        guard fileManager.fileExists(atPath: snapshotsDirectory.path) else {
            return nil
        }

        let snapshotDirectory: URL?
        let refsMain = repoDirectory.appendingPathComponent("refs/main", isDirectory: false)
        if let revision = try? String(contentsOf: refsMain, encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines),
           !revision.isEmpty {
            let candidate = snapshotsDirectory.appendingPathComponent(revision, isDirectory: true)
            snapshotDirectory = fileManager.fileExists(atPath: candidate.path) ? candidate : nil
        } else {
            let snapshots = (try? fileManager.contentsOfDirectory(
                at: snapshotsDirectory,
                includingPropertiesForKeys: [.contentModificationDateKey],
                options: [.skipsHiddenFiles]
            )) ?? []
            snapshotDirectory = snapshots.max(by: {
                let lhs = (try? $0.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
                let rhs = (try? $1.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
                return lhs < rhs
            })
        }

        guard let snapshotDirectory else {
            return nil
        }

        let hasRequiredFiles = requiredRelativePaths.allSatisfy { relativePath in
            let candidate = snapshotDirectory.appendingPathComponent(relativePath, isDirectory: false)
            guard fileManager.fileExists(atPath: candidate.path) else {
                return false
            }
            let size = (try? candidate.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
            return size > 0
        }
        return hasRequiredFiles ? snapshotDirectory : nil
    }

    private static func isTadaModelDirectoryComplete(_ modelDirectory: URL) -> Bool {
        let tokenizerURL = modelDirectory.appendingPathComponent("tokenizer.json")
        let configURL = modelDirectory.appendingPathComponent("model/config.json")

        guard fileManager.fileExists(atPath: tokenizerURL.path),
              fileManager.fileExists(atPath: configURL.path),
              let configData = try? Data(contentsOf: configURL),
              (try? JSONSerialization.jsonObject(with: configData)) != nil else {
            return false
        }

        return tadaComponentDirectories.allSatisfy { component in
            containsNonZeroSafeTensors(in: modelDirectory.appendingPathComponent(component, isDirectory: true))
        }
    }

    private static func containsNonZeroSafeTensors(in directory: URL) -> Bool {
        guard let files = try? fileManager.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: [.fileSizeKey],
            options: [.skipsHiddenFiles]
        ) else {
            return false
        }

        return files.contains { file in
            guard file.pathExtension.lowercased() == "safetensors" else { return false }
            let size = (try? file.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0
            return size > 0
        }
    }

    private static func clearCaches(modelDir: URL, repoID: Repo.ID, hubCache: HubCache) {
        try? fileManager.removeItem(at: modelDir)
        let hubRepoDir = hubCache.repoDirectory(repo: repoID, kind: .model)
        if fileManager.fileExists(atPath: hubRepoDir.path) {
            try? fileManager.removeItem(at: hubRepoDir)
        }
    }
}

public enum MLXBackendError: Error, Sendable, LocalizedError, Equatable {
    case unsupportedBackend(BackendKind)
    case unsupportedFamily(ModelFamilyID)
    case modelNotFound(ModelIdentifier)
    case inferenceError(String)
    case streamingError(String)
    case rejectedUnsafeWeightFile(path: String, fileExtension: String)
    case invalidSafeTensorsHeader(String)
    case missingSafeTensorsWeights(String)
    case pathTraversalDetected(String)

    public var errorDescription: String? {
        switch self {
        case .unsupportedBackend(let backendKind):
            return "Unsupported backend: \(backendKind.rawValue)"
        case .unsupportedFamily(let familyID):
            return "Unsupported model family for MLX inference backend: \(familyID.rawValue)"
        case .modelNotFound(let identifier):
            return "Model not found: \(identifier.rawValue)"
        case .inferenceError(let message), .streamingError(let message):
            return message
        case .rejectedUnsafeWeightFile(let path, let fileExtension):
            return "Unsafe model weight format '.\(fileExtension)' rejected at \(path). Only .safetensors weights are allowed."
        case .invalidSafeTensorsHeader(let path):
            return "Invalid safetensors header magic bytes at \(path). Refusing to load the file."
        case .missingSafeTensorsWeights(let directory):
            return "No .safetensors weight files were found in \(directory)."
        case .pathTraversalDetected(let path):
            return "Path traversal detected in model directory: \(path). Refusing to load files outside the model directory sandbox."
        }
    }
}
