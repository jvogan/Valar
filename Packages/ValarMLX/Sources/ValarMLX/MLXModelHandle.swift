import Foundation
@preconcurrency import MLX
import MLXAudioSTT
import MLXAudioTTS
import MLXAudioCore
import MLXLMCommon
import os
import ValarModelKit

public enum SynthesisExecutionMode: String, Sendable, Equatable {
    case oneShot
    case segmentedContinuation
}

public enum SynthesisExecutionEventKind: String, Sendable, Equatable {
    case started
    case heartbeat
    case segmentStarted
    case segmentCompleted
    case completed
    case cancelled
    case failed
}

public struct SynthesisExecutionEvent: Sendable, Equatable {
    public let kind: SynthesisExecutionEventKind
    public let executionMode: SynthesisExecutionMode
    public let segmentIndex: Int
    public let segmentCount: Int
    public let generatedTokenCount: Int?
    public let usesAnchorConditioning: Bool?
    public let chunkCharacterCount: Int?
    public let maxTokenCount: Int?
    public let prefillTokenCount: Int?
    public let segmentPrefillTimeSeconds: Double?
    public let segmentDecodeTimeSeconds: Double?
    public let anchorSegmentDecodeTimeSeconds: Double?
    public let continuationSegmentDecodeTimeSeconds: Double?
    public let samplingTimeSeconds: Double?
    public let evalTimeSeconds: Double?
    public let tokenMaterializationTimeSeconds: Double?
    public let embeddingAssemblyTimeSeconds: Double?
    public let talkerForwardTimeSeconds: Double?
    public let codePredictorTimeSeconds: Double?
    public let segmentWallTimeSeconds: Double?
    public let segmentAudioDurationSeconds: Double?
    public let continuationOutlier: Bool?

    public init(
        kind: SynthesisExecutionEventKind,
        executionMode: SynthesisExecutionMode,
        segmentIndex: Int = 0,
        segmentCount: Int = 1,
        generatedTokenCount: Int? = nil,
        usesAnchorConditioning: Bool? = nil,
        chunkCharacterCount: Int? = nil,
        maxTokenCount: Int? = nil,
        prefillTokenCount: Int? = nil,
        segmentPrefillTimeSeconds: Double? = nil,
        segmentDecodeTimeSeconds: Double? = nil,
        anchorSegmentDecodeTimeSeconds: Double? = nil,
        continuationSegmentDecodeTimeSeconds: Double? = nil,
        samplingTimeSeconds: Double? = nil,
        evalTimeSeconds: Double? = nil,
        tokenMaterializationTimeSeconds: Double? = nil,
        embeddingAssemblyTimeSeconds: Double? = nil,
        talkerForwardTimeSeconds: Double? = nil,
        codePredictorTimeSeconds: Double? = nil,
        segmentWallTimeSeconds: Double? = nil,
        segmentAudioDurationSeconds: Double? = nil,
        continuationOutlier: Bool? = nil
    ) {
        self.kind = kind
        self.executionMode = executionMode
        self.segmentIndex = segmentIndex
        self.segmentCount = segmentCount
        self.generatedTokenCount = generatedTokenCount
        self.usesAnchorConditioning = usesAnchorConditioning
        self.chunkCharacterCount = chunkCharacterCount
        self.maxTokenCount = maxTokenCount
        self.prefillTokenCount = prefillTokenCount
        self.segmentPrefillTimeSeconds = segmentPrefillTimeSeconds
        self.segmentDecodeTimeSeconds = segmentDecodeTimeSeconds
        self.anchorSegmentDecodeTimeSeconds = anchorSegmentDecodeTimeSeconds
        self.continuationSegmentDecodeTimeSeconds = continuationSegmentDecodeTimeSeconds
        self.samplingTimeSeconds = samplingTimeSeconds
        self.evalTimeSeconds = evalTimeSeconds
        self.tokenMaterializationTimeSeconds = tokenMaterializationTimeSeconds
        self.embeddingAssemblyTimeSeconds = embeddingAssemblyTimeSeconds
        self.talkerForwardTimeSeconds = talkerForwardTimeSeconds
        self.codePredictorTimeSeconds = codePredictorTimeSeconds
        self.segmentWallTimeSeconds = segmentWallTimeSeconds
        self.segmentAudioDurationSeconds = segmentAudioDurationSeconds
        self.continuationOutlier = continuationOutlier
    }
}

public enum SynthesisExecutionObserverContext {
    @TaskLocal public static var observer: (@Sendable (SynthesisExecutionEvent) -> Void)?
}

struct SynthesisSegmentObservationContext: Sendable {
    let executionMode: SynthesisExecutionMode
    let segmentIndex: Int
    let segmentCount: Int
    let usesAnchorConditioning: Bool?
    let chunkCharacterCount: Int?
    let maxTokenCount: Int?
    let startedAt: Date
}

enum SynthesisSegmentObservationTaskContext {
    @TaskLocal static var value: SynthesisSegmentObservationContext?
}

protocol SpeechFamilyOrchestrating: Sendable {
    func prepareExecution(
        handle: MLXModelHandle,
        request: SpeechSynthesisRequest
    ) async throws -> (any SpeechFamilyPreparedExecution)?

    func synthesize(
        handle: MLXModelHandle,
        request: SpeechSynthesisRequest,
        preparedExecution: (any SpeechFamilyPreparedExecution)?,
        session: ModelRuntimeSession
    ) async throws -> AudioChunk

    func synthesizeStream(
        handle: MLXModelHandle,
        request: SpeechSynthesisRequest,
        preparedExecution: (any SpeechFamilyPreparedExecution)?
    ) async throws -> AsyncThrowingStream<AudioChunk, Error>
}

protocol SpeechFamilyPreparedExecution {}

struct DefaultSpeechFamilyOrchestrator: SpeechFamilyOrchestrating {
    func prepareExecution(
        handle: MLXModelHandle,
        request: SpeechSynthesisRequest
    ) async throws -> (any SpeechFamilyPreparedExecution)? {
        _ = handle
        _ = request
        return nil
    }

    func synthesize(
        handle: MLXModelHandle,
        request: SpeechSynthesisRequest,
        preparedExecution: (any SpeechFamilyPreparedExecution)?,
        session: ModelRuntimeSession
    ) async throws -> AudioChunk {
        _ = preparedExecution
        return try await handle.synthesizeOneShot(
            request: request,
            in: session,
            behaviorOverride: .expressive,
            automaticQwenCeiling: nil,
            executionMode: .oneShot
        )
    }

    func synthesizeStream(
        handle: MLXModelHandle,
        request: SpeechSynthesisRequest,
        preparedExecution: (any SpeechFamilyPreparedExecution)?
    ) async throws -> AsyncThrowingStream<AudioChunk, Error> {
        _ = preparedExecution
        return try await handle.synthesizeStreamOneShot(
            request: request,
            behaviorOverride: .expressive,
            automaticQwenCeiling: nil,
            executionMode: .oneShot
        )
    }
}

struct QwenSpeechOrchestrator: SpeechFamilyOrchestrating {
    struct ExecutionPlan: Sendable {
        let behavior: SpeechSynthesisVoiceBehavior
        let mode: SynthesisExecutionMode
        let chunkPlan: QwenTextChunker.ChunkPlan
        let automaticMaxTokenCeiling: Int
    }

    private struct PreparedExecution: SpeechFamilyPreparedExecution, Sendable {
        let plan: ExecutionPlan
        let longForm: MLXModelHandle.PreparedQwenLongFormExecution?
    }

    func prepareExecution(
        handle: MLXModelHandle,
        request: SpeechSynthesisRequest
    ) async throws -> (any SpeechFamilyPreparedExecution)? {
        let plan = Self.plan(for: handle, request: request)
        let longForm = try await handle.prepareQwenLongFormExecutionIfNeeded(
            request: request,
            executionPlan: plan
        )
        return PreparedExecution(plan: plan, longForm: longForm)
    }

    func synthesize(
        handle: MLXModelHandle,
        request: SpeechSynthesisRequest,
        preparedExecution: (any SpeechFamilyPreparedExecution)?,
        session: ModelRuntimeSession
    ) async throws -> AudioChunk {
        let prepared = try await resolvedPreparedExecution(
            handle: handle,
            request: request,
            existing: preparedExecution
        )
        let plan = prepared.plan
        return try await handle.withQwenExecutionObservation(
            request: request,
            behavior: plan.behavior,
            executionMode: plan.mode,
            segmentCount: plan.chunkPlan.chunks.count
        ) {
            switch plan.mode {
            case .oneShot:
                return try await handle.synthesizeOneShot(
                    request: request,
                    in: session,
                    behaviorOverride: plan.behavior,
                    automaticQwenCeiling: plan.automaticMaxTokenCeiling,
                    executionMode: plan.mode
                )
            case .segmentedContinuation:
                return try await handle.synthesizePreparedLongForm(
                    request: request,
                    in: session,
                    preparedExecution: prepared.longForm
                )
            }
        }
    }

    func synthesizeStream(
        handle: MLXModelHandle,
        request: SpeechSynthesisRequest,
        preparedExecution: (any SpeechFamilyPreparedExecution)?
    ) async throws -> AsyncThrowingStream<AudioChunk, Error> {
        let prepared = try await resolvedPreparedExecution(
            handle: handle,
            request: request,
            existing: preparedExecution
        )
        let plan = prepared.plan

        return AsyncThrowingStream { continuation in
            let task = Task { @Sendable in
                do {
                    try await handle.withQwenExecutionObservation(
                        request: request,
                        behavior: plan.behavior,
                        executionMode: plan.mode,
                        segmentCount: plan.chunkPlan.chunks.count
                    ) {
                        let baseStream: AsyncThrowingStream<AudioChunk, Error> = switch plan.mode {
                        case .oneShot:
                            try await handle.synthesizeStreamOneShot(
                                request: request,
                                behaviorOverride: plan.behavior,
                                automaticQwenCeiling: plan.automaticMaxTokenCeiling,
                                executionMode: plan.mode
                            )
                        case .segmentedContinuation:
                            try await handle.synthesizePreparedLongFormStream(
                                request: request,
                                preparedExecution: prepared.longForm
                            )
                        }
                        for try await chunk in baseStream {
                            continuation.yield(chunk)
                        }
                    }
                    continuation.finish()
                } catch is CancellationError {
                    continuation.finish(throwing: CancellationError())
                } catch {
                    continuation.finish(throwing: error)
                }
            }

            continuation.onTermination = { @Sendable _ in
                task.cancel()
            }
        }
    }

    static func plan(
        for handle: MLXModelHandle,
        request: SpeechSynthesisRequest
    ) -> ExecutionPlan {
        let plannedExecution = MLXModelHandle.plannedQwenExecution(
            for: request,
            descriptor: handle.descriptor
        )
        let chunkPlan = QwenTextChunker.plan(text: request.text, behavior: plannedExecution.behavior)
        return ExecutionPlan(
            behavior: plannedExecution.behavior,
            mode: plannedExecution.mode,
            chunkPlan: chunkPlan,
            automaticMaxTokenCeiling: plannedExecution.automaticMaxTokenCeiling
        )
    }

    private func resolvedPreparedExecution(
        handle: MLXModelHandle,
        request: SpeechSynthesisRequest,
        existing: (any SpeechFamilyPreparedExecution)?
    ) async throws -> PreparedExecution {
        if let existing = existing as? PreparedExecution {
            return existing
        }
        if let prepared = try await prepareExecution(handle: handle, request: request) as? PreparedExecution {
            return prepared
        }
        let plan = Self.plan(for: handle, request: request)
        return PreparedExecution(plan: plan, longForm: nil)
    }
}

/// Wraps an mlx-audio-swift model instance as a ValarModel.
///
/// When mlx-audio-swift is linked, this holds a reference to the loaded model
/// (e.g., `Qwen3TTSModel`, `SopranoModel`) and delegates synthesis/recognition
/// calls to it.
public final class MLXModelHandle: ValarModel, TextToSpeechWorkflow, SpeechToTextWorkflow, ForcedAlignmentWorkflow, Sendable {
    private static let logger = Logger(subsystem: "com.valar.tts", category: "Synthesis")
    private struct InheritedObservationContexts: Sendable {
        let synthesisObserver: (@Sendable (SynthesisExecutionEvent) -> Void)?
        let heartbeatHandler: (@Sendable (AudioGenerationHeartbeat) -> Void)?
        let infoHandler: (@Sendable (AudioGenerationInfo) -> Void)?
    }

    private actor QwenHeartbeatMonitor {
        private var lastHeartbeatAt = Date()
        private var stalled = false

        func ping() {
            lastHeartbeatAt = .now
        }

        func markStalled() {
            stalled = true
        }

        func isStalled(timeoutSeconds: TimeInterval, now: Date = .now) -> Bool {
            now.timeIntervalSince(lastHeartbeatAt) > timeoutSeconds
        }

        func didStall() -> Bool {
            stalled
        }
    }

    private static func inheritedObservationContexts() -> InheritedObservationContexts {
        InheritedObservationContexts(
            synthesisObserver: SynthesisExecutionObserverContext.observer,
            heartbeatHandler: AudioGenerationObserverContext.heartbeatHandler,
            infoHandler: AudioGenerationObserverContext.infoHandler
        )
    }

    private static func withInheritedObservationContexts<T: Sendable>(
        _ contexts: InheritedObservationContexts,
        operation: @escaping @Sendable () async throws -> T
    ) async throws -> T {
        try await SynthesisExecutionObserverContext.$observer.withValue(contexts.synthesisObserver) {
            try await AudioGenerationObserverContext.$heartbeatHandler.withValue(contexts.heartbeatHandler) {
                try await AudioGenerationObserverContext.$infoHandler.withValue(contexts.infoHandler) {
                    try await operation()
                }
            }
        }
    }

    internal struct PlannedQwenExecution: Sendable, Equatable {
        let behavior: SpeechSynthesisVoiceBehavior
        let mode: SynthesisExecutionMode
        let segmentCount: Int
        let automaticMaxTokenCeiling: Int
    }

    private struct QwenClonePromptPayloadProbe: Decodable {
        let version: Int
        let refSpeakerEmbedding: Data?
        let refCode: Data?
        let refText: String?
        let numCodeGroups: Int?
        let frameCount: Int?
        let xVectorOnlyMode: Bool
        let iclMode: Bool
    }

    struct StableNarratorContinuationContext: Sendable {
        struct PreparedInput: Sendable, Equatable {
            let voice: String?
            let speaker: String?
            let instruct: String?
            let conditioning: SpeechConditioning?
            let referenceText: String?

            var resolved: ResolvedSpeechGenerationInput {
                ResolvedSpeechGenerationInput(
                    voice: voice,
                    speaker: speaker,
                    instruct: instruct,
                    conditioning: conditioning,
                    referenceAudio: nil,
                    referenceText: referenceText
                )
            }
        }

        let anchorInput: PreparedInput
        let continuationInput: PreparedInput?
        let anchorEverySegments: Int

        func preparedInput(
            segmentIndex: Int
        ) -> PreparedInput {
            let useAnchorVoice = MLXModelHandle.qwenShouldUseStableAnchorSegment(
                segmentIndex,
                anchorEverySegments: anchorEverySegments
            )
            if useAnchorVoice {
                return anchorInput
            }
            return continuationInput ?? anchorInput
        }

        func generationInput(
            segmentIndex: Int
        ) -> ResolvedSpeechGenerationInput {
            preparedInput(segmentIndex: segmentIndex).resolved
        }
    }

    struct PreparedSpeechGenerationParameters: Sendable, Equatable {
        let temperature: Float
        let topP: Float
        let repetitionPenalty: Float?
        let repetitionContextSize: Int
        let maxTokens: Int?

        func resolved(defaults: GenerateParameters) -> GenerateParameters {
            var params = defaults
            params.temperature = temperature
            params.topP = topP
            params.repetitionPenalty = repetitionPenalty
            params.repetitionContextSize = repetitionContextSize
            params.maxTokens = maxTokens
            return params
        }
    }

    struct PreparedQwenLongFormSegment: Sendable, Equatable {
        let text: String
        let characterCount: Int
        let usesAnchorConditioning: Bool
        let preparedInput: StableNarratorContinuationContext.PreparedInput?
        let generationParameters: PreparedSpeechGenerationParameters
    }

    struct PreparedQwenLongFormExecution: SpeechFamilyPreparedExecution, Sendable {
        let behavior: SpeechSynthesisVoiceBehavior
        let chunkPlan: QwenTextChunker.ChunkPlan
        let continuationContext: StableNarratorContinuationContext?
        let segments: [PreparedQwenLongFormSegment]
    }

    public let descriptor: ModelDescriptor
    public let backendKind: BackendKind = .mlx

    // mlx-audio-swift's SpeechGenerationModel is not declared Sendable, but MLXModelHandle
    // is immutable after init (all stored properties are `let`) and is only ever handed
    // across actor boundaries as a fully-constructed value. The unsafe annotation opts out
    // of the automatic Sendable check for this one property while the rest of the conformance
    // remains statically verified.
    nonisolated(unsafe) internal let mlxSpeechModel: (any SpeechGenerationModel)?
    nonisolated(unsafe) internal let mlxSpeechToTextModel: (any STTGenerationModel)?
    nonisolated(unsafe) internal let mlxForcedAlignerModel: Qwen3ForcedAlignerModel?
    nonisolated(unsafe) internal let forcedAlignmentRunner: ((ForcedAlignmentRequest) throws -> ForcedAlignmentResponse)?
    nonisolated(unsafe) internal let nativeDecoder: SpeechTokenizerDecoder?
    private let speechFamilyOrchestrator: any SpeechFamilyOrchestrating

    public init(descriptor: ModelDescriptor) {
        self.descriptor = descriptor
        self.mlxSpeechModel = nil
        self.mlxSpeechToTextModel = nil
        self.mlxForcedAlignerModel = nil
        self.forcedAlignmentRunner = nil
        self.nativeDecoder = nil
        self.speechFamilyOrchestrator = descriptor.familyID == .qwen3TTS
            ? QwenSpeechOrchestrator()
            : DefaultSpeechFamilyOrchestrator()
    }

    internal init(
        descriptor: ModelDescriptor,
        mlxModel: any SpeechGenerationModel,
        nativeDecoder: SpeechTokenizerDecoder? = nil
    ) {
        self.descriptor = descriptor
        self.mlxSpeechModel = mlxModel
        self.mlxSpeechToTextModel = nil
        self.mlxForcedAlignerModel = nil
        self.forcedAlignmentRunner = nil
        self.nativeDecoder = nativeDecoder
        self.speechFamilyOrchestrator = descriptor.familyID == .qwen3TTS
            ? QwenSpeechOrchestrator()
            : DefaultSpeechFamilyOrchestrator()
    }

    internal init(
        descriptor: ModelDescriptor,
        mlxSTTModel: any STTGenerationModel
    ) {
        self.descriptor = descriptor
        self.mlxSpeechModel = nil
        self.mlxSpeechToTextModel = mlxSTTModel
        self.mlxForcedAlignerModel = nil
        self.forcedAlignmentRunner = nil
        self.nativeDecoder = nil
        self.speechFamilyOrchestrator = descriptor.familyID == .qwen3TTS
            ? QwenSpeechOrchestrator()
            : DefaultSpeechFamilyOrchestrator()
    }

    internal init(
        descriptor: ModelDescriptor,
        mlxAlignerModel: Qwen3ForcedAlignerModel
    ) {
        self.descriptor = descriptor
        self.mlxSpeechModel = nil
        self.mlxSpeechToTextModel = nil
        self.mlxForcedAlignerModel = mlxAlignerModel
        self.forcedAlignmentRunner = { request in
            try Self.forcedAlignmentResponse(from: mlxAlignerModel, request: request)
        }
        self.nativeDecoder = nil
        self.speechFamilyOrchestrator = descriptor.familyID == .qwen3TTS
            ? QwenSpeechOrchestrator()
            : DefaultSpeechFamilyOrchestrator()
    }

    internal init(
        descriptor: ModelDescriptor,
        forcedAlignmentRunner: @escaping (ForcedAlignmentRequest) throws -> ForcedAlignmentResponse
    ) {
        self.descriptor = descriptor
        self.mlxSpeechModel = nil
        self.mlxSpeechToTextModel = nil
        self.mlxForcedAlignerModel = nil
        self.forcedAlignmentRunner = forcedAlignmentRunner
        self.nativeDecoder = nil
        self.speechFamilyOrchestrator = descriptor.familyID == .qwen3TTS
            ? QwenSpeechOrchestrator()
            : DefaultSpeechFamilyOrchestrator()
    }

    public func synthesizeLongForm(
        request: SpeechSynthesisRequest,
        in session: ModelRuntimeSession,
        onProgress: ((Int, Int) -> Void)? = nil
    ) async throws -> AudioChunk {
        try await synthesizePreparedLongForm(
            request: request,
            in: session,
            preparedExecution: nil,
            onProgress: onProgress
        )
    }

    internal func synthesizePreparedLongForm(
        request: SpeechSynthesisRequest,
        in session: ModelRuntimeSession,
        preparedExecution: PreparedQwenLongFormExecution?,
        onProgress: ((Int, Int) -> Void)? = nil
    ) async throws -> AudioChunk {
        _ = session
        guard let mlxModel = mlxSpeechModel else {
            throw MLXBackendError.modelNotFound(descriptor.id)
        }

        try request.voice?.validateCompatibility(with: descriptor.id, familyID: descriptor.familyID)

        if let chunk = try await synthesizeQwenLongFormIfNeeded(
            request: request,
            mlxModel: mlxModel,
            preparedExecution: preparedExecution,
            onProgress: onProgress
        ) {
            return chunk
        }

        return try await synthesize(request: request, in: session)
    }

    public func synthesizeLongFormStream(
        request: SpeechSynthesisRequest,
        onProgress: ((Int, Int) -> Void)? = nil
    ) async throws -> AsyncThrowingStream<AudioChunk, Error> {
        try await synthesizePreparedLongFormStream(
            request: request,
            preparedExecution: nil,
            onProgress: onProgress
        )
    }

    internal func synthesizePreparedLongFormStream(
        request: SpeechSynthesisRequest,
        preparedExecution: PreparedQwenLongFormExecution?,
        onProgress: ((Int, Int) -> Void)? = nil
    ) async throws -> AsyncThrowingStream<AudioChunk, Error> {
        guard let mlxModel = mlxSpeechModel else {
            throw MLXBackendError.modelNotFound(descriptor.id)
        }

        try request.voice?.validateCompatibility(with: descriptor.id, familyID: descriptor.familyID)

        if let stream = try await synthesizeQwenLongFormStreamIfNeeded(
            request: request,
            mlxModel: mlxModel,
            preparedExecution: preparedExecution,
            onProgress: onProgress
        ) {
            return stream
        }

        return try await synthesizeStream(request: request)
    }

    public func synthesize(
        request: SpeechSynthesisRequest,
        in session: ModelRuntimeSession
    ) async throws -> AudioChunk {
        let preparedExecution = try await speechFamilyOrchestrator.prepareExecution(
            handle: self,
            request: request
        )
        return try await speechFamilyOrchestrator.synthesize(
            handle: self,
            request: request,
            preparedExecution: preparedExecution,
            session: session
        )
    }

    public func synthesizeStream(
        request: SpeechSynthesisRequest
    ) async throws -> AsyncThrowingStream<AudioChunk, Error> {
        let preparedExecution = try await speechFamilyOrchestrator.prepareExecution(
            handle: self,
            request: request
        )
        return try await speechFamilyOrchestrator.synthesizeStream(
            handle: self,
            request: request,
            preparedExecution: preparedExecution
        )
    }

    internal func synthesizeOneShot(
        request: SpeechSynthesisRequest,
        in session: ModelRuntimeSession,
        behaviorOverride: SpeechSynthesisVoiceBehavior? = nil,
        automaticQwenCeiling: Int? = nil,
        executionMode: SynthesisExecutionMode
    ) async throws -> AudioChunk {
        _ = session
        guard let mlxModel = mlxSpeechModel else {
            throw MLXBackendError.modelNotFound(descriptor.id)
        }

        try request.voice?.validateCompatibility(with: descriptor.id, familyID: descriptor.familyID)

        let resolvedBehavior = behaviorOverride ?? Self.resolvedVoiceBehavior(for: request, descriptor: descriptor)
        let params = Self.resolvedGenerationParameters(
            from: request,
            defaults: mlxModel.defaultGenerationParameters,
            descriptor: descriptor,
            behavior: resolvedBehavior,
            text: request.text,
            ceilingOverride: automaticQwenCeiling
        )
        let startedAt = Date()
        Self.logSynthesisStart(
            descriptor: descriptor,
            request: request,
            behavior: resolvedBehavior,
            segmentCount: 1,
            maxTokens: params.maxTokens
        )
        let generationInput = Self.resolvedSpeechGenerationInput(
            from: request,
            descriptor: descriptor,
            voiceBehavior: resolvedBehavior
        )
        let audio = try await Self.withPreparedSpeechModel(mlxModel, request: request) {
            try await Self.generate(
                from: mlxModel,
                text: request.text,
                input: generationInput,
                language: request.language,
                generationParameters: params
            )
        }
        let chunk = Self.audioChunk(from: audio, sampleRate: mlxModel.sampleRate)
        Self.logSynthesisComplete(
            descriptor: descriptor,
            request: request,
            behavior: resolvedBehavior,
            segmentCount: 1,
            sampleCount: chunk.samples.count,
            sampleRate: chunk.sampleRate,
            startedAt: startedAt
        )
        return chunk
    }

    internal func synthesizeStreamOneShot(
        request: SpeechSynthesisRequest,
        behaviorOverride: SpeechSynthesisVoiceBehavior? = nil,
        automaticQwenCeiling: Int? = nil,
        executionMode: SynthesisExecutionMode
    ) async throws -> AsyncThrowingStream<AudioChunk, Error> {
        guard let mlxModel = mlxSpeechModel else {
            throw MLXBackendError.modelNotFound(descriptor.id)
        }

        try request.voice?.validateCompatibility(with: descriptor.id, familyID: descriptor.familyID)

        let resolvedBehavior = behaviorOverride ?? Self.resolvedVoiceBehavior(for: request, descriptor: descriptor)
        let params = Self.resolvedGenerationParameters(
            from: request,
            defaults: mlxModel.defaultGenerationParameters,
            descriptor: descriptor,
            behavior: resolvedBehavior,
            text: request.text,
            ceilingOverride: automaticQwenCeiling
        )
        let startedAt = Date()
        Self.logSynthesisStart(
            descriptor: descriptor,
            request: request,
            behavior: resolvedBehavior,
            segmentCount: 1,
            maxTokens: params.maxTokens
        )
        let restoreOverrides = Self.prepareSpeechModel(mlxModel, for: request)
        let stream = MLXStreamBridge.stream(
            from: mlxModel,
            descriptor: descriptor,
            request: request,
            nativeDecoder: nativeDecoder
        )

        guard let restoreOverrides else {
            return Self.loggingStream(
                stream,
                descriptor: descriptor,
                request: request,
                behavior: resolvedBehavior,
                segmentCount: 1,
                startedAt: startedAt
            )
        }

        let preparedStream = AsyncThrowingStream { continuation in
            let inheritedContexts = Self.inheritedObservationContexts()
            let task = Task { @Sendable in
                do {
                    try await Self.withInheritedObservationContexts(inheritedContexts) {
                        defer { restoreOverrides() }
                        do {
                            for try await chunk in stream {
                                continuation.yield(chunk)
                            }
                            continuation.finish()
                        } catch is CancellationError {
                            continuation.finish(throwing: CancellationError())
                        } catch {
                            continuation.finish(throwing: error)
                        }
                    }
                } catch is CancellationError {
                    continuation.finish(throwing: CancellationError())
                } catch {
                    continuation.finish(throwing: error)
                }
            }

            continuation.onTermination = { @Sendable _ in
                task.cancel()
            }
        }
        return Self.loggingStream(
            preparedStream,
            descriptor: descriptor,
            request: request,
            behavior: resolvedBehavior,
            segmentCount: 1,
            startedAt: startedAt
        )
    }

    internal func withQwenExecutionObservation<T: Sendable>(
        request: SpeechSynthesisRequest,
        behavior: SpeechSynthesisVoiceBehavior,
        executionMode: SynthesisExecutionMode,
        segmentCount: Int,
        operation: @escaping @Sendable () async throws -> T
    ) async throws -> T {
        SynthesisExecutionObserverContext.observer?(
            SynthesisExecutionEvent(
                kind: .started,
                executionMode: executionMode,
                segmentIndex: 0,
                segmentCount: max(1, segmentCount)
            )
        )

        guard descriptor.familyID == .qwen3TTS else {
            let result = try await operation()
            SynthesisExecutionObserverContext.observer?(
                SynthesisExecutionEvent(
                    kind: .completed,
                    executionMode: executionMode,
                    segmentIndex: max(1, segmentCount),
                    segmentCount: max(1, segmentCount)
                )
            )
            return result
        }

        let monitor = QwenHeartbeatMonitor()
        let observedOperation = {
            try await AudioGenerationObserverContext.$heartbeatHandler.withValue({ heartbeat in
                Task {
                    await monitor.ping()
                }
                let segmentContext = SynthesisSegmentObservationTaskContext.value
                SynthesisExecutionObserverContext.observer?(
                    SynthesisExecutionEvent(
                        kind: .heartbeat,
                        executionMode: segmentContext?.executionMode ?? executionMode,
                        segmentIndex: segmentContext?.segmentIndex ?? 0,
                        segmentCount: segmentContext?.segmentCount ?? max(1, segmentCount),
                        generatedTokenCount: heartbeat.generatedTokenCount,
                        usesAnchorConditioning: segmentContext?.usesAnchorConditioning,
                        chunkCharacterCount: segmentContext?.chunkCharacterCount,
                        maxTokenCount: segmentContext?.maxTokenCount,
                        segmentWallTimeSeconds: segmentContext.map {
                            Date().timeIntervalSince($0.startedAt)
                        }
                    )
                )
            }) {
                try await operation()
            }
        }

        let resultTask = Task {
            try await observedOperation()
        }
        let watchdogTask = Task {
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(2))
                if await monitor.isStalled(timeoutSeconds: 60) {
                    await monitor.markStalled()
                    resultTask.cancel()
                    break
                }
            }
        }

        do {
            let result = try await withTaskCancellationHandler {
                try await resultTask.value
            } onCancel: {
                resultTask.cancel()
                watchdogTask.cancel()
            }
            watchdogTask.cancel()
            SynthesisExecutionObserverContext.observer?(
                SynthesisExecutionEvent(
                    kind: .completed,
                    executionMode: executionMode,
                    segmentIndex: max(1, segmentCount),
                    segmentCount: max(1, segmentCount)
                )
            )
            return result
        } catch is CancellationError {
            watchdogTask.cancel()
            let didStall = await monitor.didStall()
            SynthesisExecutionObserverContext.observer?(
                SynthesisExecutionEvent(
                    kind: didStall ? .failed : .cancelled,
                    executionMode: executionMode,
                    segmentIndex: 0,
                    segmentCount: max(1, segmentCount)
                )
            )
            if didStall {
                throw MLXBackendError.inferenceError("Qwen synthesis stalled with no progress for 60 seconds.")
            }
            throw CancellationError()
        } catch {
            watchdogTask.cancel()
            SynthesisExecutionObserverContext.observer?(
                SynthesisExecutionEvent(
                    kind: .failed,
                    executionMode: executionMode,
                    segmentIndex: 0,
                    segmentCount: max(1, segmentCount)
                )
            )
            throw error
        }
    }

    internal func withQwenExecutionObservationStream(
        request: SpeechSynthesisRequest,
        behavior: SpeechSynthesisVoiceBehavior,
        executionMode: SynthesisExecutionMode,
        segmentCount: Int,
        operation: @escaping @Sendable () async throws -> AsyncThrowingStream<AudioChunk, Error>
    ) async throws -> AsyncThrowingStream<AudioChunk, Error> {
        let baseStream = try await withQwenExecutionObservation(
            request: request,
            behavior: behavior,
            executionMode: executionMode,
            segmentCount: segmentCount
        ) {
            try await operation()
        }
        return baseStream
    }

    public func transcribe(
        request: SpeechRecognitionRequest,
        in session: ModelRuntimeSession
    ) async throws -> RichTranscriptionResult {
        _ = session
        guard let mlxSpeechToTextModel else {
            throw MLXBackendError.modelNotFound(descriptor.id)
        }
        guard let audioChunk = request.audioChunk else {
            throw MLXBackendError.inferenceError(
                "Speech recognition requires inline audio samples in SpeechRecognitionRequest.audioChunk."
            )
        }
        guard !audioChunk.samples.isEmpty else {
            throw MLXBackendError.inferenceError("Speech recognition audio is empty.")
        }

        let baseParameters = mlxSpeechToTextModel.defaultGenerationParameters
        let generationParameters = STTGenerateParameters(
            maxTokens: baseParameters.maxTokens,
            temperature: baseParameters.temperature,
            topP: baseParameters.topP,
            topK: baseParameters.topK,
            verbose: baseParameters.verbose,
            language: Self.normalizedRecognitionLanguage(
                request.languageHint,
                fallback: baseParameters.language
            ),
            chunkDuration: baseParameters.chunkDuration,
            minChunkDuration: baseParameters.minChunkDuration
        )
        let output = mlxSpeechToTextModel.generate(
            audio: MLXArray(audioChunk.samples),
            generationParameters: generationParameters
        )
        return Self.richTranscriptionResult(from: output, modelId: descriptor.id)
    }

    public func align(
        request: ForcedAlignmentRequest,
        in session: ModelRuntimeSession
    ) async throws -> ForcedAlignmentResponse {
        _ = session
        guard let forcedAlignmentRunner else {
            throw MLXBackendError.modelNotFound(descriptor.id)
        }

        return try forcedAlignmentRunner(request)
    }

    internal static func generationParameters(
        from request: SpeechSynthesisRequest,
        defaults: GenerateParameters
    ) -> GenerateParameters {
        var params = defaults
        if let temperature = request.temperature { params.temperature = temperature }
        if let topP = request.topP { params.topP = topP }
        if let repetitionPenalty = request.repetitionPenalty { params.repetitionPenalty = repetitionPenalty }
        if let repetitionContextSize = request.repetitionContextSize { params.repetitionContextSize = repetitionContextSize }
        if let maxTokens = request.maxTokens { params.maxTokens = maxTokens }
        return params
    }

    internal static func resolvedGenerationParameters(
        from request: SpeechSynthesisRequest,
        defaults: GenerateParameters,
        descriptor: ModelDescriptor,
        behavior: SpeechSynthesisVoiceBehavior,
        text: String,
        ceilingOverride: Int? = nil
    ) -> GenerateParameters {
        var params = generationParameters(from: request, defaults: defaults)
        guard descriptor.familyID == .qwen3TTS, request.maxTokens == nil else {
            return params
        }

        let resolvedCeiling = ceilingOverride ?? qwenAutomaticMaxTokenCeiling(
            descriptor: descriptor,
            behavior: behavior,
            executionMode: QwenTextChunker.plan(text: text, behavior: behavior).isLongForm
                ? .segmentedContinuation
                : .oneShot
        )
        params.maxTokens = recommendedQwenMaxTokens(
            text: text,
            descriptor: descriptor,
            behavior: behavior,
            ceiling: min(resolvedCeiling, defaults.maxTokens ?? 8_192)
        )
        return params
    }

    internal static func recommendedQwenMaxTokens(
        text: String,
        descriptor: ModelDescriptor,
        behavior: SpeechSynthesisVoiceBehavior,
        ceiling: Int
    ) -> Int {
        let wordCount = estimatedWordCount(in: text)
        let estimatedSeconds = max(1.0, (Double(wordCount) / 165.0) * 60.0)
        let baseBudget = Int(ceil(estimatedSeconds * 12.5))

        let headroom: Double
        let baseFloor: Int
        let cappedCeiling: Int

        if descriptor.capabilities.contains(.voiceDesign), behavior != .stableNarrator {
            headroom = 1.25
            baseFloor = 48
            cappedCeiling = min(ceiling, 3_072)
        } else if behavior == .stableNarrator {
            headroom = 1.35
            baseFloor = 64
            cappedCeiling = ceiling
        } else {
            headroom = 1.25
            baseFloor = 64
            cappedCeiling = min(ceiling, 4_096)
        }

        let floor: Int
        if wordCount <= 8 {
            floor = 24
        } else if wordCount <= 24 {
            floor = min(baseFloor, 40)
        } else {
            floor = baseFloor
        }

        let recommended = Int(ceil(Double(baseBudget) * headroom))
        return max(floor, min(cappedCeiling, recommended))
    }

    internal static func qwenAutomaticMaxTokenCeiling(
        descriptor: ModelDescriptor,
        behavior: SpeechSynthesisVoiceBehavior,
        executionMode: SynthesisExecutionMode
    ) -> Int {
        switch behavior {
        case .stableNarrator:
            return executionMode == .segmentedContinuation ? 2_048 : 1_536
        case .auto, .expressive:
            if descriptor.capabilities.contains(.voiceDesign) {
                return 3_072
            }
            return 4_096
        }
    }

    internal static func plannedQwenExecution(
        for request: SpeechSynthesisRequest,
        descriptor: ModelDescriptor
    ) -> PlannedQwenExecution {
        let behavior = resolvedVoiceBehavior(for: request, descriptor: descriptor)
        let chunkPlan = QwenTextChunker.plan(text: request.text, behavior: behavior)
        let mode: SynthesisExecutionMode = chunkPlan.isLongForm ? .segmentedContinuation : .oneShot
        let automaticMaxTokenCeiling = qwenAutomaticMaxTokenCeiling(
            descriptor: descriptor,
            behavior: behavior,
            executionMode: mode
        )
        return PlannedQwenExecution(
            behavior: behavior,
            mode: mode,
            segmentCount: max(1, chunkPlan.chunks.count),
            automaticMaxTokenCeiling: automaticMaxTokenCeiling
        )
    }

    internal static func qwenShouldUseStableAnchorSegment(
        _ segmentIndex: Int,
        anchorEverySegments: Int = 4
    ) -> Bool {
        let resolvedAnchorEverySegments = max(1, anchorEverySegments)
        return segmentIndex == 0 || segmentIndex.isMultiple(of: resolvedAnchorEverySegments)
    }

    internal static func qwenStableNarratorAnchorEverySegments(
        segmentCount: Int
    ) -> Int {
        switch segmentCount {
        case ...8:
            return 4
        case 9 ... 15:
            return 6
        default:
            return 8
        }
    }

    internal static func qwenEmbeddingOnlyContinuationVoice(
        from voice: VoiceProfile
    ) -> VoiceProfile? {
        if voice.conditioningFormat == VoiceProfile.qwenSpeakerEmbeddingConditioningFormat,
           let speakerEmbedding = voice.speakerEmbedding,
           !speakerEmbedding.isEmpty {
            return VoiceProfile(
                id: voice.id,
                label: voice.label,
                backendVoiceID: voice.backendVoiceID,
                sourceModel: voice.sourceModel,
                localeIdentifier: voice.localeIdentifier,
                runtimeModel: voice.runtimeModel,
                referenceAudioAssetName: voice.referenceAudioAssetName,
                referenceTranscript: voice.referenceTranscript,
                speakerEmbedding: speakerEmbedding,
                conditioningFormat: VoiceProfile.qwenSpeakerEmbeddingConditioningFormat,
                conditioningAssets: voice.conditioningAssets,
                conditioningMetadata: voice.conditioningMetadata,
                voiceKind: .embeddingOnly
            )
        }

        guard voice.conditioningFormat == VoiceProfile.qwenClonePromptConditioningFormat,
              let payload = voice.speakerEmbedding,
              !payload.isEmpty,
              let decoded = try? JSONDecoder().decode(QwenClonePromptPayloadProbe.self, from: payload),
              let speakerEmbedding = decoded.refSpeakerEmbedding,
              !speakerEmbedding.isEmpty else {
            return nil
        }

        return VoiceProfile(
            id: voice.id,
            label: voice.label,
            backendVoiceID: voice.backendVoiceID,
            sourceModel: voice.sourceModel,
            localeIdentifier: voice.localeIdentifier,
            runtimeModel: voice.runtimeModel,
            referenceAudioAssetName: voice.referenceAudioAssetName,
            referenceTranscript: voice.referenceTranscript,
            speakerEmbedding: speakerEmbedding,
            conditioningFormat: VoiceProfile.qwenSpeakerEmbeddingConditioningFormat,
            conditioningAssets: voice.conditioningAssets,
            conditioningMetadata: voice.conditioningMetadata,
            voiceKind: .embeddingOnly
        )
    }

    private static func estimatedWordCount(in text: String) -> Int {
        let words = text.split(whereSeparator: \.isWhitespace).count
        if words > 0 {
            return words
        }

        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return 1 }
        return max(1, Int(ceil(Double(trimmed.count) / 5.0)))
    }

    internal static func resolvedVoiceSelector(
        from request: SpeechSynthesisRequest,
        descriptor: ModelDescriptor
    ) -> String? {
        if descriptor.capabilities.contains(.voiceDesign),
           let instruct = request.instruct?.trimmingCharacters(in: .whitespacesAndNewlines),
           !instruct.isEmpty {
            return instruct
        }

        return request.voice?.voiceSelector
    }

    internal static func referenceAudio(from request: SpeechSynthesisRequest) -> MLXArray? {
        if let samples = request.referenceAudioSamples, !samples.isEmpty {
            return MLXArray(samples).reshaped(1, -1)
        }
        if let pcmData = request.referenceAudioPCMFloat32LE {
            let decoded = Self.decodePCMFloat32LE(pcmData)
            guard !decoded.isEmpty else { return nil }
            return MLXArray(decoded).reshaped(1, -1)
        }
        return nil
    }

    internal static func prepareSpeechModel(
        _ model: any SpeechGenerationModel,
        for request: SpeechSynthesisRequest
    ) -> (@Sendable () -> Void)? {
        guard let chatterboxModel = model as? ChatterboxModel else {
            return nil
        }

        let previousCFGWeight = chatterboxModel.cfgWeightOverride
        let previousEmotionAdv = chatterboxModel.emotionAdvOverride
        chatterboxModel.cfgWeightOverride = request.cfgWeight
        chatterboxModel.emotionAdvOverride = request.exaggeration

        return {
            chatterboxModel.cfgWeightOverride = previousCFGWeight
            chatterboxModel.emotionAdvOverride = previousEmotionAdv
        }
    }

    internal static func withPreparedSpeechModel<T>(
        _ model: any SpeechGenerationModel,
        request: SpeechSynthesisRequest,
        operation: () async throws -> T
    ) async rethrows -> T {
        let restoreOverrides = prepareSpeechModel(model, for: request)
        defer { restoreOverrides?() }
        return try await operation()
    }

    internal struct ResolvedSpeechGenerationInput {
        let voice: String?
        let speaker: String?
        let instruct: String?
        let conditioning: SpeechConditioning?
        let referenceAudio: MLXArray?
        let referenceText: String?
    }

    internal static func resolvedVoiceBehavior(
        for request: SpeechSynthesisRequest,
        descriptor: ModelDescriptor
    ) -> SpeechSynthesisVoiceBehavior {
        if request.voiceBehavior != .auto {
            return request.voiceBehavior
        }

        guard descriptor.familyID == .qwen3TTS else {
            return .expressive
        }

        let hasReferenceAudio = request.referenceAudioSamples?.isEmpty == false
            || request.referenceAudioPCMFloat32LE?.isEmpty == false
            || request.referenceAudioAssetName?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false
        let hasReferenceTranscript = request.referenceTranscript?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .isEmpty == false

        if hasReferenceAudio && hasReferenceTranscript {
            return .stableNarrator
        }

        switch request.voice?.voiceKind {
        case .clonePrompt:
            return .stableNarrator
        case .embeddingOnly:
            if request.voice?.referenceAudioAssetName != nil,
               request.voice?.referenceTranscript?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false {
                return .stableNarrator
            }
            return .expressive
        case .preset, .namedSpeaker, .legacyPrompt, .tadaReference, .none:
            return .expressive
        }
    }

    internal static func speechConditioning(from c: VoiceConditioning) -> SpeechConditioning? {
        if c.format == VoiceConditioning.qwenClonePromptV1 {
            return SpeechConditioning(format: c.format, payload: c.payload ?? Data())
        }
        // Qwen path: payload is the raw embedding data.
        if let payload = c.payload {
            return SpeechConditioning(format: c.format, payload: payload)
        }
        // TADA path: build a JSON metadata payload and pack assetFiles into namedAssets.
        guard c.format == VoiceConditioning.tadaReferenceV1,
              !c.assetFiles.isEmpty,
              let meta = c.metadata,
              let tokenCount = meta.tokenCount,
              let acousticDim = meta.acousticDimension else {
            return nil
        }
        struct Manifest: Encodable {
            let token_count: Int
            let acoustic_dim: Int
            let frame_count: Int?
            let transcript: String
        }
        guard let payloadData = try? JSONEncoder().encode(Manifest(
            token_count: tokenCount,
            acoustic_dim: acousticDim,
            frame_count: meta.frameCount,
            transcript: meta.transcript ?? ""
        )) else {
            return nil
        }
        let namedAssets = Dictionary(
            uniqueKeysWithValues: c.assetFiles.map { ($0.filename, $0.data) }
        )
        return SpeechConditioning(format: c.format, payload: payloadData, namedAssets: namedAssets)
    }

    internal static func resolvedSpeechGenerationInput(
        from request: SpeechSynthesisRequest,
        descriptor: ModelDescriptor? = nil,
        voiceBehavior: SpeechSynthesisVoiceBehavior? = nil
    ) -> ResolvedSpeechGenerationInput {
        let isQwen = descriptor?.familyID == .qwen3TTS
        let presetVoiceID = request.voice?.backendVoiceID?.trimmingCharacters(in: .whitespacesAndNewlines)
        let hasPresetVoice = request.voice?.voiceKind == .preset && presetVoiceID?.isEmpty == false
        let resolvedBehavior = if let voiceBehavior {
            voiceBehavior
        } else if let descriptor {
            Self.resolvedVoiceBehavior(for: request, descriptor: descriptor)
        } else {
            request.voiceBehavior
        }
        let savedConditioning: SpeechConditioning? = if hasPresetVoice {
            nil
        } else {
            request.voice?.conditioning.flatMap { c in Self.speechConditioning(from: c) }
        }
        let allowsReferencePrompt = descriptor?.familyID == .qwen3TTS && resolvedBehavior == .stableNarrator
        let reusableQwenClonePrompt = isQwen
            && savedConditioning?.format == VoiceProfile.qwenClonePromptConditioningFormat
            && hasReusableQwenClonePromptPayload(savedConditioning?.payload)
        let savedReferenceAssetName = request.referenceAudioAssetName?
            .trimmingCharacters(in: .whitespacesAndNewlines)

        let refAudio: MLXArray? = if hasPresetVoice || (savedConditioning != nil && !allowsReferencePrompt) {
            nil
        } else if reusableQwenClonePrompt && savedReferenceAssetName?.isEmpty == false {
            nil
        } else if let samples = request.referenceAudioSamples {
            MLXArray(samples).reshaped(1, -1)
        } else if let pcmData = request.referenceAudioPCMFloat32LE {
            MLXArray(Self.decodePCMFloat32LE(pcmData)).reshaped(1, -1)
        } else {
            nil
        }

        let normalizedInstruct = request.instruct?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let qwenInstruct: String? = if isQwen, resolvedBehavior != .stableNarrator {
            if descriptor?.capabilities.contains(.voiceDesign) == true {
                if let normalizedInstruct, !normalizedInstruct.isEmpty {
                    normalizedInstruct
                } else if savedConditioning == nil, refAudio == nil {
                    request.voice?.voiceSelector
                } else {
                    nil
                }
            } else if let normalizedInstruct, !normalizedInstruct.isEmpty {
                normalizedInstruct
            } else {
                nil
            }
        } else {
            nil
        }

        let qwenSpeaker: String? = if isQwen,
                                      resolvedBehavior != .stableNarrator,
                                      descriptor?.capabilities.contains(.voiceDesign) == false,
                                      savedConditioning == nil,
                                      refAudio == nil {
            request.voice?.voiceSelector
        } else {
            nil
        }

        let voice: String? = if hasPresetVoice {
            request.voice?.voiceSelector
        } else if isQwen {
            nil
        } else if let descriptor, descriptor.familyID == .qwen3TTS,
                  resolvedBehavior == .stableNarrator {
            nil
        } else if let descriptor, descriptor.capabilities.contains(.voiceDesign),
                  let instruct = request.instruct?.trimmingCharacters(in: .whitespacesAndNewlines),
                  !instruct.isEmpty {
            instruct
        } else if savedConditioning != nil || refAudio != nil {
            nil
        } else {
            request.voice?.voiceSelector
        }

        let referenceText: String? = {
            let reusesReferenceText = if reusableQwenClonePrompt {
                shouldReplayReusableQwenReferenceText(
                    text: request.text,
                    behavior: resolvedBehavior
                )
            } else {
                refAudio != nil
            }

            guard reusesReferenceText else {
                return nil
            }
            let candidate = request.referenceTranscript ?? request.voice?.referenceTranscript
            guard let trimmed = candidate?.trimmingCharacters(in: .whitespacesAndNewlines),
                  !trimmed.isEmpty else {
                return nil
            }
            return trimmed
        }()
        return ResolvedSpeechGenerationInput(
            voice: voice,
            speaker: qwenSpeaker,
            instruct: qwenInstruct,
            conditioning: savedConditioning,
            referenceAudio: refAudio,
            referenceText: referenceText
        )
    }

    private static func shouldReplayReusableQwenReferenceText(
        text: String,
        behavior: SpeechSynthesisVoiceBehavior
    ) -> Bool {
        behavior == .stableNarrator
    }

    private func qwenChunkPlan(
        for request: SpeechSynthesisRequest
    ) -> QwenTextChunker.ChunkPlan {
        let behavior = Self.resolvedVoiceBehavior(for: request, descriptor: descriptor)
        return QwenTextChunker.plan(text: request.text, behavior: behavior)
    }

    static func shouldTrimQwenCache(
        afterCompletedSegment completedSegments: Int,
        totalSegments: Int,
        detectedMemoryGrowth: Bool,
        isUnderMemoryPressure: Bool
    ) -> Bool {
        guard totalSegments >= 12 else {
            return false
        }
        if isUnderMemoryPressure {
            return true
        }
        guard detectedMemoryGrowth else {
            return false
        }
        let physicalMemoryBytes = min(ProcessInfo.processInfo.physicalMemory, UInt64(Int.max))
        let trimHighWaterBytes = max(
            UInt64(16 * 1_024 * 1_024 * 1_024),
            physicalMemoryBytes / 2
        )
        guard UInt64(max(0, Int(Memory.peakMemory))) >= trimHighWaterBytes else {
            return false
        }
        let remainingSegments = totalSegments - completedSegments
        guard remainingSegments >= 4 else {
            return false
        }
        return completedSegments.isMultiple(of: 8)
    }

    private static func segmentGenerationParameters(
        from request: SpeechSynthesisRequest,
        defaults: GenerateParameters,
        descriptor: ModelDescriptor,
        behavior: SpeechSynthesisVoiceBehavior,
        text: String,
        ceilingOverride: Int? = nil
    ) -> PreparedSpeechGenerationParameters {
        let params = resolvedGenerationParameters(
            from: request,
            defaults: defaults,
            descriptor: descriptor,
            behavior: behavior,
            text: text,
            ceilingOverride: ceilingOverride
        )
        return PreparedSpeechGenerationParameters(
            temperature: params.temperature,
            topP: params.topP,
            repetitionPenalty: params.repetitionPenalty,
            repetitionContextSize: params.repetitionContextSize,
            maxTokens: params.maxTokens
        )
    }

    static func detectedQwenLongFormMemoryGrowth(
        previousPeakBytes: Int,
        currentPeakBytes: Int
    ) -> Bool {
        currentPeakBytes >= previousPeakBytes + 512 * 1_024 * 1_024
    }

    private func stableNarratorContinuationContext(
        for request: SpeechSynthesisRequest,
        mlxModel: any SpeechGenerationModel,
        segmentCount: Int
    ) throws -> StableNarratorContinuationContext? {
        guard descriptor.familyID == .qwen3TTS else {
            return nil
        }
        let behavior = Self.resolvedVoiceBehavior(for: request, descriptor: descriptor)
        guard behavior == .stableNarrator else {
            return nil
        }

        let anchorEverySegments = Self.qwenStableNarratorAnchorEverySegments(segmentCount: segmentCount)

        if let voice = request.voice {
            let continuationVoice = Self.qwenEmbeddingOnlyContinuationVoice(from: voice)
            if continuationVoice != nil || voice.conditioningFormat == VoiceProfile.qwenClonePromptConditioningFormat {
                return StableNarratorContinuationContext(
                    anchorInput: preparedStableNarratorInput(
                        from: request,
                        voice: voice
                    ),
                    continuationInput: continuationVoice.map {
                        preparedStableNarratorInput(
                            from: request,
                            voice: $0
                        )
                    },
                    anchorEverySegments: anchorEverySegments
                )
            }
        }

        guard let qwenModel = mlxModel as? Qwen3TTSModel,
              let referenceAudio = Self.referenceAudio(from: request),
              let referenceTranscript = request.referenceTranscript?
                .trimmingCharacters(in: .whitespacesAndNewlines),
              !referenceTranscript.isEmpty else {
            return nil
        }

        let conditioning = try qwenModel.createVoiceClonePromptConditioning(
            referenceAudio: referenceAudio,
            referenceTranscript: referenceTranscript
        )
        let anchorVoice = VoiceProfile(
            label: request.voice?.label ?? "Stable Narrator",
            sourceModel: descriptor.id,
            localeIdentifier: request.voice?.localeIdentifier,
            runtimeModel: descriptor.id,
            referenceAudioAssetName: request.referenceAudioAssetName,
            referenceTranscript: referenceTranscript,
            speakerEmbedding: conditioning.payload,
            conditioningFormat: conditioning.format,
            voiceKind: .clonePrompt
        )

        return StableNarratorContinuationContext(
            anchorInput: preparedStableNarratorInput(
                from: request,
                voice: anchorVoice
            ),
            continuationInput: Self.qwenEmbeddingOnlyContinuationVoice(from: anchorVoice).map {
                preparedStableNarratorInput(
                    from: request,
                    voice: $0
                )
            },
            anchorEverySegments: anchorEverySegments
        )
    }

    private func preparedStableNarratorInput(
        from request: SpeechSynthesisRequest,
        voice: VoiceProfile
    ) -> StableNarratorContinuationContext.PreparedInput {
        let resolvedInput = Self.resolvedSpeechGenerationInput(
            from: SpeechSynthesisRequest(
                model: request.model,
                text: request.text,
                voice: voice,
                language: request.language,
                referenceAudioAssetName: nil,
                referenceAudioPCMFloat32LE: nil,
                referenceAudioSamples: nil,
                referenceAudioSampleRate: nil,
                referenceTranscript: request.referenceTranscript ?? voice.referenceTranscript,
                instruct: request.instruct,
                exaggeration: request.exaggeration,
                cfgWeight: request.cfgWeight,
                sampleRate: request.sampleRate,
                responseFormat: request.responseFormat,
                temperature: request.temperature,
                topP: request.topP,
                repetitionPenalty: request.repetitionPenalty,
                repetitionContextSize: request.repetitionContextSize,
                maxTokens: request.maxTokens,
                voiceBehavior: .stableNarrator
            ),
            descriptor: descriptor,
            voiceBehavior: .stableNarrator
        )

        return StableNarratorContinuationContext.PreparedInput(
            voice: resolvedInput.voice,
            speaker: resolvedInput.speaker,
            instruct: resolvedInput.instruct,
            conditioning: resolvedInput.conditioning,
            referenceText: resolvedInput.referenceText
        )
    }

    internal func prepareQwenLongFormExecutionIfNeeded(
        request: SpeechSynthesisRequest,
        executionPlan: QwenSpeechOrchestrator.ExecutionPlan
    ) async throws -> PreparedQwenLongFormExecution? {
        guard descriptor.familyID == .qwen3TTS,
              executionPlan.mode == .segmentedContinuation,
              let mlxModel = mlxSpeechModel else {
            return nil
        }

        let continuationContext = try stableNarratorContinuationContext(
            for: request,
            mlxModel: mlxModel,
            segmentCount: executionPlan.chunkPlan.chunks.count
        )
        if let qwenModel = mlxModel as? Qwen3TTSModel {
            try Self.primeQwenReusableConditioningCacheIfNeeded(
                qwenModel: qwenModel,
                request: request,
                continuationContext: continuationContext
            )
        }

        let defaults = mlxModel.defaultGenerationParameters
        let segments = executionPlan.chunkPlan.chunks.enumerated().map { index, chunk in
            PreparedQwenLongFormSegment(
                text: chunk,
                characterCount: chunk.count,
                usesAnchorConditioning: continuationContext.map {
                    Self.qwenShouldUseStableAnchorSegment(
                        index,
                        anchorEverySegments: $0.anchorEverySegments
                    )
                } ?? false,
                preparedInput: continuationContext?.preparedInput(segmentIndex: index),
                generationParameters: Self.segmentGenerationParameters(
                    from: request,
                    defaults: defaults,
                    descriptor: descriptor,
                    behavior: executionPlan.behavior,
                    text: chunk,
                    ceilingOverride: executionPlan.automaticMaxTokenCeiling
                )
            )
        }

        return PreparedQwenLongFormExecution(
            behavior: executionPlan.behavior,
            chunkPlan: executionPlan.chunkPlan,
            continuationContext: continuationContext,
            segments: segments
        )
    }

    private static func primeQwenReusableConditioningCacheIfNeeded(
        qwenModel: Qwen3TTSModel,
        request: SpeechSynthesisRequest,
        continuationContext: StableNarratorContinuationContext?
    ) throws {
        if let continuationContext {
            try qwenModel.primeReusableConditioningCaches(for: continuationContext.anchorInput.conditioning)
            try qwenModel.primeReusableConditioningCaches(for: continuationContext.continuationInput?.conditioning)
            return
        }

        try qwenModel.primeReusableConditioningCaches(
            for: request.voice?.conditioning.flatMap { speechConditioning(from: $0) }
        )
    }

    private static func emitQwenLongFormSegmentEvent(
        kind: SynthesisExecutionEventKind,
        segment: PreparedQwenLongFormSegment,
        segmentIndex: Int,
        segmentCount: Int,
        prefillTokenCount: Int? = nil,
        segmentPrefillTimeSeconds: Double? = nil,
        segmentDecodeTimeSeconds: Double? = nil,
        samplingTimeSeconds: Double? = nil,
        evalTimeSeconds: Double? = nil,
        tokenMaterializationTimeSeconds: Double? = nil,
        embeddingAssemblyTimeSeconds: Double? = nil,
        talkerForwardTimeSeconds: Double? = nil,
        codePredictorTimeSeconds: Double? = nil,
        segmentWallTimeSeconds: Double? = nil,
        segmentAudioDurationSeconds: Double? = nil,
        continuationOutlier: Bool? = nil
    ) {
        SynthesisExecutionObserverContext.observer?(
            SynthesisExecutionEvent(
                kind: kind,
                executionMode: .segmentedContinuation,
                segmentIndex: segmentIndex,
                segmentCount: segmentCount,
                usesAnchorConditioning: segment.usesAnchorConditioning,
                chunkCharacterCount: segment.characterCount,
                maxTokenCount: segment.generationParameters.maxTokens,
                prefillTokenCount: prefillTokenCount,
                segmentPrefillTimeSeconds: segmentPrefillTimeSeconds,
                segmentDecodeTimeSeconds: segmentDecodeTimeSeconds,
                anchorSegmentDecodeTimeSeconds: segment.usesAnchorConditioning ? segmentDecodeTimeSeconds : nil,
                continuationSegmentDecodeTimeSeconds: segment.usesAnchorConditioning ? nil : segmentDecodeTimeSeconds,
                samplingTimeSeconds: samplingTimeSeconds,
                evalTimeSeconds: evalTimeSeconds,
                tokenMaterializationTimeSeconds: tokenMaterializationTimeSeconds,
                embeddingAssemblyTimeSeconds: embeddingAssemblyTimeSeconds,
                talkerForwardTimeSeconds: talkerForwardTimeSeconds,
                codePredictorTimeSeconds: codePredictorTimeSeconds,
                segmentWallTimeSeconds: segmentWallTimeSeconds,
                segmentAudioDurationSeconds: segmentAudioDurationSeconds,
                continuationOutlier: continuationOutlier
            )
        )
    }

    private static func qwenContinuationOutlier(
        segment: PreparedQwenLongFormSegment,
        info: AudioGenerationInfo?,
        segmentWallTimeSeconds: Double,
        segmentAudioDurationSeconds: Double
    ) -> Bool {
        guard segment.usesAnchorConditioning == false,
              let maxTokens = segment.generationParameters.maxTokens,
              maxTokens > 0,
              let info,
              info.generationTokenCount > 0,
              segmentAudioDurationSeconds > 0 else {
            return false
        }
        let tokenCeilingUse = Double(info.generationTokenCount) / Double(maxTokens)
        let segmentRTF = segmentWallTimeSeconds / segmentAudioDurationSeconds
        return tokenCeilingUse >= 0.85 && segmentRTF >= 2.0
    }

    private func synthesizeQwenLongFormIfNeeded(
        request: SpeechSynthesisRequest,
        mlxModel: any SpeechGenerationModel,
        preparedExecution: PreparedQwenLongFormExecution?,
        onProgress: ((Int, Int) -> Void)? = nil
    ) async throws -> AudioChunk? {
        guard descriptor.familyID == .qwen3TTS else {
            return nil
        }

        let resolvedPreparedExecution: PreparedQwenLongFormExecution?
        if let preparedExecution {
            resolvedPreparedExecution = preparedExecution
        } else {
            resolvedPreparedExecution = try await prepareQwenLongFormExecutionIfNeeded(
                request: request,
                executionPlan: QwenSpeechOrchestrator.plan(for: self, request: request)
            )
        }
        let plan = resolvedPreparedExecution?.chunkPlan ?? qwenChunkPlan(for: request)
        guard plan.isLongForm else {
            return nil
        }

        let behavior = resolvedPreparedExecution?.behavior
            ?? Self.resolvedVoiceBehavior(for: request, descriptor: descriptor)
        let continuationContext: StableNarratorContinuationContext?
        if let preparedContinuationContext = resolvedPreparedExecution?.continuationContext {
            continuationContext = preparedContinuationContext
        } else {
            continuationContext = try stableNarratorContinuationContext(
                for: request,
                mlxModel: mlxModel,
                segmentCount: plan.chunks.count
            )
        }
        let defaultLongFormInput = continuationContext == nil
            ? Self.resolvedSpeechGenerationInput(
                from: request,
                descriptor: descriptor,
                voiceBehavior: behavior
            )
            : nil
        let segments = resolvedPreparedExecution?.segments ?? plan.chunks.enumerated().map { index, chunk in
            PreparedQwenLongFormSegment(
                text: chunk,
                characterCount: chunk.count,
                usesAnchorConditioning: continuationContext.map {
                    Self.qwenShouldUseStableAnchorSegment(
                        index,
                        anchorEverySegments: $0.anchorEverySegments
                    )
                } ?? false,
                preparedInput: continuationContext?.preparedInput(segmentIndex: index),
                generationParameters: Self.segmentGenerationParameters(
                    from: request,
                    defaults: mlxModel.defaultGenerationParameters,
                    descriptor: descriptor,
                    behavior: behavior,
                    text: chunk,
                    ceilingOverride: Self.qwenAutomaticMaxTokenCeiling(
                        descriptor: descriptor,
                        behavior: behavior,
                        executionMode: .segmentedContinuation
                    )
                )
            )
        }
        let qwenPreparedPromptPhases: Qwen3TTSModel.PreparedStableNarratorPromptPhases?
        if let qwenModel = mlxModel as? Qwen3TTSModel,
           let continuationContext {
            qwenPreparedPromptPhases = try qwenModel.prepareStableNarratorPromptPhases(
                anchorConditioning: continuationContext.anchorInput.conditioning,
                continuationConditioning: continuationContext.continuationInput?.conditioning,
                continuationInstruct: continuationContext.continuationInput?.instruct,
                language: request.language ?? "auto"
            )
        } else {
            qwenPreparedPromptPhases = nil
        }
        var allSamples: [Float] = []
        var peakMemoryBytes = Int(Memory.peakMemory)

        try await Self.withPreparedSpeechModel(mlxModel, request: request) {
            for (index, segment) in segments.enumerated() {
                try Task.checkCancellation()
                let params = segment.generationParameters.resolved(defaults: mlxModel.defaultGenerationParameters)
                let generationInput = segment.preparedInput?.resolved
                    ?? continuationContext?.generationInput(segmentIndex: index)
                    ?? defaultLongFormInput
                    ?? Self.resolvedSpeechGenerationInput(from: request, descriptor: descriptor, voiceBehavior: behavior)
                Self.logger.info(
                    "Qwen long-form chunk \(index + 1)/\(plan.chunks.count) model='\(self.descriptor.id.rawValue, privacy: .public)' behavior='\(behavior.rawValue, privacy: .public)' anchor=\(segment.usesAnchorConditioning) chars=\(segment.characterCount) maxTokens=\(params.maxTokens ?? 0)"
                )
                Self.emitQwenLongFormSegmentEvent(
                    kind: .segmentStarted,
                    segment: segment,
                    segmentIndex: index + 1,
                    segmentCount: plan.chunks.count
                )
                let segmentStartedAt = Date()
                let observationContext = SynthesisSegmentObservationContext(
                    executionMode: .segmentedContinuation,
                    segmentIndex: index + 1,
                    segmentCount: plan.chunks.count,
                    usesAnchorConditioning: segment.usesAnchorConditioning,
                    chunkCharacterCount: segment.characterCount,
                    maxTokenCount: params.maxTokens,
                    startedAt: segmentStartedAt
                )
                let preparedStablePromptPhase: Qwen3TTSModel.PreparedStableNarratorPromptPhase? = if segment.usesAnchorConditioning {
                    qwenPreparedPromptPhases?.anchor
                } else {
                    qwenPreparedPromptPhases?.continuation
                }
                var segmentInfo: AudioGenerationInfo?
                let audio = try await SynthesisSegmentObservationTaskContext.$value.withValue(observationContext) {
                    try await Self.generate(
                        from: mlxModel,
                        text: segment.text,
                        input: generationInput,
                        language: request.language,
                        generationParameters: params,
                        preparedStablePromptPhase: preparedStablePromptPhase,
                        onInfo: { info in
                            segmentInfo = info
                        }
                    )
                }
                let samples = audio.squeezed().asArray(Float.self)
                allSamples.append(contentsOf: samples)
                let segmentWallTimeSeconds = Date().timeIntervalSince(segmentStartedAt)
                let segmentAudioDurationSeconds = Double(samples.count) / Double(mlxModel.sampleRate)
                let continuationOutlier = Self.qwenContinuationOutlier(
                    segment: segment,
                    info: segmentInfo,
                    segmentWallTimeSeconds: segmentWallTimeSeconds,
                    segmentAudioDurationSeconds: segmentAudioDurationSeconds
                )
                Self.emitQwenLongFormSegmentEvent(
                    kind: .segmentCompleted,
                    segment: segment,
                    segmentIndex: index + 1,
                    segmentCount: plan.chunks.count,
                    prefillTokenCount: segmentInfo?.promptTokenCount,
                    segmentPrefillTimeSeconds: segmentInfo?.prefillTime,
                    segmentDecodeTimeSeconds: segmentInfo?.generateTime,
                    samplingTimeSeconds: segmentInfo?.samplingTime,
                    evalTimeSeconds: segmentInfo?.evalTime,
                    tokenMaterializationTimeSeconds: segmentInfo?.tokenMaterializationTime,
                    embeddingAssemblyTimeSeconds: segmentInfo?.embeddingAssemblyTime,
                    talkerForwardTimeSeconds: segmentInfo?.talkerForwardTime,
                    codePredictorTimeSeconds: segmentInfo?.codePredictorTime,
                    segmentWallTimeSeconds: segmentWallTimeSeconds,
                    segmentAudioDurationSeconds: segmentAudioDurationSeconds,
                    continuationOutlier: continuationOutlier
                )
                onProgress?(index + 1, plan.chunks.count)
                let currentPeakBytes = Int(Memory.peakMemory)
                if Self.shouldTrimQwenCache(
                    afterCompletedSegment: index + 1,
                    totalSegments: plan.chunks.count,
                    detectedMemoryGrowth: Self.detectedQwenLongFormMemoryGrowth(
                        previousPeakBytes: peakMemoryBytes,
                        currentPeakBytes: currentPeakBytes
                    ),
                    isUnderMemoryPressure: false
                ) {
                    Memory.clearCache()
                }
                peakMemoryBytes = max(peakMemoryBytes, currentPeakBytes)
            }
        }

        onProgress?(plan.chunks.count, plan.chunks.count)
        return Self.audioChunk(from: allSamples, sampleRate: mlxModel.sampleRate)
    }

    private func synthesizeQwenLongFormStreamIfNeeded(
        request: SpeechSynthesisRequest,
        mlxModel: any SpeechGenerationModel,
        preparedExecution: PreparedQwenLongFormExecution?,
        onProgress: ((Int, Int) -> Void)? = nil
    ) async throws -> AsyncThrowingStream<AudioChunk, Error>? {
        guard descriptor.familyID == .qwen3TTS else {
            return nil
        }

        let resolvedPreparedExecution: PreparedQwenLongFormExecution?
        if let preparedExecution {
            resolvedPreparedExecution = preparedExecution
        } else {
            resolvedPreparedExecution = try await prepareQwenLongFormExecutionIfNeeded(
                request: request,
                executionPlan: QwenSpeechOrchestrator.plan(for: self, request: request)
            )
        }
        let plan = resolvedPreparedExecution?.chunkPlan ?? qwenChunkPlan(for: request)
        guard plan.isLongForm else {
            return nil
        }

        let behavior = resolvedPreparedExecution?.behavior
            ?? Self.resolvedVoiceBehavior(for: request, descriptor: descriptor)
        let continuationContext: StableNarratorContinuationContext?
        if let preparedContinuationContext = resolvedPreparedExecution?.continuationContext {
            continuationContext = preparedContinuationContext
        } else {
            continuationContext = try stableNarratorContinuationContext(
                for: request,
                mlxModel: mlxModel,
                segmentCount: plan.chunks.count
            )
        }
        let segments = resolvedPreparedExecution?.segments ?? plan.chunks.enumerated().map { index, chunk in
            PreparedQwenLongFormSegment(
                text: chunk,
                characterCount: chunk.count,
                usesAnchorConditioning: continuationContext.map {
                    Self.qwenShouldUseStableAnchorSegment(
                        index,
                        anchorEverySegments: $0.anchorEverySegments
                    )
                } ?? false,
                preparedInput: continuationContext?.preparedInput(segmentIndex: index),
                generationParameters: Self.segmentGenerationParameters(
                    from: request,
                    defaults: mlxModel.defaultGenerationParameters,
                    descriptor: descriptor,
                    behavior: behavior,
                    text: chunk,
                    ceilingOverride: Self.qwenAutomaticMaxTokenCeiling(
                        descriptor: descriptor,
                        behavior: behavior,
                        executionMode: .segmentedContinuation
                    )
                )
            )
        }
        let sampleRate = mlxModel.sampleRate
        let language = request.language
        let restoreOverrides = Self.prepareSpeechModel(mlxModel, for: request)
        let qwenPreparedPromptPhases: Qwen3TTSModel.PreparedStableNarratorPromptPhases?
        if let qwenModel = mlxModel as? Qwen3TTSModel,
           let continuationContext {
            qwenPreparedPromptPhases = try qwenModel.prepareStableNarratorPromptPhases(
                anchorConditioning: continuationContext.anchorInput.conditioning,
                continuationConditioning: continuationContext.continuationInput?.conditioning,
                continuationInstruct: continuationContext.continuationInput?.instruct,
                language: request.language ?? "auto"
            )
        } else {
            qwenPreparedPromptPhases = nil
        }

        struct LongFormContext: @unchecked Sendable {
            let model: any SpeechGenerationModel
            let onProgress: ((Int, Int) -> Void)?
        }
        let ctx = LongFormContext(model: mlxModel, onProgress: onProgress)

        return AsyncThrowingStream { continuation in
            let inheritedContexts = Self.inheritedObservationContexts()
            let task = Task { @Sendable in
                do {
                    try await Self.withInheritedObservationContexts(inheritedContexts) {
                        defer { restoreOverrides?() }
                        do {
                            let defaultLongFormInput = continuationContext == nil
                                ? Self.resolvedSpeechGenerationInput(
                                    from: request,
                                    descriptor: self.descriptor,
                                    voiceBehavior: behavior
                                )
                                : nil
                            var peakMemoryBytes = Int(Memory.peakMemory)
                            for (index, segment) in segments.enumerated() {
                                try Task.checkCancellation()
                                let params = segment.generationParameters.resolved(defaults: ctx.model.defaultGenerationParameters)
                                let generationInput = segment.preparedInput?.resolved
                                    ?? continuationContext?.generationInput(segmentIndex: index)
                                    ?? defaultLongFormInput
                                    ?? Self.resolvedSpeechGenerationInput(from: request, descriptor: self.descriptor, voiceBehavior: behavior)
                                Self.logger.info(
                                    "Qwen long-form stream chunk \(index + 1)/\(plan.chunks.count) model='\(self.descriptor.id.rawValue, privacy: .public)' behavior='\(behavior.rawValue, privacy: .public)' anchor=\(segment.usesAnchorConditioning) chars=\(segment.characterCount) maxTokens=\(params.maxTokens ?? 0)"
                                )
                                Self.emitQwenLongFormSegmentEvent(
                                    kind: .segmentStarted,
                                    segment: segment,
                                    segmentIndex: index + 1,
                                    segmentCount: plan.chunks.count
                                )
                                let segmentStartedAt = Date()
                                let observationContext = SynthesisSegmentObservationContext(
                                    executionMode: .segmentedContinuation,
                                    segmentIndex: index + 1,
                                    segmentCount: plan.chunks.count,
                                    usesAnchorConditioning: segment.usesAnchorConditioning,
                                    chunkCharacterCount: segment.characterCount,
                                    maxTokenCount: params.maxTokens,
                                    startedAt: segmentStartedAt
                                )
                                let preparedStablePromptPhase: Qwen3TTSModel.PreparedStableNarratorPromptPhase? = if segment.usesAnchorConditioning {
                                    qwenPreparedPromptPhases?.anchor
                                } else {
                                    qwenPreparedPromptPhases?.continuation
                                }
                                var segmentInfo: AudioGenerationInfo?
                                let audio = try await SynthesisSegmentObservationTaskContext.$value.withValue(observationContext) {
                                    try await Self.generate(
                                        from: ctx.model,
                                        text: segment.text,
                                        input: generationInput,
                                        language: language,
                                        generationParameters: params,
                                        preparedStablePromptPhase: preparedStablePromptPhase,
                                        onInfo: { info in
                                            segmentInfo = info
                                        }
                                    )
                                }
                                let samples = audio.squeezed().asArray(Float.self)
                                let segmentWallTimeSeconds = Date().timeIntervalSince(segmentStartedAt)
                                let segmentAudioDurationSeconds = Double(samples.count) / Double(sampleRate)
                                let continuationOutlier = Self.qwenContinuationOutlier(
                                    segment: segment,
                                    info: segmentInfo,
                                    segmentWallTimeSeconds: segmentWallTimeSeconds,
                                    segmentAudioDurationSeconds: segmentAudioDurationSeconds
                                )
                                Self.emitQwenLongFormSegmentEvent(
                                    kind: .segmentCompleted,
                                    segment: segment,
                                    segmentIndex: index + 1,
                                    segmentCount: plan.chunks.count,
                                    prefillTokenCount: segmentInfo?.promptTokenCount,
                                    segmentPrefillTimeSeconds: segmentInfo?.prefillTime,
                                    segmentDecodeTimeSeconds: segmentInfo?.generateTime,
                                    samplingTimeSeconds: segmentInfo?.samplingTime,
                                    evalTimeSeconds: segmentInfo?.evalTime,
                                    tokenMaterializationTimeSeconds: segmentInfo?.tokenMaterializationTime,
                                    embeddingAssemblyTimeSeconds: segmentInfo?.embeddingAssemblyTime,
                                    talkerForwardTimeSeconds: segmentInfo?.talkerForwardTime,
                                    codePredictorTimeSeconds: segmentInfo?.codePredictorTime,
                                    segmentWallTimeSeconds: segmentWallTimeSeconds,
                                    segmentAudioDurationSeconds: segmentAudioDurationSeconds,
                                    continuationOutlier: continuationOutlier
                                )
                                ctx.onProgress?(index + 1, plan.chunks.count)
                                let currentPeakBytes = Int(Memory.peakMemory)
                                if Self.shouldTrimQwenCache(
                                    afterCompletedSegment: index + 1,
                                    totalSegments: plan.chunks.count,
                                    detectedMemoryGrowth: Self.detectedQwenLongFormMemoryGrowth(
                                        previousPeakBytes: peakMemoryBytes,
                                        currentPeakBytes: currentPeakBytes
                                    ),
                                    isUnderMemoryPressure: false
                                ) {
                                    Memory.clearCache()
                                }
                                peakMemoryBytes = max(peakMemoryBytes, currentPeakBytes)
                                continuation.yield(Self.audioChunk(from: samples, sampleRate: sampleRate))
                            }
                            ctx.onProgress?(plan.chunks.count, plan.chunks.count)
                            continuation.finish()
                        } catch is CancellationError {
                            continuation.finish(throwing: CancellationError())
                        } catch {
                            continuation.finish(throwing: error)
                        }
                    }
                } catch is CancellationError {
                    continuation.finish(throwing: CancellationError())
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }

    internal static func generate(
        from model: any SpeechGenerationModel,
        text: String,
        input: ResolvedSpeechGenerationInput,
        language: String?,
        generationParameters: GenerateParameters,
        preparedStablePromptPhase: Qwen3TTSModel.PreparedStableNarratorPromptPhase? = nil,
        onInfo: ((AudioGenerationInfo) -> Void)? = nil
    ) async throws -> MLXArray {
        if let qwenModel = model as? Qwen3TTSModel {
            return try await qwenModel.generate(
                text: text,
                speaker: input.speaker,
                instruct: input.instruct,
                conditioning: input.conditioning,
                refAudio: input.referenceAudio,
                refText: input.referenceText,
                language: language,
                generationParameters: generationParameters,
                preparedStablePromptPhase: preparedStablePromptPhase,
                onInfo: onInfo
            )
        }

        if let conditioning = input.conditioning {
            guard let conditionedModel = model as? any ConditionedSpeechGenerationModel else {
                throw MLXBackendError.inferenceError(
                    "Saved voice conditioning '\(conditioning.format)' is not supported by this model."
                )
            }
            return try await conditionedModel.generate(
                text: text,
                voice: input.voice,
                conditioning: conditioning,
                refAudio: input.referenceAudio,
                refText: input.referenceText,
                language: language,
                generationParameters: generationParameters
            )
        }

        let audio = try await model.generate(
            text: text,
            voice: input.voice,
            refAudio: input.referenceAudio,
            refText: input.referenceText,
            language: language,
            generationParameters: generationParameters
        )
        if let onInfo {
            _ = onInfo
        }
        return audio
    }

    internal static func audioChunk(from audio: MLXArray, sampleRate: Int) -> AudioChunk {
        audioChunk(from: audio.squeezed().asArray(Float.self), sampleRate: sampleRate)
    }

    internal static func audioChunk(from samples: [Float], sampleRate: Int) -> AudioChunk {
        AudioChunk(samples: samples, sampleRate: Double(sampleRate))
    }

    internal static func pcmFloat32LEData(from samples: [Float]) -> Data {
        guard !samples.isEmpty else { return Data() }
        return samples.withUnsafeBufferPointer { buffer in
            guard let base = buffer.baseAddress else { return Data() }
            return Data(bytes: base, count: buffer.count * MemoryLayout<Float>.size)
        }
    }

    private static func hasReusableQwenClonePromptPayload(_ payload: Data?) -> Bool {
        guard let payload, !payload.isEmpty else { return false }

        struct QwenClonePromptPayloadProbe: Decodable {
            let refCode: Data?
            let numCodeGroups: Int?
            let frameCount: Int?
        }

        guard let decoded = try? JSONDecoder().decode(QwenClonePromptPayloadProbe.self, from: payload) else {
            return false
        }
        return decoded.refCode != nil && decoded.numCodeGroups != nil && decoded.frameCount != nil
    }

    private static func loggingStream(
        _ baseStream: AsyncThrowingStream<AudioChunk, Error>,
        descriptor: ModelDescriptor,
        request: SpeechSynthesisRequest,
        behavior: SpeechSynthesisVoiceBehavior,
        segmentCount: Int,
        startedAt: Date
    ) -> AsyncThrowingStream<AudioChunk, Error> {
        AsyncThrowingStream { continuation in
            let task = Task { @Sendable in
                var totalSamples = 0
                var outputSampleRate = 0.0
                do {
                    for try await chunk in baseStream {
                        totalSamples += chunk.samples.count
                        outputSampleRate = chunk.sampleRate
                        continuation.yield(chunk)
                    }

                    if outputSampleRate > 0 {
                        Self.logSynthesisComplete(
                            descriptor: descriptor,
                            request: request,
                            behavior: behavior,
                            segmentCount: segmentCount,
                            sampleCount: totalSamples,
                            sampleRate: outputSampleRate,
                            startedAt: startedAt
                        )
                    } else {
                        Self.logger.info(
                            "Synthesis stream completed model='\(descriptor.id.rawValue, privacy: .public)' behavior='\(behavior.rawValue, privacy: .public)' segments=\(segmentCount)"
                        )
                    }
                    continuation.finish()
                } catch is CancellationError {
                    Self.logger.info(
                        "Synthesis stream cancelled model='\(descriptor.id.rawValue, privacy: .public)' behavior='\(behavior.rawValue, privacy: .public)'"
                    )
                    continuation.finish(throwing: CancellationError())
                } catch {
                    Self.logger.error(
                        "Synthesis stream failed model='\(descriptor.id.rawValue, privacy: .public)' behavior='\(behavior.rawValue, privacy: .public)' error='\(error.localizedDescription, privacy: .public)'"
                    )
                    continuation.finish(throwing: error)
                }
            }

            continuation.onTermination = { @Sendable _ in
                task.cancel()
            }
        }
    }

    private static func logSynthesisStart(
        descriptor: ModelDescriptor,
        request: SpeechSynthesisRequest,
        behavior: SpeechSynthesisVoiceBehavior,
        segmentCount: Int,
        maxTokens: Int?
    ) {
        logger.info(
            "Synthesis started model='\(descriptor.id.rawValue, privacy: .public)' family='\(descriptor.familyID.rawValue, privacy: .public)' voiceKind='\(voiceKindLabel(request.voice?.voiceKind), privacy: .public)' behavior='\(behavior.rawValue, privacy: .public)' segmented=\(segmentCount > 1) segments=\(segmentCount) chars=\(request.text.count) maxTokens=\(maxTokens ?? 0)"
        )
    }

    private static func logSynthesisComplete(
        descriptor: ModelDescriptor,
        request: SpeechSynthesisRequest,
        behavior: SpeechSynthesisVoiceBehavior,
        segmentCount: Int,
        sampleCount: Int,
        sampleRate: Double,
        startedAt: Date
    ) {
        let wallSeconds = max(Date().timeIntervalSince(startedAt), 0)
        let audioDurationSeconds = sampleRate > 0 ? Double(sampleCount) / sampleRate : 0
        let rtf = audioDurationSeconds > 0 ? wallSeconds / audioDurationSeconds : 0
        logger.info(
            "Synthesis completed model='\(descriptor.id.rawValue, privacy: .public)' family='\(descriptor.familyID.rawValue, privacy: .public)' voiceKind='\(voiceKindLabel(request.voice?.voiceKind), privacy: .public)' behavior='\(behavior.rawValue, privacy: .public)' segmented=\(segmentCount > 1) segments=\(segmentCount) wall=\(wallSeconds, format: .fixed(precision: 2))s audio=\(audioDurationSeconds, format: .fixed(precision: 2))s rtf=\(rtf, format: .fixed(precision: 2))"
        )
    }

    private static func voiceKindLabel(_ voiceKind: VoiceKind?) -> String {
        voiceKind?.rawValue ?? "none"
    }

    internal static func decodePCMFloat32LE(_ data: Data) -> [Float] {
        let stride = MemoryLayout<Float>.size
        guard data.count.isMultiple(of: stride) else { return [] }
        let count = data.count / stride
        return data.withUnsafeBytes { raw in
            // Use load(fromByteOffset:as:) which does not require alignment —
            // safe for Data slices and arbitrary backing stores.
            (0..<count).map { i in raw.load(fromByteOffset: i * stride, as: Float.self) }
        }
    }

    private static func normalizedRecognitionLanguage(_ languageHint: String?, fallback: String) -> String {
        guard let languageHint = languageHint?.trimmingCharacters(in: .whitespacesAndNewlines),
              !languageHint.isEmpty
        else {
            return fallback
        }

        switch languageHint.lowercased() {
        case "en", "english":
            return "English"
        default:
            return languageHint
        }
    }

    private static func forcedAlignmentResponse(
        from model: Qwen3ForcedAlignerModel,
        request: ForcedAlignmentRequest
    ) throws -> ForcedAlignmentResponse {
        guard let audioChunk = request.audioChunk else {
            throw MLXBackendError.inferenceError(
                "Forced alignment requires inline audio samples in ForcedAlignmentRequest.audioChunk."
            )
        }
        guard !audioChunk.samples.isEmpty else {
            throw MLXBackendError.inferenceError("Forced alignment audio is empty.")
        }

        let result = model.generate(
            audio: MLXArray(audioChunk.samples),
            text: request.transcript,
            language: normalizedAlignmentLanguage(request.languageHint)
        )
        let segments = result.items.map { item in
            AlignmentToken(
                text: item.text,
                startTime: item.startTime,
                endTime: item.endTime
            )
        }
        return ForcedAlignmentResponse(transcript: request.transcript, segments: segments)
    }

    private static func normalizedAlignmentLanguage(_ languageHint: String?) -> String {
        guard let languageHint = languageHint?.trimmingCharacters(in: .whitespacesAndNewlines),
              !languageHint.isEmpty
        else {
            return "English"
        }

        switch languageHint.lowercased() {
        case "en", "english":
            return "English"
        case "zh", "zh-cn", "zh-hans", "chinese", "mandarin":
            return "Chinese"
        default:
            return languageHint
        }
    }

    internal static func richTranscriptionResult(from output: STTOutput, modelId: ModelIdentifier) -> RichTranscriptionResult {
        let segments: [TranscriptionSegment] = output.segments?.compactMap { segment in
            guard let text = segment["text"] as? String else { return nil }
            let startTime = segment["start"] as? Double
            let endTime = segment["end"] as? Double
            return TranscriptionSegment(text: text, startTime: startTime, endTime: endTime)
        } ?? []
        let inferenceTime = output.totalTime > 0 ? output.totalTime : nil
        return RichTranscriptionResult(
            text: output.text,
            language: output.language,
            durationSeconds: nil, // Audio duration populated by caller (ValarRuntime) from input sample count
            segments: segments,
            backendMetadata: BackendMetadata(
                modelId: modelId.rawValue,
                backendKind: .mlx,
                inferenceTimeSeconds: inferenceTime
            )
        )
    }
}

public typealias MLXASRModelHandle = MLXModelHandle
public typealias MLXAlignerModelHandle = MLXModelHandle
