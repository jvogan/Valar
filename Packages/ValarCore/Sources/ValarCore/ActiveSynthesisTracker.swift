import Foundation

public enum ActiveSynthesisTerminalState: String, Codable, Sendable, Equatable {
    case completed
    case cancelled
    case failed
    case stalled
}

public struct ActiveSynthesisRequestRecord: Sendable, Equatable {
    public let id: UUID
    public let modelID: String
    public let voiceBehavior: String
    public let executionMode: String
    public let segmentIndex: Int
    public let segmentCount: Int
    public let startedAt: Date
    public let lastHeartbeatAt: Date
    public let usesAnchorConditioning: Bool?
    public let chunkCharacterCount: Int?
    public let generatedTokenCount: Int?
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
        id: UUID,
        modelID: String,
        voiceBehavior: String,
        executionMode: String,
        segmentIndex: Int,
        segmentCount: Int,
        startedAt: Date,
        lastHeartbeatAt: Date,
        usesAnchorConditioning: Bool? = nil,
        chunkCharacterCount: Int? = nil,
        generatedTokenCount: Int? = nil,
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
        self.id = id
        self.modelID = modelID
        self.voiceBehavior = voiceBehavior
        self.executionMode = executionMode
        self.segmentIndex = segmentIndex
        self.segmentCount = segmentCount
        self.startedAt = startedAt
        self.lastHeartbeatAt = lastHeartbeatAt
        self.usesAnchorConditioning = usesAnchorConditioning
        self.chunkCharacterCount = chunkCharacterCount
        self.generatedTokenCount = generatedTokenCount
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

public struct ActiveSynthesisCompletionRecord: Sendable, Equatable {
    public let requestID: UUID
    public let modelID: String
    public let voiceBehavior: String
    public let executionMode: String
    public let terminalState: ActiveSynthesisTerminalState
    public let finishedAt: Date
    public let message: String?

    public init(
        requestID: UUID,
        modelID: String,
        voiceBehavior: String,
        executionMode: String,
        terminalState: ActiveSynthesisTerminalState,
        finishedAt: Date,
        message: String? = nil
    ) {
        self.requestID = requestID
        self.modelID = modelID
        self.voiceBehavior = voiceBehavior
        self.executionMode = executionMode
        self.terminalState = terminalState
        self.finishedAt = finishedAt
        self.message = message
    }
}

public struct ActiveSynthesisTrackerSnapshot: Sendable, Equatable {
    public let activeRequests: [ActiveSynthesisRequestRecord]
    public let lastCompletion: ActiveSynthesisCompletionRecord?

    public init(
        activeRequests: [ActiveSynthesisRequestRecord] = [],
        lastCompletion: ActiveSynthesisCompletionRecord? = nil
    ) {
        self.activeRequests = activeRequests
        self.lastCompletion = lastCompletion
    }
}

public actor ActiveSynthesisTracker {
    private struct Entry: Sendable {
        var requestID: UUID
        var modelID: String
        var voiceBehavior: String
        var executionMode: String
        var segmentIndex: Int
        var segmentCount: Int
        var startedAt: Date
        var lastHeartbeatAt: Date
        var usesAnchorConditioning: Bool?
        var chunkCharacterCount: Int?
        var generatedTokenCount: Int?
        var maxTokenCount: Int?
        var prefillTokenCount: Int?
        var segmentPrefillTimeSeconds: Double?
        var segmentDecodeTimeSeconds: Double?
        var anchorSegmentDecodeTimeSeconds: Double?
        var continuationSegmentDecodeTimeSeconds: Double?
        var samplingTimeSeconds: Double?
        var evalTimeSeconds: Double?
        var tokenMaterializationTimeSeconds: Double?
        var embeddingAssemblyTimeSeconds: Double?
        var talkerForwardTimeSeconds: Double?
        var codePredictorTimeSeconds: Double?
        var segmentWallTimeSeconds: Double?
        var segmentAudioDurationSeconds: Double?
        var continuationOutlier: Bool?
    }

    private var activeEntries: [UUID: Entry] = [:]
    private var lastCompletion: ActiveSynthesisCompletionRecord?

    public init() {}

    public func begin(
        requestID: UUID,
        modelID: String,
        voiceBehavior: String,
        executionMode: String,
        segmentCount: Int = 1,
        startedAt: Date = .now
    ) {
        activeEntries[requestID] = Entry(
            requestID: requestID,
            modelID: modelID,
            voiceBehavior: voiceBehavior,
            executionMode: executionMode,
            segmentIndex: 0,
            segmentCount: max(1, segmentCount),
            startedAt: startedAt,
            lastHeartbeatAt: startedAt,
            usesAnchorConditioning: nil,
            chunkCharacterCount: nil,
            generatedTokenCount: nil,
            maxTokenCount: nil,
            prefillTokenCount: nil,
            segmentPrefillTimeSeconds: nil,
            segmentDecodeTimeSeconds: nil,
            anchorSegmentDecodeTimeSeconds: nil,
            continuationSegmentDecodeTimeSeconds: nil,
            samplingTimeSeconds: nil,
            evalTimeSeconds: nil,
            tokenMaterializationTimeSeconds: nil,
            embeddingAssemblyTimeSeconds: nil,
            talkerForwardTimeSeconds: nil,
            codePredictorTimeSeconds: nil,
            segmentWallTimeSeconds: nil,
            segmentAudioDurationSeconds: nil,
            continuationOutlier: nil
        )
    }

    public func heartbeat(
        requestID: UUID,
        executionMode: String? = nil,
        segmentIndex: Int? = nil,
        segmentCount: Int? = nil,
        usesAnchorConditioning: Bool? = nil,
        chunkCharacterCount: Int? = nil,
        generatedTokenCount: Int? = nil,
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
        continuationOutlier: Bool? = nil,
        at timestamp: Date = .now
    ) {
        guard var entry = activeEntries[requestID] else {
            return
        }
        if let executionMode {
            entry.executionMode = executionMode
        }
        if let segmentIndex {
            entry.segmentIndex = max(0, segmentIndex)
        }
        if let segmentCount {
            entry.segmentCount = max(1, segmentCount)
        }
        if let usesAnchorConditioning {
            entry.usesAnchorConditioning = usesAnchorConditioning
        }
        if let chunkCharacterCount {
            entry.chunkCharacterCount = chunkCharacterCount
        }
        if let generatedTokenCount {
            entry.generatedTokenCount = max(0, generatedTokenCount)
        }
        if let maxTokenCount {
            entry.maxTokenCount = maxTokenCount
        }
        if let prefillTokenCount {
            let sanitized = max(0, prefillTokenCount)
            entry.prefillTokenCount = max(entry.prefillTokenCount ?? 0, sanitized)
        }
        if let segmentPrefillTimeSeconds {
            let sanitized = max(0, segmentPrefillTimeSeconds)
            entry.segmentPrefillTimeSeconds = max(entry.segmentPrefillTimeSeconds ?? 0, sanitized)
        }
        if let segmentDecodeTimeSeconds {
            let sanitized = max(0, segmentDecodeTimeSeconds)
            entry.segmentDecodeTimeSeconds = max(entry.segmentDecodeTimeSeconds ?? 0, sanitized)
        }
        if let anchorSegmentDecodeTimeSeconds {
            let sanitized = max(0, anchorSegmentDecodeTimeSeconds)
            entry.anchorSegmentDecodeTimeSeconds = max(entry.anchorSegmentDecodeTimeSeconds ?? 0, sanitized)
        }
        if let continuationSegmentDecodeTimeSeconds {
            let sanitized = max(0, continuationSegmentDecodeTimeSeconds)
            entry.continuationSegmentDecodeTimeSeconds = max(entry.continuationSegmentDecodeTimeSeconds ?? 0, sanitized)
        }
        if let samplingTimeSeconds {
            let sanitized = max(0, samplingTimeSeconds)
            entry.samplingTimeSeconds = max(entry.samplingTimeSeconds ?? 0, sanitized)
        }
        if let evalTimeSeconds {
            let sanitized = max(0, evalTimeSeconds)
            entry.evalTimeSeconds = max(entry.evalTimeSeconds ?? 0, sanitized)
        }
        if let tokenMaterializationTimeSeconds {
            let sanitized = max(0, tokenMaterializationTimeSeconds)
            entry.tokenMaterializationTimeSeconds = max(entry.tokenMaterializationTimeSeconds ?? 0, sanitized)
        }
        if let embeddingAssemblyTimeSeconds {
            let sanitized = max(0, embeddingAssemblyTimeSeconds)
            entry.embeddingAssemblyTimeSeconds = max(entry.embeddingAssemblyTimeSeconds ?? 0, sanitized)
        }
        if let talkerForwardTimeSeconds {
            let sanitized = max(0, talkerForwardTimeSeconds)
            entry.talkerForwardTimeSeconds = max(entry.talkerForwardTimeSeconds ?? 0, sanitized)
        }
        if let codePredictorTimeSeconds {
            let sanitized = max(0, codePredictorTimeSeconds)
            entry.codePredictorTimeSeconds = max(entry.codePredictorTimeSeconds ?? 0, sanitized)
        }
        if let segmentWallTimeSeconds {
            let sanitized = max(0, segmentWallTimeSeconds)
            entry.segmentWallTimeSeconds = max(entry.segmentWallTimeSeconds ?? 0, sanitized)
        }
        if let segmentAudioDurationSeconds {
            let sanitized = max(0, segmentAudioDurationSeconds)
            entry.segmentAudioDurationSeconds = max(entry.segmentAudioDurationSeconds ?? 0, sanitized)
        }
        if let continuationOutlier {
            entry.continuationOutlier = (entry.continuationOutlier ?? false) || continuationOutlier
        }
        entry.lastHeartbeatAt = timestamp
        activeEntries[requestID] = entry
    }

    public func finish(
        requestID: UUID,
        terminalState: ActiveSynthesisTerminalState,
        message: String? = nil,
        finishedAt: Date = .now
    ) {
        let entry = activeEntries.removeValue(forKey: requestID)
        guard let entry else {
            return
        }
        lastCompletion = ActiveSynthesisCompletionRecord(
            requestID: requestID,
            modelID: entry.modelID,
            voiceBehavior: entry.voiceBehavior,
            executionMode: entry.executionMode,
            terminalState: terminalState,
            finishedAt: finishedAt,
            message: message
        )
    }

    public func snapshot() -> ActiveSynthesisTrackerSnapshot {
        let activeRequests = activeEntries.values
            .map { entry in
                ActiveSynthesisRequestRecord(
                    id: entry.requestID,
                    modelID: entry.modelID,
                    voiceBehavior: entry.voiceBehavior,
                    executionMode: entry.executionMode,
                    segmentIndex: entry.segmentIndex,
                    segmentCount: entry.segmentCount,
                    startedAt: entry.startedAt,
                    lastHeartbeatAt: entry.lastHeartbeatAt,
                    usesAnchorConditioning: entry.usesAnchorConditioning,
                    chunkCharacterCount: entry.chunkCharacterCount,
                    generatedTokenCount: entry.generatedTokenCount,
                    maxTokenCount: entry.maxTokenCount,
                    prefillTokenCount: entry.prefillTokenCount,
                    segmentPrefillTimeSeconds: entry.segmentPrefillTimeSeconds,
                    segmentDecodeTimeSeconds: entry.segmentDecodeTimeSeconds,
                    anchorSegmentDecodeTimeSeconds: entry.anchorSegmentDecodeTimeSeconds,
                    continuationSegmentDecodeTimeSeconds: entry.continuationSegmentDecodeTimeSeconds,
                    samplingTimeSeconds: entry.samplingTimeSeconds,
                    evalTimeSeconds: entry.evalTimeSeconds,
                    tokenMaterializationTimeSeconds: entry.tokenMaterializationTimeSeconds,
                    embeddingAssemblyTimeSeconds: entry.embeddingAssemblyTimeSeconds,
                    talkerForwardTimeSeconds: entry.talkerForwardTimeSeconds,
                    codePredictorTimeSeconds: entry.codePredictorTimeSeconds,
                    segmentWallTimeSeconds: entry.segmentWallTimeSeconds,
                    segmentAudioDurationSeconds: entry.segmentAudioDurationSeconds,
                    continuationOutlier: entry.continuationOutlier
                )
            }
            .sorted { $0.startedAt < $1.startedAt }
        return ActiveSynthesisTrackerSnapshot(
            activeRequests: activeRequests,
            lastCompletion: lastCompletion
        )
    }
}
