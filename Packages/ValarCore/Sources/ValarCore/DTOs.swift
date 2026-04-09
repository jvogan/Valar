import Foundation
import ValarModelKit
import ValarPersistence

public struct ValarCommandEnvelope<Payload: Codable & Sendable>: Codable, Sendable {
    public let ok: Bool
    public let command: String
    public let data: Payload?
    public let error: ValarCommandErrorDTO?

    public init(
        ok: Bool,
        command: String,
        data: Payload?,
        error: ValarCommandErrorDTO?
    ) {
        self.ok = ok
        self.command = command
        self.data = data
        self.error = error
    }
}

public struct ValarCommandErrorDTO: Codable, Sendable, Equatable {
    public let code: Int
    public let kind: String
    public let message: String
    public let help: String?

    public init(
        code: Int,
        kind: String,
        message: String,
        help: String? = nil
    ) {
        self.code = code
        self.kind = kind
        self.message = message
        self.help = help
    }
}

public struct DaemonErrorEnvelopeDTO: Codable, Sendable, Equatable {
    public let ok: Bool
    public let error: ValarCommandErrorDTO

    public init(
        ok: Bool = false,
        error: ValarCommandErrorDTO
    ) {
        self.ok = ok
        self.error = error
    }
}

public struct ValarCommandSuccessPayloadDTO: Codable, Sendable, Equatable {
    public let message: String?
    public let help: String?

    public init(
        message: String? = nil,
        help: String? = nil
    ) {
        self.message = message
        self.help = help
    }
}

public struct ModelSummaryDTO: Codable, Sendable, Equatable {
    public let id: String
    public let displayName: String
    public let family: String
    public let provider: String
    public let installState: String
    public let installedPath: String?
    public let installPathStatus: String?
    public let isRecommended: Bool
    public let cachedOnDisk: Bool
    public let licenseName: String?
    public let licenseURL: String?
    public let supportTier: String
    public let releaseEligible: Bool
    public let qualityTierByLanguage: [String: String]
    public let distributionTier: String
    public let notes: String?

    public init(
        id: String,
        displayName: String,
        family: String,
        provider: String,
        installState: String,
        installedPath: String?,
        installPathStatus: String? = nil,
        isRecommended: Bool,
        cachedOnDisk: Bool = false,
        licenseName: String? = nil,
        licenseURL: String? = nil,
        supportTier: String = ModelSupportTier.supported.rawValue,
        releaseEligible: Bool = true,
        qualityTierByLanguage: [String: String] = [:],
        distributionTier: String = ModelDistributionTier.optionalInstall.rawValue,
        notes: String? = nil
    ) {
        self.id = id
        self.displayName = displayName
        self.family = family
        self.provider = provider
        self.installState = installState
        self.installedPath = installedPath
        self.installPathStatus = installPathStatus
        self.isRecommended = isRecommended
        self.cachedOnDisk = cachedOnDisk
        self.licenseName = licenseName
        self.licenseURL = licenseURL
        self.supportTier = supportTier
        self.releaseEligible = releaseEligible
        self.qualityTierByLanguage = qualityTierByLanguage
        self.distributionTier = distributionTier
        self.notes = notes
    }

    public init(from model: CatalogModel) {
        self.init(
            id: model.id.rawValue,
            displayName: model.descriptor.displayName,
            family: model.familyID.rawValue,
            provider: model.providerName,
            installState: model.installState.rawValue,
            installedPath: model.installedPath,
            installPathStatus: model.installPathStatus?.rawValue,
            isRecommended: model.isRecommended,
            cachedOnDisk: model.cachedOnDisk,
            licenseName: model.licenseName,
            licenseURL: model.licenseURL?.absoluteString,
            supportTier: model.supportTier.rawValue,
            releaseEligible: model.releaseEligible,
            qualityTierByLanguage: model.qualityTierByLanguage.mapValues(\.rawValue),
            distributionTier: model.distributionTier.rawValue,
            notes: model.notes
        )
    }
}

public struct ModelDetailPayloadDTO: Codable, Sendable, Equatable {
    public let message: String
    public let model: ModelSummaryDTO
    public let providerURL: String?
    public let manifestPath: String?
    public let artifactCount: Int
    public let supportedBackends: [String]
    public let capabilities: [String]
    public let voiceFeatures: [String]
    public let defaultSampleRate: Double?
    public let supportsReferenceAudio: Bool
    public let supportTier: String
    public let releaseEligible: Bool
    public let qualityTierByLanguage: [String: String]
    public let distributionTier: String

    public init(
        message: String,
        model: ModelSummaryDTO,
        providerURL: String? = nil,
        manifestPath: String? = nil,
        artifactCount: Int,
        supportedBackends: [String],
        capabilities: [String],
        voiceFeatures: [String] = [],
        defaultSampleRate: Double? = nil,
        supportsReferenceAudio: Bool = false,
        supportTier: String = ModelSupportTier.supported.rawValue,
        releaseEligible: Bool = true,
        qualityTierByLanguage: [String: String] = [:],
        distributionTier: String = ModelDistributionTier.optionalInstall.rawValue
    ) {
        self.message = message
        self.model = model
        self.providerURL = providerURL
        self.manifestPath = manifestPath
        self.artifactCount = artifactCount
        self.supportedBackends = supportedBackends
        self.capabilities = capabilities
        self.voiceFeatures = voiceFeatures
        self.defaultSampleRate = defaultSampleRate
        self.supportsReferenceAudio = supportsReferenceAudio
        self.supportTier = supportTier
        self.releaseEligible = releaseEligible
        self.qualityTierByLanguage = qualityTierByLanguage
        self.distributionTier = distributionTier
    }
}

public struct ModelListPayloadDTO: Codable, Sendable, Equatable {
    public let message: String
    public let models: [ModelSummaryDTO]

    public init(
        message: String,
        models: [ModelSummaryDTO]
    ) {
        self.message = message
        self.models = models
    }
}

public struct HealthStatusDTO: Codable, Sendable, Equatable {
    public let status: String
    public let version: String

    public init(
        status: String,
        version: String
    ) {
        self.status = status
        self.version = version
    }
}

public struct DaemonHealthStatusDTO: Codable, Sendable, Equatable {
    public let status: String

    public init(status: String) {
        self.status = status
    }
}

public struct DaemonReadyDTO: Codable, Sendable, Equatable {
    public let ready: Bool
    /// Human-readable explanation when `ready` is false.
    public let reason: String?
    /// IDs of models registered in the model pack store (available on disk, whether or not
    /// they are already resident in memory).
    public let installedModels: [String]
    /// IDs of installed model records whose on-disk packs are missing or stale.
    public let staleInstalledModels: [String]
    /// IDs of models found in the mlx-audio HuggingFace cache but not yet registered.
    public let cachedModels: [String]
    /// True if at least one TTS model is ready to serve immediately or can load on demand
    /// from an installed pack.
    public let ttsReady: Bool
    /// True if at least one ASR model is ready to serve immediately or can load on demand
    /// from an installed pack.
    public let asrReady: Bool
    /// True if at least one forced-alignment model is ready to serve immediately or can
    /// load on demand from an installed pack.
    public let alignmentReady: Bool
    /// Optional readiness mode that distinguishes already-resident service from lazy-load readiness.
    public let readinessMode: DaemonReadinessMode?
    /// True when at least one resident model can synthesize speech right now without a lazy load.
    public let residentTTSReady: Bool
    /// True when at least one resident model can transcribe speech right now without a lazy load.
    public let residentASRReady: Bool
    /// True when at least one resident model can align audio/text right now without a lazy load.
    public let residentAlignmentReady: Bool
    /// True when daemon-side inference assets such as the metallib bundle are present.
    public let inferenceAssetsReady: Bool?
    /// Human-readable readiness issue when daemon-side inference assets are missing.
    public let inferenceAssetIssue: String?

    public init(
        ready: Bool,
        reason: String? = nil,
        installedModels: [String] = [],
        staleInstalledModels: [String] = [],
        cachedModels: [String] = [],
        ttsReady: Bool = false,
        asrReady: Bool = false,
        alignmentReady: Bool = false,
        readinessMode: DaemonReadinessMode? = nil,
        residentTTSReady: Bool = false,
        residentASRReady: Bool = false,
        residentAlignmentReady: Bool = false,
        inferenceAssetsReady: Bool? = nil,
        inferenceAssetIssue: String? = nil
    ) {
        self.ready = ready
        self.reason = reason
        self.installedModels = installedModels
        self.staleInstalledModels = staleInstalledModels
        self.cachedModels = cachedModels
        self.ttsReady = ttsReady
        self.asrReady = asrReady
        self.alignmentReady = alignmentReady
        self.readinessMode = readinessMode
        self.residentTTSReady = residentTTSReady
        self.residentASRReady = residentASRReady
        self.residentAlignmentReady = residentAlignmentReady
        self.inferenceAssetsReady = inferenceAssetsReady
        self.inferenceAssetIssue = inferenceAssetIssue
    }
}

public struct ModelResidencyEvictionEventDTO: Codable, Sendable, Equatable {
    public let modelID: String
    public let displayName: String
    public let trigger: String
    public let reclaimedBytes: Int
    public let residentBytesAfterEviction: Int
    public let residentModelCountAfterEviction: Int
    public let occurredAt: String

    public init(
        modelID: String,
        displayName: String,
        trigger: String,
        reclaimedBytes: Int,
        residentBytesAfterEviction: Int,
        residentModelCountAfterEviction: Int,
        occurredAt: String
    ) {
        self.modelID = modelID
        self.displayName = displayName
        self.trigger = trigger
        self.reclaimedBytes = reclaimedBytes
        self.residentBytesAfterEviction = residentBytesAfterEviction
        self.residentModelCountAfterEviction = residentModelCountAfterEviction
        self.occurredAt = occurredAt
    }

    public init(from event: ModelResidencyEvictionEvent) {
        self.init(
            modelID: event.descriptor.id.rawValue,
            displayName: event.descriptor.displayName,
            trigger: event.trigger.rawValue,
            reclaimedBytes: event.reclaimedBytes,
            residentBytesAfterEviction: event.residentBytesAfterEviction,
            residentModelCountAfterEviction: event.residentModelCountAfterEviction,
            occurredAt: ValarDTOFormatting.iso8601String(from: event.occurredAt)
        )
    }
}

public struct DaemonRuntimeStatusDTO: Codable, Sendable, Equatable {
    public let processIdentifier: Int
    public let daemonPIDFilePath: String
    public let daemonPIDFilePresent: Bool
    public let daemonPIDFileMatchesProcess: Bool?
    public let residentModels: [ModelResidencySnapshotDTO]
    public let totalResidentBytes: Int
    public let memoryBudgetBytes: Int
    public let warmPolicy: String
    public let warmStartModelSource: WarmStartModelSourceDTO
    public let configuredWarmStartModels: [String]
    public let effectiveWarmStartModels: [String]
    public let orphanedModelPackPaths: [String]
    public let idleResidentExtraModels: [String]
    public let idleTrimEligibleModels: [String]
    /// True when idle resident models outside the effective warm set appear to be lingering beyond
    /// the recent-use grace window rather than simply settling after startup or on-demand use.
    public let idleResidentExtraModelsLikelyDrift: Bool?
    /// Human-readable explanation of whether idle resident models outside the effective warm set
    /// look transient or likely indicate sticky on-demand residency / warm-set drift.
    public let idleResidentExtraModelsAdvisory: String?
    public let lastIdleTrimResult: DaemonIdleTrimResultDTO?
    public let metalDeviceName: String?
    /// Whole-process physical footprint, including model memory and non-model runtime overhead.
    public let processFootprintBytes: Int
    public let processFootprintHighWaterBytes: Int?
    /// True when the current whole-process footprint exceeds the daemon's configured budget.
    public let memoryBudgetBreached: Bool
    /// Positive number of bytes the current footprint exceeds the configured daemon budget by.
    public let processFootprintOverBudgetBytes: Int
    /// Compatibility field retained for one release; this is an average CPU percentage since daemon start.
    public let processCPUPercent: Double?
    /// Best-effort point-in-time CPU percentage sampled between consecutive runtime calls.
    public let processCPUCurrentPercent: Double?
    public let processCPUCurrentHighWaterPercent: Double?
    /// Explicit name for the same cumulative average CPU figure exposed above.
    public let processCPUAveragePercentSinceStart: Double?
    public let processCPUAverageHighWaterPercentSinceStart: Double?
    public let availableDiskBytes: Int?
    public let availableDiskLowWaterBytes: Int?
    /// Compatibility field retained for one release; contains models still warming.
    public let prewarmedModels: [String]
    /// Models currently warming or preloading.
    public let warmingModels: [String]
    public let activeSynthesisCount: Int
    public let oldestActiveSynthesisAgeSeconds: Double?
    public let stalledSynthesisCount: Int
    public let activeSynthesisRequests: [ActiveSynthesisRequestDTO]
    public let lastSynthesisCompletionReason: String?
    public let recentResidentEvictions: [ModelResidencyEvictionEventDTO]
    public let uptimeSeconds: Double

    public init(
        processIdentifier: Int,
        daemonPIDFilePath: String,
        daemonPIDFilePresent: Bool,
        daemonPIDFileMatchesProcess: Bool? = nil,
        residentModels: [ModelResidencySnapshotDTO],
        totalResidentBytes: Int,
        memoryBudgetBytes: Int,
        warmPolicy: String,
        warmStartModelSource: WarmStartModelSourceDTO = .default,
        configuredWarmStartModels: [String] = [],
        effectiveWarmStartModels: [String] = [],
        orphanedModelPackPaths: [String] = [],
        idleResidentExtraModels: [String] = [],
        idleTrimEligibleModels: [String] = [],
        idleResidentExtraModelsLikelyDrift: Bool? = nil,
        idleResidentExtraModelsAdvisory: String? = nil,
        lastIdleTrimResult: DaemonIdleTrimResultDTO? = nil,
        metalDeviceName: String?,
        processFootprintBytes: Int,
        processFootprintHighWaterBytes: Int? = nil,
        memoryBudgetBreached: Bool = false,
        processFootprintOverBudgetBytes: Int = 0,
        processCPUPercent: Double? = nil,
        processCPUCurrentPercent: Double? = nil,
        processCPUCurrentHighWaterPercent: Double? = nil,
        processCPUAveragePercentSinceStart: Double? = nil,
        processCPUAverageHighWaterPercentSinceStart: Double? = nil,
        availableDiskBytes: Int? = nil,
        availableDiskLowWaterBytes: Int? = nil,
        prewarmedModels: [String],
        warmingModels: [String] = [],
        activeSynthesisCount: Int = 0,
        oldestActiveSynthesisAgeSeconds: Double? = nil,
        stalledSynthesisCount: Int = 0,
        activeSynthesisRequests: [ActiveSynthesisRequestDTO] = [],
        lastSynthesisCompletionReason: String? = nil,
        recentResidentEvictions: [ModelResidencyEvictionEventDTO] = [],
        uptimeSeconds: Double
    ) {
        self.processIdentifier = processIdentifier
        self.daemonPIDFilePath = daemonPIDFilePath
        self.daemonPIDFilePresent = daemonPIDFilePresent
        self.daemonPIDFileMatchesProcess = daemonPIDFileMatchesProcess
        self.residentModels = residentModels
        self.totalResidentBytes = totalResidentBytes
        self.memoryBudgetBytes = memoryBudgetBytes
        self.warmPolicy = warmPolicy
        self.warmStartModelSource = warmStartModelSource
        self.configuredWarmStartModels = configuredWarmStartModels
        self.effectiveWarmStartModels = effectiveWarmStartModels
        self.orphanedModelPackPaths = orphanedModelPackPaths
        self.idleResidentExtraModels = idleResidentExtraModels
        self.idleTrimEligibleModels = idleTrimEligibleModels
        self.idleResidentExtraModelsLikelyDrift = idleResidentExtraModelsLikelyDrift
        self.idleResidentExtraModelsAdvisory = idleResidentExtraModelsAdvisory
        self.lastIdleTrimResult = lastIdleTrimResult
        self.metalDeviceName = metalDeviceName
        self.processFootprintBytes = processFootprintBytes
        self.processFootprintHighWaterBytes = processFootprintHighWaterBytes
        self.memoryBudgetBreached = memoryBudgetBreached
        self.processFootprintOverBudgetBytes = processFootprintOverBudgetBytes
        self.processCPUPercent = processCPUPercent
        self.processCPUCurrentPercent = processCPUCurrentPercent
        self.processCPUCurrentHighWaterPercent = processCPUCurrentHighWaterPercent
        self.processCPUAveragePercentSinceStart = processCPUAveragePercentSinceStart ?? processCPUPercent
        self.processCPUAverageHighWaterPercentSinceStart = processCPUAverageHighWaterPercentSinceStart
        self.availableDiskBytes = availableDiskBytes
        self.availableDiskLowWaterBytes = availableDiskLowWaterBytes
        self.prewarmedModels = prewarmedModels
        self.warmingModels = warmingModels
        self.activeSynthesisCount = activeSynthesisCount
        self.oldestActiveSynthesisAgeSeconds = oldestActiveSynthesisAgeSeconds
        self.stalledSynthesisCount = stalledSynthesisCount
        self.activeSynthesisRequests = activeSynthesisRequests
        self.lastSynthesisCompletionReason = lastSynthesisCompletionReason
        self.recentResidentEvictions = recentResidentEvictions
        self.uptimeSeconds = uptimeSeconds
    }
}

public struct DaemonRuntimeTrimRequestDTO: Codable, Sendable, Equatable {
    public let modelIDs: [String]?
    public let includeWarmStartModels: Bool

    public init(
        modelIDs: [String]? = nil,
        includeWarmStartModels: Bool = false
    ) {
        self.modelIDs = modelIDs
        self.includeWarmStartModels = includeWarmStartModels
    }

    private enum CodingKeys: String, CodingKey {
        case modelIDs
        case includeWarmStartModels
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.modelIDs = try container.decodeIfPresent([String].self, forKey: .modelIDs)
        self.includeWarmStartModels = try container.decodeIfPresent(Bool.self, forKey: .includeWarmStartModels) ?? false
    }
}

public struct DaemonRuntimeTrimResultDTO: Codable, Sendable, Equatable {
    public let trimmedModelIDs: [String]
    public let skippedModelIDs: [String]
    public let effectiveWarmStartModels: [String]
    public let residentModelIDs: [String]
    public let memoryBudgetBreached: Bool
    public let processFootprintBytes: Int
    public let processFootprintOverBudgetBytes: Int
    public let reason: String

    public init(
        trimmedModelIDs: [String],
        skippedModelIDs: [String],
        effectiveWarmStartModels: [String],
        residentModelIDs: [String],
        memoryBudgetBreached: Bool,
        processFootprintBytes: Int,
        processFootprintOverBudgetBytes: Int,
        reason: String
    ) {
        self.trimmedModelIDs = trimmedModelIDs
        self.skippedModelIDs = skippedModelIDs
        self.effectiveWarmStartModels = effectiveWarmStartModels
        self.residentModelIDs = residentModelIDs
        self.memoryBudgetBreached = memoryBudgetBreached
        self.processFootprintBytes = processFootprintBytes
        self.processFootprintOverBudgetBytes = processFootprintOverBudgetBytes
        self.reason = reason
    }
}

public struct DaemonIdleTrimResultDTO: Codable, Sendable, Equatable {
    public let occurredAt: String
    public let trimmedModelIDs: [String]
    public let reason: String

    public init(
        occurredAt: String,
        trimmedModelIDs: [String],
        reason: String
    ) {
        self.occurredAt = occurredAt
        self.trimmedModelIDs = trimmedModelIDs
        self.reason = reason
    }

    public init(from result: DaemonIdleTrimResult) {
        self.init(
            occurredAt: ValarDTOFormatting.iso8601String(from: result.occurredAt),
            trimmedModelIDs: result.trimmedModelIDs,
            reason: result.reason
        )
    }
}

public struct ActiveSynthesisRequestDTO: Codable, Sendable, Equatable {
    public let id: String
    public let modelID: String
    public let voiceBehavior: String
    public let executionMode: String
    public let segmentIndex: Int
    public let segmentCount: Int
    public let startedAt: String
    public let lastHeartbeatAt: String
    public let ageSeconds: Double
    public let lastHeartbeatAgeSeconds: Double
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
        id: String,
        modelID: String,
        voiceBehavior: String,
        executionMode: String,
        segmentIndex: Int,
        segmentCount: Int,
        startedAt: String,
        lastHeartbeatAt: String,
        ageSeconds: Double,
        lastHeartbeatAgeSeconds: Double,
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
        self.ageSeconds = ageSeconds
        self.lastHeartbeatAgeSeconds = lastHeartbeatAgeSeconds
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

public struct ModelCleanupPayloadDTO: Codable, Sendable, Equatable {
    public let dryRun: Bool
    public let removedStaleModelIDs: [String]
    public let orphanedModelPackPaths: [String]
    public let removedOrphanedModelPackPaths: [String]
    public let removedLegacyCachePaths: [String]
    public let legacyMLXAudioSafeToDelete: Bool
    public let message: String

    public init(
        dryRun: Bool,
        removedStaleModelIDs: [String],
        orphanedModelPackPaths: [String],
        removedOrphanedModelPackPaths: [String] = [],
        removedLegacyCachePaths: [String] = [],
        legacyMLXAudioSafeToDelete: Bool,
        message: String
    ) {
        self.dryRun = dryRun
        self.removedStaleModelIDs = removedStaleModelIDs
        self.orphanedModelPackPaths = orphanedModelPackPaths
        self.removedOrphanedModelPackPaths = removedOrphanedModelPackPaths
        self.removedLegacyCachePaths = removedLegacyCachePaths
        self.legacyMLXAudioSafeToDelete = legacyMLXAudioSafeToDelete
        self.message = message
    }
}

public enum DaemonReadinessMode: String, Codable, Sendable, Equatable {
    case resident
    case loadOnDemand
    case unavailable
}

public enum WarmStartModelSourceDTO: String, Codable, Sendable, Equatable {
    case explicit
    case `default`
}

public struct ModelRouteDescriptorDTO: Codable, Sendable, Equatable, Identifiable {
    public let id: String
    public let displayName: String
    public let family: String
    public let provider: String
    public let installState: String
    /// True when the model is registered in the ValarTTS model pack store.
    public let installed: Bool
    /// True when model files exist on disk (either installed in model packs or
    /// present in the mlx-audio HuggingFace cache). Ready to use without downloading.
    public let cachedOnDisk: Bool
    /// Install-path health status for installed model records.
    public let installPathStatus: String?
    public let providerURL: String?
    public let capabilities: [String]
    public let voiceFeatures: [String]
    public let defaultSampleRate: Double?
    public let notes: String?
    public let licenseName: String?
    public let licenseURL: String?
    public let supportsReferenceAudio: Bool
    public let estimatedSizeBytes: Int?
    public let supportTier: String
    public let releaseEligible: Bool
    public let qualityTierByLanguage: [String: String]
    public let distributionTier: String

    public init(
        id: String,
        displayName: String,
        family: String,
        provider: String,
        installState: String,
        installed: Bool,
        cachedOnDisk: Bool,
        installPathStatus: String? = nil,
        providerURL: String? = nil,
        capabilities: [String] = [],
        voiceFeatures: [String] = [],
        defaultSampleRate: Double? = nil,
        notes: String? = nil,
        licenseName: String? = nil,
        licenseURL: String? = nil,
        supportsReferenceAudio: Bool = false,
        estimatedSizeBytes: Int? = nil,
        supportTier: String = ModelSupportTier.supported.rawValue,
        releaseEligible: Bool = true,
        qualityTierByLanguage: [String: String] = [:],
        distributionTier: String = ModelDistributionTier.optionalInstall.rawValue
    ) {
        self.id = id
        self.displayName = displayName
        self.family = family
        self.provider = provider
        self.installState = installState
        self.installed = installed
        self.cachedOnDisk = cachedOnDisk
        self.installPathStatus = installPathStatus
        self.providerURL = providerURL
        self.capabilities = capabilities
        self.voiceFeatures = voiceFeatures
        self.defaultSampleRate = defaultSampleRate
        self.notes = notes
        self.licenseName = licenseName
        self.licenseURL = licenseURL
        self.supportsReferenceAudio = supportsReferenceAudio
        self.estimatedSizeBytes = estimatedSizeBytes
        self.supportTier = supportTier
        self.releaseEligible = releaseEligible
        self.qualityTierByLanguage = qualityTierByLanguage
        self.distributionTier = distributionTier
    }

    public init(from model: CatalogModel) {
        let isInstalled = model.installState == .installed
        let caps = model.descriptor.capabilities
        let voiceSupport = model.descriptor.voiceSupport
        let entry = SupportedModelCatalog.entry(for: model.id)
        let licenseName = model.licenseName ?? entry?.manifest.licenses.first?.name
        let license = model.licenseURL?.absoluteString ?? entry?.manifest.licenses.first?.sourceURL?.absoluteString
        let totalBytes: Int? = entry.flatMap { e in
            let sizes = e.manifest.artifacts.compactMap(\.sizeBytes)
            return sizes.isEmpty ? nil : sizes.reduce(0, +)
        }
        self.init(
            id: model.id.rawValue,
            displayName: model.descriptor.displayName,
            family: model.familyID.rawValue,
            provider: model.providerName,
            installState: model.installState.rawValue,
            installed: isInstalled,
            cachedOnDisk: isInstalled || model.cachedOnDisk,
            installPathStatus: model.installPathStatus?.rawValue,
            providerURL: model.providerURL?.absoluteString,
            capabilities: caps.map(\.rawValue).sorted(),
            voiceFeatures: voiceSupport.features.map(\.rawValue),
            defaultSampleRate: model.descriptor.defaultSampleRate,
            notes: model.notes ?? model.descriptor.notes,
            licenseName: licenseName,
            licenseURL: license,
            supportsReferenceAudio: voiceSupport.supportsReferenceAudio,
            estimatedSizeBytes: totalBytes,
            supportTier: model.supportTier.rawValue,
            releaseEligible: model.releaseEligible,
            qualityTierByLanguage: model.qualityTierByLanguage.mapValues(\.rawValue),
            distributionTier: model.distributionTier.rawValue
        )
    }
}

public struct ModelInstallRequestDTO: Codable, Sendable, Equatable {
    public let model: String
    public let allowDownload: Bool
    public let refreshCache: Bool

    public init(model: String, allowDownload: Bool = false, refreshCache: Bool = false) {
        self.model = model
        self.allowDownload = allowDownload
        self.refreshCache = refreshCache
    }

    private enum CodingKeys: String, CodingKey {
        case model
        case modelID = "model_id"
        case allowDownload = "allow_download"
        case refreshCache = "refresh_cache"
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        if let model = try container.decodeIfPresent(String.self, forKey: .model),
           !model.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            self.model = model
            self.allowDownload = try container.decodeIfPresent(Bool.self, forKey: .allowDownload) ?? false
            self.refreshCache = try container.decodeIfPresent(Bool.self, forKey: .refreshCache) ?? false
            return
        }
        if let modelID = try container.decodeIfPresent(String.self, forKey: .modelID),
           !modelID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            self.model = modelID
            self.allowDownload = try container.decodeIfPresent(Bool.self, forKey: .allowDownload) ?? false
            self.refreshCache = try container.decodeIfPresent(Bool.self, forKey: .refreshCache) ?? false
            return
        }
        throw DecodingError.keyNotFound(
            CodingKeys.model,
            DecodingError.Context(
                codingPath: container.codingPath,
                debugDescription: "Expected 'model' or 'model_id'."
            )
        )
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(model, forKey: .model)
    }
}

public struct ModelInstallOperationDTO: Codable, Sendable, Equatable {
    public let operationId: String
    public let status: String

    public init(
        operationId: String,
        status: String = "queued"
    ) {
        self.operationId = operationId
        self.status = status
    }
}

public struct DaemonOperationStatusDTO: Codable, Sendable, Equatable, Identifiable {
    public let operationId: String
    public let kind: String
    public let status: String
    public let createdAt: String
    public let startedAt: String?
    public let finishedAt: String?
    public let error: String?

    public var id: String { operationId }

    public init(
        operationId: String,
        kind: String,
        status: String,
        createdAt: String,
        startedAt: String? = nil,
        finishedAt: String? = nil,
        error: String? = nil
    ) {
        self.operationId = operationId
        self.kind = kind
        self.status = status
        self.createdAt = createdAt
        self.startedAt = startedAt
        self.finishedAt = finishedAt
        self.error = error
    }
}

public struct DaemonQueueStateDTO: Codable, Sendable, Equatable {
    public let operations: [DaemonOperationStatusDTO]
    public let queuedCount: Int
    public let runningCount: Int

    public init(
        operations: [DaemonOperationStatusDTO],
        queuedCount: Int,
        runningCount: Int
    ) {
        self.operations = operations
        self.queuedCount = queuedCount
        self.runningCount = runningCount
    }
}

public struct ModelRemoveRequestDTO: Codable, Sendable, Equatable {
    public let model: String

    public init(model: String) {
        self.model = model
    }
}

public struct OKResponseDTO: Codable, Sendable, Equatable {
    public let ok: Bool

    public init(ok: Bool) {
        self.ok = ok
    }
}

public struct ModelOperationPayloadDTO: Codable, Sendable, Equatable {
    public let message: String
    public let model: ModelSummaryDTO
    public let result: String?
    public let installedPath: String?
    public let removedPath: String?
    public let warnings: [String]?

    public init(
        message: String,
        model: ModelSummaryDTO,
        result: String? = nil,
        installedPath: String? = nil,
        removedPath: String? = nil,
        warnings: [String]? = nil
    ) {
        self.message = message
        self.model = model
        self.result = result
        self.installedPath = installedPath
        self.removedPath = removedPath
        self.warnings = warnings
    }
}

public struct VoiceSummaryDTO: Codable, Sendable, Equatable {
    public let id: String
    public let label: String
    public let modelID: String
    public let preview: String
    public let voiceKind: String?
    public let isLegacyExpressive: Bool

    public init(
        id: String,
        label: String,
        modelID: String,
        preview: String,
        voiceKind: String? = nil,
        isLegacyExpressive: Bool = false
    ) {
        self.id = id
        self.label = label
        self.modelID = modelID
        self.preview = preview
        self.voiceKind = voiceKind
        self.isLegacyExpressive = isLegacyExpressive
    }

    public init(
        from voice: VoiceLibraryRecord,
        preview: String
    ) {
        self.init(
            id: voice.id.uuidString,
            label: voice.label,
            modelID: voice.modelID,
            preview: preview,
            voiceKind: voice.effectiveVoiceKind,
            isLegacyExpressive: voice.isLegacyExpressive
        )
    }
}

public struct VoiceDetailDTO: Codable, Sendable, Equatable {
    public let id: String
    public let label: String
    public let modelID: String
    public let runtimeModelID: String?
    public let preview: String
    public let sourceAssetName: String?
    public let referenceAudioAssetName: String?
    public let referenceTranscript: String?
    public let referenceDurationSeconds: Double?
    public let referenceSampleRate: Double?
    public let referenceChannelCount: Int?
    public let conditioningFormat: String?
    public let conditioningData: Data?
    public let conditioningAssetName: String?
    public let conditioningSourceModel: String?
    public let conditioningMetadata: String?
    public let voicePrompt: String?
    public let createdAt: String
    public let isClonedVoice: Bool
    public let voiceKind: String?
    public let isLegacyExpressive: Bool

    public init(
        id: String,
        label: String,
        modelID: String,
        runtimeModelID: String?,
        preview: String,
        sourceAssetName: String?,
        referenceAudioAssetName: String?,
        referenceTranscript: String?,
        referenceDurationSeconds: Double?,
        referenceSampleRate: Double?,
        referenceChannelCount: Int?,
        conditioningFormat: String?,
        conditioningData: Data?,
        conditioningAssetName: String?,
        conditioningSourceModel: String?,
        conditioningMetadata: String?,
        voicePrompt: String?,
        createdAt: String,
        isClonedVoice: Bool,
        voiceKind: String? = nil,
        isLegacyExpressive: Bool = false
    ) {
        self.id = id
        self.label = label
        self.modelID = modelID
        self.runtimeModelID = runtimeModelID
        self.preview = preview
        self.sourceAssetName = sourceAssetName
        self.referenceAudioAssetName = referenceAudioAssetName
        self.referenceTranscript = referenceTranscript
        self.referenceDurationSeconds = referenceDurationSeconds
        self.referenceSampleRate = referenceSampleRate
        self.referenceChannelCount = referenceChannelCount
        self.conditioningFormat = conditioningFormat
        self.conditioningData = conditioningData
        self.conditioningAssetName = conditioningAssetName
        self.conditioningSourceModel = conditioningSourceModel
        self.conditioningMetadata = conditioningMetadata
        self.voicePrompt = voicePrompt
        self.createdAt = createdAt
        self.isClonedVoice = isClonedVoice
        self.voiceKind = voiceKind
        self.isLegacyExpressive = isLegacyExpressive
    }

    public init(
        from voice: VoiceLibraryRecord,
        preview: String
    ) {
        self.init(
            id: voice.id.uuidString,
            label: voice.label,
            modelID: voice.modelID,
            runtimeModelID: voice.runtimeModelID,
            preview: preview,
            sourceAssetName: voice.sourceAssetName,
            referenceAudioAssetName: voice.referenceAudioAssetName,
            referenceTranscript: voice.referenceTranscript,
            referenceDurationSeconds: voice.referenceDurationSeconds,
            referenceSampleRate: voice.referenceSampleRate,
            referenceChannelCount: voice.referenceChannelCount,
            conditioningFormat: voice.conditioningFormat,
            conditioningData: nil,
            conditioningAssetName: nil,
            conditioningSourceModel: nil,
            conditioningMetadata: nil,
            voicePrompt: voice.voicePrompt,
            createdAt: ValarDTOFormatting.iso8601String(from: voice.createdAt),
            isClonedVoice: voice.isClonedVoice,
            voiceKind: voice.effectiveVoiceKind,
            isLegacyExpressive: voice.isLegacyExpressive
        )
    }
}

public struct VoiceListPayloadDTO: Codable, Sendable, Equatable {
    public let message: String
    public let databasePath: String
    public let voices: [VoiceSummaryDTO]

    public init(
        message: String,
        databasePath: String,
        voices: [VoiceSummaryDTO]
    ) {
        self.message = message
        self.databasePath = databasePath
        self.voices = voices
    }
}

public struct VoiceDetailPayloadDTO: Codable, Sendable, Equatable {
    public let message: String
    public let voice: VoiceDetailDTO
    public let previewPath: String

    public init(
        message: String,
        voice: VoiceDetailDTO,
        previewPath: String
    ) {
        self.message = message
        self.voice = voice
        self.previewPath = previewPath
    }
}

public struct SpeechSynthesisPayloadDTO: Codable, Sendable, Equatable {
    public let message: String
    public let modelID: String
    public let outputPath: String
    public let text: String
    public let voiceID: String
    public let effectiveVoiceID: String?
    public let effectiveLanguage: String?
    public let voiceSelectionMode: String?

    public init(
        message: String,
        modelID: String,
        outputPath: String,
        text: String,
        voiceID: String,
        effectiveVoiceID: String? = nil,
        effectiveLanguage: String? = nil,
        voiceSelectionMode: String? = nil
    ) {
        self.message = message
        self.modelID = modelID
        self.outputPath = outputPath
        self.text = text
        self.voiceID = voiceID
        self.effectiveVoiceID = effectiveVoiceID
        self.effectiveLanguage = effectiveLanguage
        self.voiceSelectionMode = voiceSelectionMode
    }
}

public struct AlignmentTokenDTO: Codable, Sendable, Equatable {
    public let text: String
    public let startTime: TimeInterval
    public let endTime: TimeInterval
    public let confidence: Double?

    public init(
        text: String,
        startTime: TimeInterval,
        endTime: TimeInterval,
        confidence: Double? = nil
    ) {
        self.text = text
        self.startTime = startTime
        self.endTime = endTime
        self.confidence = confidence
    }

    public init(from token: AlignmentToken) {
        self.init(
            text: token.text,
            startTime: token.startTime,
            endTime: token.endTime,
            confidence: token.confidence
        )
    }
}

public struct AlignmentPayloadDTO: Codable, Sendable, Equatable {
    public let message: String
    public let modelID: String
    public let audioPath: String
    public let outputPath: String?
    public let transcript: String
    public let tokens: [AlignmentTokenDTO]

    public init(
        message: String,
        modelID: String,
        audioPath: String,
        outputPath: String?,
        transcript: String,
        tokens: [AlignmentTokenDTO]
    ) {
        self.message = message
        self.modelID = modelID
        self.audioPath = audioPath
        self.outputPath = outputPath
        self.transcript = transcript
        self.tokens = tokens
    }
}

public struct ProjectSessionDTO: Codable, Sendable, Equatable {
    public let version: Int
    public let projectID: String
    public let title: String
    public let bundlePath: String
    public let createdAt: String
    public let openedAt: String

    public init(
        version: Int,
        projectID: String,
        title: String,
        bundlePath: String,
        createdAt: String,
        openedAt: String
    ) {
        self.version = version
        self.projectID = projectID
        self.title = title
        self.bundlePath = bundlePath
        self.createdAt = createdAt
        self.openedAt = openedAt
    }
}

public struct ProjectSessionPayloadDTO: Codable, Sendable, Equatable {
    public let message: String
    public let project: ProjectSessionDTO

    public init(
        message: String,
        project: ProjectSessionDTO
    ) {
        self.message = message
        self.project = project
    }
}

public struct ProjectInfoDTO: Codable, Sendable, Equatable {
    public let title: String
    public let projectID: String
    public let bundlePath: String
    public let createdAt: String
    public let openedAt: String
    public let chapters: Int
    public let renderJobs: Int
    public let exports: Int
    public let speakers: Int

    public init(
        title: String,
        projectID: String,
        bundlePath: String,
        createdAt: String,
        openedAt: String,
        chapters: Int,
        renderJobs: Int,
        exports: Int,
        speakers: Int
    ) {
        self.title = title
        self.projectID = projectID
        self.bundlePath = bundlePath
        self.createdAt = createdAt
        self.openedAt = openedAt
        self.chapters = chapters
        self.renderJobs = renderJobs
        self.exports = exports
        self.speakers = speakers
    }
}

public struct ProjectInfoPayloadDTO: Codable, Sendable, Equatable {
    public let message: String
    public let project: ProjectInfoDTO

    public init(
        message: String,
        project: ProjectInfoDTO
    ) {
        self.message = message
        self.project = project
    }
}

public struct ChapterDTO: Codable, Sendable, Equatable {
    public let id: String
    public let projectID: String
    public let index: Int
    public let title: String
    public let text: String
    public let textLength: Int
    public let speakerLabel: String?
    public let estimatedDurationSeconds: Double?
    public let hasSourceAudio: Bool
    public let sourceAudioAssetName: String?
    public let sourceAudioSampleRate: Double?
    public let sourceAudioDurationSeconds: Double?
    public let transcriptionJSON: String?
    public let transcriptionModelID: String?
    public let alignmentJSON: String?
    public let alignmentModelID: String?
    public let derivedTranslationText: String?

    public init(
        id: String,
        projectID: String,
        index: Int,
        title: String,
        text: String,
        textLength: Int,
        speakerLabel: String?,
        estimatedDurationSeconds: Double?,
        hasSourceAudio: Bool,
        sourceAudioAssetName: String?,
        sourceAudioSampleRate: Double?,
        sourceAudioDurationSeconds: Double?,
        transcriptionJSON: String?,
        transcriptionModelID: String?,
        alignmentJSON: String?,
        alignmentModelID: String?,
        derivedTranslationText: String?
    ) {
        self.id = id
        self.projectID = projectID
        self.index = index
        self.title = title
        self.text = text
        self.textLength = textLength
        self.speakerLabel = speakerLabel
        self.estimatedDurationSeconds = estimatedDurationSeconds
        self.hasSourceAudio = hasSourceAudio
        self.sourceAudioAssetName = sourceAudioAssetName
        self.sourceAudioSampleRate = sourceAudioSampleRate
        self.sourceAudioDurationSeconds = sourceAudioDurationSeconds
        self.transcriptionJSON = transcriptionJSON
        self.transcriptionModelID = transcriptionModelID
        self.alignmentJSON = alignmentJSON
        self.alignmentModelID = alignmentModelID
        self.derivedTranslationText = derivedTranslationText
    }

    public init(from chapter: ChapterRecord) {
        self.init(
            id: chapter.id.uuidString,
            projectID: chapter.projectID.uuidString,
            index: chapter.index,
            title: chapter.title,
            text: chapter.script,
            textLength: chapter.script.count,
            speakerLabel: chapter.speakerLabel,
            estimatedDurationSeconds: chapter.estimatedDurationSeconds,
            hasSourceAudio: chapter.hasSourceAudio,
            sourceAudioAssetName: chapter.sourceAudioAssetName,
            sourceAudioSampleRate: chapter.sourceAudioSampleRate,
            sourceAudioDurationSeconds: chapter.sourceAudioDurationSeconds,
            transcriptionJSON: chapter.transcriptionJSON,
            transcriptionModelID: chapter.transcriptionModelID,
            alignmentJSON: chapter.alignmentJSON,
            alignmentModelID: chapter.alignmentModelID,
            derivedTranslationText: chapter.derivedTranslationText
        )
    }
}

public struct ChapterListPayloadDTO: Codable, Sendable, Equatable {
    public let message: String
    public let projectTitle: String
    public let chapters: [ChapterDTO]

    public init(
        message: String,
        projectTitle: String,
        chapters: [ChapterDTO]
    ) {
        self.message = message
        self.projectTitle = projectTitle
        self.chapters = chapters
    }
}

public struct ChapterMutationPayloadDTO: Codable, Sendable, Equatable {
    public let message: String
    public let chapter: ChapterDTO?
    public let chapterID: String?

    public init(
        message: String,
        chapter: ChapterDTO? = nil,
        chapterID: String? = nil
    ) {
        self.message = message
        self.chapter = chapter
        self.chapterID = chapterID
    }
}

public struct RenderJobDTO: Codable, Sendable, Equatable {
    public let id: String
    public let projectID: String
    public let chapterID: String?
    public let modelID: String
    public let title: String
    public let outputFileName: String
    public let state: String
    public let progress: Double
    public let progressLabel: String
    public let failureReason: String?
    public let createdAt: String

    public init(
        id: String,
        projectID: String,
        chapterID: String?,
        modelID: String,
        title: String,
        outputFileName: String,
        state: String,
        progress: Double,
        progressLabel: String,
        failureReason: String?,
        createdAt: String
    ) {
        self.id = id
        self.projectID = projectID
        self.chapterID = chapterID
        self.modelID = modelID
        self.title = title
        self.outputFileName = outputFileName
        self.state = state
        self.progress = progress
        self.progressLabel = progressLabel
        self.failureReason = failureReason
        self.createdAt = createdAt
    }

    public init(from job: RenderJob) {
        let clampedProgress = max(0, min(job.progress, 1))
        self.init(
            id: job.id.uuidString,
            projectID: job.projectID.uuidString,
            chapterID: job.chapterID?.uuidString,
            modelID: job.modelID.rawValue,
            title: job.title ?? job.chapterID?.uuidString ?? "-",
            outputFileName: job.outputFileName,
            state: job.state.rawValue,
            progress: job.progress,
            progressLabel: "\(Int((clampedProgress * 100).rounded()))%",
            failureReason: job.failureReason,
            createdAt: ValarDTOFormatting.iso8601String(from: job.createdAt)
        )
    }

    public init(from job: RenderJobRecord) {
        let clampedProgress = max(0, min(job.progress, 1))
        self.init(
            id: job.id.uuidString,
            projectID: job.projectID.uuidString,
            chapterID: job.chapterIDs.first?.uuidString,
            modelID: job.modelID,
            title: job.title ?? job.chapterIDs.first?.uuidString ?? "-",
            outputFileName: job.outputFileName,
            state: job.state,
            progress: job.progress,
            progressLabel: "\(Int((clampedProgress * 100).rounded()))%",
            failureReason: job.failureReason,
            createdAt: ValarDTOFormatting.iso8601String(from: job.createdAt)
        )
    }
}

public struct RenderStatusPayloadDTO: Codable, Sendable, Equatable {
    public let message: String
    public let projectTitle: String?
    public let watched: Bool?
    public let generatedAt: String?
    public let processedCount: Int?
    public let queuedCount: Int?
    public let remainingPendingCount: Int?
    public let renders: [RenderJobDTO]

    public init(
        message: String,
        projectTitle: String? = nil,
        watched: Bool? = nil,
        generatedAt: String? = nil,
        processedCount: Int? = nil,
        queuedCount: Int? = nil,
        remainingPendingCount: Int? = nil,
        renders: [RenderJobDTO]
    ) {
        self.message = message
        self.projectTitle = projectTitle
        self.watched = watched
        self.generatedAt = generatedAt
        self.processedCount = processedCount
        self.queuedCount = queuedCount
        self.remainingPendingCount = remainingPendingCount
        self.renders = renders
    }
}

public struct RenderDetailPayloadDTO: Codable, Sendable, Equatable {
    public let message: String
    public let render: RenderJobDTO

    public init(
        message: String,
        render: RenderJobDTO
    ) {
        self.message = message
        self.render = render
    }
}

public struct ExportDTO: Codable, Sendable, Equatable {
    public let id: String
    public let projectID: String
    public let fileName: String
    public let createdAt: String
    public let checksum: String?

    public init(
        id: String,
        projectID: String,
        fileName: String,
        createdAt: String,
        checksum: String?
    ) {
        self.id = id
        self.projectID = projectID
        self.fileName = fileName
        self.createdAt = createdAt
        self.checksum = checksum
    }

    public init(from export: ExportRecord) {
        self.init(
            id: export.id.uuidString,
            projectID: export.projectID.uuidString,
            fileName: export.fileName,
            createdAt: ValarDTOFormatting.iso8601String(from: export.createdAt),
            checksum: export.checksum
        )
    }
}

public struct ExportListPayloadDTO: Codable, Sendable, Equatable {
    public let message: String
    public let projectTitle: String
    public let exports: [ExportDTO]

    public init(
        message: String,
        projectTitle: String,
        exports: [ExportDTO]
    ) {
        self.message = message
        self.projectTitle = projectTitle
        self.exports = exports
    }
}

public struct ExportCreatePayloadDTO: Codable, Sendable, Equatable {
    public let message: String
    public let chapterID: String
    public let modelID: String?
    public let outputPath: String
    public let export: ExportDTO

    public init(
        message: String,
        chapterID: String,
        modelID: String?,
        outputPath: String,
        export: ExportDTO
    ) {
        self.message = message
        self.chapterID = chapterID
        self.modelID = modelID
        self.outputPath = outputPath
        self.export = export
    }
}

public struct RuntimeConfigurationDTO: Codable, Sendable, Equatable {
    public let warmPolicy: String
    public let maxResidentBytes: Int
    public let maxResidentModels: Int
    public let maxQueuedRenderJobs: Int
    public let warmStartModelIDs: [String]?
    public let idleTrimSettleGraceSeconds: Double
    public let idleTrimRecentUseGraceSeconds: Double

    public init(
        warmPolicy: String,
        maxResidentBytes: Int,
        maxResidentModels: Int,
        maxQueuedRenderJobs: Int,
        warmStartModelIDs: [String]? = nil,
        idleTrimSettleGraceSeconds: Double,
        idleTrimRecentUseGraceSeconds: Double
    ) {
        self.warmPolicy = warmPolicy
        self.maxResidentBytes = maxResidentBytes
        self.maxResidentModels = maxResidentModels
        self.maxQueuedRenderJobs = maxQueuedRenderJobs
        self.warmStartModelIDs = warmStartModelIDs
        self.idleTrimSettleGraceSeconds = idleTrimSettleGraceSeconds
        self.idleTrimRecentUseGraceSeconds = idleTrimRecentUseGraceSeconds
    }

    public init(from configuration: RuntimeConfiguration) {
        self.init(
            warmPolicy: configuration.warmPolicy.rawValue,
            maxResidentBytes: configuration.maxResidentBytes,
            maxResidentModels: configuration.maxResidentModels,
            maxQueuedRenderJobs: configuration.maxQueuedRenderJobs,
            warmStartModelIDs: configuration.warmStartModelIDs?.map(\.rawValue),
            idleTrimSettleGraceSeconds: configuration.idleTrimSettleGraceSeconds,
            idleTrimRecentUseGraceSeconds: configuration.idleTrimRecentUseGraceSeconds
        )
    }
}

public struct AppPathsDTO: Codable, Sendable, Equatable {
    public let applicationSupport: String
    public let modelPacksDirectory: String
    public let projectsDirectory: String
    public let voiceLibraryDirectory: String
    public let cacheDirectory: String
    public let importsDirectory: String
    public let snapshotsDirectory: String
    public let databasePath: String

    public init(
        applicationSupport: String,
        modelPacksDirectory: String,
        projectsDirectory: String,
        voiceLibraryDirectory: String,
        cacheDirectory: String,
        importsDirectory: String,
        snapshotsDirectory: String,
        databasePath: String
    ) {
        self.applicationSupport = applicationSupport
        self.modelPacksDirectory = modelPacksDirectory
        self.projectsDirectory = projectsDirectory
        self.voiceLibraryDirectory = voiceLibraryDirectory
        self.cacheDirectory = cacheDirectory
        self.importsDirectory = importsDirectory
        self.snapshotsDirectory = snapshotsDirectory
        self.databasePath = databasePath
    }

    public init(from paths: ValarAppPaths) {
        self.init(
            applicationSupport: paths.applicationSupport.path,
            modelPacksDirectory: paths.modelPacksDirectory.path,
            projectsDirectory: paths.projectsDirectory.path,
            voiceLibraryDirectory: paths.voiceLibraryDirectory.path,
            cacheDirectory: paths.cacheDirectory.path,
            importsDirectory: paths.importsDirectory.path,
            snapshotsDirectory: paths.snapshotsDirectory.path,
            databasePath: paths.databaseURL.path
        )
    }
}

public struct ProjectSummaryDTO: Codable, Sendable, Equatable {
    public let id: String
    public let title: String
    public let createdAt: String
    public let updatedAt: String
    public let notes: String?

    public init(
        id: String,
        title: String,
        createdAt: String,
        updatedAt: String,
        notes: String?
    ) {
        self.id = id
        self.title = title
        self.createdAt = createdAt
        self.updatedAt = updatedAt
        self.notes = notes
    }

    public init(from project: ProjectRecord) {
        self.init(
            id: project.id.uuidString,
            title: project.title,
            createdAt: ValarDTOFormatting.iso8601String(from: project.createdAt),
            updatedAt: ValarDTOFormatting.iso8601String(from: project.updatedAt),
            notes: project.notes
        )
    }
}

public struct ModelResidencySnapshotDTO: Codable, Sendable, Equatable {
    public let id: String
    public let displayName: String
    public let domain: String
    public let state: String
    public let lastTouchedAt: String?
    public let residentRank: Int?
    public let estimatedResidentBytes: Int
    public let actualResidentBytes: Int?
    public let residencyPolicy: String
    public let activeSessionCount: Int
    public let isWarmStartModel: Bool
    public let idleTrimEligible: Bool

    public init(
        id: String,
        displayName: String,
        domain: String,
        state: String,
        lastTouchedAt: String?,
        residentRank: Int?,
        estimatedResidentBytes: Int,
        actualResidentBytes: Int?,
        residencyPolicy: String,
        activeSessionCount: Int,
        isWarmStartModel: Bool = false,
        idleTrimEligible: Bool = false
    ) {
        self.id = id
        self.displayName = displayName
        self.domain = domain
        self.state = state
        self.lastTouchedAt = lastTouchedAt
        self.residentRank = residentRank
        self.estimatedResidentBytes = estimatedResidentBytes
        self.actualResidentBytes = actualResidentBytes
        self.residencyPolicy = residencyPolicy
        self.activeSessionCount = activeSessionCount
        self.isWarmStartModel = isWarmStartModel
        self.idleTrimEligible = idleTrimEligible
    }

    public init(from snapshot: ModelResidencySnapshot) {
        self.init(
            id: snapshot.descriptor.id.rawValue,
            displayName: snapshot.descriptor.displayName,
            domain: snapshot.descriptor.domain.rawValue,
            state: snapshot.state.rawValue,
            lastTouchedAt: snapshot.lastTouchedAt.map(ValarDTOFormatting.iso8601String),
            residentRank: snapshot.residentRank,
            estimatedResidentBytes: snapshot.estimatedResidentBytes,
            actualResidentBytes: snapshot.actualResidentBytes,
            residencyPolicy: snapshot.residencyPolicy.rawValue,
            activeSessionCount: snapshot.activeSessionCount
        )
    }
}

public struct DiagnosticsDTO: Codable, Sendable, Equatable {
    public let appPaths: AppPathsDTO
    public let runtimeConfiguration: RuntimeConfigurationDTO
    public let models: [ModelSummaryDTO]
    public let modelSnapshots: [ModelResidencySnapshotDTO]
    public let projects: [ProjectSummaryDTO]
    public let voices: [VoiceSummaryDTO]
    public let renders: [RenderJobDTO]
    public let lastUpdatedAt: String

    public init(
        appPaths: AppPathsDTO,
        runtimeConfiguration: RuntimeConfigurationDTO,
        models: [ModelSummaryDTO],
        modelSnapshots: [ModelResidencySnapshotDTO],
        projects: [ProjectSummaryDTO],
        voices: [VoiceSummaryDTO],
        renders: [RenderJobDTO],
        lastUpdatedAt: String
    ) {
        self.appPaths = appPaths
        self.runtimeConfiguration = runtimeConfiguration
        self.models = models
        self.modelSnapshots = modelSnapshots
        self.projects = projects
        self.voices = voices
        self.renders = renders
        self.lastUpdatedAt = lastUpdatedAt
    }
}

// MARK: - Session DTOs

public struct SessionNewRequestDTO: Codable, Sendable {
    public let path: String

    public init(path: String) {
        self.path = path
    }
}

public struct SessionNewResponseDTO: Codable, Sendable, Equatable {
    public let sessionId: String

    public init(sessionId: String) {
        self.sessionId = sessionId
    }
}

public struct ChapterCreateRequestDTO: Codable, Sendable {
    public let title: String
    public let text: String
    public let index: Int
    public let speakerLabel: String?

    public init(title: String, text: String, index: Int, speakerLabel: String? = nil) {
        self.title = title
        self.text = text
        self.index = index
        self.speakerLabel = speakerLabel
    }
}

public struct ChapterUpdateRequestDTO: Codable, Sendable {
    public let title: String?
    public let text: String?

    public init(title: String? = nil, text: String? = nil) {
        self.title = title
        self.text = text
    }
}

public struct CapabilitySnapshotDTO: Codable, Sendable {
    public let canSpeakNow: Bool
    public let canTranscribeNow: Bool
    public let canAlignNow: Bool
    public let canCloneVoiceNow: Bool
    public let installedTTSModels: [String]
    public let installedASRModels: [String]
    public let cachedButNotRegistered: [String]
    public let installedVoiceCount: Int
    public let daemonReachable: Bool
    public let daemonReady: Bool
    public let metallibAvailable: Bool
    public let inferenceAssetIssue: String?
    public let missingPrerequisites: [String]

    public init(
        canSpeakNow: Bool,
        canTranscribeNow: Bool,
        canAlignNow: Bool,
        canCloneVoiceNow: Bool,
        installedTTSModels: [String],
        installedASRModels: [String],
        cachedButNotRegistered: [String],
        installedVoiceCount: Int,
        daemonReachable: Bool,
        daemonReady: Bool,
        metallibAvailable: Bool,
        inferenceAssetIssue: String? = nil,
        missingPrerequisites: [String]
    ) {
        self.canSpeakNow = canSpeakNow
        self.canTranscribeNow = canTranscribeNow
        self.canAlignNow = canAlignNow
        self.canCloneVoiceNow = canCloneVoiceNow
        self.installedTTSModels = installedTTSModels
        self.installedASRModels = installedASRModels
        self.cachedButNotRegistered = cachedButNotRegistered
        self.installedVoiceCount = installedVoiceCount
        self.daemonReachable = daemonReachable
        self.daemonReady = daemonReady
        self.metallibAvailable = metallibAvailable
        self.inferenceAssetIssue = inferenceAssetIssue
        self.missingPrerequisites = missingPrerequisites
    }
}

public struct ModelSharedCachePurgeDTO: Codable, Sendable, Equatable {
    public let model: String
    public let removedPaths: [String]
    public let removedCount: Int

    public init(
        model: String,
        removedPaths: [String]
    ) {
        self.model = model
        self.removedPaths = removedPaths
        self.removedCount = removedPaths.count
    }
}

private enum ValarDTOFormatting {
    static func iso8601String(from date: Date) -> String {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter.string(from: date)
    }
}
