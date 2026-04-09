import Foundation
import ValarModelKit
import ValarPersistence

public extension ValarRuntime {
    private struct DaemonWarmResidencyEvaluation: Sendable {
        let effectiveWarmStartModels: [String]
        let warmingModelIDs: [String]
        let idleResidentExtraModels: [String]
        let idleTrimEligibleModelIDs: [ModelIdentifier]
        let observation: (likelyDrift: Bool, message: String?)

        var idleTrimEligibleModels: [String] {
            idleTrimEligibleModelIDs.map(\.rawValue).sorted()
        }

        var effectiveWarmStartModelSet: Set<String> {
            Set(effectiveWarmStartModels)
        }

        var idleTrimEligibleModelSet: Set<String> {
            Set(idleTrimEligibleModels)
        }
    }

    func daemonHealthStatus() -> DaemonHealthStatusDTO {
        DaemonHealthStatusDTO(status: "ok")
    }

    func daemonReadyStatus() async -> DaemonReadyDTO {
        _ = await ensureStartupMaintenance()
        let snapshots = await modelRegistry.snapshots()
        let allModels: [CatalogModel]
        do {
            allModels = try await modelCatalog.refresh()
        } catch {
            allModels = (try? await modelCatalog.supportedModels()) ?? []
        }
        let installedModels = allModels.filter { $0.installState == .installed }
        let installedIDs = installedModels.map { $0.id.rawValue }.sorted()
        let cachedIDs = allModels
            .filter { $0.installState == .cached }
            .map { $0.id.rawValue }
            .sorted()
        let staleInstalledModels = allModels
            .filter { model in
                guard let status = model.installPathStatus else { return false }
                return !status.isValid
            }
            .map { $0.id.rawValue }
            .sorted()
        let resident = snapshots.filter { $0.state == .resident }
        let residentTTS = resident.contains { $0.descriptor.capabilities.contains(.speechSynthesis) }
        let residentASR = resident.contains { $0.descriptor.capabilities.contains(.speechRecognition) }
        let residentAlignment = resident.contains { $0.descriptor.capabilities.contains(.forcedAlignment) }
        let installedTTS = installedModels.contains { $0.descriptor.capabilities.contains(.speechSynthesis) }
        let installedASR = installedModels.contains { $0.descriptor.capabilities.contains(.speechRecognition) }
        let installedAlignment = installedModels.contains { $0.descriptor.capabilities.contains(.forcedAlignment) }
        let ttsReady = residentTTS || installedTTS
        let asrReady = residentASR || installedASR
        let alignmentReady = residentAlignment || installedAlignment
        let readinessMode: DaemonReadinessMode = if residentTTS || residentASR || residentAlignment {
            .resident
        } else if ttsReady || asrReady || alignmentReady {
            .loadOnDemand
        } else {
            .unavailable
        }
        let inferenceAssets = LocalInferenceAssetsStatus.currentProcess()
        let inferenceAssetsReady = inferenceAssets.metallibAvailable

        if ttsReady || asrReady || alignmentReady {
            return DaemonReadyDTO(
                ready: true,
                installedModels: installedIDs,
                staleInstalledModels: staleInstalledModels,
                cachedModels: cachedIDs,
                ttsReady: ttsReady,
                asrReady: asrReady,
                alignmentReady: alignmentReady,
                readinessMode: readinessMode,
                residentTTSReady: residentTTS,
                residentASRReady: residentASR,
                residentAlignmentReady: residentAlignment,
                inferenceAssetsReady: inferenceAssetsReady,
                inferenceAssetIssue: inferenceAssetsReady ? nil : inferenceAssets.failureReason
            )
        }

        let reason: String
        if !installedIDs.isEmpty {
            reason = "No models loaded in memory. \(installedIDs.count) model(s) installed on disk. " +
                "Restart the daemon, or issue the first synthesis/transcription request and let Valar load on demand."
        } else if !staleInstalledModels.isEmpty {
            reason = "No models loaded. \(staleInstalledModels.count) installed model record(s) point at missing or stale on-disk paths. " +
                "Reinstall or remove the stale records before relying on this daemon."
        } else if !cachedIDs.isEmpty {
            reason = "No models loaded. \(cachedIDs.count) model(s) cached on disk. " +
                "Run POST /v1/models/install to register them."
        } else {
            reason = "No models installed. Use POST /v1/models/install to install a model."
        }

        return DaemonReadyDTO(
            ready: false,
            reason: reason,
            installedModels: installedIDs,
            staleInstalledModels: staleInstalledModels,
            cachedModels: cachedIDs,
            ttsReady: false,
            asrReady: false,
            alignmentReady: false,
            readinessMode: readinessMode,
            residentTTSReady: residentTTS,
            residentASRReady: residentASR,
            residentAlignmentReady: residentAlignment,
            inferenceAssetsReady: inferenceAssetsReady,
            inferenceAssetIssue: inferenceAssetsReady ? nil : inferenceAssets.failureReason
        )
    }

    func daemonRuntimeStatus(startedAt: Date) async -> DaemonRuntimeStatusDTO {
        let startupMaintenance = await ensureStartupMaintenance()
        let snapshots = await modelRegistry.snapshots()
        let resident = snapshots.filter { $0.state == .resident }
        let totalBytes = await modelRegistry.residentBytes()
        let metalName: String? = nil
        let uptimeSeconds = max(0, -startedAt.timeIntervalSinceNow)
        let resourceSnapshot = await resourceMonitor.sample(
            applicationSupportURL: paths.applicationSupport,
            uptimeSeconds: uptimeSeconds
        )
        let synthesisSnapshot = await activeSynthesisTracker.snapshot()
        let now = Date()
        let activeRequests = synthesisSnapshot.activeRequests
        let warmResidency = await daemonWarmResidencyEvaluation(
            startedAt: startedAt,
            snapshots: snapshots,
            activeRequests: activeRequests,
            now: now
        )
        let oldestActiveAgeSeconds = activeRequests.first.map { now.timeIntervalSince($0.startedAt) }
        let stalledCount = activeRequests.filter { now.timeIntervalSince($0.lastHeartbeatAt) > 60 }.count
        let activeRequestDTOs = activeRequests.map { request in
            ActiveSynthesisRequestDTO(
                id: request.id.uuidString,
                modelID: request.modelID,
                voiceBehavior: request.voiceBehavior,
                executionMode: request.executionMode,
                segmentIndex: request.segmentIndex,
                segmentCount: request.segmentCount,
                startedAt: daemonRuntimeISO8601String(from: request.startedAt),
                lastHeartbeatAt: daemonRuntimeISO8601String(from: request.lastHeartbeatAt),
                ageSeconds: max(0, now.timeIntervalSince(request.startedAt)),
                lastHeartbeatAgeSeconds: max(0, now.timeIntervalSince(request.lastHeartbeatAt)),
                usesAnchorConditioning: request.usesAnchorConditioning,
                chunkCharacterCount: request.chunkCharacterCount,
                generatedTokenCount: request.generatedTokenCount,
                maxTokenCount: request.maxTokenCount,
                prefillTokenCount: request.prefillTokenCount,
                segmentPrefillTimeSeconds: request.segmentPrefillTimeSeconds,
                segmentDecodeTimeSeconds: request.segmentDecodeTimeSeconds,
                anchorSegmentDecodeTimeSeconds: request.anchorSegmentDecodeTimeSeconds,
                continuationSegmentDecodeTimeSeconds: request.continuationSegmentDecodeTimeSeconds,
                samplingTimeSeconds: request.samplingTimeSeconds,
                evalTimeSeconds: request.evalTimeSeconds,
                tokenMaterializationTimeSeconds: request.tokenMaterializationTimeSeconds,
                embeddingAssemblyTimeSeconds: request.embeddingAssemblyTimeSeconds,
                talkerForwardTimeSeconds: request.talkerForwardTimeSeconds,
                codePredictorTimeSeconds: request.codePredictorTimeSeconds,
                segmentWallTimeSeconds: request.segmentWallTimeSeconds,
                segmentAudioDurationSeconds: request.segmentAudioDurationSeconds,
                continuationOutlier: request.continuationOutlier
            )
        }
        let lastCompletionReason: String? = if let completion = synthesisSnapshot.lastCompletion {
            if let message = completion.message?.trimmingCharacters(in: .whitespacesAndNewlines),
               !message.isEmpty {
                "\(completion.terminalState.rawValue): \(message)"
            } else {
                completion.terminalState.rawValue
            }
        } else {
            nil
        }
        let warmStartModelSource: WarmStartModelSourceDTO = runtimeConfiguration.warmStartModelIDs == nil
            ? .default
            : .explicit
        let configuredWarmStartModels = runtimeConfiguration.warmStartModelIDs?.map(\.rawValue) ?? []
        let residentIDs = Set(resident.map(\.descriptor.id.rawValue))
        let prewarmedIDs = warmResidency.effectiveWarmStartModels.filter {
            residentIDs.contains($0) || Set(warmResidency.warmingModelIDs).contains($0)
        }
        let pidFileURL = paths.daemonPIDFileURL
        let pidFilePresent = FileManager.default.fileExists(atPath: pidFileURL.path)
        let pidFileContents = try? String(contentsOf: pidFileURL, encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let pidFileMatchesProcess = pidFileContents.flatMap(Int.init).map {
            $0 == Int(ProcessInfo.processInfo.processIdentifier)
        }
        let lastIdleTrimResult = await daemonIdleTrimTracker.snapshot().map(DaemonIdleTrimResultDTO.init(from:))
        let recentResidentEvictions = await modelRegistry.evictionEvents()
            .suffix(10)
            .map(ModelResidencyEvictionEventDTO.init(from:))
        let processFootprintOverBudgetBytes = max(0, resourceSnapshot.processFootprintBytes - runtimeConfiguration.maxResidentBytes)
        return DaemonRuntimeStatusDTO(
            processIdentifier: Int(ProcessInfo.processInfo.processIdentifier),
            daemonPIDFilePath: ValarPathRedaction.redact(pidFileURL),
            daemonPIDFilePresent: pidFilePresent,
            daemonPIDFileMatchesProcess: pidFileMatchesProcess,
            residentModels: resident.map {
                ModelResidencySnapshotDTO(
                    id: $0.descriptor.id.rawValue,
                    displayName: $0.descriptor.displayName,
                    domain: $0.descriptor.domain.rawValue,
                    state: $0.state.rawValue,
                    lastTouchedAt: $0.lastTouchedAt.map(daemonRuntimeISO8601String),
                    residentRank: $0.residentRank,
                    estimatedResidentBytes: $0.estimatedResidentBytes,
                    actualResidentBytes: $0.actualResidentBytes,
                    residencyPolicy: $0.residencyPolicy.rawValue,
                    activeSessionCount: $0.activeSessionCount,
                    isWarmStartModel: warmResidency.effectiveWarmStartModelSet.contains($0.descriptor.id.rawValue),
                    idleTrimEligible: warmResidency.idleTrimEligibleModelSet.contains($0.descriptor.id.rawValue)
                )
            },
            totalResidentBytes: totalBytes,
            memoryBudgetBytes: runtimeConfiguration.maxResidentBytes,
            warmPolicy: runtimeConfiguration.warmPolicy.rawValue,
            warmStartModelSource: warmStartModelSource,
            configuredWarmStartModels: configuredWarmStartModels,
            effectiveWarmStartModels: warmResidency.effectiveWarmStartModels,
            orphanedModelPackPaths: startupMaintenance.modelPackState.orphanedModelPackPaths.map(ValarPathRedaction.redact),
            idleResidentExtraModels: warmResidency.idleResidentExtraModels,
            idleTrimEligibleModels: warmResidency.idleTrimEligibleModels,
            idleResidentExtraModelsLikelyDrift: warmResidency.observation.likelyDrift,
            idleResidentExtraModelsAdvisory: warmResidency.observation.message,
            lastIdleTrimResult: lastIdleTrimResult,
            metalDeviceName: metalName,
            processFootprintBytes: resourceSnapshot.processFootprintBytes,
            processFootprintHighWaterBytes: resourceSnapshot.processFootprintHighWaterBytes,
            memoryBudgetBreached: processFootprintOverBudgetBytes > 0,
            processFootprintOverBudgetBytes: processFootprintOverBudgetBytes,
            processCPUPercent: resourceSnapshot.processCPUAveragePercentSinceStart,
            processCPUCurrentPercent: resourceSnapshot.processCPUCurrentPercent,
            processCPUCurrentHighWaterPercent: resourceSnapshot.processCPUCurrentHighWaterPercent,
            processCPUAveragePercentSinceStart: resourceSnapshot.processCPUAveragePercentSinceStart,
            processCPUAverageHighWaterPercentSinceStart: resourceSnapshot.processCPUAverageHighWaterPercentSinceStart,
            availableDiskBytes: resourceSnapshot.availableDiskBytes,
            availableDiskLowWaterBytes: resourceSnapshot.availableDiskLowWaterBytes,
            prewarmedModels: prewarmedIDs,
            warmingModels: warmResidency.warmingModelIDs,
            activeSynthesisCount: activeRequests.count,
            oldestActiveSynthesisAgeSeconds: oldestActiveAgeSeconds,
            stalledSynthesisCount: stalledCount,
            activeSynthesisRequests: activeRequestDTOs,
            lastSynthesisCompletionReason: lastCompletionReason,
            recentResidentEvictions: recentResidentEvictions,
            uptimeSeconds: uptimeSeconds
        )
    }

    func trimRouteRuntimeResidents(
        startedAt: Date,
        modelIDs: [String]? = nil,
        includeWarmStartModels: Bool = false
    ) async -> DaemonRuntimeTrimResultDTO {
        let snapshots = await modelRegistry.snapshots()
        let activeRequests = await activeSynthesisTracker.snapshot().activeRequests
        let evaluation = await daemonWarmResidencyEvaluation(
            startedAt: startedAt,
            snapshots: snapshots,
            activeRequests: activeRequests,
            now: .now
        )
        let residentSnapshots = snapshots.filter { $0.state == .resident }
        let residentModelIDs = residentSnapshots.map { $0.descriptor.id.rawValue }.sorted()
        let warmSet = evaluation.effectiveWarmStartModelSet
        let requestedSet = Set(modelIDs?.map {
            $0.trimmingCharacters(in: .whitespacesAndNewlines)
        }.filter { !$0.isEmpty } ?? [])

        let candidates: [ModelIdentifier]
        if requestedSet.isEmpty {
            candidates = includeWarmStartModels
                ? residentSnapshots
                    .filter { $0.activeSessionCount == 0 }
                    .map(\.descriptor.id)
                : residentSnapshots
                    .filter { snapshot in
                        snapshot.activeSessionCount == 0
                            && !warmSet.contains(snapshot.descriptor.id.rawValue)
                    }
                    .map(\.descriptor.id)
        } else {
            candidates = residentSnapshots.compactMap { snapshot in
                let identifier = snapshot.descriptor.id.rawValue
                guard requestedSet.contains(identifier) else { return nil }
                guard snapshot.activeSessionCount == 0 else { return nil }
                if !includeWarmStartModels, warmSet.contains(identifier) {
                    return nil
                }
                return snapshot.descriptor.id
            }
        }

        var trimmed: [String] = []
        for identifier in candidates {
            if await modelRegistry.evictResident(identifier, trigger: .idleTrim) {
                trimmed.append(identifier.rawValue)
            }
        }

        let skipped: [String]
        if requestedSet.isEmpty {
            skipped = residentModelIDs.filter { !trimmed.contains($0) }
        } else {
            skipped = requestedSet.subtracting(trimmed).sorted()
        }
        let resourceSnapshot = await resourceMonitor.sample(
            applicationSupportURL: paths.applicationSupport,
            uptimeSeconds: max(0, -startedAt.timeIntervalSinceNow)
        )
        let processFootprintOverBudgetBytes = max(0, resourceSnapshot.processFootprintBytes - runtimeConfiguration.maxResidentBytes)
        let reason: String
        if requestedSet.isEmpty {
            reason = includeWarmStartModels
                ? "Trimmed resident models regardless of warm-start policy."
                : "Trimmed resident non-warm models."
        } else {
            reason = "Processed explicit runtime trim request."
        }

        let remainingResidents = await modelRegistry.residentModels().map(\.id.rawValue).sorted()
        return DaemonRuntimeTrimResultDTO(
            trimmedModelIDs: trimmed.sorted(),
            skippedModelIDs: skipped,
            effectiveWarmStartModels: evaluation.effectiveWarmStartModels,
            residentModelIDs: remainingResidents,
            memoryBudgetBreached: processFootprintOverBudgetBytes > 0,
            processFootprintBytes: resourceSnapshot.processFootprintBytes,
            processFootprintOverBudgetBytes: processFootprintOverBudgetBytes,
            reason: reason
        )
    }

    func trimIdleNonWarmResidentsIfNeeded(startedAt: Date) async {
        let snapshots = await modelRegistry.snapshots()
        let activeRequests = await activeSynthesisTracker.snapshot().activeRequests
        let evaluation = await daemonWarmResidencyEvaluation(
            startedAt: startedAt,
            snapshots: snapshots,
            activeRequests: activeRequests,
            now: .now
        )
        guard !evaluation.idleTrimEligibleModelIDs.isEmpty else {
            return
        }

        var trimmed: [String] = []
        for identifier in evaluation.idleTrimEligibleModelIDs {
            if await modelRegistry.evictResident(identifier, trigger: .idleTrim) {
                trimmed.append(identifier.rawValue)
            }
        }

        guard !trimmed.isEmpty else {
            return
        }

        let displayLabel = trimmed.joined(separator: ", ")
        await daemonIdleTrimTracker.record(
            trimmedModelIDs: trimmed,
            reason: "Trimmed idle non-warm resident model(s) after the recent-use grace period: \(displayLabel)."
        )
    }

    func restoreMissingWarmResidentsIfNeeded() async {
        guard runtimeConfiguration.warmPolicy == .eager,
              inferenceBackend.runtimeCapabilities.features.contains(.warmStart) else {
            return
        }

        let activeRequests = await activeSynthesisTracker.snapshot().activeRequests
        guard activeRequests.isEmpty else {
            return
        }

        let warmModels = await warmStartCatalogModels()
        guard !warmModels.isEmpty else {
            return
        }

        let snapshots = await modelRegistry.snapshots()
        let residentIDs = Set(
            snapshots
                .filter { $0.state == .resident }
                .map(\.descriptor.id)
        )
        let warmingIDs = Set(
            snapshots
                .filter { $0.state == .warming }
                .map(\.descriptor.id)
        )
        let warmModelIDSet = Set(warmModels.map(\.id))
        let hasIdleExtraResidents = snapshots.contains { snapshot in
            snapshot.state == .resident && !warmModelIDSet.contains(snapshot.descriptor.id)
        }
        guard !hasIdleExtraResidents else {
            return
        }

        let missingWarmModels = warmModels.filter { model in
            !residentIDs.contains(model.id) && !warmingIDs.contains(model.id)
        }
        guard !missingWarmModels.isEmpty else {
            return
        }

        let policy = BackendSelectionPolicy()
        let backendRuntime = BackendSelectionPolicy.Runtime(
            availableBackends: [inferenceBackend.backendKind]
        )

        for model in missingWarmModels {
            guard let configuration = try? policy.runtimeConfiguration(
                for: model.descriptor,
                runtime: backendRuntime
            ) else {
                continue
            }
            do {
                let reserved = try await reserveModelWorkflowSession(
                    descriptor: model.descriptor,
                    configuration: configuration
                )
                await reserved.cleaner.finish()
            } catch {
                continue
            }
        }
    }

    func listRouteModels() async throws -> [ModelRouteDescriptorDTO] {
        _ = await ensureStartupMaintenance()
        return try await modelCatalog.refresh().map(ModelRouteDescriptorDTO.init(from:))
    }

    func installRouteModel(id: String, allowDownload: Bool = false, refreshCache: Bool = false) async throws {
        let identifier = ModelIdentifier(id)
        if let hiddenReason = CatalogVisibilityPolicy.currentProcess().hiddenReason(for: identifier) {
            throw RouteModelError.modelHidden(hiddenReason)
        }
        guard let entry = SupportedModelCatalog.entry(for: identifier) else {
            throw RouteModelError.modelNotFound(id)
        }

        let sourceKind: ModelPackSourceKind = entry.remoteURL == nil ? .localFile : .remoteURL
        let sourceLocation = entry.remoteURL?.absoluteString ?? "catalog:\(id)"
        let manifest = ModelCatalog.makePersistenceManifest(from: entry.manifest)
        let mode: ModelInstallMode = sourceKind == .remoteURL ? .downloadArtifacts : .metadataOnly

        if refreshCache {
            if sourceKind == .remoteURL, !allowDownload {
                throw RouteModelError.refreshRequiresDownload(id)
            }
            _ = try await modelInstaller.uninstall(modelID: identifier)
            _ = try await modelInstaller.purgeSharedCaches(for: identifier)
        }

        _ = try await modelInstaller.install(
            manifest: manifest,
            sourceKind: sourceKind,
            sourceLocation: sourceLocation,
            mode: mode
        )
        _ = try await modelCatalog.refresh()
    }

    func removeRouteModel(id: String) async throws {
        let identifier = ModelIdentifier(id)
        _ = try await modelInstaller.uninstall(modelID: identifier)
        _ = try await modelCatalog.refresh()
    }

    func listRouteVoices() async -> [VoiceLibraryRecord] {
        _ = await ensureStartupMaintenance()
        return await listVoices()
    }

    func createRouteVoice(name: String, description: String?) async throws -> VoiceLibraryRecord {
        try await createVoice(VoiceCreateRequest(label: name, voicePrompt: description))
    }

    static func warmStartCandidateIDs(
        from models: [CatalogModel],
        configuredModelIDs: [ModelIdentifier]? = nil
    ) -> [ModelIdentifier] {
        let installed = models.filter { $0.installState == .installed }
        guard !installed.isEmpty else { return [] }

        if let configuredModelIDs, !configuredModelIDs.isEmpty {
            let installedByID = Dictionary(uniqueKeysWithValues: installed.map { ($0.id, $0) })
            var selected: [ModelIdentifier] = []
            var selectedIDs: Set<ModelIdentifier> = []
            for identifier in configuredModelIDs {
                guard let resolved = installedByID[identifier]?.id else { continue }
                if selectedIDs.insert(resolved).inserted {
                    selected.append(resolved)
                }
            }
            return selected
        }

        let primaryTTS = installed.first(where: {
            $0.id == Self.defaultVoiceCloneRuntimeModelID &&
                $0.descriptor.capabilities.contains(.speechSynthesis)
        }) ?? installed.first(where: {
            $0.id == Self.defaultVoiceCreateModelID &&
                $0.descriptor.capabilities.contains(.speechSynthesis)
        }) ?? installed.first(where: {
            $0.familyID == .qwen3TTS && $0.descriptor.capabilities.contains(.speechSynthesis)
        }) ?? installed.first(where: {
            $0.descriptor.capabilities.contains(.speechSynthesis)
        })

        let asr = installed.first(where: {
            $0.familyID == .qwen3ASR && $0.descriptor.capabilities.contains(.speechRecognition)
        }) ?? installed.first(where: {
            $0.descriptor.capabilities.contains(.speechRecognition)
        })

        var selected: [ModelIdentifier] = []
        var selectedIDs: Set<ModelIdentifier> = []
        for candidate in [primaryTTS, asr].compactMap({ $0?.id }) {
            if selectedIDs.insert(candidate).inserted {
                selected.append(candidate)
            }
        }
        return selected
    }

    /// Returns the bounded warm-start set for this runtime.
    ///
    /// Preference order:
    /// - any explicit `warmStartModelIDs` in the runtime configuration
    /// - otherwise, the primary Qwen TTS model plus one ASR model if installed
    func warmStartCatalogModels() async -> [CatalogModel] {
        let supported: [CatalogModel]
        do {
            supported = try await modelCatalog.refresh()
        } catch {
            return []
        }

        let candidateIDs = Self.warmStartCandidateIDs(
            from: supported,
            configuredModelIDs: runtimeConfiguration.warmStartModelIDs
        )
        guard !candidateIDs.isEmpty else { return [] }

        let byID = Dictionary(uniqueKeysWithValues: supported.map { ($0.id, $0) })
        var selected: [CatalogModel] = []
        for identifier in candidateIDs {
            if let model = byID[identifier] {
                selected.append(model)
            }
        }
        return selected
    }

    /// Eagerly loads the bounded warm-start set through the shared model/session path.
    ///
    /// This keeps runtime residency reporting, doctor output, and backend state aligned:
    /// warm-started models are registered, marked resident, and immediately released back
    /// to an idle resident state instead of only being queued in the backend.
    func prewarmInstalledModels() async {
        guard inferenceBackend.runtimeCapabilities.features.contains(.warmStart) else { return }
        let warmModels = await warmStartCatalogModels()
        guard !warmModels.isEmpty else { return }

        let policy = BackendSelectionPolicy()
        let backendRuntime = BackendSelectionPolicy.Runtime(
            availableBackends: [inferenceBackend.backendKind]
        )
        for model in warmModels {
            guard let configuration = try? policy.runtimeConfiguration(
                for: model.descriptor,
                runtime: backendRuntime
            ) else {
                continue
            }
            do {
                let reserved = try await reserveModelWorkflowSession(
                    descriptor: model.descriptor,
                    configuration: configuration
                )
                await reserved.cleaner.finish()
            } catch {
                continue
            }
        }
    }

    private func daemonWarmResidencyEvaluation(
        startedAt: Date,
        snapshots: [ModelResidencySnapshot],
        activeRequests: [ActiveSynthesisRequestRecord],
        now: Date
    ) async -> DaemonWarmResidencyEvaluation {
        let resident = snapshots.filter { $0.state == .resident }
        let warmingModelIDs = snapshots
            .filter { $0.state == .warming }
            .map(\.descriptor.id.rawValue)
            .sorted()
        let effectiveWarmStartModels = await warmStartCatalogModels()
            .map(\.id.rawValue)
            .sorted()
        let warmModelIDSet = Set(effectiveWarmStartModels)
        let idleResidentExtraModels = activeRequests.isEmpty
            ? resident
                .map(\.descriptor.id.rawValue)
                .filter { !warmModelIDSet.contains($0) }
                .sorted()
            : []
        let uptimeSeconds = max(0, -startedAt.timeIntervalSinceNow)
        let observation = idleResidentExtraObservation(
            extraModelIDs: idleResidentExtraModels,
            residentSnapshots: resident,
            warmingModelIDs: Set(warmingModelIDs),
            uptimeSeconds: uptimeSeconds,
            now: now
        )
        let idleTrimEligibleModelIDs = resident.compactMap { snapshot -> ModelIdentifier? in
            guard activeRequests.isEmpty else { return nil }
            let identifier = snapshot.descriptor.id
            guard !warmModelIDSet.contains(identifier.rawValue),
                  snapshot.activeSessionCount == 0,
                  uptimeSeconds >= runtimeConfiguration.idleTrimSettleGraceSeconds,
                  warmingModelIDs.isEmpty else {
                return nil
            }
            guard let lastTouchedAt = snapshot.lastTouchedAt else {
                return identifier
            }
            return now.timeIntervalSince(lastTouchedAt) > runtimeConfiguration.idleTrimRecentUseGraceSeconds ? identifier : nil
        }
        return DaemonWarmResidencyEvaluation(
            effectiveWarmStartModels: effectiveWarmStartModels,
            warmingModelIDs: warmingModelIDs,
            idleResidentExtraModels: idleResidentExtraModels,
            idleTrimEligibleModelIDs: idleTrimEligibleModelIDs.sorted { $0.rawValue < $1.rawValue },
            observation: observation
        )
    }

    private func idleResidentExtraObservation(
        extraModelIDs: [String],
        residentSnapshots: [ModelResidencySnapshot],
        warmingModelIDs: Set<String>,
        uptimeSeconds: TimeInterval,
        now: Date
    ) -> (likelyDrift: Bool, message: String?) {
        guard !extraModelIDs.isEmpty else {
            return (false, nil)
        }

        let extraModelIDSet = Set(extraModelIDs)
        let extraSnapshots = residentSnapshots.filter { extraModelIDSet.contains($0.descriptor.id.rawValue) }
        let displayNames = extraSnapshots.map(\.descriptor.displayName).sorted()
        let displayLabel = displayNames.isEmpty ? extraModelIDs.joined(separator: ", ") : displayNames.joined(separator: ", ")

        if uptimeSeconds < runtimeConfiguration.idleTrimSettleGraceSeconds || !warmingModelIDs.isEmpty {
            return (
                false,
                "Daemon is still settling warm residency. Extra resident models (\(displayLabel)) may remain loaded briefly after startup, warmup, or recent on-demand use."
            )
        }

        let staleByAgeOrUnknownTouch = extraSnapshots.filter { snapshot in
            guard let lastTouchedAt = snapshot.lastTouchedAt else {
                return true
            }
            return now.timeIntervalSince(lastTouchedAt) > runtimeConfiguration.idleTrimRecentUseGraceSeconds
        }
        if staleByAgeOrUnknownTouch.isEmpty {
            return (
                false,
                "Daemon recently served on-demand model(s) that remain resident: \(displayLabel). This is advisory-only unless they continue to linger after a longer idle period."
            )
        }

        return (
            true,
            "Daemon has idle resident non-warm model(s) lingering beyond the recent-use grace period: \(displayLabel). This suggests warm-set drift or sticky on-demand residency."
        )
    }

    func capabilitySnapshot(
        daemonReachable: Bool,
        daemonReady: Bool,
        metallibAvailable: Bool
    ) async throws -> CapabilitySnapshotDTO {
        let catalog = try await modelCatalog.refresh()
        let installed = catalog.filter { $0.installState == .installed }
        let cached = catalog.filter { $0.installState == .cached }
        let voices = await listVoices()
        let visibilityPolicy = CatalogVisibilityPolicy.currentProcess()

        let ttsList = installed.filter { $0.descriptor.capabilities.contains(.speechSynthesis) }
        let asrList = installed.filter { $0.descriptor.capabilities.contains(.speechRecognition) }
        let alignList = installed.filter { $0.descriptor.capabilities.contains(.forcedAlignment) }
        let cloneList = installed.filter { $0.descriptor.capabilities.contains(.voiceCloning) }
        let hasVisibleTADA = catalog.contains { $0.familyID == .tadaTTS }
        let hasInstalledTADA = installed.contains { $0.familyID == .tadaTTS }

        var missing: [String] = []
        if ttsList.isEmpty { missing.append("Install a TTS model: run 'valartts models install <id>'") }
        if asrList.isEmpty { missing.append("Install an ASR model: run 'valartts models install <id>'") }
        if alignList.isEmpty { missing.append("Optional: Install alignment model for word-level timestamps: run 'valartts models install <id>'") }
        if hasVisibleTADA,
           !hasInstalledTADA,
           let tadaTokenizerIssue = ModelInstaller.tadaTokenizerInstallIssue() {
            missing.append(tadaTokenizerIssue)
        }
        if let hiddenVoxtralReason = visibilityPolicy.hiddenReason(for: VoxtralCatalog.modelIdentifier) {
            missing.append(hiddenVoxtralReason)
        }
        let inferenceAssets = LocalInferenceAssetsStatus.currentProcess()
        if !metallibAvailable { missing.append(inferenceAssets.failureReason) }
        if !daemonReachable { missing.append("Start the daemon: run 'valarttsd'") }
        return CapabilitySnapshotDTO(
            canSpeakNow: !ttsList.isEmpty && metallibAvailable,
            canTranscribeNow: !asrList.isEmpty && metallibAvailable,
            canAlignNow: !alignList.isEmpty && metallibAvailable,
            canCloneVoiceNow: !cloneList.isEmpty && metallibAvailable,
            installedTTSModels: ttsList.map(\.id.rawValue),
            installedASRModels: asrList.map(\.id.rawValue),
            cachedButNotRegistered: cached.map(\.id.rawValue),
            installedVoiceCount: voices.count,
            daemonReachable: daemonReachable,
            daemonReady: daemonReady,
            metallibAvailable: metallibAvailable,
            inferenceAssetIssue: metallibAvailable ? nil : inferenceAssets.failureReason,
            missingPrerequisites: missing
        )
    }

    func purgeRouteModelSharedCache(id: String) async throws -> ModelSharedCachePurgeDTO {
        let identifier = ModelIdentifier(id)
        if SupportedModelCatalog.entry(for: identifier) == nil,
           try await modelPackRegistry.installedRecord(for: id) == nil {
            throw RouteModelError.modelNotFound(id)
        }

        let removedPaths = try await modelInstaller.purgeSharedCaches(for: identifier).sorted()
        _ = try await modelCatalog.refresh()
        return ModelSharedCachePurgeDTO(model: id, removedPaths: removedPaths)
    }
}

private func daemonRuntimeISO8601String(from date: Date) -> String {
    let formatter = ISO8601DateFormatter()
    formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
    return formatter.string(from: date)
}

public enum RouteModelError: LocalizedError, Sendable, Equatable {
    case modelNotFound(String)
    case modelHidden(String)
    case refreshRequiresDownload(String)

    public var errorDescription: String? {
        switch self {
        case .modelNotFound(let id):
            return "Model '\(id)' is unavailable."
        case .modelHidden(let message):
            return message
        case .refreshRequiresDownload(let id):
            return "Refreshing shared cache for model '\(id)' requires network download. Retry with allow_download=true."
        }
    }
}
