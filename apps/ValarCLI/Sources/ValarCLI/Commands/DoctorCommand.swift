import ArgumentParser
import Darwin
import Foundation
import Metal
import ValarCore
import ValarModelKit
import ValarPersistence

struct DoctorCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "doctor",
        abstract: "Check system readiness for Valar."
    )

    mutating func run() async throws {
        let runtime = try ValarRuntime()
        var issues: [String] = []
        var advisories: [String] = []
        let daemonBaseURL = Self.daemonBaseURL()
        let modelPackAudit = try? await runtime.auditLocalModelPackState()
        let orphanedModelPackPaths = modelPackAudit?.orphanedModelPackPaths ?? []
        let daemonPIDStatus = daemonPIDStatus(paths: runtime.paths)

        // Architecture
        #if arch(arm64)
        let arch = "arm64 (Apple Silicon)"
        #else
        let arch = "x86_64 (Intel — MLX inference unavailable)"
        issues.append("Not running on Apple Silicon — MLX models require Apple Silicon hardware and will not work on Intel Macs. Valar requires an Apple Silicon Mac.")
        #endif

        // System memory
        let physicalMemoryMB = Int(ProcessInfo.processInfo.physicalMemory / (1024 * 1024))

        // Metal
        let metalDevice = MTLCreateSystemDefaultDevice()
        let metalDeviceName = metalDevice?.name
        let metalBudgetMB = metalDevice.map { Int($0.recommendedMaxWorkingSetSize / (1024 * 1024)) }
        if metalDeviceName == nil {
            issues.append("No Metal device found — MLX inference requires Metal GPU support and is unavailable on this system. Check that you are running on Apple Silicon.")
        }
        let localInferenceAssets = Self.localInferenceAssets()

        // Runtime configuration
        let maxResidentMB = runtime.runtimeConfiguration.maxResidentBytes / (1024 * 1024)
        let maxResidentModels = runtime.runtimeConfiguration.maxResidentModels

        // Models
        let catalog = try await runtime.modelCatalog.refresh()
        let supportedEntriesByModelID = Dictionary(
            uniqueKeysWithValues: SupportedModelCatalog.allSupportedEntries.map { ($0.id.rawValue, $0) }
        )
        let installReceipts = try await runtime.modelPackRegistry.receipts()
        let receiptByID = Dictionary(uniqueKeysWithValues: installReceipts.map { ($0.id, $0) })
        let installLedger = try await runtime.modelPackRegistry.ledgerEntries()
        let validInstallReceipts = installReceipts.filter {
            FileManager.default.fileExists(atPath: $0.installedModelPath)
        }
        let staleInstallReceipts = installReceipts.filter {
            !FileManager.default.fileExists(atPath: $0.installedModelPath)
        }
        let installedPathsByModelID = Dictionary(
            uniqueKeysWithValues: validInstallReceipts.map { ($0.modelID, $0.installedModelPath) }
        )
        let installedFamilyIDs = validInstallReceipts.map(\.familyID)
        let installed = catalog.filter { $0.installState == .installed }
        let cached = catalog.filter { $0.installState == .cached }
        let staleInstalledModels = (try? await runtime.modelCatalog.staleInstalledModels()) ?? []
        let staleInstalledModelIDs = staleInstalledModels.map(\.id.rawValue)
        let installedDetails: [DoctorInstalledModelDTO] = validInstallReceipts.map { receipt in
            let entry = supportedEntriesByModelID[receipt.modelID]
            return DoctorInstalledModelDTO(
                id: receipt.modelID,
                displayName: entry?.manifest.displayName ?? receipt.modelID,
                family: entry?.manifest.familyID.rawValue ?? receipt.familyID,
                domain: entry?.manifest.domain.rawValue ?? "unknown"
            )
        }.sorted { $0.displayName < $1.displayName }
        let cachedDetails: [DoctorInstalledModelDTO] = cached.map { model in
            DoctorInstalledModelDTO(
                id: model.id.rawValue,
                displayName: model.descriptor.displayName,
                family: model.descriptor.familyID.rawValue,
                domain: model.descriptor.domain.rawValue
            )
        }.sorted { $0.displayName < $1.displayName }
        let recentInstallEvents: [DoctorInstallEventDTO] = installLedger
            .suffix(20)
            .reversed()
            .map { entry in
                let receiptID = entry.receiptID
                let modelID = receiptID.flatMap { receiptByID[$0]?.modelID } ?? receiptID ?? "unknown"
                let supportedEntry = supportedEntriesByModelID[modelID]
                return DoctorInstallEventDTO(
                    modelID: modelID,
                    displayName: supportedEntry?.manifest.displayName ?? modelID,
                    family: supportedEntry?.manifest.familyID.rawValue
                        ?? receiptID.flatMap { receiptByID[$0]?.familyID }
                        ?? "unknown",
                    sourceKind: entry.sourceKind.rawValue,
                    recordedAt: ISO8601DateFormatter().string(from: entry.recordedAt),
                    succeeded: entry.succeeded,
                    message: entry.message ?? ""
                )
            }

        if !staleInstallReceipts.isEmpty {
            let staleModelList = staleInstallReceipts.map(\.modelID).sorted().joined(separator: ", ")
            issues.append(
                "Found \(staleInstallReceipts.count) stale installed model receipt(s) with missing ModelPacks: \(staleModelList). " +
                "Run 'valartts models status' to inspect current state, then reinstall with 'valartts models install <id>' or remove the stale record."
            )
        }
        if !staleInstalledModels.isEmpty {
            let staleModelList = staleInstalledModelIDs.sorted().joined(separator: ", ")
            issues.append(
                "Found \(staleInstalledModels.count) stale installed model record(s) with missing or broken ModelPacks: \(staleModelList). " +
                "Run 'valartts models status' to inspect current state, then reinstall with 'valartts models install <id>' or remove the stale record."
            )
        }
        if !orphanedModelPackPaths.isEmpty {
            issues.append(
                "Found \(orphanedModelPackPaths.count) orphaned ModelPack director\(orphanedModelPackPaths.count == 1 ? "y" : "ies") not registered in the install ledger. " +
                "Run 'valartts models cleanup --apply' to remove them."
            )
        }

        let hasTTSInstalled = validInstallReceipts.contains { receipt in
            supportedEntriesByModelID[receipt.modelID]?.manifest.capabilities.contains(.speechSynthesis) == true
        }
        if !hasTTSInstalled {
            issues.append("No TTS model installed. Run: valartts models install mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16")
        }

        let hasASRInstalled = validInstallReceipts.contains { receipt in
            supportedEntriesByModelID[receipt.modelID]?.manifest.capabilities.contains(.speechRecognition) == true
        }
        if !hasASRInstalled {
            issues.append("No ASR model installed. Run: valartts models install mlx-community/Qwen3-ASR-0.6B-8bit")
        }
        if (hasTTSInstalled || hasASRInstalled), !localInferenceAssets.metallibAvailable {
            issues.append(localInferenceAssets.failureReason)
        }

        let hasVisibleOrInstalledVoxtral = catalog.contains { $0.familyID == .voxtralTTS }
            || installedFamilyIDs.contains(ModelFamilyID.voxtralTTS.rawValue)
        if hasVisibleOrInstalledVoxtral, let toolingIssue = voxtralToolingIssue() {
            issues.append(toolingIssue)
        }
        let hasVisibleTADA = catalog.contains { $0.familyID == .tadaTTS }
        let hasInstalledTADA = installedFamilyIDs.contains(ModelFamilyID.tadaTTS.rawValue)
        if hasVisibleTADA,
           !hasInstalledTADA,
           let tadaTokenizerIssue = ModelInstaller.tadaTokenizerInstallIssue() {
            issues.append(tadaTokenizerIssue)
        }

        // Daemon health
        var daemonReachable = false
        if let url = daemonBaseURL?.appendingPathComponent("v1/health") {
            do {
                let (_, response) = try await URLSession.shared.data(from: url)
                if let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 {
                    daemonReachable = true
                }
            } catch {
                // Daemon not running
            }
        }

        // Daemon ready + runtime (only if reachable)
        var daemonReadyDTO: DaemonReadyDTO? = nil
        var daemonRuntimeDTO: DaemonRuntimeStatusDTO? = nil

        if daemonReachable {
            if let readyURL = daemonBaseURL?.appendingPathComponent("v1/ready") {
                if let (readyData, readyResponse) = try? await URLSession.shared.data(from: readyURL),
                   let httpResponse = readyResponse as? HTTPURLResponse {
                    if let dto = try? JSONDecoder().decode(DaemonReadyDTO.self, from: readyData) {
                        daemonReadyDTO = dto
                    } else {
                        daemonReadyDTO = DaemonReadyDTO(
                            ready: httpResponse.statusCode == 200,
                            reason: httpResponse.statusCode == 200 ? nil : "Daemon reported not ready.",
                            readinessMode: httpResponse.statusCode == 200 ? .loadOnDemand : .unavailable
                        )
                    }
                }
            }

            if let runtimeURL = daemonBaseURL?.appendingPathComponent("v1/runtime") {
                if let (runtimeData, runtimeResponse) = try? await URLSession.shared.data(from: runtimeURL),
                   let httpResponse = runtimeResponse as? HTTPURLResponse,
                   httpResponse.statusCode == 200 {
                    daemonRuntimeDTO = try? JSONDecoder().decode(DaemonRuntimeStatusDTO.self, from: runtimeData)
                }
            }
        }
        let reportReferenceTime = Date()
        if let status = daemonRuntimeDTO {
            let warmResidencyObservation = Self.idleResidentExtraObservation(
                from: status,
                referenceTime: reportReferenceTime
            )
            if status.daemonPIDFilePresent, status.daemonPIDFileMatchesProcess == false {
                issues.append(
                    "Daemon PID file does not match the running process. Check \(status.daemonPIDFilePath) and restart the daemon cleanly."
                )
            }
            if status.activeSynthesisCount == 0, !status.idleResidentExtraModels.isEmpty {
                advisories.append(
                    status.idleResidentExtraModelsAdvisory
                        ?? warmResidencyObservation.message
                        ?? "Daemon is idle but has resident non-warm models loaded: \(status.idleResidentExtraModels.joined(separator: ", "))."
                )
            }
            if !status.orphanedModelPackPaths.isEmpty {
                issues.append(
                    "Daemon startup detected \(status.orphanedModelPackPaths.count) orphaned ModelPack director\(status.orphanedModelPackPaths.count == 1 ? "y" : "ies"). Run 'valartts models cleanup --apply'."
                )
            }
        } else if daemonPIDStatus.present, daemonPIDStatus.live == false {
            issues.append(
                "Found a stale daemon PID file at \(daemonPIDStatus.path). Remove it or restart the daemon cleanly."
            )
        }

        // Storage
        let modelsDir = runtime.paths.modelPacksDirectory
        let projectsDir = runtime.paths.projectsDirectory
        let voiceLibraryDir = runtime.paths.voiceLibraryDirectory
        let cacheDir = runtime.paths.cacheDirectory
        let importsDir = runtime.paths.importsDirectory
        let snapshotsDir = runtime.paths.snapshotsDirectory
        let hfHubDir = Self.resolveHFHubCacheRoot()
        let hfMLXAudioDir = hfHubDir.appendingPathComponent("mlx-audio", isDirectory: true)
        let modelsSize = directorySize(modelsDir)
        let projectsSize = directorySize(projectsDir)
        let voiceLibrarySize = directorySize(voiceLibraryDir)
        let cacheSize = directorySize(cacheDir)
        let importsSize = directorySize(importsDir)
        let snapshotsSize = directorySize(snapshotsDir)
        let hfHubModelsSize = directorySizeOfChildren(in: hfHubDir, withPrefix: "models--")
        let hfMLXAudioSize = directorySize(hfMLXAudioDir)
        let sharedCacheAudit = analyzeSharedHFCache(
            modelIDs: Set(catalog.map { $0.id.rawValue })
                .union(SupportedModelCatalog.allSupportedEntries.map { $0.id.rawValue }),
            hfHubRoot: hfHubDir,
            hfMLXAudioRoot: hfMLXAudioDir,
            installedPathsByModelID: installedPathsByModelID,
            supportedEntriesByModelID: supportedEntriesByModelID
        )
        let duplicateSharedCacheModels = sharedCacheAudit.duplicates
        let legacyMLXAudioSafeToDelete = sharedCacheAudit.legacyOnlyModelIDs.isEmpty

        if OutputContext.jsonRequested {
            let residentDTOs: [DoctorResidentModelDTO]? = daemonRuntimeDTO.map { status in
                let extraModelIDs = Set(status.idleResidentExtraModels)
                return status.residentModels.map {
                    DoctorResidentModelDTO(
                        from: $0,
                        referenceTime: reportReferenceTime,
                        idleWarmSetExtraIDs: extraModelIDs
                    )
                }
            }
            let report = DoctorReportDTO(
                architecture: arch,
                applesilicon: arch.contains("arm64"),
                physicalMemoryMB: physicalMemoryMB,
                metalDeviceName: metalDeviceName,
                metalBudgetMB: metalBudgetMB,
                runtimeMaxResidentMB: maxResidentMB,
                runtimeMaxResidentModels: maxResidentModels,
                catalogModels: catalog.count,
                installedModels: validInstallReceipts.count,
                installedModelIDs: validInstallReceipts.map(\.modelID).sorted(),
                installedModelDetails: installedDetails,
                cachedModelIDs: cached.map { $0.id.rawValue },
                cachedModelDetails: cachedDetails,
                daemonReachable: daemonReachable,
                daemonReady: daemonReadyDTO?.ready,
                daemonReadyReason: daemonReadyDTO?.reason,
                daemonReadinessMode: daemonReadyDTO?.readinessMode?.rawValue,
                daemonResidentTTSReady: daemonReadyDTO?.residentTTSReady,
                daemonResidentASRReady: daemonReadyDTO?.residentASRReady,
                daemonResidentAlignmentReady: daemonReadyDTO?.residentAlignmentReady,
                daemonInferenceAssetsReady: daemonReadyDTO?.inferenceAssetsReady,
                localInferenceAssetsReady: localInferenceAssets.metallibAvailable,
                localInferenceAssetsFailureReason: localInferenceAssets.metallibAvailable ? nil : localInferenceAssets.failureReason,
                daemonResidentModels: residentDTOs,
                daemonPID: daemonRuntimeDTO?.processIdentifier ?? daemonPIDStatus.pid,
                daemonPIDFilePath: daemonRuntimeDTO?.daemonPIDFilePath ?? daemonPIDStatus.path,
                daemonPIDFilePresent: daemonRuntimeDTO?.daemonPIDFilePresent ?? daemonPIDStatus.present,
                daemonPIDFileMatchesProcess: daemonRuntimeDTO?.daemonPIDFileMatchesProcess ?? daemonPIDStatus.matchesProcess,
                daemonTotalResidentMB: daemonRuntimeDTO.map { $0.totalResidentBytes / (1024 * 1024) },
                daemonMemoryBudgetMB: daemonRuntimeDTO.map { $0.memoryBudgetBytes / (1024 * 1024) },
                daemonProcessFootprintMB: daemonRuntimeDTO.map { $0.processFootprintBytes / (1024 * 1024) },
                daemonProcessFootprintHighWaterMB: daemonRuntimeDTO.map { ($0.processFootprintHighWaterBytes ?? 0) / (1024 * 1024) },
                daemonProcessCPUPercent: daemonRuntimeDTO?.processCPUPercent,
                daemonProcessCPUCurrentPercent: daemonRuntimeDTO?.processCPUCurrentPercent,
                daemonProcessCPUCurrentHighWaterPercent: daemonRuntimeDTO?.processCPUCurrentHighWaterPercent,
                daemonProcessCPUAveragePercentSinceStart: daemonRuntimeDTO?.processCPUAveragePercentSinceStart,
                daemonProcessCPUAverageHighWaterPercentSinceStart: daemonRuntimeDTO?.processCPUAverageHighWaterPercentSinceStart,
                daemonUptimeSeconds: daemonRuntimeDTO?.uptimeSeconds,
                availableDiskMB: daemonRuntimeDTO.map { ($0.availableDiskBytes ?? 0) / (1024 * 1024) },
                availableDiskLowWaterMB: daemonRuntimeDTO.map { ($0.availableDiskLowWaterBytes ?? 0) / (1024 * 1024) },
                daemonWarmPolicy: daemonRuntimeDTO?.warmPolicy,
                daemonWarmStartModelSource: daemonRuntimeDTO?.warmStartModelSource,
                daemonConfiguredWarmStartModels: daemonRuntimeDTO?.configuredWarmStartModels ?? [],
                daemonEffectiveWarmStartModels: daemonRuntimeDTO?.effectiveWarmStartModels ?? [],
                daemonWarmingModels: daemonRuntimeDTO?.warmingModels ?? daemonRuntimeDTO?.prewarmedModels ?? [],
                daemonIdleResidentExtraModels: daemonRuntimeDTO?.idleResidentExtraModels ?? [],
                daemonIdleTrimEligibleModels: daemonRuntimeDTO?.idleTrimEligibleModels ?? [],
                daemonIdleResidentExtraModelsLikelyDrift: daemonRuntimeDTO.map {
                    $0.idleResidentExtraModelsLikelyDrift
                        ?? Self.idleResidentExtraObservation(from: $0, referenceTime: reportReferenceTime).likelyDrift
                },
                daemonIdleResidentExtraModelsAdvisory: daemonRuntimeDTO.flatMap { status in
                    if let advisory = status.idleResidentExtraModelsAdvisory {
                        advisory
                    } else {
                        Self.idleResidentExtraObservation(from: status, referenceTime: reportReferenceTime).message
                    }
                },
                daemonLastIdleTrimResult: daemonRuntimeDTO?.lastIdleTrimResult,
                activeSynthesisRequests: daemonRuntimeDTO?.activeSynthesisRequests ?? [],
                orphanedModelPackPaths: orphanedModelPackPaths,
                modelsPath: modelsDir.path,
                projectsPath: projectsDir.path,
                voiceLibraryPath: voiceLibraryDir.path,
                cachePath: cacheDir.path,
                importsPath: importsDir.path,
                snapshotsPath: snapshotsDir.path,
                huggingFaceHubPath: hfHubDir.path,
                huggingFaceMLXAudioPath: hfMLXAudioDir.path,
                modelsDiskMB: modelsSize,
                projectsDiskMB: projectsSize,
                voiceLibraryDiskMB: voiceLibrarySize,
                cacheDiskMB: cacheSize,
                importsDiskMB: importsSize,
                snapshotsDiskMB: snapshotsSize,
                huggingFaceHubModelsDiskMB: hfHubModelsSize,
                huggingFaceMLXAudioDiskMB: hfMLXAudioSize,
                legacyMLXAudioSafeToDelete: legacyMLXAudioSafeToDelete,
                duplicateSharedCacheModels: duplicateSharedCacheModels,
                installedBackedLegacyCacheModels: sharedCacheAudit.installedBackedLegacyModels,
                legacyOnlySharedCacheModelIDs: sharedCacheAudit.legacyOnlyModelIDs,
                staleInstalledModelIDs: staleInstalledModelIDs.sorted(),
                recentInstallEvents: recentInstallEvents,
                advisories: advisories,
                issues: issues
            )
            try OutputFormat.writeSuccess(command: OutputFormat.commandPath("doctor"), data: report)
        } else {
            print("Valar Doctor")
            print("============")
            print()

            // System
            print("System")
            print("  Architecture:    \(arch)")
            let ramGB = physicalMemoryMB >= 1024
                ? String(format: "%.0f GB", Double(physicalMemoryMB) / 1024)
                : "\(physicalMemoryMB) MB"
            print("  Physical RAM:    \(ramGB)")
            if let name = metalDeviceName {
                print("  Metal device:    \(name)")
            } else {
                print("  Metal device:    none")
            }
            if let budgetMB = metalBudgetMB {
                let budgetGB = budgetMB >= 1024
                    ? String(format: "%.0f GB", Double(budgetMB) / 1024)
                    : "\(budgetMB) MB"
                print("  Metal budget:    \(budgetGB)")
            }
            print()

            // Runtime
            print("Runtime")
            let maxMBStr = maxResidentMB >= 1024
                ? String(format: "%.0f GB", Double(maxResidentMB) / 1024)
                : "\(maxResidentMB) MB"
            print("  Max resident:    \(maxResidentModels) model(s) / \(maxMBStr)")
            print("  Local assets:    \(localInferenceAssets.metallibAvailable ? "ready" : "missing")")
            if !localInferenceAssets.metallibAvailable {
                print("  Asset note:      \(localInferenceAssets.failureReason)")
            }
            print()

            // Models
            print("Models (\(installedDetails.count) installed, \(catalog.count) visible in catalog)")
            if installed.isEmpty {
                print("  (none installed)")
            } else {
                for m in installedDetails {
                    print("  - \(m.displayName)  [\(m.family) / \(m.domain)]")
                }
            }
            if !cachedDetails.isEmpty {
                print("  Cached only:")
                for m in cachedDetails {
                    print("  - \(m.displayName)  [\(m.family) / \(m.domain)]")
                }
                print("  Register cached models with: valartts models install <id>")
            }
            if !staleInstalledModels.isEmpty {
                print("  Stale installed records:")
                for model in staleInstalledModels {
                    let status = model.installPathStatus?.rawValue ?? "unknown"
                    print("  - \(model.descriptor.displayName)  [\(model.familyID.rawValue) / \(model.descriptor.domain.rawValue)]")
                    print("      status: \(status)")
                    if let path = model.installedPath {
                        print("      path:   \(path)")
                    }
                }
                print("  Reinstall stale models with: valartts models install <id>")
            }
            print()

            // Daemon
            if daemonReachable {
                print("Daemon:          reachable (\(daemonBaseURL?.absoluteString ?? "http://127.0.0.1:8787"))")
                if let pid = daemonRuntimeDTO?.processIdentifier {
                    print("  PID:           \(pid)")
                }
                if let ready = daemonReadyDTO?.ready {
                    print("  Ready:         \(ready ? "yes" : "no")")
                }
                if let readinessMode = daemonReadyDTO?.readinessMode?.rawValue {
                    print("  Ready mode:    \(readinessMode)")
                }
                if let inferenceAssetsReady = daemonReadyDTO?.inferenceAssetsReady {
                    print("  Assets ready:  \(inferenceAssetsReady ? "yes" : "no")")
                }
                if let ready = daemonReadyDTO {
                    print("  TTS ready:     \(ready.ttsReady ? "yes" : "no")  (resident: \(ready.residentTTSReady ? "yes" : "no"))")
                    print("  ASR ready:     \(ready.asrReady ? "yes" : "no")  (resident: \(ready.residentASRReady ? "yes" : "no"))")
                    print("  Align ready:   \(ready.alignmentReady ? "yes" : "no")  (resident: \(ready.residentAlignmentReady ? "yes" : "no"))")
                    if let reason = ready.reason {
                        print("  Reason:        \(reason)")
                    }
                }
                if let status = daemonRuntimeDTO {
                    let totalMB = status.totalResidentBytes / (1024 * 1024)
                    let totalStr = totalMB >= 1024
                        ? String(format: "%.1f GB", Double(totalMB) / 1024)
                        : "\(totalMB) MB"
                    print("  Resident:      \(status.residentModels.count) model(s), \(totalStr) total")
                    let footprintMB = status.processFootprintBytes / (1024 * 1024)
                    let footprintStr = footprintMB >= 1024
                        ? String(format: "%.1f GB", Double(footprintMB) / 1024)
                        : "\(footprintMB) MB"
                    print("  RSS:           \(footprintStr)")
                    if let highWater = status.processFootprintHighWaterBytes {
                        let highWaterMB = highWater / (1024 * 1024)
                        let highWaterStr = highWaterMB >= 1024
                            ? String(format: "%.1f GB", Double(highWaterMB) / 1024)
                            : "\(highWaterMB) MB"
                        print("  RSS high:      \(highWaterStr)")
                    }
                    if let currentCPU = status.processCPUCurrentPercent {
                        print(String(format: "  CPU current:   %.1f%%", currentCPU))
                    }
                    if let currentHigh = status.processCPUCurrentHighWaterPercent {
                        print(String(format: "  CPU current hi: %.1f%%", currentHigh))
                    }
                    if let cpuPercent = status.processCPUAveragePercentSinceStart ?? status.processCPUPercent {
                        print(String(format: "  CPU avg:       %.1f%%", cpuPercent))
                    }
                    if let cpuHigh = status.processCPUAverageHighWaterPercentSinceStart {
                        print(String(format: "  CPU avg hi:    %.1f%%", cpuHigh))
                    }
                    for m in status.residentModels {
                        let bytesStr: String
                        let mb = (m.actualResidentBytes ?? m.estimatedResidentBytes) / (1024 * 1024)
                        bytesStr = mb >= 1024
                            ? String(format: "%.1f GB", Double(mb) / 1024)
                            : "\(mb) MB"
                        print("    - \(m.displayName) (\(m.state), \(bytesStr))")
                    }
                    let warmingModels = status.warmingModels.isEmpty ? status.prewarmedModels : status.warmingModels
                    if !warmingModels.isEmpty {
                        print("  Warming:       \(warmingModels.joined(separator: ", "))")
                    }
                    print("  Warm policy:   \(status.warmPolicy)")
                    let warmSourceLabel = status.warmStartModelSource == .explicit
                        ? "explicit"
                        : "default (implicit bounded daemon warm set)"
                    print("  Warm source:   \(warmSourceLabel)")
                    if status.warmStartModelSource == .default {
                        print("  Warm set:      implicit daemon default")
                    } else if !status.configuredWarmStartModels.isEmpty {
                        print("  Warm set:      \(status.configuredWarmStartModels.joined(separator: ", "))")
                    } else {
                        print("  Warm set:      (empty explicit set)")
                    }
                    if !status.effectiveWarmStartModels.isEmpty {
                        print("  Warm effective: \(status.effectiveWarmStartModels.joined(separator: ", "))")
                    }
                    if let availableDiskBytes = status.availableDiskBytes {
                        let availableGB = Double(availableDiskBytes) / 1_073_741_824
                        print("  Free disk:     \(String(format: "%.1f GB", availableGB))")
                    }
                    if let diskLowWater = status.availableDiskLowWaterBytes {
                        let availableGB = Double(diskLowWater) / 1_073_741_824
                        print("  Disk low:      \(String(format: "%.1f GB", availableGB))")
                    }
                    let uptimeStr = String(format: "%.1fs", status.uptimeSeconds)
                    print("  Uptime:        \(uptimeStr)")
                    if let metalName = status.metalDeviceName {
                        print("  Metal device:  \(metalName)")
                    }
                    if status.daemonPIDFilePresent {
                        let matchesLabel = status.daemonPIDFileMatchesProcess.map { $0 ? "yes" : "no" } ?? "unknown"
                        print("  PID file:      \(status.daemonPIDFilePath) (matches: \(matchesLabel))")
                    }
                    if !status.idleResidentExtraModels.isEmpty {
                        let warmResidencyObservation = Self.idleResidentExtraObservation(
                            from: status,
                            referenceTime: reportReferenceTime
                        )
                        let extrasLabel = (status.idleResidentExtraModelsLikelyDrift
                            ?? warmResidencyObservation.likelyDrift) ? "likely drift" : "advisory"
                        print("  Resident extras (\(extrasLabel)): \(status.idleResidentExtraModels.joined(separator: ", "))")
                        if let advisory = status.idleResidentExtraModelsAdvisory ?? warmResidencyObservation.message {
                            print("  Warm note:     \(advisory)")
                        }
                        for extra in Self.idleResidentExtraSnapshots(in: status) {
                            print(
                                "    - \(extra.displayName) (policy: \(extra.residencyPolicy), last touched: \(Self.lastTouchedAgeDescription(from: extra.lastTouchedAt, referenceTime: reportReferenceTime)))"
                            )
                        }
                    }
                    if !status.idleTrimEligibleModels.isEmpty {
                        print("  Trim eligible: \(status.idleTrimEligibleModels.joined(separator: ", "))")
                    }
                    if let lastTrim = status.lastIdleTrimResult {
                        let trimmed = lastTrim.trimmedModelIDs.isEmpty ? "-" : lastTrim.trimmedModelIDs.joined(separator: ", ")
                        print("  Last idle trim: \(trimmed) @ \(lastTrim.occurredAt)")
                        print("  Trim note:     \(lastTrim.reason)")
                    }
                    if !status.activeSynthesisRequests.isEmpty {
                        print("  Active requests:")
                        for request in status.activeSynthesisRequests {
                            print(
                                String(
                                    format: "    - %@ %@ %@ seg %d/%d age %.1fs heartbeat %.1fs",
                                    request.id,
                                    request.modelID,
                                    request.executionMode,
                                    request.segmentIndex,
                                    request.segmentCount,
                                    request.ageSeconds,
                                    request.lastHeartbeatAgeSeconds
                                )
                            )
                        }
                    }
                }
            } else {
                print("Daemon:          not running")
                if daemonPIDStatus.present {
                    let liveLabel = daemonPIDStatus.live.map { $0 ? "yes" : "no" } ?? "unknown"
                    print("  PID file:      \(daemonPIDStatus.path) (present, process alive: \(liveLabel))")
                }
            }
            print()

            // Storage
            print("Storage")
            print("  Models path:     \(modelsDir.path) (\(modelsSize) MB)")
            print("  Voice library:   \(voiceLibraryDir.path) (\(voiceLibrarySize) MB)")
            print("  App cache:       \(cacheDir.path) (\(cacheSize) MB)")
            print("  Imports:         \(importsDir.path) (\(importsSize) MB)")
            print("  Snapshots:       \(snapshotsDir.path) (\(snapshotsSize) MB)")
            print("  Projects path:   \(projectsDir.path) (\(projectsSize) MB)")
            print("  HF hub models:   \(hfHubDir.path) (\(hfHubModelsSize) MB)")
            print("  HF mlx-audio:    \(hfMLXAudioDir.path) (\(hfMLXAudioSize) MB)")
            print("  Note: HF paths are shared caches and may include non-Valar artifacts.")
            print("  New Valar installs register into ModelPacks and should not create fresh mlx-audio duplicates.")
            print("  Per-model cleanup: valartts models purge-cache <id>")
            print("  Note: reported directory sizes use du-style physical disk usage when available, so APFS clones and hard links are less misleading.")
            print("  Note: HF + ModelPacks together can still overstate total net disk usage when the same bytes are shared across buckets.")
            if !recentInstallEvents.isEmpty {
                print("  Recent install activity:")
                for event in recentInstallEvents.prefix(10) {
                    let statusLabel = event.succeeded ? "ok" : "failed"
                    print("    - [\(statusLabel)] \(event.recordedAt) \(event.displayName)")
                }
            }
            if !duplicateSharedCacheModels.isEmpty {
                print("  Duplicate shared cache:")
                for duplicate in duplicateSharedCacheModels {
                    print("    - \(duplicate.id)")
                    print("      preferred: \(duplicate.preferredPath) (\(duplicate.preferredSizeMB) MB)")
                    print("      legacy:    \(duplicate.legacyPath) (\(duplicate.legacySizeMB) MB)")
                    switch duplicate.installedPackBacking {
                    case "preferred":
                        print("      installed pack shares with preferred snapshot; remove the legacy copy first.")
                    case "legacy":
                        print("      installed pack shares with legacy mlx-audio; remove the standard HF snapshot first.")
                    default:
                        break
                    }
                }
                print("  Safe cleanup: use the per-model note above when present; otherwise prefer the standard HF snapshot path and remove the legacy mlx-audio copy.")
            }
            if !sharedCacheAudit.installedBackedLegacyModels.isEmpty {
                print("  Legacy cache backed by installed packs:")
                for legacy in sharedCacheAudit.installedBackedLegacyModels {
                    print("    - \(legacy.id)")
                    print("      installed: \(legacy.installedPath)")
                    print("      legacy:    \(legacy.legacyPath) (\(legacy.legacySizeMB) MB)")
                }
                print("  These legacy mlx-audio entries are not required for current Valar runtime use. Removing them mostly removes alternate cache paths; actual space reclaimed depends on whether the files are hard-linked into ModelPacks.")
            }
            if !sharedCacheAudit.legacyOnlyModelIDs.isEmpty {
                print("  Legacy-only cache:")
                for modelID in sharedCacheAudit.legacyOnlyModelIDs {
                    print("    - \(modelID)")
                }
                print("  Do not delete the entire mlx-audio cache blindly; these entries still exist only in the legacy layout.")
            } else if legacyMLXAudioSafeToDelete {
                print("  Legacy mlx-audio cache safe to delete for current Valar-managed workflows: yes")
                print("  Removing ~/.cache/huggingface/hub/mlx-audio will not break the models Valar currently sees as installed or cached.")
                print("  It may still remove cache entries used by older upstream mlx-audio tooling outside Valar.")
            }
            if !orphanedModelPackPaths.isEmpty {
                print("  Orphaned ModelPacks:")
                for path in orphanedModelPackPaths {
                    print("    - \(path)")
                }
                print("  Cleanup: valartts models cleanup --apply")
            }

            if !issues.isEmpty {
                print()
                print("Issues:")
                for issue in issues {
                    print("  ! \(issue)")
                }
            } else if !advisories.isEmpty {
                print()
                print("No issues found.")
            } else {
                print()
                print("No issues found.")
            }
            if !advisories.isEmpty {
                print()
                print("Advisories:")
                for advisory in advisories {
                    print("  - \(advisory)")
                }
            }
        }

        if !issues.isEmpty {
            throw ExitCode(1)
        }
    }

    private static func localInferenceAssets() -> LocalInferenceAssetsStatus {
        LocalInferenceAssetsStatus.currentProcess()
    }

    private static func daemonBaseURL(environment: [String: String] = ProcessInfo.processInfo.environment) -> URL? {
        let host = environment["VALARTTSD_BIND_HOST"]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .nonEmpty ?? "127.0.0.1"
        let port = environment["VALARTTSD_BIND_PORT"]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .nonEmpty ?? "8787"
        return URL(string: "http://\(host):\(port)")
    }

    private func daemonPIDStatus(paths: ValarAppPaths) -> (path: String, present: Bool, pid: Int?, live: Bool?, matchesProcess: Bool?) {
        let pidFileURL = paths.daemonPIDFileURL
        let present = FileManager.default.fileExists(atPath: pidFileURL.path)
        guard present,
              let raw = try? String(contentsOf: pidFileURL, encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines),
              let pid = Int(raw) else {
            return (pidFileURL.path, present, nil, nil, nil)
        }
        let live = kill(Int32(pid), 0) == 0
        return (pidFileURL.path, true, pid, live, nil)
    }

    fileprivate static func idleResidentExtraSnapshots(
        in status: DaemonRuntimeStatusDTO
    ) -> [ModelResidencySnapshotDTO] {
        let extraModelIDs = Set(status.idleResidentExtraModels)
        return status.residentModels
            .filter { extraModelIDs.contains($0.id) }
            .sorted { $0.displayName.localizedCaseInsensitiveCompare($1.displayName) == .orderedAscending }
    }

    fileprivate static func idleResidentExtraObservation(
        from status: DaemonRuntimeStatusDTO,
        referenceTime: Date
    ) -> (likelyDrift: Bool, message: String?) {
        guard !status.idleResidentExtraModels.isEmpty else {
            return (false, nil)
        }

        let extraSnapshots = idleResidentExtraSnapshots(in: status)
        let displayNames = extraSnapshots.map(\.displayName)
        let displayLabel = displayNames.isEmpty
            ? status.idleResidentExtraModels.joined(separator: ", ")
            : displayNames.joined(separator: ", ")

        if status.uptimeSeconds < 300 || !status.warmingModels.isEmpty {
            return (
                false,
                "Daemon is still settling warm residency. Extra resident models (\(displayLabel)) may remain loaded briefly after startup, warmup, or recent on-demand use."
            )
        }

        let staleByAgeOrUnknownTouch = extraSnapshots.filter {
            guard let ageSeconds = lastTouchedAgeSeconds(from: $0.lastTouchedAt, referenceTime: referenceTime) else {
                return true
            }
            return ageSeconds > 600
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

    fileprivate static func lastTouchedAgeSeconds(
        from value: String?,
        referenceTime: Date
    ) -> Double? {
        guard let value else {
            return nil
        }
        let fractionalFormatter = ISO8601DateFormatter()
        fractionalFormatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        let fallbackFormatter = ISO8601DateFormatter()
        guard let timestamp = fractionalFormatter.date(from: value) ?? fallbackFormatter.date(from: value) else {
            return nil
        }
        return max(0, referenceTime.timeIntervalSince(timestamp))
    }

    fileprivate static func lastTouchedAgeDescription(
        from value: String?,
        referenceTime: Date
    ) -> String {
        guard let ageSeconds = lastTouchedAgeSeconds(from: value, referenceTime: referenceTime) else {
            return "unknown"
        }
        if ageSeconds < 60 {
            return "\(Int(ageSeconds.rounded()))s ago"
        }
        if ageSeconds < 3_600 {
            return "\(Int((ageSeconds / 60).rounded()))m ago"
        }
        return String(format: "%.1fh ago", ageSeconds / 3_600)
    }

    private func directorySize(_ url: URL) -> Int {
        sizeOfTreesOnDisk([url])
    }

    private func directorySizeOfChildren(in url: URL, withPrefix prefix: String) -> Int {
        guard let children = try? FileManager.default.contentsOfDirectory(
            at: url,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        ) else {
            return 0
        }

        return sizeOfTreesOnDisk(children.filter { $0.lastPathComponent.hasPrefix(prefix) })
    }

    private func sizeOfTreesOnDisk(_ roots: [URL]) -> Int {
        if let physicalKilobytes = diskUsageKilobytes(for: roots) {
            return Int((physicalKilobytes + 1023) / 1024)
        }
        return sizeOfTreesDeduplicatingHardLinks(roots)
    }

    private func diskUsageKilobytes(for roots: [URL]) -> Int64? {
        let existingRoots = roots.filter { FileManager.default.fileExists(atPath: $0.path) }
        guard !existingRoots.isEmpty else {
            return 0
        }

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/du")
        process.arguments = ["-sk"] + existingRoots.map(\.path)
        let outputPipe = Pipe()
        process.standardOutput = outputPipe
        process.standardError = Pipe()

        do {
            try process.run()
            process.waitUntilExit()
        } catch {
            return nil
        }

        guard process.terminationStatus == 0 else {
            return nil
        }

        let outputData = outputPipe.fileHandleForReading.readDataToEndOfFile()
        guard let output = String(data: outputData, encoding: .utf8) else {
            return nil
        }

        return output
            .split(whereSeparator: \.isNewline)
            .reduce(into: Int64.zero) { total, line in
                guard let prefix = line.split(maxSplits: 1, whereSeparator: \.isWhitespace).first,
                      let kilobytes = Int64(prefix) else {
                    return
                }

                total += kilobytes
            }
    }

    private func sizeOfTreesDeduplicatingHardLinks(_ roots: [URL]) -> Int {
        var visited: Set<FileIdentity> = []
        let totalBytes = roots.reduce(into: Int64.zero) { total, root in
            total += uniqueFileBytes(at: root, visited: &visited)
        }
        return Int(totalBytes / (1024 * 1024))
    }

    private func uniqueFileBytes(at root: URL, visited: inout Set<FileIdentity>) -> Int64 {
        if let fileBytes = regularFileBytes(at: root, visited: &visited) {
            return fileBytes
        }
        guard let enumerator = FileManager.default.enumerator(
            at: root,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        ) else {
            return 0
        }

        var totalBytes: Int64 = 0
        for case let fileURL as URL in enumerator {
            totalBytes += regularFileBytes(at: fileURL, visited: &visited) ?? 0
        }
        return totalBytes
    }

    private func regularFileBytes(at url: URL, visited: inout Set<FileIdentity>) -> Int64? {
        guard let status = fileStatus(at: url) else {
            return nil
        }
        guard (status.st_mode & S_IFMT) == S_IFREG else {
            return nil
        }

        let identity = FileIdentity(
            device: numericCast(status.st_dev),
            inode: numericCast(status.st_ino)
        )
        guard visited.insert(identity).inserted else {
            return 0
        }
        return Int64(status.st_size)
    }

    private func fileStatus(at url: URL) -> stat? {
        var status = stat()
        let result = url.withUnsafeFileSystemRepresentation { fileSystemPath -> Int32 in
            guard let fileSystemPath else { return -1 }
            return lstat(fileSystemPath, &status)
        }
        return result == 0 ? status : nil
    }

    private func analyzeSharedHFCache(
        modelIDs: Set<String>,
        hfHubRoot: URL,
        hfMLXAudioRoot: URL,
        installedPathsByModelID: [String: String],
        supportedEntriesByModelID: [String: SupportedModelCatalogEntry],
        fileManager: FileManager = .default
    ) -> DoctorSharedCacheAudit {
        let substantialSizeThresholdMB = 100
        var duplicates: [DoctorSharedCacheDuplicateDTO] = []
        var installedBackedLegacyModels: [DoctorInstalledBackedLegacyCacheDTO] = []
        var legacyOnlyModelIDs: [String] = []

        for modelID in modelIDs {
            let preferredDirectory = hfHubRoot.appendingPathComponent(Self.hfHubRepoDirectoryName(for: modelID), isDirectory: true)
            let legacyDirectory = hfMLXAudioRoot.appendingPathComponent(Self.hfMLXAudioDirectoryName(for: modelID), isDirectory: true)

            let hasPreferred = fileManager.fileExists(atPath: preferredDirectory.path)
            let hasLegacy = fileManager.fileExists(atPath: legacyDirectory.path)
            guard hasPreferred || hasLegacy else { continue }

            let preferredSizeMB = hasPreferred ? directorySize(preferredDirectory) : 0
            let legacySizeMB = hasLegacy ? directorySize(legacyDirectory) : 0
            let hasSubstantialPreferred = hasPreferred && preferredSizeMB >= substantialSizeThresholdMB
            let hasSubstantialLegacy = hasLegacy && legacySizeMB >= substantialSizeThresholdMB

            if hasSubstantialPreferred && hasSubstantialLegacy {
                let installedPackBacking = installedPathsByModelID[modelID].flatMap { installedPath in
                    detectInstalledPackBacking(
                        installedPath: installedPath,
                        preferredDirectory: preferredDirectory,
                        legacyDirectory: legacyDirectory,
                        entry: supportedEntriesByModelID[modelID],
                        fileManager: fileManager
                    )
                }
                duplicates.append(
                    DoctorSharedCacheDuplicateDTO(
                        id: modelID,
                        preferredPath: preferredDirectory.path,
                        preferredSizeMB: preferredSizeMB,
                        legacyPath: legacyDirectory.path,
                        legacySizeMB: legacySizeMB,
                        installedPackBacking: installedPackBacking?.rawValue
                    )
                )
            } else if !hasSubstantialPreferred && hasSubstantialLegacy {
                if let installedPath = installedPathsByModelID[modelID] {
                    installedBackedLegacyModels.append(
                        DoctorInstalledBackedLegacyCacheDTO(
                            id: modelID,
                            installedPath: installedPath,
                            legacyPath: legacyDirectory.path,
                            legacySizeMB: legacySizeMB
                        )
                    )
                } else {
                    legacyOnlyModelIDs.append(modelID)
                }
            }
        }

        return DoctorSharedCacheAudit(
            duplicates: duplicates.sorted { $0.id < $1.id },
            installedBackedLegacyModels: installedBackedLegacyModels.sorted { $0.id < $1.id },
            legacyOnlyModelIDs: legacyOnlyModelIDs.sorted()
        )
    }

    private func detectInstalledPackBacking(
        installedPath: String,
        preferredDirectory: URL,
        legacyDirectory: URL,
        entry: SupportedModelCatalogEntry?,
        fileManager: FileManager = .default
    ) -> DoctorInstalledPackBacking? {
        guard let entry else { return nil }
        let installedRoot = URL(fileURLWithPath: installedPath, isDirectory: true)
        guard fileManager.fileExists(atPath: installedRoot.path) else { return nil }

        for artifact in entry.manifest.artifacts where !artifact.relativePath.hasSuffix("/") {
            let installedArtifact = installedRoot.appendingPathComponent(artifact.relativePath, isDirectory: false)
            guard fileManager.fileExists(atPath: installedArtifact.path) else { continue }

            let preferredArtifact = preferredDirectory.appendingPathComponent(artifact.relativePath, isDirectory: false)
            if fileManager.fileExists(atPath: preferredArtifact.path),
               sameFile(installedArtifact, preferredArtifact) {
                return .preferred
            }

            if let legacyArtifact = legacyArtifactURL(root: legacyDirectory, relativePath: artifact.relativePath, fileManager: fileManager),
               sameFile(installedArtifact, legacyArtifact) {
                return .legacy
            }
        }

        return nil
    }

    private func legacyArtifactURL(
        root: URL,
        relativePath: String,
        fileManager: FileManager = .default
    ) -> URL? {
        let nested = root.appendingPathComponent(relativePath, isDirectory: false)
        if fileManager.fileExists(atPath: nested.path) {
            return nested
        }

        let basename = URL(fileURLWithPath: relativePath).lastPathComponent
        let flat = root.appendingPathComponent(basename, isDirectory: false)
        return fileManager.fileExists(atPath: flat.path) ? flat : nil
    }

    private func sameFile(_ lhs: URL, _ rhs: URL) -> Bool {
        guard
            let lhsValues = try? lhs.resourceValues(forKeys: [.fileResourceIdentifierKey]),
            let rhsValues = try? rhs.resourceValues(forKeys: [.fileResourceIdentifierKey]),
            let lhsIdentifier = lhsValues.fileResourceIdentifier,
            let rhsIdentifier = rhsValues.fileResourceIdentifier
        else {
            return false
        }
        return (lhsIdentifier as AnyObject).isEqual(rhsIdentifier)
    }

    private static func resolveHFHubCacheRoot(
        environment: [String: String] = ProcessInfo.processInfo.environment,
        fileManager: FileManager = .default
    ) -> URL {
        if let explicit = environment["HF_HUB_CACHE"]?.trimmingCharacters(in: .whitespacesAndNewlines),
           !explicit.isEmpty {
            return URL(fileURLWithPath: explicit, isDirectory: true)
        }
        if let explicit = environment["HUGGINGFACE_HUB_CACHE"]?.trimmingCharacters(in: .whitespacesAndNewlines),
           !explicit.isEmpty {
            return URL(fileURLWithPath: explicit, isDirectory: true)
        }
        if let hfHome = environment["HF_HOME"]?.trimmingCharacters(in: .whitespacesAndNewlines),
           !hfHome.isEmpty {
            return URL(fileURLWithPath: hfHome, isDirectory: true)
                .appendingPathComponent("hub", isDirectory: true)
        }

        return fileManager.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/hub", isDirectory: true)
    }

    private func voxtralToolingIssue(fileManager: FileManager = .default) -> String? {
        guard let repoRoot = Self.resolveRepoRoot(fileManager: fileManager) else {
            return "Cannot locate the Valar repository root. Set VALARTTS_REPO_ROOT to the repository directory and retry."
        }
        let managedPython = repoRoot.appendingPathComponent("scripts/voxtral/.venv/bin/python3")
        let bootstrapScript = repoRoot.appendingPathComponent("scripts/voxtral/bootstrap_env.sh")
        if !fileManager.isExecutableFile(atPath: managedPython.path) {
            return "Voxtral preset voice installs require the managed toolchain. Run: bash \(bootstrapScript.path)"
        }
        let process = Process()
        process.executableURL = managedPython
        process.arguments = ["-c", "import torch"]
        do {
            try process.run()
            process.waitUntilExit()
        } catch {
            return "Voxtral managed tooling failed to launch. Run: bash \(bootstrapScript.path)"
        }
        if process.terminationStatus != 0 {
            return "Voxtral managed tooling is missing torch or is otherwise broken. Run: bash \(bootstrapScript.path)"
        }
        return nil
    }

    /// Resolves the Valar repository root at runtime.
    ///
    /// Resolution order:
    /// 1. `VALARTTS_REPO_ROOT` environment variable (explicit override).
    /// 2. Walk up from the running executable's directory looking for `.git`
    ///    (works when the binary lives inside a repository build tree, e.g. during `swift run`).
    /// 3. Walk up from the current working directory as a last resort.
    private static func resolveRepoRoot(
        fileManager: FileManager = .default,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> URL? {
        if let explicit = environment["VALARTTS_REPO_ROOT"]?.trimmingCharacters(in: .whitespacesAndNewlines),
           !explicit.isEmpty {
            let url = URL(fileURLWithPath: explicit, isDirectory: true)
            if fileManager.fileExists(atPath: url.appendingPathComponent(".git").path) {
                return url
            }
        }

        let startCandidates: [URL] = [
            Bundle.main.executableURL.map { $0.deletingLastPathComponent() },
            CommandLine.arguments.first.map { URL(fileURLWithPath: $0).deletingLastPathComponent() },
            URL(fileURLWithPath: fileManager.currentDirectoryPath, isDirectory: true),
        ].compactMap { $0 }

        for start in startCandidates {
            var candidate = start
            for _ in 0 ..< 10 {
                if fileManager.fileExists(atPath: candidate.appendingPathComponent(".git").path) {
                    return candidate
                }
                let parent = candidate.deletingLastPathComponent()
                if parent.path == candidate.path { break }
                candidate = parent
            }
        }
        return nil
    }

    private static func hfHubRepoDirectoryName(for modelID: String) -> String {
        "models--" + modelID.replacingOccurrences(of: "/", with: "--")
    }

    private static func hfMLXAudioDirectoryName(for modelID: String) -> String {
        modelID.replacingOccurrences(of: "/", with: "_")
    }
}

private extension String {
    var nonEmpty: String? {
        isEmpty ? nil : self
    }
}

private struct FileIdentity: Hashable {
    let device: UInt64
    let inode: UInt64
}

struct DoctorReportDTO: Codable, Sendable {
    let architecture: String
    let applesilicon: Bool
    let physicalMemoryMB: Int
    let metalDeviceName: String?
    let metalBudgetMB: Int?
    let runtimeMaxResidentMB: Int
    let runtimeMaxResidentModels: Int
    let catalogModels: Int
    let installedModels: Int
    let installedModelIDs: [String]
    let installedModelDetails: [DoctorInstalledModelDTO]
    let cachedModelIDs: [String]
    let cachedModelDetails: [DoctorInstalledModelDTO]
    let daemonReachable: Bool
    let daemonReady: Bool?
    let daemonReadyReason: String?
    let daemonReadinessMode: String?
    let daemonResidentTTSReady: Bool?
    let daemonResidentASRReady: Bool?
    let daemonResidentAlignmentReady: Bool?
    let daemonInferenceAssetsReady: Bool?
    let localInferenceAssetsReady: Bool
    let localInferenceAssetsFailureReason: String?
    let daemonResidentModels: [DoctorResidentModelDTO]?
    let daemonPID: Int?
    let daemonPIDFilePath: String
    let daemonPIDFilePresent: Bool
    let daemonPIDFileMatchesProcess: Bool?
    let daemonTotalResidentMB: Int?
    let daemonMemoryBudgetMB: Int?
    let daemonProcessFootprintMB: Int?
    let daemonProcessFootprintHighWaterMB: Int?
    let daemonProcessCPUPercent: Double?
    let daemonProcessCPUCurrentPercent: Double?
    let daemonProcessCPUCurrentHighWaterPercent: Double?
    let daemonProcessCPUAveragePercentSinceStart: Double?
    let daemonProcessCPUAverageHighWaterPercentSinceStart: Double?
    let daemonUptimeSeconds: Double?
    let availableDiskMB: Int?
    let availableDiskLowWaterMB: Int?
    let daemonWarmPolicy: String?
    let daemonWarmStartModelSource: WarmStartModelSourceDTO?
    let daemonConfiguredWarmStartModels: [String]
    let daemonEffectiveWarmStartModels: [String]
    let daemonWarmingModels: [String]
    let daemonIdleResidentExtraModels: [String]
    let daemonIdleTrimEligibleModels: [String]
    let daemonIdleResidentExtraModelsLikelyDrift: Bool?
    let daemonIdleResidentExtraModelsAdvisory: String?
    let daemonLastIdleTrimResult: DaemonIdleTrimResultDTO?
    let activeSynthesisRequests: [ActiveSynthesisRequestDTO]
    let orphanedModelPackPaths: [String]
    let modelsPath: String
    let projectsPath: String
    let voiceLibraryPath: String
    let cachePath: String
    let importsPath: String
    let snapshotsPath: String
    let huggingFaceHubPath: String
    let huggingFaceMLXAudioPath: String
    let modelsDiskMB: Int
    let projectsDiskMB: Int
    let voiceLibraryDiskMB: Int
    let cacheDiskMB: Int
    let importsDiskMB: Int
    let snapshotsDiskMB: Int
    let huggingFaceHubModelsDiskMB: Int
    let huggingFaceMLXAudioDiskMB: Int
    let legacyMLXAudioSafeToDelete: Bool
    let duplicateSharedCacheModels: [DoctorSharedCacheDuplicateDTO]
    let installedBackedLegacyCacheModels: [DoctorInstalledBackedLegacyCacheDTO]
    let legacyOnlySharedCacheModelIDs: [String]
    let staleInstalledModelIDs: [String]
    let recentInstallEvents: [DoctorInstallEventDTO]
    let advisories: [String]
    let issues: [String]
}

struct DoctorSharedCacheAudit: Sendable {
    let duplicates: [DoctorSharedCacheDuplicateDTO]
    let installedBackedLegacyModels: [DoctorInstalledBackedLegacyCacheDTO]
    let legacyOnlyModelIDs: [String]
}

struct DoctorInstalledModelDTO: Codable, Sendable {
    let id: String
    let displayName: String
    let family: String
    let domain: String
}

struct DoctorInstallEventDTO: Codable, Sendable {
    let modelID: String
    let displayName: String
    let family: String
    let sourceKind: String
    let recordedAt: String
    let succeeded: Bool
    let message: String
}

struct DoctorSharedCacheDuplicateDTO: Codable, Sendable {
    let id: String
    let preferredPath: String
    let preferredSizeMB: Int
    let legacyPath: String
    let legacySizeMB: Int
    let installedPackBacking: String?
}

struct DoctorInstalledBackedLegacyCacheDTO: Codable, Sendable {
    let id: String
    let installedPath: String
    let legacyPath: String
    let legacySizeMB: Int
}

enum DoctorInstalledPackBacking: String, Sendable {
    case preferred
    case legacy
}

struct DoctorResidentModelDTO: Codable, Sendable {
    let id: String
    let displayName: String
    let domain: String
    let state: String
    let lastTouchedAt: String?
    let lastTouchedAgeSeconds: Double?
    let estimatedResidentMB: Int
    let actualResidentMB: Int?
    let residencyPolicy: String
    let outsideWarmSet: Bool
    let activeSessionCount: Int
    let isWarmStartModel: Bool
    let idleTrimEligible: Bool

    init(
        from snapshot: ModelResidencySnapshotDTO,
        referenceTime: Date,
        idleWarmSetExtraIDs: Set<String> = []
    ) {
        self.id = snapshot.id
        self.displayName = snapshot.displayName
        self.domain = snapshot.domain
        self.state = snapshot.state
        self.lastTouchedAt = snapshot.lastTouchedAt
        self.lastTouchedAgeSeconds = DoctorCommand.lastTouchedAgeSeconds(
            from: snapshot.lastTouchedAt,
            referenceTime: referenceTime
        )
        self.estimatedResidentMB = snapshot.estimatedResidentBytes / (1024 * 1024)
        self.actualResidentMB = snapshot.actualResidentBytes.map { $0 / (1024 * 1024) }
        self.residencyPolicy = snapshot.residencyPolicy
        self.outsideWarmSet = idleWarmSetExtraIDs.contains(snapshot.id)
        self.activeSessionCount = snapshot.activeSessionCount
        self.isWarmStartModel = snapshot.isWarmStartModel
        self.idleTrimEligible = snapshot.idleTrimEligible
    }
}
