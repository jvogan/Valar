@preconcurrency import ArgumentParser
import Foundation
import ValarCore
import ValarModelKit
import ValarPersistence

struct ModelsCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "models",
        abstract: "List, install, remove, and inspect model packs.",
        subcommands: [List.self, Info.self, Install.self, Remove.self, PurgeCache.self, Cleanup.self, Status.self]
    )

    mutating func run() async throws {
        throw CleanExit.helpRequest(Self.self)
    }
}

extension ModelsCommand {
    struct List: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Print the supported model catalog."
        )

        mutating func run() async throws {
            let runtime = try ModelsCommand.makeRuntime()
            let models = try await runtime.modelCatalog.supportedModels()

            if OutputContext.jsonRequested {
                try OutputFormat.writeSuccess(
                    command: OutputFormat.commandPath("models list"),
                    data: ModelListPayloadDTO(
                        message: models.isEmpty ? "No supported models found." : "Loaded \(models.count) supported model(s).",
                        models: models.map { ModelSummaryDTO(from: $0) }
                    )
                )
                return
            }

            ModelsCommand.printCatalog(models)
        }
    }

    struct Install: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Download and install a supported model."
        )

        @Argument(help: "The model identifier to install.")
        var id: String

        @Flag(name: .long, help: "Permit downloading model artifacts from the network.")
        var allowDownload: Bool = false

        @Flag(name: .long, help: "Purge shared Hugging Face cache for this model before reinstalling. Remote models require --allow-download when this is set.")
        var refreshCache: Bool = false

        mutating func run() async throws {
            let runtime = try ModelsCommand.makeRuntime()
            let model = try await ModelsCommand.lookupModel(id: id, using: runtime)
            let manifest = try await runtime.modelCatalog.installationManifest(for: model.id)

            guard let manifest else {
                throw CLICommandError(message: "Model '\(id)' has no installation metadata — it may not be downloadable yet.\nRun: valartts models list to see installable models.")
            }

            if model.installState == .installed,
               !refreshCache,
               let existing = try await runtime.modelPackRegistry.installedRecord(for: model.id.rawValue) {
                let message = "Already installed: \(model.descriptor.displayName) | \(existing.installedPath)"
                if OutputContext.jsonRequested {
                    try OutputFormat.writeSuccess(
                        command: OutputFormat.commandPath("models install"),
                        data: ModelOperationPayloadDTO(
                            message: message,
                            model: ModelSummaryDTO(from: model),
                            result: "already_installed",
                            installedPath: existing.installedPath,
                            warnings: []
                        )
                    )
                } else {
                    print(message)
                }
                return
            }

            if refreshCache, model.providerURL != nil, !allowDownload {
                throw ValidationError("Refreshing shared cache for '\(id)' requires --allow-download because Valar will need to fetch a fresh snapshot.")
            }

            if refreshCache {
                _ = try await runtime.modelInstaller.uninstall(modelID: model.id)
                let purgedPaths = try await runtime.modelInstaller.purgeSharedCaches(for: model.id)
                if !OutputContext.jsonRequested {
                    if purgedPaths.isEmpty {
                        print("No shared cache entries needed purging for \(model.descriptor.displayName).")
                    } else {
                        print("Purged shared cache entries:")
                        for path in purgedPaths.sorted() {
                            print("  - \(path)")
                        }
                    }
                }
            }

            // Consent gate: block network downloads unless --allow-download is set.
            // A `.cached` model has files on disk already; only `.supported` + remoteURL requires download.
            if (model.installState == .supported || refreshCache), let sourceURL = model.providerURL, !allowDownload {
                let warningMessage = "Model '\(model.descriptor.displayName)' is not cached locally and requires a download from: \(sourceURL.absoluteString)\nRe-run with --allow-download to permit the download."
                if OutputContext.jsonRequested {
                    struct DownloadRequiredEnvelope: Encodable {
                        let ok: Bool
                        let command: String
                        let error: ErrorDetails
                        struct ErrorDetails: Encodable {
                            let code: Int
                            let kind: String
                            let message: String
                            let model: String
                            let estimatedSize: String
                        }
                    }
                    let encoder = JSONEncoder()
                    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
                    let dto = DownloadRequiredEnvelope(
                        ok: false,
                        command: OutputFormat.commandPath("models install"),
                        error: .init(code: 2, kind: "download_required", message: warningMessage, model: id, estimatedSize: "unknown")
                    )
                    print(String(decoding: try encoder.encode(dto), as: UTF8.self))
                } else {
                    print("warning: \(warningMessage)")
                }
                return
            }

            let sourceKind: ModelPackSourceKind = model.providerURL == nil ? .localFile : .remoteURL
            let sourceLocation = model.providerURL?.absoluteString ?? model.id.rawValue
            let mode: ModelInstallMode = sourceKind == .remoteURL ? .downloadArtifacts : .metadataOnly

            let progressTask: Task<Void, Never>? = OutputContext.jsonRequested ? nil : Task {
                var lastMessage = ""
                for await event in runtime.modelInstaller.progress {
                    guard event.modelID == model.id.rawValue else { continue }
                    let message = ModelsCommand.progressMessage(for: event)
                    guard message != lastMessage else { continue }
                    print(message)
                    lastMessage = message

                    switch event.status {
                    case .completed, .failed:
                        return
                    case .starting, .downloading, .verifying:
                        continue
                    }
                }
            }
            defer { progressTask?.cancel() }

            let result = try await runtime.modelInstaller.install(
                manifest: manifest,
                sourceKind: sourceKind,
                sourceLocation: sourceLocation,
                notes: model.notes,
                mode: mode
            )

            let warnings = result.report.issues.filter { $0.severity == .warning }
            let message = "Installed \(result.descriptor.displayName) | \(result.record.installedPath)"

            if OutputContext.jsonRequested {
                try OutputFormat.writeSuccess(
                    command: OutputFormat.commandPath("models install"),
                    data: ModelOperationPayloadDTO(
                        message: message,
                        model: ModelSummaryDTO(from: model),
                        result: "installed",
                        installedPath: result.record.installedPath,
                        warnings: ModelsCommand.humanReadableInstallWarnings(from: warnings)
                    )
                )
                return
            }

            for warning in ModelsCommand.humanReadableInstallWarnings(from: warnings) {
                print("warning: \(warning)")
            }

            print(message)

            if model.licenseName?.localizedCaseInsensitiveContains("NC") == true {
                let licenseName = model.licenseName ?? "a non-commercial license"
                print("Note: This model is licensed under \(licenseName) (non-commercial use only).")
            }
        }
    }

    struct Info: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Show detailed metadata for one supported model."
        )

        @Argument(help: "The model identifier to inspect.")
        var id: String

        mutating func run() async throws {
            let runtime = try ModelsCommand.makeRuntime()
            let model = try await ModelsCommand.lookupModel(id: id, using: runtime)
            let payload = ModelsCommand.detailPayload(for: model)

            if OutputContext.jsonRequested {
                try OutputFormat.writeSuccess(
                    command: OutputFormat.commandPath("models info"),
                    data: payload
                )
                return
            }

            ModelsCommand.printDetails(payload)
        }
    }

    struct Remove: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Remove an installed model pack."
        )

        @Argument(help: "The model identifier to remove.")
        var id: String

        mutating func run() async throws {
            let runtime = try ModelsCommand.makeRuntime()
            let model = try await ModelsCommand.lookupModel(id: id, using: runtime)

            guard let removed = try await runtime.modelInstaller.uninstall(descriptor: model.descriptor) else {
                throw ValidationError("Model '\(id)' is not installed. Run: valartts models status to see what is installed.")
            }

            let message = "Removed installed pack for \(model.descriptor.displayName) | \(removed.installedPath)"
            let cacheHint = "Shared Hugging Face cache entries, if any, were left intact."
            if OutputContext.jsonRequested {
                try OutputFormat.writeSuccess(
                    command: OutputFormat.commandPath("models remove"),
                    data: ModelOperationPayloadDTO(
                        message: "\(message) \(cacheHint)",
                        model: ModelSummaryDTO(from: model),
                        removedPath: removed.installedPath
                    )
                )
            } else {
                print(message)
                print(cacheHint)
            }
        }
    }

    struct PurgeCache: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            commandName: "purge-cache",
            abstract: "Remove shared Hugging Face cache entries for one model while keeping any installed Valar pack."
        )

        @Argument(help: "The model identifier whose shared cache entries should be removed.")
        var id: String

        mutating func run() async throws {
            let runtime = try ModelsCommand.makeRuntime()
            let model = try await ModelsCommand.lookupModelForMaintenance(id: id, using: runtime)
            let removedPaths = try await runtime.modelInstaller.purgeSharedCaches(for: model.id).sorted()

            let message: String
            if removedPaths.isEmpty {
                message = "No shared cache entries were present for \(model.descriptor.displayName)."
            } else {
                message = "Removed \(removedPaths.count) shared cache entr\(removedPaths.count == 1 ? "y" : "ies") for \(model.descriptor.displayName). Installed Valar packs, if any, were left intact; hard-linked installed packs may still retain the model bytes until you also remove the pack."
            }

            if OutputContext.jsonRequested {
                try OutputFormat.writeSuccess(
                    command: OutputFormat.commandPath("models purge-cache"),
                    data: ModelOperationPayloadDTO(
                        message: message,
                        model: ModelSummaryDTO(from: model),
                        result: "cache_purged",
                        warnings: removedPaths
                    )
                )
                return
            }

            print(message)
            if !removedPaths.isEmpty {
                print("Removed paths:")
                for path in removedPaths {
                    print("  - \(path)")
                }
            }
            print("If this model is still installed, hard-linked bytes may remain until you also run: valartts models remove \(model.id.rawValue)")
            print("Reinstall later with: valartts models install \(model.id.rawValue)")
        }
    }

    struct Cleanup: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Audit and optionally remove orphaned or stale model storage."
        )

        @Flag(name: .long, help: "Report what would be removed without making changes.")
        var dryRun: Bool = false

        @Flag(name: .long, help: "Apply cleanup changes. Without this flag, report what would be removed.")
        var apply: Bool = false

        mutating func run() async throws {
            if apply, dryRun {
                throw ValidationError("Use either --apply or --dry-run, not both.")
            }

            let runtime = try ModelsCommand.makeRuntime()
            let audit = try await runtime.auditLocalModelPackState()
            let staleModelIDs = audit.removedStaleModelIDs.map(\.rawValue).sorted()
            let orphanedPaths = audit.orphanedModelPackPaths.sorted()

            let catalog = try await runtime.modelCatalog.refresh()
            let installReceipts = try await runtime.modelPackRegistry.receipts()
            let validInstallReceipts = installReceipts.filter {
                FileManager.default.fileExists(atPath: $0.installedModelPath)
            }
            let installedPathsByModelID = Dictionary(
                uniqueKeysWithValues: validInstallReceipts.map { ($0.modelID, $0.installedModelPath) }
            )
            let supportedEntriesByModelID = Dictionary(
                uniqueKeysWithValues: SupportedModelCatalog.allSupportedEntries.map { ($0.id.rawValue, $0) }
            )
            let hfHubRoot = ModelsCommand.resolveHFHubCacheRoot()
            let legacyRoot = hfHubRoot.appendingPathComponent("mlx-audio", isDirectory: true)
            let sharedCacheAudit = ModelsCommand.analyzeSharedHFCache(
                modelIDs: Set(catalog.map { $0.id.rawValue })
                    .union(SupportedModelCatalog.allSupportedEntries.map { $0.id.rawValue }),
                hfHubRoot: hfHubRoot,
                hfMLXAudioRoot: legacyRoot,
                installedPathsByModelID: installedPathsByModelID,
                supportedEntriesByModelID: supportedEntriesByModelID
            )
            let legacySafeToDelete = sharedCacheAudit.legacyOnlyModelIDs.isEmpty

            var removedStaleModelIDs: [String] = []
            var removedOrphanedPaths: [String] = []
            var removedLegacyPaths: [String] = []

            if apply {
                let maintenance = await runtime.ensureStartupMaintenance()
                removedStaleModelIDs = maintenance.modelPackState.removedStaleModelIDs.map(\.rawValue).sorted()
                removedOrphanedPaths = try await runtime.removeOrphanedModelPacks(
                    paths: maintenance.modelPackState.orphanedModelPackPaths
                )
                if legacySafeToDelete, FileManager.default.fileExists(atPath: legacyRoot.path) {
                    try FileManager.default.removeItem(at: legacyRoot)
                    removedLegacyPaths = [legacyRoot.path]
                }
            }

            let payload = ModelCleanupPayloadDTO(
                dryRun: !apply,
                removedStaleModelIDs: apply ? removedStaleModelIDs : staleModelIDs,
                orphanedModelPackPaths: orphanedPaths,
                removedOrphanedModelPackPaths: removedOrphanedPaths,
                removedLegacyCachePaths: removedLegacyPaths,
                legacyMLXAudioSafeToDelete: legacySafeToDelete,
                message: apply
                    ? "Applied model storage cleanup."
                    : "Dry-run only. Re-run with --apply to remove orphaned packs, stale records, and safe legacy cache leftovers."
            )

            if OutputContext.jsonRequested {
                try OutputFormat.writeSuccess(
                    command: OutputFormat.commandPath("models cleanup"),
                    data: payload
                )
                return
            }

            print(payload.message)
            if !payload.removedStaleModelIDs.isEmpty {
                print("Stale install records:")
                for modelID in payload.removedStaleModelIDs {
                    print("  - \(modelID)")
                }
            }
            if !payload.orphanedModelPackPaths.isEmpty {
                print(apply ? "Orphaned ModelPacks removed:" : "Orphaned ModelPacks:")
                for path in apply ? payload.removedOrphanedModelPackPaths : payload.orphanedModelPackPaths {
                    print("  - \(path)")
                }
            }
            print("Legacy mlx-audio cache safe to delete: \(payload.legacyMLXAudioSafeToDelete ? "yes" : "no")")
            if !payload.removedLegacyCachePaths.isEmpty {
                print("Removed legacy cache paths:")
                for path in payload.removedLegacyCachePaths {
                    print("  - \(path)")
                }
            }
        }
    }

    struct Status: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Print install status for every supported model."
        )

        mutating func run() async throws {
            let runtime = try ModelsCommand.makeRuntime()
            let models = try await runtime.modelCatalog.supportedModels()

            if OutputContext.jsonRequested {
                try OutputFormat.writeSuccess(
                    command: OutputFormat.commandPath("models status"),
                    data: ModelListPayloadDTO(
                        message: models.isEmpty ? "No supported models found." : "Loaded status for \(models.count) model(s).",
                        models: models.map { ModelSummaryDTO(from: $0) }
                    )
                )
                return
            }

            ModelsCommand.printStatus(models)
        }
    }
}

private extension ModelsCommand {
    static func makeRuntime() throws -> ValarRuntime {
        try ValarRuntime()
    }

    static func lookupModel(id: String, using runtime: ValarRuntime) async throws -> CatalogModel {
        let models = try await runtime.modelCatalog.supportedModels()
        guard let model = models.first(where: { $0.id.rawValue == id }) else {
            if let hiddenReason = CatalogVisibilityPolicy.currentProcess().hiddenReason(for: ModelIdentifier(id)) {
                throw ValidationError(hiddenReason)
            }
            throw ValidationError("Unknown model '\(id)'. Run 'valartts models list' to see supported identifiers.")
        }
        return model
    }

    static func lookupModelForMaintenance(id: String, using runtime: ValarRuntime) async throws -> CatalogModel {
        let models = try await runtime.modelCatalog.supportedModels()
        if let model = models.first(where: { $0.id.rawValue == id }) {
            return model
        }

        let identifier = ModelIdentifier(id)
        if let entry = SupportedModelCatalog.entry(for: identifier) {
            let descriptor = ModelDescriptor(manifest: entry.manifest)
            return CatalogModel(
                id: descriptor.id,
                descriptor: descriptor,
                familyID: descriptor.familyID,
                installState: .supported,
                providerName: entry.remoteURL?.host() ?? "Valar",
                providerURL: entry.remoteURL,
                sourceKind: entry.remoteURL == nil ? .localFile : .remoteURL,
                isRecommended: entry.isRecommended,
                manifestPath: nil,
                installedPath: nil,
                artifactCount: entry.manifest.artifacts.count,
                supportedBackends: entry.manifest.supportedBackends.map(\.backendKind),
                licenseName: entry.manifest.licenses.first?.name,
                licenseURL: entry.manifest.licenses.first?.sourceURL,
                supportTier: entry.manifest.supportTier,
                releaseEligible: entry.manifest.releaseEligible,
                qualityTierByLanguage: entry.manifest.qualityTierByLanguage,
                distributionTier: entry.distributionTier,
                notes: entry.manifest.notes,
                cachedOnDisk: false
            )
        }

        if let hiddenReason = CatalogVisibilityPolicy.currentProcess().hiddenReason(for: identifier) {
            throw ValidationError(hiddenReason)
        }
        throw ValidationError("Unknown model '\(id)'. Run 'valartts models list' to see supported identifiers.")
    }

    static func installStateLabel(_ model: CatalogModel) -> String {
        model.installState == .supported ? "not installed" : model.installState.rawValue
    }

    static func humanReadableInstallWarnings(from issues: [ModelInstallValidationIssue]) -> [String] {
        let checksumWarnings = issues.filter { isMissingChecksumWarning($0.message) }
        let otherWarnings = issues
            .map(\.message)
            .filter { !isMissingChecksumWarning($0) }

        guard !checksumWarnings.isEmpty else {
            return otherWarnings
        }

        let checksumCount = checksumWarnings.count
        let noun = checksumCount == 1 ? "artifact" : "artifacts"
        let checksumSummary =
            "\(checksumCount) \(noun) in this model pack do not declare SHA-256 checksums. Valar can install them, but cannot locally verify the downloaded files."
        return otherWarnings + [checksumSummary]
    }

    static func isMissingChecksumWarning(_ message: String) -> Bool {
        message.contains("is missing a SHA-256 checksum")
    }

    static func printCatalog(_ models: [CatalogModel]) {
        if models.isEmpty {
            print("No supported models found.")
            return
        }

        for model in models {
            print([
                installStateLabel(model),
                model.id.rawValue,
                model.descriptor.displayName,
                model.familyID.rawValue,
                model.supportTier.rawValue,
                model.providerName,
            ].joined(separator: " | "))
        }
    }

    static func printStatus(_ models: [CatalogModel]) {
        if models.isEmpty {
            print("No supported models found.")
            return
        }

        for model in models {
            let location = model.installedPath ?? "-"
            print([
                installStateLabel(model),
                model.id.rawValue,
                model.descriptor.displayName,
                model.supportTier.rawValue,
                location,
            ].joined(separator: " | "))
        }

        let cachedCount = models.filter { $0.installState == .cached }.count
        if cachedCount > 0 {
            print("\nHint: \(cachedCount) model(s) are cached on disk but not registered.")
            print("Run 'valartts models install <id>' to register cached models. This avoids network download, but it may still materialize a local ModelPacks copy/link set.")
        }
        print("Update a model with: valartts models install <id> --refresh-cache --allow-download")
        print("Remove only the shared cache path: valartts models purge-cache <id>")
        print("Free bytes for an installed hard-linked model: valartts models remove <id> && valartts models purge-cache <id>")
        print("Inspect shared cache and safe cleanup with: valartts doctor")
    }

    static func detailPayload(for model: CatalogModel) -> ModelDetailPayloadDTO {
        let caps = model.descriptor.capabilities
        let voiceSupport = model.descriptor.voiceSupport
        return ModelDetailPayloadDTO(
            message: "Loaded metadata for \(model.descriptor.displayName).",
            model: ModelSummaryDTO(from: model),
            providerURL: model.providerURL?.absoluteString,
            manifestPath: model.manifestPath,
            artifactCount: model.artifactCount,
            supportedBackends: model.supportedBackends.map(\.rawValue),
            capabilities: caps.map(\.rawValue).sorted(),
            voiceFeatures: voiceSupport.features.map(\.rawValue),
            defaultSampleRate: model.descriptor.defaultSampleRate,
            supportsReferenceAudio: voiceSupport.supportsReferenceAudio,
            supportTier: model.supportTier.rawValue,
            releaseEligible: model.releaseEligible,
            qualityTierByLanguage: model.qualityTierByLanguage.mapValues(\.rawValue),
            distributionTier: model.distributionTier.rawValue
        )
    }

    static func printDetails(_ payload: ModelDetailPayloadDTO) {
        let model = payload.model
        print("ID: \(model.id)")
        print("Name: \(model.displayName)")
        print("Family: \(model.family)")
        print("Provider: \(model.provider)")
        print("Install State: \(model.installState)")
        print("Cached On Disk: \(model.cachedOnDisk ? "yes" : "no")")
        print("Support Tier: \(payload.supportTier)")
        print("Release Eligible: \(payload.releaseEligible ? "yes" : "no")")
        print("Distribution Tier: \(payload.distributionTier)")
        if let installedPath = model.installedPath {
            print("Installed Path: \(installedPath)")
        }
        if let providerURL = payload.providerURL {
            print("Provider URL: \(providerURL)")
        }
        if let licenseName = model.licenseName {
            print("License: \(licenseName)")
        }
        if let licenseURL = model.licenseURL {
            print("License URL: \(licenseURL)")
        }
        if let notes = model.notes {
            print("Notes: \(notes)")
        }
        if !payload.qualityTierByLanguage.isEmpty {
            let tiers = payload.qualityTierByLanguage
                .sorted { $0.key < $1.key }
                .map { "\($0.key): \($0.value)" }
                .joined(separator: ", ")
            print("Language Tiers: \(tiers)")
        }
        print("Artifacts: \(payload.artifactCount)")
        if let sampleRate = payload.defaultSampleRate {
            print("Sample Rate: \(Int(sampleRate)) Hz")
        }
        let voiceFeatures = payload.voiceFeatures
            .compactMap { rawValue in ModelVoiceFeature(rawValue: rawValue)?.displayName }
        if !voiceFeatures.isEmpty {
            print("Voice Features: \(voiceFeatures.joined(separator: ", "))")
        }
        print("Reference Audio: \(payload.supportsReferenceAudio ? "supported" : "not supported")")
        print("Backends: \(payload.supportedBackends.joined(separator: ", "))")
        print("Capabilities: \(payload.capabilities.joined(separator: ", "))")
        if let manifestPath = payload.manifestPath {
            print("Manifest Path: \(manifestPath)")
        }
    }

    static func progressMessage(for event: ModelInstallProgressEvent) -> String {
        switch event.status {
        case .starting:
            return "Starting install for \(event.modelID)..."
        case .downloading:
            let percent = Int((event.progress * 100).rounded())
            return "Downloading \(event.modelID): \(percent)%"
        case .verifying:
            return "Verifying \(event.modelID)..."
        case .completed:
            return "Completed install for \(event.modelID)."
        case .failed(let message):
            return "Install failed for \(event.modelID): \(message)"
        }
    }

    static func analyzeSharedHFCache(
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

            let preferredSizeMB = hasPreferred ? Self.diskUsageMegabytes(preferredDirectory) : 0
            let legacySizeMB = hasLegacy ? Self.diskUsageMegabytes(legacyDirectory) : 0
            let hasSubstantialPreferred = hasPreferred && preferredSizeMB >= substantialSizeThresholdMB
            let hasSubstantialLegacy = hasLegacy && legacySizeMB >= substantialSizeThresholdMB

            if hasSubstantialPreferred && hasSubstantialLegacy {
                let installedPackBacking = installedPathsByModelID[modelID].flatMap { installedPath in
                    Self.detectInstalledPackBacking(
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

    static func detectInstalledPackBacking(
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
               Self.sameFile(installedArtifact, preferredArtifact) {
                return .preferred
            }

            if let legacyArtifact = Self.legacyArtifactURL(root: legacyDirectory, relativePath: artifact.relativePath, fileManager: fileManager),
               Self.sameFile(installedArtifact, legacyArtifact) {
                return .legacy
            }
        }

        return nil
    }

    static func legacyArtifactURL(
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

    static func sameFile(_ lhs: URL, _ rhs: URL) -> Bool {
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

    static func resolveHFHubCacheRoot(
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

    static func hfHubRepoDirectoryName(for modelID: String) -> String {
        "models--" + modelID.replacingOccurrences(of: "/", with: "--")
    }

    static func hfMLXAudioDirectoryName(for modelID: String) -> String {
        modelID.replacingOccurrences(of: "/", with: "_")
    }

    static func diskUsageMegabytes(_ url: URL) -> Int {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/du")
        process.arguments = ["-sk", url.path]
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = Pipe()
        do {
            try process.run()
            process.waitUntilExit()
            guard process.terminationStatus == 0,
                  let output = String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8)?
                    .split(whereSeparator: \.isWhitespace)
                    .first,
                  let kilobytes = Int(output) else {
                return 0
            }
            return Int((Double(kilobytes) / 1024.0).rounded())
        } catch {
            return 0
        }
    }
}
