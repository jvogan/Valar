import Foundation
import HuggingFace

public enum ModelDirectoryKind: String, Sendable {
    case valarManagedPack
    case huggingFaceCache
    case legacyMLXAudioCache
    case other
}

public enum ModelUtils {
    private static let metadataRelativePaths = [
        "config.json",
        "params.json",
        "model/config.json",
        "model/params.json",
    ]
    private static let tadaRequiredRelativePathsWithoutTokenizer = [
        "model/config.json",
    ]
    private static let tadaComponentDirectories = ["model", "encoder", "decoder", "aligner"]
    private static let tadaDownloadPatterns = [
        "model/config.json",
        "model/*.json",
        "model/*.safetensors",
        "encoder/*.safetensors",
        "decoder/*.safetensors",
        "aligner/*.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ]
    private static let metaTokenizerFilename = "tokenizer.json"
    private static let metaTokenizerRepo = Repo.ID(rawValue: "meta-llama/Llama-3.2-1B")!
    /// TADA model repos that bundle their own tokenizer — checked before the meta-llama fallback.
    private static let tadaPreferredTokenizerRepos: [Repo.ID] = [
        Repo.ID(rawValue: "HumeAI/mlx-tada-1b")!,
        Repo.ID(rawValue: "HumeAI/mlx-tada-3b")!,
    ]
    /// meta-llama repos that carry a compatible tokenizer, used as a fallback when no TADA snapshot is cached.
    private static let tadaFallbackTokenizerRepos: [Repo.ID] = [
        Repo.ID(rawValue: "meta-llama/Llama-3.2-1B")!,
        Repo.ID(rawValue: "meta-llama/Llama-3.2-1B-Instruct")!,
        Repo.ID(rawValue: "meta-llama/Llama-3.2-3B")!,
        Repo.ID(rawValue: "meta-llama/Llama-3.2-3B-Instruct")!,
    ]

    public static func resolveModelType(
        repoID: Repo.ID,
        hfToken: String? = nil,
        cache: HubCache = .default
    ) async throws -> String? {
        let lowerRepoID = repoID.description.lowercased()
        let modelURL = try await resolveOrDownloadModel(
            repoID: repoID,
            requiredExtension: "safetensors",
            additionalMatchingPatterns: lowerRepoID.contains("tada") ? tadaDownloadPatterns : [],
            requiredRelativePaths: lowerRepoID.contains("tada") ? tadaRequiredRelativePathsWithoutTokenizer : [],
            hfToken: hfToken,
            cache: cache
        )
        return try resolveModelType(modelDirectory: modelURL, modelNameHint: repoID.name)
    }

    public static func resolveModelType(modelDirectory: URL) throws -> String? {
        try resolveModelType(modelDirectory: modelDirectory, modelNameHint: modelDirectory.lastPathComponent)
    }

    /// Resolves a model from cache or downloads it if not cached.
    /// - Parameters:
    ///   - string: The repository name
    ///   - requiredExtension: File extension that must exist for cache to be considered complete (e.g., "safetensors")
    ///   - hfToken: The huggingface token for access to gated repositories, if needed.
    /// - Returns: The model directory URL
    public static func resolveOrDownloadModel(
        repoID: Repo.ID,
        requiredExtension: String,
        additionalMatchingPatterns: [String] = [],
        requiredRelativePaths: [String] = [],
        hfToken: String? = nil,
        cache: HubCache = .default
    ) async throws -> URL {
        let client = makeHubClient(hfToken: hfToken, cache: cache)
        let resolvedCache = client.cache ?? cache
        return try await resolveOrDownloadModel(
            client: client,
            cache: resolvedCache,
            repoID: repoID,
            requiredExtension: requiredExtension,
            additionalMatchingPatterns: additionalMatchingPatterns,
            requiredRelativePaths: requiredRelativePaths
        )
    }

    /// Resolves a model from cache or downloads it if not cached.
    /// - Parameters:
    ///   - client: The HuggingFace Hub client
    ///   - cache: The HuggingFace cache
    ///   - repoID: The repository ID
    ///   - requiredExtension: File extension that must exist for cache to be considered complete (e.g., "safetensors")
    /// - Returns: The model directory URL
    public static func resolveOrDownloadModel(
        client: HubClient,
        cache: HubCache = .default,
        repoID: Repo.ID,
        requiredExtension: String,
        additionalMatchingPatterns: [String] = [],
        requiredRelativePaths: [String] = []
    ) async throws -> URL {
        let normalizedRequiredExtension = requiredExtension.hasPrefix(".")
            ? String(requiredExtension.dropFirst())
            : requiredExtension
        let isTadaRepo = repoID.description.lowercased().contains("tada")

        if let hubSnapshot = preferredHubSnapshotDirectory(
            repoID: repoID,
            cache: cache,
            requiredExtension: normalizedRequiredExtension,
            requiredRelativePaths: requiredRelativePaths
        ), !isTadaRepo || directoryHasCompleteTadaContent(
            at: hubSnapshot,
            requiresTokenizer: requiredRelativePaths.contains(metaTokenizerFilename)
        ) {
            print("Using standard Hugging Face snapshot at: \(hubSnapshot.path)")
            return hubSnapshot
        }

        let modelSubdir = repoID.description.replacingOccurrences(of: "/", with: "_")
        let modelDir = cache.cacheDirectory
            .appendingPathComponent("mlx-audio")
            .appendingPathComponent(modelSubdir)

        // Backward compatibility for older mlx-audio cache layouts already present on disk.
        if FileManager.default.fileExists(atPath: modelDir.path) {
            let hasRequiredFile = isTadaRepo
                ? directoryHasCompleteTadaContent(
                    at: modelDir,
                    requiresTokenizer: requiredRelativePaths.contains(metaTokenizerFilename)
                )
                : directoryHasRequiredContent(
                    at: modelDir,
                    requiredExtension: normalizedRequiredExtension,
                    requiredRelativePaths: requiredRelativePaths
                )

            if hasRequiredFile {
                if let metadataURL = preferredMetadataURL(in: modelDir) {
                    if let metadataData = try? Data(contentsOf: metadataURL),
                       let _ = try? JSONSerialization.jsonObject(with: metadataData) {
                        print("Using legacy mlx-audio cache at: \(modelDir.path)")
                        return modelDir
                    } else {
                        print("Cached \(metadataURL.lastPathComponent) is invalid, clearing cache...")
                        Self.clearCaches(modelDir: modelDir, repoID: repoID, hubCache: cache)
                    }
                } else {
                    print("Cached model is missing config.json/params.json, clearing cache...")
                    Self.clearCaches(modelDir: modelDir, repoID: repoID, hubCache: cache)
                }
            } else {
                print("Cached model appears incomplete, clearing cache...")
                Self.clearCaches(modelDir: modelDir, repoID: repoID, hubCache: cache)
            }
        }

        var allowedExtensions: Set<String> = [
            "*.\(normalizedRequiredExtension)",
            "*.safetensors",
            "*.json",
            "*.txt",
            "*.wav",
        ]
        allowedExtensions.formUnion(additionalMatchingPatterns)

        print("Downloading model \(repoID)...")
        let snapshotDir = try await client.downloadSnapshot(
            of: repoID,
            kind: .model,
            revision: "main",
            matching: Array(allowedExtensions),
            progressHandler: { progress in
                print("\(progress.completedUnitCount)/\(progress.totalUnitCount) files")
            }
        )

        let hasValidFile = isTadaRepo
            ? directoryHasCompleteTadaContent(
                at: snapshotDir,
                requiresTokenizer: requiredRelativePaths.contains(metaTokenizerFilename)
            )
            : directoryHasRequiredContent(
                at: snapshotDir,
                requiredExtension: normalizedRequiredExtension,
                requiredRelativePaths: requiredRelativePaths
            )

        if !hasValidFile {
            Self.clearCaches(modelDir: modelDir, repoID: repoID, hubCache: cache)
            if requiredRelativePaths.isEmpty {
                throw ModelUtilsError.incompleteDownload(repoID.description)
            }
            throw ModelUtilsError.missingRequiredFiles(repoID.description, requiredRelativePaths)
        }

        print("Model downloaded to: \(snapshotDir.path)")
        return snapshotDir
    }

    public static func installSelfContainedTadaPack(
        repoID: Repo.ID,
        hfToken: String? = nil,
        cache: HubCache = .default
    ) async throws -> URL {
        let client = makeHubClient(hfToken: hfToken, cache: cache)
        let resolvedCache = client.cache ?? cache
        let modelDir = try await resolveOrDownloadModel(
            client: client,
            cache: resolvedCache,
            repoID: repoID,
            requiredExtension: "safetensors",
            additionalMatchingPatterns: tadaDownloadPatterns,
            requiredRelativePaths: tadaRequiredRelativePathsWithoutTokenizer
        )

        try await materializeTadaTokenizerIfNeeded(
            into: modelDir,
            modelRepoID: repoID,
            hfToken: hfToken,
            cache: resolvedCache
        )
        try validateTadaPackLayout(modelDirectory: modelDir)
        return modelDir
    }

    public static func validateTadaPackLayout(modelDirectory: URL) throws {
        var missingPaths: [String] = []
        if !hasNonEmptyFile(at: modelDirectory.appendingPathComponent("model/config.json")) {
            missingPaths.append("model/config.json")
        }
        if !hasNonEmptyFile(at: modelDirectory.appendingPathComponent(metaTokenizerFilename)) {
            missingPaths.append(metaTokenizerFilename)
        }
        for component in tadaComponentDirectories where !containsNonZeroSafeTensors(in: modelDirectory.appendingPathComponent(component, isDirectory: true)) {
            missingPaths.append("\(component)/*.safetensors")
        }
        guard missingPaths.isEmpty else {
            throw ModelUtilsError.missingTadaPackFiles(modelDirectory.path, missingPaths)
        }
    }

    public static func modelDirectoryKind(_ modelDir: URL) -> ModelDirectoryKind {
        let path = modelDir.standardizedFileURL.path
        if path.contains("/Library/Application Support/ValarTTS/ModelPacks/") {
            return .valarManagedPack
        }
        if path.contains("/.cache/huggingface/hub/") {
            if path.contains("/mlx-audio/") {
                return .legacyMLXAudioCache
            }
            return .huggingFaceCache
        }
        return .other
    }

    public static func shouldAutoDeleteCorruptedModelDirectory(_ modelDir: URL) -> Bool {
        switch modelDirectoryKind(modelDir) {
        case .huggingFaceCache, .legacyMLXAudioCache:
            return true
        case .valarManagedPack, .other:
            return false
        }
    }

    private static func clearCaches(modelDir: URL, repoID: Repo.ID, hubCache: HubCache) {
        if shouldAutoDeleteCorruptedModelDirectory(modelDir) {
            try? FileManager.default.removeItem(at: modelDir)
        }
        let hubRepoDir = hubCache.repoDirectory(repo: repoID, kind: .model)
        if FileManager.default.fileExists(atPath: hubRepoDir.path) {
            print("Clearing Hub cache at: \(hubRepoDir.path)")
            try? FileManager.default.removeItem(at: hubRepoDir)
        }
    }

    private static func preferredHubSnapshotDirectory(
        repoID: Repo.ID,
        cache: HubCache,
        requiredExtension: String,
        requiredRelativePaths: [String]
    ) -> URL? {
        guard let snapshotDir = hubSnapshotDirectory(repoID: repoID, cache: cache) else {
            return nil
        }

        guard directoryHasRequiredContent(
            at: snapshotDir,
            requiredExtension: requiredExtension,
            requiredRelativePaths: requiredRelativePaths
        ) else {
            return nil
        }

        if let metadataURL = preferredMetadataURL(in: snapshotDir) {
            guard let metadataData = try? Data(contentsOf: metadataURL),
                  (try? JSONSerialization.jsonObject(with: metadataData)) != nil else {
                return nil
            }
        } else {
            return nil
        }

        return snapshotDir
    }

    private static func hubSnapshotDirectory(repoID: Repo.ID, cache: HubCache) -> URL? {
        let repoDirectory = cache.repoDirectory(repo: repoID, kind: .model)
        let snapshotsDirectory = repoDirectory.appendingPathComponent("snapshots", isDirectory: true)
        guard FileManager.default.fileExists(atPath: snapshotsDirectory.path) else {
            return nil
        }

        let refsMain = repoDirectory.appendingPathComponent("refs/main", isDirectory: false)
        if let revision = try? String(contentsOf: refsMain, encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines),
           !revision.isEmpty {
            let candidate = snapshotsDirectory.appendingPathComponent(revision, isDirectory: true)
            if FileManager.default.fileExists(atPath: candidate.path) {
                return candidate
            }
        }

        let snapshots = (try? FileManager.default.contentsOfDirectory(
            at: snapshotsDirectory,
            includingPropertiesForKeys: [.contentModificationDateKey],
            options: [.skipsHiddenFiles]
        )) ?? []
        return snapshots.max(by: {
            let lhs = (try? $0.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
            let rhs = (try? $1.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
            return lhs < rhs
        })
    }

    private static func resolveModelType(modelDirectory: URL, modelNameHint: String) throws -> String? {
        let modelNameComponents = fallbackModelNameComponents(from: modelNameHint)
        guard let metadataURL = preferredMetadataURL(in: modelDirectory) else {
            return modelNameComponents?.first?.lowercased()
        }

        let metadataJSON = try JSONSerialization.jsonObject(with: Data(contentsOf: metadataURL))
        if let metadata = metadataJSON as? [String: Any] {
            if isTadaMetadata(metadata, modelDirectory: modelDirectory) {
                return "tada"
            }
            return (metadata["model_type"] as? String)
                ?? (metadata["architecture"] as? String)
                ?? modelNameComponents?.first?.lowercased()
        }

        return nil
    }

    private static func isTadaMetadata(_ metadata: [String: Any], modelDirectory: URL) -> Bool {
        if let architecture = metadata["architecture"] as? String,
           architecture.localizedCaseInsensitiveContains("tada") {
            return true
        }

        if let architectures = metadata["architectures"] as? [String],
           architectures.contains(where: { $0.localizedCaseInsensitiveContains("tada") }) {
            return true
        }

        // Current Hume MLX configs advertise `model_type: "llama"` even though the pack
        // is a self-contained TADA family. Fall back to the pack layout to disambiguate
        // from generic llama/orpheus repositories.
        if let modelType = metadata["model_type"] as? String,
           modelType.caseInsensitiveCompare("llama") == .orderedSame,
           directoryHasCompleteTadaContent(at: modelDirectory, requiresTokenizer: false),
           metadata["diffusion_head_type"] != nil || metadata["acoustic_dim"] != nil {
            return true
        }

        return false
    }

    private static func preferredMetadataURL(in modelDirectory: URL) -> URL? {
        for relativePath in metadataRelativePaths {
            let candidate = modelDirectory.appendingPathComponent(relativePath)
            if FileManager.default.fileExists(atPath: candidate.path) {
                return candidate
            }
        }
        for fileURL in recursiveFiles(in: modelDirectory) where metadataRelativePaths.contains(fileURL.lastPathComponent) {
            return fileURL
        }
        return nil
    }

    private static func fallbackModelNameComponents(from modelNameHint: String) -> [Substring]? {
        let finalComponent = modelNameHint.split(separator: "/").last
        return finalComponent?.split(separator: "-")
    }

    private static func makeHubClient(hfToken: String?, cache: HubCache) -> HubClient {
        if let token = hfToken?.trimmingCharacters(in: .whitespacesAndNewlines), !token.isEmpty {
            print("Using HuggingFace token from configuration")
            return HubClient(host: HubClient.defaultHost, bearerToken: token, cache: cache)
        }
        return HubClient(cache: cache)
    }

    private static func directoryHasRequiredContent(
        at modelDir: URL,
        requiredExtension: String,
        requiredRelativePaths: [String]
    ) -> Bool {
        if !requiredRelativePaths.isEmpty {
            return requiredRelativePaths.allSatisfy { relativePath in
                hasNonEmptyFile(at: modelDir.appendingPathComponent(relativePath))
            }
        }

        return recursiveFiles(in: modelDir).contains { file in
            guard file.pathExtension == requiredExtension else { return false }
            return hasNonEmptyFile(at: file)
        }
    }

    private static func recursiveFiles(in directory: URL) -> [URL] {
        let enumerator = FileManager.default.enumerator(
            at: directory,
            includingPropertiesForKeys: [.isRegularFileKey, .fileSizeKey],
            options: [.skipsHiddenFiles]
        )

        var files: [URL] = []
        while let url = enumerator?.nextObject() as? URL {
            let values = try? url.resourceValues(forKeys: [.isRegularFileKey])
            if values?.isRegularFile == true {
                files.append(url)
            }
        }
        return files
    }

    private static func hasNonEmptyFile(at url: URL) -> Bool {
        guard FileManager.default.fileExists(atPath: url.path) else { return false }
        let size = (try? url.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0
        return size > 0
    }

    private static func directoryHasCompleteTadaContent(at modelDir: URL, requiresTokenizer: Bool) -> Bool {
        guard hasNonEmptyFile(at: modelDir.appendingPathComponent("model/config.json")) else {
            return false
        }
        if requiresTokenizer, !hasNonEmptyFile(at: modelDir.appendingPathComponent(metaTokenizerFilename)) {
            return false
        }
        return tadaComponentDirectories.allSatisfy { component in
            containsNonZeroSafeTensors(in: modelDir.appendingPathComponent(component, isDirectory: true))
        }
    }

    private static func containsNonZeroSafeTensors(in directory: URL) -> Bool {
        guard let files = try? FileManager.default.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: [.fileSizeKey],
            options: [.skipsHiddenFiles]
        ) else {
            return false
        }

        return files.contains { file in
            guard file.pathExtension.lowercased() == "safetensors" else { return false }
            let size = (try? file.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
            return size > 0
        }
    }

    private static func materializeTadaTokenizerIfNeeded(
        into modelDir: URL,
        modelRepoID: Repo.ID,
        hfToken: String?,
        cache: HubCache
    ) async throws {
        let targetURL = modelDir.appendingPathComponent(metaTokenizerFilename)
        if hasNonEmptyFile(at: targetURL) {
            return
        }

        // Check preferred TADA snapshots first (the TADA repos bundle their own tokenizer),
        // then fall back to meta-llama snapshots which carry a compatible tokenizer.
        let candidateRepos = tadaPreferredTokenizerRepos + tadaFallbackTokenizerRepos

        // 1. Look in standard HF hub snapshots for all candidate repos.
        for candidateRepo in candidateRepos {
            let snapshot = preferredHubSnapshotDirectory(
                repoID: candidateRepo,
                cache: cache,
                requiredExtension: "json",
                requiredRelativePaths: [metaTokenizerFilename]
            )
            if let url = snapshot?.appendingPathComponent(metaTokenizerFilename),
               hasNonEmptyFile(at: url) {
                let sourceURL = url
                do {
                    if FileManager.default.fileExists(atPath: targetURL.path) {
                        try FileManager.default.removeItem(at: targetURL)
                    }
                    try FileManager.default.copyItem(at: sourceURL, to: targetURL)
                } catch {
                    throw ModelUtilsError.tadaTokenizerMaterializationFailed(
                        modelRepoID.description,
                        "Failed to copy \(metaTokenizerFilename) into \(modelDir.path): \(error.localizedDescription)"
                    )
                }
                return
            }
        }

        // 2. Look in the legacy mlx-audio flat cache layout for all candidate repos.
        for candidateRepo in candidateRepos {
            let legacyDir = cache.cacheDirectory
                .appendingPathComponent("mlx-audio")
                .appendingPathComponent(candidateRepo.description.replacingOccurrences(of: "/", with: "_"))
            let legacyURL = legacyDir.appendingPathComponent(metaTokenizerFilename)
            if hasNonEmptyFile(at: legacyURL) {
                let sourceURL = legacyURL
                do {
                    if FileManager.default.fileExists(atPath: targetURL.path) {
                        try FileManager.default.removeItem(at: targetURL)
                    }
                    try FileManager.default.copyItem(at: sourceURL, to: targetURL)
                } catch {
                    throw ModelUtilsError.tadaTokenizerMaterializationFailed(
                        modelRepoID.description,
                        "Failed to copy \(metaTokenizerFilename) into \(modelDir.path): \(error.localizedDescription)"
                    )
                }
                return
            }
        }

        // 3. Nothing cached — download from the TADA repo if a token is available,
        //    then fall back to meta-llama/Llama-3.2-1B if the TADA repo is unavailable.
        guard let token = hfToken?.trimmingCharacters(in: .whitespacesAndNewlines), !token.isEmpty else {
            throw ModelUtilsError.tadaMetaTokenizerAccessDenied(modelRepoID.description, metaTokenizerRepo.description)
        }

        let downloadCandidates = candidateRepos
        var lastError: Error?
        var sourceURL: URL?
        let client = makeHubClient(hfToken: token, cache: cache)
        for candidateRepo in downloadCandidates {
            do {
                let snapshotDir = try await client.downloadSnapshot(
                    of: candidateRepo,
                    kind: .model,
                    revision: "main",
                    matching: [metaTokenizerFilename],
                    progressHandler: { _ in }
                )
                let candidate = snapshotDir.appendingPathComponent(metaTokenizerFilename)
                if hasNonEmptyFile(at: candidate) {
                    sourceURL = candidate
                    break
                }
            } catch {
                lastError = error
            }
        }

        guard let resolvedSourceURL = sourceURL else {
            if let err = lastError, isMetaTokenizerAccessError(err) {
                throw ModelUtilsError.tadaMetaTokenizerAccessDenied(modelRepoID.description, metaTokenizerRepo.description)
            }
            let tried = downloadCandidates.map(\.description).joined(separator: ", ")
            throw ModelUtilsError.tadaTokenizerMaterializationFailed(
                modelRepoID.description,
                "Failed to download \(metaTokenizerFilename) from any candidate repo (\(tried)): \(lastError?.localizedDescription ?? "unknown error")"
            )
        }

        guard hasNonEmptyFile(at: resolvedSourceURL) else {
            throw ModelUtilsError.tadaTokenizerMaterializationFailed(
                modelRepoID.description,
                "Downloaded \(metaTokenizerFilename) was missing or empty."
            )
        }

        do {
            if FileManager.default.fileExists(atPath: targetURL.path) {
                try FileManager.default.removeItem(at: targetURL)
            }
            try FileManager.default.copyItem(at: resolvedSourceURL, to: targetURL)
        } catch {
            throw ModelUtilsError.tadaTokenizerMaterializationFailed(
                modelRepoID.description,
                "Failed to copy \(metaTokenizerFilename) into \(modelDir.path): \(error.localizedDescription)"
            )
        }
    }

    private static func isMetaTokenizerAccessError(_ error: Error) -> Bool {
        let message = [
            error.localizedDescription,
            String(describing: error),
        ]
        .joined(separator: "\n")
        .lowercased()

        let mentionsMetaRepo = message.contains("meta-llama") || message.contains("llama-3.2-1b")
        let mentionsAccessFailure = message.contains("401")
            || message.contains("403")
            || message.contains("unauthorized")
            || message.contains("forbidden")
            || message.contains("gated")
            || message.contains("license")
            || message.contains("accept")
            || message.contains("access")

        return mentionsMetaRepo && mentionsAccessFailure
    }
}

public enum ModelUtilsError: LocalizedError {
    case incompleteDownload(String)
    case missingRequiredFiles(String, [String])
    case missingTadaPackFiles(String, [String])
    case tadaMetaTokenizerAccessDenied(String, String)
    case tadaTokenizerMaterializationFailed(String, String)

    public var errorDescription: String? {
        switch self {
        case .incompleteDownload(let repo):
            return "Downloaded model '\(repo)' has missing or zero-byte weight files. "
                + "The cache has been cleared — please try again."
        case .missingRequiredFiles(let repo, let requiredPaths):
            return "Downloaded model '\(repo)' is missing required files: \(requiredPaths.joined(separator: ", ")). "
                + "The cache has been cleared — please try again."
        case .missingTadaPackFiles(let path, let missingPaths):
            return "Incomplete TADA pack at \(path). Missing: \(missingPaths.joined(separator: ", ")). "
                + "Expected self-contained layout: model/config.json, model/weights.safetensors, encoder/weights.safetensors, decoder/weights.safetensors, aligner/weights.safetensors, tokenizer.json."
        case .tadaMetaTokenizerAccessDenied(let modelRepo, let metaRepo):
            return "Cannot install '\(modelRepo)' because tokenizer.json must be copied from gated repo '\(metaRepo)'. "
                + "Use a Hugging Face token that has accepted the Meta Llama 3.2 license for \(metaRepo), then retry the install."
        case .tadaTokenizerMaterializationFailed(let modelRepo, let reason):
            return "Failed to materialize tokenizer.json for '\(modelRepo)'. \(reason)"
        }
    }
}
