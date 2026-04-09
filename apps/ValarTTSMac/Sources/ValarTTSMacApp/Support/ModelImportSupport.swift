import Foundation
import ValarCore
import ValarModelKit
import ValarPersistence

struct ImportedModelBundleResult: Sendable, Equatable {
    let modelID: ModelIdentifier
    let displayName: String
}

enum ModelImportError: LocalizedError, Equatable {
    case unsupportedBundleExtension(String)
    case bundleNotFound(String)
    case bundleMustBeDirectory(String)
    case missingManifest
    case invalidManifest(String)
    case validationFailed([String])
    case missingWeightArtifacts
    case unsupportedWeightExtension(String)
    case missingWeightFile(String)

    var errorDescription: String? {
        switch self {
        case .unsupportedBundleExtension(let fileName):
            return "“\(fileName)” is not a .valarmodel bundle."
        case .bundleNotFound(let path):
            return "The selected bundle could not be found at \(path)."
        case .bundleMustBeDirectory(let fileName):
            return "“\(fileName)” is not a valid .valarmodel bundle directory."
        case .missingManifest:
            return "The selected bundle is missing manifest.json."
        case .invalidManifest(let message):
            return "manifest.json is invalid: \(message)"
        case .validationFailed(let messages):
            return messages.joined(separator: "\n")
        case .missingWeightArtifacts:
            return "The bundle must declare at least one .safetensors weight artifact in manifest.json."
        case .unsupportedWeightExtension(let relativePath):
            return "Weight artifact “\(relativePath)” must use the .safetensors extension."
        case .missingWeightFile(let relativePath):
            return "The bundle is missing the weight file “\(relativePath)”."
        }
    }
}

struct ImportedModelBundleCandidate: Sendable {
    let bundleURL: URL
    let manifest: ValarPersistence.ModelPackManifest
    let destinationDirectory: URL
}

struct ModelBundleImporter {
    private let fileManager: FileManager

    init(fileManager: FileManager = .default) {
        self.fileManager = fileManager
    }

    func importBundle(
        from sourceURL: URL,
        using services: ValarServiceHub
    ) async throws -> ImportedModelBundleResult {
        let startedSecurityScope = sourceURL.startAccessingSecurityScopedResource()
        defer {
            if startedSecurityScope {
                sourceURL.stopAccessingSecurityScopedResource()
            }
        }

        let appPaths = services.runtime.paths
        let candidate = try await validateBundle(at: sourceURL, using: services)
        let destinationDirectory = candidate.destinationDirectory
        let destinationParent = destinationDirectory.deletingLastPathComponent()
        let stagingDirectory = temporaryDirectory(
            siblingOf: destinationDirectory,
            suffix: "importing"
        )
        let backupDirectory = fileManager.fileExists(atPath: destinationDirectory.path)
            ? temporaryDirectory(siblingOf: destinationDirectory, suffix: "backup")
            : nil

        try removeIfPresent(stagingDirectory)
        if let backupDirectory {
            try removeIfPresent(backupDirectory)
        }

        try fileManager.createDirectory(at: appPaths.modelPacksDirectory, withIntermediateDirectories: true)
        try fileManager.createDirectory(at: destinationParent, withIntermediateDirectories: true)
        try fileManager.copyItem(at: candidate.bundleURL, to: stagingDirectory)

        var installedToDestination = false

        do {
            if let backupDirectory {
                try fileManager.moveItem(at: destinationDirectory, to: backupDirectory)
            }

            try fileManager.moveItem(at: stagingDirectory, to: destinationDirectory)
            installedToDestination = true

            _ = try await services.modelInstaller.install(
                manifest: candidate.manifest,
                sourceKind: .importedArchive,
                sourceLocation: candidate.bundleURL.path,
                notes: "Imported custom model",
                mode: .metadataOnly
            )

            if let backupDirectory {
                try removeIfPresent(backupDirectory)
            }

            return ImportedModelBundleResult(
                modelID: ModelIdentifier(candidate.manifest.modelID),
                displayName: candidate.manifest.displayName
            )
        } catch {
            try? removeIfPresent(stagingDirectory)

            if installedToDestination {
                try? removeIfPresent(destinationDirectory)
            }

            if let backupDirectory, fileManager.fileExists(atPath: backupDirectory.path) {
                try? fileManager.moveItem(at: backupDirectory, to: destinationDirectory)
            }

            throw error
        }
    }

    func validateBundle(
        at sourceURL: URL,
        using services: ValarServiceHub
    ) async throws -> ImportedModelBundleCandidate {
        let bundleURL = sourceURL.standardizedFileURL
        guard bundleURL.pathExtension.lowercased() == "valarmodel" else {
            throw ModelImportError.unsupportedBundleExtension(bundleURL.lastPathComponent)
        }
        guard fileManager.fileExists(atPath: bundleURL.path) else {
            throw ModelImportError.bundleNotFound(bundleURL.path)
        }
        guard isDirectory(bundleURL) else {
            throw ModelImportError.bundleMustBeDirectory(bundleURL.lastPathComponent)
        }

        let manifestURL = bundleURL.appendingPathComponent("manifest.json", isDirectory: false)
        try ValarAppPaths.validateContainment(manifestURL, within: bundleURL, fileManager: fileManager)
        guard fileManager.fileExists(atPath: manifestURL.path) else {
            throw ModelImportError.missingManifest
        }

        let manifest: ValarPersistence.ModelPackManifest
        do {
            manifest = try JSONDecoder().decode(
                ValarPersistence.ModelPackManifest.self,
                from: Data(contentsOf: manifestURL)
            )
        } catch {
            throw ModelImportError.invalidManifest(error.localizedDescription)
        }

        let report = await services.modelInstaller.validate(manifest)
        let validationErrors = report.issues
            .filter { $0.severity == .error }
            .map(\.message)
        if !validationErrors.isEmpty {
            throw ModelImportError.validationFailed(validationErrors)
        }

        let weightArtifacts = manifest.artifactSpecs.filter { artifact in
            artifact.kind.caseInsensitiveCompare("weights") == .orderedSame
                || artifact.relativePath.lowercased().hasSuffix(".safetensors")
        }
        guard !weightArtifacts.isEmpty else {
            throw ModelImportError.missingWeightArtifacts
        }

        for artifact in weightArtifacts {
            guard artifact.relativePath.lowercased().hasSuffix(".safetensors") else {
                throw ModelImportError.unsupportedWeightExtension(artifact.relativePath)
            }

            let weightURL = bundleURL.appendingPathComponent(artifact.relativePath, isDirectory: false)
            try ValarAppPaths.validateContainment(weightURL, within: bundleURL, fileManager: fileManager)
            guard fileManager.fileExists(atPath: weightURL.path), !isDirectory(weightURL) else {
                throw ModelImportError.missingWeightFile(artifact.relativePath)
            }
        }

        let destinationDirectory = try services.runtime.paths.modelPackDirectory(
            familyID: manifest.familyID,
            modelID: manifest.modelID
        )

        return ImportedModelBundleCandidate(
            bundleURL: bundleURL,
            manifest: manifest,
            destinationDirectory: destinationDirectory
        )
    }

    private func temporaryDirectory(siblingOf destinationDirectory: URL, suffix: String) -> URL {
        destinationDirectory
            .deletingLastPathComponent()
            .appendingPathComponent(
                ".\(destinationDirectory.lastPathComponent)-\(suffix)-\(UUID().uuidString)",
                isDirectory: true
            )
    }

    private func removeIfPresent(_ url: URL) throws {
        guard fileManager.fileExists(atPath: url.path) else { return }
        try fileManager.removeItem(at: url)
    }

    private func isDirectory(_ url: URL) -> Bool {
        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory) else {
            return false
        }
        return isDirectory.boolValue
    }
}
