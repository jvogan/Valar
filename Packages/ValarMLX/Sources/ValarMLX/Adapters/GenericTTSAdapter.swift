import Foundation
import ValarModelKit

/// Generic adapter for non-Qwen TTS models supported by mlx-audio-swift
/// (e.g., Soprano, Orpheus, Marvis, Pocket TTS).
public struct GenericTTSAdapter: ModelAdapter, Sendable {
    public let familyID: ModelFamilyID
    public let supportedCapabilities: Set<CapabilityID> = [.speechSynthesis]
    public let supportedBackends: [BackendKind] = [.mlx]

    public init(familyID: ModelFamilyID) {
        self.familyID = familyID
    }

    public func validate(manifest: ModelPackManifest) throws {
        guard !manifest.artifacts.isEmpty else {
            throw AdapterError.missingArtifacts("Model requires at least weight artifacts")
        }
    }

    public func makeDescriptor(from manifest: ModelPackManifest) throws -> ModelDescriptor {
        ModelDescriptor(manifest: manifest)
    }
}
