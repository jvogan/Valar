import Foundation
import ValarModelKit

/// Model adapter for the Qwen3-TTS family (Base, CustomVoice, VoiceDesign).
///
/// Validates manifests, creates descriptors, and provides TTS synthesis
/// by delegating to mlx-audio-swift's Qwen3TTS model.
public struct Qwen3TTSAdapter: ModelAdapter, Sendable {
    public let familyID: ModelFamilyID = .qwen3TTS
    public let supportedCapabilities: Set<CapabilityID> = [
        .speechSynthesis, .tokenization, .longFormRendering,
        .voiceCloning, .voiceDesign, .audioConditioning
    ]
    public let supportedBackends: [BackendKind] = [.mlx]

    public init() {}

    public func validate(manifest: ModelPackManifest) throws {
        guard manifest.familyID == .qwen3TTS else {
            throw AdapterError.familyMismatch(expected: .qwen3TTS, got: manifest.familyID)
        }
        guard !manifest.artifacts.isEmpty else {
            throw AdapterError.missingArtifacts("Qwen3-TTS requires at least model weights")
        }
    }

    public func makeDescriptor(from manifest: ModelPackManifest) throws -> ModelDescriptor {
        ModelDescriptor(manifest: manifest)
    }
}

public enum AdapterError: Error, Sendable {
    case familyMismatch(expected: ModelFamilyID, got: ModelFamilyID)
    case missingArtifacts(String)
    case unsupportedSurface(String)
}
