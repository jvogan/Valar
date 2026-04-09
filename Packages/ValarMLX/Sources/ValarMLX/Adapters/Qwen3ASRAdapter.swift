import Foundation
import ValarModelKit

/// Model adapter for the Qwen3-ASR speech recognition family.
public struct Qwen3ASRAdapter: ModelAdapter, Sendable {
    public let familyID: ModelFamilyID = .qwen3ASR
    public let supportedCapabilities: Set<CapabilityID> = [
        .speechRecognition, .tokenization, .translation
    ]
    public let supportedBackends: [BackendKind] = [.mlx]

    public init() {}

    public func validate(manifest: ModelPackManifest) throws {
        guard manifest.familyID == .qwen3ASR else {
            throw AdapterError.familyMismatch(expected: .qwen3ASR, got: manifest.familyID)
        }
    }

    public func makeDescriptor(from manifest: ModelPackManifest) throws -> ModelDescriptor {
        ModelDescriptor(manifest: manifest)
    }
}
