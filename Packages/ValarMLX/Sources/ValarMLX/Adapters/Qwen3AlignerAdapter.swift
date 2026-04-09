import Foundation
import ValarModelKit

/// Model adapter for the Qwen3 forced-alignment family.
public struct Qwen3AlignerAdapter: ModelAdapter, Sendable {
    public let familyID: ModelFamilyID = .qwen3ForcedAligner
    public let supportedCapabilities: Set<CapabilityID> = [
        .speechRecognition, .forcedAlignment, .tokenization
    ]
    public let supportedBackends: [BackendKind] = [.mlx]

    public init() {}

    public func validate(manifest: ModelPackManifest) throws {
        guard manifest.familyID == .qwen3ForcedAligner else {
            throw AdapterError.familyMismatch(expected: .qwen3ForcedAligner, got: manifest.familyID)
        }
    }

    public func makeDescriptor(from manifest: ModelPackManifest) throws -> ModelDescriptor {
        ModelDescriptor(manifest: manifest)
    }
}
