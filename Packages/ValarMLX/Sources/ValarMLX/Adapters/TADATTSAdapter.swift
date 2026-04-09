import Foundation
import ValarModelKit

public struct TADATTSAdapter: ModelAdapter, Sendable {
    public let familyID: ModelFamilyID = .tadaTTS
    public let supportedCapabilities: Set<CapabilityID> = [
        .speechSynthesis, .voiceCloning, .multilingual, .audioConditioning
    ]
    public let supportedBackends: [BackendKind] = [.mlx]

    public init() {}

    public func validate(manifest: ModelPackManifest) throws {
        guard manifest.familyID == .tadaTTS else {
            throw AdapterError.familyMismatch(expected: .tadaTTS, got: manifest.familyID)
        }
        guard !manifest.artifacts.isEmpty else {
            throw AdapterError.missingArtifacts("TADA requires model, encoder, decoder, aligner, and tokenizer artifacts")
        }
    }

    public func makeDescriptor(from manifest: ModelPackManifest) throws -> ModelDescriptor {
        ModelDescriptor(manifest: manifest)
    }
}
