import Foundation

public protocol SpeakerEmbeddingInferenceBackend: InferenceBackend {
    func extractSpeakerEmbedding(
        descriptor: ModelDescriptor,
        monoReferenceSamples: [Float]
    ) async throws -> Data
}
