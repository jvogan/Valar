import Foundation

public struct VoiceConditioningExtractionRequest: Sendable, Equatable, Hashable {
    public let descriptor: ModelDescriptor
    public let monoReferenceSamples: [Float]
    public let sampleRate: Double
    public let referenceTranscript: String?
    public let language: String?

    public init(
        descriptor: ModelDescriptor,
        monoReferenceSamples: [Float],
        sampleRate: Double,
        referenceTranscript: String? = nil,
        language: String? = nil
    ) {
        self.descriptor = descriptor
        self.monoReferenceSamples = monoReferenceSamples
        self.sampleRate = sampleRate
        self.referenceTranscript = referenceTranscript
        self.language = language
    }
}

public protocol VoiceConditioningInferenceBackend: InferenceBackend {
    func extractVoiceConditioning(
        _ request: VoiceConditioningExtractionRequest
    ) async throws -> VoiceConditioning
}
