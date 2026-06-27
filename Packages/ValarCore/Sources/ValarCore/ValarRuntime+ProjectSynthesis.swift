import Foundation
import ValarModelKit
import ValarPersistence

public enum ValarProjectSynthesisError: LocalizedError, Equatable, Sendable {
    case modelNotFound(ModelIdentifier)
    case unsupportedSpeechSynthesis(ModelIdentifier)
    case noCompatibleBackend(ModelIdentifier)

    public var errorDescription: String? {
        switch self {
        case let .modelNotFound(identifier):
            return "Text-to-speech model not found: \(identifier.rawValue)"
        case let .unsupportedSpeechSynthesis(identifier):
            return "Model \(identifier.rawValue) does not support speech synthesis."
        case let .noCompatibleBackend(identifier):
            return "No compatible inference backend is available for text-to-speech model \(identifier.rawValue)."
        }
    }
}

public extension ValarRuntime {
    func speechSynthesisDescriptor(for identifier: ModelIdentifier) async throws -> ModelDescriptor {
        let descriptor: ModelDescriptor?
        if let registered = await modelRegistry.descriptor(for: identifier) {
            descriptor = registered
        } else if let registered = await capabilityRegistry.descriptor(for: identifier) {
            descriptor = registered
        } else {
            descriptor = try await modelCatalog.model(for: identifier)?.descriptor
        }

        guard let resolvedDescriptor = descriptor else {
            throw ValarProjectSynthesisError.modelNotFound(identifier)
        }
        guard resolvedDescriptor.capabilities.contains(.speechSynthesis) else {
            throw ValarProjectSynthesisError.unsupportedSpeechSynthesis(resolvedDescriptor.id)
        }
        return resolvedDescriptor
    }

    func synthesizeProjectChapter(
        modelID: ModelIdentifier,
        options: RenderSynthesisOptions,
        text: String
    ) async throws -> AudioChunk {
        let descriptor = try await speechSynthesisDescriptor(for: modelID)
        let configuration = try projectSynthesisConfiguration(for: descriptor)
        let request = SpeechSynthesisRequest(
            model: descriptor.id,
            text: text,
            language: options.normalizedLanguage,
            sampleRate: descriptor.defaultSampleRate ?? 24_000,
            responseFormat: "pcm_f32le",
            temperature: options.temperature.map(Float.init),
            topP: options.topP.map(Float.init),
            repetitionPenalty: options.repetitionPenalty.map(Float.init),
            maxTokens: options.maxTokens,
            voiceBehavior: options.voiceBehavior
        )

        do {
            return try await withReservedTextToSpeechWorkflowSession(
                descriptor: descriptor,
                configuration: configuration
            ) { reserved in
                try await reserved.workflow.synthesize(request: request, in: reserved.session)
            }
        } catch let error as WorkflowReservationError {
            switch error {
            case let .unsupportedTextToSpeech(identifier):
                throw ValarProjectSynthesisError.unsupportedSpeechSynthesis(identifier)
            default:
                throw error
            }
        }
    }

    private func projectSynthesisConfiguration(
        for descriptor: ModelDescriptor
    ) throws -> ModelRuntimeConfiguration {
        let policy = BackendSelectionPolicy()

        do {
            return try policy.runtimeConfiguration(
                for: descriptor,
                residencyPolicy: .automatic,
                runtime: backendSelectionRuntime()
            )
        } catch let error as BackendSelectionPolicy.SelectionError {
            switch error {
            case let .noCompatibleBackend(identifier):
                throw ValarProjectSynthesisError.noCompatibleBackend(identifier)
            }
        }
    }

}
