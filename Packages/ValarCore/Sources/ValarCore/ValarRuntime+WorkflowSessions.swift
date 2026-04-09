import Foundation
import ValarMLX
import ValarModelKit

actor ReservedWorkflowCleaner {
    private let cleanup: @Sendable () async -> Void
    private var finished = false

    init(cleanup: @escaping @Sendable () async -> Void) {
        self.cleanup = cleanup
    }

    func finish() async {
        guard finished == false else { return }
        finished = true
        await cleanup()
    }
}

public enum WorkflowReservationError: LocalizedError, Sendable {
    case unsupportedTextToSpeech(ModelIdentifier)
    case unsupportedSpeechToText(ModelIdentifier)
    case unsupportedForcedAlignment(ModelIdentifier)

    public var errorDescription: String? {
        switch self {
        case .unsupportedTextToSpeech(let identifier):
            return "Loaded model \(identifier.rawValue) does not provide a text-to-speech workflow."
        case .unsupportedSpeechToText(let identifier):
            return "Loaded model \(identifier.rawValue) does not provide a speech-to-text workflow."
        case .unsupportedForcedAlignment(let identifier):
            return "Loaded model \(identifier.rawValue) does not provide a forced-alignment workflow."
        }
    }
}

public struct ReservedTextToSpeechWorkflowSession: Sendable {
    public let descriptor: ModelDescriptor
    public let configuration: ModelRuntimeConfiguration
    public let session: ModelRuntimeSession
    public let workflow: any TextToSpeechWorkflow

    private let cleaner: ReservedWorkflowCleaner

    init(
        descriptor: ModelDescriptor,
        configuration: ModelRuntimeConfiguration,
        session: ModelRuntimeSession,
        workflow: any TextToSpeechWorkflow,
        cleaner: ReservedWorkflowCleaner
    ) {
        self.descriptor = descriptor
        self.configuration = configuration
        self.session = session
        self.workflow = workflow
        self.cleaner = cleaner
    }

    public func finish() async {
        await cleaner.finish()
    }
}

public struct ReservedSpeechToTextWorkflowSession: Sendable {
    public let descriptor: ModelDescriptor
    public let configuration: ModelRuntimeConfiguration
    public let session: ModelRuntimeSession
    public let workflow: any SpeechToTextWorkflow

    private let cleaner: ReservedWorkflowCleaner

    init(
        descriptor: ModelDescriptor,
        configuration: ModelRuntimeConfiguration,
        session: ModelRuntimeSession,
        workflow: any SpeechToTextWorkflow,
        cleaner: ReservedWorkflowCleaner
    ) {
        self.descriptor = descriptor
        self.configuration = configuration
        self.session = session
        self.workflow = workflow
        self.cleaner = cleaner
    }

    public func finish() async {
        await cleaner.finish()
    }
}

public struct ReservedForcedAlignmentWorkflowSession: Sendable {
    public let descriptor: ModelDescriptor
    public let configuration: ModelRuntimeConfiguration
    public let session: ModelRuntimeSession
    public let workflow: any ForcedAlignmentWorkflow

    private let cleaner: ReservedWorkflowCleaner

    init(
        descriptor: ModelDescriptor,
        configuration: ModelRuntimeConfiguration,
        session: ModelRuntimeSession,
        workflow: any ForcedAlignmentWorkflow,
        cleaner: ReservedWorkflowCleaner
    ) {
        self.descriptor = descriptor
        self.configuration = configuration
        self.session = session
        self.workflow = workflow
        self.cleaner = cleaner
    }

    public func finish() async {
        await cleaner.finish()
    }
}

public extension ValarRuntime {
    func reserveTextToSpeechWorkflowSession(
        descriptor: ModelDescriptor,
        configuration: ModelRuntimeConfiguration
    ) async throws -> ReservedTextToSpeechWorkflowSession {
        let reserved = try await reserveModelWorkflowSession(
            descriptor: descriptor,
            configuration: configuration
        )
        guard let workflow = reserved.model as? any TextToSpeechWorkflow else {
            await reserved.cleaner.finish()
            throw WorkflowReservationError.unsupportedTextToSpeech(descriptor.id)
        }
        return ReservedTextToSpeechWorkflowSession(
            descriptor: descriptor,
            configuration: configuration,
            session: reserved.session,
            workflow: workflow,
            cleaner: reserved.cleaner
        )
    }

    func withReservedTextToSpeechWorkflowSession<Result: Sendable>(
        descriptor: ModelDescriptor,
        configuration: ModelRuntimeConfiguration,
        operation: @escaping @Sendable (ReservedTextToSpeechWorkflowSession) async throws -> Result
    ) async throws -> Result {
        let reserved = try await reserveTextToSpeechWorkflowSession(
            descriptor: descriptor,
            configuration: configuration
        )
        do {
            let result = try await operation(reserved)
            await reserved.finish()
            return result
        } catch {
            await reserved.finish()
            throw error
        }
    }

    func withReservedTextToSpeechWorkflowSessionStream<Element: Sendable>(
        descriptor: ModelDescriptor,
        configuration: ModelRuntimeConfiguration,
        operation: @escaping @Sendable (ReservedTextToSpeechWorkflowSession) async throws -> AsyncThrowingStream<Element, Error>
    ) async throws -> AsyncThrowingStream<Element, Error> {
        let reserved = try await reserveTextToSpeechWorkflowSession(
            descriptor: descriptor,
            configuration: configuration
        )
        do {
            let baseStream = try await operation(reserved)
            return AsyncThrowingStream { continuation in
                let task = Task {
                    do {
                        for try await element in baseStream {
                            continuation.yield(element)
                        }
                        await reserved.finish()
                        continuation.finish()
                    } catch is CancellationError {
                        await reserved.finish()
                        continuation.finish(throwing: CancellationError())
                    } catch {
                        await reserved.finish()
                        continuation.finish(throwing: error)
                    }
                }

                continuation.onTermination = { @Sendable _ in
                    task.cancel()
                }
            }
        } catch {
            await reserved.finish()
            throw error
        }
    }

    func reserveSpeechToTextWorkflowSession(
        descriptor: ModelDescriptor,
        configuration: ModelRuntimeConfiguration
    ) async throws -> ReservedSpeechToTextWorkflowSession {
        let reserved = try await reserveModelWorkflowSession(
            descriptor: descriptor,
            configuration: configuration
        )
        guard let workflow = reserved.model as? any SpeechToTextWorkflow else {
            await reserved.cleaner.finish()
            throw WorkflowReservationError.unsupportedSpeechToText(descriptor.id)
        }
        return ReservedSpeechToTextWorkflowSession(
            descriptor: descriptor,
            configuration: configuration,
            session: reserved.session,
            workflow: workflow,
            cleaner: reserved.cleaner
        )
    }

    func withReservedSpeechToTextWorkflowSession<Result: Sendable>(
        descriptor: ModelDescriptor,
        configuration: ModelRuntimeConfiguration,
        operation: @escaping @Sendable (ReservedSpeechToTextWorkflowSession) async throws -> Result
    ) async throws -> Result {
        let reserved = try await reserveSpeechToTextWorkflowSession(
            descriptor: descriptor,
            configuration: configuration
        )
        do {
            let result = try await operation(reserved)
            await reserved.finish()
            return result
        } catch {
            await reserved.finish()
            throw error
        }
    }

    func withReservedSpeechToTextWorkflowSessionStream<Element: Sendable>(
        descriptor: ModelDescriptor,
        configuration: ModelRuntimeConfiguration,
        operation: @escaping @Sendable (ReservedSpeechToTextWorkflowSession) async throws -> AsyncThrowingStream<Element, Error>
    ) async throws -> AsyncThrowingStream<Element, Error> {
        let reserved = try await reserveSpeechToTextWorkflowSession(
            descriptor: descriptor,
            configuration: configuration
        )
        do {
            let baseStream = try await operation(reserved)
            return AsyncThrowingStream { continuation in
                let task = Task {
                    do {
                        for try await element in baseStream {
                            continuation.yield(element)
                        }
                        await reserved.finish()
                        continuation.finish()
                    } catch is CancellationError {
                        await reserved.finish()
                        continuation.finish(throwing: CancellationError())
                    } catch {
                        await reserved.finish()
                        continuation.finish(throwing: error)
                    }
                }

                continuation.onTermination = { @Sendable _ in
                    task.cancel()
                }
            }
        } catch {
            await reserved.finish()
            throw error
        }
    }

    func reserveForcedAlignmentWorkflowSession(
        descriptor: ModelDescriptor,
        configuration: ModelRuntimeConfiguration
    ) async throws -> ReservedForcedAlignmentWorkflowSession {
        let reserved = try await reserveModelWorkflowSession(
            descriptor: descriptor,
            configuration: configuration
        )
        guard let workflow = reserved.model as? any ForcedAlignmentWorkflow else {
            await reserved.cleaner.finish()
            throw WorkflowReservationError.unsupportedForcedAlignment(descriptor.id)
        }
        return ReservedForcedAlignmentWorkflowSession(
            descriptor: descriptor,
            configuration: configuration,
            session: reserved.session,
            workflow: workflow,
            cleaner: reserved.cleaner
        )
    }

    func withReservedForcedAlignmentWorkflowSession<Result: Sendable>(
        descriptor: ModelDescriptor,
        configuration: ModelRuntimeConfiguration,
        operation: @escaping @Sendable (ReservedForcedAlignmentWorkflowSession) async throws -> Result
    ) async throws -> Result {
        let reserved = try await reserveForcedAlignmentWorkflowSession(
            descriptor: descriptor,
            configuration: configuration
        )
        do {
            let result = try await operation(reserved)
            await reserved.finish()
            return result
        } catch {
            await reserved.finish()
            throw error
        }
    }
}

extension ValarRuntime {
    struct ReservedModelWorkflowSession: Sendable {
        let model: any ValarModel
        let session: ModelRuntimeSession
        let cleaner: ReservedWorkflowCleaner
    }

    func reserveModelWorkflowSession(
        descriptor: ModelDescriptor,
        configuration: ModelRuntimeConfiguration
    ) async throws -> ReservedModelWorkflowSession {
        await modelRegistry.register(
            descriptor,
            estimatedResidentBytes: nil,
            runtimeConfiguration: configuration
        )
        await capabilityRegistry.register(descriptor)
        await modelRegistry.markState(.warming, for: descriptor.id)

        let model: any ValarModel
        do {
            model = try await inferenceBackend.loadModel(
                descriptor: descriptor,
                configuration: configuration
            )
        } catch {
            await modelRegistry.markState(.unloaded, for: descriptor.id)
            throw error
        }

        await modelRegistry.markState(.resident, for: descriptor.id)
        await modelRegistry.reserveSession(for: descriptor.id)
        await (inferenceBackend as? MLXInferenceBackend)?.beginSession(for: descriptor.id)

        let cleaner = ReservedWorkflowCleaner { [inferenceBackend, modelRegistry] in
            await (inferenceBackend as? MLXInferenceBackend)?.endSession(for: descriptor.id)
            await modelRegistry.releaseSession(for: descriptor.id)
            await modelRegistry.touch(descriptor.id)
        }

        let session = ModelRuntimeSession(
            descriptor: descriptor,
            backendKind: configuration.backendKind,
            configuration: configuration,
            state: .resident
        )
        return ReservedModelWorkflowSession(
            model: model,
            session: session,
            cleaner: cleaner
        )
    }
}
