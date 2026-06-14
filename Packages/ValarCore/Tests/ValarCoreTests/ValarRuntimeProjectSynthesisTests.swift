import Foundation
import XCTest
@testable import ValarCore
import ValarModelKit
import ValarPersistence

final class ValarRuntimeProjectSynthesisTests: XCTestCase {
    func testSynthesizeProjectChapterBuildsSharedRequest() async throws {
        let baseURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        defer { try? FileManager.default.removeItem(at: baseURL) }

        let tracker = ProjectSynthesisRequestTracker()
        let runtime = try ValarRuntime(
            paths: ValarAppPaths(baseURL: baseURL),
            runtimeConfiguration: RuntimeConfiguration(),
            inferenceBackend: ProjectSynthesisStubBackend(tracker: tracker)
        )
        let descriptor = ModelDescriptor(
            id: "test/project-tts",
            familyID: .qwen3TTS,
            displayName: "Project TTS",
            domain: .tts,
            capabilities: [.speechSynthesis],
            supportedBackends: [BackendRequirement(backendKind: .mlx)],
            defaultSampleRate: 44_100
        )
        await runtime.modelRegistry.register(descriptor)

        let options = RenderSynthesisOptions(
            language: "es",
            temperature: 0.62,
            topP: 0.91,
            repetitionPenalty: 1.07,
            maxTokens: 1_024,
            voiceBehavior: .stableNarrator
        )

        let chunk = try await runtime.synthesizeProjectChapter(
            modelID: descriptor.id,
            options: options,
            text: "Shared project render path"
        )
        let firstRequest = await tracker.firstRequest()
        let request = try XCTUnwrap(firstRequest)

        XCTAssertEqual(chunk, AudioChunk(samples: [0, 0.25, -0.25], sampleRate: 44_100))
        XCTAssertEqual(request.model, descriptor.id)
        XCTAssertEqual(request.text, "Shared project render path")
        XCTAssertEqual(request.language, "es")
        XCTAssertEqual(request.sampleRate, 44_100)
        XCTAssertEqual(request.responseFormat, "pcm_f32le")
        XCTAssertEqual(request.temperature, Float(0.62))
        XCTAssertEqual(request.topP, Float(0.91))
        XCTAssertEqual(request.repetitionPenalty, Float(1.07))
        XCTAssertEqual(request.maxTokens, 1_024)
        XCTAssertEqual(request.voiceBehavior, .stableNarrator)
    }

    func testSpeechSynthesisDescriptorRejectsUnsupportedModel() async throws {
        let baseURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        defer { try? FileManager.default.removeItem(at: baseURL) }

        let runtime = try ValarRuntime(
            paths: ValarAppPaths(baseURL: baseURL),
            runtimeConfiguration: RuntimeConfiguration(),
            inferenceBackend: ProjectSynthesisStubBackend(tracker: ProjectSynthesisRequestTracker())
        )
        let descriptor = ModelDescriptor(
            id: "test/asr-only",
            familyID: .qwen3ASR,
            displayName: "ASR Only",
            domain: .stt,
            capabilities: [.speechRecognition],
            supportedBackends: [BackendRequirement(backendKind: .mlx)]
        )
        await runtime.modelRegistry.register(descriptor)

        do {
            _ = try await runtime.speechSynthesisDescriptor(for: descriptor.id)
            XCTFail("Expected unsupported speech synthesis error")
        } catch let error as ValarProjectSynthesisError {
            XCTAssertEqual(error, .unsupportedSpeechSynthesis(descriptor.id))
        }
    }
}

private actor ProjectSynthesisRequestTracker {
    private var requests: [SpeechSynthesisRequest] = []

    func record(_ request: SpeechSynthesisRequest) {
        requests.append(request)
    }

    func firstRequest() -> SpeechSynthesisRequest? {
        requests.first
    }
}

private struct ProjectSynthesisStubBackend: InferenceBackend {
    let tracker: ProjectSynthesisRequestTracker

    var backendKind: BackendKind { .mlx }
    var runtimeCapabilities: BackendCapabilities { BackendCapabilities() }

    func validate(requirement: BackendRequirement) async throws {}

    func loadModel(
        descriptor: ModelDescriptor,
        configuration: ModelRuntimeConfiguration
    ) async throws -> any ValarModel {
        ProjectSynthesisStubModel(
            descriptor: descriptor,
            backendKind: configuration.backendKind,
            tracker: tracker
        )
    }

    func unloadModel(_ model: any ValarModel) async throws {}
}

private struct ProjectSynthesisStubModel: TextToSpeechWorkflow {
    let descriptor: ModelDescriptor
    let backendKind: BackendKind
    let tracker: ProjectSynthesisRequestTracker

    func synthesize(
        request: SpeechSynthesisRequest,
        in session: ModelRuntimeSession
    ) async throws -> AudioChunk {
        await tracker.record(request)
        return AudioChunk(samples: [0, 0.25, -0.25], sampleRate: request.sampleRate)
    }

    func synthesizeStream(
        request: SpeechSynthesisRequest
    ) async throws -> AsyncThrowingStream<AudioChunk, Error> {
        AsyncThrowingStream { continuation in
            continuation.finish()
        }
    }
}
