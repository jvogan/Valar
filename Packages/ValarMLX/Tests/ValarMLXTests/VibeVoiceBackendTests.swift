import Foundation
import Testing
@testable import ValarMLX
import ValarModelKit

// MARK: - Helpers

private func makeVibeVoiceDescriptor(
    id: String = "test/vibevoice-realtime"
) -> ModelDescriptor {
    let backendRequirement = BackendRequirement(
        backendKind: .mlx,
        minimumMemoryBytes: nil
    )
    return ModelDescriptor(
        id: ModelIdentifier(id),
        familyID: .vibevoiceRealtimeTTS,
        displayName: "VibeVoice Realtime 0.5B",
        domain: .tts,
        capabilities: [.speechSynthesis, .presetVoices, .streaming],
        supportedBackends: [backendRequirement]
    )
}

private let config = ModelRuntimeConfiguration(backendKind: .mlx)

/// Thread-safe flag for tracking loader invocations in concurrent test closures.
private actor TestFlag {
    var value = false
    func set() { value = true }
}

/// Thread-safe counter for tracking loader call counts in concurrent test closures.
private actor TestCounter {
    var value = 0
    func increment() { value += 1 }
}

// MARK: - Tests

@Suite("VibeVoice Backend Integration")
struct VibeVoiceBackendTests {

    @Test("supportedFamilies includes vibevoiceRealtimeTTS")
    func supportedFamiliesContainsVibeVoice() {
        let backend = MLXInferenceBackend(
            modelDirectoryResolver: { _ in nil },
            warningHandler: { _ in },
            qwenModelLoader: { descriptor in MLXModelHandle(descriptor: descriptor) }
        )
        #expect(backend.runtimeCapabilities.supportedFamilies.contains(.vibevoiceRealtimeTTS))
    }

    @Test("VibeVoice loader is registered and invoked for vibevoiceRealtimeTTS")
    func vibeVoiceLoaderIsRegistered() async throws {
        let loaderCalled = TestFlag()
        let backend = MLXInferenceBackend(
            modelDirectoryResolver: { _ in nil },
            warningHandler: { _ in },
            qwenModelLoader: { descriptor in MLXModelHandle(descriptor: descriptor) },
            vibeVoiceTTSModelLoader: { descriptor in
                await loaderCalled.set()
                return MLXModelHandle(descriptor: descriptor)
            }
        )

        let descriptor = makeVibeVoiceDescriptor()
        _ = try await backend.loadModel(descriptor: descriptor, configuration: config)
        #expect(await loaderCalled.value)
    }

    @Test("VibeVoice does not use the Qwen loader")
    func vibeVoiceDoesNotUseQwenLoader() async throws {
        let qwenLoaderCalled = TestFlag()
        let backend = MLXInferenceBackend(
            modelDirectoryResolver: { _ in nil },
            warningHandler: { _ in },
            qwenModelLoader: { descriptor in
                await qwenLoaderCalled.set()
                return MLXModelHandle(descriptor: descriptor)
            },
            vibeVoiceTTSModelLoader: { descriptor in
                MLXModelHandle(descriptor: descriptor)
            }
        )

        let descriptor = makeVibeVoiceDescriptor()
        _ = try await backend.loadModel(descriptor: descriptor, configuration: config)
        #expect(await !qwenLoaderCalled.value)
    }

    @Test("VibeVoice residency tracking works after load")
    func residencyTrackingAfterLoad() async throws {
        let backend = MLXInferenceBackend(
            modelDirectoryResolver: { _ in nil },
            warningHandler: { _ in },
            qwenModelLoader: { descriptor in MLXModelHandle(descriptor: descriptor) },
            vibeVoiceTTSModelLoader: { descriptor in
                MLXModelHandle(descriptor: descriptor)
            }
        )

        let descriptor = makeVibeVoiceDescriptor()
        _ = try await backend.loadModel(descriptor: descriptor, configuration: config)

        let snapshot = await backend.residencySnapshot()
        #expect(snapshot.count == 1)
        #expect(snapshot[0].modelId == descriptor.id)
        #expect(snapshot[0].sessionCount == 1)
    }

    @Test("VibeVoice prewarm is non-blocking")
    func prewarmIsNonBlocking() async {
        let loaderDelay = Duration.milliseconds(200)
        let backend = MLXInferenceBackend(
            modelDirectoryResolver: { _ in nil },
            warningHandler: { _ in },
            qwenModelLoader: { descriptor in MLXModelHandle(descriptor: descriptor) },
            vibeVoiceTTSModelLoader: { descriptor in
                try await Task.sleep(for: loaderDelay)
                return MLXModelHandle(descriptor: descriptor)
            }
        )

        let clock = ContinuousClock()
        let elapsed = await clock.measure {
            await backend.prewarm(
                descriptor: makeVibeVoiceDescriptor(),
                configuration: config
            )
        }

        #expect(elapsed < .milliseconds(100))
    }

    @Test("VibeVoice warm load after prewarm skips loader")
    func warmLoadSkipsLoader() async throws {
        let loaderCallCount = TestCounter()
        let backend = MLXInferenceBackend(
            modelDirectoryResolver: { _ in nil },
            warningHandler: { _ in },
            qwenModelLoader: { descriptor in MLXModelHandle(descriptor: descriptor) },
            vibeVoiceTTSModelLoader: { descriptor in
                await loaderCallCount.increment()
                return MLXModelHandle(descriptor: descriptor)
            }
        )

        let descriptor = makeVibeVoiceDescriptor()

        // Prewarm then wait for completion
        await backend.prewarm(descriptor: descriptor, configuration: config)
        try await Task.sleep(for: .milliseconds(50))

        // Load should hit cache
        _ = try await backend.loadModel(descriptor: descriptor, configuration: config)
        #expect(await loaderCallCount.value == 1)
    }
}
