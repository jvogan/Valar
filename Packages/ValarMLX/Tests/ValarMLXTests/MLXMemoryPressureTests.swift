@preconcurrency import Dispatch
import Foundation
import Testing
@testable import ValarMLX
import ValarModelKit

// MARK: - MLXMemoryPressureObserver unit tests

@Suite("MLXMemoryPressureObserver")
struct MLXMemoryPressureObserverTests {

    @Test("waitUntilClear returns immediately when pressure is normal")
    func waitUntilClearPassesThroughWhenNormal() async throws {
        let observer = MLXMemoryPressureObserver(testingMode: true)
        // No pressure injected — waitUntilClear should return without suspending.
        try await observer.waitUntilClear()
        let underPressure = await observer.isUnderPressure
        #expect(!underPressure)
    }

    @Test("isUnderPressure is true after warning event")
    func isUnderPressureAfterWarning() async {
        let observer = MLXMemoryPressureObserver(testingMode: true)
        await observer.simulatePressureEvent(.warning)
        let underPressure = await observer.isUnderPressure
        #expect(underPressure)
    }

    @Test("isUnderPressure is true after critical event")
    func isUnderPressureAfterCritical() async {
        let observer = MLXMemoryPressureObserver(testingMode: true)
        await observer.simulatePressureEvent(.critical)
        let underPressure = await observer.isUnderPressure
        #expect(underPressure)
    }

    @Test("isUnderPressure is false after normal follows warning")
    func isUnderPressureClearsOnNormal() async {
        let observer = MLXMemoryPressureObserver(testingMode: true)
        await observer.simulatePressureEvent(.warning)
        await observer.simulatePressureEvent(.normal)
        let underPressure = await observer.isUnderPressure
        #expect(!underPressure)
    }

    @Test("waitUntilClear suspends under warning then resumes on normal")
    func waitUntilClearSuspendsAndResumeOnNormal() async throws {
        let observer = MLXMemoryPressureObserver(testingMode: true)
        await observer.simulatePressureEvent(.warning)

        let resumed = LockedFlag()
        let waitTask = Task {
            try await observer.waitUntilClear()
            resumed.set(true)
        }

        // Yield so waitTask can start and suspend inside waitUntilClear.
        await Task.yield()
        await Task.yield()
        #expect(!resumed.value)

        // Clear pressure — the waiter should resume.
        await observer.simulatePressureEvent(.normal)
        try await waitTask.value
        #expect(resumed.value)
    }

    @Test("pressureEvents stream emits warning event")
    func pressureEventsEmitsWarning() async throws {
        let observer = MLXMemoryPressureObserver(testingMode: true)

        let received = LockedBox<DispatchSource.MemoryPressureEvent?>(nil)
        let stream = observer.pressureEvents
        let collectTask = Task { @Sendable in
            for await rawEvent in stream {
                received.set(DispatchSource.MemoryPressureEvent(rawValue: rawEvent))
                break
            }
        }

        // Yield to let the collectTask reach its `for await`.
        await Task.yield()
        await observer.simulatePressureEvent(.warning)
        await Task.yield()
        await Task.yield()

        collectTask.cancel()
        #expect(received.value?.contains(.warning) == true)
    }

    @Test("pressureEvents stream emits critical event")
    func pressureEventsEmitsCritical() async throws {
        let observer = MLXMemoryPressureObserver(testingMode: true)

        let received = LockedBox<DispatchSource.MemoryPressureEvent?>(nil)
        let stream = observer.pressureEvents
        let collectTask = Task { @Sendable in
            for await rawEvent in stream {
                received.set(DispatchSource.MemoryPressureEvent(rawValue: rawEvent))
                break
            }
        }

        await Task.yield()
        await observer.simulatePressureEvent(.critical)
        await Task.yield()
        await Task.yield()

        collectTask.cancel()
        #expect(received.value?.contains(.critical) == true)
    }
}

// MARK: - Concurrency-safe shared state helpers

/// Thread-safe boolean flag using NSLock. `@unchecked Sendable` because
/// NSLock provides the required synchronisation.
private final class LockedFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var _value = false
    var value: Bool { lock.withLock { _value } }
    func set(_ v: Bool) { lock.withLock { _value = v } }
}

/// Thread-safe box for any value type. `@unchecked Sendable` — NSLock guards access.
private final class LockedBox<T: Sendable>: @unchecked Sendable {
    private let lock = NSLock()
    private var _value: T
    init(_ initial: T) { _value = initial }
    var value: T { lock.withLock { _value } }
    func set(_ v: T) { lock.withLock { _value = v } }
}

// MARK: - MLXInferenceBackend memory pressure integration tests

@Suite("MLXInferenceBackend memory pressure")
struct MLXBackendMemoryPressureTests {

    // MARK: - Helpers

    private func makeDescriptor(id: String, family: ModelFamilyID = .qwen3TTS) -> ModelDescriptor {
        ModelDescriptor(
            id: ModelIdentifier(id),
            familyID: family,
            displayName: id,
            domain: .tts,
            capabilities: [.speechSynthesis]
        )
    }

    private func makeBackend(
        observer: MLXMemoryPressureObserver,
        evictor: @escaping @Sendable (URL) -> Void = { _ in }
    ) -> MLXInferenceBackend {
        MLXInferenceBackend(
            modelDirectoryResolver: { _ in nil },
            warningHandler: { _ in },
            qwenModelLoader: { descriptor in MLXModelHandle(descriptor: descriptor) },
            speakerEncoderEvictor: evictor,
            memoryPressureObserver: observer
        )
    }

    private static let config = ModelRuntimeConfiguration(backendKind: .mlx)

    // MARK: - LRU tracking

    @Test("modelLoadOrder reflects insertion order after two loads")
    func loadOrderAfterTwoLoads() async throws {
        let observer = MLXMemoryPressureObserver(testingMode: true)
        let backend = makeBackend(observer: observer)

        let descA = makeDescriptor(id: "model-a")
        let descB = makeDescriptor(id: "model-b")

        _ = try await backend.loadModel(descriptor: descA, configuration: Self.config)
        _ = try await backend.loadModel(descriptor: descB, configuration: Self.config)

        let order = await backend.modelLoadOrderSnapshot
        #expect(order.count == 2)
        #expect(order.first?.rawValue == "model-a")
        #expect(order.last?.rawValue == "model-b")
    }

    @Test("Cache hit moves model to most-recently-used position")
    func cacheHitUpdateslruOrder() async throws {
        let observer = MLXMemoryPressureObserver(testingMode: true)
        let backend = makeBackend(observer: observer)

        let descA = makeDescriptor(id: "model-a")
        let descB = makeDescriptor(id: "model-b")

        _ = try await backend.loadModel(descriptor: descA, configuration: Self.config)
        _ = try await backend.loadModel(descriptor: descB, configuration: Self.config)

        // Re-access A — it should move to the end (MRU).
        _ = try await backend.loadModel(descriptor: descA, configuration: Self.config)

        let order = await backend.modelLoadOrderSnapshot
        #expect(order.last?.rawValue == "model-a")
        #expect(order.first?.rawValue == "model-b")
    }

    @Test("unloadModel removes model from LRU order")
    func unloadRemovesFromLoadOrder() async throws {
        let observer = MLXMemoryPressureObserver(testingMode: true)
        let backend = makeBackend(observer: observer)

        let descA = makeDescriptor(id: "model-a")
        let model = try await backend.loadModel(descriptor: descA, configuration: Self.config)
        try await backend.unloadModel(model)

        let order = await backend.modelLoadOrderSnapshot
        #expect(order.isEmpty)
    }

    // MARK: - Critical pressure eviction

    @Test("Critical pressure evicts the least-recently-used model")
    func criticalPressureEvictsLRU() async throws {
        let observer = MLXMemoryPressureObserver(testingMode: true)
        let backend = makeBackend(observer: observer)

        let descA = makeDescriptor(id: "lru-model")
        let descB = makeDescriptor(id: "mru-model")

        _ = try await backend.loadModel(descriptor: descA, configuration: Self.config)
        _ = try await backend.loadModel(descriptor: descB, configuration: Self.config)

        let countBefore = await backend.loadedModelCount
        #expect(countBefore == 2)

        // Yield to let the monitor task establish its `for await`.
        await Task.yield()

        await observer.simulatePressureEvent(.critical)

        // Give the monitor task time to process the event and hop to the actor.
        for _ in 0..<10 { await Task.yield() }

        let countAfter = await backend.loadedModelCount
        #expect(countAfter == 1)

        // The LRU model (lru-model) should be gone; MRU (mru-model) should remain.
        let remainingOrder = await backend.modelLoadOrderSnapshot
        #expect(remainingOrder.first?.rawValue == "mru-model")
    }

    @Test("Critical pressure on empty backend is a no-op")
    func criticalPressureOnEmptyBackendIsNoOp() async throws {
        let observer = MLXMemoryPressureObserver(testingMode: true)
        let backend = makeBackend(observer: observer)

        await Task.yield()
        await observer.simulatePressureEvent(.critical)
        for _ in 0..<10 { await Task.yield() }

        let count = await backend.loadedModelCount
        #expect(count == 0)
    }

    // MARK: - Warning pressure pauses loads

    @Test("Warning pressure causes isUnderPressure to be true on the observer")
    func warningPressureRaisesObserverFlag() async throws {
        let observer = MLXMemoryPressureObserver(testingMode: true)
        _ = makeBackend(observer: observer)

        await observer.simulatePressureEvent(.warning)
        let underPressure = await observer.isUnderPressure
        #expect(underPressure)
    }

    @Test("Normal after warning clears pressure and allows loads")
    func normalAfterWarningClearsAndAllowsLoad() async throws {
        let observer = MLXMemoryPressureObserver(testingMode: true)
        let backend = makeBackend(observer: observer)

        await observer.simulatePressureEvent(.warning)
        await observer.simulatePressureEvent(.normal)

        // With pressure cleared, load should complete without suspending.
        let desc = makeDescriptor(id: "post-normal-model")
        let model = try await backend.loadModel(descriptor: desc, configuration: Self.config)
        #expect(model.descriptor.id.rawValue == "post-normal-model")
    }
}
