import Foundation
import Testing
@testable import ValarMLX
import ValarModelKit

// MARK: - Helpers

private func makeBackend(
    metalAllocatedSizeProvider: @escaping @Sendable () -> UInt64? = { nil }
) -> MLXInferenceBackend {
    MLXInferenceBackend(
        modelDirectoryResolver: { _ in nil },
        warningHandler: { _ in },
        qwenModelLoader: { descriptor in MLXModelHandle(descriptor: descriptor) },
        metalAllocatedSizeProvider: metalAllocatedSizeProvider
    )
}

private func makeTTSDescriptor(
    id: String = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
    minimumMemoryBytes: Int? = nil
) -> ModelDescriptor {
    let backendRequirement = BackendRequirement(
        backendKind: .mlx,
        minimumMemoryBytes: minimumMemoryBytes
    )
    return ModelDescriptor(
        id: ModelIdentifier(id),
        familyID: .qwen3TTS,
        displayName: "Test Model",
        domain: .tts,
        capabilities: [.speechSynthesis],
        supportedBackends: [backendRequirement]
    )
}

private let config = ModelRuntimeConfiguration(backendKind: .mlx)

// MARK: - Tests

@Suite("MLX Residency")
struct MLXResidencyTests {

    @Test("Residency snapshot is empty when no models are loaded")
    func residencySnapshotEmptyInitially() async {
        let backend = makeBackend()
        let snapshot = await backend.residencySnapshot()
        #expect(snapshot.isEmpty)
    }

    @Test("Residency snapshot contains one entry after loadModel")
    func residencySnapshotAfterLoad() async throws {
        let backend = makeBackend()
        let descriptor = makeTTSDescriptor()
        _ = try await backend.loadModel(descriptor: descriptor, configuration: config)

        let snapshot = await backend.residencySnapshot()
        #expect(snapshot.count == 1)
        #expect(snapshot[0].modelId == descriptor.id)
    }

    @Test("residentSince is set on first load and not changed on subsequent loads")
    func residentSinceIsStable() async throws {
        let backend = makeBackend()
        let descriptor = makeTTSDescriptor()
        _ = try await backend.loadModel(descriptor: descriptor, configuration: config)

        let firstSnapshot = await backend.residencySnapshot()
        guard let first = firstSnapshot.first else {
            Issue.record("Expected residency entry after first load")
            return
        }
        let residentSince = first.residentSince

        // Second load is a cache hit — residentSince must not change.
        _ = try await backend.loadModel(descriptor: descriptor, configuration: config)
        let secondSnapshot = await backend.residencySnapshot()
        guard let second = secondSnapshot.first else {
            Issue.record("Expected residency entry after second load")
            return
        }
        #expect(second.residentSince == residentSince)
    }

    @Test("sessionCount increments on each loadModel call")
    func sessionCountIncrements() async throws {
        let backend = makeBackend()
        let descriptor = makeTTSDescriptor()

        _ = try await backend.loadModel(descriptor: descriptor, configuration: config)
        let snap1 = await backend.residencySnapshot()
        #expect(snap1.first?.sessionCount == 1)

        _ = try await backend.loadModel(descriptor: descriptor, configuration: config)
        let snap2 = await backend.residencySnapshot()
        #expect(snap2.first?.sessionCount == 2)

        _ = try await backend.loadModel(descriptor: descriptor, configuration: config)
        let snap3 = await backend.residencySnapshot()
        #expect(snap3.first?.sessionCount == 3)
    }

    @Test("lastUsedAt is updated on each loadModel call")
    func lastUsedAtUpdates() async throws {
        let backend = makeBackend()
        let descriptor = makeTTSDescriptor()
        _ = try await backend.loadModel(descriptor: descriptor, configuration: config)

        let snap1 = await backend.residencySnapshot()
        let firstAccess = snap1.first?.lastUsedAt

        // A tiny sleep so the clock can tick.
        try await Task.sleep(nanoseconds: 1_000_000)

        _ = try await backend.loadModel(descriptor: descriptor, configuration: config)
        let snap2 = await backend.residencySnapshot()
        let secondAccess = snap2.first?.lastUsedAt

        if let t1 = firstAccess, let t2 = secondAccess {
            #expect(t2 >= t1)
        }
    }

    @Test("Residency snapshot is empty after unloadModel")
    func residencyRemovedOnUnload() async throws {
        let backend = makeBackend()
        let descriptor = makeTTSDescriptor()
        let model = try await backend.loadModel(descriptor: descriptor, configuration: config)

        let snapBefore = await backend.residencySnapshot()
        #expect(snapBefore.count == 1)

        try await backend.unloadModel(model)
        let snapAfter = await backend.residencySnapshot()
        #expect(snapAfter.isEmpty)
    }

    @Test("estimatedBytes uses descriptor minimumMemoryBytes when Metal is unavailable")
    func estimatedBytesFromDescriptor() async throws {
        let backend = makeBackend(metalAllocatedSizeProvider: { nil })
        let descriptor = makeTTSDescriptor(minimumMemoryBytes: 4_294_967_296) // 4 GB
        _ = try await backend.loadModel(descriptor: descriptor, configuration: config)

        let snapshot = await backend.residencySnapshot()
        #expect(snapshot.first?.estimatedBytes == 4_294_967_296)
    }

    @Test("estimatedBytes is nil when Metal unavailable and descriptor has no minimumMemoryBytes")
    func estimatedBytesNilWhenNoData() async throws {
        let backend = makeBackend(metalAllocatedSizeProvider: { nil })
        let descriptor = makeTTSDescriptor(minimumMemoryBytes: nil)
        _ = try await backend.loadModel(descriptor: descriptor, configuration: config)

        let snapshot = await backend.residencySnapshot()
        #expect(snapshot.first?.estimatedBytes == nil)
    }

    @Test("estimatedBytes prefers Metal allocation delta over descriptor minimumMemoryBytes")
    func estimatedBytesPrefersMetal() async throws {
        // Simulate Metal reporting 2 GB before the load and 4 GB after.
        // Use @unchecked Sendable class so the sequential provider can mutate its state.
        final class SequentialValues: @unchecked Sendable {
            private var remaining: [UInt64?]
            init(_ values: [UInt64?]) { remaining = values }
            func next() -> UInt64? { remaining.isEmpty ? nil : remaining.removeFirst() }
        }
        let seq = SequentialValues([2_147_483_648, 4_294_967_296])
        let metalProvider: @Sendable () -> UInt64? = { seq.next() }

        let backend = makeBackend(metalAllocatedSizeProvider: metalProvider)
        let descriptor = makeTTSDescriptor(minimumMemoryBytes: 1_073_741_824) // 1 GB — should be ignored
        _ = try await backend.loadModel(descriptor: descriptor, configuration: config)

        let snapshot = await backend.residencySnapshot()
        // Delta = 4 GB - 2 GB = 2 GB
        #expect(snapshot.first?.estimatedBytes == 2_147_483_648)
    }

    @Test("Residency tracks multiple models independently")
    func multipleModelResidency() async throws {
        let backend = MLXInferenceBackend(
            modelDirectoryResolver: { _ in nil },
            warningHandler: { _ in },
            qwenModelLoader: { descriptor in MLXModelHandle(descriptor: descriptor) },
            qwenASRModelLoader: { descriptor in MLXASRModelHandle(descriptor: descriptor) }
        )
        let ttsDescriptor = makeTTSDescriptor(id: "org/model-a")
        let asrDescriptor = ModelDescriptor(
            id: "org/model-b",
            familyID: .qwen3ASR,
            displayName: "ASR Model",
            domain: .stt,
            capabilities: [.speechRecognition]
        )

        _ = try await backend.loadModel(descriptor: ttsDescriptor, configuration: config)
        _ = try await backend.loadModel(descriptor: asrDescriptor, configuration: config)

        let snapshot = await backend.residencySnapshot()
        #expect(snapshot.count == 2)

        let ids = Set(snapshot.map(\.modelId))
        #expect(ids.contains(ttsDescriptor.id))
        #expect(ids.contains(asrDescriptor.id))
    }

    @Test("ResidencyInfo fields match expected values after load")
    func residencyInfoFields() async throws {
        let before = Date()
        let backend = makeBackend(metalAllocatedSizeProvider: { nil })
        let descriptor = makeTTSDescriptor(minimumMemoryBytes: 2_048)
        _ = try await backend.loadModel(descriptor: descriptor, configuration: config)
        let after = Date()

        let snapshot = await backend.residencySnapshot()
        guard let info = snapshot.first else {
            Issue.record("Expected residency info after load")
            return
        }
        #expect(info.modelId == descriptor.id)
        #expect(info.residentSince >= before)
        #expect(info.residentSince <= after)
        #expect(info.lastUsedAt >= before)
        #expect(info.sessionCount == 1)
        #expect(info.estimatedBytes == 2_048)
    }
}
