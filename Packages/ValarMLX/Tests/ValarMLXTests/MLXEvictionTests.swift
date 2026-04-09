import Foundation
import Testing
@testable import ValarMLX
import ValarModelKit

// MARK: - Helpers

private func makeTTSDescriptor(
    id: String = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
    minimumMemoryBytes: Int? = nil
) -> ModelDescriptor {
    ModelDescriptor(
        id: ModelIdentifier(id),
        familyID: .qwen3TTS,
        displayName: "Test TTS Model",
        domain: .tts,
        capabilities: [.speechSynthesis],
        supportedBackends: [BackendRequirement(backendKind: .mlx, minimumMemoryBytes: minimumMemoryBytes)]
    )
}

private func makeASRDescriptor(
    id: String = "mlx-community/Qwen3-ASR-Base",
    minimumMemoryBytes: Int? = nil
) -> ModelDescriptor {
    ModelDescriptor(
        id: ModelIdentifier(id),
        familyID: .qwen3ASR,
        displayName: "Test ASR Model",
        domain: .stt,
        capabilities: [.speechRecognition],
        supportedBackends: [BackendRequirement(backendKind: .mlx, minimumMemoryBytes: minimumMemoryBytes)]
    )
}

private func makeBackend(
    evictionPolicy: EvictionPolicy = .default
) -> MLXInferenceBackend {
    MLXInferenceBackend(
        modelDirectoryResolver: { _ in nil },
        warningHandler: { _ in },
        qwenModelLoader: { descriptor in MLXModelHandle(descriptor: descriptor) },
        qwenASRModelLoader: { descriptor in MLXASRModelHandle(descriptor: descriptor) },
        evictionPolicy: evictionPolicy
    )
}

private let config = ModelRuntimeConfiguration(backendKind: .mlx)

// MARK: - Tests

@Suite("MLX Eviction Policy", .serialized)
struct MLXEvictionTests {

    // MARK: Budget-based eviction

    @Test("evictIfNeeded with no policy set is a no-op")
    func evictIfNeededWithDefaultPolicyIsNoop() async throws {
        let backend = makeBackend(evictionPolicy: .default)
        let descriptor = makeTTSDescriptor(minimumMemoryBytes: 1_000_000)
        _ = try await backend.loadModel(descriptor: descriptor, configuration: config)

        let evicted = await backend.evictIfNeeded()
        #expect(evicted.isEmpty)
        #expect(await backend.loadedModelCount == 1)
    }

    @Test("evictIfNeeded does not evict when total bytes is within budget")
    func evictIfNeededUnderBudgetDoesNothing() async throws {
        let budget = 10_000_000
        let backend = makeBackend(evictionPolicy: EvictionPolicy(maxResidentBytes: budget))
        let descriptor = makeTTSDescriptor(minimumMemoryBytes: 1_000_000) // well under budget
        _ = try await backend.loadModel(descriptor: descriptor, configuration: config)

        let evicted = await backend.evictIfNeeded()
        #expect(evicted.isEmpty)
        #expect(await backend.loadedModelCount == 1)
    }

    @Test("evictIfNeeded evicts LRU model when total bytes exceeds budget")
    func evictIfNeededEvictsLRUWhenOverBudget() async throws {
        // Load two 3 GB models; budget is 4 GB → one must go.
        let threeGB = 3_221_225_472
        let fourGB = 4_294_967_296
        let lruDescriptor = makeTTSDescriptor(id: "org/model-lru", minimumMemoryBytes: threeGB)
        let mruDescriptor = makeTTSDescriptor(id: "org/model-mru", minimumMemoryBytes: threeGB)

        let backend = makeBackend(evictionPolicy: EvictionPolicy(maxResidentBytes: fourGB))
        _ = try await backend.loadModel(descriptor: lruDescriptor, configuration: config)
        _ = try await backend.loadModel(descriptor: mruDescriptor, configuration: config)
        // mruDescriptor is now MRU; lruDescriptor is LRU (index 0 in modelLoadOrder)

        let evicted = await backend.evictIfNeeded()
        #expect(evicted == [lruDescriptor.id])
        #expect(await backend.loadedModelCount == 1)

        // Verify the MRU model survived.
        let snapshot = await backend.residencySnapshot()
        #expect(snapshot.first?.modelId == mruDescriptor.id)
    }

    @Test("evictIfNeeded evicts multiple models until under budget")
    func evictIfNeededEvictsMultipleModelsUntilUnderBudget() async throws {
        // Three 2 GB models; budget is 2 GB → two must go.
        let twoGB = 2_147_483_648
        let descriptors = (1...3).map {
            makeTTSDescriptor(id: "org/model-\($0)", minimumMemoryBytes: twoGB)
        }

        let backend = makeBackend(evictionPolicy: EvictionPolicy(maxResidentBytes: twoGB))
        for d in descriptors {
            _ = try await backend.loadModel(descriptor: d, configuration: config)
        }

        let evicted = await backend.evictIfNeeded()
        // Two LRU models (model-1 and model-2) should be evicted; model-3 (MRU) survives.
        #expect(evicted.count == 2)
        #expect(Set(evicted) == Set([descriptors[0].id, descriptors[1].id]))
        #expect(await backend.loadedModelCount == 1)
    }

    // MARK: ASR eviction priority

    @Test("evictIfNeeded evicts ASR before TTS when budget forces one eviction")
    func evictIfNeededEvictsASRBeforeTTS() async throws {
        // Load TTS first (LRU), then ASR (MRU). Budget requires evicting one model.
        // Despite TTS being LRU, ASR's higher eviction priority means ASR goes first.
        let threeGB = 3_221_225_472
        let fourGB = 4_294_967_296
        let ttsDescriptor = makeTTSDescriptor(minimumMemoryBytes: threeGB)
        let asrDescriptor = makeASRDescriptor(minimumMemoryBytes: threeGB)

        let backend = makeBackend(evictionPolicy: EvictionPolicy(maxResidentBytes: fourGB))
        _ = try await backend.loadModel(descriptor: ttsDescriptor, configuration: config)
        _ = try await backend.loadModel(descriptor: asrDescriptor, configuration: config)
        // Order: ttsDescriptor=LRU, asrDescriptor=MRU

        let evicted = await backend.evictIfNeeded()
        // ASR is evicted even though it's the MRU — it has higher eviction priority.
        #expect(evicted == [asrDescriptor.id])
        #expect(await backend.loadedModelCount == 1)

        let snapshot = await backend.residencySnapshot()
        #expect(snapshot.first?.modelId == ttsDescriptor.id)
    }

    @Test("evictIfNeeded evicts ASR before TTS even when ASR is MRU and TTS is LRU")
    func evictIfNeededASRPriorityIgnoresLRUOrder() async throws {
        // Three models: TTS-a (oldest), TTS-b (middle), ASR (newest MRU).
        // Budget is 2 GB → forces eviction of 2 out of 3 (total 6 GB).
        // ASR should be evicted first despite being MRU, then LRU TTS (ttsA).
        let twoGB = 2_147_483_648
        let ttsA = makeTTSDescriptor(id: "org/tts-a", minimumMemoryBytes: twoGB)
        let ttsB = makeTTSDescriptor(id: "org/tts-b", minimumMemoryBytes: twoGB)
        let asrC = makeASRDescriptor(id: "org/asr-c", minimumMemoryBytes: twoGB)

        let backend = makeBackend(evictionPolicy: EvictionPolicy(maxResidentBytes: twoGB))
        _ = try await backend.loadModel(descriptor: ttsA, configuration: config)
        _ = try await backend.loadModel(descriptor: ttsB, configuration: config)
        _ = try await backend.loadModel(descriptor: asrC, configuration: config)
        // Order (LRU→MRU): ttsA, ttsB, asrC — total 6 GB, budget 2 GB → 2 models must go

        let evicted = await backend.evictIfNeeded()
        // ASR (asrC) evicted first, then the LRU TTS (ttsA)
        #expect(evicted.count == 2)
        #expect(evicted[0] == asrC.id)
        #expect(evicted[1] == ttsA.id)
    }

    // MARK: Active session protection

    @Test("evictIfNeeded does not evict models with an active session")
    func evictIfNeededSkipsActiveSessionModels() async throws {
        let threeGB = 3_221_225_472
        let fourGB = 4_294_967_296
        let lruDescriptor = makeTTSDescriptor(id: "org/model-active", minimumMemoryBytes: threeGB)
        let mruDescriptor = makeTTSDescriptor(id: "org/model-idle", minimumMemoryBytes: threeGB)

        let backend = makeBackend(evictionPolicy: EvictionPolicy(maxResidentBytes: fourGB))
        _ = try await backend.loadModel(descriptor: lruDescriptor, configuration: config)
        _ = try await backend.loadModel(descriptor: mruDescriptor, configuration: config)

        // Mark the LRU model as active — it must not be evicted.
        await backend.beginSession(for: lruDescriptor.id)

        let evicted = await backend.evictIfNeeded()
        // The active LRU model is protected; the idle MRU model is evicted instead.
        #expect(evicted == [mruDescriptor.id])
        #expect(await backend.loadedModelCount == 1)

        let snapshot = await backend.residencySnapshot()
        #expect(snapshot.first?.modelId == lruDescriptor.id)

        // Clean up the session so the backend isn't left in a dirty state.
        await backend.endSession(for: lruDescriptor.id)
    }

    @Test("evictIfNeeded evicts no one when all models have active sessions")
    func evictIfNeededDoesNothingWhenAllActive() async throws {
        let threeGB = 3_221_225_472
        let twoGB = 2_147_483_648
        let descriptor = makeTTSDescriptor(minimumMemoryBytes: threeGB)

        let backend = makeBackend(evictionPolicy: EvictionPolicy(maxResidentBytes: twoGB))
        _ = try await backend.loadModel(descriptor: descriptor, configuration: config)
        await backend.beginSession(for: descriptor.id)

        let evicted = await backend.evictIfNeeded()
        #expect(evicted.isEmpty)
        #expect(await backend.loadedModelCount == 1)

        await backend.endSession(for: descriptor.id)
    }

    @Test("endSession clamps counter at zero and does not crash")
    func endSessionClampedAtZero() async throws {
        let backend = makeBackend()
        let descriptor = makeTTSDescriptor()
        _ = try await backend.loadModel(descriptor: descriptor, configuration: config)

        // endSession on a model with no active session should be harmless.
        await backend.endSession(for: descriptor.id)
        let snapshot = await backend.residencySnapshot()
        #expect(!snapshot.isEmpty) // model is still resident
    }

    @Test("endSession applies eviction policy as soon as a model becomes idle")
    func endSessionTriggersPolicyEviction() async throws {
        let threeGB = 3_221_225_472
        let fourGB = 4_294_967_296
        let firstDescriptor = makeTTSDescriptor(id: "org/model-first", minimumMemoryBytes: threeGB)
        let secondDescriptor = makeTTSDescriptor(id: "org/model-second", minimumMemoryBytes: threeGB)

        let backend = makeBackend(evictionPolicy: EvictionPolicy(maxResidentBytes: fourGB))
        _ = try await backend.loadModel(descriptor: firstDescriptor, configuration: config)
        _ = try await backend.loadModel(descriptor: secondDescriptor, configuration: config)

        await backend.beginSession(for: firstDescriptor.id)
        await backend.beginSession(for: secondDescriptor.id)

        await backend.endSession(for: secondDescriptor.id)
        #expect(await backend.loadedModelCount == 1)

        let snapshot = await backend.residencySnapshot()
        #expect(snapshot.count == 1)
        #expect(snapshot.first?.modelId == firstDescriptor.id)

        await backend.endSession(for: firstDescriptor.id)
        #expect(await backend.loadedModelCount == 1)
    }

    // MARK: Idle-timeout eviction

    @Test("evictIfNeeded evicts models past idle timeout")
    func evictIfNeededEvictsIdleModels() async throws {
        // Use a negative timeout so any last-used timestamp is "past the cutoff".
        let backend = makeBackend(evictionPolicy: EvictionPolicy(idleTimeoutSeconds: -1))
        let descriptor = makeTTSDescriptor()
        _ = try await backend.loadModel(descriptor: descriptor, configuration: config)

        let evicted = await backend.evictIfNeeded()
        #expect(evicted == [descriptor.id])
        #expect(await backend.loadedModelCount == 0)
    }

    @Test("evictIfNeeded does not evict models within idle timeout window")
    func evictIfNeededDoesNotEvictFreshModels() async throws {
        // Timeout of 3600 seconds — freshly loaded model should not be evicted.
        let backend = makeBackend(evictionPolicy: EvictionPolicy(idleTimeoutSeconds: 3600))
        let descriptor = makeTTSDescriptor()
        _ = try await backend.loadModel(descriptor: descriptor, configuration: config)

        let evicted = await backend.evictIfNeeded()
        #expect(evicted.isEmpty)
        #expect(await backend.loadedModelCount == 1)
    }

    @Test("idle-timeout eviction skips models with active sessions")
    func idleEvictionSkipsActiveSession() async throws {
        let backend = makeBackend(evictionPolicy: EvictionPolicy(idleTimeoutSeconds: -1))
        let descriptor = makeTTSDescriptor()
        _ = try await backend.loadModel(descriptor: descriptor, configuration: config)
        await backend.beginSession(for: descriptor.id)

        let evicted = await backend.evictIfNeeded()
        #expect(evicted.isEmpty)
        #expect(await backend.loadedModelCount == 1)

        await backend.endSession(for: descriptor.id)
    }

    // MARK: Combined budget + idle timeout

    @Test("both budget and idle-timeout policies are applied in one evictIfNeeded call")
    func bothPoliciesAppliedTogether() async throws {
        // One model is idle (should be caught by timeout), one is over-budget (should be
        // caught by budget-based eviction). Both go in a single evictIfNeeded() call.
        let twoGB = 2_147_483_648
        let idleDescriptor = makeTTSDescriptor(id: "org/idle-model", minimumMemoryBytes: 1)
        let overBudgetDescriptor = makeTTSDescriptor(id: "org/over-budget-model", minimumMemoryBytes: twoGB)

        let backend = makeBackend(evictionPolicy: EvictionPolicy(
            maxResidentBytes: 1,  // any resident model over 1 byte violates the budget
            idleTimeoutSeconds: -1
        ))
        _ = try await backend.loadModel(descriptor: idleDescriptor, configuration: config)
        _ = try await backend.loadModel(descriptor: overBudgetDescriptor, configuration: config)

        let evicted = await backend.evictIfNeeded()
        #expect(evicted.count == 2)
        #expect(await backend.loadedModelCount == 0)
    }
}
