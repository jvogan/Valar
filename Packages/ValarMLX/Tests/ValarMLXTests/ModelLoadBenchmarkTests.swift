import Foundation
import Testing
@testable import ValarMLX
import ValarModelKit

/// Tests that characterise model load latency and deduplication behaviour.
///
/// All timing assertions use injected slow loaders (`Task.sleep`) to simulate the
/// wall-clock cost of a real model load. Conservative thresholds give CI headroom
/// while still documenting the intended performance contract.
///
/// Coverage:
/// - `prewarm()` must return without blocking the caller
/// - Cold `loadModel()` incurs the full loader latency
/// - `loadModel()` after a completed prewarm does not re-invoke the loader
/// - Concurrent `loadModel()` calls sharing a prewarm task invoke the loader exactly once
/// - Double-prewarm and post-load-prewarm are no-ops
/// - Repeated `loadModel()` calls for the same model hit the in-memory cache
/// - Unload then reload invokes the loader a second time
/// - Weight cache hit skips the directory resolver; cache miss calls it once
@Suite("Model Load Benchmarks")
struct ModelLoadBenchmarkTests {

    // MARK: - Non-blocking prewarm

    @Test("prewarm returns before the loader finishes")
    func prewarmIsNonBlocking() async {
        let loaderDelay = Duration.milliseconds(300)
        let backend = MLXInferenceBackend(
            modelDirectoryResolver: { _ in nil },
            warningHandler: { _ in },
            qwenModelLoader: { descriptor in
                try await Task.sleep(for: loaderDelay)
                return MLXModelHandle(descriptor: descriptor)
            }
        )

        let clock = ContinuousClock()
        let elapsed = await clock.measure {
            await backend.prewarm(
                descriptor: benchmarkDescriptor(id: "bench/prewarm-nonblocking"),
                configuration: ModelRuntimeConfiguration(backendKind: .mlx)
            )
        }

        // prewarm() must return well before the 300 ms loader finishes.
        // Threshold is half the loader delay to give CI plenty of headroom.
        #expect(elapsed < .milliseconds(150))
    }

    // MARK: - Cold vs warm loadModel latency

    @Test("Cold loadModel incurs the full loader latency")
    func coldLoadIncursFullLoaderLatency() async throws {
        let loaderDelay = Duration.milliseconds(100)
        let backend = MLXInferenceBackend(
            modelDirectoryResolver: { _ in nil },
            warningHandler: { _ in },
            qwenModelLoader: { descriptor in
                try await Task.sleep(for: loaderDelay)
                return MLXModelHandle(descriptor: descriptor)
            }
        )

        let clock = ContinuousClock()
        let elapsed = try await clock.measure {
            _ = try await backend.loadModel(
                descriptor: benchmarkDescriptor(id: "bench/cold-load"),
                configuration: ModelRuntimeConfiguration(backendKind: .mlx)
            )
        }

        // Cold load must block for at least the simulated loader delay.
        #expect(elapsed >= loaderDelay)
    }

    @Test("loadModel after a completed prewarm does not incur loader latency")
    func warmLoadIsFast() async throws {
        let loaderDelay = Duration.milliseconds(150)
        let backend = MLXInferenceBackend(
            modelDirectoryResolver: { _ in nil },
            warningHandler: { _ in },
            qwenModelLoader: { descriptor in
                try await Task.sleep(for: loaderDelay)
                return MLXModelHandle(descriptor: descriptor)
            }
        )

        let descriptor = benchmarkDescriptor(id: "bench/warm-load")
        let config = ModelRuntimeConfiguration(backendKind: .mlx)

        // Start prewarm and give it 2× the loader delay to complete.
        await backend.prewarm(descriptor: descriptor, configuration: config)
        try await Task.sleep(for: .milliseconds(300))

        let clock = ContinuousClock()
        let elapsed = try await clock.measure {
            _ = try await backend.loadModel(descriptor: descriptor, configuration: config)
        }

        // Warm loadModel reads from the in-memory cache — must be near-instant.
        #expect(elapsed < .milliseconds(50))
    }

    // MARK: - Loader deduplication

    @Test("Concurrent loadModel calls during an in-flight prewarm invoke the loader exactly once")
    func concurrentLoadsDuringPrewarmDeduplicateLoader() async throws {
        let counter = InvocationCounter()
        let backend = MLXInferenceBackend(
            modelDirectoryResolver: { _ in nil },
            warningHandler: { _ in },
            qwenModelLoader: { descriptor in
                await counter.increment()
                try await Task.sleep(for: .milliseconds(100))
                return MLXModelHandle(descriptor: descriptor)
            }
        )

        let descriptor = benchmarkDescriptor(id: "bench/concurrent-prewarm")
        let config = ModelRuntimeConfiguration(backendKind: .mlx)

        // Prewarm starts the in-flight load task.
        await backend.prewarm(descriptor: descriptor, configuration: config)

        // Three concurrent loads arrive while the prewarm task is still running.
        // Each will find the pending task in pendingLoadTasks and await it,
        // rather than starting independent load operations.
        async let m1 = backend.loadModel(descriptor: descriptor, configuration: config)
        async let m2 = backend.loadModel(descriptor: descriptor, configuration: config)
        async let m3 = backend.loadModel(descriptor: descriptor, configuration: config)

        let (handle1, handle2, handle3) = try await (m1, m2, m3)

        #expect(handle1.descriptor.id == descriptor.id)
        #expect(handle2.descriptor.id == descriptor.id)
        #expect(handle3.descriptor.id == descriptor.id)

        // Loader must have been called only once despite three concurrent demands.
        #expect(await counter.count == 1)
    }

    @Test("Double prewarm for the same model is a no-op")
    func doublePrewarmIsNoop() async throws {
        let counter = InvocationCounter()
        let backend = MLXInferenceBackend(
            modelDirectoryResolver: { _ in nil },
            warningHandler: { _ in },
            qwenModelLoader: { descriptor in
                await counter.increment()
                return MLXModelHandle(descriptor: descriptor)
            }
        )

        let descriptor = benchmarkDescriptor(id: "bench/double-prewarm")
        let config = ModelRuntimeConfiguration(backendKind: .mlx)

        await backend.prewarm(descriptor: descriptor, configuration: config)
        await backend.prewarm(descriptor: descriptor, configuration: config)

        // Allow the single prewarm task to finish.
        try await Task.sleep(for: .milliseconds(50))

        #expect(await counter.count == 1)
    }

    @Test("prewarm after model is already loaded is a no-op")
    func prewarmAfterLoadIsNoop() async throws {
        let counter = InvocationCounter()
        let backend = MLXInferenceBackend(
            modelDirectoryResolver: { _ in nil },
            warningHandler: { _ in },
            qwenModelLoader: { descriptor in
                await counter.increment()
                return MLXModelHandle(descriptor: descriptor)
            }
        )

        let descriptor = benchmarkDescriptor(id: "bench/prewarm-after-load")
        let config = ModelRuntimeConfiguration(backendKind: .mlx)

        _ = try await backend.loadModel(descriptor: descriptor, configuration: config)
        await backend.prewarm(descriptor: descriptor, configuration: config)

        #expect(await counter.count == 1)
    }

    // MARK: - In-memory model cache

    @Test("Repeated loadModel calls for the same model only invoke the loader once")
    func repeatedLoadsHitInMemoryCache() async throws {
        let counter = InvocationCounter()
        let backend = MLXInferenceBackend(
            modelDirectoryResolver: { _ in nil },
            warningHandler: { _ in },
            qwenModelLoader: { descriptor in
                await counter.increment()
                return MLXModelHandle(descriptor: descriptor)
            }
        )

        let descriptor = benchmarkDescriptor(id: "bench/in-memory-cache")
        let config = ModelRuntimeConfiguration(backendKind: .mlx)

        for _ in 0..<3 {
            _ = try await backend.loadModel(descriptor: descriptor, configuration: config)
        }

        #expect(await counter.count == 1)
    }

    @Test("Unload then reload invokes the loader a second time")
    func unloadThenReloadCallsLoaderAgain() async throws {
        let counter = InvocationCounter()
        let backend = MLXInferenceBackend(
            modelDirectoryResolver: { _ in nil },
            warningHandler: { _ in },
            qwenModelLoader: { descriptor in
                await counter.increment()
                return MLXModelHandle(descriptor: descriptor)
            }
        )

        let descriptor = benchmarkDescriptor(id: "bench/unload-reload")
        let config = ModelRuntimeConfiguration(backendKind: .mlx)

        let model = try await backend.loadModel(descriptor: descriptor, configuration: config)
        try await backend.unloadModel(model)
        _ = try await backend.loadModel(descriptor: descriptor, configuration: config)

        #expect(await counter.count == 2)
    }

    // MARK: - Weight cache fast path

    @Test("Weight cache hit skips the directory resolver")
    func weightCacheHitSkipsResolver() async throws {
        let resolverCounter = InvocationCounter()

        let directory = try makeValidSafeTensorsDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        // Pre-populate a weight cache with the directory's safetensors mtime.
        let modelID = "bench/weight-cache-hit"
        var cache = MLXModelWeightCache()
        try cache.store(modelID: modelID, directory: directory)

        let cacheFile = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString + "-cache.json")
        defer { try? FileManager.default.removeItem(at: cacheFile) }
        try cache.save(to: cacheFile)

        let backend = MLXInferenceBackend(
            modelDirectoryResolver: { _ in
                await resolverCounter.increment()
                return directory
            },
            warningHandler: { _ in },
            qwenModelLoader: { descriptor in MLXModelHandle(descriptor: descriptor) },
            weightCacheFileURL: cacheFile
        )

        let descriptor = ModelDescriptor(
            id: ModelIdentifier(modelID),
            familyID: .qwen3TTS,
            displayName: "Cache Hit Model",
            domain: .tts,
            capabilities: [.speechSynthesis]
        )

        _ = try await backend.loadModel(
            descriptor: descriptor,
            configuration: ModelRuntimeConfiguration(backendKind: .mlx)
        )

        // Cache hit — resolver must not have been called.
        #expect(await resolverCounter.count == 0)
    }

    @Test("Weight cache miss calls the directory resolver once")
    func weightCacheMissCallsResolver() async throws {
        let resolverCounter = InvocationCounter()

        let directory = try makeValidSafeTensorsDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        // Start with an empty cache so every model is a miss.
        let cacheFile = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString + "-cache.json")
        defer { try? FileManager.default.removeItem(at: cacheFile) }
        try MLXModelWeightCache().save(to: cacheFile)

        let backend = MLXInferenceBackend(
            modelDirectoryResolver: { _ in
                await resolverCounter.increment()
                return directory
            },
            warningHandler: { _ in },
            qwenModelLoader: { descriptor in MLXModelHandle(descriptor: descriptor) },
            weightCacheFileURL: cacheFile
        )

        _ = try await backend.loadModel(
            descriptor: benchmarkDescriptor(id: "bench/cache-miss"),
            configuration: ModelRuntimeConfiguration(backendKind: .mlx)
        )

        // Cache miss — resolver must have been called exactly once.
        #expect(await resolverCounter.count == 1)
    }
}

// MARK: - Private helpers

/// Returns a Qwen3TTS `ModelDescriptor` using `id` as the model identifier.
///
/// Qwen3TTS descriptors route through the injectable `qwenModelLoader` in
/// `MLXInferenceBackend`, which is required for any test that measures or
/// counts loader invocations.
private func benchmarkDescriptor(id: String) -> ModelDescriptor {
    ModelDescriptor(
        id: ModelIdentifier(id),
        familyID: .qwen3TTS,
        displayName: id,
        domain: .tts,
        capabilities: [.speechSynthesis]
    )
}

/// Creates a temporary directory containing one valid `.safetensors` file.
///
/// The file has a minimal but structurally correct safetensors header
/// (`{"}`  — 8-byte little-endian length prefix followed by an empty JSON object)
/// so that full header validation in `MLXInferenceBackend` passes.
private func makeValidSafeTensorsDirectory() throws -> URL {
    let dir = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

    var data = Data()
    var headerLength = UInt64(2).littleEndian
    withUnsafeBytes(of: &headerLength) { data.append(contentsOf: $0) }
    data.append(contentsOf: [0x7B, 0x7D]) // `{}`
    try data.write(to: dir.appendingPathComponent("model.safetensors"))

    return dir
}

// MARK: - Shared test infrastructure

/// Thread-safe invocation counter backed by Swift actor isolation.
private actor InvocationCounter {
    private(set) var count: Int = 0

    func increment() {
        count += 1
    }
}
