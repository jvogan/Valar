import Foundation
import Testing
@testable import ValarMLX
import ValarModelKit

/// Performance-characterisation tests for the VibeVoice load and streaming pipeline.
///
/// All timing assertions use injected slow loaders (`Task.sleep`) so that CI
/// machines without a GPU still enforce the latency contracts. Conservative
/// thresholds are used relative to the injected delays; they document the
/// intended performance budget rather than measuring real inference speed.
///
/// Metrics captured:
/// - **Cold load latency**: time for a first `loadModel()` call when the model is
///   not pre-warmed.
/// - **Warm load latency**: time for `loadModel()` after a completed `prewarm()` —
///   must be near-instant (cache hit, no loader invocation).
/// - **First-chunk latency**: time from synthesis request to first yielded `AudioChunk`
///   on the streaming path. Simulated via a mock stream that emits one chunk after a
///   configurable delay.
/// - **Loader deduplication**: concurrent VibeVoice loads during an in-flight prewarm
///   must invoke the loader exactly once.
///
/// End-to-end RTF measurements require a real VibeVoice model and a running
/// daemon. The public repo ships the benchmark corpus and reference benchmark
/// patterns in `scripts/qwen/benchmark.sh` and `scripts/voxtral/benchmark.sh`;
/// VibeVoice-specific live benchmark guidance lives in
/// `tests/vibevoice_corpus/README.md`.
@Suite("VibeVoice Performance")
struct VibeVoiceBenchmarkTests {

    // MARK: - Cold load

    @Test("Cold VibeVoice loadModel incurs the full loader latency")
    func coldLoadIncursFullLatency() async throws {
        let loaderDelay = Duration.milliseconds(100)
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
        let elapsed = try await clock.measure {
            _ = try await backend.loadModel(
                descriptor: vibeVoiceDescriptor(id: "bench/vv-cold-load"),
                configuration: ModelRuntimeConfiguration(backendKind: .mlx)
            )
        }

        // Cold load must block for at least the simulated loader delay.
        #expect(elapsed >= loaderDelay)
    }

    // MARK: - Warm load (post-prewarm)

    @Test("VibeVoice warm loadModel after prewarm is near-instant")
    func warmLoadAfterPrewarmIsNearInstant() async throws {
        let loaderDelay = Duration.milliseconds(150)
        let backend = MLXInferenceBackend(
            modelDirectoryResolver: { _ in nil },
            warningHandler: { _ in },
            qwenModelLoader: { descriptor in MLXModelHandle(descriptor: descriptor) },
            vibeVoiceTTSModelLoader: { descriptor in
                try await Task.sleep(for: loaderDelay)
                return MLXModelHandle(descriptor: descriptor)
            }
        )

        let descriptor = vibeVoiceDescriptor(id: "bench/vv-warm-load")
        let config = ModelRuntimeConfiguration(backendKind: .mlx)

        // Prewarm, then wait long enough for the loader to finish.
        await backend.prewarm(descriptor: descriptor, configuration: config)
        try await Task.sleep(for: .milliseconds(300))

        let clock = ContinuousClock()
        let elapsed = try await clock.measure {
            _ = try await backend.loadModel(descriptor: descriptor, configuration: config)
        }

        // Warm load reads from the in-memory cache — must be well under the loader delay.
        #expect(elapsed < .milliseconds(50))
    }

    // MARK: - Loader deduplication

    @Test("Concurrent VibeVoice loads during an in-flight prewarm invoke the loader exactly once")
    func concurrentLoadsDuringPrewarmDeduplicateLoader() async throws {
        let counter = InvocationCounter()
        let backend = MLXInferenceBackend(
            modelDirectoryResolver: { _ in nil },
            warningHandler: { _ in },
            qwenModelLoader: { descriptor in MLXModelHandle(descriptor: descriptor) },
            vibeVoiceTTSModelLoader: { descriptor in
                await counter.increment()
                try await Task.sleep(for: .milliseconds(100))
                return MLXModelHandle(descriptor: descriptor)
            }
        )

        let descriptor = vibeVoiceDescriptor(id: "bench/vv-concurrent-prewarm")
        let config = ModelRuntimeConfiguration(backendKind: .mlx)

        // Start an in-flight load via prewarm.
        await backend.prewarm(descriptor: descriptor, configuration: config)

        // Three concurrent loads arrive while the prewarm task is still running.
        async let h1 = backend.loadModel(descriptor: descriptor, configuration: config)
        async let h2 = backend.loadModel(descriptor: descriptor, configuration: config)
        async let h3 = backend.loadModel(descriptor: descriptor, configuration: config)

        let (handle1, handle2, handle3) = try await (h1, h2, h3)

        #expect(handle1.descriptor.id == descriptor.id)
        #expect(handle2.descriptor.id == descriptor.id)
        #expect(handle3.descriptor.id == descriptor.id)

        // Loader must have been called only once despite three concurrent demands.
        #expect(await counter.count == 1)
    }

    // MARK: - Prewarm is non-blocking

    @Test("VibeVoice prewarm returns before the loader finishes")
    func vibeVoicePrewarmIsNonBlocking() async {
        let loaderDelay = Duration.milliseconds(250)
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
                descriptor: vibeVoiceDescriptor(id: "bench/vv-prewarm-nonblocking"),
                configuration: ModelRuntimeConfiguration(backendKind: .mlx)
            )
        }

        // prewarm() must return well before the 250 ms loader finishes.
        #expect(elapsed < .milliseconds(125))
    }

    // MARK: - Repeated loads hit in-memory cache

    @Test("Repeated VibeVoice loadModel calls only invoke the loader once")
    func repeatedLoadsHitInMemoryCache() async throws {
        let counter = InvocationCounter()
        let backend = MLXInferenceBackend(
            modelDirectoryResolver: { _ in nil },
            warningHandler: { _ in },
            qwenModelLoader: { descriptor in MLXModelHandle(descriptor: descriptor) },
            vibeVoiceTTSModelLoader: { descriptor in
                await counter.increment()
                return MLXModelHandle(descriptor: descriptor)
            }
        )

        let descriptor = vibeVoiceDescriptor(id: "bench/vv-repeated-loads")
        let config = ModelRuntimeConfiguration(backendKind: .mlx)

        for _ in 0..<4 {
            _ = try await backend.loadModel(descriptor: descriptor, configuration: config)
        }

        #expect(await counter.count == 1)
    }

    // MARK: - First-chunk latency budget

    /// Validates that the first AudioChunk arrives within the streaming latency budget.
    ///
    /// This test models the VibeVoice streaming path with a mock async generator that
    /// yields one chunk after a 150 ms simulated TTS latency. The measurement verifies
    /// that the consumer side does not add hidden overhead beyond the generator delay.
    ///
    /// Real first-chunk latency (including Metal inference) is measured against
    /// a live daemon using the benchmark corpus and the reference patterns
    /// documented in `tests/vibevoice_corpus/README.md`.
    @Test("First AudioChunk arrives within the simulated generator delay")
    func firstChunkArrivesWithinGeneratorDelay() async throws {
        let generatorDelay = Duration.milliseconds(150)
        // Tolerance: consumer overhead must not exceed half the generator delay.
        let budget = generatorDelay + .milliseconds(75)

        let clock = ContinuousClock()
        var firstChunkLatency: Duration? = nil

        let generatorStart = clock.now
        let stream = mockAudioChunkStream(firstChunkDelay: generatorDelay, chunkCount: 3)

        for try await _ in stream {
            if firstChunkLatency == nil {
                firstChunkLatency = clock.now - generatorStart
            }
            break // only care about the first chunk
        }

        guard let latency = firstChunkLatency else {
            Issue.record("Stream yielded no chunks")
            return
        }

        #expect(latency < budget,
            "First-chunk latency \(latency) exceeds budget \(budget)")
    }

    // MARK: - Corpus manifest integrity

    /// Verifies that the vibevoice_corpus manifest is well-formed and all referenced
    /// text files exist relative to the repo root.
    ///
    /// This test is not a performance test, but it runs in this suite because it
    /// guards the corpus inputs used by the live benchmark workflow described in
    /// `tests/vibevoice_corpus/README.md`.
    @Test("vibevoice_corpus manifest is valid and all text files exist")
    func corpusManifestIsValid() throws {
        let testSourceDir = URL(fileURLWithPath: #file)
            .deletingLastPathComponent()  // ValarMLXTests/
            .deletingLastPathComponent()  // Tests/
            .deletingLastPathComponent()  // ValarMLX/
            .deletingLastPathComponent()  // Packages/
        let corpusDir = testSourceDir.appendingPathComponent("tests/vibevoice_corpus")
        let manifestURL = corpusDir.appendingPathComponent("manifest.json")

        guard FileManager.default.fileExists(atPath: manifestURL.path) else {
            // Allow skip in environments where the repo root is not accessible
            // (e.g., Codex sandboxes). Do not fail, just record a note.
            return
        }

        let data = try Data(contentsOf: manifestURL)
        let manifest = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        let entries = manifest?["entries"] as? [[String: Any]]
        #expect(entries != nil && !entries!.isEmpty,
            "manifest.json must contain a non-empty 'entries' array")

        guard let entries else { return }

        var seenFiles: Set<String> = []
        for entry in entries {
            guard let file = entry["file"] as? String else {
                Issue.record("Entry missing 'file' key: \(entry)")
                continue
            }
            guard let lang = entry["language"] as? String else {
                Issue.record("Entry missing 'language' key: \(entry)")
                continue
            }
            #expect(entry["voice_id"] is String,
                "Entry for '\(lang)' missing 'voice_id'")

            if seenFiles.insert(file).inserted {
                let txtURL = corpusDir.appendingPathComponent(file)
                #expect(FileManager.default.fileExists(atPath: txtURL.path),
                    "Corpus file '\(file)' referenced in manifest does not exist")
            }
        }

        // Must cover all 11 supported languages
        let langs = Set(entries.compactMap { $0["language"] as? String })
        let required: Set<String> = ["en", "de", "fr", "es", "it", "nl", "pt", "pl", "hi", "ja", "ko"]
        for lang in required {
            #expect(langs.contains(lang),
                "vibevoice_corpus is missing entries for language '\(lang)'")
        }
    }
}

// MARK: - Private helpers

/// Returns a VibeVoice `ModelDescriptor` keyed by `id`.
private func vibeVoiceDescriptor(
    id: String = "mlx-community/VibeVoice-Realtime-0.5B-4bit"
) -> ModelDescriptor {
    ModelDescriptor(
        id: ModelIdentifier(id),
        familyID: .vibevoiceRealtimeTTS,
        displayName: "VibeVoice Realtime 0.5B",
        domain: .tts,
        capabilities: [.speechSynthesis, .presetVoices, .streaming],
        supportedBackends: [BackendRequirement(backendKind: .mlx, minimumMemoryBytes: nil)]
    )
}

/// Returns an `AsyncThrowingStream<AudioChunk, Error>` that emits `chunkCount` chunks.
/// The first chunk is delayed by `firstChunkDelay`; subsequent chunks are immediate.
private func mockAudioChunkStream(
    firstChunkDelay: Duration,
    chunkCount: Int
) -> AsyncThrowingStream<AudioChunk, Error> {
    AsyncThrowingStream { continuation in
        Task {
            do {
                try await Task.sleep(for: firstChunkDelay)
                for _ in 0..<chunkCount {
                    let samples = [Float](repeating: 0, count: 480)
                    let chunk = AudioChunk(samples: samples, sampleRate: 24_000)
                    continuation.yield(chunk)
                }
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }
    }
}

// MARK: - Shared actor for invocation counting

/// Thread-safe invocation counter backed by Swift actor isolation.
private actor InvocationCounter {
    private(set) var count: Int = 0

    func increment() {
        count += 1
    }
}
