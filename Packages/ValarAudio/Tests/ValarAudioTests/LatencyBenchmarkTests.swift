import Foundation
import XCTest
@testable import ValarAudio

/// Latency-focused benchmarks for the streaming audio pipeline.
///
/// Throughput benchmarks in `SPSCFloatRingBufferTests` confirm the pipeline can
/// move samples quickly in bulk.  These tests ask a different question:
/// *how long does a single chunk spend in each pipeline stage?*
///
/// The canonical streaming TTS chunk size used throughout is 480 samples at 24 kHz,
/// which equals 20 ms of audio — one buffer period at the default output rate.  Every
/// latency threshold is set relative to that 20 ms budget so failures clearly
/// communicate how far a component is from real-time requirements.
final class LatencyBenchmarkTests: XCTestCase {

    // 20 ms at 24 kHz — the canonical TTS streaming chunk size.
    private let chunkFrames = 480
    private let sampleRate = 24_000.0

    // MARK: - SPSC ring buffer latency

    /// Write → read round-trip latency for a single 20 ms chunk.
    ///
    /// This is the minimum latency floor of the streaming pipeline.  Even on a
    /// busy CI machine the lock-free SPSC implementation completes each round-trip
    /// in the sub-microsecond range.  The threshold of 1 ms (1 000 µs) is set
    /// far above what is achievable so the test gates correctness rather than
    /// micro-optimisation.
    func testSPSCRingBufferChunkRoundTripLatency() {
        let ring = SPSCFloatRingBuffer(minimumCapacity: 1024)
        let chunk = [Float](repeating: 0.5, count: chunkFrames)
        var readBack = [Float](repeating: 0, count: chunkFrames)
        let iterations = 10_000

        // Warm up — discard first measurement.
        _ = ring.write(chunk)
        _ = readBack.withUnsafeMutableBufferPointer { ring.read(into: $0) }

        let elapsed = measureWallClock {
            for _ in 0..<iterations {
                _ = ring.write(chunk)
                _ = readBack.withUnsafeMutableBufferPointer { ring.read(into: $0) }
            }
        }

        let latencyMicros = elapsed / Double(iterations) * 1_000_000
        print(String(
            format: "SPSC round-trip latency: %.2f µs/chunk (%d iterations, %.4f s total)",
            latencyMicros, iterations, elapsed
        ))

        // Lock-free 480-sample write → read must complete in < 1 ms (1 000 µs).
        // In practice Apple Silicon delivers sub-microsecond throughput.
        XCTAssertLessThan(
            latencyMicros,
            1_000,
            String(format: "SPSC latency %.2f µs/chunk exceeds 1 ms threshold", latencyMicros)
        )
    }

    /// First-sample availability latency: time from producer write to the
    /// moment `availableToRead` reflects the new samples.
    ///
    /// In a streaming TTS pipeline this determines how quickly the audio engine
    /// can begin draining after the model emits its first chunk.  The threshold
    /// of 100 µs per write is generous — actual values are single-digit microseconds.
    func testSPSCFirstSampleAvailabilityLatency() {
        let chunk = [Float](repeating: 0.5, count: chunkFrames)
        let iterations = 10_000
        var totalElapsed: TimeInterval = 0

        for _ in 0..<iterations {
            // Fresh ring buffer each iteration to isolate the cold-write cost.
            let ring = SPSCFloatRingBuffer(minimumCapacity: 1024)

            let start = CFAbsoluteTimeGetCurrent()
            let written = ring.write(chunk)
            totalElapsed += CFAbsoluteTimeGetCurrent() - start

            XCTAssertEqual(written, chunkFrames)
        }

        let avgMicros = totalElapsed / Double(iterations) * 1_000_000
        print(String(
            format: "SPSC first-sample availability: %.2f µs/write (%d iterations, %.4f s total)",
            avgMicros, iterations, totalElapsed
        ))

        // Writing 480 samples must make them visible to a reader in < 100 µs.
        XCTAssertLessThan(
            avgMicros,
            100,
            String(format: "SPSC write latency %.2f µs exceeds 100 µs threshold", avgMicros)
        )
    }

    // MARK: - StreamingWAVWriter latency

    /// Per-chunk append latency and real-time headroom for `StreamingWAVWriter`.
    ///
    /// The writer is on the critical path between TTS model output and the WAV file
    /// that the audio engine reads.  It must drain each 20 ms chunk faster than the
    /// model generates the next one.  Two assertions are made:
    ///
    /// - **Latency**: each `append` must complete in < 5 ms so a single slow write
    ///   never causes an audible glitch (the 20 ms chunk budget provides 4× headroom).
    /// - **Throughput**: the sustained rate must be ≥ 5× real-time (≥ 250 chunks/s
    ///   at 24 kHz / 480 samples) so burst generation can be committed to disk
    ///   without blocking the inference loop.
    func testStreamingWAVWriterChunkLatencyAndThroughput() async throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("wav")
        defer { try? FileManager.default.removeItem(at: url) }

        let chunk = [Float](repeating: 0.5, count: chunkFrames)
        let iterations = 500

        let writer = try StreamingWAVWriter(url: url, sampleRate: sampleRate, channelCount: 1)

        // Warm up — excluded from measurement.
        try await writer.append(chunk)

        let elapsed = try await measureWallClockAsync {
            for _ in 0..<iterations {
                try await writer.append(chunk)
            }
        }

        await writer.finalize()

        let latencyMs = elapsed / Double(iterations) * 1_000
        let chunksPerSecond = Double(iterations) / elapsed
        let realTimeMultiple = chunksPerSecond / (sampleRate / Double(chunkFrames))

        print(String(
            format: "StreamingWAVWriter: %.3f ms/chunk | %.1f chunks/s | %.1f× real-time (%d iterations, %.4f s)",
            latencyMs, chunksPerSecond, realTimeMultiple, iterations, elapsed
        ))

        // Per-chunk append must be < 5 ms to avoid stalling streaming generation.
        XCTAssertLessThan(
            latencyMs,
            5.0,
            String(format: "WAV writer latency %.3f ms/chunk exceeds 5 ms budget", latencyMs)
        )

        // Sustained throughput must be at least 5× real-time.
        XCTAssertGreaterThan(
            realTimeMultiple,
            5.0,
            String(format: "WAV writer throughput %.1f× real-time is below 5× threshold", realTimeMultiple)
        )
    }

    // MARK: - Helpers

    private func measureWallClock(_ block: () -> Void) -> TimeInterval {
        let start = CFAbsoluteTimeGetCurrent()
        block()
        return CFAbsoluteTimeGetCurrent() - start
    }

    private func measureWallClockAsync(_ block: () async throws -> Void) async rethrows -> TimeInterval {
        let start = CFAbsoluteTimeGetCurrent()
        try await block()
        return CFAbsoluteTimeGetCurrent() - start
    }
}
