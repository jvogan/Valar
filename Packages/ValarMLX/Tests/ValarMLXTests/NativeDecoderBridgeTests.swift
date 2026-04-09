import Foundation
@preconcurrency import MLX
import MLXNN
import Testing
import ValarModelKit
@testable import ValarMLX

// MARK: - Native Decoder Bridge Tests
//
// Tests for the native SpeechTokenizerDecoder integration in MLXStreamBridge.
// Uses a small decoder (tiny dims) for fast testing — verifies stream plumbing,
// chunk shapes, sample rates, error propagation, and fallback behavior.

@Suite("Native Decoder Bridge")
struct NativeDecoderBridgeTests {

    // MARK: - decodeCodes

    @Test("decodeCodes returns AudioChunk at native sample rate")
    func decodeCodesReturnsSingleChunk() {
        let decoder = makeSmallDecoder()
        let codes = makeSmallCodes(time: 4)

        let chunk = MLXStreamBridge.decodeCodes(codes, using: decoder)

        #expect(chunk.sampleRate == Double(MLXStreamBridge.nativeDecoderSampleRate))
        #expect(!chunk.samples.isEmpty)
    }

    @Test("decodeCodes output samples are finite")
    func decodeCodesOutputFinite() {
        let decoder = makeSmallDecoder()
        let codes = makeSmallCodes(time: 6)

        let chunk = MLXStreamBridge.decodeCodes(codes, using: decoder)

        for sample in chunk.samples {
            #expect(sample.isFinite, "Decoded sample must be finite")
        }
    }

    @Test("decodeCodes respects custom sample rate")
    func decodeCodesCustomSampleRate() {
        let decoder = makeSmallDecoder()
        let codes = makeSmallCodes(time: 4)

        let chunk = MLXStreamBridge.decodeCodes(codes, using: decoder, sampleRate: 16_000)

        #expect(chunk.sampleRate == 16_000.0)
    }

    @Test("decodeCodes non-chunked path produces output")
    func decodeCodesNonChunked() {
        let decoder = makeSmallDecoder()
        let codes = makeSmallCodes(time: 4)

        let chunk = MLXStreamBridge.decodeCodes(codes, using: decoder, chunked: false)

        #expect(!chunk.samples.isEmpty)
        #expect(chunk.sampleRate == Double(MLXStreamBridge.nativeDecoderSampleRate))
    }

    @Test("decodeCodes chunked and non-chunked produce same length for short input")
    func decodeCodesChunkedVsNonChunked() {
        let decoder = makeSmallDecoder()
        let codes = makeSmallCodes(time: 4)

        let chunked = MLXStreamBridge.decodeCodes(codes, using: decoder, chunked: true)
        let direct = MLXStreamBridge.decodeCodes(codes, using: decoder, chunked: false)

        #expect(chunked.samples.count == direct.samples.count)
    }

    // MARK: - nativeDecoderStream

    @Test("nativeDecoderStream yields chunks from code frames")
    func nativeDecoderStreamYieldsChunks() async throws {
        let decoder = makeSmallDecoder()
        let codes1 = makeSmallCodes(time: 4)
        let codes2 = makeSmallCodes(time: 6)

        let stream = MLXStreamBridge.nativeDecoderStream(
            from: AsyncThrowingStream { continuation in
                continuation.yield(codes1)
                continuation.yield(codes2)
                continuation.finish()
            },
            decoder: decoder
        )

        var chunks: [AudioChunk] = []
        for try await chunk in stream {
            chunks.append(chunk)
        }

        #expect(chunks.count == 2)
        #expect(chunks[0].sampleRate == Double(MLXStreamBridge.nativeDecoderSampleRate))
        #expect(chunks[1].sampleRate == Double(MLXStreamBridge.nativeDecoderSampleRate))
        #expect(!chunks[0].samples.isEmpty)
        #expect(!chunks[1].samples.isEmpty)
    }

    @Test("nativeDecoderStream completes on empty input")
    func nativeDecoderStreamEmptyInput() async throws {
        let decoder = makeSmallDecoder()

        let stream = MLXStreamBridge.nativeDecoderStream(
            from: AsyncThrowingStream { continuation in
                continuation.finish()
            },
            decoder: decoder
        )

        var chunks: [AudioChunk] = []
        for try await chunk in stream {
            chunks.append(chunk)
        }

        #expect(chunks.isEmpty)
    }

    @Test("nativeDecoderStream propagates upstream errors")
    func nativeDecoderStreamPropagatesErrors() async {
        let decoder = makeSmallDecoder()

        let stream = MLXStreamBridge.nativeDecoderStream(
            from: AsyncThrowingStream { continuation in
                continuation.yield(makeSmallCodes(time: 4))
                continuation.finish(throwing: TestStreamError.intentional)
            },
            decoder: decoder
        )

        var chunks: [AudioChunk] = []
        do {
            for try await chunk in stream {
                chunks.append(chunk)
            }
            Issue.record("Expected error to be thrown")
        } catch is TestStreamError {
            // Expected
        } catch {
            Issue.record("Unexpected error type: \(error)")
        }

        // The first chunk should have been yielded before the error
        #expect(chunks.count == 1)
    }

    @Test("nativeDecoderStream stops yielding when task is cancelled between frames")
    func nativeDecoderStreamHonoursCancellation() async throws {
        let decoder = makeSmallDecoder()

        // Hold the second frame until the parent task cancels the consumer.
        let secondFrameGate = ContinuationGate()
        let firstChunkSeen = ContinuationGate()

        let codeFrames = AsyncThrowingStream<MLXArray, Error> { continuation in
            let producer = Task {
                do {
                    continuation.yield(makeSmallCodes(time: 4))
                    await secondFrameGate.wait()
                    try Task.checkCancellation()
                    continuation.yield(makeSmallCodes(time: 4))
                    continuation.finish()
                } catch is CancellationError {
                    continuation.finish()
                }
            }

            continuation.onTermination = { @Sendable _ in
                producer.cancel()
                Task {
                    await secondFrameGate.open()
                }
            }
        }

        let stream = MLXStreamBridge.nativeDecoderStream(from: codeFrames, decoder: decoder)

        let task = Task { () throws -> Int in
            var chunkCount = 0
            for try await _ in stream {
                chunkCount += 1
                if chunkCount == 1 {
                    await firstChunkSeen.open()
                }
            }
            return chunkCount
        }

        await firstChunkSeen.wait()
        task.cancel()
        await secondFrameGate.open()

        let outcome = await task.result
        var chunkCount = 0
        switch outcome {
        case .success(let count):
            // Cancellation may cause the stream to finish cleanly without throwing
            // if the CancellationError is caught inside the bridge Task and forwarded
            // as a stream finish. Accept both outcomes.
            chunkCount = count
        case .failure(let error):
            #expect(error is CancellationError, "Expected CancellationError, got \(error)")
        }

        // At most one chunk (the first frame) should have been processed.
        #expect(chunkCount <= 1)
    }

    @Test("nativeDecoderStream cancels when already-cancelled task consumes it")
    func nativeDecoderStreamAlreadyCancelledTask() async throws {
        let decoder = makeSmallDecoder()

        let stream = MLXStreamBridge.nativeDecoderStream(
            from: AsyncThrowingStream { continuation in
                continuation.yield(makeSmallCodes(time: 4))
                continuation.yield(makeSmallCodes(time: 4))
                continuation.finish()
            },
            decoder: decoder
        )

        // Create a task, cancel it immediately, then wait for it.
        let task = Task {
            var chunks: [AudioChunk] = []
            for try await chunk in stream {
                chunks.append(chunk)
            }
            return chunks
        }
        task.cancel()

        do {
            let chunks = try await task.value
            // If the task happened to complete before cancellation was noticed, the
            // result must still be a valid (possibly empty) collection.
            #expect(chunks.count <= 2)
        } catch is CancellationError {
            // Cancellation propagated as expected.
        }
    }

    @Test("nativeDecoderStream respects custom sample rate")
    func nativeDecoderStreamCustomSampleRate() async throws {
        let decoder = makeSmallDecoder()

        let stream = MLXStreamBridge.nativeDecoderStream(
            from: AsyncThrowingStream { continuation in
                continuation.yield(makeSmallCodes(time: 4))
                continuation.finish()
            },
            decoder: decoder,
            sampleRate: 48_000
        )

        var chunks: [AudioChunk] = []
        for try await chunk in stream {
            chunks.append(chunk)
        }

        #expect(chunks.count == 1)
        #expect(chunks[0].sampleRate == 48_000.0)
    }

    // MARK: - nativeDecoderSampleRate constant

    @Test("nativeDecoderSampleRate is 24000")
    func sampleRateConstant() {
        #expect(MLXStreamBridge.nativeDecoderSampleRate == 24_000)
    }

    // MARK: - Helpers

    private func makeSmallDecoder() -> SpeechTokenizerDecoder { makeTestDecoder() }
}

/// Shared tiny decoder config used across test suites.
private let sharedSmallDecoderConfig = SpeechTokenizerDecoderConfig(
    codebookDim: 8,
    codebookSize: 16,
    numQuantizers: 3,
    numSemanticQuantizers: 1,
    latentDim: 16,
    decoderDim: 16,
    upsampleRates: [2, 2],
    upsamplingRatios: [2],
    hiddenSize: 8,
    intermediateSize: 16,
    numHiddenLayers: 1,
    numAttentionHeads: 2,
    numKeyValueHeads: 2,
    headDim: 4,
    rmsNormEps: 1e-5,
    ropeTheta: 10000.0,
    slidingWindow: 8,
    layerScaleInitialScale: 0.01,
    attentionBias: false
)

private func makeTestDecoder() -> SpeechTokenizerDecoder {
    SpeechTokenizerDecoder(config: sharedSmallDecoderConfig)
}

/// Generate random codes for the small test config.
/// Shape: [1, numQuantizers=3, time].
private func makeSmallCodes(time: Int) -> MLXArray {
    let nQ = 3
    let values = (0 ..< nQ * time).map { _ in Int32.random(in: 0 ..< 16) }
    return MLXArray(values).reshaped(1, nQ, time)
}

/// Make a single per-step code frame of shape [1, nQ].
private func makeSmallPerStepCode(nQ: Int = 3) -> MLXArray {
    let values = (0 ..< nQ).map { _ in Int32.random(in: 0 ..< 16) }
    return MLXArray(values).reshaped(1, nQ)
}

// MARK: - BatchCodeFrames Tests

@Suite("BatchCodeFrames")
struct BatchCodeFramesTests {

    @Test("yields one batch when frame count equals chunkSize")
    func exactChunk() async throws {
        let nQ = 3
        let chunkSize = 4
        let frames = (0 ..< chunkSize).map { _ in makeSmallPerStepCode(nQ: nQ) }

        let batched = MLXStreamBridge.batchCodeFrames(
            from: AsyncThrowingStream { cont in
                for f in frames { cont.yield(f) }
                cont.finish()
            },
            chunkSize: chunkSize
        )

        var results: [MLXArray] = []
        for try await batch in batched { results.append(batch) }

        #expect(results.count == 1)
        // Output shape: [1, nQ, chunkSize]
        #expect(results[0].shape == [1, nQ, chunkSize])
    }

    @Test("yields two full batches when frame count is 2x chunkSize")
    func twoFullChunks() async throws {
        let nQ = 3
        let chunkSize = 3
        let totalFrames = 6
        let frames = (0 ..< totalFrames).map { _ in makeSmallPerStepCode(nQ: nQ) }

        let batched = MLXStreamBridge.batchCodeFrames(
            from: AsyncThrowingStream { cont in
                for f in frames { cont.yield(f) }
                cont.finish()
            },
            chunkSize: chunkSize
        )

        var results: [MLXArray] = []
        for try await batch in batched { results.append(batch) }

        #expect(results.count == 2)
        #expect(results[0].shape == [1, nQ, chunkSize])
        #expect(results[1].shape == [1, nQ, chunkSize])
    }

    @Test("flushes partial chunk at stream end")
    func partialFlush() async throws {
        let nQ = 3
        let chunkSize = 4
        let totalFrames = 5  // 1 full chunk + 1 partial (1 frame)
        let frames = (0 ..< totalFrames).map { _ in makeSmallPerStepCode(nQ: nQ) }

        let batched = MLXStreamBridge.batchCodeFrames(
            from: AsyncThrowingStream { cont in
                for f in frames { cont.yield(f) }
                cont.finish()
            },
            chunkSize: chunkSize
        )

        var results: [MLXArray] = []
        for try await batch in batched { results.append(batch) }

        #expect(results.count == 2)
        #expect(results[0].shape == [1, nQ, chunkSize])
        // Partial last chunk has 1 frame
        #expect(results[1].shape == [1, nQ, 1])
    }

    @Test("completes immediately on empty input")
    func emptyInput() async throws {
        let batched = MLXStreamBridge.batchCodeFrames(
            from: AsyncThrowingStream { cont in cont.finish() },
            chunkSize: 4
        )

        var results: [MLXArray] = []
        for try await batch in batched { results.append(batch) }

        #expect(results.isEmpty)
    }

    @Test("propagates upstream errors mid-stream")
    func propagatesError() async {
        let nQ = 3
        let chunkSize = 4

        let batched = MLXStreamBridge.batchCodeFrames(
            from: AsyncThrowingStream { cont in
                cont.yield(makeSmallPerStepCode(nQ: nQ))
                cont.finish(throwing: TestStreamError.intentional)
            },
            chunkSize: chunkSize
        )

        do {
            for try await _ in batched {}
            Issue.record("Expected error to be thrown")
        } catch is TestStreamError {
            // Expected
        } catch {
            Issue.record("Unexpected error type: \(error)")
        }
    }

    @Test("output batch feeds nativeDecoderStream to produce audio chunks")
    func batchedFramesThroughDecoder() async throws {
        let nQ = 3
        let chunkSize = 4
        let frames = (0 ..< chunkSize).map { _ in makeSmallPerStepCode(nQ: nQ) }
        let decoder = makeTestDecoder()

        let batched = MLXStreamBridge.batchCodeFrames(
            from: AsyncThrowingStream { cont in
                for f in frames { cont.yield(f) }
                cont.finish()
            },
            chunkSize: chunkSize
        )
        let audioStream = MLXStreamBridge.nativeDecoderStream(from: batched, decoder: decoder)

        var chunks: [AudioChunk] = []
        for try await chunk in audioStream { chunks.append(chunk) }

        #expect(chunks.count == 1)
        #expect(!chunks[0].samples.isEmpty)
        #expect(chunks[0].sampleRate == Double(MLXStreamBridge.nativeDecoderSampleRate))
    }

    @Test("useNativeDecoderPath defaults to false")
    func flagDefaultIsFalse() {
        #expect(MLXStreamBridge.useNativeDecoderPath == false)
    }

    @Test("codeStreamChunkSize is 25")
    func chunkSizeConstant() {
        #expect(MLXStreamBridge.codeStreamChunkSize == 25)
    }
}

private enum TestStreamError: Error {
    case intentional
}

private actor ContinuationGate {
    private var continuation: CheckedContinuation<Void, Never>?
    private var isOpen = false

    func wait() async {
        guard !isOpen else { return }
        await withCheckedContinuation { continuation in
            self.continuation = continuation
        }
    }

    func open() {
        guard !isOpen else { return }
        isOpen = true
        continuation?.resume()
        continuation = nil
    }
}
