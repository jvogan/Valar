import Dispatch
import XCTest
@testable import ValarAudio

final class SPSCFloatRingBufferTests: XCTestCase {

    // MARK: - Capacity

    func testCapacityRoundsUpToNextPowerOfTwo() {
        XCTAssertEqual(SPSCFloatRingBuffer(minimumCapacity: 1).ringCapacity, 2)
        XCTAssertEqual(SPSCFloatRingBuffer(minimumCapacity: 2).ringCapacity, 2)
        XCTAssertEqual(SPSCFloatRingBuffer(minimumCapacity: 3).ringCapacity, 4)
        XCTAssertEqual(SPSCFloatRingBuffer(minimumCapacity: 4).ringCapacity, 4)
        XCTAssertEqual(SPSCFloatRingBuffer(minimumCapacity: 5).ringCapacity, 8)
        XCTAssertEqual(SPSCFloatRingBuffer(minimumCapacity: 100).ringCapacity, 128)
    }

    func testNegativeOrZeroMinimumCapacityCoercesToTwo() {
        XCTAssertEqual(SPSCFloatRingBuffer(minimumCapacity: 0).ringCapacity, 2)
        XCTAssertEqual(SPSCFloatRingBuffer(minimumCapacity: -4).ringCapacity, 2)
    }

    // MARK: - Empty state

    func testNewBufferIsEmpty() {
        let ring = SPSCFloatRingBuffer(minimumCapacity: 8)
        XCTAssertTrue(ring.isEmpty)
        XCTAssertEqual(ring.availableToRead, 0)
        XCTAssertEqual(ring.availableToWrite, 8)
    }

    func testReadFromEmptyReturnsEmptyArray() {
        let ring = SPSCFloatRingBuffer(minimumCapacity: 4)
        let result = ring.read(count: 10)
        XCTAssertEqual(result, [])
    }

    func testReadZeroCountReturnsEmptyArray() {
        let ring = SPSCFloatRingBuffer(minimumCapacity: 4)
        ring.write([1, 2, 3])
        let result = ring.read(count: 0)
        XCTAssertEqual(result, [])
    }

    func testWriteEmptyArrayWritesZeroSamples() {
        let ring = SPSCFloatRingBuffer(minimumCapacity: 4)
        let written = ring.write([Float]())
        XCTAssertEqual(written, 0)
        XCTAssertTrue(ring.isEmpty)
    }

    // MARK: - Basic write / read

    func testWriteAndReadSingleSample() {
        let ring = SPSCFloatRingBuffer(minimumCapacity: 4)
        let written = ring.write([0.5])
        XCTAssertEqual(written, 1)
        XCTAssertEqual(ring.availableToRead, 1)
        XCTAssertEqual(ring.availableToWrite, 3)

        let result = ring.read(count: 1)
        XCTAssertEqual(result, [0.5])
        XCTAssertTrue(ring.isEmpty)
    }

    func testWriteAndReadAllSamples() {
        let ring = SPSCFloatRingBuffer(minimumCapacity: 4)
        let samples: [Float] = [1, 2, 3, 4]
        let written = ring.write(samples)
        XCTAssertEqual(written, 4)

        let result = ring.read(count: 4)
        XCTAssertEqual(result, samples)
        XCTAssertTrue(ring.isEmpty)
    }

    func testReadFewerThanAvailableReturnsOnlyRequestedCount() {
        let ring = SPSCFloatRingBuffer(minimumCapacity: 8)
        ring.write([1, 2, 3, 4, 5])

        let partial = ring.read(count: 3)
        XCTAssertEqual(partial, [1, 2, 3])
        XCTAssertEqual(ring.availableToRead, 2)
    }

    func testReadMoreThanAvailableReturnsOnlyWhatIsPresent() {
        let ring = SPSCFloatRingBuffer(minimumCapacity: 8)
        ring.write([1.0, 2.0])

        let result = ring.read(count: 100)
        XCTAssertEqual(result, [1.0, 2.0])
    }

    // MARK: - Full buffer

    func testWriteToFullBufferReturnsZero() {
        let ring = SPSCFloatRingBuffer(minimumCapacity: 4)
        ring.write([1, 2, 3, 4])
        XCTAssertEqual(ring.availableToWrite, 0)

        let extra = ring.write([5])
        XCTAssertEqual(extra, 0)
        XCTAssertEqual(ring.availableToRead, 4)
    }

    func testWritePartialWhenNearlyFull() {
        let ring = SPSCFloatRingBuffer(minimumCapacity: 4)
        ring.write([1, 2, 3])
        XCTAssertEqual(ring.availableToWrite, 1)

        let written = ring.write([4, 5, 6])
        XCTAssertEqual(written, 1)
        XCTAssertEqual(ring.availableToWrite, 0)

        let result = ring.read(count: 4)
        XCTAssertEqual(result, [1, 2, 3, 4])
    }

    // MARK: - Wrap-around

    func testWriteAndReadAcrossRingBoundary() {
        let ring = SPSCFloatRingBuffer(minimumCapacity: 4)

        // Fill to capacity, then consume half
        ring.write([1, 2, 3, 4])
        _ = ring.read(count: 2)

        // Write two more — these wrap to the beginning of the storage array
        let written = ring.write([5, 6])
        XCTAssertEqual(written, 2)

        // Read all four remaining; the read spans the wrap boundary
        let result = ring.read(count: 4)
        XCTAssertEqual(result, [3, 4, 5, 6])
    }

    func testMultipleFullCycles() {
        let ring = SPSCFloatRingBuffer(minimumCapacity: 4)

        for cycle in 0..<5 {
            let batch: [Float] = [Float(cycle * 4 + 1), Float(cycle * 4 + 2),
                                  Float(cycle * 4 + 3), Float(cycle * 4 + 4)]
            let written = ring.write(batch)
            XCTAssertEqual(written, 4, "Cycle \(cycle): expected 4 samples written")
            let read = ring.read(count: 4)
            XCTAssertEqual(read, batch, "Cycle \(cycle): read samples mismatch")
        }
    }

    // MARK: - discard

    func testDiscardDropsSamplesWithoutReturningThem() {
        let ring = SPSCFloatRingBuffer(minimumCapacity: 8)
        ring.write([1, 2, 3, 4, 5])

        let discarded = ring.discard(count: 2)
        XCTAssertEqual(discarded, 2)
        XCTAssertEqual(ring.availableToRead, 3)

        let result = ring.read(count: 3)
        XCTAssertEqual(result, [3, 4, 5])
    }

    func testDiscardMoreThanAvailableDiscardOnlyAvailable() {
        let ring = SPSCFloatRingBuffer(minimumCapacity: 8)
        ring.write([1, 2])

        let discarded = ring.discard(count: 100)
        XCTAssertEqual(discarded, 2)
        XCTAssertTrue(ring.isEmpty)
    }

    func testDiscardZeroIsNoOp() {
        let ring = SPSCFloatRingBuffer(minimumCapacity: 4)
        ring.write([1, 2, 3])

        let discarded = ring.discard(count: 0)
        XCTAssertEqual(discarded, 0)
        XCTAssertEqual(ring.availableToRead, 3)
    }

    func testDiscardFromEmptyBufferReturnsZero() {
        let ring = SPSCFloatRingBuffer(minimumCapacity: 4)
        let discarded = ring.discard(count: 5)
        XCTAssertEqual(discarded, 0)
    }

    // MARK: - reset

    func testResetClearsAllStateAndAllowsReuse() {
        let ring = SPSCFloatRingBuffer(minimumCapacity: 4)
        ring.write([1, 2, 3, 4])
        _ = ring.read(count: 2)

        ring.reset()

        XCTAssertTrue(ring.isEmpty)
        XCTAssertEqual(ring.availableToRead, 0)
        XCTAssertEqual(ring.availableToWrite, ring.ringCapacity)

        // Should be able to write from fresh state
        let written = ring.write([10, 20])
        XCTAssertEqual(written, 2)
        let result = ring.read(count: 2)
        XCTAssertEqual(result, [10, 20])
    }

    // MARK: - read(into:)

    func testReadIntoUnsafeBufferPointer() {
        let ring = SPSCFloatRingBuffer(minimumCapacity: 8)
        ring.write([1, 2, 3, 4])

        var destination = [Float](repeating: 0, count: 4)
        let actual = destination.withUnsafeMutableBufferPointer { ring.read(into: $0) }
        XCTAssertEqual(actual, 4)
        XCTAssertEqual(destination, [1, 2, 3, 4])
    }

    func testReadIntoEmptyDestinationWritesZero() {
        let ring = SPSCFloatRingBuffer(minimumCapacity: 8)
        ring.write([1, 2])

        var destination = [Float]()
        let actual = destination.withUnsafeMutableBufferPointer { ring.read(into: $0) }
        XCTAssertEqual(actual, 0)
        XCTAssertEqual(ring.availableToRead, 2)
    }

    // MARK: - availableToRead / availableToWrite invariant

    func testAvailableToReadPlusAvailableToWriteEqualsCapacity() {
        let ring = SPSCFloatRingBuffer(minimumCapacity: 8)
        ring.write([0.1, 0.2, 0.3])
        XCTAssertEqual(ring.availableToRead + ring.availableToWrite, ring.ringCapacity)
        _ = ring.read(count: 1)
        XCTAssertEqual(ring.availableToRead + ring.availableToWrite, ring.ringCapacity)
    }

    // MARK: - write(contentsOf:)

    func testWriteContentsOfUnsafeBufferPointer() {
        let ring = SPSCFloatRingBuffer(minimumCapacity: 8)
        let input: [Float] = [0.5, -0.5, 1.0, -1.0]
        let written = input.withUnsafeBufferPointer { ring.write(contentsOf: $0) }
        XCTAssertEqual(written, 4)
        XCTAssertEqual(ring.read(count: 4), input)
    }

    // MARK: - Concurrent correctness (SPSC)

    /// Produces N samples on a background thread and consumes them on the test thread,
    /// confirming that the lock-free SPSC contract holds under concurrent access.
    func testConcurrentProducerConsumerPreservesAllSamples() {
        let ring = SPSCFloatRingBuffer(minimumCapacity: 256)
        let totalSamples = 48_000  // one second at 48 kHz
        let chunkSize = 64

        let input = (0..<totalSamples).map { Float($0) }
        var output: [Float] = []
        output.reserveCapacity(totalSamples)

        let producerQueue = DispatchQueue(label: "spsc.producer", qos: .userInitiated)
        let doneExpectation = expectation(description: "producer done")

        // Producer: write in chunks, yielding to the consumer between writes.
        producerQueue.async {
            var offset = 0
            while offset < totalSamples {
                let end = min(offset + chunkSize, totalSamples)
                let chunk = Array(input[offset..<end])
                var sent = 0
                while sent < chunk.count {
                    let wrote = ring.write(Array(chunk[sent...]))
                    sent += wrote
                    if wrote == 0 { Thread.sleep(forTimeInterval: 0.00001) }
                }
                offset = end
            }
            doneExpectation.fulfill()
        }

        // Consumer: drain until we have all samples.
        let deadline = Date(timeIntervalSinceNow: 10)
        while output.count < totalSamples, Date() < deadline {
            let batch = ring.read(count: chunkSize)
            output.append(contentsOf: batch)
            if batch.isEmpty { Thread.sleep(forTimeInterval: 0.00001) }
        }

        wait(for: [doneExpectation], timeout: 15)

        // Drain any residual samples after producer finishes.
        output.append(contentsOf: ring.read(count: ring.availableToRead))

        XCTAssertEqual(output.count, totalSamples)
        for (index, sample) in output.enumerated() {
            XCTAssertEqual(sample, Float(index), accuracy: 1e-6)
        }
    }

    // MARK: - Throughput benchmark

    /// Measures sustained write → read throughput through the ring buffer.
    ///
    /// The assertion threshold (100 M samples/sec) is conservative — Apple Silicon
    /// delivers well above this in a tight loop — so this primarily guards against
    /// catastrophic regressions such as accidentally introducing a lock.
    func testThroughputBenchmark() {
        let ring = SPSCFloatRingBuffer(minimumCapacity: 4096)
        let totalSamples = 192_000
        let chunkSize = 512
        let input = [Float](repeating: 0.5, count: chunkSize)
        var output = [Float](repeating: 0, count: chunkSize)

        let start = CFAbsoluteTimeGetCurrent()
        var produced = 0
        var consumed = 0

        while consumed < totalSamples {
            if produced - consumed < ring.ringCapacity - chunkSize {
                let wrote = ring.write(input)
                produced += wrote
            }
            let read = output.withUnsafeMutableBufferPointer { ring.read(into: $0) }
            consumed += read
        }

        let elapsed = CFAbsoluteTimeGetCurrent() - start
        let throughputMSamples = Double(totalSamples) / elapsed / 1_000_000

        print(String(
            format: "SPSC throughput: %.2f M samples/sec (%.4f s for %d samples)",
            throughputMSamples, elapsed, totalSamples
        ))

        XCTAssertGreaterThan(
            throughputMSamples,
            100.0,
            "SPSC throughput \(throughputMSamples) M samples/sec is below threshold"
        )
    }
}
