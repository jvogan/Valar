import XCTest
@testable import ValarAudio

final class AudioPlaybackQueueAdmissionControllerTests: XCTestCase {
    func testBackpressureWaitsUntilLowWaterBeforeAdmittingFIFOHead() async throws {
        let controller = try AudioPlaybackQueueAdmissionController(
            limits: AudioPlaybackQueueLimits(
                lowWaterDuration: 1,
                highWaterDuration: 4,
                maximumScheduledBufferCount: 4,
                maximumWaitingProducerCount: 1
            )
        )
        let twoSeconds = AudioPlaybackQueueCost(frameCount: 48_000, sampleRate: 24_000)
        let first = try await controller.acquire(cost: twoSeconds)
        let second = try await controller.acquire(cost: twoSeconds)
        let blocked = Task {
            try await controller.acquire(cost: twoSeconds)
        }

        await waitForWaitingProducerCount(1, controller: controller)
        await controller.release(first)

        var snapshot = await controller.snapshot()
        XCTAssertEqual(snapshot.scheduledDuration, 2, accuracy: 0.000_001)
        XCTAssertEqual(snapshot.waitingProducerCount, 1)
        XCTAssertTrue(snapshot.isBackpressuring)

        await controller.release(second)
        let third = try await blocked.value

        snapshot = await controller.snapshot()
        XCTAssertEqual(snapshot.scheduledDuration, 2, accuracy: 0.000_001)
        XCTAssertEqual(snapshot.scheduledFrameCount, 48_000)
        XCTAssertEqual(snapshot.scheduledBufferCount, 1)
        XCTAssertEqual(snapshot.scheduledByteCount, 48_000 * 4)
        XCTAssertEqual(snapshot.waitingProducerCount, 0)
        XCTAssertFalse(snapshot.isBackpressuring)
        await controller.release(third)
    }

    func testBufferCountBoundsTinyChunksIndependentlyOfDuration() async throws {
        let controller = try AudioPlaybackQueueAdmissionController(
            limits: AudioPlaybackQueueLimits(
                lowWaterDuration: 50,
                highWaterDuration: 100,
                maximumScheduledBufferCount: 2,
                maximumWaitingProducerCount: 1
            )
        )
        let tiny = AudioPlaybackQueueCost(frameCount: 1, sampleRate: 24_000)
        let first = try await controller.acquire(cost: tiny)
        let second = try await controller.acquire(cost: tiny)
        let blocked = Task {
            try await controller.acquire(cost: tiny)
        }

        await waitForWaitingProducerCount(1, controller: controller)
        var snapshot = await controller.snapshot()
        XCTAssertEqual(snapshot.scheduledBufferCount, 2)

        await controller.release(first)
        let third = try await blocked.value
        snapshot = await controller.snapshot()
        XCTAssertEqual(snapshot.scheduledBufferCount, 2)

        await controller.release(second)
        await controller.release(third)
    }

    func testCancellationRemovesSuspendedProducer() async throws {
        let controller = try makeSingleBufferController()
        let oneSecond = AudioPlaybackQueueCost(frameCount: 24_000, sampleRate: 24_000)
        let active = try await controller.acquire(cost: oneSecond)
        let blocked = Task {
            try await controller.acquire(cost: oneSecond)
        }

        await waitForWaitingProducerCount(1, controller: controller)
        blocked.cancel()

        do {
            _ = try await blocked.value
            XCTFail("A cancelled producer must not receive a reservation")
        } catch is CancellationError {
            // Expected.
        }

        let snapshot = await controller.snapshot()
        XCTAssertEqual(snapshot.waitingProducerCount, 0)
        XCTAssertFalse(snapshot.isBackpressuring)
        await controller.release(active)
    }

    func testResetWakesWaiterAndInvalidatesExistingReservations() async throws {
        let controller = try makeSingleBufferController()
        let oneSecond = AudioPlaybackQueueCost(frameCount: 24_000, sampleRate: 24_000)
        let active = try await controller.acquire(cost: oneSecond)
        let blocked = Task {
            try await controller.acquire(cost: oneSecond)
        }

        await waitForWaitingProducerCount(1, controller: controller)
        await controller.reset()

        do {
            _ = try await blocked.value
            XCTFail("Reset must wake a suspended producer with cancellation")
        } catch is CancellationError {
            // Expected.
        }

        await controller.release(active)
        let snapshot = await controller.snapshot()
        XCTAssertEqual(snapshot.scheduledDuration, 0)
        XCTAssertEqual(snapshot.scheduledFrameCount, 0)
        XCTAssertEqual(snapshot.scheduledBufferCount, 0)
        XCTAssertEqual(snapshot.scheduledByteCount, 0)
        XCTAssertEqual(snapshot.waitingProducerCount, 0)
    }

    func testWaitingProducerLimitPreventsUnboundedSuspendedPCM() async throws {
        let controller = try makeSingleBufferController()
        let oneSecond = AudioPlaybackQueueCost(frameCount: 24_000, sampleRate: 24_000)
        let active = try await controller.acquire(cost: oneSecond)
        let firstBlocked = Task {
            try await controller.acquire(cost: oneSecond)
        }

        await waitForWaitingProducerCount(1, controller: controller)

        do {
            _ = try await controller.acquire(cost: oneSecond)
            XCTFail("A second suspended producer should be rejected")
        } catch let error as AudioPlaybackQueueAdmissionError {
            XCTAssertEqual(error, .tooManyWaitingProducers(limit: 1))
        }

        firstBlocked.cancel()
        _ = try? await firstBlocked.value
        await controller.release(active)
    }

    func testOversizedAndInvalidBuffersFailBeforeWaiting() async throws {
        let limits = AudioPlaybackQueueLimits(
            lowWaterDuration: 1,
            highWaterDuration: 2,
            maximumScheduledBufferCount: 2
        )
        let controller = try AudioPlaybackQueueAdmissionController(limits: limits)
        let oversized = AudioPlaybackQueueCost(frameCount: 48_001, sampleRate: 24_000)

        do {
            _ = try await controller.acquire(cost: oversized)
            XCTFail("An oversized buffer should require caller-side splitting")
        } catch let error as AudioPlaybackQueueAdmissionError {
            XCTAssertEqual(error, .bufferExceedsHighWater(oversized, limits))
        }

        let invalid = AudioPlaybackQueueCost(frameCount: 1, sampleRate: .nan)
        do {
            _ = try await controller.acquire(cost: invalid)
            XCTFail("A non-finite sample rate must be rejected")
        } catch let error as AudioPlaybackQueueAdmissionError {
            guard case .invalidCost(let rejectedCost) = error else {
                XCTFail("Expected invalidCost, received \(error)")
                return
            }
            XCTAssertEqual(rejectedCost.frameCount, invalid.frameCount)
            XCTAssertTrue(rejectedCost.sampleRate.isNaN)
        }
    }

    func testDecodedByteBudgetBackpressuresAndRejectsOversizedBuffers() async throws {
        let limits = AudioPlaybackQueueLimits(
            lowWaterDuration: 0.1,
            highWaterDuration: 10,
            maximumScheduledBufferCount: 2,
            maximumWaitingProducerCount: 1,
            maximumScheduledBytes: 100
        )
        let controller = try AudioPlaybackQueueAdmissionController(limits: limits)
        let eightyBytes = AudioPlaybackQueueCost(
            frameCount: 10,
            sampleRate: 10,
            channelCount: 2,
            bytesPerSample: 4
        )
        let twentyFourBytes = AudioPlaybackQueueCost(
            frameCount: 6,
            sampleRate: 10,
            channelCount: 1,
            bytesPerSample: 4
        )
        let tooLarge = AudioPlaybackQueueCost(
            frameCount: 13,
            sampleRate: 10,
            channelCount: 2,
            bytesPerSample: 4
        )

        do {
            _ = try await controller.acquire(cost: tooLarge)
            XCTFail("A single decoded buffer larger than the byte budget must fail")
        } catch let error as AudioPlaybackQueueAdmissionError {
            XCTAssertEqual(error, .bufferExceedsByteLimit(tooLarge, limits))
        }

        let first = try await controller.acquire(cost: eightyBytes)
        let blocked = Task {
            try await controller.acquire(cost: twentyFourBytes)
        }
        await waitForWaitingProducerCount(1, controller: controller)
        var snapshot = await controller.snapshot()
        XCTAssertEqual(snapshot.scheduledByteCount, 80)

        await controller.release(first)
        let second = try await blocked.value
        snapshot = await controller.snapshot()
        XCTAssertEqual(snapshot.scheduledByteCount, 24)
        await controller.release(second)
    }

    func testInvalidLimitsFailClosed() {
        let invalid = AudioPlaybackQueueLimits(
            lowWaterDuration: 4,
            highWaterDuration: 4,
            maximumScheduledBufferCount: 0,
            maximumWaitingProducerCount: 0
        )

        XCTAssertThrowsError(
            try AudioPlaybackQueueAdmissionController(limits: invalid)
        ) { error in
            XCTAssertEqual(
                error as? AudioPlaybackQueueAdmissionError,
                .invalidLimits(invalid)
            )
        }
    }

    private func makeSingleBufferController() throws -> AudioPlaybackQueueAdmissionController {
        try AudioPlaybackQueueAdmissionController(
            limits: AudioPlaybackQueueLimits(
                lowWaterDuration: 0.25,
                highWaterDuration: 1,
                maximumScheduledBufferCount: 1,
                maximumWaitingProducerCount: 1
            )
        )
    }

    private func waitForWaitingProducerCount(
        _ expectedCount: Int,
        controller: AudioPlaybackQueueAdmissionController,
        file: StaticString = #filePath,
        line: UInt = #line
    ) async {
        for _ in 0 ..< 100 {
            if await controller.snapshot().waitingProducerCount == expectedCount {
                return
            }
            await Task.yield()
        }

        XCTFail(
            "Timed out waiting for \(expectedCount) queued playback producer(s)",
            file: file,
            line: line
        )
    }
}
