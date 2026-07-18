import Foundation

/// Low/high-water limits for buffers submitted to an audio playback queue.
///
/// Decoded bytes provide the hard PCM memory bound. Duration limits producer
/// lead, `maximumScheduledBufferCount` bounds overhead from many tiny chunks,
/// and `maximumWaitingProducerCount` prevents suspended producers (and their
/// source PCM) from accumulating without limit.
public struct AudioPlaybackQueueLimits: Codable, Equatable, Sendable {
    public let lowWaterDuration: TimeInterval
    public let highWaterDuration: TimeInterval
    public let maximumScheduledBufferCount: Int
    public let maximumWaitingProducerCount: Int
    public let maximumScheduledBytes: Int64

    public init(
        lowWaterDuration: TimeInterval,
        highWaterDuration: TimeInterval,
        maximumScheduledBufferCount: Int,
        maximumWaitingProducerCount: Int = 1,
        maximumScheduledBytes: Int64 = 8 * 1_024 * 1_024
    ) {
        self.lowWaterDuration = lowWaterDuration
        self.highWaterDuration = highWaterDuration
        self.maximumScheduledBufferCount = maximumScheduledBufferCount
        self.maximumWaitingProducerCount = maximumWaitingProducerCount
        self.maximumScheduledBytes = maximumScheduledBytes
    }

    /// A deliberately small interactive window. Callers should split larger
    /// source buffers before requesting admission.
    public static let interactive = Self(
        lowWaterDuration: 1.5,
        highWaterDuration: 4,
        maximumScheduledBufferCount: 8,
        maximumWaitingProducerCount: 1,
        maximumScheduledBytes: 8 * 1_024 * 1_024
    )
}
/// The amount of PCM represented by one scheduled playback buffer.
public struct AudioPlaybackQueueCost: Codable, Equatable, Sendable {
    public let frameCount: Int64
    public let sampleRate: Double
    public let channelCount: Int
    public let bytesPerSample: Int

    public init(
        frameCount: Int64,
        sampleRate: Double,
        channelCount: Int = 1,
        bytesPerSample: Int = MemoryLayout<Float>.stride
    ) {
        self.frameCount = frameCount
        self.sampleRate = sampleRate
        self.channelCount = channelCount
        self.bytesPerSample = bytesPerSample
    }

    public var duration: TimeInterval {
        guard frameCount >= 0, sampleRate.isFinite, sampleRate > 0 else { return 0 }
        return TimeInterval(frameCount) / sampleRate
    }

    public var decodedByteCount: Int64? {
        guard frameCount >= 0, channelCount > 0, bytesPerSample > 0 else {
            return nil
        }
        let (sampleCount, sampleOverflow) = frameCount.multipliedReportingOverflow(
            by: Int64(channelCount)
        )
        guard !sampleOverflow else { return nil }
        let (byteCount, byteOverflow) = sampleCount.multipliedReportingOverflow(
            by: Int64(bytesPerSample)
        )
        return byteOverflow ? nil : byteCount
    }
}

/// An idempotently releasable claim on playback queue capacity.
public struct AudioPlaybackQueueReservation: Equatable, Sendable, Identifiable {
    public let id: UUID
    public let cost: AudioPlaybackQueueCost

    fileprivate init(id: UUID, cost: AudioPlaybackQueueCost) {
        self.id = id
        self.cost = cost
    }
}

public struct AudioPlaybackQueueAdmissionSnapshot: Equatable, Sendable {
    public let limits: AudioPlaybackQueueLimits
    public let scheduledDuration: TimeInterval
    public let scheduledFrameCount: Int64
    public let scheduledBufferCount: Int
    public let scheduledByteCount: Int64
    public let waitingProducerCount: Int
    public let isBackpressuring: Bool

    public init(
        limits: AudioPlaybackQueueLimits,
        scheduledDuration: TimeInterval,
        scheduledFrameCount: Int64,
        scheduledBufferCount: Int,
        scheduledByteCount: Int64,
        waitingProducerCount: Int,
        isBackpressuring: Bool
    ) {
        self.limits = limits
        self.scheduledDuration = scheduledDuration
        self.scheduledFrameCount = scheduledFrameCount
        self.scheduledBufferCount = scheduledBufferCount
        self.scheduledByteCount = scheduledByteCount
        self.waitingProducerCount = waitingProducerCount
        self.isBackpressuring = isBackpressuring
    }
}

public enum AudioPlaybackQueueAdmissionError: LocalizedError, Equatable, Sendable {
    case invalidLimits(AudioPlaybackQueueLimits)
    case invalidCost(AudioPlaybackQueueCost)
    case bufferExceedsHighWater(AudioPlaybackQueueCost, AudioPlaybackQueueLimits)
    case bufferExceedsByteLimit(AudioPlaybackQueueCost, AudioPlaybackQueueLimits)
    case tooManyWaitingProducers(limit: Int)

    public var errorDescription: String? {
        switch self {
        case .invalidLimits:
            return "Playback queue limits require a finite nonnegative low water, a larger finite high water, and positive counts."
        case .invalidCost:
            return "Playback queue costs require a nonnegative frame count, a finite positive sample rate, and positive channel/sample widths."
        case .bufferExceedsHighWater:
            return "The playback buffer is larger than the configured high-water duration; split it before scheduling."
        case .bufferExceedsByteLimit:
            return "The decoded playback buffer is larger than the configured byte limit; split it before scheduling."
        case .tooManyWaitingProducers(let limit):
            return "The playback queue already has its limit of \(limit) waiting producer(s)."
        }
    }
}

/// FIFO, cancellation-aware admission for a bounded playback queue.
///
/// Once a producer reaches the high-water mark, new work stays suspended until
/// released audio reaches the low-water mark. This hysteresis avoids waking a
/// producer for every render callback. `reset()` cancels all suspended producers
/// and invalidates outstanding reservations, making it suitable for `stop()`.
///
/// The controller owns no PCM or `AVAudioPCMBuffer` objects. A playback engine
/// acquires immediately before materializing/scheduling a buffer and releases
/// from its render-completion callback.
public actor AudioPlaybackQueueAdmissionController {
    private struct Waiter {
        let id: UUID
        let cost: AudioPlaybackQueueCost
        let durationNanoseconds: UInt64
        let decodedByteCount: Int64
        let continuation: CheckedContinuation<AudioPlaybackQueueReservation, any Error>
    }

    private let limits: AudioPlaybackQueueLimits
    private let lowWaterNanoseconds: UInt64
    private let highWaterNanoseconds: UInt64

    private var scheduledNanoseconds: UInt64 = 0
    private var scheduledFrameCount: Int64 = 0
    private var scheduledByteCount: Int64 = 0
    private var activeReservations: [
        UUID: (
            cost: AudioPlaybackQueueCost,
            durationNanoseconds: UInt64,
            decodedByteCount: Int64
        )
    ] = [:]
    private var waiters: [Waiter] = []
    private var isBackpressuring = false

    public init(limits: AudioPlaybackQueueLimits = .interactive) throws {
        guard let lowWaterNanoseconds = Self.nanoseconds(for: limits.lowWaterDuration),
              let highWaterNanoseconds = Self.nanoseconds(for: limits.highWaterDuration),
              lowWaterNanoseconds < highWaterNanoseconds,
              limits.maximumScheduledBufferCount > 0,
              limits.maximumWaitingProducerCount > 0,
              limits.maximumScheduledBytes > 0 else {
            throw AudioPlaybackQueueAdmissionError.invalidLimits(limits)
        }

        self.limits = limits
        self.lowWaterNanoseconds = lowWaterNanoseconds
        self.highWaterNanoseconds = highWaterNanoseconds
    }

    /// Suspends until the buffer can be scheduled without crossing the high
    /// water or buffer-count limit. Requests remain FIFO once backpressure
    /// begins; a smaller request cannot bypass a blocked earlier request.
    public func acquire(
        cost: AudioPlaybackQueueCost
    ) async throws -> AudioPlaybackQueueReservation {
        guard let durationNanoseconds = Self.durationNanoseconds(for: cost) else {
            throw AudioPlaybackQueueAdmissionError.invalidCost(cost)
        }
        guard let decodedByteCount = cost.decodedByteCount else {
            throw AudioPlaybackQueueAdmissionError.invalidCost(cost)
        }
        guard durationNanoseconds <= highWaterNanoseconds else {
            throw AudioPlaybackQueueAdmissionError.bufferExceedsHighWater(cost, limits)
        }
        guard decodedByteCount <= limits.maximumScheduledBytes else {
            throw AudioPlaybackQueueAdmissionError.bufferExceedsByteLimit(cost, limits)
        }
        try Task.checkCancellation()

        if waiters.isEmpty,
           !isBackpressuring,
           canAdmit(
               cost: cost,
               durationNanoseconds: durationNanoseconds,
               decodedByteCount: decodedByteCount
           ) {
            let reservation = makeReservation(
                cost: cost,
                durationNanoseconds: durationNanoseconds,
                decodedByteCount: decodedByteCount
            )
            do {
                try Task.checkCancellation()
                return reservation
            } catch {
                release(reservation)
                throw error
            }
        }

        guard waiters.count < limits.maximumWaitingProducerCount else {
            throw AudioPlaybackQueueAdmissionError.tooManyWaitingProducers(
                limit: limits.maximumWaitingProducerCount
            )
        }

        isBackpressuring = true
        let requestID = UUID()
        let reservation = try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                waiters.append(
                    Waiter(
                        id: requestID,
                        cost: cost,
                        durationNanoseconds: durationNanoseconds,
                        decodedByteCount: decodedByteCount,
                        continuation: continuation
                    )
                )
                // Cancellation may reach the actor before this continuation is
                // installed. Close that race after insertion so the producer
                // cannot remain suspended forever.
                if Task.isCancelled {
                    let cancelled = waiters.removeLast()
                    cancelled.continuation.resume(throwing: CancellationError())
                    if waiters.isEmpty {
                        isBackpressuring = false
                    }
                } else {
                    drainWaitersIfReady()
                }
            }
        } onCancel: {
            Task { await self.cancelPendingRequest(id: requestID) }
        }

        do {
            // `reset()` can interleave after a waiter is resumed but before
            // this actor method runs again. Never return that stale claim.
            guard activeReservations[reservation.id] != nil else {
                throw CancellationError()
            }
            try Task.checkCancellation()
            return reservation
        } catch {
            release(reservation)
            throw error
        }
    }

    /// Releases a completed buffer. Releasing an old or already released
    /// reservation is intentionally a no-op.
    public func release(_ reservation: AudioPlaybackQueueReservation) {
        guard let active = activeReservations.removeValue(forKey: reservation.id) else {
            return
        }

        scheduledNanoseconds = scheduledNanoseconds >= active.durationNanoseconds
            ? scheduledNanoseconds - active.durationNanoseconds
            : 0
        scheduledFrameCount = active.cost.frameCount <= scheduledFrameCount
            ? scheduledFrameCount - active.cost.frameCount
            : 0
        scheduledByteCount = active.decodedByteCount <= scheduledByteCount
            ? scheduledByteCount - active.decodedByteCount
            : 0
        drainWaitersIfReady()
    }

    /// Invalidates active reservations and wakes every suspended producer with
    /// `CancellationError`. Intended to be called by playback stop/reset.
    public func reset() {
        activeReservations.removeAll(keepingCapacity: true)
        scheduledNanoseconds = 0
        scheduledFrameCount = 0
        scheduledByteCount = 0
        isBackpressuring = false

        let cancelledWaiters = waiters
        waiters.removeAll(keepingCapacity: true)
        for waiter in cancelledWaiters {
            waiter.continuation.resume(throwing: CancellationError())
        }
    }

    public func snapshot() -> AudioPlaybackQueueAdmissionSnapshot {
        AudioPlaybackQueueAdmissionSnapshot(
            limits: limits,
            scheduledDuration: TimeInterval(scheduledNanoseconds) / 1_000_000_000,
            scheduledFrameCount: scheduledFrameCount,
            scheduledBufferCount: activeReservations.count,
            scheduledByteCount: scheduledByteCount,
            waitingProducerCount: waiters.count,
            isBackpressuring: isBackpressuring
        )
    }

    private func cancelPendingRequest(id: UUID) {
        guard let index = waiters.firstIndex(where: { $0.id == id }) else {
            return
        }

        let waiter = waiters.remove(at: index)
        waiter.continuation.resume(throwing: CancellationError())
        if waiters.isEmpty {
            isBackpressuring = false
        } else {
            drainWaitersIfReady()
        }
    }

    private func drainWaitersIfReady() {
        guard !waiters.isEmpty else {
            isBackpressuring = false
            return
        }

        if isBackpressuring, scheduledNanoseconds > lowWaterNanoseconds {
            return
        }

        isBackpressuring = false
        while let waiter = waiters.first,
              canAdmit(
                  cost: waiter.cost,
                  durationNanoseconds: waiter.durationNanoseconds,
                  decodedByteCount: waiter.decodedByteCount
              ) {
            waiters.removeFirst()
            let reservation = makeReservation(
                cost: waiter.cost,
                durationNanoseconds: waiter.durationNanoseconds,
                decodedByteCount: waiter.decodedByteCount
            )
            waiter.continuation.resume(returning: reservation)
        }

        if !waiters.isEmpty {
            isBackpressuring = true
        }
    }

    private func canAdmit(
        cost: AudioPlaybackQueueCost,
        durationNanoseconds: UInt64,
        decodedByteCount: Int64
    ) -> Bool {
        guard activeReservations.count < limits.maximumScheduledBufferCount,
              durationNanoseconds <= highWaterNanoseconds - scheduledNanoseconds,
              decodedByteCount <= limits.maximumScheduledBytes - scheduledByteCount,
              cost.frameCount <= Int64.max - scheduledFrameCount else {
            return false
        }
        return true
    }

    private func makeReservation(
        cost: AudioPlaybackQueueCost,
        durationNanoseconds: UInt64,
        decodedByteCount: Int64
    ) -> AudioPlaybackQueueReservation {
        let reservation = AudioPlaybackQueueReservation(id: UUID(), cost: cost)
        activeReservations[reservation.id] = (
            cost,
            durationNanoseconds,
            decodedByteCount
        )
        scheduledNanoseconds += durationNanoseconds
        scheduledFrameCount += cost.frameCount
        scheduledByteCount += decodedByteCount
        return reservation
    }

    private static func durationNanoseconds(
        for cost: AudioPlaybackQueueCost
    ) -> UInt64? {
        guard cost.frameCount >= 0,
              cost.sampleRate.isFinite,
              cost.sampleRate > 0 else {
            return nil
        }

        let duration = TimeInterval(cost.frameCount) / cost.sampleRate
        guard duration.isFinite, duration >= 0 else {
            return nil
        }
        return nanoseconds(for: duration, roundingUp: true)
    }

    private static func nanoseconds(
        for duration: TimeInterval,
        roundingUp: Bool = false
    ) -> UInt64? {
        guard duration.isFinite, duration >= 0 else { return nil }
        let scaled = duration * 1_000_000_000
        // `Double(UInt64.max)` rounds up to 2^64. Require a strict bound so
        // conversion can never trap at that rounded, unrepresentable endpoint.
        guard scaled.isFinite, scaled < TimeInterval(UInt64.max) else { return nil }
        return UInt64(roundingUp ? scaled.rounded(.up) : scaled.rounded(.down))
    }
}
