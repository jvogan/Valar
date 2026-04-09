import Atomics
import Foundation

/// A lock-free Single-Producer Single-Consumer ring buffer for `Float` audio samples.
///
/// `SPSCFloatRingBuffer` allows one *producer* thread to write samples concurrently
/// with one *consumer* thread reading them — without any locks or OS synchronization
/// primitives. Ordering between the two sides is enforced with acquire/release
/// atomics on the head and tail indices.
///
/// **Thread-safety contract**
/// - At most **one** thread may call `write(_:)` / `write(contentsOf:)` at a time.
/// - At most **one** thread may call `read(count:)` / `read(into:)` / `discard(count:)` at a time.
/// - `reset()` must be called only when no concurrent reads or writes are in progress.
///
/// **Capacity**
/// The actual capacity is the smallest power of 2 that is ≥ `minimumCapacity` (at least 2).
/// A power-of-2 capacity lets the ring use a bitmask instead of `%` in the hot path.
///
/// **Memory layout**
/// Samples are stored in a heap-allocated contiguous `Float` array. Reads and writes
/// that wrap around the ring end are split into two `memcpy`-equivalent operations.
public final class SPSCFloatRingBuffer: Sendable {

    // MARK: - Storage

    private let capacity: Int
    private let mask: Int  // capacity - 1; used for fast modular indexing
    // Raw heap storage for audio samples. Thread-safety is enforced by the SPSC protocol:
    // the producer and consumer advance independent indices (head/tail) with acquire/release
    // atomics and never touch the same memory slot concurrently. The unsafe annotation opts
    // out of the automatic Sendable check for this pointer while the rest of the class remains
    // statically verified.
    nonisolated(unsafe) private let storage: UnsafeMutablePointer<Float>

    // Monotonically increasing indices. Actual slot = index & mask.
    // Only the consumer advances `head`; only the producer advances `tail`.
    private let head: ManagedAtomic<Int>  // next read position
    private let tail: ManagedAtomic<Int>  // next write position

    // MARK: - Initialiser

    /// Creates a ring buffer with capacity ≥ `minimumCapacity` samples.
    ///
    /// The actual capacity is rounded up to the next power of 2 (minimum 2).
    public init(minimumCapacity: Int) {
        let cap = Self.nextPowerOfTwo(max(minimumCapacity, 2))
        self.capacity = cap
        self.mask = cap - 1
        self.storage = UnsafeMutablePointer<Float>.allocate(capacity: cap)
        self.storage.initialize(repeating: 0, count: cap)
        self.head = ManagedAtomic<Int>(0)
        self.tail = ManagedAtomic<Int>(0)
    }

    deinit {
        storage.deinitialize(count: capacity)
        storage.deallocate()
    }

    // MARK: - Public properties

    /// The ring buffer's element capacity (always a power of 2).
    public var ringCapacity: Int { capacity }

    /// Samples available for reading.
    ///
    /// Safe to call from the **consumer** thread. The returned count reflects the
    /// producer's most recently committed write.
    public var availableToRead: Int {
        let currentTail = tail.load(ordering: .acquiring)
        let currentHead = head.load(ordering: .relaxed)
        return currentTail &- currentHead
    }

    /// Free slots available for writing.
    ///
    /// Safe to call from the **producer** thread. The returned count reflects the
    /// consumer's most recently committed read.
    public var availableToWrite: Int {
        let currentHead = head.load(ordering: .acquiring)
        let currentTail = tail.load(ordering: .relaxed)
        return capacity &- (currentTail &- currentHead)
    }

    /// `true` when no samples are available to read.
    public var isEmpty: Bool { availableToRead == 0 }

    // MARK: - Write (producer side)

    /// Writes as many samples from `samples` as fit in the ring.
    ///
    /// - Returns: The number of samples actually written; may be less than
    ///   `samples.count` if the ring is full.
    @discardableResult
    public func write(_ samples: [Float]) -> Int {
        samples.withUnsafeBufferPointer { write(contentsOf: $0) }
    }

    /// Writes as many samples from `buffer` as fit in the ring.
    ///
    /// - Returns: The number of samples actually written.
    @discardableResult
    public func write(contentsOf buffer: UnsafeBufferPointer<Float>) -> Int {
        guard let src = buffer.baseAddress, buffer.count > 0 else { return 0 }

        // Producer reads head with acquire to observe the consumer's latest release.
        let currentHead = head.load(ordering: .acquiring)
        let currentTail = tail.load(ordering: .relaxed)  // only producer touches tail
        let free = capacity &- (currentTail &- currentHead)

        let count = min(buffer.count, free)
        guard count > 0 else { return 0 }

        let writeSlot = currentTail & mask
        let firstChunk = min(count, capacity - writeSlot)
        let secondChunk = count - firstChunk

        (storage + writeSlot).update(from: src, count: firstChunk)
        if secondChunk > 0 {
            storage.update(from: src + firstChunk, count: secondChunk)
        }

        // Release so the consumer sees the samples before seeing the new tail.
        tail.store(currentTail &+ count, ordering: .releasing)
        return count
    }

    // MARK: - Read (consumer side)

    /// Reads up to `count` samples into a newly allocated array.
    ///
    /// - Returns: An array containing the samples that were read. May be shorter
    ///   than `count` if fewer samples are available.
    public func read(count: Int) -> [Float] {
        guard count > 0 else { return [] }
        var result = [Float](repeating: 0, count: count)
        let actual = result.withUnsafeMutableBufferPointer { read(into: $0) }
        if actual < count {
            result.removeLast(count - actual)
        }
        return result
    }

    /// Reads as many samples as available (up to `buffer.count`) into `buffer`.
    ///
    /// - Returns: The number of samples actually read.
    @discardableResult
    public func read(into buffer: UnsafeMutableBufferPointer<Float>) -> Int {
        guard let dst = buffer.baseAddress, buffer.count > 0 else { return 0 }

        // Consumer reads tail with acquire to observe the producer's latest release.
        let currentTail = tail.load(ordering: .acquiring)
        let currentHead = head.load(ordering: .relaxed)  // only consumer touches head
        let available = currentTail &- currentHead

        let count = min(buffer.count, available)
        guard count > 0 else { return 0 }

        let readSlot = currentHead & mask
        let firstChunk = min(count, capacity - readSlot)
        let secondChunk = count - firstChunk

        dst.update(from: storage + readSlot, count: firstChunk)
        if secondChunk > 0 {
            (dst + firstChunk).update(from: storage, count: secondChunk)
        }

        // Release so the producer sees the freed slots after we advance head.
        head.store(currentHead &+ count, ordering: .releasing)
        return count
    }

    /// Discards up to `count` samples from the read side without copying them.
    ///
    /// - Returns: The number of samples actually discarded.
    @discardableResult
    public func discard(count: Int) -> Int {
        guard count > 0 else { return 0 }

        let currentTail = tail.load(ordering: .acquiring)
        let currentHead = head.load(ordering: .relaxed)
        let available = currentTail &- currentHead

        let toDrop = min(count, available)
        guard toDrop > 0 else { return 0 }

        head.store(currentHead &+ toDrop, ordering: .releasing)
        return toDrop
    }

    /// Resets the ring buffer to the empty state.
    ///
    /// - Warning: Only call when no concurrent reads or writes are in progress.
    public func reset() {
        head.store(0, ordering: .sequentiallyConsistent)
        tail.store(0, ordering: .sequentiallyConsistent)
    }

    // MARK: - Private helpers

    private static func nextPowerOfTwo(_ n: Int) -> Int {
        guard n > 1 else { return 1 }
        var power = 1
        while power < n { power <<= 1 }
        return power
    }
}
