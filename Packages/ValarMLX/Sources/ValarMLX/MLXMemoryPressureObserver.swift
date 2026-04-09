@preconcurrency import Dispatch
import Foundation

/// Monitors macOS kernel memory-pressure events and suspends inference callers
/// until system memory returns to normal.
///
/// `waitUntilClear()` returns immediately when memory pressure is `.normal`.
/// When the kernel delivers a `.warning` or `.critical` event, callers that
/// call `waitUntilClear()` suspend. They are all resumed together when the
/// kernel delivers a subsequent `.normal` event.
///
/// `pressureEvents` is an `AsyncStream` that emits every raw kernel pressure
/// event — subscribers can react to `.warning` or `.critical` levels beyond
/// the simple gate behaviour (e.g. LRU eviction on `.critical`).
///
/// ### Integration
/// Attach a live instance to `MLXInferenceBackend` via the `memoryPressureObserver`
/// injection point. In tests, call `simulatePressureEvent(_:)` to drive the
/// state machine without a real dispatch source.
///
/// ### Thread-safety
/// All mutable state is confined to the `Gate` actor. The dispatch source event
/// handler posts work onto `Gate` via unstructured `Task`s. The class itself is
/// `Sendable` because all stored properties are either actors, `AsyncStream`
/// value types, or immutable sendable values set once at initialisation time.
public final class MLXMemoryPressureObserver: Sendable {

    // MARK: - Gate actor

    // All mutable state lives here. Actor isolation guarantees that waiter
    // bookkeeping and pressure updates are serialised.
    private let gate = Gate()

    private let dispatchSource: (any DispatchSourceMemoryPressure)?

    // MARK: - Event stream

    /// Emits the raw `UInt32` value of every kernel memory-pressure event.
    /// Reconstruct the typed event with `DispatchSource.MemoryPressureEvent(rawValue:)`.
    /// Subscribers can react to individual levels (e.g. `.critical`) without
    /// needing to poll `isUnderPressure`. `UInt32` is used instead of the
    /// non-`Sendable` `DispatchSource.MemoryPressureEvent` so the stream
    /// can flow safely across actor and task boundaries under Swift 6.
    public let pressureEvents: AsyncStream<UInt>

    // Continuation is Sendable and written only once during init.
    private let eventContinuation: AsyncStream<UInt>.Continuation

    // MARK: - Init

    /// Creates and immediately starts observing system memory pressure.
    public init() {
        var cont: AsyncStream<UInt>.Continuation!
        let stream = AsyncStream(UInt.self, bufferingPolicy: .bufferingNewest(16)) { cont = $0 }
        self.eventContinuation = cont
        self.pressureEvents = stream

        let capturedGate = gate
        let capturedContinuation = cont!
        let source = DispatchSource.makeMemoryPressureSource(
            eventMask: [.normal, .warning, .critical],
            queue: DispatchQueue.global(qos: .utility)
        )
        source.setEventHandler { [capturedGate, capturedContinuation] in
            let rawValue = source.data.rawValue
            capturedContinuation.yield(rawValue)
            Task { await capturedGate.handle(event: DispatchSource.MemoryPressureEvent(rawValue: rawValue)) }
        }
        source.resume()
        self.dispatchSource = source
    }

    /// Creates a dormant observer with no live dispatch source — for unit tests.
    init(testingMode: Bool) {
        _ = testingMode
        var cont: AsyncStream<UInt>.Continuation!
        let stream = AsyncStream(UInt.self, bufferingPolicy: .bufferingNewest(16)) { cont = $0 }
        self.eventContinuation = cont
        self.pressureEvents = stream
        self.dispatchSource = nil
    }

    deinit {
        dispatchSource?.cancel()
        eventContinuation.finish()
    }

    // MARK: - Public API

    /// Suspends the caller while memory pressure is elevated (`.warning` or
    /// `.critical`). Returns immediately if pressure is currently `.normal`.
    ///
    /// If the calling `Task` is cancelled while waiting, this method rethrows
    /// a `CancellationError`.
    public func waitUntilClear() async throws {
        try await gate.waitIfPressured()
    }

    /// Queries the current pressure state without suspending.
    public var isUnderPressure: Bool {
        get async { await gate.pressured }
    }

    // MARK: - Testing support

    /// Injects a synthetic pressure event, bypassing the dispatch source.
    /// Yields to `pressureEvents` and updates the Gate in the same call.
    /// Only meaningful for unit tests; has no effect when a real source is
    /// already delivering events.
    func simulatePressureEvent(_ event: DispatchSource.MemoryPressureEvent) async {
        let normalized = DispatchSource.MemoryPressureEvent(rawValue: event.rawValue)
        eventContinuation.yield(normalized.rawValue)
        await gate.handle(event: normalized)
    }

    // MARK: - Gate

    fileprivate actor Gate {
        private(set) var pressured: Bool = false
        private var waiters: [CheckedContinuation<Void, Error>] = []

        func handle(event: DispatchSource.MemoryPressureEvent) {
            if event == .normal {
                pressured = false
                let all = waiters
                waiters.removeAll()
                for continuation in all {
                    continuation.resume()
                }
            } else {
                // .warning or .critical — new callers should suspend.
                pressured = true
            }
        }

        func waitIfPressured() async throws {
            guard pressured else { return }
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
                if Task.isCancelled {
                    continuation.resume(throwing: CancellationError())
                } else {
                    waiters.append(continuation)
                }
            }
        }

        func cancelAllWaiters() {
            let all = waiters
            waiters.removeAll()
            for continuation in all {
                continuation.resume(throwing: CancellationError())
            }
        }
    }
}
