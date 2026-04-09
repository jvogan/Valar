import AppKit
import Foundation
import QuartzCore

/// A display-synced playback position publisher.
///
/// Uses `CADisplayLink` (macOS 14+, created via `NSScreen.displayLink`) to
/// emit the current playback position on every display frame — typically 60 Hz
/// or 120 Hz on modern Apple hardware — replacing coarse `Task.sleep` polling.
///
/// The publisher is `@MainActor`-isolated. Call `start()` when playback begins
/// and `stop()` when it pauses; the same publisher can be restarted any number
/// of times. The underlying `AsyncStream` stays open across stop/start cycles
/// and terminates only when the publisher is deallocated.
///
/// ## Usage
/// ```swift
/// let publisher = PlaybackPositionPublisher { currentPosition }
/// publisher.start()
///
/// for await position in publisher.stream {
///     updateProgressIndicator(to: position)
/// }
/// ```
@MainActor
public final class PlaybackPositionPublisher {

    // MARK: - Public API

    /// A stream of playback position updates in seconds.
    ///
    /// Values are emitted once per display frame while the publisher is active.
    /// The stream terminates when the publisher is deallocated.
    public let stream: AsyncStream<TimeInterval>

    // MARK: - Private state

    // nonisolated(unsafe) allows deinit (which is nonisolated in Swift 6) to
    // reach these properties. Both are only mutated on the main actor during
    // normal operation, and deinit is called by ARC when the last reference
    // (held on the main actor) is released, so access in deinit is safe.
    nonisolated(unsafe) private var continuation: AsyncStream<TimeInterval>.Continuation?
    nonisolated(unsafe) private var displayLink: CADisplayLink?

    private let positionProvider: @MainActor () -> TimeInterval
    private let bridge: DisplayLinkBridge

    // MARK: - Initializer

    /// Creates a position publisher backed by `positionProvider`.
    ///
    /// - Parameter positionProvider: A closure returning the current playback
    ///   position in seconds. Invoked on the main actor once per display frame
    ///   while the publisher is active. May capture `@MainActor`-isolated state.
    public init(positionProvider: @MainActor @escaping () -> TimeInterval) {
        self.positionProvider = positionProvider

        let bridge = DisplayLinkBridge()
        self.bridge = bridge

        var capturedContinuation: AsyncStream<TimeInterval>.Continuation?
        self.stream = AsyncStream { continuation in
            capturedContinuation = continuation
        }
        self.continuation = capturedContinuation

        // Wire the per-frame callback once all stored properties are set.
        // [weak self] prevents a retain cycle: publisher → bridge → handler → [weak publisher].
        bridge.handler = { [weak self] in
            guard let self else { return }
            let position = self.positionProvider()
            self.continuation?.yield(position)
        }
    }

    // MARK: - Lifecycle

    /// Starts the display link, beginning position updates.
    ///
    /// Calling `start()` while already active is a no-op.
    ///
    /// - Note: Requires `NSScreen.main` to be non-nil. In rare headless
    ///   environments where no screen is attached, this method silently returns
    ///   without activating the display link.
    public func start() {
        guard displayLink == nil else { return }
        guard let screen = NSScreen.main else { return }
        let link = screen.displayLink(target: bridge, selector: #selector(DisplayLinkBridge.fire))
        link.add(to: .main, forMode: .common)
        displayLink = link
    }

    /// Stops the display link, pausing position updates.
    ///
    /// The stream remains open; call `start()` again to resume updates.
    /// Calling `stop()` while already inactive is a no-op.
    public func stop() {
        displayLink?.invalidate()
        displayLink = nil
    }

    deinit {
        displayLink?.invalidate()
        continuation?.finish()
    }
}

// MARK: - Internal bridge

/// Bridges `CADisplayLink`'s Objective-C selector callback to a Swift closure.
///
/// `CADisplayLink` retains its target strongly. Using a separate bridge object
/// (rather than making `PlaybackPositionPublisher` itself the target) avoids a
/// retain cycle: publisher → displayLink → bridge → handler [weak publisher].
private final class DisplayLinkBridge: NSObject {

    /// Per-frame callback, always invoked on the main thread.
    ///
    /// Stored as `nonisolated(unsafe)` because it is written exactly once
    /// during `PlaybackPositionPublisher.init` (on the main actor) and is
    /// thereafter only read from `fire()` on the main thread. The write and
    /// all reads share the same thread, so no data race can occur.
    nonisolated(unsafe) var handler: (@MainActor () -> Void)?

    /// `CADisplayLink` fires this selector on the main thread. Using
    /// `MainActor.assumeIsolated` transfers to main-actor isolation without
    /// suspension, which is valid because `CADisplayLink` callbacks always
    /// arrive on the main thread.
    @objc func fire() {
        guard let handler else { return }
        MainActor.assumeIsolated {
            handler()
        }
    }
}
