import AVFoundation
import AudioToolbox

public actor AudioEnginePlayer {
    private let engine: AVAudioEngine
    private var playerNode: AVAudioPlayerNode?

    private var pendingBuffers = AudioChunkRingBuffer<QueuedAudioBuffer>()
    private var activeFormat: AudioFormatDescriptor?
    private var totalEnqueuedFrames: Int64 = 0
    private var playedFrames: Int64 = 0
    private var playbackBaseFrames: Int64 = 0
    private var scheduledBufferCount = 0
    private var finishedStreaming = false
    private var isPlaying = false
    private var isBuffering = false
    private var didFinishPlayback = false
    private var playbackSessionID: UInt64 = 0

    public init() {
        self.engine = AVAudioEngine()
    }

    public func play(_ buffer: AudioPCMBuffer) async throws {
        stop()
        try feedChunk(buffer)
        finishStream()
    }

    public func feedChunk(_ buffer: AudioPCMBuffer) throws {
        try ensureEngineReady(for: buffer.format)

        finishedStreaming = false
        didFinishPlayback = false

        let avBuffer = try Self.makeAVAudioBuffer(from: buffer)
        let queuedBuffer = QueuedAudioBuffer(
            buffer: avBuffer,
            frameCount: Int64(buffer.frameCount)
        )

        pendingBuffers.append(queuedBuffer)
        totalEnqueuedFrames += queuedBuffer.frameCount

        schedulePendingBuffers()
        beginPlaybackIfNeeded()
    }

    /// Fast path that feeds a raw Float array directly into `AVAudioPCMBuffer.floatChannelData`
    /// via `memcpy`, bypassing any intermediate `Data` conversion.
    ///
    /// - Parameters:
    ///   - samples: Mono PCM samples in the range [-1, 1].
    ///   - sampleRate: Sample rate of the incoming audio (e.g. 24_000).
    public func feedSamples(_ samples: [Float], sampleRate: Double) throws {
        let format = AudioFormatDescriptor(sampleRate: sampleRate, channelCount: 1)
        try ensureEngineReady(for: format)

        finishedStreaming = false
        didFinishPlayback = false

        let avBuffer = try Self.makeAVAudioBufferFromSamples(samples, sampleRate: sampleRate)
        let queuedBuffer = QueuedAudioBuffer(
            buffer: avBuffer,
            frameCount: Int64(samples.count)
        )

        pendingBuffers.append(queuedBuffer)
        totalEnqueuedFrames += queuedBuffer.frameCount

        schedulePendingBuffers()
        beginPlaybackIfNeeded()
    }

    public func finishStream() {
        finishedStreaming = true
        updateTerminalStateIfNeeded()
    }

    public func stop() {
        playbackSessionID &+= 1
        playerNode?.stop()
        if engine.isRunning {
            engine.stop()
        }

        resetPlaybackState(didFinishPlayback: false)
    }

    public func playbackSnapshot() -> AudioPlaybackSnapshot {
        let sampleRate = activeFormat?.sampleRate ?? 0
        let positionFrames = currentPositionFrames()
        let queuedFrames = max(totalEnqueuedFrames - positionFrames, 0)

        return AudioPlaybackSnapshot(
            position: Self.seconds(forFrames: positionFrames, sampleRate: sampleRate),
            queuedDuration: Self.seconds(forFrames: queuedFrames, sampleRate: sampleRate),
            isPlaying: isPlaying,
            isBuffering: isBuffering,
            didFinish: didFinishPlayback
        )
    }
}

extension AudioEnginePlayer {
    private func resolvedPlayerNode() throws -> AVAudioPlayerNode {
        if let playerNode {
            return playerNode
        }

        guard Self.playbackComponentIsAvailable() else {
            throw AudioEnginePlayerError.playbackUnavailable
        }

        let playerNode = AVAudioPlayerNode()
        engine.attach(playerNode)
        engine.connect(playerNode, to: engine.mainMixerNode, format: nil)
        self.playerNode = playerNode
        return playerNode
    }

    private func ensureEngineReady(for format: AudioFormatDescriptor) throws {
        let playerNode = try resolvedPlayerNode()

        if let currentFormat = activeFormat, currentFormat != format {
            if totalEnqueuedFrames > 0 {
                throw AudioEnginePlayerError.formatChangedDuringPlayback(
                    expected: currentFormat,
                    actual: format
                )
            }

            playerNode.stop()
            if engine.isRunning {
                engine.stop()
            }

            let avFormat = try Self.makeAVAudioFormat(from: format)
            engine.disconnectNodeOutput(playerNode)
            engine.connect(playerNode, to: engine.mainMixerNode, format: avFormat)
            activeFormat = format
        } else if activeFormat == nil {
            let avFormat = try Self.makeAVAudioFormat(from: format)
            engine.disconnectNodeOutput(playerNode)
            engine.connect(playerNode, to: engine.mainMixerNode, format: avFormat)
            activeFormat = format
        }

        if !engine.isRunning {
            try engine.start()
        }
    }

    private func schedulePendingBuffers() {
        guard let playerNode else { return }

        while let queuedBuffer = pendingBuffers.popFirst() {
            scheduledBufferCount += 1
            let frameCount = queuedBuffer.frameCount
            let sessionID = playbackSessionID

            playerNode.scheduleBuffer(
                queuedBuffer.buffer,
                completionCallbackType: .dataPlayedBack
            ) { [self] _ in
                Task {
                    await markBufferPlayedBack(frameCount: frameCount, sessionID: sessionID)
                }
            }
        }
    }

    private func beginPlaybackIfNeeded() {
        guard let playerNode else { return }
        guard scheduledBufferCount > 0 else { return }
        guard !playerNode.isPlaying else {
            isPlaying = true
            isBuffering = false
            return
        }

        playbackBaseFrames = playedFrames
        playerNode.play()
        isPlaying = true
        isBuffering = false
    }

    private func markBufferPlayedBack(frameCount: Int64, sessionID: UInt64) {
        guard sessionID == playbackSessionID else { return }
        playedFrames += frameCount
        scheduledBufferCount = max(scheduledBufferCount - 1, 0)
        updateTerminalStateIfNeeded()
    }

    private func updateTerminalStateIfNeeded() {
        guard let playerNode else { return }
        guard scheduledBufferCount == 0, pendingBuffers.isEmpty else { return }

        playerNode.stop()

        if finishedStreaming {
            if engine.isRunning {
                engine.stop()
            }
            resetPlaybackState(didFinishPlayback: totalEnqueuedFrames > 0)
        } else {
            playbackBaseFrames = playedFrames
            isPlaying = false
            isBuffering = totalEnqueuedFrames > 0
            didFinishPlayback = false
        }
    }

    private func resetPlaybackState(didFinishPlayback: Bool) {
        pendingBuffers.removeAll(keepingCapacity: true)
        activeFormat = nil
        totalEnqueuedFrames = 0
        playedFrames = 0
        playbackBaseFrames = 0
        scheduledBufferCount = 0
        finishedStreaming = false
        isPlaying = false
        isBuffering = false
        self.didFinishPlayback = didFinishPlayback
    }

    private func currentPositionFrames() -> Int64 {
        guard let playerNode else { return playedFrames }
        var positionFrames = playedFrames

        if playerNode.isPlaying,
           let renderTime = playerNode.lastRenderTime,
           let playerTime = playerNode.playerTime(forNodeTime: renderTime) {
            let liveFrames = max(Int64(playerTime.sampleTime), 0)
            positionFrames = max(positionFrames, playbackBaseFrames + liveFrames)
        }

        return min(positionFrames, totalEnqueuedFrames)
    }

    private static func playbackComponentIsAvailable() -> Bool {
        var description = AudioComponentDescription(
            componentType: kAudioUnitType_Generator,
            componentSubType: kAudioUnitSubType_ScheduledSoundPlayer,
            componentManufacturer: kAudioUnitManufacturer_Apple,
            componentFlags: 0,
            componentFlagsMask: 0
        )

        return AudioComponentFindNext(nil, &description) != nil
    }

    private static func makeAVAudioFormat(from format: AudioFormatDescriptor) throws -> AVAudioFormat {
        guard let avFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: format.sampleRate,
            channels: AVAudioChannelCount(format.channelCount),
            interleaved: format.interleaved
        ) else {
            throw AudioEnginePlayerError.invalidFormat(
                sampleRate: format.sampleRate,
                channelCount: format.channelCount
            )
        }

        return avFormat
    }

    private static func makeAVAudioBuffer(from buffer: AudioPCMBuffer) throws -> AVAudioPCMBuffer {
        guard buffer.format.channelCount > 0, buffer.format.sampleRate > 0 else {
            throw AudioEnginePlayerError.invalidFormat(
                sampleRate: buffer.format.sampleRate,
                channelCount: buffer.format.channelCount
            )
        }
        let avFormat = try makeAVAudioFormat(from: buffer.format)
        guard buffer.frameCount >= 0, buffer.frameCount <= Int(AVAudioFrameCount.max) else {
            throw AudioEnginePlayerError.bufferAllocationFailed(frameCount: buffer.frameCount)
        }
        let frameCapacity = AVAudioFrameCount(buffer.frameCount)
        guard let avBuffer = AVAudioPCMBuffer(
            pcmFormat: avFormat,
            frameCapacity: frameCapacity
        ) else {
            throw AudioEnginePlayerError.bufferAllocationFailed(frameCount: buffer.frameCount)
        }

        avBuffer.frameLength = frameCapacity
        PCMCoding.fill(avBuffer, from: buffer.channels, frameCount: buffer.frameCount)

        return avBuffer
    }

    /// Fills a freshly-allocated `AVAudioPCMBuffer` from a mono Float array using `memcpy`,
    /// with no intermediate `Data` conversion.
    ///
    /// Marked `internal` so the test target can verify buffer contents via `@testable import`.
    static func makeAVAudioBufferFromSamples(_ samples: [Float], sampleRate: Double) throws -> AVAudioPCMBuffer {
        guard sampleRate > 0 else {
            throw AudioEnginePlayerError.invalidFormat(sampleRate: sampleRate, channelCount: 1)
        }
        guard let avFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: sampleRate,
            channels: 1,
            interleaved: false
        ) else {
            throw AudioEnginePlayerError.invalidFormat(sampleRate: sampleRate, channelCount: 1)
        }

        let frameCount = samples.count
        guard frameCount <= Int(AVAudioFrameCount.max) else {
            throw AudioEnginePlayerError.bufferAllocationFailed(frameCount: frameCount)
        }

        // Allocate with at least capacity 1 so AVAudioPCMBuffer always returns non-nil.
        let frameCapacity = AVAudioFrameCount(max(frameCount, 1))
        guard let avBuffer = AVAudioPCMBuffer(pcmFormat: avFormat, frameCapacity: frameCapacity) else {
            throw AudioEnginePlayerError.bufferAllocationFailed(frameCount: frameCount)
        }

        avBuffer.frameLength = AVAudioFrameCount(frameCount)

        // Directly memcpy into floatChannelData — no Data intermediary.
        if frameCount > 0, let destination = avBuffer.floatChannelData?[0] {
            _ = samples.withUnsafeBufferPointer { src in
                memcpy(destination, src.baseAddress!, frameCount * MemoryLayout<Float>.stride)
            }
        }

        return avBuffer
    }

    private static func seconds(forFrames frames: Int64, sampleRate: Double) -> TimeInterval {
        guard sampleRate > 0 else { return 0 }
        return TimeInterval(frames) / sampleRate
    }
}

private struct QueuedAudioBuffer {
    let buffer: AVAudioPCMBuffer
    let frameCount: Int64
}

private struct AudioChunkRingBuffer<Element> {
    private var storage: [Element?]
    private var head = 0
    private var tail = 0
    private(set) var count = 0

    init(minimumCapacity: Int = 8) {
        storage = Array(repeating: nil, count: max(minimumCapacity, 1))
    }

    var isEmpty: Bool {
        count == 0
    }

    mutating func append(_ element: Element) {
        if count == storage.count {
            grow()
        }

        storage[tail] = element
        tail = (tail + 1) % storage.count
        count += 1
    }

    mutating func popFirst() -> Element? {
        guard count > 0 else { return nil }

        let element = storage[head]
        storage[head] = nil
        head = (head + 1) % storage.count
        count -= 1
        return element
    }

    mutating func removeAll(keepingCapacity: Bool) {
        if keepingCapacity {
            storage = Array(repeating: nil, count: storage.count)
        } else {
            storage = [nil]
        }

        head = 0
        tail = 0
        count = 0
    }

    private mutating func grow() {
        let oldStorage = storage
        let newCapacity = max(storage.count * 2, 1)
        storage = Array(repeating: nil, count: newCapacity)

        for index in 0 ..< count {
            storage[index] = oldStorage[(head + index) % oldStorage.count]
        }

        head = 0
        tail = count
    }
}

private enum AudioEnginePlayerError: Error {
    case bufferAllocationFailed(frameCount: Int)
    case formatChangedDuringPlayback(expected: AudioFormatDescriptor, actual: AudioFormatDescriptor)
    case invalidFormat(sampleRate: Double, channelCount: Int)
    case playbackUnavailable
}
