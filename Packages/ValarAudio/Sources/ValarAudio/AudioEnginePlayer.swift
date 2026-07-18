import AVFoundation
import AudioToolbox

public actor AudioEnginePlayer {
    enum QueueDrainDisposition: Equatable {
        case awaitMoreAudio
        case finishPlayback(didPlayAudio: Bool)
    }

    private let engine: AVAudioEngine
    private let playbackQueueLimits: AudioPlaybackQueueLimits
    private let playbackAdmission: AudioPlaybackQueueAdmissionController
    private var playerNode: AVAudioPlayerNode?

    private var pendingBuffers = AudioChunkRingBuffer<QueuedAudioBuffer>()
    private var activeFormat: AudioFormatDescriptor?
    private var totalEnqueuedFrames: Int64 = 0
    private var playedFrames: Int64 = 0
    private var playbackBaseFrames: Int64 = 0
    private var playbackNodeBaseSampleTime: Int64 = 0
    private var scheduledBufferCount = 0
    private var finishedStreaming = false
    private var isPlaying = false
    private var isBuffering = false
    private var didFinishPlayback = false
    private var playbackSessionID: UInt64 = 0

    public init() {
        let limits = AudioPlaybackQueueLimits.interactive
        self.engine = AVAudioEngine()
        self.playbackQueueLimits = limits
        self.playbackAdmission = Self.makeDefaultPlaybackAdmission(limits: limits)
    }

    public init(playbackQueueLimits: AudioPlaybackQueueLimits) throws {
        self.engine = AVAudioEngine()
        self.playbackQueueLimits = playbackQueueLimits
        self.playbackAdmission = try AudioPlaybackQueueAdmissionController(
            limits: playbackQueueLimits
        )
    }

    public func play(_ buffer: AudioPCMBuffer) async throws {
        await stop()
        do {
            try Self.validatePlaybackBufferShape(buffer)
            let chunkFrameCount = try maximumPlaybackChunkFrameCount(
                for: buffer.format
            )
            var startFrame = 0
            while startFrame < buffer.frameCount {
                try Task.checkCancellation()
                let endFrame = startFrame + min(
                    chunkFrameCount,
                    buffer.frameCount - startFrame
                )
                let channels = buffer.channels.map { channel -> [Float] in
                    guard startFrame < channel.count else { return [] }
                    return Array(channel[startFrame ..< min(endFrame, channel.count)])
                }
                try await feedChunk(
                    AudioPCMBuffer(channels: channels, format: buffer.format)
                )
                startFrame = endFrame
            }
            finishStream()
        } catch {
            await stop()
            throw error
        }
    }

    public func feedChunk(_ buffer: AudioPCMBuffer) async throws {
        try Self.validatePlaybackBufferShape(buffer)
        let sessionID = playbackSessionID
        let cost = AudioPlaybackQueueCost(
            frameCount: Int64(buffer.frameCount),
            sampleRate: buffer.format.sampleRate,
            channelCount: buffer.format.channelCount,
            bytesPerSample: MemoryLayout<Float>.stride
        )
        let reservation = try await playbackAdmission.acquire(cost: cost)
        guard sessionID == playbackSessionID, !Task.isCancelled else {
            await playbackAdmission.release(reservation)
            throw CancellationError()
        }

        do {
            try ensureEngineReady(for: buffer.format)

            finishedStreaming = false
            didFinishPlayback = false

            let avBuffer = try Self.makeAVAudioBuffer(from: buffer)
            let queuedBuffer = QueuedAudioBuffer(
                buffer: avBuffer,
                frameCount: Int64(buffer.frameCount),
                reservation: reservation
            )

            pendingBuffers.append(queuedBuffer)
            totalEnqueuedFrames += queuedBuffer.frameCount

            schedulePendingBuffers()
            beginPlaybackIfNeeded()
        } catch {
            await playbackAdmission.release(reservation)
            throw error
        }
    }

    /// Fast path that feeds a raw Float array directly into `AVAudioPCMBuffer.floatChannelData`
    /// via `memcpy`, bypassing any intermediate `Data` conversion.
    ///
    /// - Parameters:
    ///   - samples: Mono PCM samples in the range [-1, 1].
    ///   - sampleRate: Sample rate of the incoming audio (e.g. 24_000).
    public func feedSamples(_ samples: [Float], sampleRate: Double) async throws {
        let sessionID = playbackSessionID
        let format = AudioFormatDescriptor(sampleRate: sampleRate, channelCount: 1)
        let cost = AudioPlaybackQueueCost(
            frameCount: Int64(samples.count),
            sampleRate: sampleRate,
            channelCount: 1,
            bytesPerSample: MemoryLayout<Float>.stride
        )
        let reservation = try await playbackAdmission.acquire(cost: cost)
        guard sessionID == playbackSessionID, !Task.isCancelled else {
            await playbackAdmission.release(reservation)
            throw CancellationError()
        }

        do {
            try ensureEngineReady(for: format)

            finishedStreaming = false
            didFinishPlayback = false

            let avBuffer = try Self.makeAVAudioBufferFromSamples(
                samples,
                sampleRate: sampleRate
            )
            let queuedBuffer = QueuedAudioBuffer(
                buffer: avBuffer,
                frameCount: Int64(samples.count),
                reservation: reservation
            )

            pendingBuffers.append(queuedBuffer)
            totalEnqueuedFrames += queuedBuffer.frameCount

            schedulePendingBuffers()
            beginPlaybackIfNeeded()
        } catch {
            await playbackAdmission.release(reservation)
            throw error
        }
    }

    public func finishStream() {
        finishedStreaming = true
        updateTerminalStateIfNeeded()
    }

    public func stop() async {
        playbackSessionID &+= 1
        playerNode?.stop()
        if engine.isRunning {
            engine.stop()
        }

        resetPlaybackState(didFinishPlayback: false)
        await playbackAdmission.reset()
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

    nonisolated static func queueDrainDisposition(
        finishedStreaming: Bool,
        totalEnqueuedFrames: Int64
    ) -> QueueDrainDisposition {
        if finishedStreaming {
            return .finishPlayback(didPlayAudio: totalEnqueuedFrames > 0)
        }
        return .awaitMoreAudio
    }

    private nonisolated static func makeDefaultPlaybackAdmission(
        limits: AudioPlaybackQueueLimits
    )
        -> AudioPlaybackQueueAdmissionController
    {
        do {
            return try AudioPlaybackQueueAdmissionController(limits: limits)
        } catch {
            preconditionFailure("Invalid library-owned interactive playback limits: \(error)")
        }
    }
}

extension AudioEnginePlayer {
    private func maximumPlaybackChunkFrameCount(
        for format: AudioFormatDescriptor
    ) throws -> Int {
        guard format.channelCount > 0,
              AVAudioChannelCount(exactly: format.channelCount) != nil,
              format.sampleRate.isFinite,
              format.sampleRate > 0 else {
            throw AudioEnginePlayerError.invalidFormat(
                sampleRate: format.sampleRate,
                channelCount: format.channelCount
            )
        }

        let (bytesPerFrame, byteWidthOverflow) = format.channelCount
            .multipliedReportingOverflow(by: MemoryLayout<Float>.stride)
        guard !byteWidthOverflow, bytesPerFrame > 0 else {
            throw AudioEnginePlayerError.invalidFormat(
                sampleRate: format.sampleRate,
                channelCount: format.channelCount
            )
        }
        let framesByBytes = playbackQueueLimits.maximumScheduledBytes
            / Int64(bytesPerFrame)

        let durationFrameCount = (
            playbackQueueLimits.highWaterDuration * format.sampleRate
        ).rounded(.down)
        guard durationFrameCount.isFinite, durationFrameCount >= 1 else {
            throw AudioEnginePlayerError.invalidFormat(
                sampleRate: format.sampleRate,
                channelCount: format.channelCount
            )
        }
        let framesByDuration = durationFrameCount >= Double(Int64.max)
            ? Int64.max
            : Int64(durationFrameCount)
        let frameCount = min(
            framesByBytes,
            framesByDuration,
            Int64(Int.max)
        )
        guard frameCount > 0 else {
            throw AudioEnginePlayerError.invalidFormat(
                sampleRate: format.sampleRate,
                channelCount: format.channelCount
            )
        }
        return Int(frameCount)
    }

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
            let reservation = queuedBuffer.reservation
            let sessionID = playbackSessionID

            playerNode.scheduleBuffer(
                queuedBuffer.buffer,
                completionCallbackType: .dataPlayedBack
            ) { [self] _ in
                Task {
                    await markBufferPlayedBack(
                        frameCount: frameCount,
                        reservation: reservation,
                        sessionID: sessionID
                    )
                }
            }
        }
    }

    private func beginPlaybackIfNeeded() {
        guard let playerNode else { return }
        guard scheduledBufferCount > 0 else { return }
        guard !playerNode.isPlaying else {
            if isBuffering {
                // The player clock continues through an underrun. Rebase it so
                // silent wait time is not counted as newly played audio.
                playbackBaseFrames = playedFrames
                playbackNodeBaseSampleTime = currentNodeSampleTime() ?? 0
            }
            isPlaying = true
            isBuffering = false
            return
        }

        playbackBaseFrames = playedFrames
        playerNode.play()
        playbackNodeBaseSampleTime = currentNodeSampleTime() ?? 0
        isPlaying = true
        isBuffering = false
    }

    private func markBufferPlayedBack(
        frameCount: Int64,
        reservation: AudioPlaybackQueueReservation,
        sessionID: UInt64
    ) async {
        await playbackAdmission.release(reservation)
        guard sessionID == playbackSessionID else { return }
        playedFrames += frameCount
        scheduledBufferCount = max(scheduledBufferCount - 1, 0)
        updateTerminalStateIfNeeded()
    }

    private func updateTerminalStateIfNeeded() {
        guard let playerNode else { return }
        guard scheduledBufferCount == 0, pendingBuffers.isEmpty else { return }

        switch Self.queueDrainDisposition(
            finishedStreaming: finishedStreaming,
            totalEnqueuedFrames: totalEnqueuedFrames
        ) {
        case .finishPlayback(let didPlayAudio):
            playerNode.stop()
            if engine.isRunning {
                engine.stop()
            }
            resetPlaybackState(didFinishPlayback: didPlayAudio)
        case .awaitMoreAudio:
            // Keep the engine and node running across a temporary underrun.
            // Scheduling the next buffer then resumes without a stop/start cycle.
            playbackBaseFrames = playedFrames
            playbackNodeBaseSampleTime = currentNodeSampleTime() ?? 0
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
        playbackNodeBaseSampleTime = 0
        scheduledBufferCount = 0
        finishedStreaming = false
        isPlaying = false
        isBuffering = false
        self.didFinishPlayback = didFinishPlayback
    }

    private func currentPositionFrames() -> Int64 {
        guard let playerNode else { return playedFrames }
        var positionFrames = playedFrames

        if playerNode.isPlaying, let nodeSampleTime = currentNodeSampleTime() {
            let liveFrames = max(nodeSampleTime - playbackNodeBaseSampleTime, 0)
            positionFrames = max(positionFrames, playbackBaseFrames + liveFrames)
        }

        return min(positionFrames, totalEnqueuedFrames)
    }

    private func currentNodeSampleTime() -> Int64? {
        guard let playerNode,
              let renderTime = playerNode.lastRenderTime,
              let playerTime = playerNode.playerTime(forNodeTime: renderTime) else {
            return nil
        }
        return Int64(playerTime.sampleTime)
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
        guard format.sampleRate.isFinite,
              format.sampleRate > 0,
              let channelCount = AVAudioChannelCount(exactly: format.channelCount),
              channelCount > 0,
              let avFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: format.sampleRate,
            channels: channelCount,
            // AudioPCMBuffer stores one Float array per channel regardless of
            // the source container layout. Keep the AVFoundation buffer planar
            // so PCMCoding can copy every channel through floatChannelData.
            interleaved: false
        ) else {
            throw AudioEnginePlayerError.invalidFormat(
                sampleRate: format.sampleRate,
                channelCount: format.channelCount
            )
        }

        return avFormat
    }

    private static func validatePlaybackBufferShape(
        _ buffer: AudioPCMBuffer
    ) throws {
        guard buffer.channels.count == buffer.format.channelCount,
              AVAudioChannelCount(exactly: buffer.format.channelCount) != nil,
              buffer.format.sampleRate.isFinite,
              buffer.format.sampleRate > 0 else {
            throw AudioEnginePlayerError.invalidFormat(
                sampleRate: buffer.format.sampleRate,
                channelCount: buffer.format.channelCount
            )
        }
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
    let reservation: AudioPlaybackQueueReservation
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
