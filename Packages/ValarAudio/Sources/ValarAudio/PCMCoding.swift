import AVFoundation

/// Shared utilities for converting between ValarAudio's `[[Float]]` channel representation
/// and AVFoundation's `AVAudioPCMBuffer`.
///
/// Three call sites previously duplicated the same channel-copy loops:
/// `AVFoundationAudioDecoder`, `AVFoundationAudioExporter`, and `AudioEnginePlayer`.
/// This type centralises the two directions so each site is a single call.
enum PCMCoding {

    /// Reads all float channels out of an `AVAudioPCMBuffer` into `[[Float]]`.
    ///
    /// Returns one `[Float]` per channel, each of length `avBuffer.frameLength`.
    /// When `frameLength` is zero, `floatChannelData` may be `nil`; in that case
    /// each channel is represented as an empty array so the channel count is preserved.
    static func channels(from avBuffer: AVAudioPCMBuffer) -> [[Float]] {
        let channelCount = Int(avBuffer.format.channelCount)
        let frameCount = Int(avBuffer.frameLength)
        var result: [[Float]] = []
        result.reserveCapacity(channelCount)
        for ch in 0..<channelCount {
            if let ptr = avBuffer.floatChannelData?[ch] {
                result.append(Array(UnsafeBufferPointer(start: ptr, count: frameCount)))
            } else {
                // floatChannelData is nil when frameLength == 0 (no backing storage).
                result.append([])
            }
        }
        return result
    }

    /// Writes `channels` into `avBuffer`, zero-padding any channel that is shorter than `frameCount`.
    ///
    /// The number of channels written is clamped to `avBuffer.format.channelCount`.
    /// Channels present in `avBuffer` but absent from `channels` are zero-filled entirely.
    static func fill(
        _ avBuffer: AVAudioPCMBuffer,
        from channels: [[Float]],
        startFrame: Int = 0,
        frameCount: Int
    ) {
        precondition(
            frameCount == Int(avBuffer.frameLength),
            "PCMCoding.fill frameCount must match avBuffer.frameLength"
        )
        precondition(startFrame >= 0, "PCMCoding.fill startFrame must be nonnegative")

        let channelCount = Int(avBuffer.format.channelCount)
        for channelIndex in 0..<channelCount {
            guard let destination = avBuffer.floatChannelData?[channelIndex] else { continue }
            let source = channelIndex < channels.count ? channels[channelIndex] : []
            let availableCount = min(frameCount, max(0, source.count - startFrame))
            source.withUnsafeBufferPointer { ptr in
                if let baseAddress = ptr.baseAddress, availableCount > 0 {
                    destination.update(
                        from: baseAddress.advanced(by: startFrame),
                        count: availableCount
                    )
                }
            }
            if availableCount < frameCount {
                destination.advanced(by: availableCount)
                    .initialize(repeating: 0, count: frameCount - availableCount)
            }
        }
    }
}
