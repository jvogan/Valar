import Foundation
import SwiftOGG
import ValarAudio

/// Decodes inbound OGG/Opus audio (e.g. Telegram voice messages) to an
/// `AudioPCMBuffer` without AVFoundation, which does not support the OGG container.
///
/// Wraps `OGGDecoder` from element-hq/swift-ogg 0.0.3. The decoder outputs
/// interleaved Float32 PCM at the sample rate declared by the Opus stream
/// (always 48 000 Hz for standard Opus/Telegram voice).
public struct ChannelAudioImporter: Sendable {
    public init() {}

    /// Decodes OGG/Opus `data` to a non-interleaved `AudioPCMBuffer`.
    ///
    /// - Parameter data: Raw OGG/Opus bytes (e.g. from a `.ogg` or `.oga` file).
    /// - Returns: PCM buffer at the rate declared by the stream (typically 48 000 Hz).
    /// - Throws: `OpusError` / `OggError` from SwiftOGG on malformed input.
    public func decode(_ data: Data) throws -> AudioPCMBuffer {
        let oggDecoder = try OGGDecoder(audioData: data)
        let pcmData = oggDecoder.pcmData

        // OGGDecoder.sampleRate and .numChannels are internal in swift-ogg 0.0.3.
        // Opus always decodes to 48 000 Hz; Telegram voice messages are always mono.
        let sampleRate: Double = 48_000
        let samples: [Float] = pcmData.withUnsafeBytes { rawBuffer in
            Array(rawBuffer.bindMemory(to: Float.self))
        }

        return AudioPCMBuffer(mono: samples, sampleRate: sampleRate, container: "ogg")
    }
}
