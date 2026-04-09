import Accelerate
import Foundation
import SwiftOGG
import ValarAudio

/// Converts a ValarAudio PCM buffer to an in-memory OGG/Opus bitstream for
/// channel delivery (Telegram voice messages, WhatsApp PTT, Matrix Element, etc.).
///
/// Wraps `OGGEncoder` (element-hq/swift-ogg 0.0.3) in an actor to satisfy
/// Swift 6 strict concurrency. Performs Float32 → Int16 conversion internally
/// using Accelerate `vDSP_vclip` + `vDSP_vsmul` + `vDSP_vfix16`.
///
/// ## Constraints
/// - Only mono (1) and stereo (2) channels are supported (Opus protocol).
/// - Sample rate must be one of: 8000, 12000, 16000, 24000, 48000 Hz.
/// - PCM rate and Opus rate must match (resampling is the caller's responsibility).
///
/// ## EOS note
/// `OGGEncoder.endstream()` is `internal` in swift-ogg 0.0.3. Input is
/// zero-padded to a complete Opus frame boundary before encoding so no audio
/// is lost. After flushing, the `e_o_s` flag is patched onto the last OGG page
/// header (byte offset 5, bit 2) to signal stream end. This prevents audio
/// truncation in players that rely on EOS (Telegram, WhatsApp).
public actor ChannelAudioExporter {

    /// Errors thrown by `encode(_:)`.
    public enum ExportError: Error, Sendable {
        case unsupportedSampleRate(Double)
        case unsupportedChannelCount(Int)
        case encodingFailed(Error)
    }

    private static let validRates: Set<Int32> = [8_000, 12_000, 16_000, 24_000, 48_000]

    public init() {}

    /// Encodes `buffer` to an in-memory OGG/Opus bitstream and returns it as `Data`.
    ///
    /// - Parameter buffer: Source PCM audio. Must be mono or stereo with a
    ///   sample rate from `{8000, 12000, 16000, 24000, 48000}`.
    /// - Returns: A complete OGG/Opus bitstream beginning with the Opus headers.
    /// - Throws: `ExportError` on validation failure, `OggError`/`OpusError` on
    ///   encoding failure.
    public func encode(_ buffer: AudioPCMBuffer) throws -> Data {
        let rate = Int32(buffer.format.sampleRate)
        let channels = Int32(buffer.format.channelCount)

        guard Self.validRates.contains(rate) else {
            throw ExportError.unsupportedSampleRate(buffer.format.sampleRate)
        }
        guard channels == 1 || channels == 2 else {
            throw ExportError.unsupportedChannelCount(buffer.format.channelCount)
        }

        let pcmBytesPerFrame = UInt32(channels) * UInt32(MemoryLayout<Int16>.size)

        let encoder = try OGGEncoder(
            pcmRate: rate,
            pcmChannels: channels,
            pcmBytesPerFrame: pcmBytesPerFrame,
            opusRate: rate,
            application: .audio
        )

        // Convert Float32 channels → interleaved Int16, clipped and frame-padded.
        let int16Data = frameAligned(toInt16Data(buffer), rate: rate, channels: Int(channels))
        guard !int16Data.isEmpty else {
            return encoder.bitstream(flush: true)
        }

        do {
            try encoder.encode(pcm: int16Data)
        } catch {
            throw ExportError.encodingFailed(error)
        }

        // endstream() is internal in swift-ogg 0.0.3; bitstream(flush: true) drains
        // all accumulated OGG pages. Because input was padded to a frame boundary
        // above, OGGEncoder's internal pcmCache is empty after encode(pcm:).
        var oggData = encoder.bitstream(flush: true)

        // Patch the EOS flag on the last OGG page. OGGEncoder can't set e_o_s = 1
        // because endstream() is internal. Some players (Telegram, WhatsApp) truncate
        // audio when the final page lacks the EOS flag.
        Self.setEOSOnLastPage(&oggData)

        return oggData
    }

    // MARK: - Private

    /// Sets the `e_o_s` (end of stream) flag on the last OGG page in `data`.
    ///
    /// OGG page header format: bytes 0-3 = "OggS", byte 5 = header type flag.
    /// Bit 2 of the header type flag (0x04) indicates end of stream.
    /// We scan backwards from the end to find the last "OggS" capture pattern,
    /// then set bit 2 on its header type byte.
    private static func setEOSOnLastPage(_ data: inout Data) {
        let pattern: [UInt8] = [0x4F, 0x67, 0x67, 0x53] // "OggS"
        let count = data.count
        guard count >= 27 else { return } // minimum OGG page size

        // Scan backwards for the last "OggS" sync pattern
        for offset in stride(from: count - 27, through: 0, by: -1) {
            if data[offset] == pattern[0],
               offset + 5 < count,
               data[offset + 1] == pattern[1],
               data[offset + 2] == pattern[2],
               data[offset + 3] == pattern[3] {
                // Found last page header — set EOS flag (bit 2 of byte 5)
                data[offset + 5] |= 0x04
                return
            }
        }
    }

    /// Converts `AudioPCMBuffer` (non-interleaved Float32) to interleaved Int16 `Data`.
    ///
    /// Clips each sample to [–1, 1] before scaling to prevent Int16 overflow wrap-around.
    private func toInt16Data(_ buffer: AudioPCMBuffer) -> Data {
        let frameCount = buffer.frameCount
        let channelCount = buffer.format.channelCount
        let (sampleCount, overflow) = frameCount.multipliedReportingOverflow(by: channelCount)
        guard !overflow, sampleCount > 0 else { return Data() }

        // Interleave channels into a single contiguous Float array.
        var floatInterleaved = [Float](repeating: 0, count: sampleCount)
        for ch in 0..<channelCount {
            guard ch < buffer.channels.count else { continue }
            let src = buffer.channels[ch]
            let available = min(frameCount, src.count)
            for f in 0..<available {
                floatInterleaved[f * channelCount + ch] = src[f]
            }
        }

        // Clip to [–1, 1] to prevent overflow when scaling to Int16 range.
        var lo = Float(-1)
        var hi = Float(1)
        var clipped = [Float](repeating: 0, count: sampleCount)
        vDSP_vclip(floatInterleaved, 1, &lo, &hi, &clipped, 1, vDSP_Length(sampleCount))

        // Scale to Int16 range.
        var scale = Float(Int16.max)
        var scaled = [Float](repeating: 0, count: sampleCount)
        vDSP_vsmul(clipped, 1, &scale, &scaled, 1, vDSP_Length(sampleCount))

        // Convert to Int16 (truncating).
        var result = [Int16](repeating: 0, count: sampleCount)
        vDSP_vfix16(&scaled, 1, &result, 1, vDSP_Length(sampleCount))

        return Data(bytes: result, count: result.count * MemoryLayout<Int16>.size)
    }

    /// Zero-pads `data` so its length is an exact multiple of one Opus frame.
    ///
    /// OGGEncoder caches any trailing PCM that doesn't fill a complete 20 ms frame.
    /// Without `endstream()` (internal in swift-ogg 0.0.3), that cached remainder
    /// would be silently dropped. Pre-padding ensures all source audio is encoded.
    ///
    /// Frame size in samples: `960 / (48000 / rate)` — e.g. 480 at 24 kHz.
    private func frameAligned(_ data: Data, rate: Int32, channels: Int) -> Data {
        guard !data.isEmpty else { return data }

        let samplesPerFrame = 960 / (48_000 / Int(rate))
        let bytesPerOpusFrame = samplesPerFrame * channels * MemoryLayout<Int16>.size
        guard bytesPerOpusFrame > 0 else { return data }

        let remainder = data.count % bytesPerOpusFrame
        guard remainder != 0 else { return data }

        return data + Data(count: bytesPerOpusFrame - remainder)
    }
}
