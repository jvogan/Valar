import AVFoundation

public struct AVFoundationAudioDecoder: AudioDecoder {
    private static let maximumTemporaryFileSize = 500_000_000

    public init() {}

    public func decode(_ data: Data, hint: String?) async throws -> AudioPCMBuffer {
        // Fast path: raw pcm_f32le — reinterpret bytes directly, no temp file needed.
        // Hint format: "pcm_f32le[:<sampleRate>[:<channelCount>]]"
        // Example: "pcm_f32le:24000:1"
        if hint?.lowercased().hasPrefix("pcm_f32le") == true {
            return try decodePCMFloat32LE(data, hint: hint!)
        }

        guard data.count <= Self.maximumTemporaryFileSize else {
            throw AudioDecoderError.fileTooLarge(
                size: data.count,
                maximumSize: Self.maximumTemporaryFileSize
            )
        }

        let tempURL = try TemporaryAudioFileSecurity.makeURL(fileExtension: hint ?? "wav")
        try data.write(to: tempURL)
        defer { try? FileManager.default.removeItem(at: tempURL) }

        let file = try AVAudioFile(forReading: tempURL)
        let format = file.processingFormat
        guard file.length >= 0, file.length <= Int64(AVAudioFrameCount.max) else {
            throw AudioPipelineError.decodingFailed("Frame count \(file.length) exceeds AVAudioPCMBuffer capacity")
        }
        let frameCount = AVAudioFrameCount(file.length)
        guard let pcmBuffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
            throw AudioPipelineError.decodingFailed("Could not allocate PCM buffer")
        }
        try file.read(into: pcmBuffer)

        let channelCount = Int(format.channelCount)
        let channels = PCMCoding.channels(from: pcmBuffer)

        return AudioPCMBuffer(
            channels: channels,
            format: AudioFormatDescriptor(
                sampleRate: format.sampleRate,
                channelCount: channelCount,
                sampleFormat: .float32,
                interleaved: false,
                container: hint ?? "wav"
            )
        )
    }

    // MARK: - PCM fast path

    /// Decodes raw pcm_f32le bytes into an `AudioPCMBuffer` without any disk I/O.
    ///
    /// The hint carries optional metadata after the format tag:
    ///   `pcm_f32le[:<sampleRate>[:<channelCount>]]`
    ///
    /// Sample rate defaults to 24 000 Hz and channel count to 1 (mono) when omitted.
    /// All current Apple platforms store floats in native little-endian order, so the
    /// raw bytes can be reinterpreted directly — no per-sample conversion required.
    private func decodePCMFloat32LE(_ data: Data, hint: String) throws -> AudioPCMBuffer {
        guard data.count.isMultiple(of: MemoryLayout<Float>.size) else {
            throw AudioPipelineError.decodingFailed(
                "pcm_f32le byte count \(data.count) is not a multiple of \(MemoryLayout<Float>.size)"
            )
        }

        let (sampleRate, channelCount) = parsePCMHint(hint)
        let totalSamples = data.count / MemoryLayout<Float>.size

        // Bulk reinterpret: on LE platforms this is a single memcpy with no conversion.
        let flat: [Float] = data.withUnsafeBytes { rawBuffer in
            Array(rawBuffer.bindMemory(to: Float.self))
        }

        let channels: [[Float]]
        if channelCount == 1 {
            channels = [flat]
        } else {
            channels = (0..<channelCount).map { ch in
                stride(from: ch, to: totalSamples, by: channelCount).map { flat[$0] }
            }
        }

        return AudioPCMBuffer(
            channels: channels,
            format: AudioFormatDescriptor(
                sampleRate: sampleRate,
                channelCount: channelCount,
                sampleFormat: .float32,
                interleaved: false,
                container: "pcm"
            )
        )
    }

    /// Parses `sampleRate` and `channelCount` from a pcm_f32le hint string.
    ///
    /// Accepted forms:
    /// - `"pcm_f32le"` → (24 000, 1)
    /// - `"pcm_f32le:24000"` → (24 000, 1)
    /// - `"pcm_f32le:24000:2"` → (24 000, 2)
    private func parsePCMHint(_ hint: String) -> (sampleRate: Double, channelCount: Int) {
        let parts = hint.lowercased().split(separator: ":", maxSplits: 2)
        let sampleRate = parts.count > 1 ? Double(parts[1]) ?? 24_000 : 24_000
        let channelCount = parts.count > 2 ? max(1, Int(parts[2]) ?? 1) : 1
        return (sampleRate, channelCount)
    }
}

private enum AudioDecoderError: Error, LocalizedError, Sendable {
    case fileTooLarge(size: Int, maximumSize: Int)

    var errorDescription: String? {
        switch self {
        case let .fileTooLarge(size, maximumSize):
            return "Audio payload size \(size) bytes exceeds temporary decode limit of \(maximumSize) bytes"
        }
    }
}
