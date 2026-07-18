import AVFoundation

public struct AVFoundationAudioDecoder: AudioDecoder {
    private static let maximumTemporaryFileSize = 128 * 1_024 * 1_024
    static let maximumDecodedPCMBytes = 256 * 1_024 * 1_024
    private static let framesPerDecodeRead: AVAudioFrameCount = 16_384

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

        return try decodeAudioFile(at: tempURL, hint: hint)
    }

    public func decode(fileAt url: URL, hint: String?) async throws -> AudioPCMBuffer {
        let resourceValues = try url.resourceValues(
            forKeys: [.fileSizeKey, .isDirectoryKey, .isRegularFileKey]
        )
        guard resourceValues.isDirectory != true,
              resourceValues.isRegularFile == true else {
            throw AudioPipelineError.decodingFailed("Audio source must be a regular file")
        }
        if let fileSize = resourceValues.fileSize,
           fileSize > Self.maximumTemporaryFileSize {
            throw AudioDecoderError.fileTooLarge(
                size: fileSize,
                maximumSize: Self.maximumTemporaryFileSize
            )
        }
        if hint?.lowercased().hasPrefix("pcm_f32le") == true {
            if let fileSize = resourceValues.fileSize,
               fileSize > Self.maximumDecodedPCMBytes {
                throw AudioDecoderError.decodedAudioTooLarge(
                    size: fileSize,
                    maximumSize: Self.maximumDecodedPCMBytes
                )
            }
            return try await decode(
                Data(contentsOf: url, options: .mappedIfSafe),
                hint: hint
            )
        }

        return try decodeAudioFile(at: url, hint: hint)
    }

    private func decodeAudioFile(at url: URL, hint: String?) throws -> AudioPCMBuffer {
        let file = try AVAudioFile(forReading: url)
        let format = file.processingFormat
        let channelCount = Int(format.channelCount)
        guard format.sampleRate.isFinite,
              format.sampleRate >= 1_000,
              format.sampleRate <= 768_000,
              (1...64).contains(channelCount) else {
            throw AudioPipelineError.decodingFailed("Decoded audio format is outside supported bounds")
        }
        guard file.length >= 0, file.length <= Int64(AVAudioFrameCount.max) else {
            throw AudioPipelineError.decodingFailed("Frame count \(file.length) exceeds AVAudioPCMBuffer capacity")
        }
        _ = try Self.validatedDecodedPCMByteCount(
            frameCount: file.length,
            channelCount: channelCount
        )
        let frameCount = Int(file.length)
        var channels = (0..<channelCount).map { _ in
            [Float](repeating: 0, count: frameCount)
        }
        guard frameCount > 0 else {
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
        let decodeCapacity = min(
            Self.framesPerDecodeRead,
            AVAudioFrameCount(frameCount)
        )
        guard let pcmBuffer = AVAudioPCMBuffer(
            pcmFormat: format,
            frameCapacity: decodeCapacity
        ) else {
            throw AudioPipelineError.decodingFailed("Could not allocate PCM buffer")
        }
        var destinationOffset = 0
        while destinationOffset < frameCount {
            let requestedFrames = AVAudioFrameCount(
                min(
                    Int(Self.framesPerDecodeRead),
                    frameCount - destinationOffset
                )
            )
            try file.read(into: pcmBuffer, frameCount: requestedFrames)
            let framesRead = Int(pcmBuffer.frameLength)
            guard framesRead > 0 else { break }
            for channelIndex in 0..<channelCount {
                guard let source = pcmBuffer.floatChannelData?[channelIndex] else {
                    throw AudioPipelineError.decodingFailed(
                        "Decoded audio did not expose Float32 channel data"
                    )
                }
                channels[channelIndex].withUnsafeMutableBufferPointer { destination in
                    destination.baseAddress?.advanced(by: destinationOffset).update(
                        from: source,
                        count: framesRead
                    )
                }
            }
            destinationOffset += framesRead
        }
        guard destinationOffset == frameCount else {
            throw AudioPipelineError.decodingFailed(
                "Audio file ended after \(destinationOffset) of \(frameCount) declared frames"
            )
        }

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

        let (sampleRate, channelCount) = try parsePCMHint(hint)
        let totalSamples = data.count / MemoryLayout<Float>.size
        guard totalSamples.isMultiple(of: channelCount) else {
            throw AudioPipelineError.decodingFailed(
                "pcm_f32le sample count \(totalSamples) is not divisible by channel count \(channelCount)"
            )
        }
        _ = try Self.validatedDecodedPCMByteCount(
            frameCount: Int64(totalSamples / channelCount),
            channelCount: channelCount
        )

        let frameCount = totalSamples / channelCount
        var channels = (0..<channelCount).map { _ in
            [Float](repeating: 0, count: frameCount)
        }
        data.withUnsafeBytes { rawBuffer in
            let source = rawBuffer.bindMemory(to: Float.self)
            if channelCount == 1 {
                channels[0].withUnsafeMutableBufferPointer { destination in
                    guard let sourceBase = source.baseAddress,
                          let destinationBase = destination.baseAddress else { return }
                    destinationBase.update(from: sourceBase, count: frameCount)
                }
            } else {
                for channelIndex in 0..<channelCount {
                    for frameIndex in 0..<frameCount {
                        channels[channelIndex][frameIndex] =
                            source[(frameIndex * channelCount) + channelIndex]
                    }
                }
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
    private func parsePCMHint(
        _ hint: String
    ) throws -> (sampleRate: Double, channelCount: Int) {
        let parts = hint.lowercased().split(separator: ":", maxSplits: 2)
        let sampleRate: Double
        if parts.count > 1 {
            guard let parsed = Double(parts[1]),
                  parsed.isFinite,
                  parsed >= 1_000,
                  parsed <= 768_000 else {
                throw AudioPipelineError.decodingFailed(
                    "pcm_f32le sample rate must be finite and between 1000 and 768000 Hz"
                )
            }
            sampleRate = parsed
        } else {
            sampleRate = 24_000
        }
        let channelCount: Int
        if parts.count > 2 {
            guard let parsed = Int(parts[2]), parsed >= 1, parsed <= 64 else {
                throw AudioPipelineError.decodingFailed(
                    "pcm_f32le channel count must be between 1 and 64"
                )
            }
            channelCount = parsed
        } else {
            channelCount = 1
        }
        return (sampleRate, channelCount)
    }

    static func validatedDecodedPCMByteCount(
        frameCount: Int64,
        channelCount: Int
    ) throws -> Int {
        guard frameCount >= 0,
              UInt64(frameCount) <= UInt64(Int.max),
              (1...64).contains(channelCount) else {
            throw AudioDecoderError.decodedAudioTooLarge(
                size: nil,
                maximumSize: maximumDecodedPCMBytes
            )
        }
        let (sampleCount, sampleOverflowed) = Int(frameCount)
            .multipliedReportingOverflow(by: channelCount)
        let (byteCount, byteOverflowed) = sampleCount
            .multipliedReportingOverflow(by: MemoryLayout<Float>.size)
        guard !sampleOverflowed,
              !byteOverflowed,
              byteCount <= maximumDecodedPCMBytes else {
            throw AudioDecoderError.decodedAudioTooLarge(
                size: byteOverflowed || sampleOverflowed ? nil : byteCount,
                maximumSize: maximumDecodedPCMBytes
            )
        }
        return byteCount
    }
}

private enum AudioDecoderError: Error, LocalizedError, Sendable {
    case fileTooLarge(size: Int, maximumSize: Int)
    case decodedAudioTooLarge(size: Int?, maximumSize: Int)

    var errorDescription: String? {
        switch self {
        case let .fileTooLarge(size, maximumSize):
            return "Audio payload size \(size) bytes exceeds temporary decode limit of \(maximumSize) bytes"
        case let .decodedAudioTooLarge(size, maximumSize):
            let observed = size.map { "\($0) bytes" } ?? "an unrepresentable size"
            return "Decoded PCM would require \(observed), exceeding the \(maximumSize)-byte allocation limit"
        }
    }
}
