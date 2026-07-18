import AVFoundation
import Darwin
import Foundation

/// A streaming WAV writer that accepts audio sample chunks incrementally.
///
/// Unlike `AVFoundationAudioExporter`, which requires a complete `AudioPCMBuffer`
/// upfront, `StreamingWAVWriter` opens a WAV file at initialization and accepts
/// incremental sample chunks via `append`. This is suited for streaming TTS
/// scenarios where audio is generated chunk by chunk from an `AsyncThrowingStream`.
///
/// `AVAudioFile` automatically updates the RIFF chunk sizes when the file handle
/// is closed. `finalize()` triggers that by releasing the file object.
///
/// ## Usage
/// ```swift
/// let writer = try StreamingWAVWriter(url: outputURL, sampleRate: 24_000, channelCount: 1)
///
/// for await chunk in audioStream {
///     try await writer.append(chunk.samples)
/// }
///
/// let exported = await writer.finalize()
/// ```
///
/// ## Thread safety
/// `StreamingWAVWriter` is an actor. All append and finalize calls are actor-isolated
/// and safe to call from concurrent async contexts.
public actor StreamingWAVWriter {
    private static let framesPerWrite = 16_384

    // MARK: - State

    private let url: URL
    private let descriptor: AudioFormatDescriptor
    private var file: AVAudioFile?
    private var totalFramesWritten: Int = 0
    private var isFinalized: Bool = false

    // MARK: - Public properties

    /// The total number of audio frames written so far.
    public var framesWritten: Int { totalFramesWritten }

    /// The format descriptor describing the output WAV file.
    public var format: AudioFormatDescriptor { descriptor }

    // MARK: - Initializer

    /// Opens a new WAV file at `url` for streaming writes.
    ///
    /// Intermediate directories are created if needed. An existing regular file
    /// at `url` is replaced; symbolic links and other entry types are rejected.
    ///
    /// - Parameters:
    ///   - url: Destination path for the WAV file.
    ///   - sampleRate: Output sample rate in Hz (e.g., 24_000).
    ///   - channelCount: Number of audio channels (1 = mono, 2 = stereo).
    /// - Throws: `AudioPipelineError.exportFailed` if the format is invalid or the
    ///   file cannot be opened.
    public init(url: URL, sampleRate: Double, channelCount: Int) throws {
        guard (1...64).contains(channelCount),
              sampleRate.isFinite,
              (1_000...768_000).contains(sampleRate) else {
            throw AudioPipelineError.exportFailed(
                "Invalid format: channelCount=\(channelCount) sampleRate=\(sampleRate)"
            )
        }
        guard url.isFileURL, !url.lastPathComponent.isEmpty else {
            throw AudioPipelineError.exportFailed(
                "Streaming WAV output must be a local file URL."
            )
        }

        // Create intermediate directories if needed.
        let url = url.standardizedFileURL
        let directory = url.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Self.requireRealDirectory(at: directory)

        // AVAudioFile writes directly to its destination. Inspect the directory
        // entry without following symbolic links before replacing anything.
        if try Self.destinationEntryExists(at: url) {
            // `unlink` cannot recursively remove a directory if the entry is
            // swapped after lstat. It also removes a swapped link itself rather
            // than following it.
            if Darwin.unlink(url.path) != 0 {
                let errorCode = errno
                guard errorCode == ENOENT else {
                    throw AudioPipelineError.exportFailed(
                        "Unable to replace the streaming WAV destination (errno \(errorCode))."
                    )
                }
            }
        }

        let settings: [String: Any] = [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: sampleRate,
            AVNumberOfChannelsKey: channelCount,
            AVLinearPCMBitDepthKey: 32,
            AVLinearPCMIsFloatKey: true,
            AVLinearPCMIsBigEndianKey: false,
            AVLinearPCMIsNonInterleaved: false,
        ]

        let file = try AVAudioFile(
            forWriting: url,
            settings: settings,
            commonFormat: .pcmFormatFloat32,
            interleaved: false
        )

        self.url = url
        self.descriptor = AudioFormatDescriptor(
            sampleRate: sampleRate,
            channelCount: channelCount,
            sampleFormat: .float32,
            interleaved: false,
            container: "wav"
        )
        self.file = file
    }

    // MARK: - Append

    /// Appends raw float32 samples to the WAV file.
    ///
    /// For **mono** audio, `samples` contains sequential frame values.
    /// For **multi-channel** audio, `samples` must be interleaved:
    /// `[L0, R0, L1, R1, …]`.
    ///
    /// Incomplete trailing frames (where `samples.count % channelCount != 0`) are
    /// silently dropped.
    ///
    /// - Parameter samples: Float32 audio samples to append.
    /// - Throws: `AudioPipelineError.exportFailed` if the writer has been finalized.
    public func append(_ samples: [Float]) throws {
        guard !samples.isEmpty else { return }
        try checkOpen()

        let channelCount = descriptor.channelCount
        let frameCount = samples.count / channelCount
        guard frameCount > 0 else { return }

        var startFrame = 0
        while startFrame < frameCount {
            let writeFrameCount = min(Self.framesPerWrite, frameCount - startFrame)
            let pcmBuffer = try makePCMBuffer(frameCount: writeFrameCount)
            deinterleave(
                samples: samples,
                startFrame: startFrame,
                frameCount: writeFrameCount,
                channelCount: channelCount,
                into: pcmBuffer
            )
            try file!.write(from: pcmBuffer)
            try addWrittenFrames(writeFrameCount)
            startFrame += writeFrameCount
        }
    }

    /// Appends an `AudioPCMBuffer` to the WAV file.
    ///
    /// Channels present in `buffer` but absent from the writer's channel count are ignored.
    /// Channels absent from `buffer` but present in the writer are zero-filled.
    ///
    /// - Parameter buffer: The buffer to append.
    /// - Throws: `AudioPipelineError.exportFailed` if the writer has been finalized.
    public func append(_ buffer: AudioPCMBuffer) throws {
        guard buffer.frameCount > 0 else { return }
        try checkOpen()

        let frameCount = buffer.frameCount
        guard buffer.channels.count == descriptor.channelCount,
              buffer.channels.allSatisfy({ $0.count == frameCount }),
              buffer.format.sampleRate.isFinite,
              abs(buffer.format.sampleRate - descriptor.sampleRate)
                <= max(0.001, descriptor.sampleRate * 1e-9) else {
            throw AudioPipelineError.exportFailed(
                "Streaming WAV input must match the writer's sample rate and equal-length channel layout."
            )
        }
        var startFrame = 0
        while startFrame < frameCount {
            let writeFrameCount = min(Self.framesPerWrite, frameCount - startFrame)
            let pcmBuffer = try makePCMBuffer(frameCount: writeFrameCount)
            PCMCoding.fill(
                pcmBuffer,
                from: buffer.channels,
                startFrame: startFrame,
                frameCount: writeFrameCount
            )
            try file!.write(from: pcmBuffer)
            try addWrittenFrames(writeFrameCount)
            startFrame += writeFrameCount
        }
    }

    // MARK: - Finalize

    /// Closes the WAV file and returns the `AudioExportedFile`.
    ///
    /// Releasing the `AVAudioFile` triggers its `deinit`, which flushes buffered
    /// samples and writes the final RIFF chunk and `data` sub-chunk sizes into the
    /// WAV header. The result is a well-formed WAV file readable by any standard
    /// audio player or decoder.
    ///
    /// Subsequent calls to `finalize()` are safe and return the same
    /// `AudioExportedFile` without side effects.
    @discardableResult
    public func finalize() -> AudioExportedFile {
        if !isFinalized {
            // Releasing the AVAudioFile triggers deinit, which flushes and writes
            // the final RIFF and data chunk sizes into the WAV header.
            file = nil
            isFinalized = true
        }
        return AudioExportedFile(url: url, format: descriptor)
    }

    // MARK: - Private helpers

    private func checkOpen() throws {
        guard !isFinalized else {
            throw AudioPipelineError.exportFailed(
                "Cannot append to a finalized StreamingWAVWriter"
            )
        }
        guard file != nil else {
            throw AudioPipelineError.exportFailed(
                "StreamingWAVWriter: file handle is not open"
            )
        }
    }

    private func makePCMBuffer(frameCount: Int) throws -> AVAudioPCMBuffer {
        // file is guaranteed non-nil by checkOpen() and actor isolation.
        guard let processingFormat = file?.processingFormat else {
            throw AudioPipelineError.exportFailed(
                "StreamingWAVWriter: file handle is not open"
            )
        }
        guard frameCount > 0,
              frameCount <= Int(AVAudioFrameCount.max),
              let pcmBuffer = AVAudioPCMBuffer(
            pcmFormat: processingFormat,
            frameCapacity: AVAudioFrameCount(frameCount)
        ) else {
            throw AudioPipelineError.exportFailed(
                "Unable to allocate AVAudioPCMBuffer (frameCount=\(frameCount))"
            )
        }
        pcmBuffer.frameLength = AVAudioFrameCount(frameCount)
        return pcmBuffer
    }

    private func addWrittenFrames(_ frameCount: Int) throws {
        let (nextTotal, overflowed) = totalFramesWritten.addingReportingOverflow(
            frameCount
        )
        guard !overflowed else {
            throw AudioPipelineError.exportFailed(
                "Streaming WAV frame count exceeds the supported integer range."
            )
        }
        totalFramesWritten = nextTotal
    }

    /// De-interleaves interleaved `samples` into the per-channel pointers of `pcmBuffer`.
    ///
    /// For mono, this is a single contiguous copy. For multi-channel, each channel
    /// is extracted from the interleaved layout.
    private func deinterleave(
        samples: [Float],
        startFrame: Int,
        frameCount: Int,
        channelCount: Int,
        into pcmBuffer: AVAudioPCMBuffer
    ) {
        if channelCount == 1 {
            guard let dst = pcmBuffer.floatChannelData?[0] else { return }
            samples.withUnsafeBufferPointer { src in
                if let base = src.baseAddress {
                    dst.update(
                        from: base.advanced(by: startFrame),
                        count: frameCount
                    )
                }
            }
        } else {
            for ch in 0..<channelCount {
                guard let dst = pcmBuffer.floatChannelData?[ch] else { continue }
                for frame in 0..<frameCount {
                    dst[frame] = samples[(startFrame + frame) * channelCount + ch]
                }
            }
        }
    }

    private static func requireRealDirectory(at url: URL) throws {
        var metadata = stat()
        guard Darwin.lstat(url.path, &metadata) == 0,
              metadata.st_mode & S_IFMT == S_IFDIR else {
            throw AudioPipelineError.exportFailed(
                "The streaming WAV destination directory must be a real, non-symbolic-link directory."
            )
        }
    }

    /// Returns whether a safe-to-replace regular file already exists.
    ///
    /// `lstat` is intentional: `FileManager.fileExists` follows links and reports
    /// false for dangling links, which could otherwise redirect the writer.
    private static func destinationEntryExists(at url: URL) throws -> Bool {
        var metadata = stat()
        if Darwin.lstat(url.path, &metadata) == 0 {
            guard metadata.st_mode & S_IFMT == S_IFREG else {
                throw AudioPipelineError.exportFailed(
                    "The streaming WAV destination must be a regular, non-symbolic-link file."
                )
            }
            return true
        }

        let errorCode = errno
        guard errorCode == ENOENT else {
            throw AudioPipelineError.exportFailed(
                "Unable to inspect the streaming WAV destination (errno \(errorCode))."
            )
        }
        return false
    }
}
