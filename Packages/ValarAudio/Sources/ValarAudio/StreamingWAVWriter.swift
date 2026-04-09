import AVFoundation
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
    /// Intermediate directories are created if needed. Any existing file at `url`
    /// is removed before opening.
    ///
    /// - Parameters:
    ///   - url: Destination path for the WAV file.
    ///   - sampleRate: Output sample rate in Hz (e.g., 24_000).
    ///   - channelCount: Number of audio channels (1 = mono, 2 = stereo).
    /// - Throws: `AudioPipelineError.exportFailed` if the format is invalid or the
    ///   file cannot be opened.
    public init(url: URL, sampleRate: Double, channelCount: Int) throws {
        guard channelCount > 0, sampleRate > 0 else {
            throw AudioPipelineError.exportFailed(
                "Invalid format: channelCount=\(channelCount) sampleRate=\(sampleRate)"
            )
        }

        // Create intermediate directories if needed.
        let directory = url.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        // Remove any stale file at the target location.
        if FileManager.default.fileExists(atPath: url.path) {
            do {
                try FileManager.default.removeItem(at: url)
            } catch let error as CocoaError where error.code == .fileNoSuchFile {
                // Another process may have removed it concurrently — ignore.
                _ = error
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

        let pcmBuffer = try makePCMBuffer(frameCount: frameCount)
        deinterleave(samples: samples, frameCount: frameCount, channelCount: channelCount, into: pcmBuffer)
        try file!.write(from: pcmBuffer)
        totalFramesWritten += frameCount
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
        let pcmBuffer = try makePCMBuffer(frameCount: frameCount)
        PCMCoding.fill(pcmBuffer, from: buffer.channels, frameCount: frameCount)
        try file!.write(from: pcmBuffer)
        totalFramesWritten += frameCount
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
        guard let pcmBuffer = AVAudioPCMBuffer(
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

    /// De-interleaves interleaved `samples` into the per-channel pointers of `pcmBuffer`.
    ///
    /// For mono, this is a single contiguous copy. For multi-channel, each channel
    /// is extracted from the interleaved layout.
    private func deinterleave(
        samples: [Float],
        frameCount: Int,
        channelCount: Int,
        into pcmBuffer: AVAudioPCMBuffer
    ) {
        if channelCount == 1 {
            guard let dst = pcmBuffer.floatChannelData?[0] else { return }
            samples.withUnsafeBufferPointer { src in
                if let base = src.baseAddress {
                    dst.update(from: base, count: frameCount)
                }
            }
        } else {
            for ch in 0..<channelCount {
                guard let dst = pcmBuffer.floatChannelData?[ch] else { continue }
                for frame in 0..<frameCount {
                    dst[frame] = samples[frame * channelCount + ch]
                }
            }
        }
    }
}
