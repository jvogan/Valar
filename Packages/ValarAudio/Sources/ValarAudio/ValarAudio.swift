import Accelerate
import Foundation

public enum AudioSampleFormat: String, CaseIterable, Codable, Sendable {
    case float32
    case int16
}

public struct AudioFormatDescriptor: Codable, Sendable, Equatable {
    public let sampleRate: Double
    public let channelCount: Int
    public let sampleFormat: AudioSampleFormat
    public let interleaved: Bool
    public let container: String

    public init(
        sampleRate: Double,
        channelCount: Int,
        sampleFormat: AudioSampleFormat = .float32,
        interleaved: Bool = false,
        container: String = "pcm"
    ) {
        self.sampleRate = sampleRate
        self.channelCount = channelCount
        self.sampleFormat = sampleFormat
        self.interleaved = interleaved
        self.container = container
    }
}

public struct AudioPCMBuffer: Codable, Sendable, Equatable {
    public var channels: [[Float]]
    public var format: AudioFormatDescriptor

    public init(channels: [[Float]], format: AudioFormatDescriptor) {
        self.channels = channels
        self.format = format
    }

    public init(mono samples: [Float], sampleRate: Double, container: String = "pcm") {
        self.init(
            channels: [samples],
            format: AudioFormatDescriptor(
                sampleRate: sampleRate,
                channelCount: 1,
                sampleFormat: .float32,
                interleaved: false,
                container: container
            )
        )
    }

    public var frameCount: Int {
        channels.map(\.count).max() ?? 0
    }

    public var flattenedSamples: [Float] {
        channels.flatMap { $0 }
    }

    public var duration: TimeInterval {
        guard format.sampleRate > 0 else { return 0 }
        return TimeInterval(frameCount) / format.sampleRate
    }
}

public struct AudioPlaybackSnapshot: Codable, Sendable, Equatable {
    public let position: TimeInterval
    public let queuedDuration: TimeInterval
    public let isPlaying: Bool
    public let isBuffering: Bool
    public let didFinish: Bool

    public init(
        position: TimeInterval,
        queuedDuration: TimeInterval,
        isPlaying: Bool,
        isBuffering: Bool,
        didFinish: Bool
    ) {
        self.position = position
        self.queuedDuration = queuedDuration
        self.isPlaying = isPlaying
        self.isBuffering = isBuffering
        self.didFinish = didFinish
    }
}

public struct AudioExportedAsset: Codable, Sendable, Equatable {
    public let data: Data
    public let format: AudioFormatDescriptor

    public init(data: Data, format: AudioFormatDescriptor) {
        self.data = data
        self.format = format
    }
}

public struct AudioChapterMarker: Codable, Sendable, Equatable {
    public let title: String
    public let startTime: TimeInterval
    public let duration: TimeInterval

    public init(title: String, startTime: TimeInterval, duration: TimeInterval) {
        self.title = title
        self.startTime = startTime
        self.duration = duration
    }
}

public struct AudioExportedFile: Codable, Sendable, Equatable {
    public let url: URL
    public let format: AudioFormatDescriptor
    public let chapterMarkers: [AudioChapterMarker]

    public init(url: URL, format: AudioFormatDescriptor, chapterMarkers: [AudioChapterMarker] = []) {
        self.url = url
        self.format = format
        self.chapterMarkers = chapterMarkers
    }
}

public struct AudioWaveformSummary: Codable, Sendable, Equatable {
    public let frameCount: Int
    public let peak: Float
    public let rms: Float
    public let bucketCount: Int

    public init(frameCount: Int, peak: Float, rms: Float, bucketCount: Int) {
        self.frameCount = frameCount
        self.peak = peak
        self.rms = rms
        self.bucketCount = bucketCount
    }
}

public protocol AudioDecoder: Sendable {
    func decode(_ data: Data, hint: String?) async throws -> AudioPCMBuffer
}

public protocol AudioResampler: Sendable {
    func resample(_ buffer: AudioPCMBuffer, to sampleRate: Double) async throws -> AudioPCMBuffer
}

public protocol AudioExporter: Sendable {
    func export(_ buffer: AudioPCMBuffer, as format: AudioFormatDescriptor) async throws -> AudioExportedAsset
    func export(
        _ buffer: AudioPCMBuffer,
        as format: AudioFormatDescriptor,
        to destinationURL: URL,
        chapterMarkers: [AudioChapterMarker]
    ) async throws -> AudioExportedFile
}

public enum AudioWaveformAnalyzer {
    public static func summarize(_ buffer: AudioPCMBuffer, bucketCount: Int = 32) -> AudioWaveformSummary {
        let samples = buffer.flattenedSamples
        guard !samples.isEmpty else {
            return AudioWaveformSummary(frameCount: 0, peak: 0, rms: 0, bucketCount: max(bucketCount, 1))
        }

        let peak = vDSP.maximumMagnitude(samples)
        let rms = sqrt(vDSP.sumOfSquares(samples) / Float(samples.count))
        return AudioWaveformSummary(
            frameCount: buffer.frameCount,
            peak: peak,
            rms: rms,
            bucketCount: max(bucketCount, 1)
        )
    }
}

public enum AudioPipelineError: Error, Sendable {
    case decodingFailed(String)
    case exportFailed(String)
    case resamplingFailed(String)
}

enum TemporaryAudioFileSecurityError: Error, LocalizedError {
    case containmentViolation(path: String, allowedDirectory: String)

    var errorDescription: String? {
        switch self {
        case let .containmentViolation(path, allowedDirectory):
            return "Resolved path '\(path)' escapes allowed directory '\(allowedDirectory)'"
        }
    }
}

enum TemporaryAudioFileSecurity {
    private static let allowedExtensions: Set<String> = ["wav", "aiff", "m4a", "mp3", "caf"]

    static func sanitizedExtension(_ raw: String) -> String {
        let cleaned = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return allowedExtensions.contains(cleaned) ? cleaned : "wav"
    }

    static func makeURL(
        fileExtension: String,
        fileManager: FileManager = .default
    ) throws -> URL {
        let safeExtension = sanitizedExtension(fileExtension)
        let directory = fileManager.temporaryDirectory
        let candidate = directory
            .appendingPathComponent(UUID().uuidString, isDirectory: false)
            .appendingPathExtension(safeExtension)
        try validateContainment(candidate, within: directory, fileManager: fileManager)
        return candidate
    }

    private static func validateContainment(
        _ candidate: URL,
        within allowedDirectory: URL,
        fileManager: FileManager
    ) throws {
        let canonicalCandidate = canonicalize(candidate, fileManager: fileManager)
        let canonicalAllowedDirectory = canonicalize(allowedDirectory, fileManager: fileManager)

        guard contains(canonicalCandidate, within: canonicalAllowedDirectory) else {
            throw TemporaryAudioFileSecurityError.containmentViolation(
                path: canonicalCandidate.path,
                allowedDirectory: canonicalAllowedDirectory.path
            )
        }
    }

    private static func canonicalize(_ url: URL, fileManager: FileManager) -> URL {
        let standardized = url.standardizedFileURL
        var existingAncestor = standardized
        var unresolvedComponents: [String] = []

        while !fileManager.fileExists(atPath: existingAncestor.path) {
            let parent = existingAncestor.deletingLastPathComponent()
            if parent.path == existingAncestor.path {
                break
            }
            unresolvedComponents.insert(existingAncestor.lastPathComponent, at: 0)
            existingAncestor = parent
        }

        let resolvedAncestor = existingAncestor.resolvingSymlinksInPath().standardizedFileURL
        return unresolvedComponents.reduce(resolvedAncestor) { partial, component in
            partial.appendingPathComponent(component, isDirectory: false)
        }
    }

    private static func contains(_ candidate: URL, within allowedDirectory: URL) -> Bool {
        let candidateComponents = candidate.standardizedFileURL.pathComponents
        let allowedComponents = allowedDirectory.standardizedFileURL.pathComponents

        guard candidateComponents.count >= allowedComponents.count else {
            return false
        }

        return zip(allowedComponents, candidateComponents).allSatisfy(==)
    }
}

public actor AudioPipeline {
    private let decoder: any AudioDecoder
    private let resampler: any AudioResampler
    private let exporter: any AudioExporter

    public init(
        decoder: any AudioDecoder = AVFoundationAudioDecoder(),
        resampler: any AudioResampler = AccelerateAudioResampler(),
        exporter: any AudioExporter = AVFoundationAudioExporter()
    ) {
        self.decoder = decoder
        self.resampler = resampler
        self.exporter = exporter
    }

    public func decode(_ data: Data, hint: String? = nil) async throws -> AudioPCMBuffer {
        try await decoder.decode(data, hint: hint)
    }

    public func resample(_ buffer: AudioPCMBuffer, to sampleRate: Double) async throws -> AudioPCMBuffer {
        try await resampler.resample(buffer, to: sampleRate)
    }

    public func export(_ buffer: AudioPCMBuffer, as format: AudioFormatDescriptor) async throws -> AudioExportedAsset {
        try await exporter.export(buffer, as: format)
    }

    public func export(
        _ buffer: AudioPCMBuffer,
        as format: AudioFormatDescriptor,
        to destinationURL: URL,
        chapterMarkers: [AudioChapterMarker] = []
    ) async throws -> AudioExportedFile {
        try await exporter.export(
            buffer,
            as: format,
            to: destinationURL,
            chapterMarkers: chapterMarkers
        )
    }

    public func normalize(_ buffer: AudioPCMBuffer) -> AudioPCMBuffer {
        var normalized = buffer
        normalized.channels = normalized.channels.map { channel in
            guard !channel.isEmpty else { return channel }
            var result = [Float](repeating: 0, count: channel.count)
            var lo: Float = -1
            var hi: Float = 1
            vDSP_vclip(channel, 1, &lo, &hi, &result, 1, vDSP_Length(channel.count))
            return result
        }
        return normalized
    }

    /// Peak-normalize audio to a target peak level in dB.
    /// Default target is -3 dB (~0.708 linear), which leaves headroom while
    /// ensuring consistent loudness across different TTS models.
    public func peakNormalize(_ buffer: AudioPCMBuffer, targetPeakDB: Float = -3.0) -> AudioPCMBuffer {
        let targetLinear = pow(10.0, targetPeakDB / 20.0)
        var out = buffer
        out.channels = out.channels.map { channel in
            guard !channel.isEmpty else { return channel }
            var peak: Float = 0
            vDSP_maxmgv(channel, 1, &peak, vDSP_Length(channel.count))
            guard peak > 1e-6 else { return channel }
            let gain = targetLinear / peak
            var result = [Float](repeating: 0, count: channel.count)
            var g = gain
            vDSP_vsmul(channel, 1, &g, &result, 1, vDSP_Length(channel.count))
            return result
        }
        return out
    }

    public func transcode(
        _ buffer: AudioPCMBuffer,
        container: String,
        sampleRate: Double? = nil
    ) async throws -> AudioExportedAsset {
        let working = if let sampleRate {
            try await resampler.resample(buffer, to: sampleRate)
        } else {
            buffer
        }

        let targetFormat = AudioFormatDescriptor(
            sampleRate: working.format.sampleRate,
            channelCount: working.format.channelCount,
            sampleFormat: working.format.sampleFormat,
            interleaved: working.format.interleaved,
            container: container
        )
        return try await exporter.export(working, as: targetFormat)
    }

    public func concatenate(_ buffers: [AudioPCMBuffer]) -> AudioPCMBuffer? {
        guard let first = buffers.first else { return nil }
        var combined = Array(repeating: [Float](), count: first.channels.count)

        for buffer in buffers {
            guard buffer.channels.count == combined.count else { return nil }
            for index in buffer.channels.indices {
                combined[index].append(contentsOf: buffer.channels[index])
            }
        }

        return AudioPCMBuffer(channels: combined, format: first.format)
    }

    public func waveform(for buffer: AudioPCMBuffer, bucketCount: Int = 32) -> AudioWaveformSummary {
        AudioWaveformAnalyzer.summarize(buffer, bucketCount: bucketCount)
    }
}
