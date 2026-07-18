import AVFoundation
import CoreMedia

/// Incremental M4A sink used by project exporters that must not retain an
/// entire multi-chapter render in memory.
public protocol M4AChapterStreamWriting: Sendable {
    func append(_ buffer: AudioPCMBuffer, chapterTitle: String) async throws
    func finalize() async throws -> AudioExportedFile
    func cancel() async
}

public struct AVFoundationAudioExporter: AudioExporter {
    private static let maximumInMemoryExportBytes = 256 * 1_024 * 1_024

    public init() {}

    public func export(_ buffer: AudioPCMBuffer, as format: AudioFormatDescriptor) async throws -> AudioExportedAsset {
        let container = Self.normalizedContainer(format.container)
        let tempURL = try TemporaryAudioFileSecurity.makeURL(fileExtension: container)
        defer { try? FileManager.default.removeItem(at: tempURL) }

        _ = try await export(buffer, as: format, to: tempURL, chapterMarkers: [])
        let values = try tempURL.resourceValues(forKeys: [.fileSizeKey])
        if let fileSize = values.fileSize,
           fileSize > Self.maximumInMemoryExportBytes {
            throw AudioPipelineError.exportFailed(
                "Encoded audio exceeds the \(Self.maximumInMemoryExportBytes)-byte in-memory export limit; use file export instead."
            )
        }
        let data = try Data(contentsOf: tempURL)

        return AudioExportedAsset(
            data: data,
            format: Self.normalizedFormat(format, container: container)
        )
    }

    public func export(
        _ buffer: AudioPCMBuffer,
        as format: AudioFormatDescriptor,
        to destinationURL: URL,
        chapterMarkers: [AudioChapterMarker]
    ) async throws -> AudioExportedFile {
        let container = Self.normalizedContainer(format.container)
        let normalizedFormat = Self.normalizedFormat(format, container: container)
        try Self.validate(buffer: buffer, format: normalizedFormat)
        let normalizedMarkers = Self.normalizedChapterMarkers(chapterMarkers)
        let fileManager = FileManager.default
        let destinationURL = destinationURL.standardizedFileURL
        let destinationDirectory = destinationURL.deletingLastPathComponent()

        try fileManager.createDirectory(
            at: destinationDirectory,
            withIntermediateDirectories: true
        )
        if fileManager.fileExists(atPath: destinationURL.path) {
            let values = try destinationURL.resourceValues(
                forKeys: [.isDirectoryKey, .isRegularFileKey, .isSymbolicLinkKey]
            )
            guard values.isDirectory != true,
                  values.isRegularFile == true,
                  values.isSymbolicLink != true else {
                throw AudioPipelineError.exportFailed(
                    "The audio destination must be a regular, non-symbolic-link file."
                )
            }
        }

        let stagingURL = destinationDirectory
            .appendingPathComponent(
                ".\(destinationURL.deletingPathExtension().lastPathComponent)-valar-\(UUID().uuidString)",
                isDirectory: false
            )
            .appendingPathExtension(container)
        defer { try? fileManager.removeItem(at: stagingURL) }

        switch container {
        case "wav":
            try await writeWaveFile(buffer, as: normalizedFormat, to: stagingURL)
        case "m4a":
            try await writeM4AFile(
                buffer,
                as: normalizedFormat,
                to: stagingURL,
                chapterMarkers: normalizedMarkers
            )
        default:
            throw AudioPipelineError.exportFailed("Unsupported container '\(container)'")
        }

        try Task.checkCancellation()
        if fileManager.fileExists(atPath: destinationURL.path) {
            _ = try fileManager.replaceItemAt(
                destinationURL,
                withItemAt: stagingURL,
                backupItemName: nil,
                options: []
            )
        } else {
            try fileManager.moveItem(at: stagingURL, to: destinationURL)
        }

        return AudioExportedFile(
            url: destinationURL,
            format: normalizedFormat,
            chapterMarkers: normalizedMarkers
        )
    }

    private func writeWaveFile(
        _ buffer: AudioPCMBuffer,
        as format: AudioFormatDescriptor,
        to destinationURL: URL
    ) async throws {
        let settings: [String: Any] = [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: format.sampleRate,
            AVNumberOfChannelsKey: format.channelCount,
            AVLinearPCMBitDepthKey: 32,
            AVLinearPCMIsFloatKey: true,
            AVLinearPCMIsBigEndianKey: false,
            AVLinearPCMIsNonInterleaved: false,
        ]

        let file = try AVAudioFile(
            forWriting: destinationURL,
            settings: settings,
            commonFormat: .pcmFormatFloat32,
            interleaved: false
        )
        let framesPerChunk = 16_384
        var startFrame = 0
        while startFrame < buffer.frameCount {
            try Task.checkCancellation()
            let frameCount = min(framesPerChunk, buffer.frameCount - startFrame)
            try file.write(
                from: try makePCMBuffer(
                    from: buffer,
                    startFrame: startFrame,
                    frameCount: frameCount,
                    format: format
                )
            )
            startFrame += frameCount
        }
    }

    private func writeM4AFile(
        _ buffer: AudioPCMBuffer,
        as format: AudioFormatDescriptor,
        to destinationURL: URL,
        chapterMarkers: [AudioChapterMarker]
    ) async throws {
        guard format.channelCount > 0, format.sampleRate > 0 else {
            throw AudioPipelineError.exportFailed("Invalid format: channelCount=\(format.channelCount) sampleRate=\(format.sampleRate)")
        }
        guard let pcmFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: format.sampleRate,
            channels: AVAudioChannelCount(format.channelCount),
            interleaved: true
        ) else {
            throw AudioPipelineError.exportFailed("Unable to create PCM export format")
        }

        let writer = try AVAssetWriter(outputURL: destinationURL, fileType: .m4a)
        let audioInput = AVAssetWriterInput(
            mediaType: .audio,
            outputSettings: [
                AVFormatIDKey: kAudioFormatMPEG4AAC,
                AVSampleRateKey: format.sampleRate,
                AVNumberOfChannelsKey: format.channelCount,
                AVEncoderBitRateKey: recommendedBitRate(for: format.channelCount),
            ],
            sourceFormatHint: pcmFormat.formatDescription
        )
        audioInput.expectsMediaDataInRealTime = false
        guard writer.canAdd(audioInput) else {
            throw AudioPipelineError.exportFailed("Unable to add AAC writer input")
        }
        writer.add(audioInput)

        let metadataAdaptor = try makeMetadataAdaptor(for: writer, chapterMarkers: chapterMarkers)

        guard writer.startWriting() else {
            throw writer.error
                ?? AudioPipelineError.exportFailed("Asset writer could not start.")
        }
        writer.startSession(atSourceTime: .zero)

        do {
            try await appendPCMChunks(
                from: buffer,
                using: pcmFormat,
                to: audioInput,
                writer: writer
            )
            try await appendChapterMarkers(
                chapterMarkers,
                with: metadataAdaptor,
                writer: writer
            )
            try await finishWriting(writer)
        } catch {
            if writer.status == .writing || writer.status == .unknown {
                writer.cancelWriting()
            }
            throw error
        }
    }

    private func appendPCMChunks(
        from buffer: AudioPCMBuffer,
        using pcmFormat: AVAudioFormat,
        to audioInput: AVAssetWriterInput,
        writer: AVAssetWriter
    ) async throws {
        let framesPerChunk = 4_096
        let totalFrames = buffer.frameCount

        guard totalFrames > 0 else {
            audioInput.markAsFinished()
            return
        }

        var nextFrame = 0
        while nextFrame < totalFrames {
            try await waitUntilReadyForMoreMediaData(
                on: audioInput,
                writer: writer
            )

            let count = min(framesPerChunk, totalFrames - nextFrame)
            let sampleBuffer = try makePCMSampleBuffer(
                from: buffer,
                startFrame: nextFrame,
                frameCount: count,
                format: pcmFormat
            )
            guard audioInput.append(sampleBuffer) else {
                audioInput.markAsFinished()
                throw writer.error
                    ?? AudioPipelineError.exportFailed("Failed to append AAC sample buffer")
            }

            nextFrame += count
        }

        audioInput.markAsFinished()
    }

    private func appendChapterMarkers(
        _ chapterMarkers: [AudioChapterMarker],
        with metadataAdaptor: MetadataAdaptor?,
        writer: AVAssetWriter
    ) async throws {
        guard let metadataAdaptor else { return }

        for marker in chapterMarkers {
            try await waitUntilReadyForMoreMediaData(
                on: metadataAdaptor.input,
                writer: writer
            )

            let item = AVMutableMetadataItem()
            item.identifier = .quickTimeUserDataChapter
            item.dataType = kCMMetadataBaseDataType_UTF8 as String
            item.extendedLanguageTag = "und"
            item.value = marker.title as NSString

            let start = CMTime(seconds: marker.startTime, preferredTimescale: 600)
            let duration = CMTime(seconds: marker.duration, preferredTimescale: 600)
            let group = AVTimedMetadataGroup(
                items: [item],
                timeRange: CMTimeRange(start: start, duration: duration)
            )

            guard metadataAdaptor.adaptor.append(group) else {
                metadataAdaptor.input.markAsFinished()
                throw AudioPipelineError.exportFailed("Failed to append chapter metadata")
            }
        }

        metadataAdaptor.input.markAsFinished()
    }

    private func finishWriting(_ writer: AVAssetWriter) async throws {
        await withCheckedContinuation { continuation in
            writer.finishWriting {
                continuation.resume()
            }
        }

        if let error = writer.error {
            throw error
        }
        guard writer.status == .completed else {
            throw AudioPipelineError.exportFailed("Asset writer finished with status \(writer.status.rawValue)")
        }
    }

    private func waitUntilReadyForMoreMediaData(
        on input: AVAssetWriterInput,
        writer: AVAssetWriter
    ) async throws {
        try Task.checkCancellation()
        var backoffMilliseconds = 1
        while !input.isReadyForMoreMediaData {
            try Task.checkCancellation()
            try Self.requireActiveWriter(writer)
            try await Task.sleep(for: .milliseconds(Int64(backoffMilliseconds)))
            backoffMilliseconds = min(backoffMilliseconds * 2, 20)
        }
        try Task.checkCancellation()
        try Self.requireActiveWriter(writer)
    }

    private static func requireActiveWriter(_ writer: AVAssetWriter) throws {
        switch writer.status {
        case .writing:
            return
        case .failed:
            throw writer.error
                ?? AudioPipelineError.exportFailed("Asset writer failed without an error.")
        case .cancelled:
            throw CancellationError()
        case .completed:
            throw AudioPipelineError.exportFailed(
                "Asset writer completed before all media was appended."
            )
        case .unknown:
            throw AudioPipelineError.exportFailed(
                "Asset writer returned to an unknown state while exporting."
            )
        @unknown default:
            throw AudioPipelineError.exportFailed("Asset writer entered an unsupported state.")
        }
    }

    private func makePCMBuffer(
        from buffer: AudioPCMBuffer,
        startFrame: Int,
        frameCount: Int,
        format: AudioFormatDescriptor
    ) throws -> AVAudioPCMBuffer {
        guard format.channelCount > 0, format.sampleRate > 0 else {
            throw AudioPipelineError.exportFailed("Invalid format: channelCount=\(format.channelCount) sampleRate=\(format.sampleRate)")
        }
        guard let avFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: format.sampleRate,
            channels: AVAudioChannelCount(format.channelCount),
            interleaved: false
        ) else {
            throw AudioPipelineError.exportFailed("Unable to create PCM buffer format")
        }

        guard startFrame >= 0,
              frameCount >= 0,
              startFrame <= buffer.frameCount,
              frameCount <= buffer.frameCount - startFrame,
              frameCount <= Int(AVAudioFrameCount.max) else {
            throw AudioPipelineError.exportFailed("Invalid PCM export frame range")
        }
        let frameCapacity = AVAudioFrameCount(frameCount)
        guard let pcmBuffer = AVAudioPCMBuffer(
            pcmFormat: avFormat,
            frameCapacity: frameCapacity
        ) else {
            throw AudioPipelineError.exportFailed("Unable to allocate PCM buffer")
        }
        pcmBuffer.frameLength = frameCapacity

        PCMCoding.fill(
            pcmBuffer,
            from: buffer.channels,
            startFrame: startFrame,
            frameCount: frameCount
        )

        return pcmBuffer
    }

    private func makePCMSampleBuffer(
        from buffer: AudioPCMBuffer,
        startFrame: Int,
        frameCount: Int,
        format: AVAudioFormat,
        presentationFrameOffset: Int = 0
    ) throws -> CMSampleBuffer {
        let (presentationFrame, overflowed) = presentationFrameOffset.addingReportingOverflow(startFrame)
        guard !overflowed,
              presentationFrame >= 0,
              presentationFrame <= Int(CMTimeValue.max) else {
            throw AudioPipelineError.exportFailed("AAC presentation timestamp overflow")
        }
        let interleavedData = makeInterleavedPCMData(
            from: buffer,
            startFrame: startFrame,
            frameCount: frameCount,
            channelCount: Int(format.channelCount)
        )

        var blockBuffer: CMBlockBuffer?
        let dataLength = interleavedData.count
        let status = CMBlockBufferCreateWithMemoryBlock(
            allocator: kCFAllocatorDefault,
            memoryBlock: nil,
            blockLength: dataLength,
            blockAllocator: nil,
            customBlockSource: nil,
            offsetToData: 0,
            dataLength: dataLength,
            flags: 0,
            blockBufferOut: &blockBuffer
        )
        guard status == kCMBlockBufferNoErr, let blockBuffer else {
            throw AudioPipelineError.exportFailed("Unable to allocate PCM block buffer")
        }

        try interleavedData.withUnsafeBytes { bytes in
            guard let baseAddress = bytes.baseAddress else { return }
            let replaceStatus = CMBlockBufferReplaceDataBytes(
                with: baseAddress,
                blockBuffer: blockBuffer,
                offsetIntoDestination: 0,
                dataLength: dataLength
            )
            guard replaceStatus == kCMBlockBufferNoErr else {
                throw AudioPipelineError.exportFailed("Unable to populate PCM block buffer")
            }
        }

        let presentationTime = CMTime(
            value: CMTimeValue(presentationFrame),
            timescale: CMTimeScale(format.sampleRate)
        )
        var sampleBuffer: CMSampleBuffer?
        let sampleStatus = CMAudioSampleBufferCreateReadyWithPacketDescriptions(
            allocator: kCFAllocatorDefault,
            dataBuffer: blockBuffer,
            formatDescription: format.formatDescription,
            sampleCount: frameCount,
            presentationTimeStamp: presentationTime,
            packetDescriptions: nil,
            sampleBufferOut: &sampleBuffer
        )
        guard sampleStatus == noErr, let sampleBuffer else {
            throw AudioPipelineError.exportFailed("Unable to create PCM sample buffer")
        }

        return sampleBuffer
    }

    func makeInterleavedPCMData(
        from buffer: AudioPCMBuffer,
        startFrame: Int,
        frameCount: Int,
        channelCount: Int
    ) -> Data {
        var data = Data(count: frameCount * channelCount * MemoryLayout<Float>.size)

        data.withUnsafeMutableBytes { raw in
            let output = raw.baseAddress!.assumingMemoryBound(to: Float.self)
            for channelIndex in 0 ..< channelCount {
                guard channelIndex < buffer.channels.count else { continue }
                let channel = buffer.channels[channelIndex]
                let availableFrames = min(frameCount, max(0, channel.count - startFrame))
                guard availableFrames > 0 else { continue }
                for frameOffset in 0 ..< availableFrames {
                    output[(frameOffset * channelCount) + channelIndex] = channel[startFrame + frameOffset]
                }
            }
        }

        return data
    }

    fileprivate func makeMetadataAdaptor(
        for writer: AVAssetWriter,
        chapterMarkers: [AudioChapterMarker],
        includeWhenEmpty: Bool = false
    ) throws -> MetadataAdaptor? {
        guard includeWhenEmpty || !chapterMarkers.isEmpty else { return nil }

        let specification: NSDictionary = [
            kCMMetadataFormatDescriptionMetadataSpecificationKey_Identifier as NSString:
                AVMetadataIdentifier.quickTimeUserDataChapter.rawValue as NSString,
            kCMMetadataFormatDescriptionMetadataSpecificationKey_DataType as NSString:
                kCMMetadataBaseDataType_UTF8 as NSString,
        ]

        var description: CMFormatDescription?
        let status = CMMetadataFormatDescriptionCreateWithMetadataSpecifications(
            allocator: kCFAllocatorDefault,
            metadataType: kCMMetadataFormatType_Boxed,
            metadataSpecifications: [specification] as CFArray,
            formatDescriptionOut: &description
        )
        guard status == noErr, let description else {
            throw AudioPipelineError.exportFailed("Unable to create chapter metadata description")
        }

        let input = AVAssetWriterInput(mediaType: .metadata, outputSettings: nil, sourceFormatHint: description)
        input.expectsMediaDataInRealTime = false
        guard writer.canAdd(input) else {
            throw AudioPipelineError.exportFailed("Unable to add chapter metadata input")
        }
        writer.add(input)

        return MetadataAdaptor(
            input: input,
            adaptor: AVAssetWriterInputMetadataAdaptor(assetWriterInput: input)
        )
    }

    fileprivate func recommendedBitRate(for channelCount: Int) -> Int {
        max(64_000, channelCount * 128_000)
    }

    private static func normalizedContainer(_ value: String) -> String {
        let normalized = value.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        switch normalized {
        case "aac", "m4a":
            return "m4a"
        case "", "wav":
            return "wav"
        default:
            return normalized
        }
    }

    private static func normalizedFormat(_ format: AudioFormatDescriptor, container: String) -> AudioFormatDescriptor {
        AudioFormatDescriptor(
            sampleRate: format.sampleRate,
            channelCount: format.channelCount,
            sampleFormat: .float32,
            interleaved: false,
            container: container
        )
    }

    private static func normalizedChapterMarkers(_ chapterMarkers: [AudioChapterMarker]) -> [AudioChapterMarker] {
        chapterMarkers
            .filter {
                !$0.title.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                    && $0.startTime.isFinite
                    && $0.startTime >= 0
                    && $0.duration.isFinite
                    && $0.duration > 0
            }
            .sorted { $0.startTime < $1.startTime }
    }

    fileprivate static func validate(
        buffer: AudioPCMBuffer,
        format: AudioFormatDescriptor
    ) throws {
        guard format.sampleRate.isFinite,
              format.sampleRate >= 1_000,
              format.sampleRate <= 768_000,
              (1...64).contains(format.channelCount),
              buffer.channels.count == format.channelCount,
              buffer.format.sampleRate.isFinite,
              abs(buffer.format.sampleRate - format.sampleRate)
                <= max(0.001, format.sampleRate * 1e-9),
              buffer.channels.dropFirst().allSatisfy({
                  $0.count == buffer.channels[0].count
              }) else {
            throw AudioPipelineError.exportFailed(
                "Audio export requires 1–64 equal-length channels and a finite 1000–768000 Hz sample rate."
            )
        }
    }
}

fileprivate struct MetadataAdaptor {
    let input: AVAssetWriterInput
    let adaptor: AVAssetWriterInputMetadataAdaptor
}

/// A bounded-memory AAC writer. Each `append` call encodes one chapter in
/// 4,096-frame pieces before returning, so callers can release that chapter's
/// PCM before producing the next one. The completed file is staged beside the
/// destination and atomically installed only after AVAssetWriter succeeds.
public actor StreamingM4AWriter: M4AChapterStreamWriting {
    private enum State: Equatable {
        case writing
        case appending
        case finishing
        case finished
        case cancelled
    }

    private static let framesPerChunk = 4_096

    private let destinationURL: URL
    private let stagingURL: URL
    private let format: AudioFormatDescriptor
    private let pcmFormat: AVAudioFormat
    private let writer: AVAssetWriter
    private let audioInput: AVAssetWriterInput
    private let metadataAdaptor: MetadataAdaptor
    private var chapterMarkers: [AudioChapterMarker] = []
    private var totalFrameCount = 0
    private var state = State.writing
    private var finalizedAsset: AudioExportedFile?

    public init(
        destinationURL: URL,
        sampleRate: Double,
        channelCount: Int
    ) throws {
        guard destinationURL.isFileURL,
              sampleRate.isFinite,
              sampleRate >= 1_000,
              sampleRate <= 768_000,
              (1...64).contains(channelCount) else {
            throw AudioPipelineError.exportFailed(
                "M4A streaming export requires a file URL, 1–64 channels, and a finite 1000–768000 Hz sample rate."
            )
        }

        let destinationURL = destinationURL.standardizedFileURL
        let fileManager = FileManager.default
        let destinationDirectory = destinationURL.deletingLastPathComponent()
        try fileManager.createDirectory(
            at: destinationDirectory,
            withIntermediateDirectories: true
        )
        try Self.validateDestination(destinationURL, fileManager: fileManager)

        let stagingURL = destinationDirectory
            .appendingPathComponent(
                ".\(destinationURL.deletingPathExtension().lastPathComponent)-valar-\(UUID().uuidString)",
                isDirectory: false
            )
            .appendingPathExtension("m4a")

        guard let pcmFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: sampleRate,
            channels: AVAudioChannelCount(channelCount),
            interleaved: true
        ) else {
            throw AudioPipelineError.exportFailed("Unable to create PCM export format")
        }

        do {
            let writer = try AVAssetWriter(outputURL: stagingURL, fileType: .m4a)
            let audioInput = AVAssetWriterInput(
                mediaType: .audio,
                outputSettings: [
                    AVFormatIDKey: kAudioFormatMPEG4AAC,
                    AVSampleRateKey: sampleRate,
                    AVNumberOfChannelsKey: channelCount,
                    AVEncoderBitRateKey: AVFoundationAudioExporter()
                        .recommendedBitRate(for: channelCount),
                ],
                sourceFormatHint: pcmFormat.formatDescription
            )
            audioInput.expectsMediaDataInRealTime = false
            guard writer.canAdd(audioInput) else {
                throw AudioPipelineError.exportFailed("Unable to add AAC writer input")
            }
            writer.add(audioInput)

            guard let metadataAdaptor = try AVFoundationAudioExporter()
                .makeMetadataAdaptor(
                    for: writer,
                    chapterMarkers: [],
                    includeWhenEmpty: true
                ) else {
                throw AudioPipelineError.exportFailed("Unable to add chapter metadata input")
            }

            guard writer.startWriting() else {
                throw writer.error
                    ?? AudioPipelineError.exportFailed("Asset writer could not start.")
            }
            writer.startSession(atSourceTime: .zero)

            self.destinationURL = destinationURL
            self.stagingURL = stagingURL
            self.format = AudioFormatDescriptor(
                sampleRate: sampleRate,
                channelCount: channelCount,
                sampleFormat: .float32,
                interleaved: false,
                container: "m4a"
            )
            self.pcmFormat = pcmFormat
            self.writer = writer
            self.audioInput = audioInput
            self.metadataAdaptor = metadataAdaptor
        } catch {
            try? fileManager.removeItem(at: stagingURL)
            throw error
        }
    }

    public func append(
        _ buffer: AudioPCMBuffer,
        chapterTitle: String
    ) async throws {
        guard state == .writing else {
            throw AudioPipelineError.exportFailed(
                "Another append or finalization operation is already active on this M4A writer"
            )
        }
        state = .appending
        do {
            try AVFoundationAudioExporter.validate(buffer: buffer, format: format)

            let (newTotalFrameCount, overflowed) = totalFrameCount
                .addingReportingOverflow(buffer.frameCount)
            guard !overflowed,
                  newTotalFrameCount <= Int(CMTimeValue.max) else {
                throw AudioPipelineError.exportFailed("M4A frame count overflow")
            }

            var nextFrame = 0
            while nextFrame < buffer.frameCount {
                try await waitUntilReadyForMoreMediaData(on: audioInput)
                let frameCount = min(
                    Self.framesPerChunk,
                    buffer.frameCount - nextFrame
                )
                let sampleBuffer = try makePCMSampleBuffer(
                    from: buffer,
                    startFrame: nextFrame,
                    frameCount: frameCount,
                    presentationFrameOffset: totalFrameCount
                )
                guard audioInput.append(sampleBuffer) else {
                    throw writer.error
                        ?? AudioPipelineError.exportFailed("Failed to append AAC sample buffer")
                }
                nextFrame += frameCount
            }

            if let marker = Self.chapterMarker(
                title: chapterTitle,
                startFrame: totalFrameCount,
                frameCount: buffer.frameCount,
                sampleRate: format.sampleRate
            ) {
                try await appendChapterMarker(marker)
                chapterMarkers.append(marker)
            }
            totalFrameCount = newTotalFrameCount
            state = .writing
        } catch {
            cancelWriterAndRemoveStagingFile()
            throw error
        }
    }

    nonisolated static func chapterMarker(
        title: String,
        startFrame: Int,
        frameCount: Int,
        sampleRate: Double
    ) -> AudioChapterMarker? {
        let title = title.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !title.isEmpty,
              startFrame >= 0,
              frameCount > 0,
              sampleRate.isFinite,
              sampleRate > 0 else {
            return nil
        }
        return AudioChapterMarker(
            title: title,
            startTime: Double(startFrame) / sampleRate,
            duration: Double(frameCount) / sampleRate
        )
    }

    public func finalize() async throws -> AudioExportedFile {
        if let finalizedAsset {
            return finalizedAsset
        }
        guard state == .writing else {
            throw AudioPipelineError.exportFailed("Cannot finalize a cancelled M4A writer")
        }

        state = .finishing
        audioInput.markAsFinished()
        metadataAdaptor.input.markAsFinished()
        do {
            await withCheckedContinuation { continuation in
                writer.finishWriting {
                    continuation.resume()
                }
            }
            if let error = writer.error {
                throw error
            }
            guard writer.status == .completed else {
                throw AudioPipelineError.exportFailed(
                    "Asset writer finished with status \(writer.status.rawValue)"
                )
            }
            try Task.checkCancellation()
            try Self.commitStagedFile(
                stagingURL,
                to: destinationURL,
                fileManager: .default
            )
            let asset = AudioExportedFile(
                url: destinationURL,
                format: format,
                chapterMarkers: chapterMarkers
            )
            finalizedAsset = asset
            state = .finished
            return asset
        } catch {
            cancelWriterAndRemoveStagingFile()
            throw error
        }
    }

    public func cancel() async {
        switch state {
        case .writing, .appending, .finishing:
            cancelWriterAndRemoveStagingFile()
        case .finished, .cancelled:
            return
        }
    }

    private func appendChapterMarker(_ marker: AudioChapterMarker) async throws {
        try await waitUntilReadyForMoreMediaData(on: metadataAdaptor.input)

        let item = AVMutableMetadataItem()
        item.identifier = .quickTimeUserDataChapter
        item.dataType = kCMMetadataBaseDataType_UTF8 as String
        item.extendedLanguageTag = "und"
        item.value = marker.title as NSString

        let group = AVTimedMetadataGroup(
            items: [item],
            timeRange: CMTimeRange(
                start: CMTime(seconds: marker.startTime, preferredTimescale: 600),
                duration: CMTime(seconds: marker.duration, preferredTimescale: 600)
            )
        )
        guard metadataAdaptor.adaptor.append(group) else {
            throw writer.error
                ?? AudioPipelineError.exportFailed("Failed to append chapter metadata")
        }
    }

    private func makePCMSampleBuffer(
        from buffer: AudioPCMBuffer,
        startFrame: Int,
        frameCount: Int,
        presentationFrameOffset: Int
    ) throws -> CMSampleBuffer {
        let (presentationFrame, overflowed) = presentationFrameOffset
            .addingReportingOverflow(startFrame)
        guard !overflowed,
              presentationFrame >= 0,
              presentationFrame <= Int(CMTimeValue.max) else {
            throw AudioPipelineError.exportFailed("AAC presentation timestamp overflow")
        }

        let interleavedData = AVFoundationAudioExporter().makeInterleavedPCMData(
            from: buffer,
            startFrame: startFrame,
            frameCount: frameCount,
            channelCount: format.channelCount
        )
        var blockBuffer: CMBlockBuffer?
        let status = CMBlockBufferCreateWithMemoryBlock(
            allocator: kCFAllocatorDefault,
            memoryBlock: nil,
            blockLength: interleavedData.count,
            blockAllocator: nil,
            customBlockSource: nil,
            offsetToData: 0,
            dataLength: interleavedData.count,
            flags: 0,
            blockBufferOut: &blockBuffer
        )
        guard status == kCMBlockBufferNoErr, let blockBuffer else {
            throw AudioPipelineError.exportFailed("Unable to allocate PCM block buffer")
        }
        try interleavedData.withUnsafeBytes { bytes in
            guard let baseAddress = bytes.baseAddress else { return }
            let replaceStatus = CMBlockBufferReplaceDataBytes(
                with: baseAddress,
                blockBuffer: blockBuffer,
                offsetIntoDestination: 0,
                dataLength: interleavedData.count
            )
            guard replaceStatus == kCMBlockBufferNoErr else {
                throw AudioPipelineError.exportFailed("Unable to populate PCM block buffer")
            }
        }

        var sampleBuffer: CMSampleBuffer?
        let sampleStatus = CMAudioSampleBufferCreateReadyWithPacketDescriptions(
            allocator: kCFAllocatorDefault,
            dataBuffer: blockBuffer,
            formatDescription: pcmFormat.formatDescription,
            sampleCount: frameCount,
            presentationTimeStamp: CMTime(
                value: CMTimeValue(presentationFrame),
                timescale: CMTimeScale(format.sampleRate)
            ),
            packetDescriptions: nil,
            sampleBufferOut: &sampleBuffer
        )
        guard sampleStatus == noErr, let sampleBuffer else {
            throw AudioPipelineError.exportFailed("Unable to create PCM sample buffer")
        }
        return sampleBuffer
    }

    private func waitUntilReadyForMoreMediaData(
        on input: AVAssetWriterInput
    ) async throws {
        try Task.checkCancellation()
        var backoffMilliseconds = 1
        while !input.isReadyForMoreMediaData {
            try Task.checkCancellation()
            try requireActiveWriter()
            try await Task.sleep(for: .milliseconds(Int64(backoffMilliseconds)))
            backoffMilliseconds = min(backoffMilliseconds * 2, 20)
        }
        try Task.checkCancellation()
        try requireActiveWriter()
    }

    private func requireActiveWriter() throws {
        switch writer.status {
        case .writing:
            return
        case .failed:
            throw writer.error
                ?? AudioPipelineError.exportFailed("Asset writer failed without an error.")
        case .cancelled:
            throw CancellationError()
        case .completed:
            throw AudioPipelineError.exportFailed(
                "Asset writer completed before all media was appended."
            )
        case .unknown:
            throw AudioPipelineError.exportFailed(
                "Asset writer returned to an unknown state while exporting."
            )
        @unknown default:
            throw AudioPipelineError.exportFailed("Asset writer entered an unsupported state.")
        }
    }

    private func cancelWriterAndRemoveStagingFile() {
        if writer.status == .writing || writer.status == .unknown {
            writer.cancelWriting()
        }
        try? FileManager.default.removeItem(at: stagingURL)
        state = .cancelled
    }

    private static func validateDestination(
        _ destinationURL: URL,
        fileManager: FileManager
    ) throws {
        guard !destinationURL.lastPathComponent.isEmpty else {
            throw AudioPipelineError.exportFailed("The M4A destination has no file name.")
        }
        guard fileManager.fileExists(atPath: destinationURL.path) else { return }
        let values = try destinationURL.resourceValues(
            forKeys: [.isDirectoryKey, .isRegularFileKey, .isSymbolicLinkKey]
        )
        guard values.isDirectory != true,
              values.isRegularFile == true,
              values.isSymbolicLink != true else {
            throw AudioPipelineError.exportFailed(
                "The audio destination must be a regular, non-symbolic-link file."
            )
        }
    }

    private static func commitStagedFile(
        _ stagingURL: URL,
        to destinationURL: URL,
        fileManager: FileManager
    ) throws {
        try validateDestination(destinationURL, fileManager: fileManager)
        if fileManager.fileExists(atPath: destinationURL.path) {
            _ = try fileManager.replaceItemAt(
                destinationURL,
                withItemAt: stagingURL,
                backupItemName: nil,
                options: []
            )
        } else {
            try fileManager.moveItem(at: stagingURL, to: destinationURL)
        }
    }
}
