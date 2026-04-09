import AVFoundation
import CoreMedia

public struct AVFoundationAudioExporter: AudioExporter {
    public init() {}

    public func export(_ buffer: AudioPCMBuffer, as format: AudioFormatDescriptor) async throws -> AudioExportedAsset {
        let container = Self.normalizedContainer(format.container)
        let tempURL = try TemporaryAudioFileSecurity.makeURL(fileExtension: container)
        defer { try? FileManager.default.removeItem(at: tempURL) }

        _ = try await export(buffer, as: format, to: tempURL, chapterMarkers: [])
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
        let normalizedMarkers = Self.normalizedChapterMarkers(chapterMarkers)

        try FileManager.default.createDirectory(
            at: destinationURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        if FileManager.default.fileExists(atPath: destinationURL.path) {
            do {
                try FileManager.default.removeItem(at: destinationURL)
            } catch let error as CocoaError where error.code == .fileNoSuchFile {
                // Another process or AVFoundation may have already removed the stale target.
            }
        }

        switch container {
        case "wav":
            try writeWaveFile(buffer, as: normalizedFormat, to: destinationURL)
        case "m4a":
            try await writeM4AFile(
                buffer,
                as: normalizedFormat,
                to: destinationURL,
                chapterMarkers: normalizedMarkers
            )
        default:
            throw AudioPipelineError.exportFailed("Unsupported container '\(container)'")
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
    ) throws {
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
        try file.write(from: try makePCMBuffer(from: buffer, format: format))
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

        writer.startWriting()
        writer.startSession(atSourceTime: .zero)

        try await appendPCMChunks(from: buffer, using: pcmFormat, to: audioInput, writer: writer)
        try await appendChapterMarkers(chapterMarkers, with: metadataAdaptor)

        try await finishWriting(writer)
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
            try await waitUntilReadyForMoreMediaData(on: audioInput)

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
        with metadataAdaptor: MetadataAdaptor?
    ) async throws {
        guard let metadataAdaptor else { return }

        for marker in chapterMarkers {
            try await waitUntilReadyForMoreMediaData(on: metadataAdaptor.input)

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

    private func waitUntilReadyForMoreMediaData(on input: AVAssetWriterInput) async throws {
        while !input.isReadyForMoreMediaData {
            try Task.checkCancellation()
            await Task.yield()
        }
    }

    private func makePCMBuffer(from buffer: AudioPCMBuffer, format: AudioFormatDescriptor) throws -> AVAudioPCMBuffer {
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

        guard buffer.frameCount >= 0, buffer.frameCount <= Int(AVAudioFrameCount.max) else {
            throw AudioPipelineError.exportFailed("Frame count \(buffer.frameCount) exceeds AVAudioPCMBuffer capacity")
        }
        let frameCapacity = AVAudioFrameCount(buffer.frameCount)
        guard let pcmBuffer = AVAudioPCMBuffer(
            pcmFormat: avFormat,
            frameCapacity: frameCapacity
        ) else {
            throw AudioPipelineError.exportFailed("Unable to allocate PCM buffer")
        }
        pcmBuffer.frameLength = frameCapacity

        PCMCoding.fill(pcmBuffer, from: buffer.channels, frameCount: buffer.frameCount)

        return pcmBuffer
    }

    private func makePCMSampleBuffer(
        from buffer: AudioPCMBuffer,
        startFrame: Int,
        frameCount: Int,
        format: AVAudioFormat
    ) throws -> CMSampleBuffer {
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
            value: CMTimeValue(startFrame),
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

    private func makeMetadataAdaptor(
        for writer: AVAssetWriter,
        chapterMarkers: [AudioChapterMarker]
    ) throws -> MetadataAdaptor? {
        guard !chapterMarkers.isEmpty else { return nil }

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

    private func recommendedBitRate(for channelCount: Int) -> Int {
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
            .filter { !$0.title.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && $0.duration > 0 }
            .sorted { $0.startTime < $1.startTime }
    }
}

private struct MetadataAdaptor {
    let input: AVAssetWriterInput
    let adaptor: AVAssetWriterInputMetadataAdaptor
}
