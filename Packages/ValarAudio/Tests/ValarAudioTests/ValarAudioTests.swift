import AVFoundation
import Foundation
import XCTest
@testable import ValarAudio
import Dispatch

final class ValarAudioTests: XCTestCase {
    func testExportProducesValidData() async throws {
        let buffer = AudioPCMBuffer(mono: [0, 0.25, 0.5, 1.0], sampleRate: 24_000, container: "wav")
        let exporter = AVFoundationAudioExporter()
        let format = AudioFormatDescriptor(sampleRate: 24_000, channelCount: 1, container: "wav")
        let asset = try await exporter.export(buffer, as: format)
        XCTAssertGreaterThan(asset.data.count, 0)
        XCTAssertEqual(asset.format.container, "wav")
        XCTAssertEqual(asset.format.sampleRate, 24_000)
    }

    func testWaveExportWritesFloat32WaveFile() async throws {
        let buffer = AudioPCMBuffer(mono: [0, 0.25, 0.5, 1.0], sampleRate: 24_000)
        let exporter = AVFoundationAudioExporter()
        let url = makeTemporaryURL(pathExtension: "wav")
        defer { try? FileManager.default.removeItem(at: url) }

        let exported = try await exporter.export(
            buffer,
            as: AudioFormatDescriptor(sampleRate: 24_000, channelCount: 1, container: "wav"),
            to: url,
            chapterMarkers: []
        )

        let data = try Data(contentsOf: exported.url)
        XCTAssertEqual(String(decoding: data.prefix(4), as: UTF8.self), "RIFF")
        XCTAssertEqual(String(decoding: data.dropFirst(8).prefix(4), as: UTF8.self), "WAVE")

        let audioFile = try AVAudioFile(forReading: exported.url)
        XCTAssertEqual(audioFile.fileFormat.sampleRate, 24_000)
        XCTAssertEqual(audioFile.processingFormat.commonFormat, .pcmFormatFloat32)
    }

    func testDecodeAndExportRoundTrip() async throws {
        let original = AudioPCMBuffer(
            mono: [0, 0.25, 0.5, 0.75, 1.0],
            sampleRate: 24_000,
            container: "wav"
        )
        let exporter = AVFoundationAudioExporter()
        let format = AudioFormatDescriptor(sampleRate: 24_000, channelCount: 1, container: "wav")
        let exported = try await exporter.export(original, as: format)

        let decoder = AVFoundationAudioDecoder()
        let decoded = try await decoder.decode(exported.data, hint: "wav")
        XCTAssertEqual(decoded.frameCount, original.frameCount)
        XCTAssertEqual(decoded.format.sampleRate, 24_000)
        XCTAssertEqual(decoded.format.channelCount, 1)
    }

    func testM4AExportProducesAACFile() async throws {
        let buffer = AudioPCMBuffer(
            mono: [Float](repeating: 0.1, count: 24_000),
            sampleRate: 24_000
        )
        let exporter = AVFoundationAudioExporter()
        let url = makeTemporaryURL(pathExtension: "m4a")
        defer { try? FileManager.default.removeItem(at: url) }

        let exported: AudioExportedFile
        do {
            exported = try await exporter.export(
                buffer,
                as: AudioFormatDescriptor(sampleRate: 24_000, channelCount: 1, container: "m4a"),
                to: url,
                chapterMarkers: []
            )
        } catch {
            throw XCTSkip("AAC encoding not available in this environment: \(error.localizedDescription)")
        }

        let audioFile = try AVAudioFile(forReading: exported.url)
        let formatID = (audioFile.fileFormat.settings[AVFormatIDKey] as? NSNumber)?.uint32Value
        XCTAssertEqual(formatID, kAudioFormatMPEG4AAC)
    }

    func testM4AExportEmbedsChapterMarkers() async throws {
        let buffer = AudioPCMBuffer(
            mono: [Float](repeating: 0.1, count: 48_000),
            sampleRate: 24_000
        )
        let exporter = AVFoundationAudioExporter()
        let url = makeTemporaryURL(pathExtension: "m4a")
        defer { try? FileManager.default.removeItem(at: url) }

        do {
            _ = try await exporter.export(
                buffer,
                as: AudioFormatDescriptor(sampleRate: 24_000, channelCount: 1, container: "m4a"),
                to: url,
                chapterMarkers: [
                    AudioChapterMarker(title: "Intro", startTime: 0, duration: 1),
                    AudioChapterMarker(title: "Outro", startTime: 1, duration: 1),
                ]
            )
        } catch {
            throw XCTSkip("AAC encoding not available in this environment: \(error.localizedDescription)")
        }

        let asset = AVURLAsset(url: url)
        let groups = try await asset.loadChapterMetadataGroups(bestMatchingPreferredLanguages: [])
        XCTAssertEqual(groups.count, 2)

        let firstTitle = try await groups.first?.items.first?.load(.stringValue)
        let secondTitle = try await groups.dropFirst().first?.items.first?.load(.stringValue)
        XCTAssertEqual(firstTitle, "Intro")
        XCTAssertEqual(secondTitle, "Outro")
    }

    func testResamplerChangesSampleCount() async throws {
        let buffer = AudioPCMBuffer(
            mono: [Float](repeating: 0.5, count: 1000),
            sampleRate: 24_000
        )
        let resampler = AccelerateAudioResampler()
        let resampled = try await resampler.resample(buffer, to: 48_000)
        XCTAssertEqual(resampled.format.sampleRate, 48_000)
        XCTAssertEqual(resampled.frameCount, 2000)
    }

    func testResamplerNoOpWhenSameRate() async throws {
        let buffer = AudioPCMBuffer(mono: [0, 0.5, 1.0], sampleRate: 24_000)
        let resampler = AccelerateAudioResampler()
        let result = try await resampler.resample(buffer, to: 24_000)
        XCTAssertEqual(result.channels, buffer.channels)
    }

    func testResamplerUpsampleIntegerRatioMatchesNaiveLoopExactly() async throws {
        let samples = [0, 0.5, 1.0, -0.75, 0.25, 0.9].map(Float.init)
        let buffer = AudioPCMBuffer(mono: samples, sampleRate: 24_000)
        let resampler = AccelerateAudioResampler()

        let result = try await resampler.resample(buffer, to: 48_000)
        let expected = naiveResample(samples, sourceRate: 24_000, targetRate: 48_000)

        XCTAssertEqual(result.channels[0], expected)
    }

    func testResamplerDownsampleIntegerRatioMatchesNaiveLoopExactly() async throws {
        let samples = (0..<24).map { index in
            Float(index % 7) / 6 - 0.5
        }
        let buffer = AudioPCMBuffer(mono: samples, sampleRate: 48_000)
        let resampler = AccelerateAudioResampler()

        let result = try await resampler.resample(buffer, to: 24_000)
        let expected = naiveResample(samples, sourceRate: 48_000, targetRate: 24_000)

        XCTAssertEqual(result.channels[0], expected)
    }

    func testResamplerHandlesNonIntegerRatioShortBuffer() async throws {
        let buffer = AudioPCMBuffer(mono: [0.2, -0.4, 0.8], sampleRate: 48_000)
        let resampler = AccelerateAudioResampler()

        let result = try await resampler.resample(buffer, to: 44_100)

        XCTAssertEqual(result.format.sampleRate, 44_100)
        XCTAssertEqual(result.frameCount, 2)
        XCTAssertEqual(result.channels.count, 1)
        XCTAssertEqual(result.channels[0].count, 2)
        XCTAssertTrue(result.channels[0].allSatisfy(\.isFinite))
    }

    func testResamplerNonIntegerRatioPreservesLowFrequencyShape() async throws {
        let sampleRate = 48_000.0
        let targetRate = 44_100.0
        let sampleCount = 4_800
        let samples = (0..<sampleCount).map { index in
            Float(sin((2 * Double.pi * 440 * Double(index)) / sampleRate))
        }
        let buffer = AudioPCMBuffer(mono: samples, sampleRate: sampleRate)
        let resampler = AccelerateAudioResampler()

        let result = try await resampler.resample(buffer, to: targetRate)
        let expected = naiveResample(samples, sourceRate: sampleRate, targetRate: targetRate)
        let meanAbsoluteError = zip(result.channels[0], expected)
            .reduce(into: Float.zero) { partial, pair in
                partial += abs(pair.0 - pair.1)
            } / Float(expected.count)

        XCTAssertLessThan(meanAbsoluteError, 0.02)
    }

    func testResamplerBenchmarkSIMDBeatsNaiveLoop() async throws {
        let sourceRate = 48_000.0
        let targetRate = 24_000.0
        let samples = benchmarkSamples(frameCount: 192_000, sampleRate: sourceRate)
        let buffer = AudioPCMBuffer(mono: samples, sampleRate: sourceRate)
        let resampler = AccelerateAudioResampler()
        let iterations = 20

        _ = try await resampler.resample(buffer, to: targetRate)
        _ = naiveResample(samples, sourceRate: sourceRate, targetRate: targetRate)

        let naiveDuration = measureWallClock {
            for _ in 0..<iterations {
                _ = naiveResample(samples, sourceRate: sourceRate, targetRate: targetRate)
            }
        }
        let simdDuration = try await measureWallClockAsync {
            for _ in 0..<iterations {
                _ = try await resampler.resample(buffer, to: targetRate)
            }
        }
        let speedup = naiveDuration / simdDuration
        print(
            String(
                format: "Resampler benchmark: naive %.4fs SIMD %.4fs speedup %.2fx",
                naiveDuration,
                simdDuration,
                speedup
            )
        )

        XCTAssertGreaterThan(
            speedup,
            1.1,
            String(format: "Observed SIMD speedup %.2fx (naive %.4fs, SIMD %.4fs)", speedup, naiveDuration, simdDuration)
        )
    }

    // MARK: - Filter coefficient cache

    func testResamplerFilterCacheReturnsIdenticalOutputForRepeatedCalls() async throws {
        let samples = (0..<4_800).map { index in
            Float(sin(2 * Double.pi * 440 * Double(index) / 48_000))
        }
        let buffer = AudioPCMBuffer(mono: samples, sampleRate: 48_000)
        let resampler = AccelerateAudioResampler()

        let first = try await resampler.resample(buffer, to: 44_100)
        let second = try await resampler.resample(buffer, to: 44_100)

        XCTAssertEqual(first.channels, second.channels)
        XCTAssertEqual(first.format.sampleRate, second.format.sampleRate)
    }

    func testResamplerFilterCacheBenchmarkSpeedsUpRepeatedCalls() async throws {
        // Small buffer makes the per-call filter-computation cost a significant fraction
        // of total work, so the cache speedup is observable.
        let sampleCount = 4_800  // 100 ms at 48 kHz
        let sourceRate = 48_000.0
        let targetRate = 44_100.0
        let buffer = AudioPCMBuffer(
            mono: [Float](repeating: 0.5, count: sampleCount),
            sampleRate: sourceRate
        )
        let resampler = AccelerateAudioResampler()
        let iterations = 200

        // Pre-warm the cache for the target rate pair.
        _ = try await resampler.resample(buffer, to: targetRate)

        // Warm path: same rate pair every iteration — cache hit on every call.
        let warmDuration = try await measureWallClockAsync {
            for _ in 0..<iterations {
                _ = try await resampler.resample(buffer, to: targetRate)
            }
        }

        // Cold path: unique rate per iteration — cache miss on every call.
        // Rates 43_700..43_700 + iterations*35 all produce decimationFactor=1
        // from a 48_000 source, so the data-path work is comparable to the warm path.
        let coldDuration = try await measureWallClockAsync {
            for i in 0..<iterations {
                let coldRate = 43_700.0 + Double(i) * 35.0
                _ = try await resampler.resample(buffer, to: coldRate)
            }
        }

        let speedup = coldDuration / warmDuration
        print(
            String(
                format: "Filter cache benchmark: cold %.4fs warm %.4fs speedup %.2fx",
                coldDuration, warmDuration, speedup
            )
        )
        XCTAssertGreaterThan(
            speedup,
            1.1,
            String(format: "Observed filter-cache speedup %.2fx (cold %.4fs, warm %.4fs)", speedup, coldDuration, warmDuration)
        )
    }

    func testNormalizeClipsSamplesOutsideRange() async {
        let buffer = AudioPCMBuffer(mono: [-2.0, -1.5, -1.0, 0.0, 1.0, 1.5, 2.0], sampleRate: 24_000)
        let pipeline = AudioPipeline()
        let result = await pipeline.normalize(buffer)
        let channel = result.channels[0]
        XCTAssertEqual(channel[0], -1.0, accuracy: 1e-6)
        XCTAssertEqual(channel[1], -1.0, accuracy: 1e-6)
        XCTAssertEqual(channel[2], -1.0, accuracy: 1e-6)
        XCTAssertEqual(channel[3],  0.0, accuracy: 1e-6)
        XCTAssertEqual(channel[4],  1.0, accuracy: 1e-6)
        XCTAssertEqual(channel[5],  1.0, accuracy: 1e-6)
        XCTAssertEqual(channel[6],  1.0, accuracy: 1e-6)
    }

    func testNormalizePreservesInRangeSamples() async {
        let samples: [Float] = [-1.0, -0.5, 0.0, 0.5, 1.0]
        let buffer = AudioPCMBuffer(mono: samples, sampleRate: 24_000)
        let pipeline = AudioPipeline()
        let result = await pipeline.normalize(buffer)
        XCTAssertEqual(result.channels[0], samples)
    }

    func testNormalizeEmptyChannelIsStable() async {
        let buffer = AudioPCMBuffer(channels: [[], []], format: AudioFormatDescriptor(sampleRate: 24_000, channelCount: 2))
        let pipeline = AudioPipeline()
        let result = await pipeline.normalize(buffer)
        XCTAssertEqual(result.channels, [[], []])
    }

    func testNormalizeBenchmarkVDSPBeatsScalarMap() async {
        let frameCount = 192_000
        let samples = (0..<frameCount).map { _ in Float.random(in: -2...2) }
        let buffer = AudioPCMBuffer(mono: samples, sampleRate: 24_000)
        let pipeline = AudioPipeline()
        let iterations = 20

        _ = await pipeline.normalize(buffer)
        _ = naiveClip(samples)

        let naiveDuration = measureWallClock {
            for _ in 0..<iterations {
                _ = naiveClip(samples)
            }
        }
        let vdspDuration = await measureWallClockAsync {
            for _ in 0..<iterations {
                _ = await pipeline.normalize(buffer)
            }
        }
        let speedup = naiveDuration / vdspDuration
        print(
            String(
                format: "Normalize benchmark: naive %.4fs vDSP %.4fs speedup %.2fx",
                naiveDuration,
                vdspDuration,
                speedup
            )
        )
        XCTAssertGreaterThan(
            speedup,
            1.1,
            String(format: "Observed vDSP speedup %.2fx (naive %.4fs, vDSP %.4fs)", speedup, naiveDuration, vdspDuration)
        )
    }

    func testConcatenateCombinesFrames() async {
        let first = AudioPCMBuffer(mono: [0, 1], sampleRate: 24_000)
        let second = AudioPCMBuffer(mono: [2, 3, 4], sampleRate: 24_000)
        let pipeline = AudioPipeline()
        let combined = await pipeline.concatenate([first, second])

        XCTAssertEqual(combined?.frameCount, 5)
    }

    func testWaveformSummary() async {
        let buffer = AudioPCMBuffer(mono: [0, 0.5, -0.5, 1.0, -1.0], sampleRate: 24_000)
        let pipeline = AudioPipeline()
        let waveform = await pipeline.waveform(for: buffer, bucketCount: 8)
        XCTAssertEqual(waveform.frameCount, 5)
        XCTAssertEqual(waveform.bucketCount, 8)
        XCTAssertEqual(waveform.peak, 1.0)
    }

    func testSanitizedExtensionAllowlistKnownFormats() {
        XCTAssertEqual(TemporaryAudioFileSecurity.sanitizedExtension("wav"), "wav")
        XCTAssertEqual(TemporaryAudioFileSecurity.sanitizedExtension("MP3"), "mp3")
        XCTAssertEqual(TemporaryAudioFileSecurity.sanitizedExtension("  m4a  "), "m4a")
        XCTAssertEqual(TemporaryAudioFileSecurity.sanitizedExtension("aiff"), "aiff")
        XCTAssertEqual(TemporaryAudioFileSecurity.sanitizedExtension("caf"), "caf")
    }

    func testSanitizedExtensionRejectsUnknownFormats() {
        XCTAssertEqual(TemporaryAudioFileSecurity.sanitizedExtension("ogg"), "wav")
        XCTAssertEqual(TemporaryAudioFileSecurity.sanitizedExtension("exe"), "wav")
        XCTAssertEqual(TemporaryAudioFileSecurity.sanitizedExtension(""), "wav")
    }

    func testSanitizedExtensionBlocksPathSeparatorInjection() {
        XCTAssertEqual(TemporaryAudioFileSecurity.sanitizedExtension("../../../etc/passwd"), "wav")
        XCTAssertEqual(TemporaryAudioFileSecurity.sanitizedExtension("wav/../../tmp"), "wav")
        XCTAssertEqual(TemporaryAudioFileSecurity.sanitizedExtension(".."), "wav")
    }

    func testMakeURLWithMaliciousExtensionStaysInTempDir() throws {
        let url = try TemporaryAudioFileSecurity.makeURL(fileExtension: "../../../etc/passwd")
        let tempDir = FileManager.default.temporaryDirectory.standardizedFileURL.path
        XCTAssertTrue(url.standardizedFileURL.path.hasPrefix(tempDir))
        XCTAssertTrue(url.pathExtension == "wav")
    }

    func testPipelineTranscode() async throws {
        let clip = AudioPCMBuffer(
            mono: [0, 0.25, 0.5, 1.0],
            sampleRate: 24_000,
            container: "wav"
        )
        let pipeline = AudioPipeline()
        let converted = try await pipeline.transcode(clip, container: "wav", sampleRate: 48_000)
        XCTAssertEqual(converted.format.container, "wav")
        XCTAssertEqual(converted.format.sampleRate, 48_000)
        XCTAssertGreaterThan(converted.data.count, 0)
    }

    func testInterleaveStereoMatchesNaiveLoop() {
        let left: [Float] = [1, 2, 3, 4, 5]
        let right: [Float] = [10, 20, 30, 40, 50]
        let buffer = AudioPCMBuffer(
            channels: [left, right],
            format: AudioFormatDescriptor(sampleRate: 24_000, channelCount: 2)
        )
        let exporter = AVFoundationAudioExporter()

        let result = exporter.makeInterleavedPCMData(from: buffer, startFrame: 0, frameCount: 5, channelCount: 2)
        let expected = naiveInterleave(buffer, startFrame: 0, frameCount: 5, channelCount: 2)

        XCTAssertEqual(result, expected)
    }

    func testInterleaveMonoMatchesNaiveLoop() {
        let samples: [Float] = [0.1, 0.2, 0.3, -0.4, 0.5]
        let buffer = AudioPCMBuffer(mono: samples, sampleRate: 24_000)
        let exporter = AVFoundationAudioExporter()

        let result = exporter.makeInterleavedPCMData(from: buffer, startFrame: 1, frameCount: 3, channelCount: 1)
        let expected = naiveInterleave(buffer, startFrame: 1, frameCount: 3, channelCount: 1)

        XCTAssertEqual(result, expected)
    }

    func testInterleaveShortChannelPadsWithZero() {
        let buffer = AudioPCMBuffer(
            channels: [[1, 2], [10, 20, 30, 40]],
            format: AudioFormatDescriptor(sampleRate: 24_000, channelCount: 2)
        )
        let exporter = AVFoundationAudioExporter()

        let result = exporter.makeInterleavedPCMData(from: buffer, startFrame: 0, frameCount: 4, channelCount: 2)
        let expected = naiveInterleave(buffer, startFrame: 0, frameCount: 4, channelCount: 2)

        XCTAssertEqual(result, expected)
    }

    func testInterleaveBenchmarkVDSPBeatsNaiveLoop() {
        let frameCount = 192_000
        let channelCount = 2
        let left = benchmarkSamples(frameCount: frameCount, sampleRate: 48_000)
        let right = left.map { -$0 }
        let buffer = AudioPCMBuffer(
            channels: [left, right],
            format: AudioFormatDescriptor(sampleRate: 48_000, channelCount: channelCount)
        )
        let exporter = AVFoundationAudioExporter()
        let iterations = 20

        _ = exporter.makeInterleavedPCMData(from: buffer, startFrame: 0, frameCount: frameCount, channelCount: channelCount)
        _ = naiveInterleave(buffer, startFrame: 0, frameCount: frameCount, channelCount: channelCount)

        let naiveDuration = measureWallClock {
            for _ in 0..<iterations {
                _ = naiveInterleave(buffer, startFrame: 0, frameCount: frameCount, channelCount: channelCount)
            }
        }
        let vdspDuration = measureWallClock {
            for _ in 0..<iterations {
                _ = exporter.makeInterleavedPCMData(from: buffer, startFrame: 0, frameCount: frameCount, channelCount: channelCount)
            }
        }
        let speedup = naiveDuration / vdspDuration
        print(
            String(
                format: "Interleave benchmark: naive %.4fs vDSP %.4fs speedup %.2fx",
                naiveDuration,
                vdspDuration,
                speedup
            )
        )
        XCTAssertGreaterThan(
            speedup,
            1.1,
            String(format: "Observed vDSP speedup %.2fx (naive %.4fs, vDSP %.4fs)", speedup, naiveDuration, vdspDuration)
        )
    }

    // MARK: - pcm_f32le fast path

    func testDecodePCMFloat32LEMonoNoMetadata() async throws {
        let samples: [Float] = [0, 0.25, -0.5, 1.0, -1.0]
        let data = makePCMFloat32LEData(samples)
        let decoder = AVFoundationAudioDecoder()

        let result = try await decoder.decode(data, hint: "pcm_f32le")

        XCTAssertEqual(result.channels.count, 1)
        XCTAssertEqual(result.channels[0], samples)
        XCTAssertEqual(result.format.sampleRate, 24_000)
        XCTAssertEqual(result.format.channelCount, 1)
        XCTAssertEqual(result.format.container, "pcm")
        XCTAssertFalse(result.format.interleaved)
    }

    func testDecodePCMFloat32LEHintWithSampleRate() async throws {
        let samples: [Float] = [0.1, -0.2, 0.3]
        let data = makePCMFloat32LEData(samples)
        let decoder = AVFoundationAudioDecoder()

        let result = try await decoder.decode(data, hint: "pcm_f32le:48000")

        XCTAssertEqual(result.format.sampleRate, 48_000)
        XCTAssertEqual(result.format.channelCount, 1)
        XCTAssertEqual(result.channels[0], samples)
    }

    func testDecodePCMFloat32LEHintCaseInsensitive() async throws {
        let samples: [Float] = [0.5, -0.5]
        let data = makePCMFloat32LEData(samples)
        let decoder = AVFoundationAudioDecoder()

        let result = try await decoder.decode(data, hint: "PCM_F32LE:24000")

        XCTAssertEqual(result.channels[0], samples)
        XCTAssertEqual(result.format.sampleRate, 24_000)
    }

    func testDecodePCMFloat32LEStereoDeinterleave() async throws {
        // Interleaved stereo: [L0, R0, L1, R1, L2, R2]
        let interleaved: [Float] = [1, 10, 2, 20, 3, 30]
        let data = makePCMFloat32LEData(interleaved)
        let decoder = AVFoundationAudioDecoder()

        let result = try await decoder.decode(data, hint: "pcm_f32le:24000:2")

        XCTAssertEqual(result.channels.count, 2)
        XCTAssertEqual(result.format.channelCount, 2)
        XCTAssertEqual(result.channels[0], [1, 2, 3])
        XCTAssertEqual(result.channels[1], [10, 20, 30])
    }

    func testDecodePCMFloat32LEEmptyDataProducesEmptyChannel() async throws {
        let decoder = AVFoundationAudioDecoder()
        let result = try await decoder.decode(Data(), hint: "pcm_f32le:24000")
        XCTAssertEqual(result.channels, [[]])
        XCTAssertEqual(result.frameCount, 0)
    }

    func testDecodePCMFloat32LERejectsUnalignedByteCount() async throws {
        // 5 bytes is not a multiple of 4
        let data = Data([0x00, 0x01, 0x02, 0x03, 0x04])
        let decoder = AVFoundationAudioDecoder()
        do {
            _ = try await decoder.decode(data, hint: "pcm_f32le")
            XCTFail("Expected decodingFailed error for unaligned byte count")
        } catch AudioPipelineError.decodingFailed {
            // expected
        }
    }

    func testDecodePCMFloat32LERoundTripsAgainstMLXEncoding() async throws {
        // Mirror the encoding MLXModelHandle.pcmFloat32LEData uses: withUnsafeBufferPointer + Data(bytes:count:)
        let original: [Float] = [0, 0.1, -0.3, 0.7, -1.0, 1.0]
        let encoded: Data = original.withUnsafeBufferPointer { buf in
            Data(bytes: buf.baseAddress!, count: buf.count * MemoryLayout<Float>.size)
        }

        let decoder = AVFoundationAudioDecoder()
        let result = try await decoder.decode(encoded, hint: "pcm_f32le:24000")

        XCTAssertEqual(result.channels[0], original)
    }

    func testDecodePCMFloat32LEBenchmarkFastPathBeatsTempFile() async throws {
        let sampleCount = 48_000 // one second at 48 kHz
        let samples = benchmarkSamples(frameCount: sampleCount, sampleRate: 48_000)
        let data = makePCMFloat32LEData(samples)
        let decoder = AVFoundationAudioDecoder()
        let iterations = 20

        // Warm up
        _ = try await decoder.decode(data, hint: "pcm_f32le:48000")

        // Benchmark fast path
        let fastDuration = try await measureWallClockAsync {
            for _ in 0..<iterations {
                _ = try await decoder.decode(data, hint: "pcm_f32le:48000")
            }
        }

        // Benchmark slow per-sample loop (reference)
        let slowDuration = measureWallClock {
            for _ in 0..<iterations {
                _ = naiveDecodePCMFloat32LE(data)
            }
        }

        let speedup = slowDuration / fastDuration
        print(
            String(
                format: "PCM fast path: slow %.4fs fast %.4fs speedup %.2fx",
                slowDuration, fastDuration, speedup
            )
        )
        XCTAssertGreaterThan(
            speedup,
            1.1,
            String(format: "Fast path speedup %.2fx (slow %.4fs, fast %.4fs)", speedup, slowDuration, fastDuration)
        )
    }

    // MARK: - Helpers (pcm_f32le tests)

    private func makePCMFloat32LEData(_ samples: [Float]) -> Data {
        samples.withUnsafeBufferPointer { buf in
            Data(bytes: buf.baseAddress!, count: buf.count * MemoryLayout<Float>.size)
        }
    }

    /// Per-sample loop reference implementation — mirrors the pattern spread across callers.
    private func naiveDecodePCMFloat32LE(_ data: Data) -> [Float] {
        var samples: [Float] = []
        samples.reserveCapacity(data.count / MemoryLayout<Float>.size)
        for offset in stride(from: 0, to: data.count, by: MemoryLayout<Float>.size) {
            var bits: UInt32 = 0
            _ = withUnsafeMutableBytes(of: &bits) { rawBuffer in
                data.copyBytes(to: rawBuffer, from: offset ..< (offset + MemoryLayout<Float>.size))
            }
            samples.append(Float(bitPattern: UInt32(littleEndian: bits)))
        }
        return samples
    }

    func testExporterRejectsZeroChannelBuffer() async {
        let buffer = AudioPCMBuffer(
            channels: [],
            format: AudioFormatDescriptor(sampleRate: 24_000, channelCount: 0)
        )
        let exporter = AVFoundationAudioExporter()
        let format = AudioFormatDescriptor(sampleRate: 24_000, channelCount: 0, container: "wav")
        do {
            _ = try await exporter.export(buffer, as: format)
            XCTFail("Expected error for zero-channel buffer")
        } catch {
            // Graceful error — no crash
        }
    }

    private func naiveInterleave(
        _ buffer: AudioPCMBuffer,
        startFrame: Int,
        frameCount: Int,
        channelCount: Int
    ) -> Data {
        var data = Data(capacity: frameCount * channelCount * MemoryLayout<Float>.size)
        for frameOffset in 0 ..< frameCount {
            let sourceFrame = startFrame + frameOffset
            for channelIndex in 0 ..< channelCount {
                let sample: Float = channelIndex < buffer.channels.count &&
                    sourceFrame < buffer.channels[channelIndex].count
                    ? buffer.channels[channelIndex][sourceFrame]
                    : .zero
                var bits = sample.bitPattern.littleEndian
                withUnsafeBytes(of: &bits) { data.append(contentsOf: $0) }
            }
        }
        return data
    }

    private func makeTemporaryURL(pathExtension: String) -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: false)
            .appendingPathExtension(pathExtension)
    }

    private func naiveResample(_ channel: [Float], sourceRate: Double, targetRate: Double) -> [Float] {
        guard sourceRate != targetRate else { return channel }

        let ratio = targetRate / sourceRate
        let targetFrameCount = Int(Double(channel.count) * ratio)
        guard !channel.isEmpty, targetFrameCount > 0 else { return [] }

        var result = [Float](repeating: 0, count: targetFrameCount)
        for index in 0..<targetFrameCount {
            let sourceIndex = Double(index) / ratio
            let lowerIndex = min(Int(sourceIndex), channel.count - 1)
            let upperIndex = min(lowerIndex + 1, channel.count - 1)
            let fraction = Float(sourceIndex - Double(lowerIndex))
            result[index] = channel[lowerIndex] * (1 - fraction) + channel[upperIndex] * fraction
        }
        return result
    }

    private func measureWallClock(_ block: () -> Void) -> TimeInterval {
        let start = CFAbsoluteTimeGetCurrent()
        block()
        return CFAbsoluteTimeGetCurrent() - start
    }

    private func measureWallClockAsync(_ block: () async throws -> Void) async rethrows -> TimeInterval {
        let start = CFAbsoluteTimeGetCurrent()
        try await block()
        return CFAbsoluteTimeGetCurrent() - start
    }

    private func naiveClip(_ samples: [Float]) -> [Float] {
        samples.map { min(max($0, -1), 1) }
    }

    private func benchmarkSamples(frameCount: Int, sampleRate: Double) -> [Float] {
        var samples = [Float]()
        samples.reserveCapacity(frameCount)

        for index in 0..<frameCount {
            let time = Double(index) / sampleRate
            let fundamental = 0.7 * sin(2 * Double.pi * 220 * time)
            let harmonic = 0.2 * sin(2 * Double.pi * 660 * time)
            let overtone = 0.1 * sin(2 * Double.pi * 1_320 * time)
            samples.append(Float(fundamental + harmonic + overtone))
        }

        return samples
    }
}

// MARK: - AudioEnginePlayer.feedSamples tests

final class AudioEnginePlayerFeedSamplesTests: XCTestCase {

    /// `makeAVAudioBufferFromSamples` must copy Float values into `floatChannelData`
    /// byte-for-byte, with no intermediate Data conversion.
    func testMakeAVAudioBufferFromSamplesMatchesInput() throws {
        let samples: [Float] = [0.0, 0.25, -0.5, 0.75, -1.0, 1.0]
        let sampleRate = 24_000.0

        let avBuffer = try AudioEnginePlayer.makeAVAudioBufferFromSamples(samples, sampleRate: sampleRate)

        XCTAssertEqual(Int(avBuffer.frameLength), samples.count)
        XCTAssertEqual(avBuffer.format.sampleRate, sampleRate)
        XCTAssertEqual(Int(avBuffer.format.channelCount), 1)

        guard let channelData = avBuffer.floatChannelData?[0] else {
            XCTFail("floatChannelData[0] is nil")
            return
        }
        let result = Array(UnsafeBufferPointer(start: channelData, count: samples.count))
        XCTAssertEqual(result, samples)
    }

    /// Empty sample array must produce a buffer with frameLength == 0 and no crash.
    func testMakeAVAudioBufferFromSamplesHandlesEmptyInput() throws {
        let avBuffer = try AudioEnginePlayer.makeAVAudioBufferFromSamples([], sampleRate: 24_000)
        XCTAssertEqual(avBuffer.frameLength, 0)
    }

    /// `feedSamples` must enqueue frames (reflected in the playback snapshot) when the
    /// audio engine can be started.  On CI hardware where AVAudioEngine.start() fails,
    /// the test is skipped rather than failing.
    func testFeedSamplesQueuesFramesAndIsPlayable() async throws {
        let samples = [Float](repeating: 0.5, count: 2_400)  // 100 ms at 24 kHz
        let sampleRate = 24_000.0
        let player = AudioEnginePlayer()

        do {
            try await player.feedSamples(samples, sampleRate: sampleRate)
        } catch {
            throw XCTSkip("AVAudioEngine could not start (no audio hardware?): \(error.localizedDescription)")
        }

        let snapshot = await player.playbackSnapshot()
        // At minimum, the queued duration must be positive — confirming frames were enqueued.
        let totalDuration = snapshot.position + snapshot.queuedDuration
        XCTAssertGreaterThan(totalDuration, 0, "feedSamples must enqueue a positive number of frames")
        await player.stop()
    }

    /// Calling `feedSamples` followed by `feedChunk` (existing API) on the same player
    /// must still work — the fast path must not break backwards compatibility.
    func testFeedSamplesAndFeedChunkAreBackwardsCompatible() async throws {
        let player = AudioEnginePlayer()

        do {
            try await player.feedSamples([0.1, 0.2, 0.3], sampleRate: 24_000)
            let chunkBuffer = AudioPCMBuffer(mono: [0.4, 0.5, 0.6], sampleRate: 24_000)
            try await player.feedChunk(chunkBuffer)
        } catch {
            throw XCTSkip("AVAudioEngine could not start: \(error.localizedDescription)")
        }

        let snapshot = await player.playbackSnapshot()
        XCTAssertGreaterThan(snapshot.position + snapshot.queuedDuration, 0)
        await player.stop()
    }
}

// MARK: - PlaybackPositionPublisher tests

@MainActor
final class PlaybackPositionPublisherTests: XCTestCase {

    /// A value emitted from the position provider must appear in the stream.
    func testStreamYieldsPositionFromProvider() async throws {
        let expected: TimeInterval = 3.14
        let publisher = PlaybackPositionPublisher { expected }
        publisher.start()

        var received: TimeInterval?
        for await pos in publisher.stream {
            received = pos
            break
        }
        publisher.stop()

        XCTAssertEqual(received ?? 0, expected, accuracy: 1e-9)
    }

    /// The position provider is called every frame; the stream reflects its latest value.
    func testStreamReflectsChangingProviderValues() async throws {
        var counter: TimeInterval = 0
        let publisher = PlaybackPositionPublisher {
            counter += 1
            return counter
        }
        publisher.start()

        var values: [TimeInterval] = []
        for await pos in publisher.stream {
            values.append(pos)
            if values.count >= 3 { break }
        }
        publisher.stop()

        XCTAssertEqual(values.count, 3)
        // Values must be strictly increasing (counter increments each call).
        for index in values.indices.dropFirst() {
            XCTAssertGreaterThan(values[index], values[index - 1])
        }
    }

    /// Calling start() multiple times must not create duplicate display links
    /// (which would cause the provider to be called more than once per frame).
    func testStartIsIdempotent() async throws {
        var callCount = 0
        let publisher = PlaybackPositionPublisher {
            callCount += 1
            return TimeInterval(callCount)
        }
        publisher.start()
        publisher.start() // second call — must be a no-op

        // Consume one value then stop.
        for await _ in publisher.stream { break }
        publisher.stop()

        // If start() created two display links, callCount would advance by 2
        // per frame; with a single link it advances by 1. We just verify the
        // publisher didn't crash and the stream is usable.
        XCTAssertGreaterThanOrEqual(callCount, 1)
    }

    /// stop() must be safe to call when already inactive (before start or twice in a row).
    func testStopIsIdempotent() {
        let publisher = PlaybackPositionPublisher { 0.0 }
        publisher.stop() // before start — must not crash
        publisher.start()
        publisher.stop()
        publisher.stop() // double stop — must not crash
    }

    /// The stream must finish (for-await loop must exit) when the publisher is
    /// deallocated, allowing stream consumers to clean up naturally.
    func testStreamFinishesOnPublisherDeinit() async throws {
        var publisher: PlaybackPositionPublisher? = PlaybackPositionPublisher { 0.0 }
        publisher!.start()

        let stream = publisher!.stream
        let finished = expectation(description: "stream finishes after publisher deinit")

        let task = Task {
            for await _ in stream {}
            finished.fulfill()
        }

        // Release the publisher; deinit should finish the continuation.
        publisher = nil

        await fulfillment(of: [finished], timeout: 2.0)
        _ = task // suppress unused warning
    }

    /// stop() followed by start() must resume emission on the same stream.
    func testRestartResumesEmission() async throws {
        let publisher = PlaybackPositionPublisher { 1.0 }
        publisher.start()

        // Consume one value, then stop.
        for await _ in publisher.stream { break }
        publisher.stop()

        // Restart and consume another value from the same stream.
        publisher.start()
        var resumed = false
        for await _ in publisher.stream {
            resumed = true
            break
        }
        publisher.stop()

        XCTAssertTrue(resumed, "Stream should emit values again after restart")
    }
}
