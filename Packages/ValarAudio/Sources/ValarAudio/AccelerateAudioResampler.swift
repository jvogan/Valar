import Accelerate
import Foundation
import os

public struct AccelerateAudioResampler: AudioResampler {
    private static let integerRatioTolerance = 1e-12
    private static let minimumFilterLength = 32
    private static let maximumFilterLength = 128

    public init() {}

    public func resample(_ buffer: AudioPCMBuffer, to targetRate: Double) async throws -> AudioPCMBuffer {
        guard buffer.format.sampleRate != targetRate else { return buffer }

        let ratio = targetRate / buffer.format.sampleRate
        let targetFrameCount = max(Int(Double(buffer.frameCount) * ratio), 0)
        let sourceRate = buffer.format.sampleRate
        let sharedOffsets = ratio == 1 ? [] : Self.makeOffsets(count: targetFrameCount, ratio: ratio)
        let downsamplePlan = ratio < 1 ? Self.makeDownsamplePlan(sourceRate: sourceRate, targetRate: targetRate) : nil

        let resampledChannels = buffer.channels.map { channel -> [Float] in
            Self.resampleChannel(
                channel,
                targetFrameCount: targetFrameCount,
                ratio: ratio,
                sharedOffsets: sharedOffsets,
                downsamplePlan: downsamplePlan
            )
        }

        return AudioPCMBuffer(
            channels: resampledChannels,
            format: AudioFormatDescriptor(
                sampleRate: targetRate,
                channelCount: buffer.format.channelCount,
                sampleFormat: .float32,
                interleaved: false,
                container: buffer.format.container
            )
        )
    }
}

private extension AccelerateAudioResampler {
    struct FilterCacheKey: Hashable, Sendable {
        let overallRatio: Double
        let decimationFactor: Int
    }

    // OSAllocatedUnfairLock is Sendable; the dictionary value type ([Float]) is Sendable.
    // Swift 6: static let of a Sendable type is permitted without actor isolation.
    static let filterCache = OSAllocatedUnfairLock(initialState: [FilterCacheKey: [Float]]())

    static func cachedLowPassFilter(overallRatio: Double, decimationFactor: Int) -> [Float] {
        let key = FilterCacheKey(overallRatio: overallRatio, decimationFactor: decimationFactor)

        if let hit = filterCache.withLock({ $0[key] }) {
            return hit
        }

        // Compute outside the lock; two simultaneous misses produce identical values,
        // so the second store is harmless.
        let filter = makeLowPassFilter(overallRatio: overallRatio, decimationFactor: decimationFactor)
        filterCache.withLock { $0[key] = filter }
        return filter
    }

    struct DownsamplePlan {
        let exactFactor: Int?
        let decimationFactor: Int
        let filter: [Float]
        let residualRatio: Double
    }

    static func resampleChannel(
        _ channel: [Float],
        targetFrameCount: Int,
        ratio: Double,
        sharedOffsets: [Float],
        downsamplePlan: DownsamplePlan?
    ) -> [Float] {
        guard !channel.isEmpty else { return [] }
        guard targetFrameCount > 0 else { return [] }
        guard channel.count > 1 else {
            return [Float](repeating: channel[0], count: targetFrameCount)
        }

        if ratio > 1 {
            return linearResample(channel, targetFrameCount: targetFrameCount, ratio: ratio, sharedOffsets: sharedOffsets)
        }

        guard let downsamplePlan else {
            return linearResample(channel, targetFrameCount: targetFrameCount, ratio: ratio, sharedOffsets: sharedOffsets)
        }

        if let exactFactor = downsamplePlan.exactFactor {
            return desample(
                channel,
                decimationFactor: exactFactor,
                filter: [1],
                outputCount: targetFrameCount
            )
        }

        let filteredCount = max(1, Int(ceil(Double(channel.count) / Double(downsamplePlan.decimationFactor))))
        let filtered = desample(
            channel,
            decimationFactor: downsamplePlan.decimationFactor,
            filter: downsamplePlan.filter,
            outputCount: filteredCount
        )

        if downsamplePlan.residualRatio == 1 {
            if filtered.count == targetFrameCount {
                return filtered
            }
            return linearResample(
                filtered,
                targetFrameCount: targetFrameCount,
                ratio: Double(targetFrameCount) / Double(filtered.count),
                sharedOffsets: makeOffsets(
                    count: targetFrameCount,
                    ratio: Double(targetFrameCount) / Double(filtered.count)
                )
            )
        }

        return linearResample(
            filtered,
            targetFrameCount: targetFrameCount,
            ratio: downsamplePlan.residualRatio,
            sharedOffsets: makeOffsets(count: targetFrameCount, ratio: downsamplePlan.residualRatio)
        )
    }

    static func makeDownsamplePlan(sourceRate: Double, targetRate: Double) -> DownsamplePlan {
        let inverseRatio = sourceRate / targetRate
        let roundedFactor = Int(inverseRatio.rounded())

        if roundedFactor > 1,
           abs(inverseRatio - Double(roundedFactor)) <= integerRatioTolerance
        {
            return DownsamplePlan(
                exactFactor: roundedFactor,
                decimationFactor: roundedFactor,
                filter: [1],
                residualRatio: 1
            )
        }

        let decimationFactor = max(1, Int(floor(inverseRatio)))
        let residualRatio = targetRate / (sourceRate / Double(decimationFactor))
        return DownsamplePlan(
            exactFactor: nil,
            decimationFactor: decimationFactor,
            filter: cachedLowPassFilter(overallRatio: targetRate / sourceRate, decimationFactor: decimationFactor),
            residualRatio: residualRatio
        )
    }

    static func linearResample(
        _ samples: [Float],
        targetFrameCount: Int,
        ratio: Double,
        sharedOffsets: [Float]
    ) -> [Float] {
        guard targetFrameCount > 0 else { return [] }
        guard !samples.isEmpty else { return [] }
        guard samples.count > 1 else {
            return [Float](repeating: samples[0], count: targetFrameCount)
        }

        let maxSourceIndex = Float(samples.count - 1)
        let offsets: [Float]
        if sharedOffsets.count == targetFrameCount {
            if let lastOffset = sharedOffsets.last, lastOffset > maxSourceIndex {
                offsets = sharedOffsets.map { min($0, maxSourceIndex) }
            } else {
                offsets = sharedOffsets
            }
        } else {
            offsets = makeOffsets(count: targetFrameCount, ratio: ratio).map { min($0, maxSourceIndex) }
        }

        let paddedSamples = samples + [samples[samples.count - 1]]
        var result = [Float](repeating: 0, count: targetFrameCount)

        paddedSamples.withUnsafeBufferPointer { sampleBuffer in
            offsets.withUnsafeBufferPointer { offsetBuffer in
                result.withUnsafeMutableBufferPointer { resultBuffer in
                    guard let sampleBase = sampleBuffer.baseAddress,
                          let offsetBase = offsetBuffer.baseAddress,
                          let resultBase = resultBuffer.baseAddress else {
                        return
                    }

                    vDSP_vlint(
                        sampleBase,
                        offsetBase,
                        1,
                        resultBase,
                        1,
                        vDSP_Length(targetFrameCount),
                        vDSP_Length(paddedSamples.count)
                    )
                }
            }
        }

        return result
    }

    static func desample(
        _ samples: [Float],
        decimationFactor: Int,
        filter: [Float],
        outputCount: Int
    ) -> [Float] {
        guard outputCount > 0 else { return [] }
        guard !samples.isEmpty else { return [] }

        let leadingPadding = max((filter.count - 1) / 2, 0)
        let minimumTrailingPadding = max(filter.count - 1 - leadingPadding, 0)
        let requiredInputCount = ((outputCount - 1) * decimationFactor) + filter.count
        let trailingPadding = max(
            minimumTrailingPadding,
            requiredInputCount - leadingPadding - samples.count
        )
        let paddedSamples = pad(
            samples,
            leadingCount: leadingPadding,
            trailingCount: trailingPadding
        )
        var result = [Float](repeating: 0, count: outputCount)

        paddedSamples.withUnsafeBufferPointer { sampleBuffer in
            filter.withUnsafeBufferPointer { filterBuffer in
                result.withUnsafeMutableBufferPointer { resultBuffer in
                    guard let sampleBase = sampleBuffer.baseAddress,
                          let filterBase = filterBuffer.baseAddress,
                          let resultBase = resultBuffer.baseAddress else {
                        return
                    }

                    vDSP_desamp(
                        sampleBase,
                        vDSP_Stride(decimationFactor),
                        filterBase,
                        resultBase,
                        vDSP_Length(outputCount),
                        vDSP_Length(filter.count)
                    )
                }
            }
        }

        return result
    }

    static func makeOffsets(count: Int, ratio: Double) -> [Float] {
        guard count > 0 else { return [] }

        var start: Float = 0
        var increment = Float(1 / ratio)
        var offsets = [Float](repeating: 0, count: count)
        vDSP_vramp(&start, &increment, &offsets, 1, vDSP_Length(count))
        return offsets
    }

    static func makeLowPassFilter(overallRatio: Double, decimationFactor: Int) -> [Float] {
        let tapCount = min(
            max(decimationFactor * 16, minimumFilterLength),
            maximumFilterLength
        )
        let cutoff = 0.5 * min(max(overallRatio, 0), 1)
        let center = Double(tapCount - 1) / 2
        var taps = [Float](repeating: 0, count: tapCount)

        for index in taps.indices {
            let position = Double(index) - center
            let sinc: Double
            if abs(position) < .ulpOfOne {
                sinc = 2 * cutoff
            } else {
                sinc = sin(2 * Double.pi * cutoff * position) / (Double.pi * position)
            }

            let windowPosition = Double(index) / Double(tapCount - 1)
            let window =
                0.42
                - (0.5 * cos(2 * Double.pi * windowPosition))
                + (0.08 * cos(4 * Double.pi * windowPosition))
            taps[index] = Float(sinc * window)
        }

        let gain = taps.reduce(0, +)
        guard gain != 0 else { return taps }
        return taps.map { $0 / gain }
    }

    static func pad(_ samples: [Float], leadingCount: Int, trailingCount: Int) -> [Float] {
        guard let firstSample = samples.first, let lastSample = samples.last else { return samples }

        return [Float](repeating: firstSample, count: max(leadingCount, 0))
            + samples
            + [Float](repeating: lastSample, count: max(trailingCount, 0))
    }
}
