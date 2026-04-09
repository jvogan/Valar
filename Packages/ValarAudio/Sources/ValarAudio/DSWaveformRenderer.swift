import Accelerate

public struct DSWaveformRenderer {
    public init() {}

    public func waveformSamples(from buffer: AudioPCMBuffer, bucketCount: Int = 200) -> [Float] {
        guard let firstChannel = buffer.channels.first, !firstChannel.isEmpty else { return [] }
        let samplesPerBucket = max(1, firstChannel.count / bucketCount)

        return stride(from: 0, to: firstChannel.count, by: samplesPerBucket).map { start in
            let end = min(start + samplesPerBucket, firstChannel.count)
            let slice = firstChannel[start..<end]
            return vDSP.maximumMagnitude(slice)
        }
    }
}
