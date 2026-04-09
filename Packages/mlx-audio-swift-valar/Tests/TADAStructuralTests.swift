import Foundation
import Testing
@testable import MLXAudioTTS
@testable import MLXAudioCodecs
import MLX

@Suite("TADA Structural Tests", .serialized)
struct TADAStructuralTests {

    @Test func grayCodeRoundTripsDurations() {
        let values = MLXArray([Int32(0), 1, 2, 7, 31, 255]).reshaped([1, 6])
        let encoded = TADAGrayCodeDurationCodec.encode(values)
        let decoded = TADAGrayCodeDurationCodec.decode(encoded)

        #expect(decoded.shape == values.shape)
        #expect(decoded.asArray(Int32.self) == values.asArray(Int32.self))
    }

    @Test func encoderProducesTokenAlignedEmbeddings() {
        let encoder = TADATTSEncoder()
        let audio = MLXRandom.normal([1, 4_800, 1])
        let tokenMask = MLXArray.zeros([1, 12], dtype: .int32)
        let tokenPositions = MLXArray([Int32(1), 3, 5, 7]).reshaped([1, 4])

        let frameHidden = encoder.frameEncode(audio, tokenMask: tokenMask)
        #expect(frameHidden.shape[0] == 1)
        #expect(frameHidden.shape[1] == 12)
        #expect(frameHidden.shape[2] == 1_024)

        let tokenValues = encoder(audio, tokenPositions: tokenPositions, tokenMask: tokenMask)
        #expect(tokenValues.shape == [1, 4, 512])
    }

    @Test func decoderExpandsTokenDurationsAndUpsamplesWaveform() {
        let decoder = TADATTSDecoder()
        let tokenValues = MLXRandom.normal([1, 4, 512])
        let timeBefore = MLXArray([Int32(1), 2, 1, 3]).reshaped([1, 4])
        let timeAfter = MLXArray([Int32(0), 1, 0, 0]).reshaped([1, 4])

        let expanded = decoder.expandTokenFeatures(tokenValues, timeBefore: timeBefore, timeAfter: timeAfter)
        #expect(expanded.values.shape == [1, 8, 512])
        #expect(expanded.frameMask.shape == [1, 8])

        let waveform = decoder(tokenValues, timeBefore: timeBefore, timeAfter: timeAfter)
        #expect(waveform.shape[0] == 1)
        #expect(waveform.shape[2] == 1)
        #expect(waveform.shape[1] >= 8 * 480)
    }

    @Test func vibeVoiceProducesAcousticAndDurationOutputs() {
        let model = TADAVibeVoice(hiddenSize: 64, acousticDim: 16, headLayers: 2)
        let noise = MLXRandom.normal([1, 5, 32])
        let conditioning = MLXRandom.normal([1, 5, 64])

        let sample = model.solve(
            noise: noise,
            conditioning: conditioning,
            numSteps: 4,
            cfgSchedule: .constant
        )

        #expect(sample.acoustic.shape == [1, 5, 16])
        #expect(sample.timeBefore.shape == [1, 5])
        #expect(sample.timeAfter.shape == [1, 5])
        #expect(sample.latent.shape == [1, 5, 32])
    }

    @Test func sanitizeStripsEncoderAndDecoderPrefixes() {
        let encoderWeights = TADATTSEncoder.sanitize(weights: [
            "encoder.hidden_linear.weight": MLXArray.zeros([512, 1_024]),
            "encoder.wav_encoder.conv.parametrizations.weight.original0": MLXArray.zeros([64, 1, 1]),
            "encoder.wav_encoder.conv.parametrizations.weight.original1": MLXArray.zeros([64, 7, 1]),
        ])

        #expect(encoderWeights["hidden_linear.weight"] != nil)
        #expect(encoderWeights["wav_encoder.conv.weight_g"] != nil)
        #expect(encoderWeights["wav_encoder.conv.weight_v"] != nil)

        let decoderWeights = TADATTSDecoder.sanitize(weights: [
            "decoder.decoder_proj.weight": MLXArray.zeros([1_024, 512]),
            "decoder.wav_decoder.conv.parametrizations.weight.original0": MLXArray.zeros([1_536, 1, 1]),
            "decoder.wav_decoder.conv.parametrizations.weight.original1": MLXArray.zeros([1_536, 7, 1_024]),
        ])

        #expect(decoderWeights["decoder_proj.weight"] != nil)
        #expect(decoderWeights["wav_decoder.conv.weight_g"] != nil)
        #expect(decoderWeights["wav_decoder.conv.weight_v"] != nil)
    }
}
