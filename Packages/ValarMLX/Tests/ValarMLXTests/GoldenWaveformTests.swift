import Foundation
import Testing
@preconcurrency import MLX
import MLXNN
@testable import ValarMLX

/// Golden waveform comparison tests for the SpeechTokenizerDecoder.
///
/// Uses a small decoder config with deterministic weights (seeded pseudo-random
/// values) to produce repeatable output. The golden fixture stores expected
/// sample values; if the fixture is missing, the test generates it on first run.
///
/// This catches regressions in the decoder pipeline: VQ → pre_conv → transformer
/// → upsample → decoder blocks → clip. Any change to the forward pass math will
/// cause the golden comparison to fail.
@Suite("Golden Waveform Comparison")
struct GoldenWaveformTests {

    // MARK: - Fixture types

    struct GoldenWaveform: Codable {
        let config: String
        let inputCodes: [Int32]
        let inputShape: [Int]
        let expectedSamples: [Float]
        let expectedShape: [Int]
        let tolerance: Float
        let tag: String
    }

    // MARK: - Deterministic RNG

    /// SplitMix64 PRNG for deterministic weight initialization.
    ///
    /// Produces identical sequences across platforms and runs given the same seed.
    private struct SplitMix64: RandomNumberGenerator {
        var state: UInt64

        init(seed: UInt64) { self.state = seed }

        mutating func next() -> UInt64 {
            state &+= 0x9E3779B97F4A7C15
            var z = state
            z = (z ^ (z &>> 30)) &* 0xBF58476D1CE4E5B9
            z = (z ^ (z &>> 27)) &* 0x94D049BB133111EB
            return z ^ (z &>> 31)
        }
    }

    // MARK: - Configuration

    /// Small config matching SpeechTokenizerDecoderTests.smallConfig().
    ///
    /// decoderDim=48, upsampleRates=[2,3], upsamplingRatios=[2]
    /// → totalUpsample=12, outputDim=12, decoder has 5 elements.
    static func smallConfig() -> SpeechTokenizerDecoderConfig {
        SpeechTokenizerDecoderConfig(
            codebookDim: 16,
            codebookSize: 32,
            numQuantizers: 2,
            numSemanticQuantizers: 1,
            latentDim: 32,
            decoderDim: 48,
            upsampleRates: [2, 3],
            upsamplingRatios: [2],
            hiddenSize: 16,
            intermediateSize: 32,
            numHiddenLayers: 1,
            numAttentionHeads: 2,
            numKeyValueHeads: 2,
            headDim: 8,
            rmsNormEps: 1e-5,
            ropeTheta: 10000.0,
            slidingWindow: 8,
            layerScaleInitialScale: 0.01,
            attentionBias: false
        )
    }

    // MARK: - Deterministic decoder factory

    /// Creates a SpeechTokenizerDecoder with deterministic weights.
    ///
    /// All parameters are replaced with values from a seeded SplitMix64 PRNG
    /// in the range [-0.05, 0.05]. This ensures outputs are identical across
    /// runs and MLX library versions, independent of MLX's random state.
    static func makeDeterministicDecoder() throws -> SpeechTokenizerDecoder {
        let config = smallConfig()
        let decoder = SpeechTokenizerDecoder(config: config)

        var rng = SplitMix64(seed: 42)
        let params = decoder.parameters().flattened()
        let seeded = params.map { (key, value) -> (String, MLXArray) in
            let count = value.shape.reduce(1, *)
            let values = (0 ..< count).map { _ in
                Float.random(in: -0.05 ... 0.05, using: &rng)
            }
            return (key, MLXArray(values).reshaped(value.shape))
        }
        try decoder.update(
            parameters: ModuleParameters.unflattened(seeded),
            verify: .noUnusedKeys
        )

        return decoder
    }

    /// Standard input codes: [batch=1, numQuantizers=2, time=4].
    ///
    /// Uses varied codes so quantizer lookup is non-trivial.
    static let goldenInputCodes: [Int32] = [
        0, 1, 2, 3, // semantic quantizer
        4, 5, 6, 7, // acoustic quantizer
    ]

    static func inputCodes() -> MLXArray {
        MLXArray(goldenInputCodes).reshaped(1, 2, 4)
    }

    // MARK: - Fixture I/O

    static func fixtureURL() -> URL {
        let thisFile = URL(fileURLWithPath: #filePath)
        return thisFile
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures")
            .appendingPathComponent("waveform_golden.json")
    }

    static func loadFixture() throws -> GoldenWaveform {
        let data = try Data(contentsOf: fixtureURL())
        return try JSONDecoder().decode(GoldenWaveform.self, from: data)
    }

    static func saveFixture(_ golden: GoldenWaveform) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(golden)
        try data.write(to: fixtureURL())
    }

    /// Runs the deterministic decoder and returns flat output samples.
    static func runDeterministicDecode() throws -> [Float] {
        let decoder = try makeDeterministicDecoder()
        let codes = inputCodes()
        let output = decoder(codes)
        MLX.eval(output)
        return output.reshaped(-1).asArray(Float.self)
    }

    // MARK: - Golden Comparison Tests

    @Test("Golden fixture exists and has expected metadata")
    func fixtureMetadata() throws {
        let url = Self.fixtureURL()
        guard FileManager.default.fileExists(atPath: url.path) else {
            // Generate the fixture on first run
            let samples = try Self.runDeterministicDecode()
            let golden = GoldenWaveform(
                config: "small",
                inputCodes: Self.goldenInputCodes,
                inputShape: [1, 2, 4],
                expectedSamples: samples,
                expectedShape: [1, 1, 48],
                tolerance: 1e-5,
                tag: "small_decoder_splitmix64_seed42"
            )
            try Self.saveFixture(golden)
            Issue.record("Generated golden fixture at \(url.path) — re-run tests to validate")
            return
        }

        let golden = try Self.loadFixture()
        #expect(golden.config == "small")
        #expect(golden.inputShape == [1, 2, 4])
        #expect(golden.expectedShape == [1, 1, 48])
        #expect(golden.expectedSamples.count == 48)
        #expect(golden.tag == "small_decoder_splitmix64_seed42")
    }

    @Test("Full decode output matches golden fixture sample-by-sample")
    func goldenFullDecode() throws {
        let url = Self.fixtureURL()
        guard FileManager.default.fileExists(atPath: url.path) else {
            Issue.record("Golden fixture missing — run fixtureMetadata test first")
            return
        }

        let samples = try Self.runDeterministicDecode()
        let golden = try Self.loadFixture()

        #expect(
            samples.count == golden.expectedSamples.count,
            "Sample count: \(samples.count) vs expected \(golden.expectedSamples.count)"
        )

        for (i, (actual, expected)) in zip(samples, golden.expectedSamples).enumerated() {
            #expect(
                abs(actual - expected) < golden.tolerance,
                "Sample \(i): \(actual) vs \(expected) (delta \(abs(actual - expected)))"
            )
        }
    }

    @Test("Re-constructed deterministic decoder produces identical output")
    func deterministicReproducibility() throws {
        let samples1 = try Self.runDeterministicDecode()
        let samples2 = try Self.runDeterministicDecode()
        #expect(samples1 == samples2, "Two runs with identical seeded weights must match exactly")
    }

    @Test("Chunked decode matches full decode within tolerance")
    func chunkedMatchesFullDecode() throws {
        let decoder = try Self.makeDeterministicDecoder()
        let codes = Self.inputCodes()

        let full = decoder(codes)
        MLX.eval(full)
        let fullSamples = full.reshaped(-1).asArray(Float.self)

        let chunked = decoder.chunkedDecode(codes, chunkSize: 2, leftContextSize: 1)
        MLX.eval(chunked)
        let chunkedSamples = chunked.reshaped(-1).asArray(Float.self)

        #expect(
            fullSamples.count == chunkedSamples.count,
            "Chunked count \(chunkedSamples.count) vs full \(fullSamples.count)"
        )

        for (i, (f, c)) in zip(fullSamples, chunkedSamples).enumerated() {
            #expect(
                abs(f - c) < 1e-4,
                "Sample \(i): full=\(f) vs chunked=\(c) (delta \(abs(f - c)))"
            )
        }
    }

    @Test("Golden output shape is correct")
    func outputShape() throws {
        let decoder = try Self.makeDeterministicDecoder()
        let codes = Self.inputCodes()
        let output = decoder(codes)
        MLX.eval(output)
        // time=4, totalUpsample=12 → 48 samples
        #expect(output.shape == [1, 1, 48])
    }

    @Test("Golden output is clipped to [-1, 1] with no NaN or Inf")
    func outputRangeAndValidity() throws {
        let decoder = try Self.makeDeterministicDecoder()
        let codes = Self.inputCodes()
        let output = decoder(codes)
        MLX.eval(output)

        let flat = output.reshaped(-1).asArray(Float.self)
        let maxVal = flat.max() ?? 0
        let minVal = flat.min() ?? 0
        #expect(maxVal <= 1.0, "Max value \(maxVal) exceeds 1.0")
        #expect(minVal >= -1.0, "Min value \(minVal) below -1.0")

        let hasNan = flat.contains { $0.isNaN }
        let hasInf = flat.contains { $0.isInfinite }
        #expect(!hasNan, "Output contains NaN values")
        #expect(!hasInf, "Output contains Inf values")
    }

    @Test("Different input codes produce different golden output")
    func differentCodesProduceDifferentOutput() throws {
        let decoder = try Self.makeDeterministicDecoder()

        let codes1 = MLXArray([Int32](repeating: 0, count: 8)).reshaped(1, 2, 4)
        let out1 = decoder(codes1)
        MLX.eval(out1)
        let samples1 = out1.reshaped(-1).asArray(Float.self)

        let codes2 = Self.inputCodes()
        let out2 = decoder(codes2)
        MLX.eval(out2)
        let samples2 = out2.reshaped(-1).asArray(Float.self)

        #expect(samples1 != samples2, "Different codes should produce different waveforms")
    }
}
