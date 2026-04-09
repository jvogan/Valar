import Foundation
@preconcurrency import MLX
import MLXNN
import Testing
@testable import ValarMLX

// MARK: - Smoke Decode Tests
//
// Load real Qwen3-TTS speech tokenizer weights from the local HuggingFace cache
// and run a forward pass through the decoder. These tests validate that:
//   1. Weight loading succeeds with no unused keys
//   2. The decoder produces audio in the expected shape
//   3. Output is clipped to [-1, 1], contains no NaN/Inf, and is non-trivial
//
// Tests are skipped when the weight directory is not present (e.g., CI).

@Suite("Smoke Decode (Real Weights)")
struct SmokeDecodeTests {

    /// Well-known HuggingFace cache paths for the Qwen3-TTS speech tokenizer.
    private static let candidateDirectories: [String] = [
        // Standard HuggingFace Hub cache (symlinked snapshots)
        NSHomeDirectory() + "/.cache/huggingface/hub/models--mlx-community--Qwen3-TTS-12Hz-1.7B-Base-bf16/snapshots",
        // mlx-audio alternate cache layout
        NSHomeDirectory() + "/.cache/huggingface/hub/mlx-audio/mlx-community_Qwen3-TTS-12Hz-1.7B-Base-bf16",
    ]

    /// Resolve the speech_tokenizer directory, returning nil if not available.
    private static func findSpeechTokenizerDirectory() -> URL? {
        let fm = FileManager.default
        for candidate in candidateDirectories {
            let url = URL(fileURLWithPath: candidate, isDirectory: true)
            if candidate.hasSuffix("snapshots") {
                // HuggingFace snapshot layout — pick the first snapshot
                guard let contents = try? fm.contentsOfDirectory(
                    at: url, includingPropertiesForKeys: nil
                ) else { continue }
                for snapshot in contents {
                    let tokenizer = snapshot.appendingPathComponent("speech_tokenizer")
                    if fm.fileExists(atPath: tokenizer.path) {
                        return tokenizer
                    }
                }
            } else {
                let tokenizer = url.appendingPathComponent("speech_tokenizer")
                if fm.fileExists(atPath: tokenizer.path) {
                    return tokenizer
                }
            }
        }
        return nil
    }

    // MARK: - Weight Loading

    @Test("Load real weights without unused keys")
    func loadWeightsNoUnusedKeys() throws {
        guard let dir = Self.findSpeechTokenizerDirectory() else {
            print("Skipping: speech_tokenizer weights not found on disk")
            return
        }

        let decoder = SpeechTokenizerDecoder()
        let result = try decoder.loadWeights(from: dir)

        // All decoder keys consumed — no unused keys (enforced by .noUnusedKeys verify)
        // Check that the loader found safetensors files
        let rawKeys = result.weights.keys.filter { $0.hasPrefix("decoder.") }
        #expect(rawKeys.count > 200, "Expected 200+ decoder weight keys, got \(rawKeys.count)")
    }

    // MARK: - Forward Pass

    @Test("Smoke decode: random codes produce valid audio waveform")
    func smokeDecodeRandomCodes() throws {
        guard let dir = Self.findSpeechTokenizerDirectory() else {
            print("Skipping: speech_tokenizer weights not found on disk")
            return
        }

        let decoder = SpeechTokenizerDecoder()
        try decoder.loadWeights(from: dir)

        let config = SpeechTokenizerDecoderConfig.default
        let batch = 1
        let numQ = config.numQuantizers
        let time = 10

        // Random codes in valid range [0, codebookSize)
        let codes = MLXArray(
            (0 ..< batch * numQ * time).map { _ in
                Int32.random(in: 0 ..< Int32(config.codebookSize))
            }
        ).reshaped(batch, numQ, time)

        let audio = decoder(codes)
        MLX.eval(audio)

        // Shape: [batch, 1, time * totalUpsample]
        let expectedSamples = time * config.totalUpsample
        #expect(audio.shape == [batch, 1, expectedSamples],
                "Expected [1, 1, \(expectedSamples)], got \(audio.shape)")

        // Range: clipped to [-1, 1]
        let flat = audio.reshaped(-1).asArray(Float.self)
        let minVal = flat.min() ?? 0
        let maxVal = flat.max() ?? 0
        #expect(minVal >= -1.0, "Output min \(minVal) < -1.0")
        #expect(maxVal <= 1.0, "Output max \(maxVal) > 1.0")

        // No NaN or Inf
        let hasNaN = flat.contains { $0.isNaN }
        let hasInf = flat.contains { $0.isInfinite }
        #expect(!hasNaN, "Output contains NaN")
        #expect(!hasInf, "Output contains Inf")

        // Non-trivial: with real weights and random codes, output shouldn't be all zeros
        let hasNonZero = flat.contains { abs($0) > 1e-6 }
        #expect(hasNonZero, "Output is all zeros — weights may not have loaded correctly")
    }

    @Test("Smoke decode: semantic-only codes (1 quantizer)")
    func smokeDecodeSemanticOnly() throws {
        guard let dir = Self.findSpeechTokenizerDirectory() else {
            print("Skipping: speech_tokenizer weights not found on disk")
            return
        }

        let decoder = SpeechTokenizerDecoder()
        try decoder.loadWeights(from: dir)

        let config = SpeechTokenizerDecoderConfig.default
        let batch = 1
        let time = 8

        // Only semantic quantizer (1 code layer)
        let codes = MLXArray(
            (0 ..< batch * 1 * time).map { _ in
                Int32.random(in: 0 ..< Int32(config.codebookSize))
            }
        ).reshaped(batch, 1, time)

        let audio = decoder(codes)
        MLX.eval(audio)

        let expectedSamples = time * config.totalUpsample
        #expect(audio.shape == [batch, 1, expectedSamples])

        let flat = audio.reshaped(-1).asArray(Float.self)
        let hasNaN = flat.contains { $0.isNaN }
        #expect(!hasNaN, "Semantic-only decode produced NaN")
    }

    @Test("Chunked decode matches single-pass decode for short input")
    func chunkedDecodeMatchesSinglePass() throws {
        guard let dir = Self.findSpeechTokenizerDirectory() else {
            print("Skipping: speech_tokenizer weights not found on disk")
            return
        }

        let decoder = SpeechTokenizerDecoder()
        try decoder.loadWeights(from: dir)

        let config = SpeechTokenizerDecoderConfig.default
        let batch = 1
        let numQ = config.numQuantizers
        let time = 20

        let codes = MLXArray(
            (0 ..< batch * numQ * time).map { _ in
                Int32.random(in: 0 ..< Int32(config.codebookSize))
            }
        ).reshaped(batch, numQ, time)

        // Single pass
        let singlePass = decoder(codes)
        MLX.eval(singlePass)

        // Chunked with chunk larger than input (should be identical)
        let chunked = decoder.chunkedDecode(codes, chunkSize: 300, leftContextSize: 25)
        MLX.eval(chunked)

        #expect(singlePass.shape == chunked.shape,
                "Shape mismatch: single \(singlePass.shape) vs chunked \(chunked.shape)")

        // Values should be identical when chunk > total frames
        let sFlat = singlePass.reshaped(-1).asArray(Float.self)
        let cFlat = chunked.reshaped(-1).asArray(Float.self)
        #expect(sFlat.count == cFlat.count)

        var maxDiff: Float = 0
        for i in 0 ..< sFlat.count {
            maxDiff = max(maxDiff, abs(sFlat[i] - cFlat[i]))
        }
        #expect(maxDiff < 1e-5, "Single-pass and chunked outputs differ by \(maxDiff)")
    }

    @Test("Decode output sample count at 24 kHz")
    func decodeSampleRate() throws {
        guard let dir = Self.findSpeechTokenizerDirectory() else {
            print("Skipping: speech_tokenizer weights not found on disk")
            return
        }

        let decoder = SpeechTokenizerDecoder()
        try decoder.loadWeights(from: dir)

        let config = SpeechTokenizerDecoderConfig.default

        // totalUpsample = 8*5*4*3 * 2*2 = 1920
        // At ~12.5 Hz code rate, this yields ~24 kHz audio
        let totalUpsample = config.totalUpsample
        #expect(totalUpsample == 1920,
                "Expected totalUpsample=1920 (8*5*4*3 * 2*2), got \(totalUpsample)")

        let codeFrames = 12
        let codes = MLXArray(
            Array(repeating: Int32(0), count: config.numQuantizers * codeFrames)
        ).reshaped(1, config.numQuantizers, codeFrames)

        let audio = decoder(codes)
        MLX.eval(audio)

        let expectedSamples = codeFrames * totalUpsample
        #expect(audio.dim(2) == expectedSamples,
                "Expected \(expectedSamples) samples for \(codeFrames) code frames")
    }
}
