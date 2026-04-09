import Foundation
import Testing
import MLX
import MLXLMCommon

@testable import MLXAudioTTS

@Suite("Qwen3TTS Unit Tests")
struct Qwen3TTSUnitTests {
    private func tinyTalkerConfig() throws -> Qwen3TTSTalkerConfig {
        try JSONDecoder().decode(
            Qwen3TTSTalkerConfig.self,
            from: Data(
                """
                {
                  "vocab_size": 32,
                  "hidden_size": 12,
                  "intermediate_size": 24,
                  "num_hidden_layers": 0,
                  "num_attention_heads": 2,
                  "num_key_value_heads": 1,
                  "head_dim": 6,
                  "text_hidden_size": 12,
                  "text_vocab_size": 16
                }
                """.utf8
            )
        )
    }

    @Test("Repetition window keeps only the configured suffix")
    func repetitionWindowCapsHistory() {
        var recentTokens: [Int] = []
        var recentUniqueTokens: [Int] = []
        var recentUniqueTokenSet = Set<Int>()

        for token in [1, 2, 3, 2, 4, 5] {
            Qwen3TTSModel.updateRepetitionWindow(
                tokenId: token,
                repetitionContextSize: 3,
                recentTokens: &recentTokens,
                recentUniqueTokens: &recentUniqueTokens,
                recentUniqueTokenSet: &recentUniqueTokenSet
            )
        }

        #expect(recentTokens == [2, 4, 5])
        #expect(recentUniqueTokens == [2, 4, 5])
        #expect(recentUniqueTokenSet == Set([2, 4, 5]))
    }

    @Test("Zero repetition window disables history tracking")
    func repetitionWindowCanBeDisabled() {
        var recentTokens = [7, 8]
        var recentUniqueTokens = [7, 8]
        var recentUniqueTokenSet: Set<Int> = [7, 8]

        Qwen3TTSModel.updateRepetitionWindow(
            tokenId: 9,
            repetitionContextSize: 0,
            recentTokens: &recentTokens,
            recentUniqueTokens: &recentUniqueTokens,
            recentUniqueTokenSet: &recentUniqueTokenSet
        )

        #expect(recentTokens.isEmpty)
        #expect(recentUniqueTokens.isEmpty)
        #expect(recentUniqueTokenSet.isEmpty)
    }

    @Test("Reduced-candidate top-p path keeps the dominant top-k token")
    func reducedCandidateTopPSamplingCanCollapseToSingleToken() throws {
        let logits = MLXArray([Float(10.0), 9.0, 0.0, -1.0]).reshaped(1, 1, 4)

        let token = try #require(
            Qwen3TTSModel.reducedCandidateSampleToken(
            logits,
            temperature: 0.9,
            topP: 0.5,
            topK: 2,
            minP: 0.0
            )
        )

        #expect(token[0, 0].item(Int32.self) == 0)
    }

    @Test("Reduced-candidate min-p path keeps only the strongest top-k token when threshold is high")
    func reducedCandidateMinPSamplingCanCollapseToSingleToken() throws {
        let logits = MLXArray([Float(10.0), 5.0, 4.0, -1.0]).reshaped(1, 1, 4)

        let token = try #require(
            Qwen3TTSModel.reducedCandidateSampleToken(
            logits,
            temperature: 0.9,
            topP: 1.0,
            topK: 3,
            minP: 0.5
            )
        )

        #expect(token[0, 0].item(Int32.self) == 0)
    }

    @Test("Reduced-candidate sampling keeps the fast path when EOS must remain eligible")
    func reducedCandidateSamplingCanRetainEOSWithoutLeavingFastPath() throws {
        let logits = MLXArray([Float(10.0), 9.0, -50.0, 0.0]).reshaped(1, 1, 4)

        let token = try #require(
            Qwen3TTSModel.reducedCandidateSampleToken(
                logits,
                temperature: 0.9,
                topP: 0.5,
                topK: 1,
                minP: 0.0,
                eosTokenId: 1
            )
        )

        #expect(token[0, 0].item(Int32.self) == 0)
    }

    @Test("Restoring an empty cache snapshot resets reusable KV caches to offset zero")
    func restoringEmptySnapshotResetsTrimmableCaches() {
        var caches: [any KVCache] = [KVCacheSimple(), KVCacheSimple()]
        let emptySnapshot = snapshotKVCaches(caches)
        let keys = MLXArray.zeros([1, 1, 2, 1], dtype: .float32)
        let values = MLXArray.zeros([1, 1, 2, 1], dtype: .float32)

        _ = caches[0].update(keys: keys, values: values)
        _ = caches[1].update(keys: keys, values: values)

        #expect(caches.allSatisfy { $0.offset == 2 })

        restoreKVCaches(&caches, from: emptySnapshot)

        #expect(caches.allSatisfy { $0.offset == 0 })
    }

    @Test("Resetting KV caches for reuse rewinds simple cache offsets")
    func resettingKVCachesForReuseRewindsOffsets() throws {
        let first = KVCacheSimple()
        let second = KVCacheSimple()
        first.offset = 64
        second.offset = 17
        var caches: [any KVCache] = [first, second]

        resetKVCachesForReuse(&caches)

        #expect(try #require(caches[0] as? KVCacheSimple).offset == 0)
        #expect(try #require(caches[1] as? KVCacheSimple).offset == 0)
    }

    @Test("Prepared talker generation positions expose single-token slices with the expected offset")
    func preparedTalkerGenerationPositionsPreservePerIndexSlices() throws {
        guard mlxRuntimeReadyForCurrentProcess() else {
            return
        }
        let config = try tinyTalkerConfig()
        let model = Qwen3TTSTalkerModel(config: config)
        let prepared = try #require(
            model.prepareGenerationPositions(
                startOffset: 7,
                count: 3,
                dtype: .float32
            )
        )

        #expect(prepared.cos.shape == [1, 1, 3, config.headDim])
        #expect(prepared.sin.shape == [1, 1, 3, config.headDim])
        #expect(prepared[safe: -1] == nil)
        #expect(prepared[safe: 3] == nil)

        let positions = MLXArray(Int32(7) ..< Int32(10)).reshaped(1, 3)
        let positionIDs = stacked([positions, positions, positions], axis: 0)
        let placeholder = MLXArray.zeros([1, 3, 1], dtype: .float32)
        let (expectedCos, expectedSin) = model.rotaryEmb(placeholder, positionIds: positionIDs)
        let expandedExpectedCos = expandedDimensions(expectedCos, axis: 1)
        let expandedExpectedSin = expandedDimensions(expectedSin, axis: 1)

        #expect(prepared.cos.asArray(Float.self) == expandedExpectedCos.asArray(Float.self))
        #expect(prepared.sin.asArray(Float.self) == expandedExpectedSin.asArray(Float.self))

        let first = try #require(prepared[safe: 0])
        let second = try #require(prepared[safe: 1])
        #expect(first.0.shape == [1, 1, 1, config.headDim])
        #expect(first.1.shape == [1, 1, 1, config.headDim])
        #expect(first.0.asArray(Float.self) != second.0.asArray(Float.self))
    }

    @Test("Prepared talker generation positions return nil for empty ranges")
    func preparedTalkerGenerationPositionsRejectEmptyRanges() throws {
        guard mlxRuntimeReadyForCurrentProcess() else {
            return
        }
        let config = try tinyTalkerConfig()
        let model = Qwen3TTSTalkerModel(config: config)
        #expect(model.prepareGenerationPositions(startOffset: 0, count: 0, dtype: .float32) == nil)
    }
}
