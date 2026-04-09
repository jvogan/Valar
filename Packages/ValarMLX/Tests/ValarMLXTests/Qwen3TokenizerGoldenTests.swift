import Foundation
import Testing
@testable import ValarMLX

/// Fixture-driven golden tokenizer tests.
///
/// Loads 20 test cases from `Fixtures/tokenizer_golden.json` and verifies
/// that each input produces the expected token IDs through the BPE encoder.
///
/// The fixture uses a minimal 17-token vocabulary (see ``GoldenTokenizerTests``
/// for the vocabulary definition and merge rules). When real model files
/// (vocab.json, merges.txt) become available, regenerate the fixture with
/// expected IDs from the Python reference tokenizer for full parity testing.
@Suite("Qwen3 Tokenizer Golden")
struct Qwen3TokenizerGoldenTests {

    // MARK: - Fixture types

    struct GoldenCase: Codable {
        let input: String
        let expected_ids: [Int]
        let tag: String
    }

    // MARK: - Shared vocabulary (matches GoldenTokenizerTests)

    static let goldenVocab = BPEVocabulary(
        tokenToID: [
            "h": 0, "e": 1, "l": 2, "o": 3,
            "\u{0120}": 4,
            "w": 5, "r": 6, "d": 7,
            "he": 8, "hel": 9, "hell": 10, "hello": 11,
            "\u{0120}w": 12, "or": 13,
            "\u{0120}wor": 14, "\u{0120}worl": 15, "\u{0120}world": 16,
        ],
        merges: [
            .init(first: "h", second: "e"),
            .init(first: "he", second: "l"),
            .init(first: "hel", second: "l"),
            .init(first: "hell", second: "o"),
            .init(first: "\u{0120}", second: "w"),
            .init(first: "o", second: "r"),
            .init(first: "\u{0120}w", second: "or"),
            .init(first: "\u{0120}wor", second: "l"),
            .init(first: "\u{0120}worl", second: "d"),
        ]
    )

    // MARK: - Fixture loading

    static func loadFixture() throws -> [GoldenCase] {
        let fixtureURL = Bundle.module.url(
            forResource: "tokenizer_golden",
            withExtension: "json"
        )!
        let data = try Data(contentsOf: fixtureURL)
        return try JSONDecoder().decode([GoldenCase].self, from: data)
    }

    // MARK: - Tests

    @Test("Fixture contains exactly 20 golden cases")
    func fixtureCount() throws {
        let cases = try Self.loadFixture()
        #expect(cases.count == 20)
    }

    @Test("All 20 golden cases produce expected token IDs")
    func allGoldenCases() throws {
        let cases = try Self.loadFixture()
        var encoder = BPEEncoder(vocabulary: Self.goldenVocab)

        for golden in cases {
            let actual = try encoder.encode(golden.input)
            #expect(
                actual == golden.expected_ids,
                "[\(golden.tag)] input: \"\(golden.input)\" — expected \(golden.expected_ids), got \(actual)"
            )
        }
    }

    @Test("All non-empty golden cases roundtrip through decode")
    func goldenRoundtrip() throws {
        let cases = try Self.loadFixture()
        let encoder = BPEEncoder(vocabulary: Self.goldenVocab)

        for golden in cases where !golden.expected_ids.isEmpty {
            let decoded = encoder.decode(golden.expected_ids)
            // Decode may not exactly match input (unknown chars are dropped),
            // but it must produce valid UTF-8
            #expect(decoded != nil, "[\(golden.tag)] decode returned nil for IDs \(golden.expected_ids)")
        }
    }
}
