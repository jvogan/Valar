import Foundation
import Testing
@testable import ValarMLX

/// Golden tokenizer test corpus: 20 hand-verified cases covering every layer
/// of the BPE tokenization pipeline (pre-tokenizer, vocabulary, encoder,
/// special tokens, and chat template).
@Suite("Golden Tokenizer Corpus")
struct GoldenTokenizerTests {

    // MARK: - Shared fixtures

    /// Minimal BPE vocabulary for golden tests.
    ///
    /// Covers "hello world" end-to-end with 9 merge rules and 17 tokens.
    /// Merge priorities are designed so the full chain resolves deterministically:
    ///   h+e → he → he+l → hel → hel+l → hell → hell+o → hello
    ///   Ġ+w → Ġw, o+r → or → Ġw+or → Ġwor → +l → Ġworl → +d → Ġworld
    static let goldenVocab = BPEVocabulary(
        tokenToID: [
            "h": 0, "e": 1, "l": 2, "o": 3,
            "\u{0120}": 4,  // Ġ = space byte (0x20) in GPT-2 encoding
            "w": 5, "r": 6, "d": 7,
            "he": 8, "hel": 9, "hell": 10, "hello": 11,
            "\u{0120}w": 12, "or": 13,
            "\u{0120}wor": 14, "\u{0120}worl": 15, "\u{0120}world": 16,
        ],
        merges: [
            .init(first: "h", second: "e"),            // 0: h e → he
            .init(first: "he", second: "l"),           // 1: he l → hel
            .init(first: "hel", second: "l"),          // 2: hel l → hell
            .init(first: "hell", second: "o"),         // 3: hell o → hello
            .init(first: "\u{0120}", second: "w"),     // 4: Ġ w → Ġw
            .init(first: "o", second: "r"),            // 5: o r → or
            .init(first: "\u{0120}w", second: "or"),   // 6: Ġw or → Ġwor
            .init(first: "\u{0120}wor", second: "l"),  // 7: Ġwor l → Ġworl
            .init(first: "\u{0120}worl", second: "d"), // 8: Ġworl d → Ġworld
        ]
    )

    static func makeEncoder() -> BPEEncoder {
        BPEEncoder(vocabulary: goldenVocab)
    }

    // MARK: - G01–G07: BPEPreTokenizer

    @Test("G01: Pre-tokenize empty string returns empty array")
    func preTokenizeEmpty() {
        #expect(BPEPreTokenizer.preTokenize("") == [])
    }

    @Test("G02: Pre-tokenize single word")
    func preTokenizeSingleWord() {
        #expect(BPEPreTokenizer.preTokenize("Hello") == ["Hello"])
    }

    @Test("G03: Pre-tokenize two words splits on space boundary")
    func preTokenizeTwoWords() {
        // Space byte (0x20) encodes to Ġ (U+0120) in GPT-2 byte mapping
        let result = BPEPreTokenizer.preTokenize("hello world")
        #expect(result == ["hello", "\u{0120}world"])
    }

    @Test("G04: Pre-tokenize splits contractions")
    func preTokenizeContractions() {
        // GPT-2 regex has explicit patterns for 's, 't, 're, 've, 'm, 'll, 'd
        let result = BPEPreTokenizer.preTokenize("I'm don't we'll")
        #expect(result == ["I", "'m", "\u{0120}don", "'t", "\u{0120}we", "'ll"])
    }

    @Test("G05: Byte encode/decode roundtrip for multibyte UTF-8")
    func byteRoundtripMultibyte() {
        let inputs = ["🎤", "café", "日本語"]
        for input in inputs {
            let encoded = BPEPreTokenizer.byteEncode(input)
            // Each UTF-8 byte becomes one character in the encoded string
            #expect(encoded.count == input.utf8.count)
            let decoded = BPEPreTokenizer.byteDecode(encoded)
            #expect(decoded == input)
        }
    }

    @Test("G06: Byte decode returns nil for unmapped character")
    func byteDecodeNilForUnmapped() {
        // € (U+20AC) is not in the GPT-2 byte-to-unicode table
        #expect(BPEPreTokenizer.byteDecode("\u{20AC}") == nil)
    }

    @Test("G07: Pre-tokenize mixed letters, digits, and punctuation")
    func preTokenizeMixed() {
        let result = BPEPreTokenizer.preTokenize("Say 42 things!")
        #expect(result == ["Say", "\u{0120}42", "\u{0120}things", "!"])
    }

    // MARK: - G08–G10: BPEVocabulary

    @Test("G08: In-memory vocabulary init and lookup")
    func vocabInMemory() {
        let vocab = Self.goldenVocab
        #expect(vocab.count == 17)
        #expect(vocab.id(for: "hello") == 11)
        #expect(vocab.token(for: 16) == "\u{0120}world")
        #expect(vocab.priority(of: .init(first: "h", second: "e")) == 0)
        #expect(vocab.priority(of: .init(first: "\u{0120}worl", second: "d")) == 8)
        #expect(vocab.priority(of: .init(first: "x", second: "y")) == nil)
    }

    @Test("G09: Load vocabulary from valid files on disk")
    func vocabFromDisk() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        try #"{"a": 0, "b": 1, "ab": 2}"#
            .write(to: dir.appendingPathComponent("vocab.json"),
                   atomically: true, encoding: .utf8)
        try "#version: 0.2\na b\n"
            .write(to: dir.appendingPathComponent("merges.txt"),
                   atomically: true, encoding: .utf8)

        let vocab = try BPEVocabulary(directory: dir)
        #expect(vocab.count == 3)
        #expect(vocab.id(for: "ab") == 2)
        #expect(vocab.merges.count == 1)
        #expect(vocab.merges[0] == BPEVocabulary.MergePair(first: "a", second: "b"))
    }

    @Test("G10: Invalid merge line throws BPELoadError")
    func vocabInvalidMerge() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        try #"{"a": 0}"#
            .write(to: dir.appendingPathComponent("vocab.json"),
                   atomically: true, encoding: .utf8)
        try "only_one_token\n"
            .write(to: dir.appendingPathComponent("merges.txt"),
                   atomically: true, encoding: .utf8)

        #expect(throws: BPELoadError.self) {
            try BPEVocabulary(directory: dir)
        }
    }

    // MARK: - G11–G16: BPEEncoder

    @Test("G11: Single character word produces one token")
    func encoderSingleChar() throws {
        var encoder = Self.makeEncoder()
        #expect(try encoder.encode("d") == [7])
    }

    @Test("G12: Full merge chain produces single merged token")
    func encoderMergeChain() throws {
        var encoder = Self.makeEncoder()
        // Merge chain: h+e→he, he+l→hel, hel+l→hell, hell+o→hello
        #expect(try encoder.tokenize("hello") == ["hello"])
    }

    @Test("G13: Encode 'hello world' to expected IDs")
    func encoderToIDs() throws {
        var encoder = Self.makeEncoder()
        // "hello" → ID 11, " world" (byte-encoded Ġworld) → ID 16
        #expect(try encoder.encode("hello world") == [11, 16])
    }

    @Test("G14: Decode IDs back to original text")
    func encoderDecode() {
        let encoder = Self.makeEncoder()
        // [11, 16] → "hello" + "Ġworld" → byteDecode → "hello world"
        #expect(encoder.decode([11, 16]) == "hello world")
    }

    @Test("G15: Encode empty string returns empty array")
    func encoderEmpty() throws {
        var encoder = Self.makeEncoder()
        #expect(try encoder.encode("") == [])
    }

    @Test("G16: Unknown symbols silently skipped during encoding")
    func encoderUnknownSkipped() throws {
        var encoder = Self.makeEncoder()
        // "hex" → BPE: h+e→"he", "x" stays → ["he", "x"]
        // "he" → ID 8, "x" not in vocab → skipped
        #expect(try encoder.encode("hex") == [8])
    }

    @Test("G16a: Cache eviction preserves BPE correctness after churn")
    func encoderCacheEvictionPreservesCorrectness() throws {
        var encoder = Self.makeEncoder()

        for index in 0...10_000 {
            _ = encoder.bpe("token\(index)")
        }

        #expect(try encoder.encode("hello world") == [11, 16])

        let cacheCount = Mirror(reflecting: encoder).children
            .first(where: { $0.label == "cache" })
            .flatMap { $0.value as? [String: [String]] }
            .map(\.count)

        #expect(cacheCount != nil)
        if let cacheCount {
            #expect(cacheCount <= 10_000)
        }
    }

    // MARK: - G17–G18: BPESpecialTokens

    @Test("G17: Qwen3-TTS well-known token IDs are stable")
    func specialTokenConstants() {
        #expect(BPESpecialTokens.Qwen3TTS.imStart == 151644)
        #expect(BPESpecialTokens.Qwen3TTS.imEnd == 151645)
        #expect(BPESpecialTokens.Qwen3TTS.ttsPad == 151671)
        #expect(BPESpecialTokens.Qwen3TTS.ttsBos == 151672)
        #expect(BPESpecialTokens.Qwen3TTS.ttsEos == 151673)
    }

    @Test("G18: Special token registry lookup and isSpecial filtering")
    func specialTokenRegistry() {
        let registry = BPESpecialTokens(tokens: [
            .init(id: 100, content: "<|special|>", special: true),
            .init(id: 200, content: "regular", special: false),
        ])

        // id(for:) only returns IDs for tokens marked special
        #expect(registry.id(for: "<|special|>") == 100)
        #expect(registry.id(for: "regular") == nil)

        // content(for:) returns content for any registered token
        #expect(registry.content(for: 100) == "<|special|>")
        #expect(registry.content(for: 200) == "regular")
        #expect(registry.content(for: 999) == nil)

        // isSpecial only returns true for special: true tokens
        #expect(registry.isSpecial("<|special|>") == true)
        #expect(registry.isSpecial("regular") == false)
        #expect(registry.isSpecial("unknown") == false)
    }

    // MARK: - G19–G20: Qwen3ChatTemplate

    @Test("G19: Chat template format produces correct string")
    func chatTemplateFormat() {
        let template = Qwen3ChatTemplate()
        let expected = "<|im_start|>assistant\nhello<|im_end|>\n<|im_start|>assistant\n"
        #expect(template.format("hello") == expected)
    }

    @Test("G20: Chat template encode wraps text with special token IDs")
    func chatTemplateEncode() throws {
        let template = Qwen3ChatTemplate()
        var encoder = Self.makeEncoder()
        let ids = try template.encode("hello", using: &encoder)

        // Structure: im_start(151644), BPE("assistant\nhello"), im_end(151645),
        //            BPE("\n"), im_start(151644), BPE("assistant\n")
        // With our mini vocab only "hello" resolves to an ID (11);
        // "assistant" and newline chars are not in the vocab and are skipped.
        #expect(ids.first == BPESpecialTokens.Qwen3TTS.imStart)
        #expect(ids.contains(11))  // "hello" token
        #expect(ids.contains(BPESpecialTokens.Qwen3TTS.imEnd))
        #expect(ids == [151644, 11, 151645, 151644])
    }
}
