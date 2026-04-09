import Foundation

/// GPT-2-style byte-level BPE pre-tokenizer.
///
/// Implements the two-stage pre-processing used by GPT-2 and derived models:
/// 1. Regex-based splitting into word-like chunks (contractions, letter runs, etc.)
/// 2. Byte-to-unicode encoding so every byte sequence has a lossless string representation
///
/// This is the pre-tokenization layer only — actual BPE merge logic lives elsewhere.
public struct BPEPreTokenizer: Sendable {

    // MARK: - Byte ↔ Unicode tables

    /// Bijective mapping from byte values (0–255) to Unicode characters.
    ///
    /// Printable ASCII (33–126) and Latin-1 supplement (161–172, 174–255) map to themselves.
    /// The remaining 68 bytes map to U+0100 through U+0143.
    /// This matches Python's `bytes_to_unicode()` from the GPT-2 encoder exactly.
    public static let byteToUnicode: [UInt8: Character] = {
        // Printable byte ranges that map to their own Unicode codepoint
        var printable: [Int] = []
        printable.append(contentsOf: 33...126)
        printable.append(contentsOf: 161...172)
        printable.append(contentsOf: 174...255)
        let printableSet = Set(printable)

        var table = [UInt8: Character](minimumCapacity: 256)

        for b in printable {
            table[UInt8(b)] = Character(Unicode.Scalar(b)!)
        }

        // Non-printable bytes get sequential codepoints starting at U+0100
        var offset = 0
        for b in 0..<256 {
            if !printableSet.contains(b) {
                table[UInt8(b)] = Character(Unicode.Scalar(256 + offset)!)
                offset += 1
            }
        }

        return table
    }()

    /// Inverse mapping: Unicode character → original byte value.
    public static let unicodeToByte: [Character: UInt8] = {
        var table = [Character: UInt8](minimumCapacity: 256)
        for (byte, char) in byteToUnicode {
            table[char] = byte
        }
        return table
    }()

    // MARK: - Pre-tokenization

    /// GPT-2 pre-tokenization pattern.
    ///
    /// Splits on: contractions ('s, 't, 're, 've, 'm, 'll, 'd),
    /// optional-space + letters, optional-space + digits,
    /// optional-space + other non-whitespace, trailing whitespace, and whitespace runs.
    nonisolated(unsafe) private static let gpt2Pattern = try! Regex(
        #"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"#
    )

    /// Pre-tokenize input text into byte-encoded chunks for BPE.
    ///
    /// 1. Applies the GPT-2 regex to split text into word-like pieces.
    /// 2. Encodes each piece's UTF-8 bytes through the byte-to-unicode mapping.
    ///
    /// - Parameter text: The raw input text.
    /// - Returns: An array of byte-encoded strings, each ready for BPE merging.
    public static func preTokenize(_ text: String) -> [String] {
        guard !text.isEmpty else { return [] }

        var result: [String] = []
        for match in text.matches(of: gpt2Pattern) {
            result.append(byteEncode(String(text[match.range])))
        }
        return result
    }

    /// Encode a string's UTF-8 bytes using the GPT-2 byte-to-unicode mapping.
    ///
    /// Each byte of the UTF-8 representation is replaced by its corresponding
    /// character from the byte-to-unicode table, producing a lossless string encoding.
    ///
    /// - Parameter text: The raw text to encode.
    /// - Returns: A string where each character represents one UTF-8 byte.
    public static func byteEncode(_ text: String) -> String {
        var encoded = ""
        encoded.reserveCapacity(text.utf8.count)
        for byte in text.utf8 {
            encoded.append(byteToUnicode[byte]!)
        }
        return encoded
    }

    /// Decode a byte-encoded string back to the original UTF-8 text.
    ///
    /// - Parameter encoded: A byte-encoded string produced by ``byteEncode(_:)``.
    /// - Returns: The original text, or `nil` if the string contains
    ///   characters outside the byte-to-unicode table.
    public static func byteDecode(_ encoded: String) -> String? {
        var bytes: [UInt8] = []
        bytes.reserveCapacity(encoded.count)
        for char in encoded {
            guard let byte = unicodeToByte[char] else { return nil }
            bytes.append(byte)
        }
        return String(bytes: bytes, encoding: .utf8)
    }
}
