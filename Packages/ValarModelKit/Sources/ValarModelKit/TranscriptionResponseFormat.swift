import Foundation

/// Public response/output formats supported by Valar transcription surfaces.
///
/// This keeps daemon and MCP request validation aligned with the existing
/// formatter support already used by the CLI.
public enum TranscriptionResponseFormat: String, Codable, CaseIterable, Sendable {
    case text
    case json
    case verbose_json
    case srt
    case vtt

    public init?(apiValue: String?) {
        guard let normalized = apiValue?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased(),
              !normalized.isEmpty,
              let format = Self(rawValue: normalized) else {
            return nil
        }
        self = format
    }

    public static var supportedValuesDescription: String {
        Self.allCases.map(\.rawValue).joined(separator: ", ")
    }

    public var contentType: String {
        switch self {
        case .text, .srt:
            return "text/plain; charset=utf-8"
        case .vtt:
            return "text/vtt; charset=utf-8"
        case .json, .verbose_json:
            return "application/json"
        }
    }

    public func render(_ result: RichTranscriptionResult) throws -> String {
        switch self {
        case .text:
            return result.text
        case .json:
            return try encodeJSONString(TranscriptionResult(result))
        case .verbose_json:
            return try encodeJSONString(result)
        case .srt:
            return TranscriptionFormatter.srt(from: result)
        case .vtt:
            return TranscriptionFormatter.vtt(from: result)
        }
    }

    private func encodeJSONString<T: Encodable>(_ value: T) throws -> String {
        let data = try JSONEncoder().encode(value)
        return String(decoding: data, as: UTF8.self)
    }
}
