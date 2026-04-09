import Foundation

enum GRDBRecordCodingError: Error, LocalizedError, Equatable {
    case jsonEncodingFailed(table: String, column: String, underlying: String)
    case jsonEncodingProducedInvalidUTF8(table: String, column: String)
    case jsonDecodingFailed(table: String, column: String, underlying: String)

    var errorDescription: String? {
        switch self {
        case let .jsonEncodingFailed(table, column, underlying):
            return "Failed to encode JSON for \(table).\(column): \(underlying)"
        case let .jsonEncodingProducedInvalidUTF8(table, column):
            return "Failed to encode JSON for \(table).\(column): encoder produced non-UTF-8 data"
        case let .jsonDecodingFailed(table, column, underlying):
            return "Failed to decode JSON for \(table).\(column): \(underlying)"
        }
    }
}

func decodeJSONColumn<Value: Decodable>(
    _ type: Value.Type,
    from jsonString: String,
    table: String,
    column: String,
    decoder: JSONDecoder = JSONDecoder()
) throws -> Value {
    do {
        return try decoder.decode(type, from: Data(jsonString.utf8))
    } catch {
        throw GRDBRecordCodingError.jsonDecodingFailed(
            table: table,
            column: column,
            underlying: error.localizedDescription
        )
    }
}

func encodeJSONColumn<Value: Encodable>(
    _ value: Value,
    table: String,
    column: String,
    encoder: JSONEncoder = JSONEncoder()
) throws -> String {
    let data: Data
    do {
        data = try encoder.encode(value)
    } catch {
        throw GRDBRecordCodingError.jsonEncodingFailed(
            table: table,
            column: column,
            underlying: error.localizedDescription
        )
    }

    guard let string = String(data: data, encoding: .utf8) else {
        throw GRDBRecordCodingError.jsonEncodingProducedInvalidUTF8(table: table, column: column)
    }
    return string
}
