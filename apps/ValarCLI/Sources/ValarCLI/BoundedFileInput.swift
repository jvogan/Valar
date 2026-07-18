import Darwin
import Foundation

enum BoundedFileInputError: LocalizedError, Equatable, Sendable {
    case invalidLimit
    case cannotOpen(label: String, message: String)
    case notRegularFile(label: String)
    case exceedsLimit(label: String, observedBytes: Int64, maximumBytes: Int)
    case changedDuringRead(label: String)

    var errorDescription: String? {
        switch self {
        case .invalidLimit:
            return "The file input byte limit is invalid."
        case .cannotOpen(let label, let message):
            return "Could not open \(label): \(message)"
        case .notRegularFile(let label):
            return "\(label) must be a regular file."
        case .exceedsLimit(let label, let observedBytes, let maximumBytes):
            return "\(label) is \(observedBytes) bytes; the limit is \(maximumBytes) bytes."
        case .changedDuringRead(let label):
            return "\(label) changed while it was being read."
        }
    }
}

enum CLIAudioInputLimits {
    /// Leaves room under the daemon's 15,000,000-byte multipart body ceiling
    /// for boundaries, headers, and ordinary text fields.
    static let daemonMultipartFileBytes = 14_000_000
    /// Base64 expands by roughly 4/3 before the JSON/data-URL envelope.
    static let daemonInlineReferenceBytes = 10_000_000
    /// Matches the common file-aware decoder's encoded-input ceiling.
    static let localEncodedAudioBytes = 128 * 1_024 * 1_024
}

enum BoundedFileInput {
    private static let readChunkBytes = 1_024 * 1_024

    /// Opens one resolved regular file descriptor, validates its size before
    /// allocation, drains it in bounded chunks, and rejects concurrent changes.
    static func readData(
        from url: URL,
        maximumByteCount: Int,
        label: String
    ) throws -> Data {
        guard maximumByteCount >= 0, maximumByteCount < Int.max else {
            throw BoundedFileInputError.invalidLimit
        }

        let resolvedURL = url.resolvingSymlinksInPath().standardizedFileURL
        let descriptor = Darwin.open(
            resolvedURL.path,
            O_RDONLY | O_CLOEXEC | O_NOFOLLOW
        )
        guard descriptor >= 0 else {
            throw BoundedFileInputError.cannotOpen(
                label: label,
                message: String(cString: strerror(errno))
            )
        }
        let handle = FileHandle(fileDescriptor: descriptor, closeOnDealloc: true)
        defer { try? handle.close() }

        var initialInfo = stat()
        guard fstat(descriptor, &initialInfo) == 0,
              initialInfo.st_mode & S_IFMT == S_IFREG,
              initialInfo.st_size >= 0 else {
            throw BoundedFileInputError.notRegularFile(label: label)
        }
        guard initialInfo.st_size <= off_t(maximumByteCount) else {
            throw BoundedFileInputError.exceedsLimit(
                label: label,
                observedBytes: Int64(initialInfo.st_size),
                maximumBytes: maximumByteCount
            )
        }

        var data = Data()
        data.reserveCapacity(Int(initialInfo.st_size))
        while true {
            let remainingIncludingSentinel = maximumByteCount - data.count + 1
            let requestedBytes = min(readChunkBytes, remainingIncludingSentinel)
            guard let chunk = try handle.read(upToCount: requestedBytes),
                  !chunk.isEmpty else {
                break
            }
            guard chunk.count <= maximumByteCount - data.count else {
                throw BoundedFileInputError.exceedsLimit(
                    label: label,
                    observedBytes: Int64(data.count) + Int64(chunk.count),
                    maximumBytes: maximumByteCount
                )
            }
            data.append(chunk)
        }

        var finalInfo = stat()
        guard fstat(descriptor, &finalInfo) == 0,
              finalInfo.st_dev == initialInfo.st_dev,
              finalInfo.st_ino == initialInfo.st_ino,
              finalInfo.st_mode & S_IFMT == S_IFREG,
              finalInfo.st_size == initialInfo.st_size,
              data.count == Int(initialInfo.st_size) else {
            throw BoundedFileInputError.changedDuringRead(label: label)
        }
        return data
    }
}
