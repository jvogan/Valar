import Foundation

// MARK: - Cache Entry

/// A single model's weight-cache record.
///
/// Stores the resolved model directory path alongside the relative path and
/// modification time of every `.safetensors` file that was present when the
/// cache entry was written. An entry is considered valid when every stored
/// file still exists on disk with an identical modification time.
struct MLXModelWeightCacheEntry: Codable, Sendable {
    /// Absolute path to the resolved model directory.
    let directoryPath: String
    /// Per-file records, one per `.safetensors` file found in the directory tree.
    let weightFiles: [WeightFileRecord]

    struct WeightFileRecord: Codable, Sendable {
        /// Path relative to `directoryPath`, using `/` as the separator.
        let relativePath: String
        /// Modification date recorded at cache-write time.
        let modificationDate: Date
    }
}

// MARK: - Cache

/// JSON index of resolved model directories and their `.safetensors` modification times.
///
/// `MLXModelWeightCache` enables warm restarts: on subsequent process launches the
/// cache is read from disk, modification times are compared against the live
/// filesystem, and a valid cache hit skips the HuggingFace Hub resolution call.
/// `MLXInferenceBackend` still performs a fresh directory scan before trusting the
/// cached path.
///
/// ### Validation strategy
/// A cache entry is valid when every stored `.safetensors` file still exists at its
/// recorded path with an identical modification time. Any discrepancy (file missing,
/// mtime changed, new safetensors added) counts as a miss and triggers a full
/// re-resolution.
///
/// ### Thread-safety
/// `MLXModelWeightCache` is a value type. All mutations produce a new copy, making
/// it safe to store as an actor-isolated `var` in `MLXInferenceBackend`.
struct MLXModelWeightCache: Sendable {
    private static let modificationDateTolerance: TimeInterval = 0.001

    // MARK: - Errors

    enum CacheError: Error, Sendable {
        case ioError(Swift.Error)
        case decodingFailed(Swift.Error)
        case encodingFailed(Swift.Error)
    }

    // MARK: - Storage

    private var entries: [String: MLXModelWeightCacheEntry]

    // MARK: - Init

    /// Creates an empty in-memory cache.
    init() {
        self.entries = [:]
    }

    private init(entries: [String: MLXModelWeightCacheEntry]) {
        self.entries = entries
    }

    // MARK: - Disk I/O

    /// Reads a persisted cache from `url`.
    ///
    /// Returns an empty cache when the file does not exist or cannot be decoded.
    /// A corrupt or incompatible cache file is silently discarded — the next
    /// successful validation will rebuild it.
    static func load(from url: URL) throws -> MLXModelWeightCache {
        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch let error as NSError
            where error.domain == NSCocoaErrorDomain
            && (error.code == NSFileNoSuchFileError || error.code == NSFileReadNoSuchFileError) {
            return MLXModelWeightCache()
        } catch {
            throw CacheError.ioError(error)
        }

        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .secondsSince1970
        guard let entries = try? decoder.decode([String: MLXModelWeightCacheEntry].self, from: data) else {
            // Treat an undecodable cache file as empty rather than crashing.
            return MLXModelWeightCache()
        }
        return MLXModelWeightCache(entries: entries)
    }

    /// Atomically writes the cache to `url`, creating any intermediate directories.
    func save(to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .secondsSince1970
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]

        let data: Data
        do {
            data = try encoder.encode(entries)
        } catch {
            throw CacheError.encodingFailed(error)
        }

        do {
            let directory = url.deletingLastPathComponent()
            try FileManager.default.createDirectory(
                at: directory,
                withIntermediateDirectories: true
            )
            try data.write(to: url, options: .atomic)
        } catch {
            throw CacheError.ioError(error)
        }
    }

    // MARK: - Cache Operations

    /// Returns the cached directory `URL` for `modelID` if all stored `.safetensors`
    /// modification times still match the current filesystem state, or `nil` on any
    /// mismatch or missing entry.
    ///
    /// This call is intentionally synchronous and allocation-light: it reads only the
    /// mtime attribute for each recorded file, which is served from the VFS metadata
    /// cache on macOS without touching data blocks.
    func cachedDirectory(for modelID: String) -> URL? {
        guard let entry = entries[modelID] else { return nil }
        let directory = URL(fileURLWithPath: entry.directoryPath, isDirectory: true)

        // Validate that the cached path is within the expected HF cache root to prevent
        // JSON-poisoning attacks where an attacker writes an arbitrary path to the cache file.
        let resolved = directory.standardized.resolvingSymlinksInPath()
        let hfCache = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/hub")
            .resolvingSymlinksInPath()
        let hfCachePath = hfCache.path.hasSuffix("/") ? hfCache.path : "\(hfCache.path)/"
        guard resolved.path.hasPrefix(hfCachePath) else { return nil }

        let fileManager = FileManager.default

        for record in entry.weightFiles {
            let fileURL = directory.appendingPathComponent(record.relativePath)
            guard
                let attributes = try? fileManager.attributesOfItem(atPath: fileURL.path),
                let mtime = attributes[.modificationDate] as? Date,
                abs(mtime.timeIntervalSince(record.modificationDate)) <= Self.modificationDateTolerance
            else {
                return nil
            }
        }

        return directory
    }

    /// Records `directory` and the current modification times of all `.safetensors`
    /// files it contains, replacing any prior entry for `modelID`.
    ///
    /// Only `.safetensors` files contribute to the cache entry — other files in the
    /// directory do not affect cache validity.
    mutating func store(modelID: String, directory: URL) throws {
        let fileManager = FileManager.default
        let resolvedDirectory = directory.resolvingSymlinksInPath()

        guard let enumerator = fileManager.enumerator(
            at: resolvedDirectory,
            includingPropertiesForKeys: [.isRegularFileKey, .contentModificationDateKey],
            options: []
        ) else {
            return
        }

        var records: [MLXModelWeightCacheEntry.WeightFileRecord] = []
        let dirPrefix = resolvedDirectory.path.hasSuffix("/")
            ? resolvedDirectory.path
            : "\(resolvedDirectory.path)/"

        for case let fileURL as URL in enumerator {
            guard
                let values = try? fileURL.resourceValues(
                    forKeys: [.isRegularFileKey, .contentModificationDateKey]
                ),
                values.isRegularFile == true,
                fileURL.pathExtension.lowercased() == "safetensors",
                let mtime = values.contentModificationDate
            else {
                continue
            }

            let resolvedFilePath = fileURL.resolvingSymlinksInPath().path
            let relativePath = resolvedFilePath.hasPrefix(dirPrefix)
                ? String(resolvedFilePath.dropFirst(dirPrefix.count))
                : resolvedFilePath

            records.append(.init(relativePath: relativePath, modificationDate: mtime))
        }

        // Sort for deterministic JSON output and reproducible cache comparisons.
        records.sort { $0.relativePath < $1.relativePath }

        entries[modelID] = MLXModelWeightCacheEntry(
            directoryPath: resolvedDirectory.path,
            weightFiles: records
        )
    }

    /// Removes the cache entry for `modelID`.
    mutating func evict(modelID: String) {
        entries.removeValue(forKey: modelID)
    }

    /// Number of entries currently in the cache.
    var entryCount: Int { entries.count }
}
