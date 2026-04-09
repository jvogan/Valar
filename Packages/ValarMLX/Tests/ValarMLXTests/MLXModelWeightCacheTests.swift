import Foundation
import Testing
@testable import ValarMLX

/// Unit tests for `MLXModelWeightCache`.
///
/// Exercises cache construction, disk I/O, hit/miss logic, staleness detection,
/// eviction, and error handling — all without requiring a real MLX model.
@Suite("MLXModelWeightCache")
struct MLXModelWeightCacheTests {

    // MARK: - Helpers

    private func makeTemporaryDirectory() throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    /// Creates a real `.safetensors` file in `directory` and returns its URL.
    private func writeSafeTensorsFile(
        named name: String = "model.safetensors",
        in directory: URL,
        payload: Data = Data("weight-bytes".utf8)
    ) throws -> URL {
        let url = directory.appendingPathComponent(name)
        try payload.write(to: url, options: .atomic)
        return url
    }

    // MARK: - Initial state

    @Test("New cache has zero entries")
    func emptyInitHasZeroEntries() {
        let cache = MLXModelWeightCache()
        #expect(cache.entryCount == 0)
    }

    @Test("New cache returns nil for any model ID")
    func emptyInitReturnsCacheMiss() {
        let cache = MLXModelWeightCache()
        #expect(cache.cachedDirectory(for: "any-model") == nil)
    }

    // MARK: - Load from disk

    @Test("Load from missing file returns empty cache without throwing")
    func loadFromMissingFileReturnsEmpty() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("json")
        let cache = try MLXModelWeightCache.load(from: url)
        #expect(cache.entryCount == 0)
    }

    @Test("Load from corrupt JSON returns empty cache without throwing")
    func loadFromCorruptJSONReturnsEmpty() throws {
        let dir = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }

        let url = dir.appendingPathComponent("weight_cache.json")
        try Data("not-valid-json!!!".utf8).write(to: url, options: .atomic)

        let cache = try MLXModelWeightCache.load(from: url)
        #expect(cache.entryCount == 0)
    }

    @Test("Load from empty JSON object returns empty cache")
    func loadFromEmptyJSONReturnsEmpty() throws {
        let dir = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }

        let url = dir.appendingPathComponent("weight_cache.json")
        try Data("{}".utf8).write(to: url, options: .atomic)

        let cache = try MLXModelWeightCache.load(from: url)
        #expect(cache.entryCount == 0)
    }

    // MARK: - Save and reload

    @Test("Save and load round-trips entry count")
    func saveLoadPreservesEntryCount() throws {
        let modelDir = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: modelDir) }

        let cacheDir = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: cacheDir) }

        _ = try writeSafeTensorsFile(in: modelDir)

        var cache = MLXModelWeightCache()
        try cache.store(modelID: "test-model", directory: modelDir)

        let cacheURL = cacheDir.appendingPathComponent("weight_cache.json")
        try cache.save(to: cacheURL)

        let reloaded = try MLXModelWeightCache.load(from: cacheURL)
        #expect(reloaded.entryCount == 1)
    }

    @Test("Reloaded cache hits on the stored model ID")
    func reloadedCacheHitsOnStoredModel() throws {
        let modelDir = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: modelDir) }

        let cacheDir = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: cacheDir) }

        _ = try writeSafeTensorsFile(in: modelDir)

        var cache = MLXModelWeightCache()
        try cache.store(modelID: "my-model", directory: modelDir)

        let cacheURL = cacheDir.appendingPathComponent("weight_cache.json")
        try cache.save(to: cacheURL)

        let reloaded = try MLXModelWeightCache.load(from: cacheURL)
        let hit = reloaded.cachedDirectory(for: "my-model")
        #expect(hit != nil)
    }

    @Test("Save creates intermediate directories automatically")
    func saveCreatesIntermediateDirectories() throws {
        let modelDir = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: modelDir) }

        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        defer { try? FileManager.default.removeItem(at: base) }

        _ = try writeSafeTensorsFile(in: modelDir)

        var cache = MLXModelWeightCache()
        try cache.store(modelID: "nested-model", directory: modelDir)

        // Deep nested path — the directory does not exist yet.
        let deepURL = base
            .appendingPathComponent("a/b/c", isDirectory: true)
            .appendingPathComponent("weight_cache.json")

        try cache.save(to: deepURL)
        #expect(FileManager.default.fileExists(atPath: deepURL.path))
    }

    // MARK: - cachedDirectory: hit / miss

    @Test("cachedDirectory returns directory URL on cache hit")
    func cachedDirectoryHitReturnsURL() throws {
        let modelDir = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: modelDir) }

        _ = try writeSafeTensorsFile(in: modelDir)

        var cache = MLXModelWeightCache()
        try cache.store(modelID: "hit-model", directory: modelDir)

        let result = cache.cachedDirectory(for: "hit-model")
        #expect(result != nil)
        let resolvedExpected = modelDir.resolvingSymlinksInPath()
        let resolvedActual = result!.resolvingSymlinksInPath()
        #expect(resolvedActual.path == resolvedExpected.path)
    }

    @Test("cachedDirectory returns nil for unknown model ID")
    func cachedDirectoryMissForUnknownModel() throws {
        let modelDir = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: modelDir) }

        _ = try writeSafeTensorsFile(in: modelDir)

        var cache = MLXModelWeightCache()
        try cache.store(modelID: "model-a", directory: modelDir)

        #expect(cache.cachedDirectory(for: "model-b") == nil)
    }

    @Test("cachedDirectory returns nil after safetensors file is deleted")
    func cachedDirectoryMissWhenFileMissing() throws {
        let modelDir = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: modelDir) }

        let weightFile = try writeSafeTensorsFile(in: modelDir)

        var cache = MLXModelWeightCache()
        try cache.store(modelID: "stale-model", directory: modelDir)

        // Confirm hit before deletion.
        #expect(cache.cachedDirectory(for: "stale-model") != nil)

        // Delete the weight file — cache entry is now stale.
        try FileManager.default.removeItem(at: weightFile)
        #expect(cache.cachedDirectory(for: "stale-model") == nil)
    }

    @Test("cachedDirectory returns nil after safetensors file is touched (mtime changes)")
    func cachedDirectoryMissWhenFileTouched() throws {
        let modelDir = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: modelDir) }

        let weightFile = try writeSafeTensorsFile(in: modelDir)

        var cache = MLXModelWeightCache()
        try cache.store(modelID: "touched-model", directory: modelDir)
        #expect(cache.cachedDirectory(for: "touched-model") != nil)

        // Advance the modification date by 2 seconds.
        let futureDate = Date(timeIntervalSinceNow: 2)
        try FileManager.default.setAttributes(
            [.modificationDate: futureDate],
            ofItemAtPath: weightFile.path
        )

        #expect(cache.cachedDirectory(for: "touched-model") == nil)
    }

    // MARK: - Store

    @Test("Store with no safetensors files still records an entry")
    func storeEmptyDirectoryCreatesEntry() throws {
        let modelDir = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: modelDir) }

        var cache = MLXModelWeightCache()
        try cache.store(modelID: "empty-dir", directory: modelDir)

        // Entry is recorded, but with no weight files the hit check will pass
        // (nothing to invalidate).
        #expect(cache.entryCount == 1)
        let result = cache.cachedDirectory(for: "empty-dir")
        #expect(result != nil)
    }

    @Test("Store replaces an existing entry for the same model ID")
    func storeReplacesExistingEntry() throws {
        let dirA = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: dirA) }
        let dirB = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: dirB) }

        _ = try writeSafeTensorsFile(named: "a.safetensors", in: dirA)
        _ = try writeSafeTensorsFile(named: "b.safetensors", in: dirB)

        var cache = MLXModelWeightCache()
        try cache.store(modelID: "model", directory: dirA)
        try cache.store(modelID: "model", directory: dirB)

        // Should still have exactly 1 entry, pointing at dirB.
        #expect(cache.entryCount == 1)
        let hit = cache.cachedDirectory(for: "model")
        let resolvedB = dirB.resolvingSymlinksInPath()
        let resolvedHit = hit?.resolvingSymlinksInPath()
        #expect(resolvedHit?.path == resolvedB.path)
    }

    @Test("Store records multiple models independently")
    func storeMultipleModels() throws {
        let dirA = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: dirA) }
        let dirB = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: dirB) }

        _ = try writeSafeTensorsFile(in: dirA)
        _ = try writeSafeTensorsFile(in: dirB)

        var cache = MLXModelWeightCache()
        try cache.store(modelID: "model-a", directory: dirA)
        try cache.store(modelID: "model-b", directory: dirB)

        #expect(cache.entryCount == 2)
        #expect(cache.cachedDirectory(for: "model-a") != nil)
        #expect(cache.cachedDirectory(for: "model-b") != nil)
    }

    // MARK: - Evict

    @Test("Evict removes the specified model entry")
    func evictRemovesEntry() throws {
        let modelDir = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: modelDir) }

        _ = try writeSafeTensorsFile(in: modelDir)

        var cache = MLXModelWeightCache()
        try cache.store(modelID: "evict-me", directory: modelDir)
        #expect(cache.entryCount == 1)

        cache.evict(modelID: "evict-me")
        #expect(cache.entryCount == 0)
        #expect(cache.cachedDirectory(for: "evict-me") == nil)
    }

    @Test("Evict of unknown model ID is a no-op")
    func evictUnknownModelIsNoOp() throws {
        let modelDir = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: modelDir) }

        _ = try writeSafeTensorsFile(in: modelDir)

        var cache = MLXModelWeightCache()
        try cache.store(modelID: "keep-me", directory: modelDir)

        cache.evict(modelID: "no-such-model")

        #expect(cache.entryCount == 1)
        #expect(cache.cachedDirectory(for: "keep-me") != nil)
    }

    @Test("Evict one of multiple models leaves others intact")
    func evictOneOfManyModels() throws {
        let dirA = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: dirA) }
        let dirB = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: dirB) }

        _ = try writeSafeTensorsFile(in: dirA)
        _ = try writeSafeTensorsFile(in: dirB)

        var cache = MLXModelWeightCache()
        try cache.store(modelID: "alpha", directory: dirA)
        try cache.store(modelID: "beta", directory: dirB)

        cache.evict(modelID: "alpha")

        #expect(cache.entryCount == 1)
        #expect(cache.cachedDirectory(for: "alpha") == nil)
        #expect(cache.cachedDirectory(for: "beta") != nil)
    }

    // MARK: - Multi-shard discovery

    @Test("Store discovers multiple safetensors shards in subdirectories")
    func storeDiscoversShardsInSubdirectory() throws {
        let modelDir = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: modelDir) }

        let shardsDir = modelDir.appendingPathComponent("weights", isDirectory: true)
        try FileManager.default.createDirectory(at: shardsDir, withIntermediateDirectories: true)

        _ = try writeSafeTensorsFile(named: "shard-00001.safetensors", in: shardsDir)
        _ = try writeSafeTensorsFile(named: "shard-00002.safetensors", in: shardsDir)

        var cache = MLXModelWeightCache()
        try cache.store(modelID: "sharded-model", directory: modelDir)
        #expect(cache.entryCount == 1)

        // Both shards must still be present for a cache hit.
        #expect(cache.cachedDirectory(for: "sharded-model") != nil)
    }
}
