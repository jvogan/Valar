import Foundation
@preconcurrency import MLX
@preconcurrency import MLXLMCommon

public struct KVCacheSnapshot {
    let state: [MLXArray]
    let metaState: [String]
}

func snapshotKVCaches(_ caches: [any KVCache]) -> [KVCacheSnapshot] {
    caches.map { cache in
        KVCacheSnapshot(state: cache.state, metaState: cache.metaState)
    }
}

func restoreKVCaches(
    _ caches: inout [any KVCache],
    from snapshots: [KVCacheSnapshot]
) {
    precondition(
        caches.count == snapshots.count,
        "KV cache snapshot count must match the cache collection."
    )
    for index in caches.indices {
        let snapshot = snapshots[index]
        if snapshot.state.isEmpty, snapshot.metaState.isEmpty, caches[index].isTrimmable {
            _ = caches[index].trim(caches[index].offset)
            continue
        }
        caches[index].state = snapshot.state
        caches[index].metaState = snapshot.metaState
    }
}

func cloneKVCaches(
    from snapshots: [KVCacheSnapshot],
    using makeCaches: () -> [any KVCache]
) -> [any KVCache] {
    var caches = makeCaches()
    restoreKVCaches(&caches, from: snapshots)
    return caches
}

func resetKVCachesForReuse(_ caches: inout [any KVCache]) {
    for index in caches.indices {
        if let cache = caches[index] as? KVCacheSimple {
            cache.offset = 0
            continue
        }
        if caches[index].isTrimmable {
            _ = caches[index].trim(caches[index].offset)
            continue
        }
        caches[index].state = []
        caches[index].metaState = []
    }
}
