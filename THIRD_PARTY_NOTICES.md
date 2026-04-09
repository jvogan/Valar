# Third-Party Notices

Valar is MIT-licensed at the repository root. This file records third-party
source provenance and license boundaries that still apply to preserved files in
the public tree.

The root [LICENSE](LICENSE) covers Valar-specific packaging, glue code,
documentation, and repository-level changes. It does not remove or replace
preserved notices in vendored or adapted upstream files.

Model checkpoints downloaded at runtime from Hugging Face or other registries
are not distributed in this repository. Those weights remain subject to their
own upstream model licenses and usage terms.

## Vendored And Adapted Source

- `Packages/mlx-audio-swift-valar/` is Valar's adapted fork of the upstream
  `mlx-audio-swift` project and remains subject to preserved upstream notices in
  file headers where present.
- Some vendored or adapted source files in that tree preserve additional
  provenance notices from their original upstream projects. Those notices remain
  in the file headers and continue to apply to the copied source.

## Swift Package Dependencies

The following Swift packages are resolved as direct or transitive dependencies
via Swift Package Manager. They are not vendored in this repository but are
fetched at build time.

Direct dependencies:

- **GRDB** (`groue/GRDB.swift`) — MIT
- **Hummingbird** (`hummingbird-project/hummingbird`) — Apache 2.0
- **FluidAudio** (`FluidInference/FluidAudio`) — MIT
- **swift-atomics** (`apple/swift-atomics`) — Apache 2.0
- **swift-ogg** (`element-hq/swift-ogg`) — Apache 2.0
- **swift-argument-parser** (`apple/swift-argument-parser`) — Apache 2.0
- **swift-transformers** (`huggingface/swift-transformers`) — Apache 2.0
- **swift-huggingface** (`huggingface/swift-huggingface`) — Apache 2.0
- **mlx-swift** (`ml-explore/mlx-swift`) — MIT
- **mlx-swift-lm** (`ml-explore/mlx-swift-lm`) — MIT

Notable transitive dependencies (resolved via Hummingbird):

- **swift-nio** (`apple/swift-nio`) — Apache 2.0
- **swift-crypto** (`apple/swift-crypto`) — Apache 2.0

## Summary

- Root repository license: MIT
- Runtime-downloaded model weights: upstream model-specific licenses
- Additional vendored/adapted source notices: preserved in file headers and in
  this file where applicable
- Included Apache 2.0 text: `LICENSES/Apache-2.0.txt`
