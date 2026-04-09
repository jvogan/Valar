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

The following Swift packages are linked as dependencies via Swift Package
Manager. They are not vendored in this repository but are resolved at build
time.

- **GRDB** (`groue/GRDB.swift`) — MIT
- **DSWaveformImage** (`dmrschmidt/DSWaveformImage`) — MIT
- **Hummingbird** (`hummingbird-project/hummingbird`) — Apache 2.0

## Summary

- Root repository license: MIT
- Runtime-downloaded model weights: upstream model-specific licenses
- Additional vendored/adapted source notices: preserved in file headers and in
  this file where applicable
- Included Apache 2.0 text: `LICENSES/Apache-2.0.txt`
