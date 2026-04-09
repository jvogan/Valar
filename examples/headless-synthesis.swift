#!/usr/bin/env swift
//
// headless-synthesis.swift
// Valar — Examples
//
// Demonstrates the intended shape of programmatic synthesis using the public
// Valar runtime packages without opening the macOS app.
//
// Status: sketch. Prefer the public CLI quickstart for the fastest working path.
//
// Requirements:
//   macOS 14+, Apple Silicon, Swift 6, Xcode 15+
//

print(
    """
    Valar headless synthesis sketch

    This example is intentionally a runbook, not a drop-in package target.

    Suggested flow:
    1. make bootstrap-native
    2. make build-cli
    3. bash scripts/build_metallib.sh
    4. swift run --package-path apps/ValarCLI valartts models install mlx-community/Soprano-1.1-80M-bf16 --allow-download
    5. swift run --package-path apps/ValarCLI valartts speak --model mlx-community/Soprano-1.1-80M-bf16 --text "Hello from Valar." --output /tmp/valar-soprano.wav

    Once that path works, turn this sketch into a real local package or app target that imports Valar packages directly.
    """
)

// import ValarModelKit
// import ValarCore
// import ValarMLX

// 1. Resolve an installed model from the public catalog.
//
// let catalog = SupportedModelCatalog.shared
// guard let modelEntry = catalog.model(id: "mlx-community/Soprano-1.1-80M-bf16") else {
//     print("Model not found in the public catalog. Install it first with `valartts models install`.")
//     exit(1)
// }

// 2. Construct a synthesis request.
//
// let request = SynthesisRequest(
//     text: "Hello from Valar headless synthesis.",
//     speakerID: nil,
//     speed: 1.0,
//     format: .wav
// )

// 3. Load the MLX backend.
//
// let backend = MLXInferenceBackend(modelEntry: modelEntry)
// try await backend.load()

// 4. Run inference and write the result.
//
// let result = try await backend.synthesize(request)
// let outputURL = URL(fileURLWithPath: "output.wav")
// try result.audioData.write(to: outputURL)
// print("Wrote \\(result.audioData.count) bytes to \\(outputURL.path)")

// Notes:
//
// - Install model weights first via the public CLI quickstart.
// - Build `mlx.metallib` with `bash scripts/build_metallib.sh` before using SPM-built binaries.
// - For the fastest working path, use the CLI and daemon docs in `docs/model-quickstart.md`.
