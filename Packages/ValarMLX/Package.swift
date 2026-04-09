// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "ValarMLX",
    platforms: [
        .macOS(.v14),
    ],
    products: [
        .library(name: "ValarMLX", targets: ["ValarMLX"]),
    ],
    dependencies: [
        .package(path: "../ValarModelKit"),
        .package(path: "../mlx-audio-swift-valar"), // Valar fork of mlx-audio-swift — adds .codes() enum case + generateCodeStream() for native decoder path (upstream: Blaizzy/mlx-audio-swift @ a1532367)
        .package(
            url: "https://github.com/FluidInference/FluidAudio.git",
            revision: "e5c6456dd9cbbd6bcdc3aeefbddfcd483c5d3ca6"
        ), // Upstream FluidAudio main @ 2026-04-02 — Swift 6 compatible and free of the older swift-transformers pin conflict.
    ],
    targets: [
        .target(
            name: "ValarMLX",
            dependencies: [
                .product(name: "ValarModelKit", package: "ValarModelKit"),
                .product(name: "MLXAudioTTS", package: "mlx-audio-swift-valar"),
                .product(name: "MLXAudioSTT", package: "mlx-audio-swift-valar"),
                .product(name: "MLXAudioCore", package: "mlx-audio-swift-valar"),
                .product(name: "FluidAudio", package: "FluidAudio"),
            ]
        ),
        .testTarget(
            name: "ValarMLXTests",
            dependencies: [
                .target(name: "ValarMLX"),
                .product(name: "FluidAudio", package: "FluidAudio"),
            ],
            resources: [
                .copy("Fixtures/tokenizer_golden.json"),
                .copy("Fixtures/e2e_parity_golden.json"),
            ]
        ),
    ]
)
