// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "ValarAudio",
    platforms: [
        .macOS(.v14),
    ],
    products: [
        .library(name: "ValarAudio", targets: ["ValarAudio"]),
        .library(name: "ValarExport", targets: ["ValarExport"]),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-atomics.git", from: "1.2.0"),
        .package(url: "https://github.com/element-hq/swift-ogg", exact: "0.0.3"),
    ],
    targets: [
        .target(
            name: "ValarAudio",
            dependencies: [
                .product(name: "Atomics", package: "swift-atomics"),
            ]
        ),
        .target(
            name: "ValarExport",
            dependencies: [
                "ValarAudio",
                .product(name: "SwiftOGG", package: "swift-ogg"),
            ],
            path: "Sources/ValarExport"
        ),
        .testTarget(
            name: "ValarAudioTests",
            dependencies: [
                "ValarAudio",
            ]
        ),
        .testTarget(
            name: "ValarExportTests",
            dependencies: [
                "ValarExport",
                "ValarAudio",
            ],
            path: "Tests/ValarExportTests"
        ),
    ]
)
