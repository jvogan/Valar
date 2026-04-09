// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "ValarCore",
    platforms: [
        .macOS(.v14),
    ],
    products: [
        .library(name: "ValarCore", targets: ["ValarCore"]),
    ],
    dependencies: [
        .package(path: "../ValarAudio"),
        // Keep runtime composition in ValarCore instead of introducing a separate
        // ValarRuntime package. ValarRuntime already lives here, and this adds a
        // clean one-way edge because ValarMLX depends on ValarModelKit, not ValarCore.
        .package(path: "../ValarMLX"),
        .package(path: "../ValarModelKit"),
        .package(path: "../ValarPersistence"),
    ],
    targets: [
        .target(
            name: "ValarCore",
            dependencies: [
                "ValarAudio",
                .product(name: "ValarMLX", package: "ValarMLX"),
                "ValarModelKit",
                "ValarPersistence",
            ]
        ),
        .testTarget(
            name: "ValarCoreTests",
            dependencies: ["ValarCore"]
        ),
    ]
)
