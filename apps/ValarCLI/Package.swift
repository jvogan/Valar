// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "ValarCLI",
    platforms: [
        .macOS(.v14),
    ],
    products: [
        .executable(name: "valartts", targets: ["valartts"]),
    ],
    dependencies: [
        .package(path: "../../Packages/ValarCore"),
        .package(path: "../../Packages/ValarAudio"),
        .package(path: "../../Packages/ValarMLX"),
        .package(path: "../../Packages/ValarModelKit"),
        .package(path: "../../Packages/ValarPersistence"),
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.5.0"),
    ],
    targets: [
        .executableTarget(
            name: "valartts",
            dependencies: [
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
                .product(name: "ValarAudio", package: "ValarAudio"),
                .product(name: "ValarExport", package: "ValarAudio"),
                .product(name: "ValarCore", package: "ValarCore"),
                .product(name: "ValarMLX", package: "ValarMLX"),
                .product(name: "ValarModelKit", package: "ValarModelKit"),
                .product(name: "ValarPersistence", package: "ValarPersistence"),
            ],
            path: "Sources/ValarCLI"
        ),
        .testTarget(
            name: "ValarCLITests",
            dependencies: [
                "valartts",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
                .product(name: "ValarModelKit", package: "ValarModelKit"),
            ],
            path: "Tests/ValarCLITests"
        ),
    ]
)
