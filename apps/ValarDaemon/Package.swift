// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "ValarDaemon",
    platforms: [
        .macOS(.v14),
    ],
    products: [
        .executable(name: "valarttsd", targets: ["valarttsd"]),
    ],
    dependencies: [
        .package(url: "https://github.com/hummingbird-project/hummingbird.git", exact: "2.21.1"),
        .package(path: "../../Packages/ValarCore"),
        .package(path: "../../Packages/ValarMLX"),
        .package(path: "../../Packages/ValarAudio"),
        .package(path: "../../Packages/ValarPersistence"),
    ],
    targets: [
        .executableTarget(
            name: "valarttsd",
            dependencies: [
                .product(name: "Hummingbird", package: "hummingbird"),
                .product(name: "ValarCore", package: "ValarCore"),
                .product(name: "ValarMLX", package: "ValarMLX"),
                .product(name: "ValarAudio", package: "ValarAudio"),
                .product(name: "ValarExport", package: "ValarAudio"),
                .product(name: "ValarPersistence", package: "ValarPersistence"),
            ],
            path: "Sources/ValarDaemon"
        ),
    ]
)
