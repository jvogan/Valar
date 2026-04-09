// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "ValarModelKit",
    platforms: [
        .macOS(.v14),
    ],
    products: [
        .library(name: "ValarModelKit", targets: ["ValarModelKit"]),
    ],
    targets: [
        .target(
            name: "ValarModelKit"
        ),
        .testTarget(
            name: "ValarModelKitTests",
            dependencies: ["ValarModelKit"]
        ),
    ]
)
