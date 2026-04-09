// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "ValarPersistence",
    platforms: [
        .macOS(.v14),
    ],
    products: [
        .library(name: "ValarPersistence", targets: ["ValarPersistence"]),
    ],
    dependencies: [
        .package(url: "https://github.com/groue/GRDB.swift.git", from: "7.0.0"),
        .package(path: "../ValarModelKit"),
    ],
    targets: [
        .target(
            name: "ValarPersistence",
            dependencies: [
                .product(name: "GRDB", package: "GRDB.swift"),
                "ValarModelKit",
            ]
        ),
        .testTarget(
            name: "ValarPersistenceTests",
            dependencies: [
                "ValarPersistence",
                "ValarModelKit",
                .product(name: "GRDB", package: "GRDB.swift"),
            ]
        ),
    ]
)
