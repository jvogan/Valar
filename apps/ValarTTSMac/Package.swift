// swift-tools-version: 6.0
import PackageDescription

let appInfoPlistPath = "\(Context.packageDirectory)/ValarTTSMac-Info.plist"

let package = Package(
    name: "ValarTTSMac",
    platforms: [
        .macOS(.v14),
    ],
    products: [
        .executable(name: "ValarTTSMac", targets: ["ValarTTSMacApp"]),
    ],
    dependencies: [
        .package(path: "../../Packages/ValarAudio"),
        .package(path: "../../Packages/ValarCore"),
        .package(path: "../../Packages/ValarModelKit"),
        .package(path: "../../Packages/ValarPersistence"),
        .package(path: "../../Packages/ValarMLX"),
    ],
    targets: [
        .target(
            name: "ValarSystemMemorySupport",
            path: "Sources/ValarSystemMemorySupport",
            publicHeadersPath: "include"
        ),
        .executableTarget(
            name: "ValarTTSMacApp",
            dependencies: [
                .product(name: "ValarAudio", package: "ValarAudio"),
                .product(name: "ValarCore", package: "ValarCore"),
                .product(name: "ValarModelKit", package: "ValarModelKit"),
                .product(name: "ValarPersistence", package: "ValarPersistence"),
                .product(name: "ValarMLX", package: "ValarMLX"),
                "ValarSystemMemorySupport",
            ],
            linkerSettings: [
                .unsafeFlags([
                    "-Xlinker", "-sectcreate",
                    "-Xlinker", "__TEXT",
                    "-Xlinker", "__info_plist",
                    "-Xlinker", appInfoPlistPath,
                ], .when(platforms: [.macOS])),
            ]
        ),
        .testTarget(
            name: "ValarTTSMacAppTests",
            dependencies: [
                .target(name: "ValarTTSMacApp")
            ]
        ),
    ]
)
