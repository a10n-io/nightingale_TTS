// swift-tools-version:6.0
import PackageDescription

let package = Package(
    name: "VerifyLive",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(path: "../.."),
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.3.0"),
    ],
    targets: [
        .executableTarget(
            name: "VerifyLive",
            dependencies: [
                .product(name: "Nightingale", package: "swift"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "."
        ),
    ]
)
