// swift-tools-version:6.0
import PackageDescription

let package = Package(
    name: "VerifyLive",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(path: "../.."),
    ],
    targets: [
        .executableTarget(
            name: "VerifyLive",
            dependencies: [
                .product(name: "Nightingale", package: "swift"),
            ],
            path: "."
        ),
    ]
)
