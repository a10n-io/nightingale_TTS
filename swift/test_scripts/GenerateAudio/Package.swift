// swift-tools-version:6.0
import PackageDescription

let package = Package(
    name: "GenerateAudio",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(path: "../.."),
    ],
    targets: [
        .executableTarget(
            name: "GenerateAudio",
            dependencies: [
                .product(name: "Nightingale", package: "swift"),
            ],
            path: "."
        ),
    ]
)
