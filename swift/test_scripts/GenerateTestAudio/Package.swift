// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "GenerateTestAudio",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(path: "../../")
    ],
    targets: [
        .executableTarget(
            name: "GenerateTestAudio",
            dependencies: [
                .product(name: "Nightingale", package: "swift")
            ]
        )
    ]
)
