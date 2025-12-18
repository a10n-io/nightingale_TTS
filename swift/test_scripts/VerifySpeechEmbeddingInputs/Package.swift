// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "VerifySpeechEmbeddingInputs",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.21.1"),
        .package(path: "../../")
    ],
    targets: [
        .executableTarget(
            name: "VerifySpeechEmbeddingInputs",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "Nightingale", package: "swift")
            ],
            path: "."
        )
    ]
)
