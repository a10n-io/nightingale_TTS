// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "VerifyStep5S3GenEmbedding",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.21.1"),
        .package(path: "../../")
    ],
    targets: [
        .executableTarget(
            name: "VerifyStep5S3GenEmbedding",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "Nightingale", package: "swift")
            ],
            path: "."
        )
    ]
)
