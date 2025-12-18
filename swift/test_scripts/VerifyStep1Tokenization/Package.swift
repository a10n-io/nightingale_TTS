// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "VerifyStep1Tokenization",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.21.1")
    ],
    targets: [
        .executableTarget(
            name: "VerifyStep1Tokenization",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "."
        )
    ]
)
