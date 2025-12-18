// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "VerifyStep2Conditioning",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.21.1"),
        .package(path: "../../")
    ],
    targets: [
        .executableTarget(
            name: "VerifyStep2Conditioning",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "Nightingale", package: "swift")
            ],
            path: "."
        )
    ]
)
