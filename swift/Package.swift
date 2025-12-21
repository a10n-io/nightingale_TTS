// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "Nightingale",
    platforms: [
        .iOS(.v17),
        .macOS(.v14)
    ],
    products: [
        .library(
            name: "Nightingale",
            targets: ["Nightingale"]
        ),
        .executable(
            name: "TestLoadVoice",
            targets: ["TestLoadVoice"]
        ),
        .executable(
            name: "TestT3Generate",
            targets: ["TestT3Generate"]
        ),
        .executable(
            name: "TestS3GenVocoding",
            targets: ["TestS3GenVocoding"]
        ),
        .executable(
            name: "VerifyStep6S3GenEncoder",
            targets: ["VerifyStep6S3GenEncoder"]
        ),
        .executable(
            name: "VerifyE2ESteps1_7",
            targets: ["VerifyE2ESteps1_7"]
        ),
        .executable(
            name: "LiveFireTest",
            targets: ["LiveFireTest"]
        ),
        .executable(
            name: "VerifyLive",
            targets: ["VerifyLive"]
        ),
        .executable(
            name: "GenerateAudio",
            targets: ["GenerateAudio"]
        ),
        .executable(
            name: "GenerateAudioE2E",
            targets: ["GenerateAudioE2E"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.21.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "0.1.14"),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.2.0"),
    ],
    targets: [
        .target(
            name: "Nightingale",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXFFT", package: "mlx-swift"),
                .product(name: "Transformers", package: "swift-transformers"),
            ],
            path: "Sources/Nightingale"
        ),
        .executableTarget(
            name: "TestLoadVoice",
            dependencies: ["Nightingale"],
            path: "test_scripts/TestLoadVoice"
        ),
        .executableTarget(
            name: "TestT3Generate",
            dependencies: ["Nightingale"],
            path: "test_scripts/TestT3Generate"
        ),
        .executableTarget(
            name: "TestS3GenVocoding",
            dependencies: ["Nightingale"],
            path: "test_scripts/TestS3GenVocoding"
        ),
        .executableTarget(
            name: "VerifyStep6S3GenEncoder",
            dependencies: ["Nightingale"],
            path: "test_scripts/VerifyStep6S3GenEncoder"
        ),
        .executableTarget(
            name: "VerifyE2ESteps1_7",
            dependencies: ["Nightingale"],
            path: "test_scripts/VerifyE2ESteps1_7"
        ),
        .executableTarget(
            name: "LiveFireTest",
            dependencies: ["Nightingale"],
            path: "test_scripts/LiveFireTest"
        ),
        .executableTarget(
            name: "VerifyLive",
            dependencies: [
                "Nightingale",
                .product(name: "ArgumentParser", package: "swift-argument-parser")
            ],
            path: "test_scripts/VerifyLive"
        ),
        .executableTarget(
            name: "GenerateAudio",
            dependencies: ["Nightingale"],
            path: "test_scripts/GenerateAudio"
        ),
        .executableTarget(
            name: "GenerateAudioE2E",
            dependencies: [
                "Nightingale",
                .product(name: "ArgumentParser", package: "swift-argument-parser")
            ],
            path: "test_scripts/GenerateAudioE2E"
        ),
    ]
)
