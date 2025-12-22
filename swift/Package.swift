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
        .executable(
            name: "TestVocoder",
            targets: ["TestVocoder"]
        ),
        .executable(
            name: "VerifyDecoderLayerByLayer",
            targets: ["VerifyDecoderLayerByLayer"]
        ),
        .executable(
            name: "VerifyBlock1Detail",
            targets: ["VerifyBlock1Detail"]
        ),
        .executable(
            name: "VerifyDown0Detail",
            targets: ["VerifyDown0Detail"]
        ),
        .executable(
            name: "VerifyFirstTransformer",
            targets: ["VerifyFirstTransformer"]
        ),
        .executable(
            name: "TestLayerNorm",
            targets: ["TestLayerNorm"]
        ),
        .executable(
            name: "VerifyNorm1Manual",
            targets: ["VerifyNorm1Manual"]
        ),
        .executable(
            name: "DebugMLXStats",
            targets: ["DebugMLXStats"]
        ),
        .executable(
            name: "VerifyDecoderDownBlock",
            targets: ["VerifyDecoderDownBlock"]
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
        .executableTarget(
            name: "TestVocoder",
            dependencies: ["Nightingale"],
            path: "test_scripts/TestVocoder"
        ),
        .executableTarget(
            name: "VerifyDecoderLayerByLayer",
            dependencies: ["Nightingale"],
            path: "test_scripts/VerifyDecoderLayerByLayer"
        ),
        .executableTarget(
            name: "VerifyBlock1Detail",
            dependencies: ["Nightingale"],
            path: "test_scripts/VerifyBlock1Detail"
        ),
        .executableTarget(
            name: "VerifyDown0Detail",
            dependencies: ["Nightingale"],
            path: "test_scripts/VerifyDown0Detail"
        ),
        .executableTarget(
            name: "VerifyFirstTransformer",
            dependencies: ["Nightingale"],
            path: "test_scripts/VerifyFirstTransformer"
        ),
        .executableTarget(
            name: "TestLayerNorm",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ],
            path: "test_scripts/TestLayerNorm"
        ),
        .executableTarget(
            name: "VerifyNorm1Manual",
            dependencies: ["Nightingale"],
            path: "test_scripts/VerifyNorm1Manual"
        ),
        .executableTarget(
            name: "DebugMLXStats",
            dependencies: ["Nightingale"],
            path: "test_scripts/DebugMLXStats"
        ),
        .executableTarget(
            name: "VerifyDecoderDownBlock",
            dependencies: ["Nightingale"],
            path: "test_scripts/VerifyDecoderDownBlock"
        ),
    ]
)
