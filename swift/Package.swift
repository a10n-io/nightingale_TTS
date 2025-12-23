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
            name: "GenerateAudio",
            targets: ["GenerateAudio"]
        ),
        .executable(
            name: "CrossValidate",
            targets: ["CrossValidate"]
        ),
        .executable(
            name: "DecoderTest",
            targets: ["DecoderTest"]
        ),
        .executable(
            name: "VocoderTest",
            targets: ["VocoderTest"]
        ),
        .executable(
            name: "VerifyEncoderProd",
            targets: ["VerifyEncoderProd"]
        ),
        .executable(
            name: "TestEncoderFix",
            targets: ["TestEncoderFix"]
        ),
        .executable(
            name: "SaveIntermediateEncoderStages",
            targets: ["SaveIntermediateEncoderStages"]
        ),
        .executable(
            name: "SaveDecoderMel",
            targets: ["SaveDecoderMel"]
        ),
        .executable(
            name: "CheckFixedNoise",
            targets: ["CheckFixedNoise"]
        ),
        .executable(
            name: "SaveDecoderIntermediates",
            targets: ["SaveDecoderIntermediates"]
        ),
        .executable(
            name: "SaveVocoderOutput",
            targets: ["SaveVocoderOutput"]
        ),
        .executable(
            name: "SaveVocoderCrossValidation",
            targets: ["SaveVocoderCrossValidation"]
        ),
        .executable(
            name: "TraceVocoderLayers",
            targets: ["TraceVocoderLayers"]
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
            name: "GenerateAudio",
            dependencies: ["Nightingale"],
            path: "test_scripts/GenerateAudio"
        ),
        .executableTarget(
            name: "CrossValidate",
            dependencies: ["Nightingale"],
            path: "test_scripts/CrossValidate"
        ),
        .executableTarget(
            name: "DecoderTest",
            dependencies: ["Nightingale"],
            path: "test_scripts/DecoderTest"
        ),
        .executableTarget(
            name: "VocoderTest",
            dependencies: ["Nightingale"],
            path: "test_scripts/VocoderTest"
        ),
        .executableTarget(
            name: "DeterministicTest",
            dependencies: ["Nightingale"],
            path: "test_scripts/DeterministicTest"
        ),
        .executableTarget(
            name: "VocoderShapeTest",
            dependencies: ["Nightingale"],
            path: "test_scripts/VocoderShapeTest"
        ),
        .executableTarget(
            name: "SaveSwiftMel",
            dependencies: ["Nightingale"],
            path: "test_scripts",
            sources: ["SaveSwiftMel.swift"]
        ),
        .executableTarget(
            name: "SaveSwiftMelForensic",
            dependencies: ["Nightingale"],
            path: "test_scripts",
            sources: ["SaveSwiftMelForensic.swift"]
        ),
        .executableTarget(
            name: "SaveEncoderOutput",
            dependencies: ["Nightingale"],
            path: "test_scripts",
            sources: ["SaveEncoderOutput.swift"]
        ),
        .executableTarget(
            name: "VerifyEncoderProd",
            dependencies: ["Nightingale"],
            path: "test_scripts",
            sources: ["VerifyEncoderProd.swift"]
        ),
        .executableTarget(
            name: "TestEncoderFix",
            dependencies: ["Nightingale"],
            path: "test_scripts",
            sources: ["TestEncoderFix.swift"]
        ),
        .executableTarget(
            name: "SaveIntermediateEncoderStages",
            dependencies: ["Nightingale"],
            path: "test_scripts",
            sources: ["SaveIntermediateEncoderStages.swift"]
        ),
        .executableTarget(
            name: "SaveDecoderMel",
            dependencies: ["Nightingale"],
            path: "test_scripts",
            sources: ["SaveDecoderMel.swift"]
        ),
        .executableTarget(
            name: "CheckFixedNoise",
            dependencies: ["Nightingale"],
            path: "test_scripts",
            sources: ["CheckFixedNoise.swift"]
        ),
        .executableTarget(
            name: "SaveDecoderIntermediates",
            dependencies: ["Nightingale"],
            path: "test_scripts",
            sources: ["SaveDecoderIntermediates.swift"]
        ),
        .executableTarget(
            name: "SaveVocoderOutput",
            dependencies: ["Nightingale"],
            path: "test_scripts",
            sources: ["SaveVocoderOutput.swift"]
        ),
        .executableTarget(
            name: "SaveVocoderCrossValidation",
            dependencies: ["Nightingale"],
            path: "test_scripts",
            sources: ["SaveVocoderCrossValidation.swift"]
        ),
        .executableTarget(
            name: "TraceVocoderLayers",
            dependencies: ["Nightingale"],
            path: "test_scripts",
            sources: ["TraceVocoderLayers.swift"]
        ),
        .executableTarget(
            name: "CheckEmbedLinearWeights",
            dependencies: ["Nightingale"],
            path: "test_scripts",
            sources: ["CheckEmbedLinearWeights.swift"]
        ),
    ]
)
