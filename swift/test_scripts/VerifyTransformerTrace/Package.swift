// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "VerifyTransformerTrace",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(path: "../..")
    ],
    targets: [
        .executableTarget(
            name: "VerifyTransformerTrace",
            dependencies: [
                .product(name: "Nightingale", package: "swift")
            ],
            path: ".",
            swiftSettings: [
                .enableUpcomingFeature("BareSlashRegexLiterals")
            ]
        )
    ]
)
