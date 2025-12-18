// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "VerifyStep7Velocity",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(path: "../..")
    ],
    targets: [
        .executableTarget(
            name: "VerifyStep7Velocity",
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
