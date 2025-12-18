#!/usr/bin/env swift

// Simple script to trace encoder sequence lengths at each stage

import Foundation

// Run the GenerateStep6Reference script which already goes through encoder
// and capture its output to see sequence lengths

let projectRoot = "/Users/a10n/Projects/nightingale/swift"
let scriptPath = "\(projectRoot)/test_scripts/GenerateStep6Reference"

print("=" + String(repeating: "=", count: 79))
print("SWIFT ENCODER - SEQUENCE LENGTH ANALYSIS")
print("=" + String(repeating: "=", count: 79))
print()
print("Running GenerateStep6Reference to trace encoder...")
print()

// Change to script directory and run it
let process = Process()
process.currentDirectoryURL = URL(fileURLWithPath: scriptPath)
process.executableURL = URL(fileURLWithPath: "/usr/bin/swift")
process.arguments = ["run"]

let outputPipe = Pipe()
let errorPipe = Pipe()
process.standardOutput = outputPipe
process.standardError = errorPipe

do {
    try process.run()
    process.waitUntilExit()

    let outputData = outputPipe.fileHandleForReading.readDataToEndOfFile()
    let errorData = errorPipe.fileHandleForReading.readDataToEndOfFile()

    if let output = String(data: outputData, encoding: .utf8) {
        print(output)
    }

    if let error = String(data: errorData, encoding: .utf8), !error.isEmpty {
        print("STDERR:")
        print(error)
    }

    if process.terminationStatus == 0 {
        print()
        print("✅ Script completed successfully")
    } else {
        print()
        print("❌ Script failed with exit code: \(process.terminationStatus)")
    }
} catch {
    print("❌ Failed to run script: \(error)")
}
