import Foundation
import MLX
import MLXNN

@main
struct TestReflectionPadding {
    static func main() async throws {
        print("Testing Swift reflection padding against PyTorch reference...")

        // Create test signal matching Python: [1.0, 2.0, 3.0, 4.0, 5.0]
        let testSignal = MLXArray([1.0, 2.0, 3.0, 4.0, 5.0]).expandedDimensions(axis: 0)  // [1, 5]
        let padLen = 2

        print("\nTest signal: \(Array(testSignal[0].asArray(Float.self)))")

        // Apply reflection padding
        let padded = reflectionPad1D(testSignal, padLen: padLen)

        let paddedValues = Array(padded[0].asArray(Float.self))
        print("Swift reflect-padded: \(paddedValues)")
        print("Expected from PyTorch: [3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0]")

        let expected: [Float] = [3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0]
        let match = zip(paddedValues, expected).allSatisfy { abs($0 - $1) < 0.0001 }

        if match {
            print("✅ Reflection padding MATCHES PyTorch!")
        } else {
            print("❌ Reflection padding MISMATCH!")
            print("   Difference: \(zip(paddedValues, expected).map { $0 - $1 })")
        }
    }

    // Copy of the reflection padding function from S3Gen.swift
    static func reflectionPad1D(_ x: MLXArray, padLen: Int) -> MLXArray {
        // x: [B, T]
        let T = x.shape[1]

        // Left padding: mirror positions 1 to padLen (excluding edge at 0)
        let leftRegion = x[0..., 1...(padLen)]  // [B, padLen] starting from index 1
        let leftIndices = MLXArray(Array((0..<padLen).reversed()))  // [padLen-1, ..., 1, 0]
        let leftPad = leftRegion[0..., leftIndices]  // Reverse it

        // Right padding: mirror positions (T-padLen-1) to (T-2) (excluding edge at T-1)
        let rightRegion = x[0..., (T-padLen-1)..<(T-1)]  // [B, padLen] ending before last element
        let rightIndices = MLXArray(Array((0..<padLen).reversed()))
        let rightPad = rightRegion[0..., rightIndices]  // Reverse it

        // Concatenate: [leftPad | x | rightPad]
        return concatenated([leftPad, x, rightPad], axis: 1)
    }
}
