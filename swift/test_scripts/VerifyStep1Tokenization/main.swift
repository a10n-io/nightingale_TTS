import Foundation
import MLX

// MARK: - Tokenization

func loadVocab(from url: URL) throws -> ([String: Int], [(String, String)]) {
    let data = try Data(contentsOf: url)
    let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
    guard let model = json?["model"] as? [String: Any],
          let vocabDict = model["vocab"] as? [String: Int] else {
        fatalError("Invalid tokenizer.json format")
    }

    // Load BPE merges
    var merges: [(String, String)] = []
    if let mergeStrings = model["merges"] as? [String] {
        for mergeStr in mergeStrings {
            let parts = mergeStr.split(separator: " ", maxSplits: 1)
            if parts.count == 2 {
                merges.append((String(parts[0]), String(parts[1])))
            }
        }
    }
    print("Loaded BPE vocab with \(vocabDict.count) tokens and \(merges.count) merge rules")
    return (vocabDict, merges)
}

func bpeEncode(word: String, vocab: [String: Int], merges: [(String, String)]) -> [Int] {
    // Start with individual characters
    var symbols = word.map { String($0) }

    // Create a set of merge rules for fast lookup, preserving order via index
    var mergeRanks: [String: Int] = [:]
    for (index, merge) in merges.enumerated() {
        let key = "\(merge.0) \(merge.1)"
        mergeRanks[key] = index
    }

    // Apply merges iteratively
    while symbols.count > 1 {
        // Find the pair with the lowest merge rank
        var bestPair: (Int, String, String)? = nil  // (index, left, right)
        var bestRank = Int.max

        for i in 0..<(symbols.count - 1) {
            let pair = "\(symbols[i]) \(symbols[i + 1])"
            if let rank = mergeRanks[pair], rank < bestRank {
                bestRank = rank
                bestPair = (i, symbols[i], symbols[i + 1])
            }
        }

        // If no merge found, we're done
        guard let (idx, left, right) = bestPair else { break }

        // Apply the merge
        symbols[idx] = left + right
        symbols.remove(at: idx + 1)
    }

    // Convert symbols to token IDs
    var tokenIds: [Int] = []
    for symbol in symbols {
        if let tokenId = vocab[symbol] {
            tokenIds.append(tokenId)
        } else {
            // Unknown token - use [UNK] = 1
            tokenIds.append(1)
        }
    }

    return tokenIds
}

func tokenize(_ text: String, vocab: [String: Int], merges: [(String, String)]) -> [Int] {
    var tokens: [Int] = []
    let words = text.components(separatedBy: .whitespaces).filter { !$0.isEmpty }

    for word in words {
        let wordTokens = bpeEncode(word: word, vocab: vocab, merges: merges)
        tokens.append(contentsOf: wordTokens)
    }

    return tokens
}

// MARK: - NPY Loader

struct NPYLoader {
    static func load(contentsOf url: URL) throws -> MLXArray {
        let data = try Data(contentsOf: url)
        var offset = 0

        // Parse magic string
        guard data.count >= 6 else { throw NPYError.invalidFormat }
        let magic = data[0..<6]
        guard magic[0] == 0x93 && magic[1] == 0x4E && magic[2] == 0x55 &&
              magic[3] == 0x4D && magic[4] == 0x50 && magic[5] == 0x59 else {
            throw NPYError.invalidMagic
        }
        offset = 6

        // Parse version
        guard data.count >= offset + 2 else { throw NPYError.invalidFormat }
        let major = data[offset]
        let minor = data[offset + 1]
        offset += 2

        // Parse header length
        var headerLen: Int
        if major == 1 {
            guard data.count >= offset + 2 else { throw NPYError.invalidFormat }
            headerLen = Int(data[offset]) | (Int(data[offset + 1]) << 8)
            offset += 2
        } else if major == 2 || major == 3 {
            guard data.count >= offset + 4 else { throw NPYError.invalidFormat }
            headerLen = Int(data[offset]) | (Int(data[offset + 1]) << 8) |
                       (Int(data[offset + 2]) << 16) | (Int(data[offset + 3]) << 24)
            offset += 4
        } else {
            throw NPYError.unsupportedVersion
        }

        // Parse header dict
        guard data.count >= offset + headerLen else { throw NPYError.invalidFormat }
        let headerData = data[offset..<(offset + headerLen)]
        guard let headerStr = String(data: headerData, encoding: .utf8) else {
            throw NPYError.invalidFormat
        }
        offset += headerLen

        // Parse shape
        guard let shapeMatch = headerStr.range(of: "'shape':\\s*\\(([^)]*)\\)", options: .regularExpression) else {
            throw NPYError.invalidFormat
        }
        let shapeStr = String(headerStr[shapeMatch]).replacingOccurrences(of: "'shape': (", with: "").replacingOccurrences(of: ")", with: "")
        let shape = shapeStr.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }

        // Parse dtype
        guard let dtypeMatch = headerStr.range(of: "'descr':\\s*'([^']*)'", options: .regularExpression) else {
            throw NPYError.invalidFormat
        }
        let dtypeStr = String(headerStr[dtypeMatch]).replacingOccurrences(of: "'descr': '", with: "").replacingOccurrences(of: "'", with: "")

        // Load array data
        let arrayData = data[offset...]
        let array: MLXArray

        if dtypeStr.hasSuffix("f4") {
            let count = arrayData.count / MemoryLayout<Float>.size
            let floats = arrayData.withUnsafeBytes { $0.bindMemory(to: Float.self) }
            array = MLXArray(Array(floats.prefix(count)))
        } else if dtypeStr.hasSuffix("f8") {
            let count = arrayData.count / MemoryLayout<Double>.size
            let doubles = arrayData.withUnsafeBytes { $0.bindMemory(to: Double.self) }
            let floats = doubles.prefix(count).map { Float($0) }
            array = MLXArray(floats)
        } else if dtypeStr.hasSuffix("i4") {
            let count = arrayData.count / MemoryLayout<Int32>.size
            let ints = arrayData.withUnsafeBytes { $0.bindMemory(to: Int32.self) }
            array = MLXArray(Array(ints.prefix(count)))
        } else if dtypeStr.hasSuffix("i8") {
            let count = arrayData.count / MemoryLayout<Int64>.size
            let longs = arrayData.withUnsafeBytes { $0.bindMemory(to: Int64.self) }
            let ints = longs.prefix(count).map { Int32($0) }
            array = MLXArray(ints)
        } else {
            throw NPYError.unsupportedDtype(dtypeStr)
        }

        return shape.isEmpty ? array : array.reshaped(shape)
    }

    enum NPYError: Error {
        case invalidFormat
        case invalidMagic
        case unsupportedVersion
        case unsupportedDtype(String)
    }
}

// MARK: - Paths

// Project root is 4 levels up from this script: swift/test_scripts/VerifyStep1Tokenization/
let projectRoot = URL(fileURLWithPath: #file)
    .deletingLastPathComponent()  // Remove main.swift
    .deletingLastPathComponent()  // Remove VerifyStep1Tokenization
    .deletingLastPathComponent()  // Remove test_scripts
    .deletingLastPathComponent()  // Remove swift
    .path

// MARK: - Main

let testText = "Hello world"
let tokenizerPath = "\(projectRoot)/models/tokenizer.json"
let refPath = "\(projectRoot)/verification_outputs/step1/tokens.npy"

print(String(repeating: "=", count: 80))
print("STEP 1: TEXT TOKENIZATION")
print(String(repeating: "=", count: 80))
print("Text: \"\(testText)\"")

do {
    let tokenizerURL = URL(fileURLWithPath: tokenizerPath)
    let (vocab, merges) = try loadVocab(from: tokenizerURL)
    let swiftTokens = tokenize(testText, vocab: vocab, merges: merges)

    let refURL = URL(fileURLWithPath: refPath)
    let refArray = try NPYLoader.load(contentsOf: refURL)
    let pythonTokens = refArray.asArray(Int32.self).map { Int($0) }

    print("Swift:  \(swiftTokens)")
    print("Python: \(pythonTokens)")

    var allMatch = swiftTokens.count == pythonTokens.count
    for i in 0..<max(swiftTokens.count, pythonTokens.count) {
        let swift = i < swiftTokens.count ? swiftTokens[i] : -1
        let python = i < pythonTokens.count ? pythonTokens[i] : -1
        if swift != python {
            print("❌ Token[\(i)]: Swift=\(swift), Python=\(python)")
            allMatch = false
        }
    }

    print(String(repeating: "=", count: 80))
    if allMatch {
        print("✅ PASSED (max_diff = 0.0)")
    } else {
        print("❌ FAILED")
        exit(1)
    }
    print(String(repeating: "=", count: 80))

} catch {
    print("❌ ERROR: \(error)")
    exit(1)
}
