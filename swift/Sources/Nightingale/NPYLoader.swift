import Foundation
import MLX

/// NPY file format loader for MLX
/// Supports loading NumPy array files (.npy) into MLXArray
public struct NPYLoader {

    /// NPY file header information
    struct NPYHeader {
        let majorVersion: UInt8
        let minorVersion: UInt8
        let headerLen: Int
        let descr: String
        let fortranOrder: Bool
        let shape: [Int]
    }

    /// Load an NPY file and return an MLXArray
    public static func load(contentsOf url: URL) throws -> MLXArray {
        let data = try Data(contentsOf: url)
        return try load(data: data)
    }

    /// Load NPY data and return an MLXArray
    public static func load(data: Data) throws -> MLXArray {
        // Parse header
        let header = try parseHeader(data: data)
        let headerSize = 10 + header.headerLen // magic(6) + version(2) + headerLen(2) + header

        // Get raw data
        let rawData = data.subdata(in: headerSize..<data.count)

        // Determine dtype and create MLXArray
        return try createArray(from: rawData, header: header)
    }

    private static func parseHeader(data: Data) throws -> NPYHeader {
        // Check magic number: \x93NUMPY
        guard data.count > 10 else {
            throw NPYError.invalidFormat("File too small")
        }

        let magic = data.prefix(6)
        guard magic[0] == 0x93,
              magic[1] == 0x4E, // N
              magic[2] == 0x55, // U
              magic[3] == 0x4D, // M
              magic[4] == 0x50, // P
              magic[5] == 0x59  // Y
        else {
            throw NPYError.invalidFormat("Invalid magic number")
        }

        let majorVersion = data[6]
        let minorVersion = data[7]

        // Header length (little endian)
        let headerLen: Int
        if majorVersion == 1 {
            headerLen = Int(data[8]) | (Int(data[9]) << 8)
        } else {
            // Version 2.0 uses 4 bytes for header length
            headerLen = Int(data[8]) | (Int(data[9]) << 8) | (Int(data[10]) << 16) | (Int(data[11]) << 24)
        }

        // Parse header string (Python dict literal)
        let headerStart = majorVersion == 1 ? 10 : 12
        let headerData = data.subdata(in: headerStart..<(headerStart + headerLen))
        guard let headerStr = String(data: headerData, encoding: .ascii) else {
            throw NPYError.invalidFormat("Cannot parse header")
        }

        // Extract dtype, fortran_order, shape from header
        let (descr, fortranOrder, shape) = try parseHeaderDict(headerStr)

        return NPYHeader(
            majorVersion: majorVersion,
            minorVersion: minorVersion,
            headerLen: headerLen,
            descr: descr,
            fortranOrder: fortranOrder,
            shape: shape
        )
    }

    private static func parseHeaderDict(_ header: String) throws -> (descr: String, fortranOrder: Bool, shape: [Int]) {
        // Simple parsing of Python dict: {'descr': '<f4', 'fortran_order': False, 'shape': (10, 20), }

        var descr = "<f4"
        var fortranOrder = false
        var shape: [Int] = []

        // Extract descr
        if let range = header.range(of: "'descr':\\s*'([^']+)'", options: .regularExpression) {
            let match = header[range]
            if let valueRange = match.range(of: "'[^']+'$", options: .regularExpression) {
                descr = String(match[valueRange]).trimmingCharacters(in: CharacterSet(charactersIn: "'"))
            }
        }

        // Extract fortran_order
        fortranOrder = header.contains("'fortran_order': True")

        // Extract shape
        if let range = header.range(of: "'shape':\\s*\\(([^)]+)\\)", options: .regularExpression) {
            let match = header[range]
            if let valueRange = match.range(of: "\\([^)]+\\)", options: .regularExpression) {
                let shapeStr = String(match[valueRange])
                    .trimmingCharacters(in: CharacterSet(charactersIn: "()"))
                    .replacingOccurrences(of: " ", with: "")

                if !shapeStr.isEmpty {
                    shape = shapeStr.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
                }
            }
        }

        return (descr, fortranOrder, shape)
    }

    private static func createArray(from data: Data, header: NPYHeader) throws -> MLXArray {
        let shape = header.shape
        let descr = header.descr

        // Determine element count
        let elementCount = shape.isEmpty ? 1 : shape.reduce(1, *)

        // Parse dtype
        // Format: <f4 (little-endian float32), <f8 (float64), <i4 (int32), etc.
        let byteOrder = descr.first ?? "<"
        let typeChar = descr.dropFirst().first ?? "f"
        let byteSize = Int(String(descr.dropFirst(2))) ?? 4

        switch typeChar {
        case "f": // float
            if byteSize == 4 {
                return loadFloat32(data: data, shape: shape, count: elementCount, littleEndian: byteOrder == "<")
            } else if byteSize == 8 {
                return loadFloat64(data: data, shape: shape, count: elementCount, littleEndian: byteOrder == "<")
            }
        case "i": // signed int
            if byteSize == 4 {
                return loadInt32(data: data, shape: shape, count: elementCount, littleEndian: byteOrder == "<")
            } else if byteSize == 8 {
                return loadInt64(data: data, shape: shape, count: elementCount, littleEndian: byteOrder == "<")
            }
        case "u": // unsigned int
            if byteSize == 4 {
                return loadUInt32(data: data, shape: shape, count: elementCount, littleEndian: byteOrder == "<")
            }
        default:
            break
        }

        throw NPYError.unsupportedDtype(descr)
    }

    private static func loadFloat32(data: Data, shape: [Int], count: Int, littleEndian: Bool) -> MLXArray {
        let floats = data.withUnsafeBytes { buffer -> [Float] in
            let ptr = buffer.bindMemory(to: Float.self)
            return Array(ptr.prefix(count))
        }
        return MLXArray(floats).reshaped(shape)
    }

    private static func loadFloat64(data: Data, shape: [Int], count: Int, littleEndian: Bool) -> MLXArray {
        let doubles = data.withUnsafeBytes { buffer -> [Double] in
            let ptr = buffer.bindMemory(to: Double.self)
            return Array(ptr.prefix(count))
        }
        // Convert to Float32 for MLX
        let floats = doubles.map { Float($0) }
        return MLXArray(floats).reshaped(shape)
    }

    private static func loadInt32(data: Data, shape: [Int], count: Int, littleEndian: Bool) -> MLXArray {
        let ints = data.withUnsafeBytes { buffer -> [Int32] in
            let ptr = buffer.bindMemory(to: Int32.self)
            return Array(ptr.prefix(count))
        }
        return MLXArray(ints).reshaped(shape)
    }

    private static func loadInt64(data: Data, shape: [Int], count: Int, littleEndian: Bool) -> MLXArray {
        let ints = data.withUnsafeBytes { buffer -> [Int64] in
            let ptr = buffer.bindMemory(to: Int64.self)
            return Array(ptr.prefix(count))
        }
        // Convert to Int32 for MLX
        let int32s = ints.map { Int32($0) }
        return MLXArray(int32s).reshaped(shape)
    }

    private static func loadUInt32(data: Data, shape: [Int], count: Int, littleEndian: Bool) -> MLXArray {
        let uints = data.withUnsafeBytes { buffer -> [UInt32] in
            let ptr = buffer.bindMemory(to: UInt32.self)
            return Array(ptr.prefix(count))
        }
        // Convert to Int32 for MLX
        let int32s = uints.map { Int32(bitPattern: $0) }
        return MLXArray(int32s).reshaped(shape)
    }

    enum NPYError: Error {
        case invalidFormat(String)
        case unsupportedDtype(String)
    }
}

// MARK: - Convenience Extensions

extension MLXArray {
    /// Load from NPY file
    public static func load(npy url: URL) throws -> MLXArray {
        try NPYLoader.load(contentsOf: url)
    }

    /// Load from NPY file path
    public static func load(npy path: String) throws -> MLXArray {
        try NPYLoader.load(contentsOf: URL(fileURLWithPath: path))
    }

    /// Save to NPY file
    public func save(npy url: URL) throws {
        // Convert to float array
        let floats = self.asArray(Float.self)
        let shape = self.shape

        // Create NPY header
        var header = "{'descr': '<f4', 'fortran_order': False, 'shape': ("
        header += shape.map { String($0) }.joined(separator: ", ")
        if shape.count == 1 {
            header += ","  // Python tuple with single element needs trailing comma
        }
        header += "), }"

        // Pad header to align data
        while (10 + header.count) % 64 != 0 {
            header += " "
        }
        header += "\n"

        var data = Data()

        // Magic number
        data.append(contentsOf: [0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59])  // \x93NUMPY

        // Version 1.0
        data.append(contentsOf: [0x01, 0x00])

        // Header length (little endian, 2 bytes)
        let headerLen = UInt16(header.count)
        data.append(UInt8(headerLen & 0xFF))
        data.append(UInt8((headerLen >> 8) & 0xFF))

        // Header
        data.append(header.data(using: .ascii)!)

        // Data (float32, little endian)
        for value in floats {
            var v = value
            data.append(contentsOf: withUnsafeBytes(of: &v) { Data($0) })
        }

        try data.write(to: url)
    }
}
