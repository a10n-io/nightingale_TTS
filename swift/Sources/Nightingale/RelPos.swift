import MLX
import MLXNN
import MLXRandom
import Foundation

// MARK: - Relative Positional Encoding

/// Relative Positional Encoding (ESPNet style)
public class EspnetRelPositionalEncoding: Module {
    public let dModel: Int
    public let dropoutRate: Float
    public let xscale: Float
    
    // We keep pe as state, typically regular PE just computes it.
    // ESPNet implementation extends it.
    // Non-optional so Module.update() can load weights into it
    public var pe: MLXArray
    
    public init(dModel: Int, dropoutRate: Float = 0.1) {
        self.dModel = dModel
        self.dropoutRate = dropoutRate
        self.xscale = sqrt(Float(dModel))
        // Initialize with empty pe - will be extended on first use or loaded from weights
        self.pe = MLXArray.zeros([1, 1, dModel])
        super.init()
    }
    
    private func extendPe(size: Int) {
        if pe.shape[1] >= size * 2 - 1 {
            return
        }
        
        let position = MLXArray(Array(0..<size).map { Float($0) }).expandedDimensions(axis: 1) // [size, 1]
        let divTerm = exp(
            MLXArray(stride(from: 0, to: dModel, by: 2).map { Float($0) }) *
            -(log(10000.0) / Float(dModel))
        )
        
        // Positive positions
        let posSin = sin(position * divTerm)
        let posCos = cos(position * divTerm)
        
        // Interleave manually: 0::2 = sin, 1::2 = cos
        // MLX doesn't support sophisticated slicing assignment easily in one go for strides
        // Construct interleaved array
        // [size, dModel/2] each
        // Stack along last axis [size, dModel/2, 2] then flatten?
        let posStack = stacked([posSin, posCos], axis: 2).reshaped([size, dModel])
        let pePositive = posStack
        
        // Negative positions
        let negPosition = position * -1
        let negSin = sin(negPosition * divTerm)
        let negCos = cos(negPosition * divTerm)
        let negStack = stacked([negSin, negCos], axis: 2).reshaped([size, dModel])
        let peNegative = negStack
        
        // Reverse positive: [0, 1, 2] -> [2, 1, 0]
        // pePositive is [T, D]
        // Create indices for reversal
        let revIndices = MLXArray(Array(stride(from: size - 1, through: 0, by: -1)).map { Int32($0) })
        let pePosRev = pePositive[revIndices]
        
        // Negative: skip first (0), take rest
        // peNegative is [size, dModel]
        let peNegRest = peNegative[1...]
        
        // Concatenate: [PosRev, NegRest] -> [2*T - 1, D]
        let peFull = concatenated([pePosRev, peNegRest], axis: 0)
        
        // Add batch dim [1, 2*T-1, D]
        self.pe = peFull.expandedDimensions(axis: 0)
    }
    
    public static var posEncDebug: Bool = false

    public func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        // x: [B, T, D]
        let size = x.shape[1]

        if Self.posEncDebug {
            print("EspnetRelPos DEBUG: x.shape = \(x.shape), dModel=\(dModel)"); fflush(stdout)
        }

        extendPe(size: size)

        if Self.posEncDebug {
            print("EspnetRelPos DEBUG: after extendPe, pe.shape = \(pe.shape)"); fflush(stdout)
            print("EspnetRelPos DEBUG: about to compute xScaled = x * xscale (xscale=\(xscale))"); fflush(stdout)
        }

        let xScaled = x * xscale

        if Self.posEncDebug {
            print("EspnetRelPos DEBUG: xScaled computed, shape = \(xScaled.shape)"); fflush(stdout)
        }

        // pos_emb: [1, 2*T-1, D] center crop
        let center = pe.shape[1] / 2
        let start = center - size + 1
        let end = center + size

        if Self.posEncDebug {
            print("EspnetRelPos DEBUG: center=\(center), start=\(start), end=\(end)"); fflush(stdout)
            print("EspnetRelPos DEBUG: about to slice pe[\(0)..<\(pe.shape[0]), \(start)..<\(end), \(0)..<\(pe.shape[2])]"); fflush(stdout)
        }

        // Use Range for slicing
        let posEmb = pe[0..., start..<end, 0...]

        if Self.posEncDebug {
            print("EspnetRelPos DEBUG: posEmb computed, shape = \(posEmb.shape)"); fflush(stdout)
            print("EspnetRelPos DEBUG: forcing eval of xScaled..."); fflush(stdout)
            eval(xScaled)
            print("EspnetRelPos DEBUG: xScaled eval done"); fflush(stdout)
            print("EspnetRelPos DEBUG: forcing eval of posEmb..."); fflush(stdout)
            eval(posEmb)
            print("EspnetRelPos DEBUG: posEmb eval done"); fflush(stdout)
        }

        return (xScaled, posEmb)
    }
}

// MARK: - Relative Multi-Head Attention

public class RelPositionMultiHeadAttention: Module {
    let numHeads: Int
    let dModel: Int
    let dHead: Int
    let scale: Float

    // Use same property names as remapping expects: linear_q -> queryProj, etc.
    public let queryProj: Linear
    public let keyProj: Linear
    public let valueProj: Linear
    public let outProj: Linear

    // RelPos specific
    public let linearPos: Linear
    public var posBiasU: MLXArray // [H, D_h] - var so weights can be loaded
    public var posBiasV: MLXArray // [H, D_h] - var so weights can be loaded

    public init(dModel: Int, numHeads: Int, dropout: Float = 0.1) {
        self.dModel = dModel
        self.numHeads = numHeads
        self.dHead = dModel / numHeads
        self.scale = 1.0 / sqrt(Float(dHead))

        self.queryProj = Linear(dModel, dModel)
        self.keyProj = Linear(dModel, dModel)
        self.valueProj = Linear(dModel, dModel)
        self.outProj = Linear(dModel, dModel)

        self.linearPos = Linear(dModel, dModel, bias: false)

        // Initialize biases
        self.posBiasU = MLXRandom.uniform(low: -1, high: 1, [numHeads, dHead])
        self.posBiasV = MLXRandom.uniform(low: -1, high: 1, [numHeads, dHead])

        super.init()
    }
    
    private func relShift(_ x: MLXArray) -> MLXArray {
        // x: [B, H, T, 2*T-1]
        let B = x.shape[0]
        let H = x.shape[1]
        let T = x.shape[2]
        
        // Prepend column of zeros: [B, H, T, 2*T]
        let zeroPad = MLXArray.zeros([B, H, T, 1], dtype: x.dtype)
        let xPad = concatenated([zeroPad, x], axis: -1) // [B, H, T, 2T]
        
        let xR = xPad.reshaped([B, H, 2 * T, T])
        // Slice: drop first "row"
        let xS = xR[0..., 0..., 1...] // [B, H, 2T-1, T] 
        let xFinalFull = xS.reshaped([B, H, T, 2 * T - 1])
        
        // Take first T columns
        return xFinalFull[0..., 0..., 0..., 0..<T]
    }
    
    public static var debugEnabled: Bool = false

    public func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil, posEmb: MLXArray) -> MLXArray {
        let B = x.shape[0]
        let T = x.shape[1]
        let D = x.shape[2]

        if Self.debugEnabled {
            print("ATTENTION DEBUG:")
            print("  Input x: \(x.shape) (expected [B, T, 512])")
            print("  posEmb: \(posEmb.shape) (expected [1, 2*T-1, 512])")
            print("  numHeads=\(numHeads), dHead=\(dHead), dModel=\(dModel)")
            print("  queryProj.weight: \(queryProj.weight.shape)")
            print("  linearPos.weight: \(linearPos.weight.shape)")
            print("  posBiasU: \(posBiasU.shape)")

            // Check dimensions
            if D != dModel {
                print("  ❌ WARNING: x dimension \(D) != dModel \(dModel)!")
            }
            if posEmb.shape[2] != dModel {
                print("  ❌ WARNING: posEmb dimension \(posEmb.shape[2]) != dModel \(dModel)!")
            }
        }

        let q = queryProj(x).reshaped([B, T, numHeads, dHead]).transposed(0, 2, 1, 3) // [B, H, T, D_h]
        let k = keyProj(x).reshaped([B, T, numHeads, dHead]).transposed(0, 2, 1, 3)
        let v = valueProj(x).reshaped([B, T, numHeads, dHead]).transposed(0, 2, 1, 3)

        if Self.debugEnabled {
            print("  After Q/K/V projection: q=\(q.shape), k=\(k.shape), v=\(v.shape)")
        }

        // Pos embeddings
        let nBatchPos = posEmb.shape[0]
        let posEmbProj = linearPos(posEmb)
        if Self.debugEnabled {
            print("  After linearPos: \(posEmbProj.shape)")
        }
        let p = posEmbProj.reshaped([nBatchPos, -1, numHeads, dHead]).transposed(0, 2, 1, 3)
        if Self.debugEnabled {
            print("  After reshape/transpose p: \(p.shape)")
        }

        // Add biases
        let biasUReshaped = posBiasU.reshaped([1, numHeads, 1, dHead])
        let biasVReshaped = posBiasV.reshaped([1, numHeads, 1, dHead])
        if Self.debugEnabled {
            print("  biasU reshaped: \(biasUReshaped.shape)")
            print("  biasV reshaped: \(biasVReshaped.shape)")
            print("  About to add q + biasU: q=\(q.shape), biasU=\(biasUReshaped.shape)")
        }
        let qWithBiasU = q + biasUReshaped
        let qWithBiasV = q + biasVReshaped

        // Attention Scores
        let matrixAC = matmul(qWithBiasU, k.transposed(0, 1, 3, 2)) // [B, H, T, T]
        let matrixBD = matmul(qWithBiasV, p.transposed(0, 1, 3, 2)) // [B, H, T, 2T-1]

        // RelShift matrixBD
        let matrixBDBshifted = relShift(matrixBD) // [B, H, T, T]

        var scores = (matrixAC + matrixBDBshifted) * scale

        if let m = mask {
            scores = scores + m
        }

        let probs = softmax(scores, axis: -1)

        var output = matmul(probs, v) // [B, H, T, D_h]
        output = output.transposed(0, 2, 1, 3).reshaped([B, T, dModel])

        return outProj(output)
    }

    /// Load weights for relative position attention
    public func load(weights: [String: MLXArray], prefix: String) {
        // Load Q, K, V projections
        if let w = weights["\(prefix).linear_q.weight"] {
            queryProj.update(parameters: ModuleParameters.unflattened(["weight": w]))
        }
        if let b = weights["\(prefix).linear_q.bias"] {
            queryProj.update(parameters: ModuleParameters.unflattened(["bias": b]))
        }
        if let w = weights["\(prefix).linear_k.weight"] {
            keyProj.update(parameters: ModuleParameters.unflattened(["weight": w]))
        }
        if let b = weights["\(prefix).linear_k.bias"] {
            keyProj.update(parameters: ModuleParameters.unflattened(["bias": b]))
        }
        if let w = weights["\(prefix).linear_v.weight"] {
            valueProj.update(parameters: ModuleParameters.unflattened(["weight": w]))
        }
        if let b = weights["\(prefix).linear_v.bias"] {
            valueProj.update(parameters: ModuleParameters.unflattened(["bias": b]))
        }

        // Load output projection
        if let w = weights["\(prefix).linear_out.weight"] {
            outProj.update(parameters: ModuleParameters.unflattened(["weight": w]))
        }
        if let b = weights["\(prefix).linear_out.bias"] {
            outProj.update(parameters: ModuleParameters.unflattened(["bias": b]))
        }

        // Load positional projection (no bias)
        if let w = weights["\(prefix).linear_pos.weight"] {
            linearPos.update(parameters: ModuleParameters.unflattened(["weight": w]))
        }

        // Load positional biases
        if let u = weights["\(prefix).pos_bias_u"] {
            posBiasU = u
        }
        if let v = weights["\(prefix).pos_bias_v"] {
            posBiasV = v
        }
    }
}
