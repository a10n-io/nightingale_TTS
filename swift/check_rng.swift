import Foundation
import MLX

MLXRandom.seed(42)
let r = MLXRandom.normal([1])
print("Swift: \(r.item(Float.self))")
