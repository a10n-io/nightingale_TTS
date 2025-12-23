import Foundation
import MLX

// Test MLX slicing to verify syntax
let arr = MLXArray(0..<10).reshaped([1, 1, 10])  // [1, 1, 10] with values 0-9
print("Full array: \(arr.asArray(Int32.self))")

// Test slicing from index 5 to end
let slice1 = arr[0..., 0..., 5...]
print("arr[0..., 0..., 5...]: \(slice1.asArray(Int32.self))")  // Should be [5,6,7,8,9]

// Test slicing from 0 to 5
let slice2 = arr[0..., 0..., 0..<5]
print("arr[0..., 0..., 0..<5]: \(slice2.asArray(Int32.self))")  // Should be [0,1,2,3,4]

// Test with real-like dimensions
let xt = MLXArray(-5..<5).asType(.float32).reshaped([1, 1, 10])  // [-5,-4,-3,-2,-1,0,1,2,3,4]
print("\nxt values: \(xt.asArray(Float.self))")
print("xt range: [\(xt.min().item(Float.self)), \(xt.max().item(Float.self))]")

let L_pm = 5
let generated = xt[0..., 0..., L_pm...]
print("\ngenerated = xt[0..., 0..., \(L_pm)...]")
print("generated values: \(generated.asArray(Float.self))")  // Should be [0,1,2,3,4]
print("generated range: [\(generated.min().item(Float.self)), \(generated.max().item(Float.self))]")
