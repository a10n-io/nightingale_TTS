import Foundation
import MLX
import MLXRandom
import MLXNN
import Nightingale

let PROJECT_ROOT = "/Users/a10n/Projects/nightingale_TTS"
let modelDir = URL(fileURLWithPath: "\(PROJECT_ROOT)/models/chatterbox")
let refDir = "\(PROJECT_ROOT)/E2E/reference_outputs/samantha/expressive_surprise_en"

print("=" + String(repeating: "=", count: 79))
print("DOWN BLOCK 0 DETAILED VERIFICATION")
print("=" + String(repeating: "=", count: 79))

// Helper to load .npy files
func loadNpy(_ path: String) throws -> MLXArray {
    return try NPYLoader.load(contentsOf: URL(fileURLWithPath: path))
}

// Load Python reference intermediate values
print("\n📥 Loading Python reference values...")
// Use the detailed trace files which have correct masked outputs
let pyTimeEmbRaw = try loadNpy("\(refDir)/dec_trace_time_emb_raw.npy")
let pyTimeEmb = try loadNpy("\(refDir)/dec_trace_time_emb.npy")
let pyHConcat = try loadNpy("\(refDir)/detailed_step2_h_concat.npy")
let pyAfterBlock1 = try loadNpy("\(refDir)/detailed_step5d_block1_output.npy")  // Use detailed trace
let pyMlpOut = try loadNpy("\(refDir)/dec_trace_down0_mlp_out.npy")
let pyWithTime = try loadNpy("\(refDir)/dec_trace_down0_with_time.npy")
let pyAfterBlock2 = try loadNpy("\(refDir)/dec_trace_down0_after_block2.npy")
let pyResConv = try loadNpy("\(refDir)/dec_trace_down0_res_conv.npy")
let pyFinalOut = try loadNpy("\(refDir)/dec_trace_down0_resnet_out.npy")

let maskFull = try loadNpy("\(refDir)/step7_cond_T.npy")
let mask = maskFull[0...0]  // Take only first batch element: [1, 1, 696]
let t = try loadNpy("\(refDir)/step7_step1_t.npy")

eval(pyTimeEmbRaw, pyTimeEmb, pyHConcat, pyAfterBlock1, pyMlpOut, pyWithTime, pyAfterBlock2, pyResConv, pyFinalOut, mask, t)
print("✅ Loaded Python reference values")

// Load decoder weights
print("\n📦 Loading decoder weights...")
let decoderURL = modelDir.appendingPathComponent("decoder_weights.safetensors")
let weights = try MLX.loadArrays(url: decoderURL)
print("  Loaded \(weights.count) tensors")

// Create TimeMLP
print("\n🔧 Creating TimeMLP...")
let timeMlp = TimeMLP(inputDim: 320, embDim: 1024)

// Load TimeMLP weights (need to transpose from PyTorch [out, in] to MLX [in, out])
let mlp1W_raw = weights["s3gen.flow.decoder.estimator.time_mlp.linear_1.weight"]!
let mlp1W = mlp1W_raw.T  // [1024, 320] -> [320, 1024]
let mlp1B = weights["s3gen.flow.decoder.estimator.time_mlp.linear_1.bias"]!
let mlp2W_raw = weights["s3gen.flow.decoder.estimator.time_mlp.linear_2.weight"]!
let mlp2W = mlp2W_raw.T  // [1024, 1024] -> [1024, 1024]
let mlp2B = weights["s3gen.flow.decoder.estimator.time_mlp.linear_2.bias"]!

timeMlp.linear1.weight = mlp1W
timeMlp.linear1.bias = mlp1B
timeMlp.linear2.weight = mlp2W
timeMlp.linear2.bias = mlp2B
print("✅ TimeMLP weights loaded")

// Create ResNet block
print("\n🔧 Creating CausalResNetBlock...")
let resnet = CausalResNetBlock(dim: 320, dimOut: 256, timeEmbDim: 1024)

// Load ResNet weights
let prefix = "s3gen.flow.decoder.estimator.down_blocks.0.0"

// Block1
let b1_conv_w_raw = weights["\(prefix).block1.block.0.weight"]!
let b1_conv_w = b1_conv_w_raw.swappedAxes(1, 2)  // PyTorch -> MLX format
let b1_conv_b = weights["\(prefix).block1.block.0.bias"]!
let b1_norm_w = weights["\(prefix).block1.block.2.weight"]!
let b1_norm_b = weights["\(prefix).block1.block.2.bias"]!

resnet.block1.conv.update(parameters: ModuleParameters.unflattened(["conv.weight": b1_conv_w, "conv.bias": b1_conv_b]))
resnet.block1.norm.update(parameters: ModuleParameters.unflattened(["weight": b1_norm_w, "bias": b1_norm_b]))

// MLP (transpose from PyTorch [out, in] to MLX [in, out])
let mlp_w_raw = weights["\(prefix).mlp.1.weight"]!
let mlp_w = mlp_w_raw.T  // [256, 1024] -> [1024, 256]
let mlp_b = weights["\(prefix).mlp.1.bias"]!
resnet.mlpLinear.weight = mlp_w
resnet.mlpLinear.bias = mlp_b

// Block2
let b2_conv_w_raw = weights["\(prefix).block2.block.0.weight"]!
let b2_conv_w = b2_conv_w_raw.swappedAxes(1, 2)  // PyTorch -> MLX format
let b2_conv_b = weights["\(prefix).block2.block.0.bias"]!
let b2_norm_w = weights["\(prefix).block2.block.2.weight"]!
let b2_norm_b = weights["\(prefix).block2.block.2.bias"]!

resnet.block2.conv.update(parameters: ModuleParameters.unflattened(["conv.weight": b2_conv_w, "conv.bias": b2_conv_b]))
resnet.block2.norm.update(parameters: ModuleParameters.unflattened(["weight": b2_norm_w, "bias": b2_norm_b]))

// ResConv (regular Conv1d, not CausalConv1d)
let res_conv_w_raw = weights["\(prefix).res_conv.weight"]!
let res_conv_w = res_conv_w_raw.swappedAxes(1, 2)  // PyTorch -> MLX format
let res_conv_b = weights["\(prefix).res_conv.bias"]!
resnet.resConv.update(parameters: ModuleParameters.unflattened(["weight": res_conv_w, "bias": res_conv_b]))

print("✅ ResNet weights loaded")

// Test 1: Time embedding
print("\n" + String(repeating: "=", count: 80))
print("TEST 1: TIME EMBEDDING")
print(String(repeating: "=", count: 80))

// TimeMLP applies sinusoidal embedding internally
eval(t, mlp1W, mlp1B, mlp2W, mlp2B)
print("t shape: \(t.shape)")
print("mlp1W shape: \(mlp1W.shape)")
print("mlp1B shape: \(mlp1B.shape)")
print("mlp2W shape: \(mlp2W.shape)")
print("mlp2B shape: \(mlp2B.shape)")
fflush(stdout)

let swiftTimeEmb = timeMlp(t)
eval(swiftTimeEmb)
print("Python time emb: [\(pyTimeEmb.min().item(Float.self)), \(pyTimeEmb.max().item(Float.self))]")
print("Swift  time emb: [\(swiftTimeEmb.min().item(Float.self)), \(swiftTimeEmb.max().item(Float.self))]")
let diffTimeEmb = swiftTimeEmb - pyTimeEmb
eval(diffTimeEmb)
print("Difference:      max abs = \(diffTimeEmb.abs().max().item(Float.self))")

// Test 2: After block1
print("\n" + String(repeating: "=", count: 80))
print("TEST 2: AFTER BLOCK1")
print(String(repeating: "=", count: 80))

print("pyHConcat shape: \(pyHConcat.shape)")
eval(pyHConcat)
print("pyHConcat range: [\(pyHConcat.min().item(Float.self)), \(pyHConcat.max().item(Float.self))]")
print("mask shape: \(mask.shape)")
print("mask sum: \(mask.sum().item(Float.self))")

let swiftAfterBlock1 = resnet.block1(pyHConcat, mask: mask)
eval(swiftAfterBlock1)
print("Python after block1: [\(pyAfterBlock1.min().item(Float.self)), \(pyAfterBlock1.max().item(Float.self))]")
print("Swift  after block1: [\(swiftAfterBlock1.min().item(Float.self)), \(swiftAfterBlock1.max().item(Float.self))]")
let diffBlock1 = swiftAfterBlock1 - pyAfterBlock1
eval(diffBlock1)
print("Difference:          max abs = \(diffBlock1.abs().max().item(Float.self))")

// Test 3: MLP output
print("\n" + String(repeating: "=", count: 80))
print("TEST 3: MLP OUTPUT")
print(String(repeating: "=", count: 80))

let swiftMlpOut = resnet.mlpLinear(mish(swiftTimeEmb))
eval(swiftMlpOut)
print("Python mlp out: [\(pyMlpOut.min().item(Float.self)), \(pyMlpOut.max().item(Float.self))]")
print("Swift  mlp out: [\(swiftMlpOut.min().item(Float.self)), \(swiftMlpOut.max().item(Float.self))]")
let diffMlp = swiftMlpOut - pyMlpOut
eval(diffMlp)
print("Difference:     max abs = \(diffMlp.abs().max().item(Float.self))")

// Test 4: After adding time
print("\n" + String(repeating: "=", count: 80))
print("TEST 4: AFTER ADDING TIME")
print(String(repeating: "=", count: 80))

let swiftWithTime = swiftAfterBlock1 + swiftMlpOut.expandedDimensions(axis: 2)
eval(swiftWithTime)
print("Python with time: [\(pyWithTime.min().item(Float.self)), \(pyWithTime.max().item(Float.self))]")
print("Swift  with time: [\(swiftWithTime.min().item(Float.self)), \(swiftWithTime.max().item(Float.self))]")
let diffWithTime = swiftWithTime - pyWithTime
eval(diffWithTime)
print("Difference:       max abs = \(diffWithTime.abs().max().item(Float.self))")

// Test 5: After block2
print("\n" + String(repeating: "=", count: 80))
print("TEST 5: AFTER BLOCK2")
print(String(repeating: "=", count: 80))

let swiftAfterBlock2 = resnet.block2(swiftWithTime, mask: mask)
eval(swiftAfterBlock2)
print("Python after block2: [\(pyAfterBlock2.min().item(Float.self)), \(pyAfterBlock2.max().item(Float.self))]")
print("Swift  after block2: [\(swiftAfterBlock2.min().item(Float.self)), \(swiftAfterBlock2.max().item(Float.self))]")
let diffBlock2 = swiftAfterBlock2 - pyAfterBlock2
eval(diffBlock2)
print("Difference:          max abs = \(diffBlock2.abs().max().item(Float.self))")

// Test 6: Res conv
print("\n" + String(repeating: "=", count: 80))
print("TEST 6: RES CONV")
print(String(repeating: "=", count: 80))

// Apply mask before res conv (like Python)
let maskedInput = pyHConcat * mask
var swiftResConv = maskedInput.transposed(0, 2, 1)  // [B, T, C]
swiftResConv = resnet.resConv(swiftResConv)         // [B, T, Cout]
swiftResConv = swiftResConv.transposed(0, 2, 1)     // [B, Cout, T]
eval(swiftResConv)
print("Python res conv: [\(pyResConv.min().item(Float.self)), \(pyResConv.max().item(Float.self))]")
print("Swift  res conv: [\(swiftResConv.min().item(Float.self)), \(swiftResConv.max().item(Float.self))]")
let diffResConv = swiftResConv - pyResConv
eval(diffResConv)
print("Difference:      max abs = \(diffResConv.abs().max().item(Float.self))")

// Test 7: Final output
print("\n" + String(repeating: "=", count: 80))
print("TEST 7: FINAL RESNET OUTPUT")
print(String(repeating: "=", count: 80))

let swiftFinalOut = swiftAfterBlock2 + swiftResConv
eval(swiftFinalOut)
print("Python final out: [\(pyFinalOut.min().item(Float.self)), \(pyFinalOut.max().item(Float.self))]")
print("Swift  final out: [\(swiftFinalOut.min().item(Float.self)), \(swiftFinalOut.max().item(Float.self))]")
let diffFinal = swiftFinalOut - pyFinalOut
eval(diffFinal)
print("Difference:       max abs = \(diffFinal.abs().max().item(Float.self))")
let rmse = MLX.sqrt((diffFinal * diffFinal).mean()).item(Float.self)
print("RMSE:             \(rmse)")

print("\n" + String(repeating: "=", count: 80))
if rmse < 0.01 {
    print("✅ PERFECT MATCH!")
} else if rmse < 0.1 {
    print("✅ GOOD MATCH (RMSE < 0.1)")
} else {
    print("❌ MISMATCH FOUND!")
}
print(String(repeating: "=", count: 80))
