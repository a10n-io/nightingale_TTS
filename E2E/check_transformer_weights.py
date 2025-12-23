"""
Check what transformer weight keys exist and what they should map to in Swift.
"""
from safetensors.torch import load_file

state_dict = load_file("/Users/a10n/Projects/nightingale_TTS/models/chatterbox/s3gen.safetensors")

print("=" * 80)
print("TRANSFORMER WEIGHTS IN PYTHON")
print("=" * 80)

# Find all transformer keys for the first down block
print("\nFirst down block, first transformer (down_blocks.0.1.0):")
for key in sorted(state_dict.keys()):
    if 'down_blocks.0.1.0' in key:
        weight = state_dict[key]
        print(f"  {key}")
        print(f"    Shape: {weight.shape}")
        if 'weight' in key:
            print(f"    [0,:5]: {weight[0, :5] if len(weight.shape) == 2 else 'N/A'}")

print("\n" + "=" * 80)
print("EXPECTED SWIFT KEY MAPPING")
print("=" * 80)

print("\nPython -> Swift mapping:")
mappings = {
    # Attention
    "down_blocks.0.1.0.attn1.to_q.weight": "downBlocks.0.transformers.0.attention.queryProj.weight",
    "down_blocks.0.1.0.attn1.to_k.weight": "downBlocks.0.transformers.0.attention.keyProj.weight",
    "down_blocks.0.1.0.attn1.to_v.weight": "downBlocks.0.transformers.0.attention.valueProj.weight",
    "down_blocks.0.1.0.attn1.to_out.weight": "downBlocks.0.transformers.0.attention.outProj.weight",

    # Norms
    "down_blocks.0.1.0.norm1.weight": "downBlocks.0.transformers.0.norm1.weight",
    "down_blocks.0.1.0.norm1.bias": "downBlocks.0.transformers.0.norm1.bias",
    "down_blocks.0.1.0.norm3.weight": "downBlocks.0.transformers.0.norm2.weight",
    "down_blocks.0.1.0.norm3.bias": "downBlocks.0.transformers.0.norm2.bias",

    # FF (FlowMLP)
    "down_blocks.0.1.0.ff.net.0.weight": "downBlocks.0.transformers.0.ff.layers.0.weight",
    "down_blocks.0.1.0.ff.net.0.bias": "downBlocks.0.transformers.0.ff.layers.0.bias",
    "down_blocks.0.1.0.ff.net.2.weight": "downBlocks.0.transformers.0.ff.layers.1.weight",
    "down_blocks.0.1.0.ff.net.2.bias": "downBlocks.0.transformers.0.ff.layers.1.bias",
}

for py_key, swift_key in mappings.items():
    full_py_key = f"flow.decoder.estimator.{py_key}"
    if full_py_key in state_dict:
        print(f"✅ {py_key}")
        print(f"   -> {swift_key}")
    else:
        print(f"❌ {py_key} NOT FOUND")
