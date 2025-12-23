"""
Test ONLY the first linear layer (embed.linear) to isolate where divergence starts.
This bypasses all encoder complexity and just tests: tokens -> embedding lookup -> embed.linear
"""
import torch
from safetensors.torch import load_file
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

print("=" * 80)
print("TEST EMBED.LINEAR ONLY")
print("=" * 80)

# Load model weights
models_dir = PROJECT_ROOT / "models" / "chatterbox"
s3gen_path = models_dir / "s3gen.safetensors"
flow_data = load_file(str(s3gen_path))

# Load tokens
tokens_path = PROJECT_ROOT / "test_audio/cross_validate/python_speech_tokens.safetensors"
tokens_data = load_file(str(tokens_path))
tokens = tokens_data["speech_tokens"]  # [T]

# Load input_embedding
input_emb_weight = flow_data["flow.input_embedding.weight"]  # [vocab_size, 512]

# Load embed.linear weights (this is encoder.embed.out.0 in Python)
embed_linear_weight = None
embed_linear_bias = None
for key in flow_data.keys():
    if "encoder.embed.out.0.weight" in key or "encoder.embed.linear.weight" in key:
        embed_linear_weight = flow_data[key]
        print(f"✅ Found embed.linear.weight: {key}")
    if "encoder.embed.out.0.bias" in key or "encoder.embed.linear.bias" in key:
        embed_linear_bias = flow_data[key]
        print(f"✅ Found embed.linear.bias: {key}")

if embed_linear_weight is None:
    # Try the flow.encoder.embed pattern
    embed_keys = [k for k in flow_data.keys() if 'encoder.embed' in k and 'weight' in k]
    print(f"\n Available encoder.embed keys: {embed_keys[:10]}")
    exit(1)

print(f"\n📊 Weights loaded:")
print(f"   input_embedding: {input_emb_weight.shape}")
print(f"   embed.linear.weight: {embed_linear_weight.shape}")
if embed_linear_bias is not None:
    print(f"   embed.linear.bias: {embed_linear_bias.shape}")

# ==============================================================================
# Forward pass: tokens -> embedding lookup -> embed.linear
# ==============================================================================

print(f"\n" + "=" * 80)
print("FORWARD PASS")
print("=" * 80)

# Step 1: Embedding lookup
print(f"\n1. Embedding lookup")
print(f"   tokens shape: {tokens.shape} = {tokens.shape}")
print(f"   First 10 tokens: {tokens[:10].tolist()}")

# Gather embeddings
token_embs = input_emb_weight[tokens]  # [T, 512]
print(f"   token_embs shape: {token_embs.shape}")
print(f"   token_embs stats: mean={token_embs.mean().item():.6f}, std={token_embs.std().item():.6f}")

# Step 2: embed.linear (Conv1d with kernel_size=1, or equivalently a Linear layer)
print(f"\n2. embed.linear")
print(f"   Weight shape: {embed_linear_weight.shape}")

# Check if it's Conv1d or Linear format
if embed_linear_weight.ndim == 3:
    # Conv1d format: [out_channels, in_channels, kernel_size]
    print(f"   Format: Conv1d [out={embed_linear_weight.shape[0]}, in={embed_linear_weight.shape[1]}, kernel={embed_linear_weight.shape[2]}]")
    if embed_linear_weight.shape[2] == 1:
        # kernel_size=1, equivalent to Linear
        print(f"   → Equivalent to Linear({embed_linear_weight.shape[1]}, {embed_linear_weight.shape[0]})")
        # For Conv1d, input is [B, C, T], output is [B, C_out, T]
        # We need to transpose token_embs from [T, C] to [1, C, T]
        x_in = token_embs.T.unsqueeze(0)  # [1, 512, T]
        # Conv1d weight is [out, in, 1], squeeze to [out, in] for matrix mult
        weight_2d = embed_linear_weight.squeeze(2)  # [out, in]
        # Do: output = input @ weight.T for Conv1d with kernel=1
        # Actually, Conv1d does: out[b,i,t] = sum_j (w[i,j,0] * in[b,j,t])
        # Which is: out = weight @ in for each position
        out = torch.nn.functional.conv1d(x_in, embed_linear_weight, bias=embed_linear_bias)
        print(f"   output shape: {out.shape} (Conv1d format [1, out, T])")
        # Transpose back to [T, out]
        output = out.squeeze(0).T  # [T, out]
    else:
        print(f"   ⚠️  Unexpected kernel_size={embed_linear_weight.shape[2]}")
        exit(1)
elif embed_linear_weight.ndim == 2:
    # Linear format: [out_features, in_features] (PyTorch format)
    print(f"   Format: Linear [out={embed_linear_weight.shape[0]}, in={embed_linear_weight.shape[1]}]")
    # PyTorch Linear: y = x @ W.T + b
    output = token_embs @ embed_linear_weight.T
    if embed_linear_bias is not None:
        output = output + embed_linear_bias
else:
    print(f"   ❌ Unexpected weight dimensionality: {embed_linear_weight.ndim}")
    exit(1)

print(f"   output shape: {output.shape}")
print(f"   output stats: mean={output.mean().item():.6f}, std={output.std().item():.6f}, range=[{output.min().item():.6f}, {output.max().item():.6f}]")

# Save for comparison
from safetensors.torch import save_file
save_file({
    "token_embs": token_embs,
    "embed_linear_output": output,
}, str(PROJECT_ROOT / "test_audio/forensic/python_embed_linear_only.safetensors"))

print(f"\n✅ Saved to: test_audio/forensic/python_embed_linear_only.safetensors")

print(f"\n" + "=" * 80)
print("NEXT: Compare with Swift's embed.linear output")
print("=" * 80)
