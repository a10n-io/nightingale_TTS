"""
Test FixedLinear in isolation to verify the math is correct.
"""
import torch
import numpy as np
from safetensors.torch import load_file

# Load the actual transformer weights
state_dict = load_file("/Users/a10n/Projects/nightingale_TTS/models/chatterbox/s3gen.safetensors")

# Get the first transformer's query projection weight
# Python key: flow.decoder.estimator.down_blocks.0.1.0.attn1.to_q.weight
# After remapping: flow.decoder.estimator.downBlocks.0.transformers.0.attention.queryProj.weight
key = "flow.decoder.estimator.down_blocks.0.1.0.attn1.to_q.weight"

if key in state_dict:
    weight_pytorch = state_dict[key]  # [out, in] = [256, 256]
    print(f"PyTorch weight shape: {weight_pytorch.shape}")
    print(f"PyTorch weight[0,:5]: {weight_pytorch[0, :5]}")
    print(f"PyTorch weight[:5,0]: {weight_pytorch[:5, 0]}")

    # Transpose for MLX format [in, out]
    weight_mlx = weight_pytorch.T  # [256, 256]
    print(f"\nMLX weight shape: {weight_mlx.shape}")
    print(f"MLX weight[0,:5]: {weight_mlx[0, :5]}")
    print(f"MLX weight[:5,0]: {weight_mlx[:5, 0]}")

    # Test with deterministic input
    batch = 2
    seq = 696
    dim = 256

    # Create test input: sin(index/10)
    count = batch * seq * dim
    flat = np.arange(count, dtype=np.float32)
    data = np.sin(flat / 10.0)
    x = torch.from_numpy(data).reshape(batch, seq, dim)

    print(f"\nInput x shape: {x.shape}")
    print(f"Input x[0,0,:5]: {x[0,0,:5]}")

    # PyTorch: x @ weight.T (because weight is [out, in])
    # x: [B, T, in] @ weight.T: [in, out] = [B, T, out]
    out_pytorch = torch.matmul(x, weight_pytorch.T)

    print(f"\nPyTorch output shape: {out_pytorch.shape}")
    print(f"PyTorch output[0,0,:5]: {out_pytorch[0,0,:5]}")
    print(f"PyTorch output mean: {out_pytorch.mean():.6f}")
    print(f"PyTorch output std: {out_pytorch.std():.6f}")

    # MLX: x @ weight (because weight is already [in, out])
    # x: [B, T, in] @ weight: [in, out] = [B, T, out]
    out_mlx = torch.matmul(x, weight_mlx)

    print(f"\nMLX-style output shape: {out_mlx.shape}")
    print(f"MLX-style output[0,0,:5]: {out_mlx[0,0,:5]}")
    print(f"MLX-style output mean: {out_mlx.mean():.6f}")
    print(f"MLX-style output std: {out_mlx.std():.6f}")

    # Check if they match
    print(f"\n✅ Outputs match: {torch.allclose(out_pytorch, out_mlx, atol=1e-5)}")

else:
    print(f"❌ Key '{key}' not found")
    print("Available transformer keys:")
    for k in sorted(state_dict.keys()):
        if 'down_blocks.0.1' in k and 'attn' in k:
            print(f"  {k}")
