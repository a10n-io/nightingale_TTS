import torch
import torch.nn as nn
from safetensors.torch import load_file

# Load weights
state_dict = load_file("/Users/a10n/Projects/nightingale_TTS/models/chatterbox/s3gen.safetensors")

# Get resConv weight for first down block
key = "flow.decoder.estimator.down_blocks.0.0.res_conv.weight"
if key in state_dict:
    resConvW = state_dict[key]
    print(f"🔍 Python ResNet[0] resConv Weight Shape: {resConvW.shape}")
    print(f"🔍 Python resConv weight (PyTorch format): {resConvW[0, 0, :5]}")

    # MLX format is transposed: [Out, In, K] -> [In, Out, K]
    # Wait, Conv1d is different. Let me check the actual shape first.
    print(f"\n🔍 Full shape: {resConvW.shape}")

    # For Conv1d:
    # PyTorch: [out_channels, in_channels, kernel_size] = [256, 320, 1]
    # MLX Conv1d weight: [out_channels, kernel_size, in_channels] = [256, 1, 320]

    # So we need to transpose from [Out, In, K] to [Out, K, In]
    mlx_format = resConvW.permute(0, 2, 1)
    print(f"\n🔍 MLX format shape: {mlx_format.shape}")
    print(f"🔍 MLX format weight[0,0,:5]: {mlx_format[0, 0, :5]}")

else:
    print(f"❌ Key '{key}' not found in state_dict")
    print(f"Available keys: {[k for k in state_dict.keys() if 'res_conv' in k]}")
