"""
Compare input embeddings (token embeddings BEFORE encoder).
"""
import torch
from safetensors.torch import load_file
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models" / "chatterbox"
VOICE_DIR = PROJECT_ROOT / "baked_voices" / "samantha"

print("=" * 80)
print("COMPARE INPUT EMBEDDINGS")
print("=" * 80)

# Load model weights
flow_weights = load_file(str(MODELS_DIR / "s3gen.safetensors"))
input_embedding_weight = flow_weights["flow.input_embedding.weight"]

print(f"\nInput embedding weight:")
print(f"  Shape: {input_embedding_weight.shape}")
print(f"  Mean: {input_embedding_weight.mean().item():.6f}")
print(f"  Std: {input_embedding_weight.std().item():.6f}")
print(f"  First 5 values [0,:5]: {input_embedding_weight[0,:5].tolist()}")

# Load tokens
voice_data = load_file(str(VOICE_DIR / "baked_voice.safetensors"))
tokens_data = load_file(str(PROJECT_ROOT / "test_audio/cross_validate/python_speech_tokens.safetensors"))

prompt_token = voice_data["gen.prompt_token"]  # [1, T_prompt]
speech_tokens = tokens_data["speech_tokens"]    # [T_speech]

print(f"\nPrompt token shape: {prompt_token.shape}")
print(f"Speech tokens shape: {speech_tokens.shape}")

# Concatenate tokens (same as Swift does)
full_tokens = torch.cat([prompt_token.squeeze(0), speech_tokens], dim=0)
print(f"Full tokens shape: {full_tokens.shape}")
print(f"First 10 tokens: {full_tokens[:10].tolist()}")

# Look up embeddings
token_embs = input_embedding_weight[full_tokens.long()]
print(f"\nToken embeddings shape: {token_embs.shape}")
print(f"  Mean: {token_embs.mean().item():.6f}")
print(f"  Std: {token_embs.std().item():.6f}")
print(f"  Range: [{token_embs.min().item():.6f}, {token_embs.max().item():.6f}]")

# Save for comparison
from safetensors.torch import save_file
output_file = PROJECT_ROOT / "test_audio" / "forensic" / "python_input_embeddings.safetensors"
save_file({"input_embeddings": token_embs.contiguous()}, str(output_file))

print(f"\n✅ Saved to: {output_file}")
print("=" * 80)
