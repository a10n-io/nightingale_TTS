"""
Check if input_embedding weights match between Python and Swift.
This is the FIRST thing to verify - if the embedding table is wrong, everything downstream is wrong.
"""
import torch
from safetensors.torch import load_file
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

print("=" * 80)
print("CHECK INPUT_EMBEDDING WEIGHTS")
print("=" * 80)

# Load Python model weights
models_dir = PROJECT_ROOT / "models" / "chatterbox"
s3gen_path = models_dir / "s3gen.safetensors"

if not s3gen_path.exists():
    print(f"\n❌ Model file not found: {s3gen_path}")
    exit(1)

flow_data = load_file(str(s3gen_path))

# Find input_embedding weight
input_emb_key = None
for key in flow_data.keys():
    if 'input_embedding' in key.lower() and 'weight' in key:
        input_emb_key = key
        break

if input_emb_key is None:
    print(f"\n❌ input_embedding.weight not found in model!")
    print(f"   Available keys with 'embedding': {[k for k in flow_data.keys() if 'embedding' in k.lower()]}")
    exit(1)

input_emb = flow_data[input_emb_key]

print(f"\n✅ Found input_embedding: {input_emb_key}")
print(f"   Shape: {input_emb.shape}")
print(f"   Range: [{input_emb.min().item():.8f}, {input_emb.max().item():.8f}]")
print(f"   Mean: {input_emb.mean().item():.8f}")
print(f"   Std: {input_emb.std().item():.8f}")

print(f"\n📊 First 10 embeddings for token 0:")
print(f"   {input_emb[0, :10].tolist()}")

print(f"\n📊 First 10 embeddings for token 500:")
if input_emb.shape[0] > 500:
    print(f"   {input_emb[500, :10].tolist()}")

print(f"\n📊 First 10 embeddings for token 1000:")
if input_emb.shape[0] > 1000:
    print(f"   {input_emb[1000, :10].tolist()}")

# Check shape (should be [vocab_size, embedding_dim])
# Python's nn.Embedding stores as [vocab_size, embedding_dim]
# MLX Embedding also expects [vocab_size, embedding_dim]
# So NO transposition should be needed for Embedding layers
print(f"\n" + "=" * 80)
print("EXPECTED FORMAT")
print("=" * 80)
print(f"PyTorch Embedding: [vocab_size, embedding_dim] = {input_emb.shape}")
print(f"MLX Embedding: [vocab_size, embedding_dim] (same)")
print(f"\n✅ No transposition needed for Embedding layers")
print(f"   BUT: Check if Swift loaded this weight correctly")

print(f"\n" + "=" * 80)
print("NEXT STEP")
print("=" * 80)
print(f"Compare this with Swift's inputEmbedding.weight")
print(f"Swift should have loaded from key: 'flow.input_embedding.weight'")
print(f"Check if Swift's weights match these values exactly")
print(f"\n" + "=" * 80)
