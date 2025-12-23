"""
Check vocoder weights to verify they're loading correctly in Swift.
"""
from safetensors.torch import load_file

state_dict = load_file("/Users/a10n/Projects/nightingale_TTS/models/chatterbox/s3gen.safetensors")

print("=" * 80)
print("VOCODER WEIGHTS CHECK")
print("=" * 80)

# Check conv_pre weight (should have weight_norm parametrization)
conv_pre_keys = [k for k in state_dict.keys() if 'mel2wav.conv_pre' in k]
print("\nconv_pre keys:")
for k in sorted(conv_pre_keys):
    print(f"  {k}: {state_dict[k].shape}")

# Check a specific ups layer
ups_keys = [k for k in state_dict.keys() if 'mel2wav.ups.0' in k and 'weight' in k]
print("\nups.0 keys:")
for k in sorted(ups_keys):
    print(f"  {k}: {state_dict[k].shape}")

# Check mSource (SourceModuleHnNSF) linear weights
msource_keys = [k for k in state_dict.keys() if 'm_source.l_linear' in k]
print("\nmSource.linear keys:")
for k in sorted(msource_keys):
    w = state_dict[k]
    print(f"  {k}: {w.shape}")
    if 'weight' in k:
        print(f"    Values: {w.flatten()[:10].tolist()}")
        print(f"    Sum: {w.sum().item():.6f}")

# Check if there are any other m_source keys
other_msource = [k for k in state_dict.keys() if 'm_source' in k and 'l_linear' not in k]
print("\nOther m_source keys:")
for k in sorted(other_msource):
    print(f"  {k}: {state_dict[k].shape}")
