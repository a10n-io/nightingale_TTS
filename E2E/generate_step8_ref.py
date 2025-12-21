#!/usr/bin/env python3
"""Generate Step 8 (Vocoder) reference outputs for Swift verification."""

import torch
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")
sys.path.insert(0, str(PROJECT_ROOT / "python"))

def main():
    print("=" * 80)
    print("GENERATING STEP 8 (VOCODER) REFERENCE OUTPUTS")
    print("=" * 80)

    # Load Python model
    print("\nLoading Python model...")
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    model_dir = PROJECT_ROOT / "models" / "chatterbox"
    model = ChatterboxMultilingualTTS.from_local(str(model_dir), device="cpu")
    vocoder = model.s3gen.mel2wav

    # Reference directory
    ref_dir = PROJECT_ROOT / "E2E" / "reference_outputs" / "samantha" / "expressive_surprise_en"

    # Load mel spectrogram from Step 7
    print("\nLoading Step 7 mel output...")
    mel_path = ref_dir / "step7_final_mel.npy"
    if not mel_path.exists():
        # Try alternative path
        mel_path = ref_dir / "step7_mel.npy"

    if not mel_path.exists():
        print(f"ERROR: Could not find mel file at {mel_path}")
        print("Please run generate_step7_ref.py first")
        return

    mel = torch.from_numpy(np.load(mel_path))  # [1, 80, T]
    print(f"  mel shape: {mel.shape}")
    print(f"  mel range: [{mel.min().item():.4f}, {mel.max().item():.4f}]")

    # Create output dict for reference values
    outputs = {}
    outputs["mel_input"] = mel.numpy()

    print("\n" + "=" * 80)
    print("RUNNING VOCODER WITH HOOKS")
    print("=" * 80)

    # Register hooks to capture intermediate values
    hooks = []

    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            if out is not None and isinstance(out, torch.Tensor):
                outputs[name] = out.detach().numpy()
                print(f"  {name}: shape={list(out.shape)}, range=[{out.min().item():.4f}, {out.max().item():.4f}]")
        return hook

    # Hook F0 predictor
    hooks.append(vocoder.f0_predictor.register_forward_hook(make_hook("f0_predictor")))

    # Hook conv_pre
    hooks.append(vocoder.conv_pre.register_forward_hook(make_hook("conv_pre")))

    # Hook upsampling layers
    for i, up in enumerate(vocoder.ups):
        hooks.append(up.register_forward_hook(make_hook(f"ups_{i}")))

    # Hook conv_post
    hooks.append(vocoder.conv_post.register_forward_hook(make_hook("conv_post")))

    # Run vocoder
    print("\nRunning vocoder inference...")
    with torch.no_grad():
        # The vocoder expects mel in [B, 80, T] format
        result = vocoder.inference(mel, cache_source=torch.zeros(1, 1, 0))

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Handle tuple return (audio, cache_source)
    if isinstance(result, tuple):
        audio = result[0]
        print(f"  (vocoder returned tuple, using first element)")
    else:
        audio = result

    print(f"\nFinal audio output:")
    print(f"  shape: {audio.shape}")
    print(f"  range: [{audio.min().item():.4f}, {audio.max().item():.4f}]")
    print(f"  duration: {audio.shape[1] / 24000:.3f}s at 24kHz")

    outputs["audio"] = audio.numpy()

    # Save outputs
    print(f"\nSaving Step 8 reference outputs to {ref_dir}...")
    for key, value in outputs.items():
        save_path = ref_dir / f"step8_{key}.npy"
        np.save(save_path, value)
        print(f"  Saved step8_{key}.npy: shape={value.shape}")

    print("\n" + "=" * 80)
    print("STEP 8 REFERENCE OUTPUTS GENERATED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    main()
