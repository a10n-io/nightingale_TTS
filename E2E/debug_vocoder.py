#!/usr/bin/env python3
"""Debug vocoder intermediate outputs for Swift comparison."""

import torch
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")
sys.path.insert(0, str(PROJECT_ROOT / "python"))

def main():
    print("=" * 80)
    print("DEBUGGING VOCODER INTERMEDIATE OUTPUTS")
    print("=" * 80)

    # Load model
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    model_dir = PROJECT_ROOT / "models" / "chatterbox"
    model = ChatterboxMultilingualTTS.from_local(str(model_dir), device="cpu")
    vocoder = model.s3gen.mel2wav

    # Reference directory
    ref_dir = PROJECT_ROOT / "E2E" / "reference_outputs" / "samantha" / "expressive_surprise_en"

    # Load mel
    mel = torch.from_numpy(np.load(ref_dir / "step7_final_mel.npy"))
    print(f"Mel input: {mel.shape}")

    outputs = {}

    # Hook F0 predictor
    def f0_hook(module, input, output):
        outputs["f0"] = output.detach().numpy()
        print(f"F0 predictor: shape={list(output.shape)}, range=[{output.min().item():.4f}, {output.max().item():.4f}]")
        print(f"  F0 first 20: {output[0, :20].tolist()}")

    h1 = vocoder.f0_predictor.register_forward_hook(f0_hook)

    # Hook conv_pre
    def conv_pre_hook(module, input, output):
        outputs["conv_pre"] = output.detach().numpy()
        print(f"conv_pre: shape={list(output.shape)}, range=[{output.min().item():.4f}, {output.max().item():.4f}]")
        print(f"  conv_pre [0, :5, :3]: {output[0, :5, :3].tolist()}")

    h2 = vocoder.conv_pre.register_forward_hook(conv_pre_hook)

    # Hook ups
    up_hooks = []
    for i, up in enumerate(vocoder.ups):
        def make_up_hook(idx):
            def hook(module, input, output):
                outputs[f"ups_{idx}"] = output.detach().numpy()
                print(f"ups_{idx}: shape={list(output.shape)}, range=[{output.min().item():.4f}, {output.max().item():.4f}]")
            return hook
        up_hooks.append(up.register_forward_hook(make_up_hook(i)))

    # Hook source module
    def source_hook(module, input, output):
        # output is (sine, uv, noise)
        sine, uv, noise = output
        outputs["source_sine"] = sine.detach().numpy()
        print(f"source_sine: shape={list(sine.shape)}, range=[{sine.min().item():.4f}, {sine.max().item():.4f}]")
        print(f"  source_sine first 20: {sine[0, :20, 0].tolist()}")

    h3 = vocoder.m_source.register_forward_hook(source_hook)

    # Hook conv_post
    def conv_post_hook(module, input, output):
        outputs["conv_post"] = output.detach().numpy()
        print(f"conv_post: shape={list(output.shape)}, range=[{output.min().item():.4f}, {output.max().item():.4f}]")

    h4 = vocoder.conv_post.register_forward_hook(conv_post_hook)

    # Run vocoder
    print("\nRunning vocoder...")
    with torch.no_grad():
        result = vocoder.inference(mel, cache_source=torch.zeros(1, 1, 0))

    if isinstance(result, tuple):
        audio = result[0]
    else:
        audio = result

    print(f"\nFinal audio: shape={audio.shape}, range=[{audio.min().item():.4f}, {audio.max().item():.4f}]")

    # Remove hooks
    h1.remove()
    h2.remove()
    h3.remove()
    h4.remove()
    for h in up_hooks:
        h.remove()

    # Save intermediates
    print("\nSaving intermediate outputs...")
    for key, value in outputs.items():
        save_path = ref_dir / f"vocoder_debug_{key}.npy"
        np.save(save_path, value)
        print(f"  Saved vocoder_debug_{key}.npy: {value.shape}")

if __name__ == "__main__":
    main()
