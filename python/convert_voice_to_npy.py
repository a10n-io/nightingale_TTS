#!/usr/bin/env python3
"""
Convert a baked voice (.pt) to NPY format for Swift/MLX.

The Swift ChatterboxEngine expects a directory with these NPY files:
- soul_t3_256.npy    -> T3 speaker embedding [1, 256]
- soul_s3_192.npy    -> S3Gen speaker embedding [1, 192]
- t3_cond_tokens.npy -> T3 conditioning tokens [1, N]
- prompt_token.npy   -> S3Gen prompt tokens [1, N]
- prompt_feat.npy    -> S3Gen prompt features [1, N, 80]
"""

import torch
import numpy as np
from pathlib import Path
import argparse


def convert_voice_to_npy(pt_path: str, output_dir: str):
    """Convert a baked voice .pt file to NPY directory format."""
    pt_path = Path(pt_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading baked voice from: {pt_path}")
    voice = torch.load(pt_path, map_location='cpu')

    print(f"Output directory: {output_dir}")
    print()

    # T3 components
    t3 = voice['t3']

    # Speaker embedding for T3
    speaker_emb = t3['speaker_emb'].detach().numpy()
    np.save(output_dir / 'soul_t3_256.npy', speaker_emb)
    print(f"  soul_t3_256.npy: {speaker_emb.shape} ({speaker_emb.dtype})")

    # Conditioning tokens for T3
    cond_tokens = t3['cond_prompt_speech_tokens'].detach().numpy().astype(np.int32)
    np.save(output_dir / 't3_cond_tokens.npy', cond_tokens)
    print(f"  t3_cond_tokens.npy: {cond_tokens.shape} ({cond_tokens.dtype})")

    # GEN (S3Gen) components
    gen = voice['gen']

    # Speaker embedding for S3Gen
    s3_emb = gen['embedding'].detach().numpy()
    np.save(output_dir / 'soul_s3_192.npy', s3_emb)
    print(f"  soul_s3_192.npy: {s3_emb.shape} ({s3_emb.dtype})")

    # Prompt tokens for S3Gen
    prompt_token = gen['prompt_token'].detach().numpy().astype(np.int32)
    np.save(output_dir / 'prompt_token.npy', prompt_token)
    print(f"  prompt_token.npy: {prompt_token.shape} ({prompt_token.dtype})")

    # Prompt features for S3Gen
    prompt_feat = gen['prompt_feat'].detach().numpy()
    np.save(output_dir / 'prompt_feat.npy', prompt_feat)
    print(f"  prompt_feat.npy: {prompt_feat.shape} ({prompt_feat.dtype})")

    print()
    print(f"Conversion complete! Voice saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert baked voice to NPY format for Swift")
    parser.add_argument("--input", "-i", default="../baked_voices/baked_voice.pt",
                        help="Path to baked voice .pt file")
    parser.add_argument("--output", "-o", default="../baked_voices/baked_voice_npy",
                        help="Output directory for NPY files")
    args = parser.parse_args()

    convert_voice_to_npy(args.input, args.output)
