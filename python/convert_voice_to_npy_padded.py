#!/usr/bin/env python3
"""
Convert a baked voice (.pt) to NPY format for Swift/MLX with padding.

Pads the voice to match samantha_full's dimensions which the Swift code was tested with:
- prompt_token: [1, 449]  (we pad from 250)
- prompt_feat: [1, 898, 80]  (we pad from 500)
"""

import torch
import numpy as np
from pathlib import Path
import argparse


# Target dimensions from samantha_full voice
TARGET_PROMPT_TOKEN_LEN = 449
TARGET_PROMPT_FEAT_LEN = 898


def convert_voice_to_npy_padded(pt_path: str, output_dir: str):
    """Convert a baked voice .pt file to NPY directory format with padding."""
    pt_path = Path(pt_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading baked voice from: {pt_path}")
    voice = torch.load(pt_path, map_location='cpu')

    print(f"Output directory: {output_dir}")
    print()

    # T3 components - no padding needed
    t3 = voice['t3']

    speaker_emb = t3['speaker_emb'].detach().numpy()
    np.save(output_dir / 'soul_t3_256.npy', speaker_emb)
    print(f"  soul_t3_256.npy: {speaker_emb.shape} ({speaker_emb.dtype})")

    cond_tokens = t3['cond_prompt_speech_tokens'].detach().numpy().astype(np.int32)
    np.save(output_dir / 't3_cond_tokens.npy', cond_tokens)
    print(f"  t3_cond_tokens.npy: {cond_tokens.shape} ({cond_tokens.dtype})")

    # GEN (S3Gen) components - need padding
    gen = voice['gen']

    s3_emb = gen['embedding'].detach().numpy()
    np.save(output_dir / 'soul_s3_192.npy', s3_emb)
    print(f"  soul_s3_192.npy: {s3_emb.shape} ({s3_emb.dtype})")

    # Pad prompt tokens to target length
    prompt_token = gen['prompt_token'].detach().numpy().astype(np.int32)
    current_len = prompt_token.shape[1]
    if current_len < TARGET_PROMPT_TOKEN_LEN:
        # Repeat the tokens to fill (better than zeros for audio)
        pad_len = TARGET_PROMPT_TOKEN_LEN - current_len
        # Use the last token repeated for padding
        pad_value = prompt_token[0, -1]
        padding = np.full((1, pad_len), pad_value, dtype=np.int32)
        prompt_token = np.concatenate([prompt_token, padding], axis=1)
        print(f"  prompt_token.npy: {prompt_token.shape} ({prompt_token.dtype}) [PADDED from {current_len}]")
    else:
        print(f"  prompt_token.npy: {prompt_token.shape} ({prompt_token.dtype})")
    np.save(output_dir / 'prompt_token.npy', prompt_token)

    # Pad prompt features to target length
    prompt_feat = gen['prompt_feat'].detach().numpy()
    current_feat_len = prompt_feat.shape[1]
    if current_feat_len < TARGET_PROMPT_FEAT_LEN:
        pad_len = TARGET_PROMPT_FEAT_LEN - current_feat_len
        # For mel features, repeat the last frame (better than zeros for audio quality)
        last_frame = prompt_feat[:, -1:, :]  # [1, 1, 80]
        padding = np.repeat(last_frame, pad_len, axis=1)  # [1, pad_len, 80]
        prompt_feat = np.concatenate([prompt_feat, padding], axis=1)
        print(f"  prompt_feat.npy: {prompt_feat.shape} ({prompt_feat.dtype}) [PADDED from {current_feat_len}]")
    else:
        print(f"  prompt_feat.npy: {prompt_feat.shape} ({prompt_feat.dtype})")
    np.save(output_dir / 'prompt_feat.npy', prompt_feat)

    print()
    print(f"Conversion complete! Padded voice saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert baked voice to NPY format with padding for Swift")
    parser.add_argument("--voice", "-v", default=None,
                        help="Voice name (e.g., 'samantha'). Will look for baked_voices/{name}/baked_voice.pt")
    parser.add_argument("--input", "-i", default=None,
                        help="Path to baked voice .pt file (alternative to --voice)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output directory for NPY files (defaults to baked_voices/{voice}_npy)")
    args = parser.parse_args()

    # Determine input/output paths
    if args.voice:
        input_path = f"../baked_voices/{args.voice}/baked_voice.pt"
        output_dir = args.output or f"../baked_voices/{args.voice}/npy"
    elif args.input:
        input_path = args.input
        output_dir = args.output or args.input.replace('.pt', '_npy')
    else:
        print("Error: Please provide either --voice or --input")
        exit(1)

    convert_voice_to_npy_padded(input_path, output_dir)
