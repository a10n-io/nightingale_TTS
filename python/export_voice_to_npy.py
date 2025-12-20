#!/usr/bin/env python3
"""
Export baked voice (.pt) to NPY format WITHOUT padding.

This creates NPY files that exactly match the values in baked_voice.pt,
which is what Python actually uses during inference.

For true E2E verification, both Python and Swift must use identical source values.

Usage:
    python export_voice_to_npy.py --voice samantha
    python export_voice_to_npy.py --voice sujano
"""

import torch
import numpy as np
from pathlib import Path
import argparse


def export_voice_to_npy(voice_name: str):
    """Export baked voice to NPY without padding - exact match to baked_voice.pt."""
    voice_dir = Path(f"../baked_voices/{voice_name}")
    pt_path = voice_dir / "baked_voice.pt"
    output_dir = voice_dir / "npy_original"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading baked voice from: {pt_path}")
    voice = torch.load(pt_path, map_location='cpu', weights_only=True)

    print(f"Output directory: {output_dir}")
    print()

    # =========================================================================
    # T3 Components (model.conds.t3)
    # =========================================================================
    t3 = voice['t3']

    # Speaker embedding [1, 256]
    speaker_emb = t3['speaker_emb'].detach().numpy().astype(np.float32)
    np.save(output_dir / 'soul_t3_256.npy', speaker_emb)
    print(f"  soul_t3_256.npy: {speaker_emb.shape} ({speaker_emb.dtype})")

    # T3 conditioning speech tokens [1, 150]
    cond_tokens = t3['cond_prompt_speech_tokens'].detach().numpy().astype(np.int32)
    np.save(output_dir / 't3_cond_tokens.npy', cond_tokens)
    print(f"  t3_cond_tokens.npy: {cond_tokens.shape} ({cond_tokens.dtype})")

    # Emotion advancement value [1, 1, 1]
    emotion_adv = t3.get('emotion_adv', torch.tensor([[[0.5]]]))
    if isinstance(emotion_adv, (int, float)):
        emotion_adv = torch.tensor([[[emotion_adv]]])
    emotion_adv = emotion_adv.detach().numpy().astype(np.float32)
    np.save(output_dir / 'emotion_adv.npy', emotion_adv)
    print(f"  emotion_adv.npy: {emotion_adv.shape} ({emotion_adv.dtype}) value={emotion_adv.flatten()[0]}")

    # =========================================================================
    # S3Gen Components (model.conds.gen)
    # =========================================================================
    gen = voice['gen']

    # Speaker embedding for S3Gen [1, 192]
    s3_emb = gen['embedding'].detach().numpy().astype(np.float32)
    np.save(output_dir / 'soul_s3_192.npy', s3_emb)
    print(f"  soul_s3_192.npy: {s3_emb.shape} ({s3_emb.dtype})")

    # Prompt speech tokens [1, N] - NO PADDING
    prompt_token = gen['prompt_token'].detach().numpy().astype(np.int32)
    np.save(output_dir / 'prompt_token.npy', prompt_token)
    print(f"  prompt_token.npy: {prompt_token.shape} ({prompt_token.dtype})")

    # Prompt token length [1]
    prompt_token_len = gen['prompt_token_len'].detach().numpy().astype(np.int32)
    np.save(output_dir / 'prompt_token_len.npy', prompt_token_len)
    print(f"  prompt_token_len.npy: {prompt_token_len.shape} ({prompt_token_len.dtype}) value={prompt_token_len[0]}")

    # Prompt mel features [1, N, 80] - NO PADDING
    prompt_feat = gen['prompt_feat'].detach().numpy().astype(np.float32)
    np.save(output_dir / 'prompt_feat.npy', prompt_feat)
    print(f"  prompt_feat.npy: {prompt_feat.shape} ({prompt_feat.dtype})")

    # Prompt feat length (may be None)
    prompt_feat_len = gen.get('prompt_feat_len')
    if prompt_feat_len is not None:
        prompt_feat_len = prompt_feat_len.detach().numpy().astype(np.int32)
        np.save(output_dir / 'prompt_feat_len.npy', prompt_feat_len)
        print(f"  prompt_feat_len.npy: {prompt_feat_len.shape}")
    else:
        # Infer from prompt_feat shape
        prompt_feat_len = np.array([prompt_feat.shape[1]], dtype=np.int32)
        np.save(output_dir / 'prompt_feat_len.npy', prompt_feat_len)
        print(f"  prompt_feat_len.npy: {prompt_feat_len.shape} (inferred) value={prompt_feat_len[0]}")

    print()
    print(f"✓ Export complete! Original (non-padded) voice saved to: {output_dir}")
    print()
    print("These files exactly match the values in baked_voice.pt")
    print("Use these for true E2E verification between Python and Swift")

    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export baked voice to NPY (no padding)")
    parser.add_argument("--voice", "-v", required=True,
                        help="Voice name (e.g., 'samantha', 'sujano')")
    args = parser.parse_args()

    export_voice_to_npy(args.voice)
