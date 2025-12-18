#!/Users/a10n/Projects/nightingale_TTS/python/venv/bin/python
"""
E2E Test Runner for Python/PyTorch Chatterbox TTS.

This script generates reference outputs for all test cases that Swift must match.
Uses deterministic settings (temperature=0, fixed seeds) for reproducibility.

Usage:
    python run_python_e2e.py
    python run_python_e2e.py --voice samantha --sentence basic_greeting
"""

import torch
import numpy as np
import random
import json
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# Project paths
PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")
MODEL_DIR = PROJECT_ROOT / "models" / "chatterbox"
VOICE_DIR = PROJECT_ROOT / "baked_voices"
E2E_DIR = PROJECT_ROOT / "E2E"
OUTPUT_DIR = E2E_DIR / "reference_outputs"

# Deterministic settings
SEED = 42
TEMPERATURE = 0.001  # Very low temperature for near-deterministic argmax behavior


@dataclass
class TestCase:
    voice: str
    sentence_id: str
    text: str
    language: str


def set_deterministic_seeds(seed: int = SEED):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, 'mps') and hasattr(torch.mps, 'manual_seed'):
        torch.mps.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Additional determinism settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"[SEED] Set all random seeds to {seed}")


def load_test_sentences() -> list:
    """Load test sentences from JSON."""
    with open(E2E_DIR / "test_sentences.json", "r") as f:
        return json.load(f)


def get_test_cases(voice_filter: Optional[str] = None,
                   sentence_filter: Optional[str] = None) -> list[TestCase]:
    """Generate all test cases (voice × sentence × language)."""
    voices = ["samantha", "sujano"]
    sentences = load_test_sentences()
    languages = ["en", "nl"]

    test_cases = []
    for voice in voices:
        if voice_filter and voice != voice_filter:
            continue
        for sentence in sentences:
            if sentence_filter and sentence["id"] != sentence_filter:
                continue
            for lang in languages:
                text_key = f"text_{lang}"
                if text_key in sentence:
                    test_cases.append(TestCase(
                        voice=voice,
                        sentence_id=sentence["id"],
                        text=sentence[text_key],
                        language=lang
                    ))
    return test_cases


def run_t3_conditioning(model, device: str) -> dict:
    """Run T3 conditioning and return intermediate outputs."""
    t3 = model.t3
    conds = model.conds.t3

    # Speaker embedding projection
    speaker_emb = conds.speaker_emb.to(device)
    speaker_token = t3.cond_enc.spkr_enc(speaker_emb).unsqueeze(1)

    # Perceiver resampler
    cond_speech_tokens = conds.cond_prompt_speech_tokens.to(device)
    speech_emb = t3.speech_emb(cond_speech_tokens)
    positions = torch.arange(cond_speech_tokens.shape[1], device=device).unsqueeze(0)
    speech_pos_emb = t3.speech_pos_emb(positions)
    cond_speech_emb = speech_emb + speech_pos_emb
    perceiver_out = t3.cond_enc.perceiver(cond_speech_emb)

    # Emotion adversarial FC (from baked voice, not hardcoded!)
    emotion_value = conds.emotion_adv.to(device)
    emotion_token = t3.cond_enc.emotion_adv_fc(emotion_value)

    # Final conditioning
    final_cond = torch.cat([speaker_token, perceiver_out, emotion_token], dim=1)

    return {
        "speaker_token": speaker_token.detach().cpu().numpy(),
        "perceiver_out": perceiver_out.detach().cpu().numpy(),
        "emotion_token": emotion_token.detach().cpu().numpy(),
        "emotion_value": emotion_value.detach().cpu().numpy(),
        "final_cond": final_cond.detach().cpu().numpy(),
    }


def run_test_case(model, test_case: TestCase, device: str) -> dict:
    """Run a single test case and return all intermediate outputs."""
    from chatterbox.mtl_tts import Conditionals, punc_norm
    from chatterbox.models.s3gen.s3gen import drop_invalid_tokens
    import torch.nn.functional as F

    # Load voice
    voice_path = VOICE_DIR / test_case.voice / "baked_voice.pt"
    model.conds = Conditionals.load(str(voice_path), map_location=device)

    outputs = {}

    # Step 1: Tokenization (using model's tokenizer with language ID)
    text = punc_norm(test_case.text)
    text_tokens = model.tokenizer.text_to_tokens(text, language_id=test_case.language.lower())
    text_tokens = text_tokens.to(device)

    # Save raw tokens
    outputs["step1_text_tokens"] = text_tokens[0].detach().cpu().numpy().astype(np.int32)

    # Prepare for T3 (duplicate for CFG, add SOT/EOT)
    text_tokens_cfg = torch.cat([text_tokens, text_tokens], dim=0)
    sot = model.t3.hp.start_text_token
    eot = model.t3.hp.stop_text_token
    text_tokens_cfg = F.pad(text_tokens_cfg, (1, 0), value=sot)
    text_tokens_cfg = F.pad(text_tokens_cfg, (0, 1), value=eot)
    outputs["step1_text_tokens_padded"] = text_tokens_cfg[0].detach().cpu().numpy().astype(np.int32)

    # Step 2: T3 Conditioning
    cond_outputs = run_t3_conditioning(model, device)
    outputs["step2_speaker_token"] = cond_outputs["speaker_token"]
    outputs["step2_perceiver_out"] = cond_outputs["perceiver_out"]
    outputs["step2_emotion_token"] = cond_outputs["emotion_token"]
    outputs["step2_emotion_value"] = cond_outputs["emotion_value"]
    outputs["step2_final_cond"] = cond_outputs["final_cond"]

    # Step 3: T3 Generation (speech tokens)
    # Use temperature=0 for deterministic argmax sampling
    with torch.inference_mode():
        speech_tokens = model.t3.inference(
            t3_cond=model.conds.t3,
            text_tokens=text_tokens_cfg,
            max_new_tokens=1000,
            temperature=TEMPERATURE,
            cfg_weight=0.5,
            repetition_penalty=2.0,
            min_p=0.05,
            top_p=1.0,
        )
        # Extract only the conditional batch and ensure proper shape
        speech_tokens = speech_tokens[0]  # First batch (CFG output)
        if speech_tokens.dim() == 1:
            speech_tokens = speech_tokens.unsqueeze(0)  # Add batch dim for drop_invalid_tokens
        speech_tokens = drop_invalid_tokens(speech_tokens)
        speech_tokens = speech_tokens.to(device)

    outputs["step3_speech_tokens"] = speech_tokens.squeeze(0).detach().cpu().numpy().astype(np.int32)

    # Step 4: S3Gen Embedding (gen_conds is a dict)
    gen_conds = model.conds.gen

    # Get prompt conditioning (it's a dict, not a dataclass)
    prompt_token = gen_conds["prompt_token"].to(device)
    prompt_feat = gen_conds["prompt_feat"].to(device)
    embedding = gen_conds["embedding"].to(device)

    outputs["step4_prompt_token"] = prompt_token.detach().cpu().numpy()
    outputs["step4_prompt_feat"] = prompt_feat.detach().cpu().numpy()
    outputs["step4_embedding"] = embedding.detach().cpu().numpy()

    # Step 5: S3Gen Encoder
    # This requires more internal access - save what we can

    # Save metadata
    outputs["metadata"] = {
        "voice": test_case.voice,
        "sentence_id": test_case.sentence_id,
        "text": test_case.text,
        "language": test_case.language,
        "seed": SEED,
        "temperature": TEMPERATURE,
        "device": device,
    }

    return outputs


def save_outputs(outputs: dict, test_case: TestCase):
    """Save test outputs to disk."""
    case_dir = OUTPUT_DIR / test_case.voice / f"{test_case.sentence_id}_{test_case.language}"
    case_dir.mkdir(parents=True, exist_ok=True)

    for key, value in outputs.items():
        if key == "metadata":
            with open(case_dir / "metadata.json", "w") as f:
                json.dump(value, f, indent=2)
        else:
            np.save(case_dir / f"{key}.npy", value)

    return case_dir


def verify_outputs(outputs: dict) -> bool:
    """Basic sanity checks on outputs."""
    # Check shapes are reasonable
    if outputs["step1_text_tokens"].shape[0] == 0:
        return False
    if outputs["step2_final_cond"].shape[1] != 34:  # 1 + 32 + 1
        return False
    if outputs["step3_speech_tokens"].shape[0] == 0:
        return False
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run E2E tests for Python TTS")
    parser.add_argument("--voice", "-v", default=None, help="Filter to specific voice")
    parser.add_argument("--sentence", "-s", default=None, help="Filter to specific sentence ID")
    parser.add_argument("--device", "-d", default="cpu", help="Device (cpu, mps, cuda)")
    args = parser.parse_args()

    device = args.device

    print("=" * 80)
    print("PYTHON E2E TEST RUNNER")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Seed: {SEED}")
    print(f"Temperature: {TEMPERATURE} (near-deterministic)")
    print()

    # Set seeds before loading model
    set_deterministic_seeds(SEED)

    # Load model once
    print("Loading model...")
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    model = ChatterboxMultilingualTTS.from_local(str(MODEL_DIR), device=device)
    print("Model loaded successfully")
    print()

    # Get test cases
    test_cases = get_test_cases(args.voice, args.sentence)
    total = len(test_cases)
    print(f"Running {total} test cases...")
    print()

    passed = 0
    failed = 0

    for i, test_case in enumerate(test_cases, 1):
        print(f"[{i}/{total}] {test_case.voice}/{test_case.sentence_id}/{test_case.language}")
        print(f"  Text: {test_case.text[:50]}{'...' if len(test_case.text) > 50 else ''}")

        try:
            # Reset seeds for each test case for reproducibility
            set_deterministic_seeds(SEED)

            outputs = run_test_case(model, test_case, device)

            if not verify_outputs(outputs):
                print(f"  FAILED: Output verification failed")
                failed += 1
                print("\n" + "=" * 80)
                print("STOPPING ON FIRST FAILURE")
                print("=" * 80)
                sys.exit(1)

            case_dir = save_outputs(outputs, test_case)
            print(f"  PASSED: Saved to {case_dir.relative_to(PROJECT_ROOT)}")
            passed += 1

        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1
            print("\n" + "=" * 80)
            print("STOPPING ON FIRST FAILURE")
            print("=" * 80)
            import traceback
            traceback.print_exc()
            sys.exit(1)

    print()
    print("=" * 80)
    print(f"E2E TEST COMPLETE: {passed}/{total} passed")
    print("=" * 80)
    print(f"Reference outputs saved to: {OUTPUT_DIR}")
    print()
    print("Next step: Run Swift E2E test to compare:")
    print("  cd E2E && swift build && .build/debug/SwiftE2E")


if __name__ == "__main__":
    main()
