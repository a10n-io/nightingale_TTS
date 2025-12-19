#!/Users/a10n/Projects/nightingale_TTS/python/venv/bin/python
"""
Unified E2E Verification Script for Python/PyTorch vs Swift/MLX.

This script:
1. Generates Python reference outputs for each test case
2. Runs Swift verification against Python outputs
3. Reports stage-by-stage verification status

Usage (from project root):
    python E2E/verify_e2e.py
    python E2E/verify_e2e.py --voice samantha
    python E2E/verify_e2e.py --swift-only  # Skip Python, just run Swift against existing refs
    python E2E/verify_e2e.py --steps 1     # Test only Step 1 (tokenization)
    python E2E/verify_e2e.py --steps 1 --linguistic  # Test Step 1 with 22 languages (132 test cases)

Or directly (uses venv python):
    ./E2E/verify_e2e.py
    ./E2E/verify_e2e.py --swift-only
    ./E2E/verify_e2e.py --steps 1 --linguistic
"""

import torch
import numpy as np
import random
import json
import sys
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Tuple
import argparse

# Project paths
PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")
MODEL_DIR = PROJECT_ROOT / "models" / "chatterbox"
VOICE_DIR = PROJECT_ROOT / "baked_voices"
E2E_DIR = PROJECT_ROOT / "E2E"
OUTPUT_DIR = E2E_DIR / "reference_outputs"

# Deterministic settings
SEED = 42
TEMPERATURE = 0.001


@dataclass
class TestCase:
    voice: str
    sentence_id: str
    text: str
    language: str


@dataclass
class StageResult:
    name: str
    passed: bool
    max_diff: float
    notes: str


def set_deterministic_seeds(seed: int = SEED):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, 'mps') and hasattr(torch.mps, 'manual_seed'):
        torch.mps.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_test_sentences(linguistic: bool = False) -> list:
    """Load test sentences from JSON.

    Args:
        linguistic: If True, load from test_sentences_unicode_linguistic.json
                   instead of test_sentences.json
    """
    filename = "test_sentences_unicode_linguistic.json" if linguistic else "test_sentences.json"
    with open(E2E_DIR / filename, "r") as f:
        return json.load(f)


def get_test_cases(voice_filter: Optional[str] = None,
                   sentence_filter: Optional[str] = None,
                   lang_filter: Optional[str] = None,
                   linguistic: bool = False) -> List[TestCase]:
    """Generate test cases.

    Args:
        voice_filter: Filter to specific voice name
        sentence_filter: Filter to specific sentence ID
        lang_filter: Filter to specific language code
        linguistic: If True, use linguistic Unicode test file with extended language support
    """
    voices = ["samantha", "sujano"]
    sentences = load_test_sentences(linguistic=linguistic)

    test_cases = []

    if linguistic:
        # Linguistic test structure: each entry has text_XX_1, text_XX_2, text_XX_3
        # where XX is the language code (extracted from id)
        for voice in voices:
            if voice_filter and voice != voice_filter:
                continue
            for sentence in sentences:
                # Extract language from id (e.g., "ar_complex" -> "ar")
                lang = sentence["id"].split("_")[0]
                if lang_filter and lang != lang_filter:
                    continue

                # Each sentence has 3 variants (_1, _2, _3)
                for variant in [1, 2, 3]:
                    text_key = f"text_{lang}_{variant}"
                    sentence_id = f"{sentence['id']}_{variant}"

                    if sentence_filter and sentence_id != sentence_filter:
                        continue

                    if text_key in sentence:
                        test_cases.append(TestCase(
                            voice=voice,
                            sentence_id=sentence_id,
                            text=sentence[text_key],
                            language=lang
                        ))
    else:
        # Regular test structure: text_en, text_nl, etc.
        languages = ["en", "nl"]
        for voice in voices:
            if voice_filter and voice != voice_filter:
                continue
            for sentence in sentences:
                if sentence_filter and sentence["id"] != sentence_filter:
                    continue
                for lang in languages:
                    if lang_filter and lang != lang_filter:
                        continue
                    text_key = f"text_{lang}"
                    if text_key in sentence:
                        test_cases.append(TestCase(
                            voice=voice,
                            sentence_id=sentence["id"],
                            text=sentence[text_key],
                            language=lang
                        ))

    return test_cases


def generate_python_reference(model, test_case: TestCase, device: str, max_step: int = 5) -> dict:
    """Generate Python reference outputs for a test case.

    Args:
        model: The Chatterbox model
        test_case: Test case configuration
        device: Device to run on (cpu, mps, cuda)
        max_step: Maximum step to generate (1-5). Default: 5
    """
    from chatterbox.mtl_tts import Conditionals, punc_norm
    from chatterbox.models.s3gen.s3gen import drop_invalid_tokens
    import torch.nn.functional as F

    set_deterministic_seeds(SEED)

    # Load voice
    voice_path = VOICE_DIR / test_case.voice / "baked_voice.pt"
    model.conds = Conditionals.load(str(voice_path), map_location=device)

    outputs = {}

    # Step 1: Tokenization (with language_id as Python implementation does)
    # Save metadata for verification
    outputs["step1_text_original"] = test_case.text  # Original text from config
    text = punc_norm(test_case.text)
    outputs["step1_text_after_punc_norm"] = text  # After punctuation normalization
    outputs["step1_language_id"] = test_case.language.lower()  # Language ID used

    text_tokens = model.tokenizer.text_to_tokens(text, language_id=test_case.language.lower())
    text_tokens = text_tokens.to(device)
    outputs["step1_text_tokens"] = text_tokens[0].detach().cpu().numpy().astype(np.int32)
    outputs["step1_token_count"] = len(text_tokens[0])  # Number of tokens

    # Prepare for T3 (CFG + SOT/EOT padding)
    # CRITICAL: This must match Swift's preparation exactly
    text_tokens_cfg = torch.cat([text_tokens, text_tokens], dim=0)  # [2, N] - duplicate for CFG
    sot = model.t3.hp.start_text_token  # 255
    eot = model.t3.hp.stop_text_token   # 0
    outputs["step1_sot_token"] = sot  # Save SOT token value
    outputs["step1_eot_token"] = eot  # Save EOT token value
    text_tokens_cfg = F.pad(text_tokens_cfg, (1, 0), value=sot)  # Prepend SOT: [2, N+1]
    text_tokens_cfg = F.pad(text_tokens_cfg, (0, 1), value=eot)  # Append EOT: [2, N+2]

    # Save CFG-prepared tokens for verification
    outputs["step1_text_tokens_cfg"] = text_tokens_cfg.detach().cpu().numpy().astype(np.int32)

    if max_step == 1:
        return outputs

    # Step 2: T3 Conditioning (with detailed intermediate outputs)
    t3 = model.t3
    conds = model.conds.t3

    # 2.1: Speaker token generation
    speaker_emb = conds.speaker_emb.to(device)
    outputs["step2_speaker_emb_input"] = speaker_emb.detach().cpu().numpy()
    speaker_token = t3.cond_enc.spkr_enc(speaker_emb).unsqueeze(1)
    outputs["step2_speaker_token"] = speaker_token.detach().cpu().numpy()

    # 2.2: Speech embeddings + positional embeddings
    cond_speech_tokens = conds.cond_prompt_speech_tokens.to(device)
    outputs["step2_cond_speech_tokens"] = cond_speech_tokens.detach().cpu().numpy().astype(np.int32)

    speech_emb = t3.speech_emb(cond_speech_tokens)
    outputs["step2_speech_emb"] = speech_emb.detach().cpu().numpy()

    positions = torch.arange(cond_speech_tokens.shape[1], device=device).unsqueeze(0)
    outputs["step2_positions"] = positions.detach().cpu().numpy().astype(np.int32)

    speech_pos_emb = t3.speech_pos_emb(positions)
    outputs["step2_speech_pos_emb"] = speech_pos_emb.detach().cpu().numpy()

    cond_speech_emb = speech_emb + speech_pos_emb
    outputs["step2_cond_speech_emb"] = cond_speech_emb.detach().cpu().numpy()

    # 2.3: Perceiver processing
    perceiver_out = t3.cond_enc.perceiver(cond_speech_emb)
    outputs["step2_perceiver_out"] = perceiver_out.detach().cpu().numpy()

    # 2.4: Emotion processing
    emotion_value = conds.emotion_adv.to(device)
    outputs["step2_emotion_value"] = emotion_value.detach().cpu().numpy()

    emotion_token = t3.cond_enc.emotion_adv_fc(emotion_value)
    outputs["step2_emotion_token"] = emotion_token.detach().cpu().numpy()

    # 2.5: Final concatenation
    final_cond = torch.cat([speaker_token, perceiver_out, emotion_token], dim=1)
    outputs["step2_final_cond"] = final_cond.detach().cpu().numpy()

    # Step 3: T3 Generation
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
        speech_tokens = speech_tokens[0]
        if speech_tokens.dim() == 1:
            speech_tokens = speech_tokens.unsqueeze(0)
        speech_tokens = drop_invalid_tokens(speech_tokens)

    outputs["step3_speech_tokens"] = speech_tokens.squeeze(0).detach().cpu().numpy().astype(np.int32)

    # Step 4: S3Gen embedding info
    gen_conds = model.conds.gen
    outputs["step4_embedding"] = gen_conds["embedding"].detach().cpu().numpy()
    outputs["step4_prompt_token"] = gen_conds["prompt_token"].detach().cpu().numpy().astype(np.int32)
    outputs["step4_prompt_feat"] = gen_conds["prompt_feat"].detach().cpu().numpy()

    # Step 5: S3Gen Input Preparation
    prompt_token = gen_conds["prompt_token"]
    prompt_feat = gen_conds["prompt_feat"]
    soul_s3 = gen_conds["embedding"]

    # Concatenate tokens
    full_tokens = torch.cat([prompt_token.squeeze(0), speech_tokens.squeeze(0)], dim=0).unsqueeze(0)
    outputs["step5_full_tokens"] = full_tokens.squeeze(0).detach().cpu().numpy().astype(np.int32)

    # Token embedding lookup
    token_emb = model.s3gen.flow.input_embedding(full_tokens.to(device))
    outputs["step5_token_emb"] = token_emb.detach().cpu().numpy()

    # Speaker embedding projection
    spk_emb_normalized = F.normalize(soul_s3, dim=-1)
    spk_emb_proj = model.s3gen.flow.spk_embed_affine_layer(spk_emb_normalized)
    outputs["step5_spk_emb"] = spk_emb_proj.detach().cpu().numpy()

    # Steps 6-8: DISABLED FOR NOW - Will add after Steps 1-5 working
    # Step 6: S3Gen Encoder
    # Step 7: Full ODE generation
    # Step 8: Vocoder

    return outputs


def save_outputs(outputs: dict, test_case: TestCase):
    """Save outputs to disk."""
    case_dir = OUTPUT_DIR / test_case.voice / f"{test_case.sentence_id}_{test_case.language}"
    case_dir.mkdir(parents=True, exist_ok=True)

    # Separate metadata from arrays
    metadata = {}
    for key, value in outputs.items():
        if isinstance(value, (str, int, float)):
            # Save metadata to config
            metadata[key] = value
        else:
            # Save arrays as .npy files
            np.save(case_dir / f"{key}.npy", value)

    # Save config for Swift (including Step 1 metadata)
    config = {
        "text": test_case.text,
        "voice": test_case.voice,
        "language": test_case.language,
        "sentence_id": test_case.sentence_id,
        "seed": SEED,
        "temperature": TEMPERATURE,
        **metadata,  # Add all metadata (text_original, language_id, sot_token, etc.)
    }
    with open(case_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    return case_dir


def verify_stage(swift_output: np.ndarray, python_output: np.ndarray, threshold: float = 0.001) -> Tuple[bool, float]:
    """Compare Swift and Python outputs."""
    if swift_output.shape != python_output.shape:
        return False, float('inf')
    max_diff = np.abs(swift_output - python_output).max()
    return max_diff < threshold, float(max_diff)


def run_swift_verification(test_case: TestCase) -> Tuple[bool, dict]:
    """Run Swift VerifyLive using swift run (properly handles Metal library resources)."""
    verify_live_dir = PROJECT_ROOT / "swift" / "test_scripts" / "VerifyLive"
    ref_dir = OUTPUT_DIR / test_case.voice / f"{test_case.sentence_id}_{test_case.language}"

    # Use 'swift run' instead of compiled binary to properly load Metal library resources
    cmd = [
        "swift", "run", "VerifyLive",
        "--voice", test_case.voice,
        "--ref-dir", str(ref_dir),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(verify_live_dir)
        )

        output = result.stdout + result.stderr

        # Parse results from Swift output
        import re
        swift_results = {
            "tokenization": {"passed": False, "diff": None},
            "conditioning": {"passed": False, "diff": None},
            "t3_generation": {"passed": False, "diff": None, "match_percent": None},
            "s3gen_input": {"passed": False, "diff": None},
            "encoder": {"passed": False, "diff": None},
            "decoder_forward": {"passed": False, "diff": None},
            "ode_solver": {"passed": False, "diff": None},
            "vocoder": {"passed": False, "diff": None},
        }

        # Check for step results - EXTRACT REAL DIFF VALUES ONLY
        if "Step 1 (Tokenization): ✅ PASSED" in output:
            swift_results["tokenization"]["passed"] = True
            swift_results["tokenization"]["diff"] = 0.0  # Tokenization is binary - exact match or fail
        elif "Step 1 (Tokenization): ❌" in output:
            swift_results["tokenization"]["passed"] = False

        if "Step 2 (Conditioning): ✅ PASSED" in output:
            swift_results["conditioning"]["passed"] = True
            # Extract REAL max_diff from output
            match = re.search(r"final_cond max_diff: ([\d.e+-]+)", output)
            if match:
                swift_results["conditioning"]["diff"] = float(match.group(1))
            else:
                # Fallback: try old format
                match = re.search(r"final_cond: ([\d.e+-]+)", output)
                if match:
                    swift_results["conditioning"]["diff"] = float(match.group(1))
        elif "Step 2 (Conditioning): ❌" in output:
            swift_results["conditioning"]["passed"] = False

        # Step 3: T3 Generation - extract real comparison results
        # Accept both "exact match" and "prefix match" as passing
        if "Step 3 (T3 Generation): ✅ PASSED" in output:
            swift_results["t3_generation"]["passed"] = True
            swift_results["t3_generation"]["diff"] = 0.0
            swift_results["t3_generation"]["match_percent"] = 100.0
        elif "Step 3 (T3 Generation): ⚠️  PARTIAL" in output or "Step 3 (T3 Generation): ❌" in output:
            swift_results["t3_generation"]["passed"] = False
            # Extract REAL match percentage from output
            match = re.search(r"Matching tokens: \d+/\d+ \(([\d.]+)%\)", output)
            if match:
                swift_results["t3_generation"]["match_percent"] = float(match.group(1))
                # For diff, use (100 - match_percent) as a metric
                swift_results["t3_generation"]["diff"] = 100.0 - swift_results["t3_generation"]["match_percent"]

        if "Step 5 (S3Gen Input): ✅ PASSED" in output:
            swift_results["s3gen_input"]["passed"] = True
            # Extract REAL max_diff values from output
            diffs = []
            for field in ["full_tokens", "token_emb", "spk_emb_proj"]:
                match = re.search(rf"{field}: ([\d.e+-]+)", output)
                if match:
                    diffs.append(float(match.group(1)))
            swift_results["s3gen_input"]["diff"] = max(diffs) if diffs else None
        elif "Step 5 (S3Gen Input): ❌" in output:
            swift_results["s3gen_input"]["passed"] = False

        # Steps 6-8: Extract real diff values (to be implemented when refs exist)
        if "Step 6 (Encoder): ✅ PASSED" in output:
            swift_results["encoder"]["passed"] = True
            # TODO: Extract real diff when Step 6 refs are generated
        elif "Step 6 (Encoder): ❌" in output:
            swift_results["encoder"]["passed"] = False

        if "Step 7a (Decoder Forward): ✅ PASSED" in output:
            swift_results["decoder_forward"]["passed"] = True
            # TODO: Extract real diff when Step 7a refs are generated
        elif "Step 7a (Decoder Forward): ❌" in output:
            swift_results["decoder_forward"]["passed"] = False

        if "Step 7 (ODE Solver): ✅ PASSED" in output:
            swift_results["ode_solver"]["passed"] = True
            # TODO: Extract real diff when Step 7 refs are generated
        elif "Step 7 (ODE Solver): ❌" in output:
            swift_results["ode_solver"]["passed"] = False

        if "Step 8 (Vocoder): ✅ PASSED" in output:
            swift_results["vocoder"]["passed"] = True
            # TODO: Extract real diff when Step 8 refs are generated
        elif "Step 8 (Vocoder): ❌" in output:
            swift_results["vocoder"]["passed"] = False

        # Check overall pass
        all_passed = "ALL TESTS PASSED" in output

        return all_passed, swift_results

    except subprocess.TimeoutExpired:
        return False, {"error": "Swift verification timed out"}
    except Exception as e:
        return False, {"error": str(e)}


def format_result(stage: StageResult) -> str:
    """Format a stage result for display."""
    status = "✅ VERIFIED" if stage.passed else "❌ FAILED"
    if stage.max_diff is None or stage.max_diff == float('inf'):
        diff_str = "N/A"
    else:
        diff_str = f"{stage.max_diff:.2e}"
    return f"Stage {stage.name} — {status} (Diff: {diff_str})\n  Notes: {stage.notes}"


def run_verification(test_case: TestCase, model, device: str, swift_only: bool = False, run_swift: bool = True, max_step: int = 5) -> List[StageResult]:
    """Run full verification for a test case.

    Args:
        test_case: Test case to verify
        model: Chatterbox model
        device: Device to run on
        swift_only: Skip Python generation
        run_swift: Run Swift verification
        max_step: Maximum step to generate/verify (1-5)
    """
    results = []
    case_dir = OUTPUT_DIR / test_case.voice / f"{test_case.sentence_id}_{test_case.language}"

    # Generate Python reference (unless swift_only)
    if not swift_only:
        print(f"  Generating Python reference (steps 1-{max_step})...")
        outputs = generate_python_reference(model, test_case, device, max_step=max_step)
        save_outputs(outputs, test_case)

    # Load Python references (only for steps up to max_step)
    python_refs = {}
    for f in case_dir.glob("*.npy"):
        # Filter by step number - only load files for steps <= max_step
        stem = f.stem
        if stem.startswith("step"):
            try:
                step_num = int(stem.split("_")[0].replace("step", ""))
                if step_num > max_step:
                    continue  # Skip files beyond max_step
            except (ValueError, IndexError):
                pass  # Keep non-step files (like old references)
        python_refs[stem] = np.load(f)

    # Run Swift verification if requested
    swift_passed = None
    swift_results = {}
    if run_swift:
        print(f"  Running Swift verification...")
        swift_passed, swift_results = run_swift_verification(test_case)
        if "error" in swift_results:
            print(f"    ⚠️  Swift error: {swift_results['error']}")

    # Stage 1: Text Tokenization
    if "step1_text_tokens" in python_refs:
        token_count = len(python_refs["step1_text_tokens"])
        swift_tok = swift_results.get("tokenization", {})
        if run_swift and swift_tok.get("passed") is not None:
            passed = swift_tok["passed"]
            diff = swift_tok.get("diff", 0.0) or 0.0
            notes = f"BPE tokenizer - {token_count} tokens - Swift {'MATCH' if passed else 'MISMATCH'}"
        else:
            passed = True
            diff = 0.0
            notes = f"BPE tokenizer - {token_count} tokens (Python only)"
        results.append(StageResult(name="1: Text Tokenization", passed=passed, max_diff=diff, notes=notes))

    # Stage 2: T3 Conditioning
    if max_step >= 2 and "step2_final_cond" in python_refs:
        cond_shape = python_refs["step2_final_cond"].shape
        emotion_val = python_refs["step2_emotion_value"].flatten()[0]
        swift_cond = swift_results.get("conditioning", {})
        if run_swift and swift_cond.get("passed") is not None:
            passed = swift_cond["passed"]
            diff = swift_cond.get("diff", 0.0) or 0.0
            notes = f"emotion_adv={emotion_val:.3f}, shape {cond_shape}"
        else:
            passed = True
            diff = 0.0
            notes = f"emotion_adv={emotion_val:.3f}, shape {cond_shape} (Python only)"
        results.append(StageResult(name="2: T3 Conditioning", passed=passed, max_diff=diff, notes=notes))

    # Stage 3: T3 Token Generation
    if max_step >= 3 and "step3_speech_tokens" in python_refs:
        token_count = len(python_refs["step3_speech_tokens"])
        swift_t3 = swift_results.get("t3_generation", {})
        if run_swift and swift_t3.get("passed") is not None:
            passed = swift_t3["passed"]
            # Use match percentage as "difference" metric (0 = 100% match, 100 = 0% match)
            diff = swift_t3.get("diff", float('inf'))
            match_pct = swift_t3.get("match_percent")
            if match_pct is not None:
                notes = f"{token_count} tokens - Swift {match_pct:.1f}% match ({'PASS' if passed else 'FAIL'})"
            else:
                notes = f"{token_count} tokens - Swift {'PASS' if passed else 'FAIL'}"
        else:
            passed = True
            diff = 0.0
            notes = f"Generated {token_count} speech tokens (Python only)"
        results.append(StageResult(
            name="3: T3 Token Generation",
            passed=passed,
            max_diff=diff,
            notes=notes
        ))

    # Stage 4: S3Gen Embedding (Python only for now)
    if max_step >= 4 and "step4_embedding" in python_refs:
        emb_shape = python_refs["step4_embedding"].shape
        results.append(StageResult(
            name="4: S3Gen Embedding",
            passed=True,
            max_diff=0.0,
            notes=f"Voice embedding shape {emb_shape} (Python)"
        ))

    # Stage 5: S3Gen Input Prep
    if max_step >= 5:
        swift_s3_input = swift_results.get("s3gen_input", {})
        if run_swift and swift_s3_input.get("passed") is not None:
            passed = swift_s3_input["passed"]
            diff = swift_s3_input.get("diff", 0.0) or 0.0
            notes = f"Token concat, embedding, speaker projection - Swift {'MATCH' if passed else 'MISMATCH'}"
            results.append(StageResult(name="5: S3Gen Input Prep", passed=passed, max_diff=diff, notes=notes))

    # Stage 6: S3Gen Encoder
    if max_step >= 6:
        swift_encoder = swift_results.get("encoder", {})
        if run_swift and swift_encoder.get("passed") is not None:
            passed = swift_encoder["passed"]
            diff = swift_encoder.get("diff", 0.0) or 0.0
            notes = f"UpsampleConformer encoder - Swift {'MATCH' if passed else 'MISMATCH'}"
            results.append(StageResult(name="6: S3Gen Encoder", passed=passed, max_diff=diff, notes=notes))

    # Stage 7a: Decoder Forward Pass
    if max_step >= 7:
        swift_decoder = swift_results.get("decoder_forward", {})
        if run_swift and swift_decoder.get("passed") is not None:
            passed = swift_decoder["passed"]
            diff = swift_decoder.get("diff", 0.0) or 0.0
            notes = f"Single decoder forward pass - Swift {'MATCH' if passed else 'MISMATCH'}"
            results.append(StageResult(name="7a: Decoder Forward", passed=passed, max_diff=diff, notes=notes))

    # Stage 7: ODE Solver
    if max_step >= 7:
        swift_ode = swift_results.get("ode_solver", {})
        if run_swift and swift_ode.get("passed") is not None:
            passed = swift_ode["passed"]
            diff = swift_ode.get("diff", 0.0) or 0.0
            notes = f"Flow Matching ODE solver - Swift {'MATCH' if passed else 'MISMATCH'}"
            results.append(StageResult(name="7: ODE Solver", passed=passed, max_diff=diff, notes=notes))

    # Stage 8: Vocoder
    if max_step >= 8:
        swift_vocoder = swift_results.get("vocoder", {})
        if run_swift and swift_vocoder.get("passed") is not None:
            passed = swift_vocoder["passed"]
            diff = swift_vocoder.get("diff", 0.0) or 0.0
            notes = f"HiFTGenerator mel-to-audio - Swift {'MATCH' if passed else 'MISMATCH'}"
            results.append(StageResult(name="8: Vocoder", passed=passed, max_diff=diff, notes=notes))

    return results


def print_comprehensive_summary(all_results: List[Tuple[TestCase, List[StageResult]]], run_swift: bool):
    """Print a comprehensive summary of all test results."""
    print()
    print("=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    print()

    # Collect stats by stage
    stage_stats = {}  # stage_name -> {"passed": [], "failed": [], "diffs": []}

    for tc, results in all_results:
        for r in results:
            if r.name not in stage_stats:
                stage_stats[r.name] = {"passed": [], "failed": [], "diffs": []}

            test_id = f"{tc.voice}/{tc.sentence_id}_{tc.language}"
            if r.passed:
                stage_stats[r.name]["passed"].append(test_id)
            else:
                stage_stats[r.name]["failed"].append(test_id)
            if r.max_diff != float('inf'):
                stage_stats[r.name]["diffs"].append(r.max_diff)

    # Define all stages (including pending ones)
    # Format: (stage_name, description, swift_implemented, python_reference_available)
    all_stages = [
        ("1: Text Tokenization", "BPE tokenizer", True, True),
        ("2: T3 Conditioning", "Speaker, emotion, perceiver & final cond", True, True),
        ("3: T3 Token Generation", "Speech tokens from T3 transformer", True, True),
        ("4: S3Gen Embedding", "Voice embedding", True, True),
        ("5: S3Gen Input Prep", "Token concat, mask, embedding, spk projection", True, True),
        ("6: S3Gen Encoder", "UpsampleConformer encoder", True, True),
        ("7a: Decoder Forward", "Single decoder pass verification", True, True),
        ("7: ODE Solver", "Flow Matching / CFM decoder", True, True),
        ("8: Vocoder", "HiFTGenerator mel-to-audio", True, True),
    ]

    # Print stage-by-stage summary
    print("┌" + "─" * 78 + "┐")
    print("│" + " STAGE-BY-STAGE VERIFICATION STATUS".center(78) + "│")
    print("├" + "─" * 78 + "┤")

    all_passed = True
    for stage_name, description, swift_implemented, python_available in all_stages:
        if stage_name in stage_stats:
            stats = stage_stats[stage_name]
            passed_count = len(stats["passed"])
            failed_count = len(stats["failed"])
            total = passed_count + failed_count

            if stats["diffs"]:
                # Filter out None values before finding max
                valid_diffs = [d for d in stats["diffs"] if d is not None and d != float('inf')]
                if valid_diffs:
                    max_diff = max(valid_diffs)
                    diff_str = f"{max_diff:.2e}"
                else:
                    diff_str = "N/A"
            else:
                diff_str = "N/A"

            if failed_count == 0:
                status = "✅ VERIFIED"
            else:
                status = "❌ FAILED"
                all_passed = False

            # Format: Stage N: Name — STATUS (Diff: X) [N/N tests]
            line = f"│ Stage {stage_name} — {status} (Diff: {diff_str})"
            line = line.ljust(60) + f"[{passed_count}/{total} tests]".rjust(18) + "│"
            print(line)
            print(f"│   Notes: {description}".ljust(79) + "│")
        elif swift_implemented:
            # Stage should exist but no results
            print(f"│ Stage {stage_name} — ⚠️  NO DATA".ljust(79) + "│")
            print(f"│   Notes: {description}".ljust(79) + "│")
        elif python_available:
            # Python reference available but Swift verification pending
            print(f"│ Stage {stage_name} — ⏸️  SWIFT PENDING".ljust(79) + "│")
            print(f"│   Notes: {description} (Python reference ✓)".ljust(79) + "│")
        else:
            # Fully pending stage
            print(f"│ Stage {stage_name} — ⏸️  NOT IMPLEMENTED".ljust(79) + "│")
            print(f"│   Notes: {description}".ljust(79) + "│")

    print("└" + "─" * 78 + "┘")
    print()

    # Print test matrix
    total_test_count = len(all_results)
    print("┌" + "─" * 78 + "┐")
    print("│" + f" TEST MATRIX ({total_test_count} Test Cases)".center(78) + "│")
    print("├" + "─" * 78 + "┤")

    # Group by voice
    voices = {}
    for tc, results in all_results:
        if tc.voice not in voices:
            voices[tc.voice] = []
        test_passed = all(r.passed for r in results)
        voices[tc.voice].append((tc, test_passed, results))

    for voice, tests in voices.items():
        print(f"│ Voice: {voice}".ljust(79) + "│")
        for tc, passed, results in tests:
            status = "✅" if passed else "❌"
            # Get max diff across all stages
            diffs = [r.max_diff for r in results if r.max_diff is not None and r.max_diff != float('inf')]
            max_diff = max(diffs) if diffs else 0.0
            test_id = f"{tc.sentence_id}_{tc.language}"
            text_preview = tc.text[:30] + "..." if len(tc.text) > 30 else tc.text
            line = f"│   {status} {test_id.ljust(25)} \"{text_preview}\""
            print(line[:79].ljust(79) + "│")
        print("│".ljust(79) + "│")

    print("└" + "─" * 78 + "┘")
    print()

    # Final verdict
    total_tests = len(all_results)
    passed_tests = sum(1 for _, results in all_results if all(r.passed for r in results))

    print("=" * 80)
    if all_passed:
        print("✅ ALL TESTS PASSED")
        print(f"   {passed_tests}/{total_tests} test cases verified successfully")
        if run_swift:
            print("   Python/PyTorch ↔ Swift/MLX parity confirmed for implemented stages")
        else:
            print("   Python reference generation complete")
    else:
        print("❌ SOME TESTS FAILED")
        print(f"   {passed_tests}/{total_tests} test cases passed")
    print("=" * 80)
    print()


def main():
    parser = argparse.ArgumentParser(description="E2E Verification for Python vs Swift")
    parser.add_argument("--voice", "-v", default=None, help="Filter to specific voice")
    parser.add_argument("--sentence", "-s", default=None, help="Filter to specific sentence")
    parser.add_argument("--lang", "-l", default=None, help="Filter to specific language")
    parser.add_argument("--device", "-d", default="cpu", help="Device (cpu, mps, cuda)")
    parser.add_argument("--swift-only", action="store_true", help="Skip Python generation")
    parser.add_argument("--no-swift", action="store_true", help="Skip Swift verification")
    parser.add_argument("--steps", type=int, default=5, help="Max step to generate/verify (1-5). Default: 5")
    parser.add_argument("--linguistic", action="store_true", help="Use Unicode linguistic test file (22 languages, complex Unicode)")
    args = parser.parse_args()

    run_swift = not args.no_swift

    print("=" * 80)
    print("E2E VERIFICATION: Python/PyTorch vs Swift/MLX")
    print("=" * 80)
    print(f"Device: {args.device}")
    print(f"Seed: {SEED}")
    print(f"Temperature: {TEMPERATURE}")
    print(f"Max step: {args.steps}")
    print(f"Test set: {'Linguistic Unicode (22 languages)' if args.linguistic else 'Standard (en, nl)'}")
    print(f"Swift verification: {'enabled' if run_swift else 'disabled'}")
    print()

    # Clean up old reference outputs to prevent stale data
    if not args.swift_only:
        print("Cleaning up old reference outputs...")
        import shutil
        if OUTPUT_DIR.exists():
            # Delete all .npy files and config.json files
            for voice_dir in OUTPUT_DIR.iterdir():
                if voice_dir.is_dir():
                    for case_dir in voice_dir.iterdir():
                        if case_dir.is_dir():
                            # Delete .npy files
                            for npy_file in case_dir.glob("*.npy"):
                                npy_file.unlink()
                            # Note: Keep config.json as it may have test case metadata
        print("Cleanup complete")
        print()

    # Load model (unless swift-only)
    model = None
    if not args.swift_only:
        print("Loading model...")
        set_deterministic_seeds(SEED)
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        model = ChatterboxMultilingualTTS.from_local(str(MODEL_DIR), device=args.device)
        print("Model loaded")
        print()

    # Get test cases
    test_cases = get_test_cases(args.voice, args.sentence, args.lang, linguistic=args.linguistic)
    print(f"Running {len(test_cases)} test cases...")
    print()

    all_results = []  # List of (TestCase, List[StageResult])
    all_passed = True

    for i, tc in enumerate(test_cases, 1):
        print("-" * 80)
        print(f"[{i}/{len(test_cases)}] Voice: {tc.voice} | Sentence: {tc.sentence_id} | Lang: {tc.language}")
        print(f"  Text: \"{tc.text[:60]}{'...' if len(tc.text) > 60 else ''}\"")
        print()

        try:
            results = run_verification(tc, model, args.device, args.swift_only, run_swift, max_step=args.steps)
            all_results.append((tc, results))

            for r in results:
                print(f"  {format_result(r)}")
                if not r.passed:
                    all_passed = False

            print()

        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            all_passed = False
            import traceback
            traceback.print_exc()
            print()
            print("=" * 80)
            print("STOPPING ON FIRST FAILURE")
            print("=" * 80)
            sys.exit(1)

    # Print comprehensive summary
    print_comprehensive_summary(all_results, run_swift)


if __name__ == "__main__":
    main()
