#!/usr/bin/env python3
"""
Debug script to understand token probability distribution at each generation step.
Helps identify where "Now" hallucination is coming from.
"""
import logging
import sys
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logging.getLogger('numba').setLevel(logging.ERROR)
logging.getLogger('torio').setLevel(logging.ERROR)
logging.getLogger('torch').setLevel(logging.ERROR)
logging.getLogger('urllib3').setLevel(logging.ERROR)
logging.getLogger('chatterbox').setLevel(logging.ERROR)

sys.path.insert(0, '/Users/a10n/Projects/nightingale_TTS/python/chatterbox/src')

import torch
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")
BAKED_VOICES_DIR = PROJECT_ROOT / "baked_voices"


def debug_generation():
    """Trace through generation to find where 'Now' comes from."""
    from chatterbox.tts import ChatterboxTTS, punc_norm
    import torch.nn.functional as F
    from tqdm import tqdm
    from transformers.generation.logits_process import (
        RepetitionPenaltyLogitsProcessor,
        TopPLogitsWarper,
        MinPLogitsWarper,
    )
    from chatterbox.models.t3.inference.logits_processors import (
        ReferenceSuppressionLogitsProcessor,
        StepwiseTemperatureLogitsProcessor,
    )
    from chatterbox.models.t3.inference.alignment_stream_analyzer import AlignmentStreamAnalyzer

    print("=" * 70)
    print("DEBUGGING TOKEN PROBABILITIES FOR 'Hi.' (WITH CFG ANNEALING)")
    print("=" * 70)
    print()

    device = "mps"
    model = ChatterboxTTS.from_pretrained(device=device)

    voice = "samantha"
    ref_audio_path = BAKED_VOICES_DIR / voice / "ref_audio.wav"
    model.prepare_conditionals(str(ref_audio_path), exaggeration=0.5)

    text = "Hi."
    norm_text = punc_norm(text)
    text_tokens = model.tokenizer.text_to_tokens(norm_text).to(device)

    sot = model.t3.hp.start_text_token
    eot = model.t3.hp.stop_text_token
    text_tokens_padded = F.pad(text_tokens, (1, 0), value=sot)
    text_tokens_padded = F.pad(text_tokens_padded, (0, 1), value=eot)
    text_len = text_tokens_padded.size(-1)

    print(f"Text: '{text}'")
    print(f"Text tokens: {text_tokens_padded[0].tolist()}")
    print(f"Text length: {text_len}")
    print()

    # Set up the generation (mimicking t3.inference)
    t3 = model.t3
    t3_cond = model.conds.t3
    cfg_weight_base = 0.5

    # CFG annealing parameters (matching t3.py)
    cfg_warmup_steps = 5   # Shorter warmup for aggressive correction
    cfg_start = 5.0        # Very high CFG = extreme text adherence
    cfg_end = cfg_weight_base
    is_short_text = text_len < 16

    print(f"CFG Annealing: start={cfg_start}, end={cfg_end}, warmup={cfg_warmup_steps}")
    print(f"Is short text: {is_short_text}")
    print()

    # Duplicate text tokens for CFG
    text_tokens_cfg = torch.cat([text_tokens_padded, text_tokens_padded], dim=0)

    # Prepare custom input embeds
    embeds, len_cond = t3.prepare_input_embeds(
        t3_cond=t3_cond,
        text_tokens=text_tokens_cfg,
        speech_tokens=t3.hp.start_speech_token * torch.ones_like(text_tokens_cfg[:, :1]),
        cfg_weight=cfg_weight_base,
    )

    # BOS token
    bos_token = torch.tensor([[t3.hp.start_speech_token]], dtype=torch.long, device=device)
    bos_embed = t3.speech_emb(bos_token)
    bos_embed = bos_embed + t3.speech_pos_emb.get_fixed_embedding(0)
    bos_embed = torch.cat([bos_embed, bos_embed])

    inputs_embeds = torch.cat([embeds, bos_embed], dim=1)
    generated_ids = bos_token.clone()

    # Get the conditioning tokens for reference
    cond_tokens = t3_cond.cond_prompt_speech_tokens.flatten().tolist()
    cond_set = set(cond_tokens)

    # Processors
    reference_suppressor = ReferenceSuppressionLogitsProcessor(
        reference_tokens=t3_cond.cond_prompt_speech_tokens,
        penalty=5.0,
        min_text_tokens=16,
    )
    repetition_penalty_processor = RepetitionPenaltyLogitsProcessor(penalty=1.2)
    min_p_warper = MinPLogitsWarper(min_p=0.05)
    top_p_warper = TopPLogitsWarper(top_p=1.0)

    # Compile model
    alignment_stream_analyzer = AlignmentStreamAnalyzer(
        t3.tfmr,
        None,
        text_tokens_slice=(len_cond, len_cond + text_len),
        alignment_layer_idx=9,
        eos_idx=t3.hp.stop_speech_token,
    )

    from chatterbox.models.t3.inference.t3_hf_backend import T3HuggingfaceBackend
    patched_model = T3HuggingfaceBackend(
        config=t3.cfg,
        llama=t3.tfmr,
        speech_enc=t3.speech_emb,
        speech_head=t3.speech_head,
        alignment_stream_analyzer=alignment_stream_analyzer,
    )

    # Initial forward pass
    with torch.inference_mode():
        output = patched_model(
            inputs_embeds=inputs_embeds,
            past_key_values=None,
            use_cache=True,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past = output.past_key_values

        print("=" * 70)
        print("STEP-BY-STEP TOKEN ANALYSIS (First 20 steps)")
        print("=" * 70)
        print()

        for step in range(20):
            logits_step = output.logits[:, -1, :]

            # CFG combine with stepwise annealing
            cond = logits_step[0:1, :]
            uncond = logits_step[1:2, :]

            # Calculate current CFG weight (matching t3.py logic)
            if is_short_text and step < cfg_warmup_steps:
                progress = step / cfg_warmup_steps
                current_cfg = cfg_start + (cfg_end - cfg_start) * progress
            else:
                current_cfg = cfg_weight_base

            cfg = torch.as_tensor(current_cfg, device=cond.device, dtype=cond.dtype)
            logits_raw = cond + cfg * (cond - uncond)

            # Get top tokens BEFORE any processing
            probs_raw = torch.softmax(logits_raw, dim=-1)
            top_k_raw = torch.topk(probs_raw[0], k=5)

            print(f"--- Step {step} (CFG={current_cfg:.2f}) ---")
            print(f"  AFTER CFG combine:")
            for i, (idx, prob) in enumerate(zip(top_k_raw.indices, top_k_raw.values)):
                in_cond = "⚠️ IN_COND" if idx.item() in cond_set else ""
                print(f"    #{i+1}: token {idx.item():5d} prob={prob.item():.4f} {in_cond}")

            # Apply reference suppression
            logits = reference_suppressor(generated_ids[:1, ...], logits_raw.clone(), text_len=text_len)
            probs_after_ref = torch.softmax(logits, dim=-1)
            top_k_after_ref = torch.topk(probs_after_ref[0], k=5)

            print(f"  AFTER reference suppression:")
            for i, (idx, prob) in enumerate(zip(top_k_after_ref.indices, top_k_after_ref.values)):
                in_cond = "⚠️ IN_COND" if idx.item() in cond_set else ""
                print(f"    #{i+1}: token {idx.item():5d} prob={prob.item():.4f} {in_cond}")

            # Apply alignment stream analyzer
            if alignment_stream_analyzer is not None:
                if logits.dim() == 1:
                    logits = logits.unsqueeze(0)
                last_token = generated_ids[0, -1].item() if len(generated_ids[0]) > 0 else None
                logits = alignment_stream_analyzer.step(logits, next_token=last_token)

            # Apply repetition penalty
            ids_for_proc = generated_ids[:1, ...]
            logits = repetition_penalty_processor(ids_for_proc, logits)

            # Apply temperature (0.8)
            temperature = 0.8
            logits = logits / temperature

            print(f"  AFTER all processing (temp={temperature}):")
            probs_final = torch.softmax(logits, dim=-1)
            top_k_final = torch.topk(probs_final[0], k=5)
            for i, (idx, prob) in enumerate(zip(top_k_final.indices, top_k_final.values)):
                in_cond = "⚠️ IN_COND" if idx.item() in cond_set else ""
                print(f"    #{i+1}: token {idx.item():5d} prob={prob.item():.4f} {in_cond}")

            # Apply min_p and top_p
            logits = min_p_warper(ids_for_proc, logits)
            logits = top_p_warper(ids_for_proc, logits)

            # Sample
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            in_cond = "⚠️ IN_COND" if next_token.item() in cond_set else ""
            print(f"  SAMPLED: token {next_token.item()} {in_cond}")
            print()

            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            if next_token.view(-1) == t3.hp.stop_speech_token:
                print(f"EOS at step {step}")
                break

            # Next step embedding
            next_token_embed = t3.speech_emb(next_token)
            next_token_embed = next_token_embed + t3.speech_pos_emb.get_fixed_embedding(step + 1)
            next_token_embed = torch.cat([next_token_embed, next_token_embed])

            output = patched_model(
                inputs_embeds=next_token_embed,
                past_key_values=past,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past = output.past_key_values

    # Cleanup
    alignment_stream_analyzer.cleanup()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_tokens = generated_ids[0].tolist()
    print(f"Generated {len(all_tokens)} tokens: {all_tokens}")
    matches = sum(1 for t in all_tokens if t in cond_set)
    print(f"Tokens matching conditioning: {matches} ({100*matches/len(all_tokens):.1f}%)")


if __name__ == "__main__":
    debug_generation()
