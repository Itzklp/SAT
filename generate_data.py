#!/usr/bin/env python3
"""
generate_data_fixed.py
Quick-fix version that avoids Flash Attention import (GLIBC issues) and uses `dtype`.
"""

import json
import math
import random
import time
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ------------- CONFIG -------------
MODEL_ID = "NousResearch/Meta-Llama-3-8B-Instruct"  # may require HF auth
OUTPUT_FILE = "synthetic_dpo_data.json"
N_SAMPLES = 1000
BATCH_SIZE = 8
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.8
# -----------------------------------

PERSONAS = ["Pro Gamer", "Budget Student", "Professional Photographer", "Grandmother buying gifts"]

def safe_dtype():
    """Return a dtype safe to pass to from_pretrained (or None)."""
    if not torch.cuda.is_available():
        return None
    # prefer bfloat16 only when CUDA + driver supports it
    try:
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
    except Exception:
        pass
    return None

def get_generator():
    """Load tokenizer + model and return a text-generation pipeline.
    Avoids passing attn_implementation to prevent lazy import of flash_attn.
    Uses `dtype` (transformers deprecation fix) and falls back on low_cpu_mem_usage.
    """
    print(f"Loading Teacher Model: {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dtype = safe_dtype()
    model_kwargs = dict(device_map="auto", trust_remote_code=True)
    if dtype is not None:
        model_kwargs["dtype"] = dtype

    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **model_kwargs)
    except Exception as e:
        print("Primary model load failed, retrying with low_cpu_mem_usage=True. Error:", e)
        try:
            model = AutoModelForCausalLM.from_pretrained(MODEL_ID, low_cpu_mem_usage=True, **model_kwargs)
        except Exception as e2:
            # Last resort: attempt CPU load (very slow) so the script fails more informatively
            print("Retry with CPU (very slow). Error:", e2)
            model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map={"": "cpu"}, trust_remote_code=True)

    gen = pipeline("text-generation", model=model, tokenizer=tokenizer, batch_size=BATCH_SIZE)
    return gen

def prompt_from_persona(persona):
    sys_msg = (
        "You are a synthetic data generator. Create a DPO training example for a Shopping Assistant.\n"
        f"Context: The user is a {persona}.\n"
        "Generate a JSON object with 3 fields:\n"
        "1. 'user_query': A plausible shopping question.\n"
        "2. 'chosen': A helpful, persona-aware, safe response.\n"
        "3. 'rejected': A generic, robotic, or slightly hallucinated response.\n"
        "Output ONLY valid JSON.\n"
        "Respond with the JSON object only."
    )
    return sys_msg + "\nGenerate example."

def generate_prompt_batch(personas, batch_size):
    return [prompt_from_persona(random.choice(personas)) for _ in range(batch_size)]

def extract_first_json(raw_text):
    """Extract first JSON object found in raw_text. Return dict or None."""
    start = raw_text.find('{')
    end = raw_text.rfind('}') + 1
    if start == -1 or end == 0 or end <= start:
        return None
    candidate = raw_text[start:end]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None

def main():
    pipe = get_generator()
    dataset = []

    print(f"Generating up to {N_SAMPLES} synthetic samples (batch size {BATCH_SIZE})...")
    n_batches = math.ceil(N_SAMPLES / BATCH_SIZE)

    for _ in tqdm(range(n_batches)):
        prompts = generate_prompt_batch(PERSONAS, BATCH_SIZE)
        try:
            results = pipe(
                prompts,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=True,
                return_full_text=False
            )
        except Exception as e:
            print("Generation error, sleeping 5s and retrying batch. Error:", e)
            time.sleep(5)
            try:
                results = pipe(prompts, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, do_sample=True, return_full_text=False)
            except Exception as e2:
                print("Retry failed; skipping this batch. Error:", e2)
                continue

        # results: list of dicts {'generated_text': "..."} or list of list/dicts depending on pipeline version
        for item in results:
            # handle both shapes: {'generated_text': '...'} or [{'generated_text': '...'}]
            raw_text = None
            if isinstance(item, dict) and "generated_text" in item:
                raw_text = item["generated_text"]
            elif isinstance(item, list) and len(item) > 0 and isinstance(item[0], dict) and "generated_text" in item[0]:
                raw_text = item[0]["generated_text"]
            else:
                # try to stringify item
                raw_text = str(item)

            data = extract_first_json(raw_text)
            if not data:
                continue

            entry = {
                "prompt": f"Persona: {random.choice(PERSONAS)}\nUser: {data.get('user_query','')}",
                "chosen": data.get('chosen', ''),
                "rejected": data.get('rejected', '')
            }
            dataset.append(entry)

        # optional early stop if we've reached N_SAMPLES
        if len(dataset) >= N_SAMPLES:
            break

    # Fallback safety
    if len(dataset) < 10:
        print("Warning: Generation low yield. Adding dummy data.")
        dataset.append({
            "prompt": "Persona: Gamer\nUser: Best laptop?",
            "chosen": "For gaming, look for RTX 40 series.",
            "rejected": "I don't know."
        })

    # Trim dataset to N_SAMPLES if overshot
    dataset = dataset[:N_SAMPLES]

    with open(OUTPUT_FILE, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"Successfully saved {len(dataset)} samples to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

