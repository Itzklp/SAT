#!/usr/bin/env python3
"""
build_eval_questions.py -- Pass 2: LLM phrasing of the selected eval
candidates into natural-language questions. The LLM only writes the
QUESTION TEXT here -- it never sees or influences the gold labels
(review_ids/sentences/contradiction/answerable), which were already fixed
deterministically in Pass 1 (build_eval_dataset.py / sample_eval_candidates.py).

Checkpoints after every batch (lesson learned: an earlier long-running
generation was killed mid-run with nothing saved).
"""

import argparse
import json
import os
import warnings
warnings.filterwarnings("ignore", message=".*Failed to load image Python extension.*")

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

MODEL_ID = "NousResearch/Meta-Llama-3-8B-Instruct"
BATCH_SIZE = 8
MAX_NEW_TOKENS = 40


def safe_dtype():
    if not torch.cuda.is_available():
        return None
    try:
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
    except Exception:
        pass
    return None


def get_generator():
    print(f"Loading Teacher Model: {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    dtype = safe_dtype()
    model_kwargs = dict(device_map="auto", trust_remote_code=True)
    if dtype is not None:
        model_kwargs["dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **model_kwargs)
    return pipeline("text-generation", model=model, tokenizer=tokenizer, batch_size=BATCH_SIZE)


def chat_prompt(tokenizer, system, user):
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def gen_batch(gen, prompts, max_new_tokens=MAX_NEW_TOKENS, temperature=0.9):
    if not prompts:
        return []
    outs = gen(prompts, max_new_tokens=max_new_tokens, temperature=temperature, do_sample=True,
               return_full_text=False, pad_token_id=gen.tokenizer.eos_token_id)
    results = []
    for o in outs:
        if isinstance(o, list):
            o = o[0]
        results.append(o.get("generated_text", "").strip().strip('"').strip())
    return results


SYS_ASPECT = (
    "You write realistic, specific shopping questions a customer would ask about a smartphone. "
    "Refer to it generically as \"this phone\" or \"the phone\" -- do NOT invent or name a specific "
    "brand or model (no \"Galaxy\", \"iPhone\", \"Pixel\", etc.). "
    "Output ONLY the question, one sentence, no preamble or quotes."
)
SYS_FEATURE = (
    "You write realistic shopping questions asking whether a smartphone has or supports a specific "
    "feature. Refer to it generically as \"this phone\" or \"the phone\" -- do NOT invent or name a "
    "specific brand or model (no \"Galaxy\", \"iPhone\", \"Pixel\", etc.). "
    "Output ONLY the question, one sentence, no preamble or quotes."
)


def build_prompt(tok, item, category):
    if category in ("aspect_specific", "contradiction", "unsupported_aspect"):
        return chat_prompt(tok, SYS_ASPECT,
            f"Aspect to ask about: {item['aspect']}\nWrite one natural question a customer would ask about a smartphone's {item['aspect']}.")
    if category == "persona_aware":
        return chat_prompt(tok, SYS_ASPECT,
            f"Persona: {item['persona']}\nAspect to ask about: {item['aspect']}\n"
            f"Write one natural question this persona would ask about a smartphone's {item['aspect']}.")
    if category == "multi_aspect":
        a1, a2 = item["aspect"]
        return chat_prompt(tok, SYS_ASPECT,
            f"Write one natural question that asks about BOTH a smartphone's {a1} AND its {a2} together.")
    if category == "overall_suitability":
        return chat_prompt(tok, SYS_ASPECT,
            "Write one natural question asking whether a smartphone is a good overall choice / worth buying.")
    if category == "unsupported_feature":
        return chat_prompt(tok, SYS_FEATURE,
            f"Feature: {item['aspect']}\nWrite one natural question asking whether a smartphone supports or has this feature.")
    raise ValueError(category)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_file", type=str, default="eval_selected.json")
    ap.add_argument("--out_file", type=str, default="eval_dataset.json")
    ap.add_argument("--limit", type=int, default=None, help="only phrase the first N (for smoke testing)")
    args = ap.parse_args()

    selected = json.load(open(args.in_file))

    # Flatten everything except "ambiguous" (already has query_template, no LLM needed) into one list.
    flat = []
    for category, items in selected.items():
        if category == "ambiguous":
            continue
        for it in items:
            flat.append({**it, "category": category})

    if args.limit is not None:
        flat = flat[:args.limit]

    out_path = args.out_file
    done = []
    if os.path.exists(out_path):
        # Only count non-ambiguous entries for resume purposes -- ambiguous
        # items are appended once, only after the phrasing loop fully
        # completes (see below), so they must never affect start_idx.
        done = [d for d in json.load(open(out_path)) if d.get("category") != "ambiguous"]
        print(f"Resuming: {len(done)} already phrased.")
    start_idx = len(done)

    gen = get_generator()

    for i in tqdm(range(start_idx, len(flat), BATCH_SIZE), desc="Phrasing eval questions"):
        batch = flat[i:i + BATCH_SIZE]
        prompts = [build_prompt(gen.tokenizer, item, item["category"]) for item in batch]
        questions = gen_batch(gen, prompts)
        for item, q in zip(batch, questions):
            item["query"] = q
            done.append(item)
        with open(out_path, "w") as f:
            json.dump(done, f, indent=2)

    # Only append the ambiguous category (no LLM needed) once the full,
    # unlimited flat list has actually been phrased -- guards against a
    # --limit smoke-test run silently baking in all 40 ambiguous items,
    # which would corrupt start_idx on a subsequent full resume.
    if args.limit is None and len(done) >= len(flat):
        for it in selected["ambiguous"]:
            done.append({**it, "category": "ambiguous", "query": it["query_template"]})
        with open(out_path, "w") as f:
            json.dump(done, f, indent=2)

    print(f"Done. {len(done)} total eval questions saved to {out_path}")


if __name__ == "__main__":
    main()
