#!/usr/bin/env python3
"""
regenerate_rejected.py

Targeted patch: regenerates ONLY the "rejected" field of an existing
synthetic_dpo_data.json, using the length-matched ungrounded prompt (see
generate_data.py). The original run's "rejected" side was truncated
mid-sentence in 95% of examples (no length constraint on that prompt,
same 200-token cap as the constrained "chosen" side) -- a length confound
for DPO. Everything else (question, chosen, evidence, provenance) is left
untouched; only "rejected" is replaced, in place, with a backup of the
original file kept alongside it.
"""

import argparse
import json
import os
import shutil
import warnings
warnings.filterwarnings("ignore", message=".*Failed to load image Python extension.*")

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

MODEL_ID = "NousResearch/Meta-Llama-3-8B-Instruct"
BATCH_SIZE = 8
MAX_NEW_TOKENS_ANS = 200


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


def gen_batch(gen, prompts, max_new_tokens, temperature):
    if not prompts:
        return []
    outs = gen(
        prompts, max_new_tokens=max_new_tokens, temperature=temperature, do_sample=temperature > 0,
        return_full_text=False, pad_token_id=gen.tokenizer.eos_token_id,
    )
    results = []
    for o in outs:
        if isinstance(o, list):
            o = o[0]
        results.append(o.get("generated_text", "").strip())
    return results


def extract_question(prompt_text):
    # prompt format: "Persona: {persona}\nUser: {question}"
    return prompt_text.split("\nUser:", 1)[-1].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=str, default="synthetic_dpo_data.json")
    ap.add_argument("--limit", type=int, default=None, help="only regenerate the first N examples (for smoke testing)")
    args = ap.parse_args()

    backup_path = args.file + ".pre_rejected_fix.bak"
    if not os.path.exists(backup_path):
        shutil.copy(args.file, backup_path)
        print(f"Backed up original to {backup_path}")

    with open(args.file) as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} examples.")

    n = len(dataset) if args.limit is None else min(args.limit, len(dataset))

    gen = get_generator()

    for i in tqdm(range(0, n, BATCH_SIZE), desc="Regenerating rejected"):
        batch = dataset[i:i + BATCH_SIZE]
        prompts = [
            chat_prompt(
                gen.tokenizer,
                "You are a helpful shopping assistant. Answer the user's question as best you can. "
                "Adapt tone to the persona. Keep the answer to 3-5 sentences.",
                f"Persona: {e['persona']}\nProduct: {e['product_title']}\nQuestion: {extract_question(e['prompt'])}",
            )
            for e in batch
        ]
        answers = gen_batch(gen, prompts, MAX_NEW_TOKENS_ANS, temperature=0.9)
        for e, ans in zip(batch, answers):
            if ans.strip():
                e["rejected"] = ans.strip()

        with open(args.file, "w") as f:
            json.dump(dataset, f, indent=2)

    print(f"Done. Rewrote 'rejected' for {n} examples in {args.file}")


if __name__ == "__main__":
    main()
