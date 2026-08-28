#!/usr/bin/env python3
"""
generate_data.py

Grounded synthetic SFT/DPO data generator for SAT.

Replaces the prior version, which generated generic shopping-assistant Q&A
(camera bags, craft gifts, gaming keyboards) with zero connection to phones
or to any review evidence. This version is grounded end-to-end:

  1. Samples a product from the TRAIN split (dataset/sat/product_splits.json)
     -- eval-split products are never touched here, so the later evaluation
     set is genuinely held out.
  2. Uses the real Librarian (librarian.py) to retrieve real evidence
     sentences for a persona-biased aspect on that product.
  3. Teacher LLM generates a natural question for that persona/aspect.
  4. "chosen":
       - if enough evidence was retrieved: an answer generated WITH the
         evidence in context, constrained to it (grounded_aspect example).
       - if not: a deterministic, templated abstention response (no LLM
         call needed -- correctness of the safety-critical "chosen" side
         of an abstention pair shouldn't depend on generation quality).
  5. "rejected": the SAME question answered by the SAME model with NO
     evidence access at all. This is a principled way to get a realistic
     contrastive negative (naturally generic / prone to hallucination
     since the model has nothing to ground on) instead of instructing the
     model to "write a bad answer" on command.

Every example carries provenance (parent_asin, aspect, persona, category,
evidence review_ids) for later leakage/quality auditing -- required before
trusting any downstream SFT/DPO run on this data.

Usage:
  python generate_data.py --n 20 --out smoke_test_data.json      # smoke test
  python generate_data.py --n 800 --out synthetic_dpo_data.json  # real run
"""

import argparse
import json
import os
import random
import warnings
warnings.filterwarnings("ignore", message=".*Failed to load image Python extension.*")

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from librarian import Librarian

MODEL_ID = "NousResearch/Meta-Llama-3-8B-Instruct"
SPLIT_FILE = "dataset/sat/product_splits.json"

BATCH_SIZE = 8
MAX_NEW_TOKENS_Q = 40
MAX_NEW_TOKENS_ANS = 200

ASPECTS = [
    "battery life", "camera quality", "display quality", "performance and speed",
    "build quality and durability", "value for money", "ease of use", "overall product quality",
]

PERSONA_ASPECT_BIAS = {
    "Budget Student": ["value for money", "battery life", "build quality and durability"],
    "Professional Photographer": ["camera quality", "display quality"],
    "Pro Gamer": ["performance and speed", "display quality", "battery life"],
    "Frequent Traveler": ["battery life", "build quality and durability", "value for money"],
    "App Developer": ["performance and speed", "overall product quality"],
}
PERSONAS = list(PERSONA_ASPECT_BIAS.keys())

MIN_SCORE = 0.55
MIN_EVIDENCE_SENTENCES = 3


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
    tokenizer.padding_side = "left"  # correct for batched causal-LM generation

    dtype = safe_dtype()
    model_kwargs = dict(device_map="auto", trust_remote_code=True)
    if dtype is not None:
        model_kwargs["dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **model_kwargs)
    gen = pipeline("text-generation", model=model, tokenizer=tokenizer, batch_size=BATCH_SIZE)
    return gen


def chat_prompt(tokenizer, system, user):
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def pick_aspect(persona):
    if random.random() < 0.7 and persona in PERSONA_ASPECT_BIAS:
        return random.choice(PERSONA_ASPECT_BIAS[persona])
    return random.choice(ASPECTS)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="number of examples to generate")
    ap.add_argument("--out", type=str, default="synthetic_dpo_data.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="continue an interrupted run by loading --out if it exists")
    args = ap.parse_args()

    random.seed(args.seed)

    splits = json.load(open(SPLIT_FILE))
    train_asins = splits["train_asins"]
    print(f"Train products available: {len(train_asins)} (eval products excluded: {len(splits['eval_asins'])})")

    print("Loading Librarian (BLAIR retriever + product review store)...")
    lib = Librarian()

    gen = get_generator()

    dataset = []
    if args.resume and os.path.exists(args.out):
        with open(args.out) as f:
            dataset = json.load(f)
        print(f"Resuming from {args.out}: {len(dataset)} examples already generated.")

    pbar = tqdm(total=args.n, initial=len(dataset), desc="Generating")

    while len(dataset) < args.n:
        b = min(BATCH_SIZE, args.n - len(dataset))

        picks = []
        for _ in range(b):
            asin = random.choice(train_asins)
            persona = random.choice(PERSONAS)
            aspect = pick_aspect(persona)
            picks.append({"asin": asin, "persona": persona, "aspect": aspect})

        for p in picks:
            cur = lib._conn.cursor()
            cur.execute("SELECT product_title FROM reviews WHERE parent_asin=? LIMIT 1", (p["asin"],))
            row = cur.fetchone()
            p["product_title"] = row[0] if row else "this phone"

            retrieval = lib.retrieve_and_prune(p["asin"], p["aspect"], top_k=8, mode="hybrid", min_score=MIN_SCORE)
            p["evidence"] = [r["sentence"] for r in retrieval["results"]]
            p["evidence_review_ids"] = [r["review_id"] for r in retrieval["results"]]
            p["sufficient_evidence"] = len(p["evidence"]) >= MIN_EVIDENCE_SENTENCES

        # Stage 1: question generation
        q_prompts = [
            chat_prompt(
                gen.tokenizer,
                "You write realistic, specific shopping questions a customer would ask about a smartphone. "
                "Output ONLY the question, one sentence, no preamble or quotes.",
                f"Persona: {p['persona']}\nAspect to ask about: {p['aspect']}\n"
                f"Write one natural question this persona would ask about a smartphone's {p['aspect']}.",
            )
            for p in picks
        ]
        questions = gen_batch(gen, q_prompts, MAX_NEW_TOKENS_Q, temperature=0.9)
        for p, q in zip(picks, questions):
            p["question"] = q.strip().strip('"').strip()

        # Stage 2: grounded "chosen" (evidence-sufficient picks only)
        grounded_idx = [i for i, p in enumerate(picks) if p["sufficient_evidence"] and p["question"]]
        grounded_prompts = [
            chat_prompt(
                gen.tokenizer,
                "You are a helpful shopping assistant. Answer the user's question using ONLY the evidence "
                "below, drawn from real customer reviews. Do not invent facts not present in the evidence. "
                "Adapt tone to the persona. If the evidence is mixed or contradictory, say so explicitly. "
                "Keep the answer to 3-5 sentences.",
                f"Persona: {picks[i]['persona']}\nProduct: {picks[i]['product_title']}\n"
                f"Question: {picks[i]['question']}\n\nEvidence from customer reviews:\n"
                + "\n".join(f"- {s}" for s in picks[i]["evidence"]),
            )
            for i in grounded_idx
        ]
        grounded_answers = gen_batch(gen, grounded_prompts, MAX_NEW_TOKENS_ANS, temperature=0.3)
        for i, ans in zip(grounded_idx, grounded_answers):
            picks[i]["chosen"] = ans.strip()

        # Stage 3: ungrounded answer -- used as "rejected" for both categories
        ug_idx = [i for i, p in enumerate(picks) if p["question"]]
        ungrounded_prompts = [
            chat_prompt(
                gen.tokenizer,
                "You are a helpful shopping assistant. Answer the user's question as best you can. "
                "Adapt tone to the persona. Keep the answer to 3-5 sentences.",
                f"Persona: {picks[i]['persona']}\nProduct: {picks[i]['product_title']}\nQuestion: {picks[i]['question']}",
            )
            for i in ug_idx
        ]
        ungrounded_answers = gen_batch(gen, ungrounded_prompts, MAX_NEW_TOKENS_ANS, temperature=0.9)
        for i, ans in zip(ug_idx, ungrounded_answers):
            picks[i]["ungrounded_answer"] = ans.strip()

        # Assemble
        for p in picks:
            if not p.get("question") or not p.get("ungrounded_answer"):
                continue
            prompt_text = f"Persona: {p['persona']}\nUser: {p['question']}"
            base = {
                "prompt": prompt_text,
                "rejected": p["ungrounded_answer"],
                "parent_asin": p["asin"],
                "product_title": p["product_title"],
                "aspect": p["aspect"],
                "persona": p["persona"],
                "evidence_review_ids": p["evidence_review_ids"],
            }
            if p["sufficient_evidence"] and p.get("chosen"):
                base.update({"chosen": p["chosen"], "category": "grounded_aspect"})
            else:
                base.update({
                    "chosen": (
                        f"Based on the available customer reviews for {p['product_title']}, there isn't enough "
                        f"evidence about {p['aspect']} to give a confident answer. I don't want to guess -- "
                        f"would you like me to tell you what the reviews do cover?"
                    ),
                    "category": "abstention",
                })
            dataset.append(base)

        pbar.update(len(picks))

        # Checkpoint after every batch -- a killed/interrupted run (session
        # teardown, OOM, etc.) loses at most one batch instead of everything.
        with open(args.out, "w") as f:
            json.dump(dataset, f, indent=2)

    pbar.close()

    n_grounded = sum(1 for d in dataset if d["category"] == "grounded_aspect")
    n_abstain = sum(1 for d in dataset if d["category"] == "abstention")
    print(f"Saved {len(dataset)} examples to {args.out} ({n_grounded} grounded_aspect, {n_abstain} abstention)")


if __name__ == "__main__":
    main()
