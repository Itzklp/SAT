#!/usr/bin/env python3
"""
run_evaluation.py

Generates responses from three systems (A: LLM-only, B: vanilla RAG,
E: full SAT quad-layer pipeline) against eval_test.json, all using the SAME
underlying DPO-tuned model weights so the comparison isolates the
architecture's contribution rather than confounding it with model tuning.

Checkpoints per-system after every batch. Run per-system via --system A/B/E
so a crash in one doesn't lose the others; --limit for smoke testing.
"""

import argparse
import json
import os
import re
import time
import warnings
warnings.filterwarnings("ignore", message=".*Failed to load image Python extension.*")

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, BitsAndBytesConfig, pipeline
from peft import PeftModel

from librarian import Librarian
from eval_metrics import is_abstention

BASE_MODEL_ID = "NousResearch/Meta-Llama-3-8B-Instruct"
RETRIEVER_ID = "hyp1231/blair-roberta-large"
ADAPTER_PATH = "./rufus_checkpoints/final_dpo_adapter"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8

MIN_SCORE = 0.55
MIN_EVIDENCE_SENTENCES = 3
TOP_K = 8

CANONICAL_ASPECTS = [
    "battery life", "camera quality", "display quality", "performance and speed",
    "build quality and durability", "value for money", "ease of use", "overall product quality",
]
_ASPECT_FALLBACK_KEYWORDS = {
    "battery life": ["batt"], "camera quality": ["camera", "photo", "picture", "video", "lens"],
    "display quality": ["screen", "display", "resolution", "brightness"],
    "performance and speed": ["performance", "speed", "fast", "slow", "lag", "processor", "cpu", "gaming", "game"],
    "build quality and durability": ["durab", "sturdy", "build quality", "crack", "scratch", "drop", "waterproof"],
    "value for money": ["price", "value", "worth", "money", "cheap", "expensive"],
    "ease of use": ["interface", "usab", "intuitive", "user friendly", "setup", "navigat"],
}


def normalize_aspect(raw):
    if raw in CANONICAL_ASPECTS:
        return raw
    low = (raw or "").lower()
    for canonical, kws in _ASPECT_FALLBACK_KEYWORDS.items():
        if any(kw in low for kw in kws):
            return canonical
    return "overall product quality"


def load_generator_and_tokenizer():
    print(f"Loading tokenizer + BLAIR ({RETRIEVER_ID}) ...")
    ret_tok = AutoTokenizer.from_pretrained(RETRIEVER_ID, use_fast=True, trust_remote_code=True)
    ret_model = AutoModel.from_pretrained(RETRIEVER_ID, trust_remote_code=True).to(DEVICE).eval()

    def encode_blair(texts, batch_size=32):
        import numpy as np
        if not texts:
            return np.zeros((0, ret_model.config.hidden_size), dtype=np.float32)
        outs = []
        for i in range(0, len(texts), batch_size):
            b = texts[i:i + batch_size]
            inputs = ret_tok(b, padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                v = ret_model(**inputs).last_hidden_state[:, 0]
                v = torch.nn.functional.normalize(v, p=2, dim=1)
            outs.append(v.cpu().numpy())
        return np.vstack(outs).astype(np.float32)

    print(f"Loading base LLM ({BASE_MODEL_ID}) + DPO adapter ...")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=True, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, quantization_config=bnb, device_map="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(base, ADAPTER_PATH, is_trainable=False)
    model.eval()
    gen = pipeline("text-generation", model=model, tokenizer=tok, batch_size=BATCH_SIZE)
    return gen, encode_blair


def chat_prompt(tok, system, user):
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def gen_batch(gen, prompts, max_new_tokens, temperature=0.0):
    if not prompts:
        return [], 0.0
    t0 = time.time()
    outs = gen(prompts, max_new_tokens=max_new_tokens, temperature=temperature, do_sample=temperature > 0,
               return_full_text=False, pad_token_id=gen.tokenizer.eos_token_id)
    dt = time.time() - t0
    results = []
    for o in outs:
        if isinstance(o, list):
            o = o[0]
        results.append(o.get("generated_text", "").strip())
    return results, dt


def token_count(tok, text):
    return len(tok(text, truncation=True, max_length=4096)["input_ids"])


# ---------------------------------------------------------------------
# SYSTEM A: LLM-only
# ---------------------------------------------------------------------
def run_system_a(gen, items, out_path):
    done = json.load(open(out_path)) if os.path.exists(out_path) else []
    start = len(done)
    for i in tqdm(range(start, len(items), BATCH_SIZE), desc="System A (LLM-only)"):
        batch = items[i:i + BATCH_SIZE]
        prompts = [
            chat_prompt(gen.tokenizer,
                "You are a helpful shopping assistant. Answer the user's question as best you can.",
                f"Persona: {e.get('persona') or 'General Shopper'}\nProduct: {e['product_title']}\nQuestion: {e['query']}")
            for e in batch
        ]
        responses, dt = gen_batch(gen, prompts, max_new_tokens=220, temperature=0.0)
        per_item_latency = dt / max(len(batch), 1)
        for e, resp in zip(batch, responses):
            done.append({
                "id": e["id"], "category": e["category"], "system": "A", "parent_asin": e["parent_asin"],
                "query": e["query"], "response": resp, "latency": per_item_latency,
                "input_tokens": token_count(gen.tokenizer, prompts[0]), "output_tokens": token_count(gen.tokenizer, resp),
                "retrieved_review_ids": [], "gold_review_ids": e.get("gold_review_ids"),
                "answerable": e.get("answerable"), "expected_action": e.get("expected_action"),
                "action_taken": "SEARCH",
            })
        json.dump(done, open(out_path, "w"), indent=2)
    print(f"System A done: {len(done)} responses -> {out_path}")


# ---------------------------------------------------------------------
# SYSTEM B: vanilla RAG (unfiltered dense retrieval, single-shot, no Doorman/Analyst/abstention)
# ---------------------------------------------------------------------
def run_system_b(gen, lib, items, out_path):
    done = json.load(open(out_path)) if os.path.exists(out_path) else []
    start = len(done)
    for i in tqdm(range(start, len(items), BATCH_SIZE), desc="System B (vanilla RAG)"):
        batch = items[i:i + BATCH_SIZE]
        retrievals = []
        for e in batch:
            r = lib.retrieve_and_prune(e["parent_asin"], e["query"], top_k=TOP_K, mode="blair", min_score=None)
            retrievals.append(r)
        prompts = []
        for e, r in zip(batch, retrievals):
            context = "\n".join(f"- {x['sentence']}" for x in r["results"]) or "(no reviews retrieved)"
            prompts.append(chat_prompt(gen.tokenizer,
                "You are a helpful shopping assistant. Answer the user's question using the customer "
                "reviews provided below.",
                f"Persona: {e.get('persona') or 'General Shopper'}\nProduct: {e['product_title']}\n"
                f"Question: {e['query']}\n\nCustomer reviews:\n{context}"))
        responses, dt = gen_batch(gen, prompts, max_new_tokens=220, temperature=0.0)
        per_item_latency = dt / max(len(batch), 1)
        for e, r, resp, prompt in zip(batch, retrievals, responses, prompts):
            done.append({
                "id": e["id"], "category": e["category"], "system": "B", "parent_asin": e["parent_asin"],
                "query": e["query"], "response": resp, "latency": per_item_latency,
                "input_tokens": token_count(gen.tokenizer, prompt), "output_tokens": token_count(gen.tokenizer, resp),
                "retrieved_review_ids": [x["review_id"] for x in r["results"]], "gold_review_ids": e.get("gold_review_ids"),
                "answerable": e.get("answerable"), "expected_action": e.get("expected_action"),
                "action_taken": "SEARCH", "n_candidates": r["stats"]["n_candidates"], "n_kept": r["stats"]["n_kept"],
            })
        json.dump(done, open(out_path, "w"), indent=2)
    print(f"System B done: {len(done)} responses -> {out_path}")


# ---------------------------------------------------------------------
# SYSTEM E: full SAT pipeline (Doorman -> Librarian -> Analyst -> Spokesperson)
# ---------------------------------------------------------------------
def doorman_parse_batch(gen, items):
    sys_prompt = (
        "You are a strict shopping-query router. OUTPUT ONLY valid JSON matching this schema:\n"
        '{"action": "SEARCH"|"CLARIFY", "aspects": ["..."], "persona": "..."|null, "clarify_question": "..."}\n'
        f"- aspects: choose ONLY from this exact list (pick all that the query touches on): {CANONICAL_ASPECTS}. "
        "Do not invent new aspect names or synonyms.\n"
        "- persona: if the query itself signals a persona, name it; otherwise null.\n"
        "- If the query is too vague to search (e.g. \"is it good?\"), set action=CLARIFY, and write an ACTUAL "
        "clarifying question -- never repeat the user's own query back as the clarify_question.\n"
        "Do not add commentary."
    )
    prompts = [
        chat_prompt(gen.tokenizer, sys_prompt, f"Session persona (may be overridden by the query): {e.get('persona') or 'General Shopper'}\nQuery: {e['query']}")
        for e in items
    ]
    responses, dt = gen_batch(gen, prompts, max_new_tokens=100, temperature=0.0)
    parsed_list = []
    for resp in responses:
        try:
            m = re.search(r"\{.*\}", resp, re.DOTALL)
            parsed = json.loads(m.group(0)) if m else {}
        except Exception:
            parsed = {}
        action = parsed.get("action", "SEARCH")
        raw_aspects = parsed.get("aspects") or ["overall product quality"]
        aspects = list(dict.fromkeys(normalize_aspect(a) for a in raw_aspects))
        persona = parsed.get("persona")
        clarify_q = parsed.get("clarify_question") or "Which matters most to you: price, camera, battery, or performance?"
        parsed_list.append({"action": action, "aspects": aspects, "persona": persona, "clarify_question": clarify_q})
    return parsed_list, dt


def analyst_organize(retrieval, aspect_label):
    results = retrieval["results"]
    positive = [r for r in results if r.get("rating") is not None and r["rating"] >= 4]
    negative = [r for r in results if r.get("rating") is not None and r["rating"] <= 2]
    neutral = [r for r in results if r not in positive and r not in negative]
    return {
        "aspect": aspect_label, "positive": positive, "negative": negative, "neutral": neutral,
        "contradiction": len(positive) > 0 and len(negative) > 0,
        "sufficient_evidence": len(results) >= MIN_EVIDENCE_SENTENCES,
        "stats": retrieval["stats"],
    }


def format_evidence(evidence):
    lines = []
    for label, bucket in [("Positive", evidence["positive"]), ("Negative", evidence["negative"]), ("Mixed/neutral", evidence["neutral"])]:
        for r in bucket:
            lines.append(f"- [{label}, {r['rating']}★] {r['sentence']}")
    return "\n".join(lines)


def run_system_e(gen, lib, items, out_path):
    done = json.load(open(out_path)) if os.path.exists(out_path) else []
    done_ids = {d["id"] for d in done}
    remaining = [e for e in items if e["id"] not in done_ids]
    if not remaining:
        print(f"System E already complete: {len(done)} -> {out_path}")
        return

    for i in tqdm(range(0, len(remaining), BATCH_SIZE), desc="System E (full SAT) - Doorman+Librarian+Analyst"):
        batch = remaining[i:i + BATCH_SIZE]
        parsed_list, doorman_dt = doorman_parse_batch(gen, batch)
        doorman_latency = doorman_dt / max(len(batch), 1)

        batch_results = []
        needs_generation = []
        for e, parsed in zip(batch, parsed_list):
            if parsed["action"] == "CLARIFY":
                batch_results.append({
                    "id": e["id"], "category": e["category"], "system": "E", "parent_asin": e["parent_asin"],
                    "query": e["query"], "response": parsed["clarify_question"], "latency": doorman_latency,
                    "input_tokens": None, "output_tokens": None,
                    "retrieved_review_ids": [], "gold_review_ids": e.get("gold_review_ids"),
                    "answerable": e.get("answerable"), "expected_action": e.get("expected_action"),
                    "action_taken": "CLARIFY", "aspects_detected": parsed["aspects"], "persona_detected": parsed["persona"],
                })
                continue

            aspect_label = ", ".join(parsed["aspects"])
            persona = parsed["persona"] or e.get("persona") or "General Shopper"
            retrieval = lib.retrieve_and_prune(e["parent_asin"], e["query"], top_k=TOP_K, mode="hybrid", min_score=MIN_SCORE)
            evidence = analyst_organize(retrieval, aspect_label)

            base_record = {
                "id": e["id"], "category": e["category"], "system": "E", "parent_asin": e["parent_asin"],
                "query": e["query"], "retrieved_review_ids": [r["review_id"] for r in retrieval["results"]],
                "gold_review_ids": e.get("gold_review_ids"), "answerable": e.get("answerable"),
                "expected_action": e.get("expected_action"), "action_taken": "SEARCH",
                "aspects_detected": parsed["aspects"], "persona_detected": persona,
                "n_candidates": retrieval["stats"]["n_candidates"], "n_kept": retrieval["stats"]["n_kept"],
                "contradiction_flagged": evidence["contradiction"], "sufficient_evidence": evidence["sufficient_evidence"],
            }

            if not evidence["sufficient_evidence"]:
                resp = (f"Based on the available customer reviews for {e['product_title']}, there isn't enough "
                        f"evidence about {aspect_label} to give a confident answer. I don't want to guess -- "
                        f"would you like me to tell you what the reviews do cover?")
                base_record.update({"response": resp, "latency": doorman_latency, "input_tokens": None, "output_tokens": None})
                batch_results.append(base_record)
            else:
                needs_generation.append((base_record, evidence, persona, e))

        if needs_generation:
            prompts = []
            for base_record, evidence, persona, e in needs_generation:
                prompts.append(chat_prompt(gen.tokenizer,
                    "You are a helpful shopping assistant. Answer the user's question using ONLY the evidence "
                    "below, drawn from real customer reviews. Do not invent facts not present in the evidence. "
                    "Adapt tone to the persona. If the evidence is mixed or contradictory, say so explicitly. "
                    "Keep the answer to 3-5 sentences.",
                    f"Persona: {persona}\nProduct: {e['product_title']}\nQuestion: {e['query']}\n\n"
                    f"Evidence from customer reviews:\n{format_evidence(evidence)}"))
            responses, gen_dt = gen_batch(gen, prompts, max_new_tokens=220, temperature=0.0)
            per_item_latency = doorman_latency + gen_dt / max(len(needs_generation), 1)
            for (base_record, evidence, persona, e), resp, prompt in zip(needs_generation, responses, prompts):
                base_record.update({
                    "response": resp, "latency": per_item_latency,
                    "input_tokens": token_count(gen.tokenizer, prompt), "output_tokens": token_count(gen.tokenizer, resp),
                })
                batch_results.append(base_record)

        done.extend(batch_results)
        json.dump(done, open(out_path, "w"), indent=2)

    print(f"System E done: {len(done)} responses -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=["A", "B", "E"], required=True)
    ap.add_argument("--eval_file", type=str, default="eval_test.json")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    items = json.load(open(args.eval_file))
    if args.limit:
        items = items[:args.limit]
    print(f"Loaded {len(items)} eval items from {args.eval_file}")

    gen, encode_blair = load_generator_and_tokenizer()

    out_path = f"eval_results_{args.system}.json"
    if args.system == "A":
        run_system_a(gen, items, out_path)
    else:
        lib = Librarian(encode_fn=encode_blair)
        if args.system == "B":
            run_system_b(gen, lib, items, out_path)
        else:
            run_system_e(gen, lib, items, out_path)


if __name__ == "__main__":
    main()
