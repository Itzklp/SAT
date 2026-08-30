#!/usr/bin/env python3
"""
judge_grounding.py

LLM-judge grounding/hallucination check on a stratified sample of System B
and System E responses (System A's evidence-citation-fabrication check is
already fully objective -- see eval_metrics.fabricates_review_citation --
no judge needed there since we know with certainty it was given zero
evidence).

Methodology limitation, disclosed rather than hidden: the judge is the SAME
base model (adapter disabled) used to generate the responses, not an
independent stronger model or a human. This is a real limitation -- a judge
grading outputs from its own weight family is not as trustworthy as human
annotation or a different model family -- and results should be read as a
secondary, exploratory signal alongside the fully-objective abstention and
citation-fabrication metrics, not as ground truth.
"""

import argparse
import json
import random
import re
import warnings
warnings.filterwarnings("ignore", message=".*Failed to load image Python extension.*")

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL_ID = "NousResearch/Meta-Llama-3-8B-Instruct"
ADAPTER_PATH = "./rufus_checkpoints/final_dpo_adapter"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ANSWERABLE_CATEGORIES = ["aspect_specific", "persona_aware", "multi_aspect", "contradiction", "overall_suitability"]
SAMPLE_PER_CATEGORY_PER_SYSTEM = 20
SEED = 55


def load_judge():
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=True, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, quantization_config=bnb, device_map="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(base, ADAPTER_PATH, is_trainable=False)
    model.eval()
    return model, tok


def judge_one(model, tok, evidence_text, response_text, query):
    sys_prompt = (
        "You are a strict fact-checker. Given EVIDENCE (customer reviews) and a RESPONSE that was supposed "
        "to be grounded in that evidence, identify whether the response makes any specific factual claim "
        "that is NOT supported by the evidence. Vague hedging (\"may vary\", \"it seems\") is fine and not "
        "a hallucination. A specific claim about a spec, feature, or review content that isn't in the "
        "evidence IS a hallucination. Output ONLY JSON: "
        '{"hallucinated": true|false, "unsupported_claims": ["..."], "reasoning": "one sentence"}'
    )
    user_prompt = f"QUERY: {query}\n\nEVIDENCE:\n{evidence_text or '(none provided)'}\n\nRESPONSE:\n{response_text}"
    with model.disable_adapter():
        messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to(DEVICE)
        with torch.no_grad():
            out = model.generate(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                                  max_new_tokens=150, do_sample=False, pad_token_id=tok.eos_token_id)
        raw = tok.decode(out[0, enc["input_ids"].shape[-1]:], skip_special_tokens=True).strip()

    try:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        parsed = json.loads(m.group(0)) if m else {}
    except Exception:
        parsed = {}
    return {
        "hallucinated": parsed.get("hallucinated"),
        "unsupported_claims": parsed.get("unsupported_claims", []),
        "reasoning": parsed.get("reasoning", ""),
        "raw_judge_output": raw,
    }


def format_evidence_from_record(record):
    # For System B/E records we stored retrieved_review_ids but not the sentence text itself
    # in eval_results_*.json (to keep file size down) -- reconstruct via Librarian.
    from librarian import Librarian
    lib = Librarian(encode_fn=lambda texts: None)
    cur = lib._conn.cursor()
    if not record.get("retrieved_review_ids"):
        return ""
    placeholders = ",".join("?" * len(record["retrieved_review_ids"]))
    cur.execute(f"SELECT id, review_text, review_title FROM reviews WHERE id IN ({placeholders})", record["retrieved_review_ids"])
    rows = {r[0]: (r[1] or r[2] or "") for r in cur.fetchall()}
    return "\n".join(f"- {rows.get(rid, '')}" for rid in record["retrieved_review_ids"] if rows.get(rid))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="judge_results.json")
    args = ap.parse_args()

    random.seed(SEED)
    results_b = json.load(open("eval_results_B.json"))
    results_e = json.load(open("eval_results_E.json"))

    sample = []
    for system_name, results in [("B", results_b), ("E", results_e)]:
        by_cat = {}
        for r in results:
            if r["category"] in ANSWERABLE_CATEGORIES:
                by_cat.setdefault(r["category"], []).append(r)
        for cat, items in by_cat.items():
            # For system E, skip CLARIFY / templated-abstention responses (nothing to judge against evidence)
            items = [it for it in items if it.get("action_taken", "SEARCH") == "SEARCH" and it.get("sufficient_evidence", True)]
            random.shuffle(items)
            sample.extend(items[:SAMPLE_PER_CATEGORY_PER_SYSTEM])

    print(f"Judging {len(sample)} responses ({SAMPLE_PER_CATEGORY_PER_SYSTEM} per category per system, systems B & E)...")

    model, tok = load_judge()

    done = json.load(open(args.out)) if __import__("os").path.exists(args.out) else []
    done_keys = {(d["system"], d["id"]) for d in done}

    for record in tqdm(sample, desc="Judging"):
        key = (record["system"], record["id"])
        if key in done_keys:
            continue
        evidence_text = format_evidence_from_record(record)
        verdict = judge_one(model, tok, evidence_text, record["response"], record["query"])
        done.append({"system": record["system"], "id": record["id"], "category": record["category"], **verdict})
        json.dump(done, open(args.out, "w"), indent=2)

    print(f"Done. {len(done)} judgments saved to {args.out}")


if __name__ == "__main__":
    main()
