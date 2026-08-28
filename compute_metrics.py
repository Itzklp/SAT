#!/usr/bin/env python3
"""
compute_metrics.py

Aggregates eval_results_{A,B,E}.json (+ judge_results.json if present) into
a single metrics.json summary. Pure aggregation, no model calls -- fast,
re-runnable any time the underlying result files change.
"""

import json
import os
import statistics
from collections import defaultdict

from eval_metrics import (
    is_abstention, acknowledges_contradiction, brand_hallucination,
    fabricates_review_citation, recall_at_k, precision_at_k, extract_candidate_brands,
)

ANSWERABLE_CATEGORIES = {"aspect_specific", "persona_aware", "multi_aspect", "contradiction", "overall_suitability"}
UNANSWERABLE_CATEGORIES = {"unsupported_aspect", "unsupported_feature"}


def load(system):
    path = f"eval_results_{system}.json"
    if not os.path.exists(path):
        return None
    return json.load(open(path))


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def percentile(xs, p):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * p
    f, c = int(k), min(int(k) + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def product_title_for(record, eval_by_id):
    e = eval_by_id.get(record["id"])
    return e["product_title"] if e else ""


_review_text_cache = {}


def evidence_text_for(record):
    """Reconstruct the review text a B/E response was actually shown, so
    brand_hallucination() can tell a faithful evidence citation apart from
    a true fabrication. System A got no evidence -- always "" for it."""
    ids = record.get("retrieved_review_ids") or []
    if not ids:
        return ""
    missing = [i for i in ids if i not in _review_text_cache]
    if missing:
        import sqlite3
        conn = sqlite3.connect("dataset/sat/reviews.db")
        cur = conn.cursor()
        placeholders = ",".join("?" * len(missing))
        cur.execute(f"SELECT id, review_text, review_title FROM reviews WHERE id IN ({placeholders})", missing)
        for rid, text, title in cur.fetchall():
            _review_text_cache[rid] = text or title or ""
        conn.close()
    return " ".join(_review_text_cache.get(i, "") for i in ids)


def main():
    eval_test = json.load(open("eval_test.json"))
    eval_by_id = {e["id"]: e for e in eval_test}

    systems = {}
    for sysname in ["A", "B", "E"]:
        data = load(sysname)
        if data:
            systems[sysname] = data

    metrics = {"systems_present": list(systems.keys()), "n_test_total": len(eval_test)}

    # -----------------------------------------------------------------
    # 1. ABSTENTION / HALLUCINATION on unanswerable questions (the core,
    #    fully objective ground-truth-backed hallucination measurement)
    # -----------------------------------------------------------------
    metrics["abstention_on_unanswerable"] = {}
    for sysname, data in systems.items():
        unans = [r for r in data if r["category"] in UNANSWERABLE_CATEGORIES]
        if not unans:
            continue
        # A CLARIFY response is a distinct failure mode (tracked separately in
        # doorman_false_clarify) -- it isn't confidently fabricating an answer,
        # so it shouldn't inflate "hallucination_rate" alongside true confident
        # fabrications. Excluded from both the numerator and denominator here.
        answered = [r for r in unans if r.get("action_taken", "SEARCH") != "CLARIFY"]
        abstained = [r for r in answered if is_abstention(r["response"])]
        n_clarify = len(unans) - len(answered)
        metrics["abstention_on_unanswerable"][sysname] = {
            "n": len(unans), "n_excluded_as_clarify": n_clarify, "n_scored": len(answered),
            "abstention_rate": len(abstained) / len(answered) if answered else None,
            "hallucination_rate": 1 - len(abstained) / len(answered) if answered else None,
        }

    # -----------------------------------------------------------------
    # 2. Over-abstention (false refusal) on ANSWERABLE questions
    # -----------------------------------------------------------------
    metrics["over_abstention_on_answerable"] = {}
    for sysname, data in systems.items():
        ans = [r for r in data if r["category"] in ANSWERABLE_CATEGORIES]
        if not ans:
            continue
        wrongly_abstained = [r for r in ans if is_abstention(r["response"])]
        metrics["over_abstention_on_answerable"][sysname] = {
            "n": len(ans),
            "over_abstention_rate": len(wrongly_abstained) / len(ans),
        }

    # -----------------------------------------------------------------
    # 2b. Doorman false-CLARIFY rate + brand hallucination WITHIN the
    #     clarify_question itself (a distinct failure mode from
    #     answer-time hallucination -- found via manual inspection)
    # -----------------------------------------------------------------
    if "E" in systems:
        data = systems["E"]
        non_ambig = [r for r in data if r["category"] != "ambiguous"]
        clarify_misfires = [r for r in non_ambig if r.get("action_taken") == "CLARIFY"]
        clarify_with_hallucinated_brand = []
        for r in clarify_misfires:
            title = product_title_for(r, eval_by_id)
            mentioned = extract_candidate_brands(r["response"])
            if mentioned and not any(b.lower() in title.lower() for b in mentioned):
                clarify_with_hallucinated_brand.append(r)
        metrics["doorman_false_clarify"] = {
            "n_non_ambiguous": len(non_ambig),
            "false_clarify_rate": len(clarify_misfires) / len(non_ambig) if non_ambig else None,
            "n_false_clarify": len(clarify_misfires),
            "n_with_hallucinated_brand_in_question": len(clarify_with_hallucinated_brand),
        }

    # -----------------------------------------------------------------
    # 3. Fabricated review citation (System A specific -- zero evidence given)
    # -----------------------------------------------------------------
    if "A" in systems:
        ans = [r for r in systems["A"] if r["category"] in ANSWERABLE_CATEGORIES | UNANSWERABLE_CATEGORIES]
        fab = [r for r in ans if fabricates_review_citation(r["response"])]
        metrics["system_a_fabricated_review_citation_rate"] = {"n": len(ans), "rate": len(fab) / len(ans) if ans else None}

    # -----------------------------------------------------------------
    # 4. Brand/model hallucination (objective, all systems)
    # -----------------------------------------------------------------
    metrics["brand_hallucination_rate"] = {}
    for sysname, data in systems.items():
        checkable = [r for r in data if r["category"] != "ambiguous"]
        hall = []
        for r in checkable:
            title = product_title_for(r, eval_by_id)
            evidence_text = evidence_text_for(r)  # "" for System A -- it got no evidence
            if brand_hallucination(r["response"], title, evidence_text):
                hall.append(r)
        metrics["brand_hallucination_rate"][sysname] = {"n": len(checkable), "rate": len(hall) / len(checkable) if checkable else None}

    # -----------------------------------------------------------------
    # 5. Retrieval Recall@K / Precision@K (systems B & E, categories with gold_review_ids)
    # -----------------------------------------------------------------
    # overall_suitability's "gold" set is deliberately ALL of a product's
    # reviews (see build_eval_dataset.py) -- not a selective relevance
    # signal, so it's excluded here or Recall@K would be trivially ~1.0.
    RETRIEVAL_EVAL_CATEGORIES = {"aspect_specific", "persona_aware", "multi_aspect", "contradiction"}

    metrics["retrieval"] = {}
    for sysname in ["B", "E"]:
        if sysname not in systems:
            continue
        recalls, precisions = [], []
        for r in systems[sysname]:
            if r["category"] not in RETRIEVAL_EVAL_CATEGORIES:
                continue
            gold = r.get("gold_review_ids")
            if not gold:
                continue
            rec = recall_at_k(r.get("retrieved_review_ids", []), gold)
            prec = precision_at_k(r.get("retrieved_review_ids", []), gold)
            if rec is not None:
                recalls.append(rec)
            if prec is not None:
                precisions.append(prec)
        metrics["retrieval"][sysname] = {
            "n": len(recalls), "mean_recall": mean(recalls), "mean_precision": mean(precisions),
        }

    # -----------------------------------------------------------------
    # 6. Context reduction (pruning) -- system E vs B
    # -----------------------------------------------------------------
    metrics["context_reduction"] = {}
    for sysname in ["B", "E"]:
        if sysname not in systems:
            continue
        cands = [r.get("n_candidates") for r in systems[sysname] if r.get("n_candidates") is not None]
        kepts = [r.get("n_kept") for r in systems[sysname] if r.get("n_kept") is not None]
        if cands:
            avg_reduction = 1 - (sum(kepts) / sum(cands))
            metrics["context_reduction"][sysname] = {
                "n": len(cands), "avg_n_candidates": mean(cands), "avg_n_kept": mean(kepts),
                "avg_reduction_pct": avg_reduction * 100,
            }

    # -----------------------------------------------------------------
    # 7. Contradiction acknowledgment rate (contradiction category, systems B & E)
    # -----------------------------------------------------------------
    metrics["contradiction_acknowledgment"] = {}
    for sysname in ["B", "E"]:
        if sysname not in systems:
            continue
        contras = [r for r in systems[sysname] if r["category"] == "contradiction"]
        ack = [r for r in contras if acknowledges_contradiction(r["response"])]
        metrics["contradiction_acknowledgment"][sysname] = {"n": len(contras), "rate": len(ack) / len(contras) if contras else None}

    # -----------------------------------------------------------------
    # 8. CLARIFY precision/recall (System E only -- other systems have no clarify mechanism)
    # -----------------------------------------------------------------
    if "E" in systems:
        data = systems["E"]
        ambiguous = [r for r in data if r["category"] == "ambiguous"]
        non_ambiguous = [r for r in data if r["category"] != "ambiguous"]
        clarify_recall = mean([1.0 if r.get("action_taken") == "CLARIFY" else 0.0 for r in ambiguous])
        false_clarify = [r for r in non_ambiguous if r.get("action_taken") == "CLARIFY"]
        metrics["clarify_behavior"] = {
            "n_ambiguous": len(ambiguous), "clarify_recall": clarify_recall,
            "n_non_ambiguous": len(non_ambiguous), "false_clarify_rate": len(false_clarify) / len(non_ambiguous) if non_ambiguous else None,
        }

    # -----------------------------------------------------------------
    # 9. Efficiency: latency, tokens
    # -----------------------------------------------------------------
    metrics["efficiency"] = {}
    for sysname, data in systems.items():
        lat = [r.get("latency") for r in data]
        in_tok = [r.get("input_tokens") for r in data]
        out_tok = [r.get("output_tokens") for r in data]
        metrics["efficiency"][sysname] = {
            "p50_latency_s": percentile(lat, 0.5), "p95_latency_s": percentile(lat, 0.95),
            "mean_input_tokens": mean(in_tok), "mean_output_tokens": mean(out_tok),
        }

    # -----------------------------------------------------------------
    # 10. LLM-judge grounding (if judge_results.json exists)
    # -----------------------------------------------------------------
    if os.path.exists("judge_results.json"):
        judged = json.load(open("judge_results.json"))
        by_system = defaultdict(list)
        for j in judged:
            by_system[j["system"]].append(j)
        metrics["judge_grounding"] = {}
        for sysname, items in by_system.items():
            valid = [j for j in items if j["hallucinated"] is not None]
            hall = [j for j in valid if j["hallucinated"]]
            metrics["judge_grounding"][sysname] = {
                "n": len(valid), "hallucination_rate": len(hall) / len(valid) if valid else None,
                "n_judge_parse_failures": len(items) - len(valid),
            }

    json.dump(metrics, open("metrics.json", "w"), indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
