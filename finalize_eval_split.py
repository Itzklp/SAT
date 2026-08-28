#!/usr/bin/env python3
"""
finalize_eval_split.py

Normalizes eval_dataset.json into a consistent schema, then splits into
a dev set (for calibrating thresholds -- e.g. the deferred retrieval-gate
fix in PROJECT_NOTES.md) and a FROZEN test set (touched only for final
reported numbers, never for tuning). Stratified by category, fixed seed.
"""

import json
import random
from collections import defaultdict, Counter

ANSWERABLE_CATEGORIES = {"aspect_specific", "persona_aware", "multi_aspect", "overall_suitability", "contradiction"}
UNANSWERABLE_CATEGORIES = {"unsupported_aspect", "unsupported_feature"}

DEV_FRACTION = 0.30
SEED = 99


def normalize(entries):
    out = []
    for i, e in enumerate(entries):
        cat = e["category"]
        if cat in ANSWERABLE_CATEGORIES:
            answerable = True
        elif cat in UNANSWERABLE_CATEGORIES:
            answerable = False
        else:  # ambiguous
            answerable = None

        out.append({
            "id": f"eval_{i:04d}",
            "category": cat,
            "parent_asin": e["parent_asin"],
            "product_title": e["product_title"],
            "aspect": e.get("aspect"),
            "persona": e.get("persona"),
            "query": e["query"],
            "gold_review_ids": e.get("gold_review_ids"),
            "gold_sentences": e.get("gold_sentences"),
            "contradiction": e.get("contradiction"),
            "n_pos": e.get("n_pos"),
            "n_neg": e.get("n_neg"),
            "n_neutral": e.get("n_neutral"),
            "answerable": answerable,
            "expected_action": "CLARIFY" if cat == "ambiguous" else "SEARCH",
        })
    return out


def stratified_split(entries, dev_fraction, seed):
    by_cat = defaultdict(list)
    for e in entries:
        by_cat[e["category"]].append(e)

    rnd = random.Random(seed)
    dev, test = [], []
    for cat, items in by_cat.items():
        rnd.shuffle(items)
        n_dev = round(len(items) * dev_fraction)
        dev.extend(items[:n_dev])
        test.extend(items[n_dev:])

    rnd.shuffle(dev)
    rnd.shuffle(test)
    return dev, test


def main():
    entries = json.load(open("eval_dataset.json"))
    normalized = normalize(entries)

    dev, test = stratified_split(normalized, DEV_FRACTION, SEED)

    print(f"Total: {len(normalized)}  Dev: {len(dev)}  Test: {len(test)}")
    print("Dev category counts:", Counter(e["category"] for e in dev))
    print("Test category counts:", Counter(e["category"] for e in test))

    json.dump(normalized, open("eval_dataset.json", "w"), indent=2)
    json.dump(dev, open("eval_dev.json", "w"), indent=2)
    json.dump(test, open("eval_test.json", "w"), indent=2)
    print("\nSaved eval_dataset.json (normalized, full), eval_dev.json, eval_test.json")
    print("eval_test.json is now FROZEN -- do not tune against it.")


if __name__ == "__main__":
    main()
