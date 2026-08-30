#!/usr/bin/env python3
"""
sample_eval_candidates.py -- Pass 1.5: sample a balanced, diverse final set
from eval_candidates.json before spending LLM compute phrasing questions.

Quotas (total ~720, within the report's 500-1000 range):
"""

import json
import random
from collections import defaultdict

QUOTAS = {
    "aspect_specific": 150,
    "persona_aware": 150,
    "multi_aspect": 80,
    "overall_suitability": 80,
    "contradiction": 100,
    "unsupported_aspect": 70,
    "unsupported_feature": 50,
    "ambiguous": 40,
}


def diverse_sample(items, n, key_fn, seed):
    """Sample n items trying to spread across distinct key_fn(item) values
    (e.g. parent_asin, or (aspect,persona)) before allowing repeats."""
    rnd = random.Random(seed)
    by_key = defaultdict(list)
    for it in items:
        by_key[key_fn(it)].append(it)
    for bucket in by_key.values():
        rnd.shuffle(bucket)

    keys = list(by_key.keys())
    rnd.shuffle(keys)
    out = []
    idx = 0
    while len(out) < n and any(by_key[k] for k in keys):
        k = keys[idx % len(keys)]
        if by_key[k]:
            out.append(by_key[k].pop())
        idx += 1
        if idx > n * 50:  # safety valve
            break
    return out[:n]


def main():
    candidates = json.load(open("eval_candidates.json"))
    selected = {}

    selected["aspect_specific"] = diverse_sample(
        candidates["aspect_specific"], QUOTAS["aspect_specific"],
        key_fn=lambda it: (it["parent_asin"], it["aspect"]), seed=1,
    )
    selected["persona_aware"] = diverse_sample(
        candidates["persona_aware"], QUOTAS["persona_aware"],
        key_fn=lambda it: it["persona"], seed=2,  # spread across the 5 personas first
    )
    selected["multi_aspect"] = diverse_sample(
        candidates["multi_aspect"], QUOTAS["multi_aspect"],
        key_fn=lambda it: it["parent_asin"], seed=3,
    )
    selected["overall_suitability"] = diverse_sample(
        candidates["overall_suitability"], QUOTAS["overall_suitability"],
        key_fn=lambda it: it["parent_asin"], seed=4,
    )
    selected["contradiction"] = diverse_sample(
        candidates["contradiction"], QUOTAS["contradiction"],
        key_fn=lambda it: (it["parent_asin"], it["aspect"]), seed=5,
    )
    selected["unsupported_aspect"] = diverse_sample(
        candidates["unsupported_aspect"], QUOTAS["unsupported_aspect"],
        key_fn=lambda it: it["aspect"], seed=6,  # spread across the 7 aspects first
    )
    selected["unsupported_feature"] = diverse_sample(
        candidates["unsupported_feature"], QUOTAS["unsupported_feature"],
        key_fn=lambda it: it["aspect"], seed=7,  # spread across the 6 features first
    )
    selected["ambiguous"] = diverse_sample(
        candidates["ambiguous"], QUOTAS["ambiguous"],
        key_fn=lambda it: it["parent_asin"], seed=8,
    )

    total = sum(len(v) for v in selected.values())
    print("Selected counts:")
    for cat, items in selected.items():
        print(f"  {cat:22s}: {len(items)} (quota {QUOTAS[cat]})")
    print(f"TOTAL: {total}")

    all_asins = set()
    for items in selected.values():
        for it in items:
            all_asins.add(it["parent_asin"])
    print(f"Distinct products touched: {len(all_asins)}")

    if selected["persona_aware"]:
        from collections import Counter
        print("Persona distribution:", Counter(it["persona"] for it in selected["persona_aware"]))
    if selected["unsupported_aspect"]:
        from collections import Counter
        print("Unsupported-aspect distribution:", Counter(it["aspect"] for it in selected["unsupported_aspect"]))
    if selected["unsupported_feature"]:
        from collections import Counter
        print("Unsupported-feature distribution:", Counter(it["aspect"] for it in selected["unsupported_feature"]))

    json.dump(selected, open("eval_selected.json", "w"), indent=2)
    print("\nSaved eval_selected.json")


if __name__ == "__main__":
    main()
