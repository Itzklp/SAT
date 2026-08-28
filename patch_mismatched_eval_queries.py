#!/usr/bin/env python3
"""Regenerates only the eval_dataset.json queries that hallucinated a
specific brand/model name unrelated to the actual gold product (see
PROJECT_NOTES.md) -- a targeted patch, not a full regeneration."""
import json
import re
import shutil

from build_eval_questions import get_generator, gen_batch, build_prompt

KNOWN_BRANDS = ['Samsung', 'Galaxy', 'iPhone', 'Apple', 'OnePlus', 'Google', 'Pixel', 'Xiaomi', 'Motorola', 'Moto',
                'Nokia', 'Sony', 'Xperia', 'LG', 'Oppo', 'Realme', 'Poco', 'Redmi', 'Asus', 'Honor', 'Huawei', 'BLU']


def mentioned_brands(query):
    return [b for b in KNOWN_BRANDS if re.search(rf'\b{b}\b', query)]


def is_mismatched(e):
    mentioned = mentioned_brands(e["query"])
    if not mentioned:
        return False
    title_low = e["product_title"].lower()
    return not any(b.lower() in title_low for b in mentioned)


def main():
    path = "eval_dataset.json"
    shutil.copy(path, path + ".pre_brand_fix.bak")
    d = json.load(open(path))

    mismatched_idx = [i for i, e in enumerate(d) if is_mismatched(e)]
    print(f"Found {len(mismatched_idx)} mismatched entries to regenerate.")

    gen = get_generator()
    BATCH = 8
    for start in range(0, len(mismatched_idx), BATCH):
        idxs = mismatched_idx[start:start + BATCH]
        items = [d[i] for i in idxs]
        prompts = [build_prompt(gen.tokenizer, it, it["category"]) for it in items]
        questions = gen_batch(gen, prompts)
        for i, q in zip(idxs, questions):
            d[i]["query"] = q
        with open(path, "w") as f:
            json.dump(d, f, indent=2)

    still_bad = [i for i in mismatched_idx if is_mismatched(d[i])]
    print(f"Remaining mismatches after patch: {len(still_bad)}")
    for i in still_bad[:10]:
        print("  -", d[i]["category"], repr(d[i]["query"][:100]))

    print(f"Done. Patched {len(mismatched_idx)} entries in {path} (backup at {path}.pre_brand_fix.bak)")


if __name__ == "__main__":
    main()
