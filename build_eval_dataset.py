#!/usr/bin/env python3
"""
build_eval_dataset.py -- Pass 1: deterministic candidate selection.

Scans all eval-split products, applies eval_lexicon.py's keyword matching
(independent of the real Librarian's semantic scoring) to pick which
(product, aspect(s), persona, category) tuples are worth turning into
evaluation questions. No LLM calls in this pass -- fast, cheap, fully
deterministic and reproducible (given a fixed seed).

Output: eval_candidates.json -- reviewed for sane category counts before
Pass 2 (build_eval_questions.py) spends LLM compute phrasing them into
natural-language queries.
"""

import json
import random

from librarian import Librarian
from eval_lexicon import ASPECT_KEYWORDS, ABSENT_FEATURE_KEYWORDS, find_aspect_sentences, aspect_has_zero_matches, polarity_split

MIN_REVIEWS = 10
MIN_ASPECT_MATCHES = 3
MIN_CONTRADICTION_EACH_SIDE = 2
MIN_TOTAL_SENTENCES_FOR_OVERALL = 15

PERSONA_ASPECT_BIAS = {
    "Budget Student": ["value for money", "battery life", "build quality and durability"],
    "Professional Photographer": ["camera quality", "display quality"],
    "Pro Gamer": ["performance and speed", "display quality", "battery life"],
    "Frequent Traveler": ["battery life", "build quality and durability", "value for money"],
    "App Developer": ["performance and speed", "overall product quality"],
}

AMBIGUOUS_TEMPLATES = [
    "is it good?", "worth it?", "how is it?", "should I buy it?",
    "what do you think about it?", "any good?", "does it work well?", "is it worth the price?",
]


def main():
    splits = json.load(open("dataset/sat/product_splits.json"))
    eval_asins = splits["eval_asins"]
    print(f"Eval products available: {len(eval_asins)}")

    # No embeddings needed for this pass -- dummy encode_fn, never called.
    lib = Librarian(encode_fn=lambda texts: None)

    candidates = {
        "aspect_specific": [], "persona_aware": [], "multi_aspect": [],
        "overall_suitability": [], "contradiction": [],
        "unsupported_aspect": [], "unsupported_feature": [], "ambiguous": [],
    }

    random.seed(123)
    products_used = 0

    for asin in eval_asins:
        cur = lib._conn.cursor()
        cur.execute("SELECT COUNT(*) FROM reviews WHERE parent_asin=?", (asin,))
        n_reviews = cur.fetchone()[0]
        if n_reviews < MIN_REVIEWS:
            continue

        sentences = lib.get_product_sentences(asin)
        if len(sentences) < 5:
            continue
        products_used += 1

        cur.execute("SELECT product_title FROM reviews WHERE parent_asin=? LIMIT 1", (asin,))
        title = cur.fetchone()[0]

        per_aspect = {}
        for aspect in ASPECT_KEYWORDS:
            matched = find_aspect_sentences(sentences, aspect)
            pos, neg, neutral = polarity_split(matched)
            per_aspect[aspect] = {"matched": matched, "pos": pos, "neg": neg, "neutral": neutral}

        # -- aspect_specific + persona_aware --
        clear_aspects = []
        for aspect, d in per_aspect.items():
            if len(d["matched"]) < MIN_ASPECT_MATCHES:
                continue
            is_contradiction = len(d["pos"]) >= MIN_CONTRADICTION_EACH_SIDE and len(d["neg"]) >= MIN_CONTRADICTION_EACH_SIDE
            entry = {
                "parent_asin": asin, "product_title": title, "aspect": aspect,
                "gold_review_ids": [s["review_id"] for s in d["matched"]],
                "gold_sentences": [s["sentence"] for s in d["matched"]],
                "n_pos": len(d["pos"]), "n_neg": len(d["neg"]), "n_neutral": len(d["neutral"]),
            }
            if is_contradiction:
                candidates["contradiction"].append({**entry, "category": "contradiction", "contradiction": True})
            else:
                clear_aspects.append(aspect)
                candidates["aspect_specific"].append({**entry, "category": "aspect_specific", "contradiction": False})
                for persona, biased in PERSONA_ASPECT_BIAS.items():
                    if aspect in biased:
                        candidates["persona_aware"].append({**entry, "category": "persona_aware", "persona": persona, "contradiction": False})

        # -- multi_aspect --
        if len(clear_aspects) >= 2:
            a1, a2 = random.sample(clear_aspects, 2)
            d1, d2 = per_aspect[a1], per_aspect[a2]
            candidates["multi_aspect"].append({
                "parent_asin": asin, "product_title": title, "aspect": [a1, a2],
                "gold_review_ids": [s["review_id"] for s in d1["matched"] + d2["matched"]],
                "gold_sentences": [s["sentence"] for s in d1["matched"] + d2["matched"]],
                "category": "multi_aspect", "contradiction": False,
            })

        # -- overall_suitability --
        if len(sentences) >= MIN_TOTAL_SENTENCES_FOR_OVERALL:
            candidates["overall_suitability"].append({
                "parent_asin": asin, "product_title": title, "aspect": "overall product quality",
                "gold_review_ids": [s["review_id"] for s in sentences],
                "gold_sentences": None,  # too broad to enumerate; system's own retrieval evaluated instead
                "category": "overall_suitability", "contradiction": None,
            })

        # -- unsupported_aspect: a standard aspect genuinely never discussed --
        for aspect in ASPECT_KEYWORDS:
            if aspect_has_zero_matches(sentences, aspect):
                candidates["unsupported_aspect"].append({
                    "parent_asin": asin, "product_title": title, "aspect": aspect,
                    "gold_review_ids": [], "gold_sentences": [],
                    "category": "unsupported_aspect", "contradiction": False, "answerable": False,
                })

        # -- unsupported_feature: rare/out-of-domain feature, verified absent --
        for feature in ABSENT_FEATURE_KEYWORDS:
            if aspect_has_zero_matches(sentences, feature, keyword_map=ABSENT_FEATURE_KEYWORDS):
                candidates["unsupported_feature"].append({
                    "parent_asin": asin, "product_title": title, "aspect": feature,
                    "gold_review_ids": [], "gold_sentences": [],
                    "category": "unsupported_feature", "contradiction": False, "answerable": False,
                })

    # -- ambiguous: template-based, paired with random eval products --
    sample_asins = random.sample(eval_asins, min(len(eval_asins), 100))
    for asin in sample_asins:
        title = lib.catalog_title(asin) if hasattr(lib, "catalog_title") else None
        if title is None:
            cur = lib._conn.cursor()
            cur.execute("SELECT product_title FROM reviews WHERE parent_asin=? LIMIT 1", (asin,))
            row = cur.fetchone()
            title = row[0] if row else "this phone"
        candidates["ambiguous"].append({
            "parent_asin": asin, "product_title": title, "aspect": None,
            "query_template": random.choice(AMBIGUOUS_TEMPLATES),
            "category": "ambiguous", "contradiction": None, "answerable": None,
        })

    print(f"\nProducts with usable review data: {products_used}")
    print("Candidate counts by category:")
    for cat, items in candidates.items():
        print(f"  {cat:22s}: {len(items)}")

    json.dump(candidates, open("eval_candidates.json", "w"), indent=2)
    print("\nSaved eval_candidates.json")


if __name__ == "__main__":
    main()
