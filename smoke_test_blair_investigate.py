#!/usr/bin/env python3
"""
smoke_test_blair_investigate.py

Follow-up investigation into why BLAIR-dense underperformed BM25 in
smoke_test_retrieval_compare.py. Tests two hypotheses cheaply on the same
two real products before deciding how to build the production Librarian:

  H1: Sentence-level indexing (not whole-review) helps, since a relevant
      sentence buried in an otherwise off-topic review gets diluted when
      the whole review is embedded as one CLS vector.
  H2: Mean-pooling over token embeddings discriminates better than the
      documented CLS-pooling for this fine-grained aspect-matching task.
"""

import csv
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import spacy
from transformers import AutoTokenizer, AutoModel
from rank_bm25 import BM25Okapi

CSV_PATH = "dataset/sat/phone_reviews.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEST_CASES = [
    ("B00M78E4MS", "battery life", ["batt"]),
    ("B08MXXC8TX", "camera quality", ["camera", "cámara", "selfie"]),
]

tok = AutoTokenizer.from_pretrained("hyp1231/blair-roberta-large")
model = AutoModel.from_pretrained("hyp1231/blair-roberta-large").to(DEVICE).eval()
nlp = spacy.load("en_core_web_sm")


def encode(texts, pooling="cls", bs=16):
    outs = []
    for i in range(0, len(texts), bs):
        b = texts[i:i + bs]
        inp = tok(b, padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            hidden = model(**inp).last_hidden_state  # (B, T, H)
            if pooling == "cls":
                v = hidden[:, 0]
            elif pooling == "mean":
                mask = inp["attention_mask"].unsqueeze(-1).float()
                v = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            else:
                raise ValueError(pooling)
            v = torch.nn.functional.normalize(v, p=2, dim=1)
        outs.append(v.cpu().numpy())
    return np.vstack(outs).astype(np.float32)


def load_reviews(asin):
    out = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["parent_asin"] == asin:
                t = (row["review_text"] or "").strip()
                ti = (row["review_title"] or "").strip()
                full = f"{ti}. {t}".strip(". ").strip()
                if full:
                    out.append(full)
    return out


def to_sentences(reviews):
    sents = []
    for r in reviews:
        doc = nlp(r)
        for s in doc.sents:
            s = s.text.strip()
            if len(s) > 3:
                sents.append(s)
    return sents


def relevant(text, keywords):
    t = text.lower()
    return any(k in t for k in keywords)


def report(label, ranked, texts, scores, keywords):
    hits = sum(1 for i in ranked if relevant(texts[i], keywords))
    print(f"  {label}: {hits}/{len(ranked)} relevant in top-{len(ranked)}")
    for i in ranked[:5]:
        mark = "✓" if relevant(texts[i], keywords) else " "
        print(f"    [{mark}] ({scores[i]:.3f}) {texts[i][:110]!r}")


def main():
    for asin, query, keywords in TEST_CASES:
        print("\n" + "=" * 90)
        reviews = load_reviews(asin)
        sentences = to_sentences(reviews)
        print(f"ASIN={asin} QUERY={query!r}  reviews={len(reviews)} sentences={len(sentences)}")

        # --- Baseline: BM25 at sentence level (for reference) ---
        tokenized = [s.lower().split() for s in sentences]
        bm25 = BM25Okapi(tokenized)
        bm25_scores = bm25.get_scores(query.lower().split())
        bm25_ranked = list(np.argsort(-bm25_scores)[:5])
        report("BM25 @ sentence-level", bm25_ranked, sentences, bm25_scores, keywords)

        # --- H1: BLAIR CLS, sentence-level indexing ---
        embs = encode(sentences, pooling="cls")
        q = encode([query], pooling="cls")
        scores = (embs @ q.T).flatten()
        ranked = list(np.argsort(-scores)[:5])
        report("H1: BLAIR CLS @ sentence-level", ranked, sentences, scores, keywords)

        # --- H2: BLAIR mean-pooling, review-level indexing (compare to CLS review-level from prior test) ---
        embs_r = encode(reviews, pooling="mean")
        q_r = encode([query], pooling="mean")
        scores_r = (embs_r @ q_r.T).flatten()
        ranked_r = list(np.argsort(-scores_r)[:5])
        report("H2: BLAIR mean-pool @ review-level", ranked_r, reviews, scores_r, keywords)

        # --- H1+H2 combined: mean-pooling at sentence level ---
        embs_s2 = encode(sentences, pooling="mean")
        q_s2 = encode([query], pooling="mean")
        scores_s2 = (embs_s2 @ q_s2.T).flatten()
        ranked_s2 = list(np.argsort(-scores_s2)[:5])
        report("H1+H2: BLAIR mean-pool @ sentence-level", ranked_s2, sentences, scores_s2, keywords)


if __name__ == "__main__":
    main()
