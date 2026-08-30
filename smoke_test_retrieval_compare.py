#!/usr/bin/env python3
"""
smoke_test_retrieval_compare.py

Diagnoses the retrieval-quality issue found in smoke_test_librarian.py by
comparing BM25 (lexical), BLAIR dense, and a simple hybrid on the same
real product/query pairs. Small, cheap, no full-corpus commitment.
"""

import csv
import warnings
warnings.filterwarnings("ignore", message=".*Failed to load image Python extension.*")

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from rank_bm25 import BM25Okapi

CSV_PATH = "dataset/sat/phone_reviews.csv"
RETRIEVER_ID = "hyp1231/blair-roberta-large"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEST_CASES = [
    ("B00M78E4MS", "How is the battery life?"),
    ("B08MXXC8TX", "How good is the camera?"),
]


def load_reviews_for_asin(asin):
    out = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["parent_asin"] == asin:
                text = (row["review_text"] or "").strip()
                title = (row["review_title"] or "").strip()
                full = f"{title}. {text}".strip(". ").strip()
                if full:
                    out.append(full)
    return out


def main():
    print(f"Loading BLAIR-RoBERTa ({RETRIEVER_ID}) on {DEVICE} ...")
    tok = AutoTokenizer.from_pretrained(RETRIEVER_ID)
    model = AutoModel.from_pretrained(RETRIEVER_ID).to(DEVICE)
    model.eval()

    def encode(texts, batch_size=16):
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = tok(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                out = model(**inputs).last_hidden_state[:, 0]
                out = torch.nn.functional.normalize(out, p=2, dim=1)
            all_embs.append(out.cpu().numpy())
        return np.vstack(all_embs).astype(np.float32)

    for asin, query in TEST_CASES:
        print("\n" + "=" * 90)
        print(f"ASIN={asin}  QUERY={query!r}")
        texts = load_reviews_for_asin(asin)
        print(f"({len(texts)} reviews)")

        # --- BM25 ---
        tokenized = [t.lower().split() for t in texts]
        bm25 = BM25Okapi(tokenized)
        bm25_scores = bm25.get_scores(query.lower().split())
        bm25_ranked = np.argsort(-bm25_scores)[:5]

        # --- BLAIR dense ---
        embs = encode(texts)
        q_vec = encode([query])
        dense_scores = (embs @ q_vec.T).flatten()
        dense_ranked = np.argsort(-dense_scores)[:5]

        # --- Hybrid: normalize both to [0,1] and average ---
        def normalize(x):
            x = np.array(x, dtype=np.float64)
            if x.max() - x.min() < 1e-9:
                return np.zeros_like(x)
            return (x - x.min()) / (x.max() - x.min())

        hybrid_scores = 0.5 * normalize(bm25_scores) + 0.5 * normalize(dense_scores)
        hybrid_ranked = np.argsort(-hybrid_scores)[:5]

        def show(label, ranked, scores):
            print(f"\n--- {label} top-5 ---")
            for idx in ranked:
                mark = "✓" if any(w in texts[idx].lower() for w in ["battery", "batt"] if "battery" in query.lower()) or \
                              any(w in texts[idx].lower() for w in ["camera", "photo", "picture"] if "camera" in query.lower()) else " "
                print(f"  [{mark}] (score={scores[idx]:.3f}) {texts[idx][:130]!r}")

        show("BM25 (lexical)", bm25_ranked, bm25_scores)
        show("BLAIR dense", dense_ranked, dense_scores)
        show("Hybrid (0.5 BM25 + 0.5 dense)", hybrid_ranked, hybrid_scores)


if __name__ == "__main__":
    main()
