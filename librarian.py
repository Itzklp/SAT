#!/usr/bin/env python3
"""
librarian.py

Layer 2 of the SAT quad-layer architecture: retrieval + pruning.

Design decision (see smoke_test_retrieval_compare.py / smoke_test_blair_investigate.py):
retrieval and pruning are unified into ONE step operating at SENTENCE
granularity, not review granularity. Whole-review BLAIR-CLS embeddings were
found to dilute signal badly (0/5 relevant top-5 results on real queries);
sentence-level BLAIR-CLS embeddings (still the documented BLaIR usage: CLS
token, L2-normalized) perform close to BM25 and catch paraphrases BM25
misses. So "retrieve chunks -> prune sentences" collapses into "retrieve
sentences directly", ranked by BM25, BLAIR dense, or a hybrid of both.

Retrieval is scoped PER PRODUCT (per parent_asin), matching the report's
objective of "aspect-specific insights for individual products" rather than
a single global cross-product index. Products have a median of ~7 reviews,
so per-product indices are cheap to build on first access and cached to
disk for reuse.
"""

import os
import pickle
import sqlite3
import warnings

warnings.filterwarnings("ignore", message=".*Failed to load image Python extension.*")

import numpy as np
import torch
import faiss
import spacy
from transformers import AutoTokenizer, AutoModel
from rank_bm25 import BM25Okapi

DB_PATH = "dataset/sat/reviews.db"
CACHE_DIR = "cache_librarian"
RETRIEVER_ID = "hyp1231/blair-roberta-large"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(CACHE_DIR, exist_ok=True)


class Librarian:
    def __init__(self, encode_fn=None, nlp=None):
        """
        encode_fn: optional externally-supplied embedding function
                   (texts: List[str]) -> np.ndarray[float32, L2-normalized, CLS-pooled],
                   so a caller (e.g. the full pipeline) can share one loaded
                   BLAIR model instead of loading a second copy. If not
                   given, Librarian loads its own.
        nlp: optional shared spaCy pipeline; loaded if not given.
        """
        self._own_model = encode_fn is None
        if self._own_model:
            print(f">>> [Librarian] Loading BLAIR-RoBERTa ({RETRIEVER_ID}) on {DEVICE} ...")
            self._tok = AutoTokenizer.from_pretrained(RETRIEVER_ID)
            self._model = AutoModel.from_pretrained(RETRIEVER_ID).to(DEVICE)
            self._model.eval()
            self.encode = self._encode_blair
        else:
            self.encode = encode_fn

        self.nlp = nlp or spacy.load("en_core_web_sm")
        self._conn = sqlite3.connect(DB_PATH)

    def _encode_blair(self, texts, batch_size=32):
        if not texts:
            return np.zeros((0, self._model.config.hidden_size), dtype=np.float32)
        outs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self._tok(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                v = self._model(**inputs).last_hidden_state[:, 0]
                v = torch.nn.functional.normalize(v, p=2, dim=1)
            outs.append(v.cpu().numpy())
        return np.vstack(outs).astype(np.float32)

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------
    def get_product_sentences(self, parent_asin):
        """Return list of dicts: sentence, review_id, rating, verified_purchase, helpful_vote."""
        cur = self._conn.cursor()
        cur.execute(
            "SELECT id, review_title, review_text, rating, verified_purchase, helpful_vote FROM reviews WHERE parent_asin = ?",
            (parent_asin,),
        )
        rows = cur.fetchall()
        sentences = []
        for review_id, title, text, rating, verified, helpful in rows:
            full = f"{title}. {text}".strip(". ").strip() if title else text
            if not full:
                continue
            doc = self.nlp(full)
            for s in doc.sents:
                s_text = s.text.strip()
                if len(s_text) > 3:
                    sentences.append({
                        "sentence": s_text,
                        "review_id": review_id,
                        "rating": rating,
                        "verified_purchase": verified,
                        "helpful_vote": helpful,
                    })
        return sentences

    # ------------------------------------------------------------------
    # Per-product index cache
    # ------------------------------------------------------------------
    def _cache_path(self, parent_asin):
        return os.path.join(CACHE_DIR, f"{parent_asin}.pkl")

    def _load_or_build_product_cache(self, parent_asin):
        path = self._cache_path(parent_asin)
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)

        sentences = self.get_product_sentences(parent_asin)
        texts = [s["sentence"] for s in sentences]
        embs = self.encode(texts) if texts else np.zeros((0, 1024), dtype=np.float32)
        tokenized = [t.lower().split() for t in texts]
        bm25 = BM25Okapi(tokenized) if tokenized else None

        cache = {"sentences": sentences, "texts": texts, "embs": embs, "bm25": bm25}
        with open(path, "wb") as f:
            pickle.dump(cache, f)
        return cache

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize(x):
        x = np.array(x, dtype=np.float64)
        if x.size == 0 or x.max() - x.min() < 1e-9:
            return np.zeros_like(x)
        return (x - x.min()) / (x.max() - x.min())

    def retrieve_and_prune(self, parent_asin, query, top_k=10, mode="hybrid", min_score=None):
        """
        Returns dict:
          results: list of {sentence, score, review_id, rating, verified_purchase, helpful_vote}
                   ranked, length <= top_k
          stats: {n_candidates, n_kept, reduction_pct}
        mode: "bm25" | "blair" | "hybrid"
        min_score: optional absolute cutoff (only meaningful for mode="blair", where
                   scores are cosine similarities in [-1,1]). For "bm25"/"hybrid" scores
                   aren't on a fixed scale, so ranking + top_k is the primary mechanism;
                   min_score is applied on the normalized [0,1] score when given.
        """
        cache = self._load_or_build_product_cache(parent_asin)
        texts = cache["texts"]
        n_candidates = len(texts)
        if n_candidates == 0:
            return {"results": [], "stats": {"n_candidates": 0, "n_kept": 0, "reduction_pct": 0.0}}

        bm25_scores = np.array(cache["bm25"].get_scores(query.lower().split())) if cache["bm25"] else np.zeros(n_candidates)
        q_vec = self.encode([query])
        dense_scores = (cache["embs"] @ q_vec.T).flatten() if cache["embs"].shape[0] else np.zeros(n_candidates)

        if mode == "bm25":
            scores = bm25_scores
            norm_scores = self._normalize(scores)
        elif mode == "blair":
            scores = dense_scores
            norm_scores = self._normalize(scores)
        elif mode == "hybrid":
            norm_scores = 0.5 * self._normalize(bm25_scores) + 0.5 * self._normalize(dense_scores)
            scores = norm_scores
        else:
            raise ValueError(f"Unknown mode: {mode}")

        order = np.argsort(-scores)
        if min_score is not None:
            order = [i for i in order if norm_scores[i] >= min_score]
        order = order[:top_k]

        results = []
        for i in order:
            s = cache["sentences"][i]
            results.append({
                "sentence": s["sentence"],
                "score": float(scores[i]),
                "review_id": s["review_id"],
                "rating": s["rating"],
                "verified_purchase": s["verified_purchase"],
                "helpful_vote": s["helpful_vote"],
            })

        n_kept = len(results)
        reduction_pct = 100.0 * (1 - n_kept / n_candidates) if n_candidates else 0.0
        return {"results": results, "stats": {"n_candidates": n_candidates, "n_kept": n_kept, "reduction_pct": reduction_pct}}


if __name__ == "__main__":
    import sys
    lib = Librarian()
    asin = sys.argv[1] if len(sys.argv) > 1 else "B00M78E4MS"
    query = sys.argv[2] if len(sys.argv) > 2 else "How is the battery life?"
    mode = sys.argv[3] if len(sys.argv) > 3 else "hybrid"
    out = lib.retrieve_and_prune(asin, query, top_k=8, mode=mode)
    print(f"ASIN={asin} query={query!r} mode={mode}")
    print("stats:", out["stats"])
    for r in out["results"]:
        print(f"  ({r['score']:.3f}) [{r['rating']}★ verified={r['verified_purchase']}] {r['sentence'][:140]!r}")
