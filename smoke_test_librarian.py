#!/usr/bin/env python3
"""
smoke_test_librarian.py

Compute-safety smoke test for the real Librarian (BLAIR-RoBERTa + FAISS +
sentence-level pruning) BEFORE running it over the full 582,798-review corpus.

Validates, on two real products pulled from dataset/sat/phone_reviews.csv:
  1. BLAIR-RoBERTa loads and encodes review text.
  2. A per-product FAISS index builds and searches correctly.
  3. spaCy sentence-splitting + aspect-similarity pruning behaves sensibly.
  4. Throughput, so we can project full-corpus embedding time/memory before committing.

Does NOT touch the full corpus. Does NOT write any checkpoints.
"""

import csv
import time
import warnings
warnings.filterwarnings("ignore", message=".*Failed to load image Python extension.*")

import numpy as np
import torch
import faiss
import spacy
from transformers import AutoTokenizer, AutoModel

CSV_PATH = "dataset/sat/phone_reviews.csv"
TEST_ASINS = ["B08MXXC8TX", "B00M78E4MS"]  # DOOGEE N30 (20ish reviews), BLU Studio C Mini
RETRIEVER_ID = "hyp1231/blair-roberta-large"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_reviews_for_asins(asins):
    wanted = set(asins)
    out = {a: [] for a in asins}
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            a = row["parent_asin"]
            if a in wanted:
                text = (row["review_text"] or "").strip()
                title = (row["review_title"] or "").strip()
                full = f"{title}. {text}".strip(". ").strip()
                if full:
                    out[a].append({
                        "text": full,
                        "rating": row["rating"],
                        "verified_purchase": row["verified_purchase"],
                        "helpful_vote": row["helpful_vote"],
                    })
    return out


def main():
    print(f"[1] Loading reviews for test ASINs {TEST_ASINS} from {CSV_PATH} ...")
    t0 = time.time()
    reviews_by_asin = load_reviews_for_asins(TEST_ASINS)
    print(f"    Done in {time.time()-t0:.1f}s")
    for a, revs in reviews_by_asin.items():
        print(f"    {a}: {len(revs)} reviews")
        if revs:
            print(f"      sample: {revs[0]['text'][:120]!r}")

    print(f"\n[2] Loading BLAIR-RoBERTa ({RETRIEVER_ID}) on {DEVICE} ...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(RETRIEVER_ID)
    model = AutoModel.from_pretrained(RETRIEVER_ID).to(DEVICE)
    model.eval()
    print(f"    Loaded in {time.time()-t0:.1f}s")

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

    # Throughput measurement on the larger of the two test sets
    test_asin = max(reviews_by_asin, key=lambda a: len(reviews_by_asin[a]))
    texts = [r["text"] for r in reviews_by_asin[test_asin]]
    print(f"\n[3] Encoding {len(texts)} reviews for {test_asin} to measure throughput ...")
    t0 = time.time()
    embs = encode(texts)
    dt = time.time() - t0
    rate = len(texts) / dt if dt > 0 else float("inf")
    print(f"    Encoded {len(texts)} reviews in {dt:.2f}s -> {rate:.1f} reviews/sec")
    print(f"    Projected time for full corpus (582,798 reviews): {582798/rate/60:.1f} minutes")

    print(f"\n[4] Building per-product FAISS index for {test_asin} (dim={embs.shape[1]}) ...")
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    print(f"    Index size: {index.ntotal}")

    query = "How is the battery life?"
    q_vec = encode([query])
    k = min(5, len(texts))
    D, I = index.search(q_vec, k)
    print(f"\n[5] Top-{k} retrieval for query: {query!r}")
    for rank, (score, idx) in enumerate(zip(D[0], I[0])):
        print(f"    #{rank+1} (score={score:.3f}): {texts[idx][:150]!r}")

    print(f"\n[6] Sentence-level aspect pruning (spaCy) on top-{k} retrieved reviews ...")
    nlp = spacy.load("en_core_web_sm")
    aspect_query = "battery life"
    aspect_vec = encode([aspect_query])
    threshold = 0.45
    kept = []
    for idx in I[0]:
        doc = nlp(texts[idx])
        sents = [s.text.strip() for s in doc.sents if s.text.strip()]
        if not sents:
            continue
        sent_vecs = encode(sents)
        sims = (sent_vecs @ aspect_vec.T).flatten()
        for s, sim in zip(sents, sims):
            if sim > threshold:
                kept.append((float(sim), s))
    kept.sort(reverse=True)
    print(f"    Retained {len(kept)} sentences above threshold {threshold} out of retrieved reviews:")
    for sim, s in kept[:10]:
        print(f"      ({sim:.3f}) {s[:150]!r}")

    print("\n[SMOKE TEST PASSED] BLAIR encoding, per-product FAISS index, and sentence pruning all work end-to-end.")


if __name__ == "__main__":
    main()
