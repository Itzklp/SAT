#!/usr/bin/env python3
"""
build_review_store.py

One-time preprocessing: converts dataset/sat/phone_reviews.csv into an
indexed SQLite store keyed by parent_asin, so the Librarian can do fast
per-product review lookups instead of scanning a 670MB CSV on every query.

Run once. Safe to re-run (drops and rebuilds the table).
"""

import csv
import sqlite3
import time

CSV_PATH = "dataset/sat/phone_reviews.csv"
DB_PATH = "dataset/sat/reviews.db"


def main():
    t0 = time.time()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS reviews")
    cur.execute("""
        CREATE TABLE reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            parent_asin TEXT,
            product_title TEXT,
            brand TEXT,
            review_title TEXT,
            review_text TEXT,
            rating REAL,
            verified_purchase TEXT,
            helpful_vote INTEGER,
            timestamp INTEGER
        )
    """)

    batch = []
    n = 0
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = (row["review_text"] or "").strip()
            title = (row["review_title"] or "").strip()
            if not text and not title:
                continue
            batch.append((
                row["parent_asin"],
                row["product_title"],
                row["brand"],
                title,
                text,
                float(row["rating"]) if row["rating"] else None,
                row["verified_purchase"],
                int(row["helpful_vote"]) if row["helpful_vote"] else 0,
                int(row["timestamp"]) if row["timestamp"] else None,
            ))
            n += 1
            if len(batch) >= 5000:
                cur.executemany(
                    "INSERT INTO reviews (parent_asin, product_title, brand, review_title, review_text, rating, verified_purchase, helpful_vote, timestamp) VALUES (?,?,?,?,?,?,?,?,?)",
                    batch,
                )
                batch = []
    if batch:
        cur.executemany(
            "INSERT INTO reviews (parent_asin, product_title, brand, review_title, review_text, rating, verified_purchase, helpful_vote, timestamp) VALUES (?,?,?,?,?,?,?,?,?)",
            batch,
        )

    print(f"Inserted {n} reviews. Building index on parent_asin ...")
    cur.execute("CREATE INDEX idx_parent_asin ON reviews(parent_asin)")
    conn.commit()
    conn.close()
    print(f"Done in {time.time()-t0:.1f}s. DB at {DB_PATH}")


if __name__ == "__main__":
    main()
