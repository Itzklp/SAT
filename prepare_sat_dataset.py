import gzip
import json
import os
import csv
from collections import defaultdict
from tqdm import tqdm


# ============================================================
# Paths
# ============================================================

META_PATH = "dataset/smartphones/meta_phones_only.jsonl.gz"
REVIEWS_PATH = "dataset/smartphones/reviews_phones_only.jsonl.gz"

OUTPUT_DIR = "dataset/sat"
OUTPUT_PATH = f"{OUTPUT_DIR}/phone_reviews.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. Load phone metadata
# ============================================================

print("=" * 60)
print("SAT DATASET PREPARATION")
print("=" * 60)

print("\n[1] Loading phone metadata...")

products = {}

with gzip.open(
    META_PATH,
    "rt",
    encoding="utf-8"
) as f:

    for line in tqdm(
        f,
        desc="Loading products"
    ):

        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        parent_asin = obj.get("parent_asin")

        if not parent_asin:
            continue

        details = obj.get("details") or {}

        products[parent_asin] = {
            "parent_asin": parent_asin,
            "product_title": obj.get("title", ""),
            "price": obj.get("price"),
            "average_rating": obj.get("average_rating"),
            "rating_number": obj.get("rating_number"),
            "features": " | ".join(
                obj.get("features") or []
            ),
            "store": obj.get("store", ""),
            "brand": details.get("Brand", ""),
            "model_name": details.get("Model Name", ""),
            "screen_size": details.get("Screen Size", ""),
            "os": details.get("OS", ""),
            "storage": details.get(
                "Memory Storage Capacity",
                ""
            ),
            "wireless_carrier": details.get(
                "Wireless Carrier",
                ""
            ),
            "cellular_technology": details.get(
                "Cellular Technology",
                ""
            )
        }


print(
    f"[1] Loaded {len(products):,} phone products."
)


# ============================================================
# 2. Join reviews with products
# ============================================================

print("\n[2] Joining reviews with products...")

fieldnames = [
    "parent_asin",
    "product_title",
    "brand",
    "model_name",
    "price",
    "average_rating",
    "rating_number",
    "screen_size",
    "os",
    "storage",
    "wireless_carrier",
    "cellular_technology",
    "features",
    "store",
    "review_title",
    "review_text",
    "rating",
    "verified_purchase",
    "helpful_vote",
    "timestamp",
    "user_id"
]


total_reviews = 0
joined_reviews = 0
missing_products = 0
empty_reviews = 0


with open(
    OUTPUT_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as fout:

    writer = csv.DictWriter(
        fout,
        fieldnames=fieldnames
    )

    writer.writeheader()

    with gzip.open(
        REVIEWS_PATH,
        "rt",
        encoding="utf-8"
    ) as fin:

        for line in tqdm(
            fin,
            desc="Joining reviews"
        ):

            total_reviews += 1

            try:
                review = json.loads(line)
            except json.JSONDecodeError:
                continue

            parent_asin = review.get(
                "parent_asin"
            )

            product = products.get(
                parent_asin
            )

            if product is None:
                missing_products += 1
                continue

            review_text = (
                review.get("text") or ""
            ).strip()

            review_title = (
                review.get("title") or ""
            ).strip()

            # Ignore reviews with no textual content
            if not review_text and not review_title:
                empty_reviews += 1
                continue

            row = dict(product)

            row.update({
                "review_title": review_title,
                "review_text": review_text,
                "rating": review.get("rating"),
                "verified_purchase": review.get(
                    "verified_purchase"
                ),
                "helpful_vote": review.get(
                    "helpful_vote",
                    0
                ),
                "timestamp": review.get(
                    "timestamp"
                ),
                "user_id": review.get(
                    "user_id",
                    ""
                )
            })

            writer.writerow(row)

            joined_reviews += 1


# ============================================================
# 3. Summary
# ============================================================

print("\n" + "=" * 60)
print("DATASET PREPARATION COMPLETE")
print("=" * 60)

print(
    f"Products:          {len(products):,}"
)

print(
    f"Input reviews:      {total_reviews:,}"
)

print(
    f"Joined reviews:     {joined_reviews:,}"
)

print(
    f"Missing products:   {missing_products:,}"
)

print(
    f"Empty reviews:      {empty_reviews:,}"
)

print(
    f"\nSaved to:"
)

print(
    OUTPUT_PATH
)

print("=" * 60)
