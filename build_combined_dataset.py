import os, gzip, json, random, re
from datasets import Dataset, concatenate_datasets
from tqdm import tqdm
import numpy as np

# -----------------------------
# Config
# -----------------------------
DATASET_DIR = "dataset"  # root directory containing domain subdirs
CACHE_DIR = "./cache_hf"  # local cache for HuggingFace datasets
OUTPUT_PATH = os.path.join(DATASET_DIR, "combined_dataset")
MAX_PRODUCTS = 10_000  # cap on sampled products
MAX_REVIEWS_PER_PRODUCT = 100
N_PROC = 8  # number of parallel workers for map
BATCH_SIZE = 1024  # batching for map to avoid memory issues

os.makedirs(CACHE_DIR, exist_ok=True)

# -----------------------------
# Helpers
# -----------------------------
def clean_price(val):
    if val is None:
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    val = str(val).replace(",", "").replace("₹", "").replace("$", "").strip().lower()
    val = re.sub(r"[^\d\.]", "", val)
    try:
        return float(val) if val else np.nan
    except:
        return np.nan

def safe_float(val):
    return float(val) if val is not None else np.nan

def stream_jsonl_gz_meta(path, domain):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            yield {
                "parent_asin": obj.get("parent_asin"),
                "title": obj.get("title"),
                "main_category": obj.get("main_category"),
                "average_rating": safe_float(obj.get("average_rating")),
                "rating_number": safe_float(obj.get("rating_number")),
                "price": clean_price(obj.get("price")),
                "domain": domain,
            }

def stream_jsonl_gz_reviews(path, domain):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            yield {
                "parent_asin": obj.get("parent_asin"),
                "review_title": obj.get("title"),
                "review_text": obj.get("text"),
                "review_rating": safe_float(obj.get("rating")),
                "verified_purchase": obj.get("verified_purchase"),
                "domain": domain,
            }

def load_meta_dataset(path, domain):
    return Dataset.from_generator(lambda: stream_jsonl_gz_meta(path, domain), cache_dir=CACHE_DIR)

def load_review_dataset(path, domain):
    return Dataset.from_generator(lambda: stream_jsonl_gz_reviews(path, domain), cache_dir=CACHE_DIR)

# -----------------------------
# 1. Collect and Sample Products
# -----------------------------
def sample_products(meta_dataset):
    total = len(meta_dataset)
    sample_size = min(MAX_PRODUCTS, total)
    sampled = meta_dataset.shuffle(seed=42).select(range(sample_size))
    return sampled

# -----------------------------
# 2. Join Reviews with Meta (batched)
# -----------------------------
def join_meta_reviews(meta_dataset, reviews_dataset):
    asin_to_meta = {row["parent_asin"]: row for row in tqdm(meta_dataset, desc="Building meta lookup")}

    def attach_meta_batch(batch):
        out = {
            "parent_asin": [],
            "product_title": [],
            "main_category": [],
            "average_rating": [],
            "rating_number": [],
            "price": [],
            "review_title": [],
            "review_text": [],
            "review_rating": [],
            "verified_purchase": [],
            "domain": [],
        }

        for i in range(len(batch["parent_asin"])):
            asin = batch["parent_asin"][i]
            if asin in asin_to_meta:
                m = asin_to_meta[asin]
                out["parent_asin"].append(asin)
                out["product_title"].append(m.get("title"))
                out["main_category"].append(m.get("main_category"))
                out["average_rating"].append(safe_float(m.get("average_rating")))
                out["rating_number"].append(safe_float(m.get("rating_number")))
                out["price"].append(clean_price(m.get("price")))
                out["review_title"].append(batch["review_title"][i])
                out["review_text"].append(batch["review_text"][i])
                out["review_rating"].append(safe_float(batch["review_rating"][i]))
                out["verified_purchase"].append(batch["verified_purchase"][i])
                out["domain"].append(batch["domain"][i])
        return out

    reviews_filtered = reviews_dataset.map(
        attach_meta_batch,
        batched=True,
        batch_size=BATCH_SIZE,
        num_proc=N_PROC,
        remove_columns=reviews_dataset.column_names
    )
    return reviews_filtered

# -----------------------------
# 3. Cap Reviews per Product
# -----------------------------
def cap_reviews(dataset):
    grouped = {}
    for row in tqdm(dataset, desc="Capping reviews"):
        if not row["verified_purchase"]:
            continue
        asin = row["parent_asin"]
        if asin not in grouped:
            grouped[asin] = []
        if len(grouped[asin]) < MAX_REVIEWS_PER_PRODUCT:
            grouped[asin].append(row)

    flat = [r for group in grouped.values() for r in group]
    return Dataset.from_list(flat)

# -----------------------------
# 4. Run ETL Across Domains
# -----------------------------
def run_etl():
    all_domain_datasets = []

    for domain in os.listdir(DATASET_DIR):
        domain_path = os.path.join(DATASET_DIR, domain)
        if not os.path.isdir(domain_path):
            continue

        meta_file = os.path.join(domain_path, "meta.jsonl.gz")
        reviews_file = os.path.join(domain_path, "reviews.jsonl.gz")
        if not os.path.exists(meta_file) or not os.path.exists(reviews_file):
            continue

        print(f"\n[Domain: {domain}] Loading meta and reviews...")
        meta_ds = load_meta_dataset(meta_file, domain)
        rev_ds = load_review_dataset(reviews_file, domain)

        print(f"[Domain: {domain}] Sampling products...")
        sampled_meta = sample_products(meta_ds)

        print(f"[Domain: {domain}] Joining with reviews...")
        joined = join_meta_reviews(sampled_meta, rev_ds)

        print(f"[Domain: {domain}] Capping reviews...")
        capped = cap_reviews(joined)

        all_domain_datasets.append(capped)

    if not all_domain_datasets:
        print("No datasets processed!")
        return

    print("\nMerging all domains...")
    combined = concatenate_datasets(all_domain_datasets)

    print(f"Final dataset size: {len(combined)} rows")
    print(f"Saving to {OUTPUT_PATH} ...")
    combined.save_to_disk(OUTPUT_PATH)
    print(">>> ETL Complete. Combined dataset saved.")

# -----------------------------
# Runner
# -----------------------------
if __name__ == "__main__":
    run_etl()

