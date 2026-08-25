import warnings
warnings.filterwarnings(
    "ignore",
    message=".*Failed to load image Python extension.*"
)

import os
import gzip
import json
import random
import re
import logging

from tqdm import tqdm
from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import torch
import numpy as np


# ============================================================
# Hugging Face / Transformers logging
# ============================================================

from transformers import logging as hf_logging

hf_logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)


# ============================================================
# Paths and configuration
# ============================================================

DATA_PATH = "dataset/smartphones"

META_PATH = f"{DATA_PATH}/meta.jsonl.gz"
REVIEWS_PATH = f"{DATA_PATH}/reviews.jsonl.gz"

META_OUT = f"{DATA_PATH}/meta_phones_only.jsonl.gz"
REVIEWS_OUT = f"{DATA_PATH}/reviews_phones_only.jsonl.gz"

MODEL_DIR = "./deberta_mini_phone_classifier"

BOOTSTRAP_CACHE = "/content/drive/MyDrive/SAT/labeled_bootstrap.json"

# Maximum number of products read from meta.jsonl.gz
# None = process everything
MAX_PRODUCTS = 1_000_000

# DeBERTa inference batch size
BATCH_SIZE = 32

# LLaMA bootstrap settings
BOOTSTRAP_SAMPLE_SIZE = 5000
BOOTSTRAP_BATCH_SIZE = 4

# Start fresh so the previous broken cache containing
# only one label is not reused.
FORCE_REFRESH_BOOTSTRAP = False

DEVICE = 0 if torch.cuda.is_available() else -1


# ============================================================
# 1. LLaMA 3 8B — Bootstrap Label Generator
# ============================================================

print("[1] Initializing LLaMA bootstrap generator...")

LLAMA_MODEL = "NousResearch/Meta-Llama-3-8B-Instruct"

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)


# ------------------------------------------------------------
# Load tokenizer
# ------------------------------------------------------------

llm_tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL, padding_side="left")


# ------------------------------------------------------------
# Load LLaMA 3 8B in 4-bit
# ------------------------------------------------------------

llm_model = AutoModelForCausalLM.from_pretrained(
    LLAMA_MODEL,
    quantization_config=quant_config,
    device_map="auto"
)


# ------------------------------------------------------------
# Padding configuration
# ------------------------------------------------------------

if llm_tokenizer.pad_token is None:
    llm_tokenizer.pad_token = llm_tokenizer.eos_token

llm_model.config.pad_token_id = llm_tokenizer.pad_token_id

print("[1] LLaMA 3 8B loaded successfully.")


# ============================================================
# Bootstrap labeling
# ============================================================

def bootstrap_labels(
    input_path,
    sample_size=5000,
    batch_size=4,
    force_refresh=False
):
    """
    Use LLaMA to generate weak labels for product titles.

    Labels:
        0 -> phone
        1 -> accessory

    The generated labels will later be used to train
    the smaller DeBERTa classifier.
    """

    # --------------------------------------------------------
    # Load existing cache if allowed
    # --------------------------------------------------------

    if os.path.exists(BOOTSTRAP_CACHE) and not force_refresh:

        print(
            f"[1] Loading cached bootstrap labels "
            f"from {BOOTSTRAP_CACHE}"
        )

        with open(BOOTSTRAP_CACHE, "r") as f:
            return json.load(f)


    # --------------------------------------------------------
    # Read product titles
    # --------------------------------------------------------

    print("[1] Reading product titles...")

    titles = []

    with gzip.open(
        input_path,
        "rt",
        encoding="utf-8"
    ) as f:

        for i, line in enumerate(f):

            if MAX_PRODUCTS and i >= MAX_PRODUCTS:
                break

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            title = obj.get("title", "")

            if title:
                titles.append((obj, title))


    print(
        f"[1] Available titles: {len(titles):,}"
    )


    # --------------------------------------------------------
    # Random sampling
    # --------------------------------------------------------

    sample_size = min(
        sample_size,
        len(titles)
    )

    sample = random.sample(
        titles,
        sample_size
    )

    print(
        f"[1] Selected {len(sample):,} titles "
        f"for LLaMA bootstrapping."
    )


    # --------------------------------------------------------
    # Few-shot prompt
    # --------------------------------------------------------

    fewshot_prompt = (
        "Classify the product title as exactly one of "
        "'phone' or 'accessory'.\n\n"

        "A phone is a smartphone itself, such as "
        "an iPhone, Samsung Galaxy, Google Pixel, "
        "OnePlus, Xiaomi, Motorola, etc.\n\n"

        "An accessory is an add-on such as a case, "
        "cover, charger, cable, battery, screen protector, "
        "adapter, earbuds, headset, dock, stand, mount, "
        "strap, or similar product.\n\n"

        "Examples:\n"
        "Apple iPhone 14 Pro Max → phone\n"
        "Spigen Ultra Hybrid Case for iPhone 14 → accessory\n"
        "Samsung Galaxy S23 Ultra 5G → phone\n"
        "Anker 20W USB-C Fast Charger for iPhone → accessory\n\n"

        "Return ONLY the word 'phone' or 'accessory'.\n\n"
    )


    # --------------------------------------------------------
    # Generate prompts
    # --------------------------------------------------------

    prompts = [
        fewshot_prompt
        + f"Product title: {title}\n"
        + "Answer:"
        for _, title in sample
    ]


    # --------------------------------------------------------
    # Bootstrap
    # --------------------------------------------------------

    labeled = []

    phone_count = 0
    accessory_count = 0
    unknown_count = 0


    for i in tqdm(
        range(0, len(prompts), batch_size),
        desc="Bootstrapping labels with LLM"
    ):

        batch_prompts = prompts[
            i:i + batch_size
        ]


        # ----------------------------------------------------
        # Tokenize
        # ----------------------------------------------------

        inputs = llm_tokenizer(
            batch_prompts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )


        # ----------------------------------------------------
        # Move tensors to model device
        # ----------------------------------------------------

        inputs = {
            k: v.to(llm_model.device)
            for k, v in inputs.items()
        }


        input_length = inputs["input_ids"].shape[1]


        # ----------------------------------------------------
        # Generate
        # ----------------------------------------------------

        with torch.no_grad():

            outputs = llm_model.generate(
                **inputs,
                max_new_tokens=4,
                do_sample=False,
                pad_token_id=llm_tokenizer.pad_token_id
            )


        # ----------------------------------------------------
        # IMPORTANT:
        # Keep ONLY newly generated tokens.
        #
        # The original prompt is removed.
        # ----------------------------------------------------

        generated_tokens = outputs[:, input_length:]


        decoded_outputs = llm_tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True
        )


        # ----------------------------------------------------
        # Parse each answer
        # ----------------------------------------------------

        for (_, title), result_text in zip(
            sample[i:i + batch_size],
            decoded_outputs
        ):

            output = result_text.strip().lower()


            # Remove punctuation
            output = re.sub(
                r"[^a-zA-Z ]",
                " ",
                output
            )

            output = output.strip()


            # ------------------------------------------------
            # Determine label
            # ------------------------------------------------

            if re.search(
                r"\baccessory\b",
                output
            ):

                label = "accessory"

                accessory_count += 1


            elif re.search(
                r"\bphone\b",
                output
            ):

                label = "phone"

                phone_count += 1


            else:

                label = None

                unknown_count += 1


            # ------------------------------------------------
            # Store valid label
            # ------------------------------------------------

            if label is not None:

                labeled.append(
                    {
                        "title": title,
                        "label": label
                    }
                )


        # ----------------------------------------------------
        # Progress information every 100 batches
        # ----------------------------------------------------

        if (
            (i // batch_size) % 100 == 0
            and i > 0
        ):

            print(
                f"\n[1] Current labels: "
                f"{len(labeled)} | "
                f"phones: {phone_count} | "
                f"accessories: {accessory_count} | "
                f"unknown: {unknown_count}"
            )


    # ========================================================
    # Bootstrap complete
    # ========================================================

    print("\n[1] Bootstrap generation completed.")

    print(
        f"[1] Total valid labels: {len(labeled)}"
    )

    print(
        f"[1] Phones: {phone_count}"
    )

    print(
        f"[1] Accessories: {accessory_count}"
    )

    print(
        f"[1] Unknown/unusable: {unknown_count}"
    )


    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if phone_count == 0 or accessory_count == 0:

        raise RuntimeError(
            "\nLLaMA did not produce both classes.\n"
            f"Phones: {phone_count}\n"
            f"Accessories: {accessory_count}\n\n"
            "Do not train DeBERTa yet."
        )


    # --------------------------------------------------------
    # Save cache AFTER all batches finish
    # --------------------------------------------------------

    with open(
        BOOTSTRAP_CACHE,
        "w"
    ) as f:

        json.dump(
            labeled,
            f,
            indent=2
        )


    print(
        f"[1] Cached bootstrap labels to "
        f"{BOOTSTRAP_CACHE}"
    )


    return labeled


# ============================================================
# 2. DeBERTa-small Classifier
# ============================================================

def train_deberta(labeled):

    print(
        "[2] Preparing DeBERTa training data..."
    )


    labels = {
        "phone": 0,
        "accessory": 1
    }


    # --------------------------------------------------------
    # Convert labels
    # --------------------------------------------------------

    texts = [
        ex["title"]
        for ex in labeled
    ]

    y = [
        labels[ex["label"]]
        for ex in labeled
    ]


    # --------------------------------------------------------
    # Separate classes
    # --------------------------------------------------------

    phones = [
        t
        for t, l in zip(texts, y)
        if l == 0
    ]

    accessories = [
        t
        for t, l in zip(texts, y)
        if l == 1
    ]


    print(
        f"[2] Phone labels: {len(phones)}"
    )

    print(
        f"[2] Accessory labels: {len(accessories)}"
    )


    # --------------------------------------------------------
    # Balance classes
    # --------------------------------------------------------

    min_len = min(
        len(phones),
        len(accessories)
    )


    if min_len < 2:

        raise RuntimeError(
            "Not enough examples in both classes "
            "to train DeBERTa."
        )


    random.shuffle(phones)
    random.shuffle(accessories)


    phones = phones[:min_len]
    accessories = accessories[:min_len]


    texts_balanced = (
        phones +
        accessories
    )

    y_balanced = (
        [0] * min_len +
        [1] * min_len
    )


    print(
        f"[2] Balanced training examples: "
        f"{len(texts_balanced)}"
    )


    # --------------------------------------------------------
    # Train/validation split
    # --------------------------------------------------------

    train_texts, val_texts, train_labels, val_labels = (
        train_test_split(
            texts_balanced,
            y_balanced,
            test_size=0.2,
            random_state=42,
            stratify=y_balanced
        )
    )


    # --------------------------------------------------------
    # DeBERTa tokenizer
    # --------------------------------------------------------

    model_name = (
        "microsoft/deberta-v3-small"
    )


    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )


    def tokenize(batch):

        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=128
        )


    # --------------------------------------------------------
    # Create datasets
    # --------------------------------------------------------

    train_dataset = Dataset.from_dict(
        {
            "text": train_texts,
            "label": train_labels
        }
    )

    val_dataset = Dataset.from_dict(
        {
            "text": val_texts,
            "label": val_labels
        }
    )


    # --------------------------------------------------------
    # Tokenize
    # --------------------------------------------------------

    train_dataset = train_dataset.map(
        tokenize,
        batched=True
    )

    val_dataset = val_dataset.map(
        tokenize,
        batched=True
    )


    # --------------------------------------------------------
    # Load DeBERTa
    # --------------------------------------------------------

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            model_name,
            num_labels=2
        )
    )


    # --------------------------------------------------------
    # Training arguments
    # --------------------------------------------------------

    training_args = TrainingArguments(
        output_dir="./results",

        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,

        num_train_epochs=2,

        save_strategy="no",

        report_to="none"
    )


    # --------------------------------------------------------
    # Trainer
    # --------------------------------------------------------

    trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=train_dataset,
      eval_dataset=val_dataset
    )


    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print(
        "[2] Training DeBERTa-small classifier..."
    )

    trainer.train()


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print(
        "[2] Validating classifier..."
    )

    preds = trainer.predict(
        val_dataset
    )


    y_pred = np.argmax(
        preds.predictions,
        axis=1
    )


    print(
        classification_report(
            val_labels,
            y_pred,
            target_names=[
                "phone",
                "accessory"
            ]
        )
    )


    print(
        "Confusion Matrix:"
    )

    print(
        confusion_matrix(
            val_labels,
            y_pred
        )
    )


    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )


    model.save_pretrained(
        MODEL_DIR
    )

    tokenizer.save_pretrained(
        MODEL_DIR
    )


    print(
        f"[2] Model saved to {MODEL_DIR}"
    )


    return tokenizer, model


# ============================================================
# Load or train DeBERTa
# ============================================================

def load_or_train_deberta(labeled):

    if os.path.exists(MODEL_DIR):

        print(
            f"[2] Loading saved model from "
            f"{MODEL_DIR}..."
        )


        tokenizer = (
            AutoTokenizer.from_pretrained(
                MODEL_DIR
            )
        )


        model = (
            AutoModelForSequenceClassification
            .from_pretrained(
                MODEL_DIR
            )
        )


        return tokenizer, model


    else:

        return train_deberta(
            labeled
        )


# ============================================================
# 3. Regex accessory filter
# ============================================================

ACCESSORY_REGEX = re.compile(
    r"(case|cover|charger|cable|protector|"
    r"battery|earbuds?|headset|adapter|dock|"
    r"stand|mount|strap|screen protector)",
    re.IGNORECASE
)


def regex_filter(title):

    return bool(
        ACCESSORY_REGEX.search(title)
    )


# ============================================================
# 4. DeBERTa batch processing
# ============================================================

def _process_batch(
    batch_objs,
    batch_texts,
    tokenizer,
    model,
    kept_lines,
    keep_asins,
    device
):

    enc = tokenizer(
        batch_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )


    enc = {
        k: v.to(device)
        for k, v in enc.items()
    }


    with torch.no_grad():

        logits = model(
            **enc
        ).logits


        probs = torch.softmax(
            logits,
            dim=-1
        ).cpu().numpy()


        preds = np.argmax(
            probs,
            axis=1
        )


        scores = np.max(
            probs,
            axis=1
        )


    for obj, pred, score in zip(
        batch_objs,
        preds,
        scores
    ):

        pred_label = (
            "phone"
            if pred == 0
            else "accessory"
        )


        # ----------------------------------------------------
        # Keep only confident phone predictions
        # and reject obvious accessory titles.
        # ----------------------------------------------------

        if (
            pred_label == "phone"
            and score > 0.60
            and not regex_filter(
                obj.get("title", "")
            )
        ):

            kept_lines.append(
                json.dumps(obj)
            )


            parent_asin = obj.get(
                "parent_asin"
            )


            if parent_asin:

                keep_asins.add(
                    parent_asin
                )


# ============================================================
# Filter complete Amazon dataset
# ============================================================

def filter_dataset(
    meta_path,
    reviews_path,
    tokenizer,
    model
):

    print(
        "[3] Filtering full dataset..."
    )


    device = torch.device(
        "cuda"
        if DEVICE >= 0
        else "cpu"
    )


    model.to(device)
    model.eval()


    keep_asins = set()

    kept_lines = []


    # --------------------------------------------------------
    # Process product metadata
    # --------------------------------------------------------

    batch_objs = []
    batch_texts = []

    total_products = 0


    with gzip.open(
        meta_path,
        "rt",
        encoding="utf-8"
    ) as fin:

        for line in tqdm(
            fin,
            desc="Reading meta.jsonl.gz"
        ):

            if (
                MAX_PRODUCTS
                and total_products >= MAX_PRODUCTS
            ):

                break


            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue


            title = obj.get(
                "title",
                ""
            )


            if not title:
                continue


            batch_objs.append(obj)
            batch_texts.append(title)


            total_products += 1


            if len(batch_texts) >= BATCH_SIZE:

                _process_batch(
                    batch_objs,
                    batch_texts,
                    tokenizer,
                    model,
                    kept_lines,
                    keep_asins,
                    device
                )


                batch_objs = []
                batch_texts = []


        # ----------------------------------------------------
        # Remaining batch
        # ----------------------------------------------------

        if batch_texts:

            _process_batch(
                batch_objs,
                batch_texts,
                tokenizer,
                model,
                kept_lines,
                keep_asins,
                device
            )


    # --------------------------------------------------------
    # Save filtered metadata
    # --------------------------------------------------------

    with gzip.open(
        META_OUT,
        "wt",
        encoding="utf-8"
    ) as f:

        for line in kept_lines:

            f.write(
                line + "\n"
            )


    percentage = (
        len(kept_lines)
        / total_products
        if total_products
        else 0
    )


    print(
        f"[3] Meta filtering done. "
        f"Kept {len(kept_lines)}/"
        f"{total_products} products "
        f"({percentage:.1%})"
    )


    # --------------------------------------------------------
    # Filter reviews
    # --------------------------------------------------------

    kept_reviews = []

    total_reviews = 0


    with gzip.open(
        reviews_path,
        "rt",
        encoding="utf-8"
    ) as fin:

        for line in tqdm(
            fin,
            desc="Filtering reviews.jsonl.gz"
        ):

            total_reviews += 1


            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue


            if (
                obj.get("parent_asin")
                in keep_asins
            ):

                kept_reviews.append(
                    json.dumps(obj)
                )


    # --------------------------------------------------------
    # Save filtered reviews
    # --------------------------------------------------------

    with gzip.open(
        REVIEWS_OUT,
        "wt",
        encoding="utf-8"
    ) as f:

        for line in kept_reviews:

            f.write(
                line + "\n"
            )


    review_percentage = (
        len(kept_reviews)
        / total_reviews
        if total_reviews
        else 0
    )


    print(
        f"[3] Reviews filtering done. "
        f"Kept {len(kept_reviews)}/"
        f"{total_reviews} reviews "
        f"({review_percentage:.1%})"
    )


    print(
        f"[3] Saved {META_OUT}"
    )

    print(
        f"[3] Saved {REVIEWS_OUT}"
    )


    return (
        META_OUT,
        REVIEWS_OUT
    )


# ============================================================
# Runner
# ============================================================

def run_pipeline():

    print(
        ">>> Starting Phone vs Accessory "
        "Filtering Pipeline <<<"
    )


    # --------------------------------------------------------
    # Stage 1
    # --------------------------------------------------------

    labeled = bootstrap_labels(
        META_PATH,
        sample_size=BOOTSTRAP_SAMPLE_SIZE,
        batch_size=BOOTSTRAP_BATCH_SIZE,
        force_refresh=FORCE_REFRESH_BOOTSTRAP
    )


    print(
        f"[1] Bootstrapped {len(labeled)} "
        f"samples with LLaMA"
    )


    # --------------------------------------------------------
    # Stage 2
    # --------------------------------------------------------

    tokenizer, model = (
        load_or_train_deberta(
            labeled
        )
    )


    print(
        "[2] Classifier ready."
    )


    # --------------------------------------------------------
    # Stage 3
    # --------------------------------------------------------

    filter_dataset(
        META_PATH,
        REVIEWS_PATH,
        tokenizer,
        model
    )


    print(
        ">>> Final phone-only datasets ready <<<"
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    run_pipeline()