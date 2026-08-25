import warnings
warnings.filterwarnings("ignore", message=".*Failed to load image Python extension.*")

import os, sys, pickle
from tqdm import tqdm
import spacy
import torch
import faiss
import numpy as np
from sklearn.preprocessing import normalize
from datasets import load_dataset
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    logging
)

# -------------------------------
# Config
# -------------------------------
N_WORKERS = 16
BATCH_SIZE = 64
EMB_CACHE_DIR = "cache_embeddings"
os.makedirs(EMB_CACHE_DIR, exist_ok=True)

# -------------------------------
# Load models
# -------------------------------
print("Loading models...")
nlp = spacy.load("en_core_web_sm")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

model_name = "NousResearch/Meta-Llama-3-8B-Instruct"
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)
summarizer = pipeline("text-generation", model=model, tokenizer=tokenizer)

device = "cuda" if torch.cuda.is_available() else "cpu"
blair_tokenizer = AutoTokenizer.from_pretrained("hyp1231/blair-roberta-large")
blair_model = AutoModel.from_pretrained("hyp1231/blair-roberta-large").to(device)
print("Models loaded.")

# -------------------------------
# Utility functions
# -------------------------------
def get_embeddings_blair(texts, batch_size=64, show_progress=False):
    all_embs = []
    iterator = range(0, len(texts), batch_size)
    iterator = tqdm(iterator, desc="Encoding with BLaIR") if show_progress else iterator
    for i in iterator:
        batch = texts[i:i+batch_size]
        inputs = blair_tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = blair_model(**inputs, return_dict=True).last_hidden_state[:, 0]
            outputs = outputs / outputs.norm(dim=1, keepdim=True)
        all_embs.append(outputs.cpu().numpy())
    return np.vstack(all_embs)

def load_or_compute_embeddings(domain_name, items, prefix):
    cache_file = os.path.join(EMB_CACHE_DIR, f"{domain_name}_{prefix}.pkl")
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    emb_dict = {item: get_embeddings_blair([item])[0] for item in items}
    with open(cache_file, "wb") as f:
        pickle.dump(emb_dict, f)
    return emb_dict

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# -------------------------------
# FAISS index builder
# -------------------------------
def build_index(index_name, texts, batch_size=64):
    index_path = f"{index_name}.index"
    reviews_path = f"{index_name}.pkl"
    emb_path = f"{index_name}_embs.pkl"

    if os.path.exists(index_path) and os.path.exists(reviews_path) and os.path.exists(emb_path):
        print(f"Loading FAISS index and embeddings from disk...")
        index = faiss.read_index(index_path)
        with open(reviews_path, "rb") as f:
            reviews = pickle.load(f)
        with open(emb_path, "rb") as f:
            review_embs = pickle.load(f)
        print(f"Loaded {len(reviews)} reviews; index size = {index.ntotal}")
        return index, reviews, review_embs

    print("Computing embeddings for all reviews...")
    review_embs = get_embeddings_blair(texts, batch_size=batch_size, show_progress=True)
    d = blair_model.config.hidden_size
    index = faiss.IndexFlatL2(d)
    index.add(review_embs)
    
    with open(reviews_path, "wb") as f:
        pickle.dump(texts, f)
    with open(emb_path, "wb") as f:
        pickle.dump(review_embs, f)
    faiss.write_index(index, index_path)
    
    print(f"FAISS index, reviews, and embeddings saved: {index_path}, {reviews_path}, {emb_path}")
    return index, texts, review_embs

# -------------------------------
# Load combined dataset
# -------------------------------
def load_combined_reviews():
    ds = load_dataset("arrow", data_files="dataset/combined_dataset/data-00000-of-00001.arrow")["train"]
    reviews = [
        f"Product: {r.get('product_title','Unknown')}, "
        f"Domain: {r.get('domain','Unknown')}, "
        f"Title: {r.get('review_title','')}, "
        f"Review: {r.get('review_text','')}"
        for r in tqdm(ds, desc="Preparing reviews") if r.get('review_text','').strip()
    ]
    print(f"Total reviews loaded: {len(reviews)}")
    return reviews

# -------------------------------
# Domain-specific config loader
# -------------------------------
def load_domain_info(domain_name):
    import importlib.util
    path = f"domains/{domain_name}.py"
    spec = importlib.util.spec_from_file_location(domain_name, path)
    domain_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(domain_module)
    return domain_module.ASPECTS, domain_module.PERSONAS, domain_module.PROMPT_TEMPLATE

# -------------------------------
# Review retrieval
# -------------------------------
def retrieve_reviews(query, index, reviews, review_embs, aspect_vecs=None, persona_vecs=None, aspect=None, persona=None, top_k=3, sim_thresh=0.5):
    q_vec = get_embeddings_blair([query], batch_size=1)
    D, I = index.search(q_vec, top_k * 5)
    candidates = [reviews[i] for i in I[0] if 0 <= i < len(reviews)]

    filtered = []
    for idx in I[0]:
        if idx < 0 or idx >= len(reviews):
            continue
        r_vec = review_embs[idx]
        keep = True
        if aspect and aspect_vecs:
            keep &= cosine_sim(r_vec, aspect_vecs[aspect]) >= sim_thresh
        if persona and persona_vecs:
            keep &= cosine_sim(r_vec, persona_vecs[persona]) >= sim_thresh
        if keep:
            filtered.append(reviews[idx])
    return filtered[:top_k] if filtered else candidates[:top_k]

# -------------------------------
# Summarization
# -------------------------------
def summarize_reviews(snippets, domain_prompt, persona=None, aspects=None):
    if not snippets:
        return "No reviews found."
    snippets = snippets[:5]
    cleaned = [" ".join(s.split())[:300] for s in snippets]  # truncate for speed
    persona_str = f"a {persona}" if persona else "a general customer"
    aspect_str = f" focusing on {', '.join(aspects)}" if aspects else ""
    context = "\n".join([f"{i+1}. {t}" for i, t in enumerate(cleaned)])
    prompt = domain_prompt.format(persona=persona_str, aspects=aspect_str, reviews=context)
    out = summarizer(prompt, max_new_tokens=120, do_sample=False, temperature=0, pad_token_id=tokenizer.eos_token_id)
    result = out[0]["generated_text"].strip()
    return result.split("Summary:")[-1].strip() if "Summary:" in result else result

# -------------------------------
# Main CLI
# -------------------------------
def run_demo(index_name):
    reviews = load_combined_reviews()
    index, reviews, review_embs = build_index(index_name, reviews)
    
    print("\nPersona-Driven Review Consultant\n")
    while True:
        query = input("Enter your query (or 'exit'): ")
        if query.lower() == "exit":
            break
        
        # Simple domain detection
        domain = "smartphones" if "phone" in query.lower() else "office_products" if "office" in query.lower() else "beauty"
        print("Loading domain ", domain, "...")
        ASPECTS, PERSONAS, PROMPT_TEMPLATE = load_domain_info(domain)
        
        aspect_vecs = load_or_compute_embeddings(domain, ASPECTS, "aspects")
        persona_vecs = load_or_compute_embeddings(domain, PERSONAS, "personas")
        print("Loaded Aspects and Personas")

        detected_aspects = classifier(query, ASPECTS, multi_label=True)
        aspects = [l for l, s in zip(detected_aspects["labels"], detected_aspects["scores"]) if s>0.5]
        aspect = aspects[0] if aspects else None

        detected_personas = classifier(query, PERSONAS, multi_label=True)
        persona_list = [l for l, s in zip(detected_personas["labels"], detected_personas["scores"]) if s>0.5]
        persona = persona_list[0] if persona_list else None
        print("Detected Aspects and Personas")

        retrieved = retrieve_reviews(query, index, reviews, review_embs, aspect_vecs, persona_vecs, aspect, persona)
        summary = summarize_reviews(retrieved, PROMPT_TEMPLATE, persona, aspects)

        print("\n--- RESULT ---")
        print(f"Aspects: {aspects if aspects else 'Not Detected'}")
        print(f"Persona: {persona if persona else 'Not Detected'}")
        print(f"Summary: {summary}")
        print("Evidence:")
        for r in retrieved:
            print(f"  - {r[:200]}...")
        print("--------------\n")

# -------------------------------
# Entry point
# -------------------------------
if __name__ == "__main__":
    if len(sys.argv)!=2:
        print("Usage: python blair.py <index_name_without_extension>")
        sys.exit(1)
    run_demo(sys.argv[1])

