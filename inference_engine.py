#!/usr/bin/env python3
"""
inference_engine.py

Integrated SAT quad-layer conversational pipeline:
  Doorman -> Librarian -> Analyst -> Spokesperson

Replaces the earlier prototype, which ran against a hardcoded 5-sentence
dummy index and the base (non-tuned) model. This version uses:
  - librarian.py's real per-product retrieval (BM25 / BLAIR / hybrid,
    sentence-granularity, backed by dataset/sat/reviews.db)
  - the validated SFT+DPO adapter at rufus_checkpoints/final_dpo_adapter

Design notes:
  - BLAIR-RoBERTa and spaCy are loaded ONCE here and shared into Librarian
    (which supports external encode_fn/nlp for exactly this reason)
    rather than loaded a second time internally.
  - Analyst does NOT re-run extract+consolidate LLM calls over the
    evidence. The DPO-tuned Spokesperson already demonstrated (see
    project notes) that it reasons well directly over raw evidence
    sentences -- correctly surfacing contradictions and abstaining on
    thin evidence in one pass. Re-extracting with more LLM calls would
    mostly add latency. Instead Analyst organizes evidence into an
    INSPECTABLE structure (positive / negative / neutral buckets +
    contradiction flag), which the report requires (Sec 15) for
    explainability and later evaluation, using each sentence's source
    review's star rating as a transparent, cheap polarity proxy --
    documented limitation: this is review-level rating, not sentence-
    level sentiment, so a complaint inside an otherwise-5-star review
    is bucketed as "positive". Good enough for a structural signal;
    not a claim of perfect per-sentence sentiment.
  - Product selection is new: the report frames this as "aspect-specific
    insights for individual products", but the prior prototype never
    asked which product was being discussed. This version does.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import re
import sqlite3
import warnings
from typing import Dict, List

warnings.filterwarnings("ignore", message=".*Failed to load image Python extension.*")

import numpy as np
import torch
import spacy
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    GenerationConfig,
)
from peft import PeftModel

from librarian import Librarian, DB_PATH

# ---------------------------
# CONFIG
# ---------------------------
ADAPTER_PATH = "./rufus_checkpoints/final_dpo_adapter"
BASE_MODEL_ID = "NousResearch/Meta-Llama-3-8B-Instruct"
RETRIEVER_ID = "hyp1231/blair-roberta-large"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Must match the thresholds used in generate_data.py -- the Spokesperson
# was trained to abstain below this bar, so inference has to gate at the
# same point or its learned abstention behavior is being fed inputs it
# never saw that distribution of.
MIN_SCORE = 0.55
MIN_EVIDENCE_SENTENCES = 3
TOP_K = 8

KNOWN_BRANDS = [
    "Samsung", "Apple", "OnePlus", "Google", "Xiaomi", "Motorola", "Moto", "Nokia",
    "Sony", "LG", "Oppo", "Realme", "Poco", "Redmi", "Asus", "Honor", "Huawei", "BLU",
]

# Must match the aspect vocabulary used everywhere else in the project
# (generate_data.py's ASPECTS, eval_lexicon.py's ASPECT_KEYWORDS). Doorman
# is constrained to this closed set rather than free text -- an earlier,
# unconstrained version happily emitted synonyms like "user interface"
# instead of "ease of use", which doesn't hurt retrieval (raw query text is
# used, not the aspect label) but breaks abstention-message accuracy and
# would corrupt aspect-detection-accuracy metrics against the eval set.
CANONICAL_ASPECTS = [
    "battery life", "camera quality", "display quality", "performance and speed",
    "build quality and durability", "value for money", "ease of use", "overall product quality",
]

# Cheap keyword-based fallback in case the model still emits something
# outside CANONICAL_ASPECTS despite the prompt constraint.
_ASPECT_FALLBACK_KEYWORDS = {
    "battery life": ["batt"],
    "camera quality": ["camera", "photo", "picture", "video", "lens"],
    "display quality": ["screen", "display", "resolution", "brightness"],
    "performance and speed": ["performance", "speed", "fast", "slow", "lag", "processor", "cpu", "gaming", "game"],
    "build quality and durability": ["durab", "sturdy", "build quality", "crack", "scratch", "drop", "waterproof"],
    "value for money": ["price", "value", "worth", "money", "cheap", "expensive"],
    "ease of use": ["interface", "usab", "intuitive", "user friendly", "user-friendly", "setup", "navigat"],
}


def normalize_aspect(raw: str) -> str:
    if raw in CANONICAL_ASPECTS:
        return raw
    low = (raw or "").lower()
    for canonical, keywords in _ASPECT_FALLBACK_KEYWORDS.items():
        if any(kw in low for kw in keywords):
            return canonical
    return "overall product quality"


# ---------------------------
# ENGINE (shared BLAIR + LLM)
# ---------------------------
class ModelEngine:
    def __init__(self):
        print(">>> [System] Initializing Neural Engines...")

        try:
            self.nlp = spacy.load("en_core_web_sm")
        except Exception:
            print("Downloading spaCy model...")
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

        print(f">>> [System] Loading Retriever: {RETRIEVER_ID}")
        self.ret_tokenizer = AutoTokenizer.from_pretrained(RETRIEVER_ID, use_fast=True, trust_remote_code=True)
        self.ret_model = AutoModel.from_pretrained(RETRIEVER_ID, trust_remote_code=True).to(DEVICE)
        self.ret_model.eval()

        print(f">>> [System] Loading Base LLM: {BASE_MODEL_ID}")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        self.llm_base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID, quantization_config=bnb_config, device_map="auto", trust_remote_code=True,
        )
        self.llm_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=True, trust_remote_code=True)
        self.llm_tokenizer.padding_side = "left"
        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token

        if os.path.exists(ADAPTER_PATH):
            print(f">>> [System] Attaching SFT+DPO Adapter: {ADAPTER_PATH}")
            self.llm = PeftModel.from_pretrained(self.llm_base, ADAPTER_PATH, is_trainable=False)
        else:
            print("!!! [WARNING] Adapter not found -- running base model only (no SFT/DPO alignment).")
            self.llm = self.llm_base
        self.llm.eval()

    def encode_blair(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.ret_model.config.hidden_size), dtype=np.float32)
        outs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.ret_tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                v = self.ret_model(**inputs).last_hidden_state[:, 0]
                v = torch.nn.functional.normalize(v, p=2, dim=1)
            outs.append(v.cpu().numpy())
        return np.vstack(outs).astype(np.float32)

    def generate(self, system_prompt: str, user_prompt: str, max_tokens=220, temperature=0.0, top_p=1.0) -> str:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        prompt = self.llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = self.llm_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(DEVICE)

        gen_config = GenerationConfig(
            max_new_tokens=max_tokens,
            temperature=float(temperature),
            top_p=float(top_p),
            do_sample=(float(temperature) > 0.0),
            pad_token_id=int(self.llm_tokenizer.eos_token_id),
        )
        with torch.no_grad():
            out = self.llm.generate(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], generation_config=gen_config)
        resp = self.llm_tokenizer.decode(out[0, enc["input_ids"].shape[-1]:], skip_special_tokens=True)
        return resp.strip()


# ---------------------------
# PRODUCT LOOKUP / SELECTION
# ---------------------------
class ProductCatalog:
    def __init__(self):
        self._conn = sqlite3.connect(DB_PATH)

    def search(self, keyword: str, limit: int = 10):
        cur = self._conn.cursor()
        cur.execute(
            "SELECT parent_asin, product_title, COUNT(*) as n FROM reviews WHERE product_title LIKE ? "
            "GROUP BY parent_asin ORDER BY n DESC LIMIT ?",
            (f"%{keyword}%", limit),
        )
        return cur.fetchall()

    def title_for(self, asin: str) -> str:
        cur = self._conn.cursor()
        cur.execute("SELECT product_title FROM reviews WHERE parent_asin=? LIMIT 1", (asin,))
        row = cur.fetchone()
        return row[0] if row else "this phone"


# ---------------------------
# PERSONA MANAGER
# ---------------------------
class UserManager:
    def __init__(self):
        self.personas = {
            "1": "Professional Photographer",
            "2": "Budget Student",
            "3": "Pro Gamer",
            "4": "Frequent Traveler",
            "5": "App Developer",
        }

    def get_persona(self, user_id: str) -> str:
        return self.personas.get(user_id, "General Shopper")


# ---------------------------
# LAYER 1: DOORMAN
# ---------------------------
class Doorman:
    def __init__(self, engine: ModelEngine):
        self.engine = engine

    def parse(self, query: str, session_persona: str) -> Dict:
        sys_prompt = (
            "You are a strict shopping-query router. OUTPUT ONLY valid JSON matching this schema:\n"
            '{"action": "SEARCH"|"CLARIFY", "aspects": ["..."], "persona": "..."|null, "clarify_question": "..."}\n'
            f"- aspects: choose ONLY from this exact list (pick all that the query touches on): "
            f"{CANONICAL_ASPECTS}. Do not invent new aspect names or synonyms.\n"
            "- persona: if the query itself signals a persona (e.g. mentions gaming, photography, travel, "
            "being a student, being a developer), name it; otherwise null.\n"
            "- If the query is too vague to search (e.g. \"is it good?\"), set action=CLARIFY, and write an "
            "ACTUAL clarifying question asking what the user cares about -- never repeat the user's own "
            "query back as the clarify_question.\n"
            "Do not add commentary."
        )
        user_prompt = f"Session persona (may be overridden by the query): {session_persona}\nQuery: {query}"
        response = self.engine.generate(sys_prompt, user_prompt, max_tokens=100, temperature=0.0)
        try:
            m = re.search(r"\{.*\}", response, re.DOTALL)
            parsed = json.loads(m.group(0)) if m else {}
        except Exception:
            parsed = {}

        action = parsed.get("action", "SEARCH")
        raw_aspects = parsed.get("aspects") or ["overall product quality"]
        aspects = [normalize_aspect(a) for a in raw_aspects]
        # de-dup while preserving order (normalization can collapse synonyms together)
        aspects = list(dict.fromkeys(aspects))
        persona = parsed.get("persona") or session_persona
        clarify_question = parsed.get("clarify_question") or "Which matters most to you: price, camera, battery, or performance?"
        return {"action": action, "aspects": aspects, "persona": persona, "clarify_question": clarify_question}


# ---------------------------
# LAYER 2: LIBRARIAN (wraps librarian.Librarian)
# ---------------------------
# (real Librarian imported above; instantiated in SATAgent with shared encode/nlp)


# ---------------------------
# LAYER 3: ANALYST
# ---------------------------
class Analyst:
    """Organizes Librarian evidence into an inspectable structure. See
    module docstring for why this doesn't re-run LLM extraction."""

    def organize(self, retrieval: Dict, aspect: str) -> Dict:
        results = retrieval["results"]
        positive, negative, neutral = [], [], []
        for r in results:
            rating = r.get("rating")
            bucket = positive if (rating is not None and rating >= 4) else \
                     negative if (rating is not None and rating <= 2) else neutral
            bucket.append(r)

        contradiction = len(positive) > 0 and len(negative) > 0
        sufficient = len(results) >= MIN_EVIDENCE_SENTENCES
        return {
            "aspect": aspect,
            "positive": positive,
            "negative": negative,
            "neutral": neutral,
            "contradiction": contradiction,
            "sufficient_evidence": sufficient,
            "stats": retrieval["stats"],
        }

    @staticmethod
    def format_for_prompt(evidence: Dict) -> str:
        lines = []
        for label, bucket in [("Positive", evidence["positive"]), ("Negative", evidence["negative"]), ("Mixed/neutral", evidence["neutral"])]:
            for r in bucket:
                lines.append(f"- [{label}, {r['rating']}★] {r['sentence']}")
        return "\n".join(lines)


# ---------------------------
# LAYER 4: SPOKESPERSON
# ---------------------------
class Spokesperson:
    def __init__(self, engine: ModelEngine):
        self.engine = engine

    def respond(self, persona: str, product_title: str, question: str, evidence: Dict) -> str:
        if not evidence["sufficient_evidence"]:
            return (
                f"Based on the available customer reviews for {product_title}, there isn't enough "
                f"evidence about {evidence['aspect']} to give a confident answer. I don't want to guess -- "
                f"would you like me to tell you what the reviews do cover?"
            )

        sys_prompt = (
            "You are a helpful shopping assistant. Answer the user's question using ONLY the evidence "
            "below, drawn from real customer reviews. Do not invent facts not present in the evidence. "
            "Adapt tone to the persona. If the evidence is mixed or contradictory, say so explicitly. "
            "Keep the answer to 3-5 sentences."
        )
        user_prompt = (
            f"Persona: {persona}\nProduct: {product_title}\nQuestion: {question}\n\n"
            f"Evidence from customer reviews:\n{Analyst.format_for_prompt(evidence)}"
        )
        return self.engine.generate(sys_prompt, user_prompt, max_tokens=220, temperature=0.0)


def extract_candidate_models(text: str) -> List[str]:
    candidates = set()
    for brand in KNOWN_BRANDS:
        for m in re.finditer(rf"\b{re.escape(brand)}\b(?:\s+[\w\-]+){{0,2}}", text, flags=re.IGNORECASE):
            candidates.add(m.group(0).strip())
    return list(candidates)


def verify_candidates(candidates: List[str], evidence_text: str) -> List[str]:
    low = evidence_text.lower()
    return [c for c in candidates if c.lower() in low]


# ---------------------------
# AGENT (orchestration)
# ---------------------------
class SATAgent:
    def __init__(self):
        self.engine = ModelEngine()
        self.catalog = ProductCatalog()
        self.user_manager = UserManager()
        self.librarian = Librarian(encode_fn=self.engine.encode_blair, nlp=self.engine.nlp)
        self.analyst = Analyst()
        self.spokesperson = Spokesperson(self.engine)
        self.doorman = Doorman(self.engine)
        self.user_id = "1"
        self.current_asin = None
        self.current_title = None

    def set_user(self, uid):
        self.user_id = uid
        print(f">>> [System] Persona set: {self.user_manager.get_persona(uid)}")

    def select_product(self, asin: str):
        self.current_asin = asin
        self.current_title = self.catalog.title_for(asin)
        print(f">>> [System] Active product: {self.current_title} ({asin})")

    def chat(self, user_input: str):
        if not self.current_asin:
            print("Agent: No product selected. Use 'product <search term>' first.")
            return

        session_persona = self.user_manager.get_persona(self.user_id)
        intent = self.doorman.parse(user_input, session_persona)

        if intent["action"] == "CLARIFY":
            print(f"Agent: {intent['clarify_question']}")
            return

        # Use all detected aspects (not just the first) so multi-aspect
        # queries get an accurate label for logging/abstention messages --
        # retrieval itself already searches the full raw query regardless.
        aspect_label = ", ".join(intent["aspects"])
        persona = intent["persona"]
        print(f"    -> [Doorman] aspects={intent['aspects']} persona={persona!r}")

        retrieval = self.librarian.retrieve_and_prune(self.current_asin, user_input, top_k=TOP_K, mode="hybrid", min_score=MIN_SCORE)
        print(f"    -> [Librarian] {retrieval['stats']}")

        evidence = self.analyst.organize(retrieval, aspect_label)
        print(f"    -> [Analyst] pos={len(evidence['positive'])} neg={len(evidence['negative'])} "
              f"neutral={len(evidence['neutral'])} contradiction={evidence['contradiction']}")

        response = self.spokesperson.respond(persona, self.current_title, user_input, evidence)

        if evidence["sufficient_evidence"]:
            candidates = extract_candidate_models(response)
            evidence_text = Analyst.format_for_prompt(evidence) + " " + self.current_title
            verified = verify_candidates(candidates, evidence_text)
            unverified = [c for c in candidates if c not in verified and c.lower() not in self.current_title.lower()]
            if unverified:
                print(f"    -> [!] Unverified model mentions in response: {unverified}")

        print(f"Agent: {response}")


# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    agent = SATAgent()

    print("\n" + "=" * 70)
    print(" SAT -- Conversational Review Assistant")
    print(" Quad-Layer Architecture: Doorman | Librarian | Analyst | Spokesperson")
    print("=" * 70)
    print("Commands: 'persona <1-5>', 'product <search term>', 'exit'")
    print("Personas: 1=Photographer 2=Student 3=Gamer 4=Traveler 5=Developer")

    print("\nType 'exit' to quit.")
    while True:
        txt = input("\nYou: ").strip()
        if not txt:
            continue
        if txt.lower() == "exit":
            break
        if txt.lower().startswith("persona "):
            agent.set_user(txt.split(" ", 1)[1].strip())
            continue
        if txt.lower().startswith("product "):
            keyword = txt.split(" ", 1)[1].strip()
            results = agent.catalog.search(keyword)
            if not results:
                print("No products found.")
                continue
            print("Matches:")
            for i, (asin, title, n) in enumerate(results):
                print(f"  [{i}] {title[:80]} ({n} reviews) -- {asin}")
            choice = input("Pick a number: ").strip()
            try:
                asin = results[int(choice)][0]
                agent.select_product(asin)
            except Exception:
                print("Invalid choice.")
            continue

        agent.chat(txt)
