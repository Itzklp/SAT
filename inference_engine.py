#!/usr/bin/env python3
"""
inference_engine_strict_verif.py

Single-file inference engine with strong anti-hallucination and model-verification:
 - Deterministic Doorman (JSON output)
 - Analyst returns empty when evidence is weak
 - Final-response: extract candidate model names from generated text and verify they
   appear in retrieved chunks or verified facts before returning them to the user.
 - If no verified model, refuse to invent names and return prioritized criteria.
 - Left-padding for decoder-only tokenizer
 - Avoid flash_attn import, safe BitsAndBytes settings
 - Suppress tokenizers parallelism fork warning
 - Use GenerationConfig so sampling params (temperature/top_p) are honored
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import re
from typing import List, Dict, Tuple
import numpy as np
import faiss
import spacy
import torch
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    GenerationConfig,
)
from peft import PeftModel

# ---------------------------
# CONFIG
# ---------------------------
ADAPTER_PATH = "./rufus_checkpoints/final_dpo_adapter"
BASE_MODEL_ID = "NousResearch/Meta-Llama-3-8B-Instruct"
RETRIEVER_ID = "hyp1231/blair-roberta-large"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SARESG_THRESHOLD = 0.45
MAX_RETRIEVAL_CHUNKS = 50

# Known brand tokens (for stronger heuristics)
KNOWN_BRANDS = [
    "Samsung", "Apple", "OnePlus", "Google", "Xiaomi", "Motorola", "Moto", "Nokia",
    "Sony", "LG", "Oppo", "Realme", "Poco", "Redmi", "Asus", "Honor"
]

# ---------------------------
# MODEL ENGINE
# ---------------------------
class ModelEngine:
    def __init__(self):
        print(">>> [System] Initializing Neural Engines...")

        # Spacy
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except Exception:
            print("Downloading Spacy model...")
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

        # Retriever: Blair-Roberta
        print(f">>> [System] Loading Retriever: {RETRIEVER_ID}")
        self.ret_tokenizer = AutoTokenizer.from_pretrained(RETRIEVER_ID, use_fast=True, trust_remote_code=True)
        self.ret_model = AutoModel.from_pretrained(RETRIEVER_ID, trust_remote_code=True).to(DEVICE)
        self.ret_model.eval()

        # LLM base + PEFT adapter
        print(f">>> [System] Loading Base LLM: {BASE_MODEL_ID}")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        # Avoid attn_implementation to prevent flash_attn import issues
        self.llm_base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

        # Tokenizer: enforce left padding for decoder-only
        self.llm_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=True, trust_remote_code=True)
        self.llm_tokenizer.padding_side = "left"
        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
        if self.llm_tokenizer.pad_token_id is None:
            self.llm_tokenizer.pad_token_id = self.llm_tokenizer.eos_token_id

        # Attach adapter if available
        if os.path.exists(ADAPTER_PATH):
            print(f">>> [System] Attaching DPO Adapter: {ADAPTER_PATH}")
            self.llm = PeftModel.from_pretrained(self.llm_base, ADAPTER_PATH, is_trainable=False)
        else:
            print("!!! [WARNING] Adapter not found. Running Base Model only.")
            self.llm = self.llm_base

        try:
            self.llm.eval()
        except Exception:
            pass

    def encode_blair(self, texts: List[str]) -> np.ndarray:
        """Return normalized float32 embeddings (CPU numpy) for FAISS IndexFlatIP."""
        if len(texts) == 0:
            # If model not loaded, safe fallback dims
            try:
                dim = self.ret_model.config.hidden_size
            except Exception:
                dim = 768
            return np.zeros((0, dim), dtype=np.float32)
        inputs = self.ret_tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.ret_model(**inputs)
            emb = out.last_hidden_state[:, 0]
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            emb = emb.cpu().float().numpy()
        return emb

    def _build_prompt_from_messages(self, system_prompt: str, user_prompt: str) -> str:
        return (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_prompt}<|eot_id|>\n"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_prompt}<|eot_id|>\n"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

    def generate(self, system_prompt: str, user_prompt: str, max_tokens=256, temperature=0.0, top_p=1.0, repetition_penalty=1.0) -> str:
        """
        Robust generation using GenerationConfig so temperature/top_p are honored.
        Deterministic by default (temperature=0.0).
        """
        try:
            if hasattr(self.llm_tokenizer, "apply_chat_template"):
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                enc = self.llm_tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
                input_ids = enc["input_ids"].to(DEVICE)
                attention_mask = enc.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(DEVICE)
            else:
                raise AttributeError("no apply_chat_template")
        except Exception:
            prompt = self._build_prompt_from_messages(system_prompt, user_prompt)
            enc = self.llm_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048)
            input_ids = enc["input_ids"].to(DEVICE)
            attention_mask = enc.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(DEVICE)

        gen_config = GenerationConfig(
            max_new_tokens=max_tokens,
            temperature=float(temperature),
            top_p=float(top_p),
            do_sample=(float(temperature) > 0.0),
            repetition_penalty=float(repetition_penalty),
            pad_token_id=int(self.llm_tokenizer.eos_token_id),
        )

        gen_kwargs = {"generation_config": gen_config, "input_ids": input_ids}
        if attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask

        with torch.no_grad():
            outputs = self.llm.generate(**gen_kwargs)

        if outputs is None or outputs.shape[0] == 0:
            return ""

        try:
            gen_ids = outputs[0, input_ids.shape[-1]:]
        except Exception:
            gen_ids = outputs[0]

        resp = self.llm_tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        return resp

# ---------------------------
# USER MANAGER
# ---------------------------
class UserManager:
    def __init__(self):
        self.personas = {
            "1": "Professional Photographer. Cares about ISO, Sensor Size, Lens compatibility.",
            "2": "Budget Student. Cares about Price, Durability, Battery Life.",
            "3": "Hardcore Gamer. Cares about GPU, Refresh Rate, Cooling."
        }

    def get_persona(self, user_id: str) -> str:
        return self.personas.get(user_id, "General Shopper")

# ---------------------------
# LIBRARIAN (RETRIEVAL + SARESG)
# ---------------------------
class Librarian:
    def __init__(self, engine: ModelEngine):
        self.engine = engine
        self.index = None
        self.chunks = []
        self._build_dummy_index()

    def _build_dummy_index(self):
        print(">>> [Librarian] Building Knowledge Index...")
        raw_reviews = [
            "The camera is amazing in low light. ISO 3200 is clean. However, the battery drains fast.",
            "Terrible autofocus. It hunts in the dark. Good colors though.",
            "Great laptop for gaming. The RTX 4080 destroys Cyberpunk. Fans are loud.",
            "Battery lasts 12 hours. Perfect for school. Screen is dim outdoors.",
            "The lens is sharp but heavy. Bokeh is creamy at f/1.2."
        ]
        embeddings = self.engine.encode_blair(raw_reviews)
        d = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(embeddings.astype(np.float32))
        self.chunks = raw_reviews

    def retrieve_and_prune(self, query: str, aspects: List[str]) -> Tuple[str, List[str]]:
        """
        Returns (pruned_context_text, retrieved_chunks_list)
        retrieved_chunks_list are the top-k chunks used for retrieval (for verification)
        """
        q_vec = self.engine.encode_blair([query])
        k = min(3, len(self.chunks))
        D, I = self.index.search(q_vec.astype(np.float32), k)
        retrieved_chunks = [self.chunks[i] for i in I[0]]

        if not aspects:
            return " ".join(retrieved_chunks), retrieved_chunks

        aspect_vec = self.engine.encode_blair([" ".join(aspects)])
        kept_sentences = []
        for chunk in retrieved_chunks:
            doc = self.engine.nlp(chunk)
            sents = [s.text for s in doc.sents]
            if not sents:
                continue
            sent_vecs = self.engine.encode_blair(sents)
            sims = np.dot(sent_vecs, aspect_vec.T).flatten()
            for text, score in zip(sents, sims):
                if float(score) > SARESG_THRESHOLD:
                    kept_sentences.append(text)
        result = " ".join(kept_sentences)
        if result:
            return result, retrieved_chunks
        else:
            # If pruning yields nothing, return full retrieved chunks (so verification can still check)
            return " ".join(retrieved_chunks), retrieved_chunks

# ---------------------------
# ANALYST (weak-evidence handling)
# ---------------------------
class Analyst:
    def __init__(self, engine: ModelEngine):
        self.engine = engine

    def process(self, context: str, query: str) -> str:
        # Extraction (deterministic)
        sys_prompt = "You are a rigorous data analyst. Extract direct quotes from the text relevant to the user query."
        evidence = self.engine.generate(sys_prompt, f"Text: {context}\nQuery: {query}\nQuotes:", max_tokens=150, temperature=0.0)
        if not evidence or len(evidence.strip()) < 30 or "No specific" in evidence:
            return ""  # signal weak evidence
        # Consolidation (deterministic)
        sys_prompt2 = "You are an opinion synthesizer. Summarize the evidence into a list of verified facts. Note contradictions."
        facts = self.engine.generate(sys_prompt2, f"Evidence: {evidence}\nFacts:", max_tokens=200, temperature=0.0)
        if not facts or len(facts.strip()) < 30:
            return ""
        return facts

# ---------------------------
# UTIL: model candidate extraction & verification
# ---------------------------
def extract_candidate_models(text: str) -> List[str]:
    """
    Heuristic extraction of candidate model/brand phrases from text.
    We capture:
      - Known brand tokens + following word(s)
      - Capitalized multi-word sequences (2-3 words)
    Return list of unique candidates.
    """
    candidates = set()

    # 1) known-brand + subsequent token(s)
    for brand in KNOWN_BRANDS:
        for match in re.finditer(rf"\b{re.escape(brand)}\b(?:\s+[\w\-]+){0,2}", text, flags=re.IGNORECASE):
            candidates.add(match.group(0).strip())

    # 2) Capitalized sequences (2-3 words) e.g., "Galaxy S21", "iPhone 14 Pro"
    for match in re.finditer(r"\b([A-Z][A-Za-z0-9]+(?:\s+[A-Z0-9][A-Za-z0-9]+){0,2})\b", text):
        phrase = match.group(1).strip()
        # Avoid single common words like "Phone" unless followed by model tokens
        if len(phrase.split()) == 1 and phrase.lower() in ("phone", "camera", "battery", "model"):
            continue
        candidates.add(phrase)

    # 3) bracketed model mentions like [Phone Model]
    for match in re.finditer(r"\[(.*?)\]", text):
        inner = match.group(1).strip()
        if inner:
            candidates.add(inner)

    return list(candidates)

def verify_candidates(candidates: List[str], retrieved_chunks: List[str], verified_facts: str) -> List[str]:
    """
    Return subset of candidates that appear (case-insensitive substring) in retrieved_chunks or verified_facts.
    """
    valid = []
    lower_facts = verified_facts.lower() if verified_facts else ""
    chunks_joined = " ".join(retrieved_chunks).lower() if retrieved_chunks else ""
    for c in candidates:
        lc = c.lower()
        if lc in chunks_joined or lc in lower_facts:
            valid.append(c)
            continue
        # Also check brand-only presence as weaker signal
        for brand in KNOWN_BRANDS:
            if brand.lower() in lc and brand.lower() in chunks_joined:
                valid.append(c)
                break
    return valid

# ---------------------------
# AMAZON RUFUS AGENT (Doorman deterministic, strict verification)
# ---------------------------
class AmazonRufusAgent:
    def __init__(self):
        self.engine = ModelEngine()
        self.user_manager = UserManager()
        self.librarian = Librarian(self.engine)
        self.analyst = Analyst(self.engine)
        self.user_id = "1"
        self.history = []

    def set_user(self, uid):
        self.user_id = uid
        print(f">>> [System] User Switched: {self.user_manager.get_persona(uid)}")

    def _doorman_intent(self, query: str, persona: str) -> Dict:
        """
        Deterministic intent classifier returning strict JSON:
        {action: "SEARCH"|"CLARIFY", aspects: [...], clarify_question: "..."}
        """
        sys_prompt = (
            "You are a strict shopping assistant router. OUTPUT ONLY valid JSON matching this schema:\n"
            '{"action": "SEARCH"|"CLARIFY", "aspects": ["..."], "clarify_question": "..."}\n'
            "- If specific enough to search, set action=SEARCH and fill aspects.\n"
            "- If vague, set action=CLARIFY and provide a short clarifying question.\n"
            "Do not add commentary."
        )
        user_prompt = f"Persona: {persona}\nQuery: {query}"
        response = self.engine.generate(sys_prompt, user_prompt, max_tokens=80, temperature=0.0)
        try:
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                if parsed.get("action") == "CLARIFY" and not parsed.get("clarify_question"):
                    parsed["clarify_question"] = "Which feature matters most: price, camera, or battery life?"
                if parsed.get("action") == "SEARCH" and not parsed.get("aspects"):
                    parsed["aspects"] = ["general quality"]
                return parsed
        except Exception:
            pass
        return {"action": "SEARCH", "aspects": ["price", "battery life", "durability"], "clarify_question": ""}

    def chat(self, user_input: str):
        persona = self.user_manager.get_persona(self.user_id)

        # 1. Doorman
        intent = self._doorman_intent(user_input, persona)

        if intent.get("action") == "CLARIFY":
            print(f"Agent: {intent.get('clarify_question')}")
            return

        # 2. Librarian
        aspects = intent.get("aspects", [])
        print(f"    -> [Doorman] Intent: Search {aspects}")
        raw_context, retrieved_chunks = self.librarian.retrieve_and_prune(user_input, aspects)

        # 3. Analyst
        verified_facts = self.analyst.process(raw_context, user_input)
        if not verified_facts:
            clarify = intent.get("clarify_question") or "Which is most important: price, camera, or battery life?"
            print(f"Agent: I don't have a grounded product name from available data. {clarify}")
            return

        print(f"    -> [Analyst] Verified Facts: {verified_facts.replace(chr(10), ' ')}")

        # 4. Spokesperson (DPO) - rules to avoid hallucination
        sys_prompt = (
            "You are Amazon Rufus. Use the provided VERIFIED FACTS to answer the user. ADOPT THESE RULES:\n"
            "1) Do NOT invent or fabricate product names. If you cannot identify a specific model from facts, say so and provide prioritized criteria instead.\n"
            "2) Provide 2-3 concise bullet points strictly derived from VERIFIED FACTS.\n"
            "3) Keep response concise (max 6 lines)."
        )
        user_prompt = f"Persona: {persona}\nQuery: {user_input}\nVerified Facts: {verified_facts}"
        final_response = self.engine.generate(sys_prompt, user_prompt, max_tokens=220, temperature=0.0, top_p=1.0)

        # Extract candidate model names from assistant text
        candidates = extract_candidate_models(final_response)
        verified_models = verify_candidates(candidates, retrieved_chunks, verified_facts)

        # If model(s) present in retrieved chunks or verified facts, trust and return only verified info
        if verified_models:
            # sanitize response to include only verified model mentions and facts
            # keep the assistant's text but ensure that each verified model occurs in the returned text
            # If assistant suggested many things, we prune to lines mentioning verified models
            lines = [ln.strip() for ln in final_response.splitlines() if ln.strip()]
            filtered = []
            for ln in lines:
                for vm in verified_models:
                    if vm.lower() in ln.lower():
                        filtered.append(ln)
                        break
            # if no line directly mentions verified model, create a concise verified reply
            if not filtered:
                filtered = [f"I can confirm the following model(s) appear in the retrieved data: {', '.join(verified_models)}. "
                            "Here are the facts we verified:"]
                filtered.append(verified_facts)
            out = "\n".join(filtered[:6])
            print(f"Agent: {out}")
            return

        # No verified model found — sanitize and return prioritized criteria (do not hallucinate)
        criteria_reply = (
            "I don't have enough evidence to recommend a specific model from the available data. "
            "Based on your priorities, prioritize: 1) battery >= 4000mAh, 2) IP67/68 rating for durability, "
            "3) good value SoC (balanced CPU/GPU). Would you like me to search for models matching these?"
        )
        print(f"Agent: {criteria_reply}")

# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    agent = AmazonRufusAgent()

    print("\n" + "="*70)
    print(" AMAZON Product Analyst")
    print(" Quad-Layer Architecture: REGEN | SARESG | DECOMPOSED | DPO")
    print("="*70)
    print("Select User ID:")
    print("1. Pro Photographer")
    print("2. Budget Student")
    print("3. Gamer")

    uid = input("Enter ID (1-3): ").strip()
    agent.set_user(uid)

    print("\nType 'exit' to quit.")
    while True:
        txt = input("\nYou: ")
        if txt.lower() == "exit":
            break
        agent.chat(txt)

