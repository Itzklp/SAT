#!/usr/bin/env python3
"""Shared, objective (non-LLM) classifiers used across the evaluation."""
import re

ABSTENTION_PATTERNS = [
    r"isn't enough evidence", r"is not enough evidence", r"don't have enough (evidence|information)",
    r"do not have enough (evidence|information)", r"not enough information", r"cannot provide (an |a )?answer",
    r"can't provide (an |a )?answer", r"don't want to guess", r"no mention of", r"not addressed in",
    r"not mentioned in the (evidence|reviews)", r"reviews (do not|don't) (mention|cover|discuss|address)",
    r"unable to (find|determine|confirm)", r"i (can't|cannot) confirm", r"no (specific )?information (is )?available",
    r"not specified in", r"i don't have (access to|specific)", r"insufficient (evidence|information)",
    # broadened after inspecting real System E responses that abstain in phrasing
    # the original pattern list missed (see PROJECT_NOTES.md). Deliberately
    # NOT including bare "none of the reviews mention ..." -- inspection
    # showed that phrase is commonly used as SUPPORTING detail inside a
    # confident answer ("None of the reviews mention drawbacks, so it's a
    # good value"), not as a refusal; including it caused false positives.
    r"couldn't find (any |any specific )?mention of", r"could not find (any |any specific )?mention of",
    r"don't mention anything about", r"do not mention anything about",
    r"doesn't (specifically )?mention (the|any|whether)", r"does not (specifically )?mention (the|any|whether)",
]
_ABSTENTION_RE = re.compile("|".join(ABSTENTION_PATTERNS), re.IGNORECASE)


def is_abstention(text: str) -> bool:
    return bool(_ABSTENTION_RE.search(text or ""))


CONTRADICTION_MARKERS = [
    r"mixed", r"contradict", r"however", r"on the other hand", r"some (reviewers|customers|users) .* (others|while)",
    r"while some", r"varies?", r"differ", r"not (everyone|all) (agree|had)", r"one reviewer .* another",
    r"both positive and negative", r"disagree",
]
_CONTRADICTION_RE = re.compile("|".join(CONTRADICTION_MARKERS), re.IGNORECASE)


def acknowledges_contradiction(text: str) -> bool:
    return bool(_CONTRADICTION_RE.search(text or ""))


KNOWN_BRANDS = [
    "Samsung", "Galaxy", "iPhone", "Apple", "OnePlus", "Xiaomi", "Motorola", "Moto",
    "Nokia", "Sony", "Xperia", "LG", "Oppo", "Realme", "Poco", "Redmi", "Asus", "Honor", "Huawei", "BLU",
    "ZTE", "Alcatel", "HTC", "BlackBerry", "Blackberry",
]
# "Pixel"/"Google" excluded from the plain list above -- "pixel" collides
# constantly with "pixel density"/"pixel count" in camera/display text, a
# false-positive found during report review. Only match it in a phone-brand
# context (preceded by "Google" or followed by a model number).
_PIXEL_PAT = re.compile(r"\bGoogle Pixel\b|\bPixel\s?\d", re.IGNORECASE)


def extract_candidate_brands(text: str):
    text = text or ""
    found = [b for b in KNOWN_BRANDS if re.search(rf"\b{b}\b", text, re.IGNORECASE)]
    if _PIXEL_PAT.search(text):
        found.append("Pixel")
    return found


def brand_hallucination(response_text: str, product_title: str, evidence_text: str = "") -> bool:
    """True if the response confidently names a brand that isn't the product's own
    brand AND doesn't appear anywhere in the evidence it was given (a mention that's
    faithfully relaying what a review said -- e.g. "compared to my old iPhone" --
    is not a hallucination; only an brand appearing from nowhere is)."""
    mentioned = extract_candidate_brands(response_text)
    if not mentioned:
        return False
    title_low = (product_title or "").lower()
    evidence_low = (evidence_text or "").lower()
    return not any(b.lower() in title_low or b.lower() in evidence_low for b in mentioned)


REVIEW_CITATION_PATTERNS = [
    r"customer reviews?", r"reviewers?", r"reviews? (mention|say|state|indicate|show|suggest)",
    r"some (customers|users|people) (feel|think|say|mention|report)", r"one reviewer",
    r"customers? (feel|think|say|mention|report)", r"based on the reviews?", r"according to (the )?reviews?",
]
_REVIEW_CITATION_RE = re.compile("|".join(REVIEW_CITATION_PATTERNS), re.IGNORECASE)


def fabricates_review_citation(text: str) -> bool:
    """For a system given ZERO evidence (e.g. the LLM-only baseline), any
    claim to be drawing on 'customer reviews' is categorically fabricated --
    it was never shown any review text."""
    return bool(_REVIEW_CITATION_RE.search(text or ""))


def recall_at_k(retrieved_ids, gold_ids, k=None):
    if not gold_ids:
        return None  # undefined -- no gold set for this query type
    r = retrieved_ids[:k] if k else retrieved_ids
    hit = len(set(r) & set(gold_ids))
    return hit / len(set(gold_ids))


def precision_at_k(retrieved_ids, gold_ids, k=None):
    if not gold_ids:
        return None
    r = retrieved_ids[:k] if k else retrieved_ids
    if not r:
        return 0.0
    hit = len(set(r) & set(gold_ids))
    return hit / len(set(r))
