#!/usr/bin/env python3
"""
eval_lexicon.py

Deterministic, keyword-based aspect matching used ONLY for building
evaluation-set ground truth -- deliberately independent of librarian.py's
BLAIR/hybrid semantic scoring, so the eval set isn't circular (a system
shouldn't be evaluated against labels its own retrieval mechanism produced).

This is intentionally simpler and less "smart" than the real Librarian:
plain substring matching on lemmatized-ish keyword lists. That's the point
-- it's a different, independently-inspectable signal to check the real
system against, not a competing retrieval method.
"""

import re

ASPECT_KEYWORDS = {
    "battery life": ["batt"],
    "camera quality": ["camera", "photo", "picture", "video record", "selfie", "lens", "megapixel", " mp "],
    "display quality": ["screen", "display", "resolution", "brightness", "bezel"],
    "performance and speed": ["fast", "slow", "lag", "speed", "processor", "cpu", "snapdragon", "performance",
                               "smooth", "freeze", "hang", "responsive"],
    "build quality and durability": ["durable", "durability", "sturdy", "build quality", "crack", "scratch",
                                      "drop", "broke", "waterproof", "water resistant", "solid"],
    "value for money": ["price", "value", "worth", "money", "cheap", "expensive", "affordable", "overpriced"],
    "ease of use": ["easy to use", "intuitive", "user friendly", "user-friendly", "interface", "setup",
                     "straightforward", "complicated", "confusing"],
}

# Features that are NEVER present in this (2014-2023-era, mostly budget/mid)
# corpus in practice -- used to construct genuinely unsupported questions
# with an objective, verifiable absence rather than a guessed one.
ABSENT_FEATURE_KEYWORDS = {
    "satellite messaging": ["satellite"],
    "Thread/Matter smart home support": ["thread protocol", "matter protocol", "smart home hub"],
    "under-display fingerprint sensor generation": ["under-display fingerprint", "under display fingerprint"],
    "periscope zoom lens": ["periscope"],
    "solid state buttons": ["solid state button", "haptic button"],
    "satellite SOS": ["sos via satellite", "emergency sos satellite"],
}


def find_aspect_sentences(sentences, aspect):
    """sentences: list of dicts with 'sentence' key (from Librarian.get_product_sentences).
    Returns the subset whose text contains any keyword for `aspect`."""
    keywords = ASPECT_KEYWORDS[aspect]
    out = []
    for s in sentences:
        low = s["sentence"].lower()
        if any(kw in low for kw in keywords):
            out.append(s)
    return out


def aspect_has_zero_matches(sentences, aspect_or_feature, keyword_map=None):
    keywords = (keyword_map or ASPECT_KEYWORDS)[aspect_or_feature]
    for s in sentences:
        low = s["sentence"].lower()
        if any(kw in low for kw in keywords):
            return False
    return True


def polarity_split(matched_sentences):
    """Independent contradiction signal: split by source-review star rating."""
    pos = [s for s in matched_sentences if s.get("rating") is not None and s["rating"] >= 4]
    neg = [s for s in matched_sentences if s.get("rating") is not None and s["rating"] <= 2]
    neutral = [s for s in matched_sentences if s not in pos and s not in neg]
    return pos, neg, neutral
