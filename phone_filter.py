#!/usr/bin/env python3
"""
phone_filter.py

Deterministic post-filter catching non-phone products (standalone SIM
cards, prepaid plans, power banks) that leaked through the DeBERTa
phone/accessory classifier in meta_phones_only.jsonl.gz (~3.3% of the
catalog on inspection -- see project notes). This is a targeted, minimum-
necessary rule-based patch on top of the validated classifier, NOT a
replacement for it: it only catches the specific residual failure modes
actually observed (SIM cards, power banks, screen protectors), not a
general accessory detector.

Logic: a title is non-phone if an accessory keyword (sim card / power
bank / screen protector / tempered glass) appears, and either no "phone"/
"smartphone" word appears at all, or the accessory keyword appears BEFORE
the phone word (i.e. the accessory is the product, and "phone" is just
describing what it's compatible with/for -- e.g. "SIM Card for Smart
Phones"). If "phone"/"smartphone" appears first, the accessory keyword is
almost always a bundled feature of a real phone listing (e.g. "LG ...
Prepaid Smartphone ... Sim Card Included").

Known residual gap (accepted, not fixed): titles where a phone word
precedes the accessory keyword purely by coincidental phrasing on an
actual accessory listing, e.g. "Iridium Satellite Phone Global Prepaid
SIM Card" (a SIM card, but "Phone" appears first as part of the service
name). Rare in practice; left as a documented limitation rather than
over-fit further.
"""

import re

# Phrases that are reliably diagnostic of a non-phone product on their own --
# a title containing "tablet" or "protection plan" is essentially never a
# hardware phone listing, even when "phone" also appears (it usually does,
# as the thing being protected / bundled-with -- e.g. "Prepaid Phone
# Accidental Protection Plan", "Google Fi Wireless ... Plan"). No position
# check needed; always excluded when matched.
ABSOLUTE_NONPHONE_PAT = re.compile(
    r"\btablet\b|\bipad\b|\bipod\b|protection plan|accidental protection|"
    r"insurance plan|wireless plan|unlimited plan",
    re.I,
)

# Accessory phrases that DO legitimately co-occur with genuine phone bundle
# listings (e.g. "... Prepaid Smartphone ... Sim Card Included"), so these
# are only excluded when the accessory phrase appears BEFORE any phone word
# -- i.e. the accessory is the product, and "phone" merely describes what
# it's compatible with/for (e.g. "SIM Card for Smart Phones").
ACCESSORY_PAT = re.compile(
    r"(micro |mini |nano )?sim\s*card|power\s*bank|screen protector|tempered glass|"
    r"airtime card|minutes? card|refill card|top[- ]?up card",
    re.I,
)
PHONE_PAT = re.compile(r"\bphones?\b|\bsmartphones?\b", re.I)


def is_non_phone(title: str) -> bool:
    if not title:
        return False
    if ABSOLUTE_NONPHONE_PAT.search(title):
        return True
    acc = ACCESSORY_PAT.search(title)
    if not acc:
        return False
    ph = PHONE_PAT.search(title)
    if not ph:
        return True
    return acc.start() < ph.start()


if __name__ == "__main__":
    import gzip
    import json

    n_total = 0
    n_excluded = 0
    with gzip.open("dataset/smartphones/meta_phones_only.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            n_total += 1
            obj = json.loads(line)
            if is_non_phone(obj.get("title", "") or ""):
                n_excluded += 1
    print(f"Total: {n_total}, flagged non-phone: {n_excluded} ({100*n_excluded/n_total:.2f}%)")
