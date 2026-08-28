# SAT Project Notes

Running record of validated state, known limitations, and artifact provenance.
Kept up to date as the project progresses — this is the source of truth for
"what's actually been verified" versus what merely runs.

## Pipeline status (as of this session)

| Stage | Status | Artifact |
|---|---|---|
| Phone/accessory filtering (DeBERTa) | Validated (~0.99 F1) | `deberta_mini_phone_classifier/` |
| Residual non-phone contamination filter | Built, validated (see below) | `phone_filter.py` |
| Product review store | Built | `dataset/sat/reviews.db` |
| Train/eval product split (leak-safe) | Built | `dataset/sat/product_splits.json` |
| Grounded synthetic SFT/DPO data | Generated, audited | `synthetic_dpo_data.json` (800 examples) |
| Librarian (retrieval + pruning) | Built, validated | `librarian.py` |
| SFT | Trained, adapter loads + generates | `rufus_checkpoints/sft_final/` |
| DPO | Trained, validated on held-out products | `rufus_checkpoints/final_dpo_adapter/` |
| Doorman / Analyst / Spokesperson integration | Built, smoke-tested | `inference_engine.py` |
| Evaluation dataset (dev + frozen test) | Built, audited | `eval_dev.json` (216), `eval_test.json` (504) |
| Baselines (A: LLM-only, B: vanilla RAG, E: full SAT) / metrics | Run, report generated | `SAT_Evaluation_Report.pdf`, `metrics.json` |
| Full 5-tier ablation (retrieval/+pruning/+Analyst/+full) | Not done -- only 3 systems compared | — |

## Known limitations (deliberately deferred, not hidden)

### 1. Retrieval sufficiency gate uses relative, not absolute, scores

`Librarian.retrieve_and_prune`'s `min_score` filter compares against
`_normalize()`'s per-query min-max-scaled score, which by construction always
stretches the best available candidate toward 1.0 regardless of whether
anything in the pool is genuinely relevant. Confirmed empirically: querying a
product about "satellite messaging and Thread/Matter smart home protocols"
(a topic the corpus never discusses) still returned "sufficient evidence"
(5 sentences above the 0.55 threshold), with the top-scoring sentence
(0.953) being about generic SMS messaging complaints, not satellite features.

In practice the system still behaved safely in every case tested, because
the Spokesperson reads the evidence *content*, not just the retrieval
count/score, and correctly declined to answer using irrelevant evidence.
This is a real but redundant safety net, not a substitute for a correct gate.

This also means `generate_data.py` used the same flawed gate to decide the
grounded/abstention split for the 800 training examples, so some portion of
that split may be less rigorous than intended (though the resulting model
still demonstrably learned real, generalizing abstention behavior on unseen
eval-split products -- see the inference validation in this session).

**Deferred fix**: switch the gate to raw (non-normalized) dense cosine
similarity, an absolute measure, with a threshold calibrated against a
labeled dev set of known-relevant / known-irrelevant aspect-product pairs --
this is naturally produced by the evaluation-dataset-building stage (report
§18-19), rather than another guessed constant.

### 2. `phone_filter.py` residual gaps (documented, low-impact)

Deterministic post-filter catches ~4.28% of the catalog (600/14,003
products) that leaked past the DeBERTa classifier: SIM cards, power banks,
airtime cards, tablets, iPads/iPods, protection/wireless/service plans.
Known residual false negatives (accepted, not chased further):
- Titles where a phone word coincidentally precedes an accessory keyword on
  an actual accessory listing, e.g. "Iridium Satellite Phone Global Prepaid
  SIM Card" (a SIM card, but "Phone" appears first as part of a service name).
- Long-tail one-off miscategorized items (e.g. a lanyard, a single stylus-only
  listing) not covered by the current keyword set.

### 3. Doorman persona detection

Fixed during this session (was previously session-menu-only); now the
Doorman also infers persona from the query text itself, falling back to the
session-selected persona. Not yet evaluated systematically against a labeled
persona-detection set.

## Evaluation dataset (`eval_dev.json` / `eval_test.json`)

720 total queries (216 dev / 504 frozen test, stratified 30/70 by category,
seed 99), built entirely from `eval_asins` (held out from all training).
Ground truth is deliberately **independent** of the real Librarian's
BLAIR/hybrid scoring -- see `eval_lexicon.py` -- to avoid evaluating the
system against labels its own retrieval mechanism produced. Two-pass
construction: `build_eval_dataset.py` (deterministic keyword+rating-based
candidate selection, no LLM) -> `sample_eval_candidates.py` (diversity-aware
quota sampling) -> `build_eval_questions.py` (LLM phrases the question text
only; never influences gold labels) -> `finalize_eval_split.py` (schema
normalization + dev/test split).

Categories (counts are dev+test): aspect_specific (150), persona_aware (150),
contradiction (100, `n_pos`/`n_neg` counts stored so skew can be analyzed),
multi_aspect (80), overall_suitability (80), unsupported_aspect (70, a
standard aspect verified absent from that product's own reviews),
unsupported_feature (50, a feature verified absent via a separate keyword
list -- satellite messaging, Thread/Matter, periscope zoom, etc.), ambiguous
(40, template-based vague queries like "is it good?", expects `CLARIFY`).

**Bug found and fixed during construction**: the `unsupported_feature`
question-phrasing prompt didn't tell the LLM which product it was writing
about, so it invented specific real flagship names (e.g. "Samsung Galaxy
S22 Ultra") unprompted -- 28% of that category (14/50) ended up asking about
a named phone that had nothing to do with the actual gold product being
tested. Smaller leakage (3/80, 6/80) in `multi_aspect`/`overall_suitability`
for the same root cause. Fixed by explicitly instructing the prompt to stay
generic ("this phone"/"the phone", no brand names) and patching only the
affected 23 entries in place (`patch_mismatched_eval_queries.py`) rather
than regenerating the whole set. 0 mismatches remain.

Audited: 0 leakage into train products, 0 empty queries, only expected
duplicates (ambiguous category's small template pool, by design).

## Evaluation results (`SAT_Evaluation_Report.pdf`, `metrics.json`)

Ran on the frozen `eval_test.json` (504 queries). Three systems, same
underlying SFT+DPO weights: A (LLM-only), B (vanilla RAG, unfiltered
BLAIR-dense top-8, no Doorman/Analyst/abstention), E (full pipeline).

**Core result (objective, ground-truth-backed)**: hallucination rate on
questions verified to have zero relevant evidence -- A 50.0%, B 42.9%,
E 30.4%. E meaningfully outperforms both baselines, including B specifically
(same retrieval, no sufficiency gate), isolating the value of the
gate + structured evidence rather than just "having retrieval at all."

**Bugs found and fixed during report construction** (methodology, not just
results, needed real correction -- see full report for detail):
- `eval_metrics.is_abstention()` initially had false negatives on real
  abstention phrasings ("I couldn't find any mention of...") and, after a
  first broaden, a false positive (flagged a confident positive answer that
  merely said "none of the reviews mention any drawbacks" as supporting
  color). Both fixed; verified against concrete examples before trusting
  the metric.
- `brand_hallucination()` originally flagged "Pixel" as a hallucinated
  brand whenever the response said "pixel density"/"pixel count" (English
  word collision, nothing to do with Google Pixel), and flagged legitimate
  evidence citations ("compared to my old iPhone" -- from an actual review)
  as fabrication. Fixed by requiring "Pixel" to appear in a real brand
  context, and by checking mentioned brands against the evidence actually
  shown to the model, not just the product title.
- Found (not previously known): Doorman's CLARIFY branch occasionally
  (~1.7% of non-ambiguous queries) writes a clarifying question that names
  a specific real phone unrelated to the actual product -- same root cause
  as the earlier interactive-CLI echo bug, reduced but not eliminated by
  that fix. Small but real; not yet fixed in code, tracked here.
- LLM-judge secondary check (Section 4 of the report) turned out to be
  unreliable: 30-36% structured-output parse failures, and near-zero
  discrimination among what did parse. Reported honestly as a negative
  finding about same-model-family self-judging, not presented as a clean
  "0% hallucination" result.

**Post-delivery correction**: the report's Findings section originally claimed
"the Analyst's positive/negative evidence structuring appears to directly
help" on contradiction acknowledgment, but the underlying numbers show the
opposite -- vanilla RAG (B) acknowledged verified contradictions MORE often
than the full pipeline (73% vs 61% for E). Caught on a post-delivery re-check
of the generated text against `metrics.json`, not during the original
page-by-page layout review. Fixed `generate_report_pdf.py`'s findings
synthesis to state the direction supported by the actual numbers (and flag it
as worth investigating, not a win) instead of asserting the opposite by
template default; corrected PDF re-sent.

**Known-still-open from this pass**: over-abstention on answerable questions
is somewhat high across all systems (26-30%) -- worth investigating whether
the abstention-pattern classifier is still too broad, or the systems
genuinely hedge unnecessarily this often. Retrieval-gate threshold
(MIN_SCORE=0.55) still not calibrated against the dev set (the point of
building it) -- this evaluation is the BEFORE, not the tuned AFTER.

## Reproducibility notes

- Product train/eval split: seed 42, `dataset/sat/product_splits.json`,
  min 10 reviews/product, 5,034 train / 888 eval products after exclusions.
- Synthetic data generation: seed 42, `generate_data.py`, hybrid retrieval
  (`MIN_SCORE=0.55`, `MIN_EVIDENCE_SENTENCES=3`), teacher =
  `NousResearch/Meta-Llama-3-8B-Instruct`.
- SFT: 1 epoch, LoRA r=16/alpha=32, lr=2e-4, batch 8 x grad-accum 2.
- DPO: 1 epoch, same LoRA continued (not a fresh adapter), lr=5e-6,
  batch 2 x grad-accum 8, beta=0.1. Final DPO loss 0.199 (from 0.616),
  reward accuracy reached 1.0, reward margin grew to 1.56.
- Hardware: single NVIDIA A30 (24GB).
