#!/usr/bin/env python3
"""
generate_report_pdf.py

Renders metrics.json (+ manual findings/limitations text) into a multi-page
PDF report via matplotlib's PdfPages. No external PDF library needed.
"""

import json
import textwrap
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

SYSTEM_LABELS = {"A": "A: LLM-only", "B": "B: Vanilla RAG", "E": "E: Full SAT"}
SYSTEM_COLORS = {"A": "#c0392b", "B": "#e67e22", "E": "#2471a3"}

PAGE_SIZE = (8.5, 11)


def new_page():
    fig = plt.figure(figsize=PAGE_SIZE)
    return fig


MIN_Y = 0.10  # below this, a new paragraph/heading no longer fits -- start a new page


def text_page(pdf, title, paragraphs, footer=None):
    """Auto-paginates: if content would run past the footer, starts a new
    page (title suffixed "(cont.)") rather than silently overflowing or
    overlapping the footer -- a real bug found during report review where
    a long Findings/Limitations section got cut off at the page boundary."""
    page_num = 1

    def start_page():
        fig = new_page()
        page_title = title if page_num == 1 else f"{title} (cont.)"
        fig.text(0.08, 0.94, page_title, fontsize=18, fontweight="bold", va="top")
        return fig, 0.87

    fig, y = start_page()
    for para in paragraphs:
        if isinstance(para, tuple):
            heading, body = para
            block_lines = 1 + len(textwrap.wrap(body, width=95))
        else:
            heading, body = None, para
            block_lines = len(textwrap.wrap(body, width=95))
        block_height = block_lines * 0.026 + 0.018 + (0.035 - 0.026 if heading else 0)

        if y - block_height < MIN_Y:
            if footer:
                fig.text(0.08, 0.04, footer, fontsize=8, color="gray", va="top")
            pdf.savefig(fig)
            plt.close(fig)
            page_num += 1
            fig, y = start_page()

        if heading:
            fig.text(0.08, y, heading, fontsize=13, fontweight="bold", va="top")
            y -= 0.035
        for line in textwrap.wrap(body, width=95):
            fig.text(0.08, y, line, fontsize=10.5, va="top", family="sans-serif")
            y -= 0.026
        y -= 0.018

    if footer:
        fig.text(0.08, 0.04, footer, fontsize=8, color="gray", va="top")
    pdf.savefig(fig)
    plt.close(fig)
    plt.close(fig)


def bar_chart_page(pdf, title, subtitle, systems, values, ylabel, pct=True, ylim=None, annotate_fmt="{:.0%}"):
    fig = new_page()
    fig.text(0.08, 0.94, title, fontsize=16, fontweight="bold", va="top")
    subtitle_lines = textwrap.wrap(subtitle, width=95) if subtitle else []
    y = 0.905
    for line in subtitle_lines:
        fig.text(0.08, y, line, fontsize=9.5, color="dimgray", va="top")
        y -= 0.022
    chart_top = y - 0.04
    ax = fig.add_axes([0.12, 0.12, 0.78, chart_top - 0.12])

    labels = [SYSTEM_LABELS.get(s, s) for s in systems]
    colors = [SYSTEM_COLORS.get(s, "#555") for s in systems]
    bars = ax.bar(labels, [v if v is not None else 0 for v in values], color=colors, width=0.5)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    for b, v in zip(bars, values):
        if v is None:
            continue
        ax.annotate(annotate_fmt.format(v), (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    pdf.savefig(fig)
    plt.close(fig)


def table_page(pdf, title, headers, rows, subtitle=None):
    fig = new_page()
    fig.text(0.08, 0.94, title, fontsize=16, fontweight="bold", va="top")
    subtitle_lines = textwrap.wrap(subtitle, width=95) if subtitle else []
    y = 0.905
    for line in subtitle_lines:
        fig.text(0.08, y, line, fontsize=9.5, color="dimgray", va="top")
        y -= 0.022
    table_top = y - 0.04

    ax = fig.add_axes([0.08, table_top - 0.35, 0.84, 0.30])
    ax.axis("off")
    tab = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(9.5)
    tab.auto_set_column_width(col=list(range(len(headers))))
    tab.scale(1, 2.2)
    for (r, c), cell in tab.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2471a3")
            cell.set_text_props(color="white", fontweight="bold")
    pdf.savefig(fig)
    plt.close(fig)


def pct_or_na(x):
    return f"{x:.1%}" if isinstance(x, (int, float)) else "N/A"


def num_or_na(x, fmt="{:.2f}"):
    return fmt.format(x) if isinstance(x, (int, float)) else "N/A"


def main():
    metrics = json.load(open("metrics.json"))
    systems = metrics.get("systems_present", ["A", "B", "E"])
    out_path = "SAT_Evaluation_Report.pdf"

    with PdfPages(out_path) as pdf:
        # ---------------- Title page ----------------
        fig = new_page()
        fig.text(0.5, 0.62, "SAT", fontsize=44, fontweight="bold", ha="center")
        fig.text(0.5, 0.55, "Conversational E-commerce Review Analysis", fontsize=16, ha="center")
        fig.text(0.5, 0.51, "System Evaluation Report", fontsize=16, ha="center", color="dimgray")
        fig.text(0.5, 0.40, f"Frozen test set: {metrics.get('n_test_total', '?')} queries, 8 categories", fontsize=11, ha="center")
        fig.text(0.5, 0.36, f"Systems compared: {', '.join(SYSTEM_LABELS.get(s, s) for s in systems)}", fontsize=11, ha="center")
        fig.text(0.5, 0.10, f"Generated {date.today().isoformat()}", fontsize=9, ha="center", color="gray")
        pdf.savefig(fig)
        plt.close(fig)

        # ---------------- Executive summary ----------------
        halluc = metrics.get("abstention_on_unanswerable", {})
        summary_lines = []
        if halluc:
            for s in ["A", "B", "E"]:
                if s in halluc:
                    summary_lines.append(f"System {s}: {halluc[s]['hallucination_rate']:.1%} hallucination rate on questions with zero real evidence.")
        text_page(pdf, "Executive Summary", [
            ("What this measures", "This report evaluates SAT's full quad-layer pipeline (System E) against two "
             "baselines using the SAME underlying fine-tuned model weights, so differences are attributable to "
             "the retrieval/reasoning architecture, not to model tuning. All numbers come from the frozen test "
             "split (eval_test.json), never used to tune any threshold in this system."),
            ("The headline hallucination number", " ".join(summary_lines) if summary_lines else "See Section 3."),
            ("How hallucination was measured", "The primary metric is fully objective, not a judge's opinion: "
             "for questions where the review corpus genuinely contains zero relevant evidence (verified "
             "independently via keyword matching, not the system's own retrieval), any confident answer is by "
             "construction a fabrication. A secondary, disclosed-as-imperfect LLM-judge check on answerable "
             "questions provides a second signal (Section 4)."),
            ("Read next", "Section 2: retrieval quality. Section 3: the core hallucination results. "
             "Section 4: judge-based grounding check. Section 5: contradiction handling & clarification. "
             "Section 6: efficiency. Section 7: findings and what should be fixed next. Section 8: limitations."),
        ])

        # ---------------- Methodology ----------------
        text_page(pdf, "1. Methodology", [
            ("Three systems, one model", "System A (LLM-only) answers from parametric knowledge with zero "
             "retrieval. System B (vanilla RAG) retrieves the top-8 sentences by raw BLAIR-dense similarity with "
             "no aspect-aware pruning threshold and answers in one shot, no persona/abstention logic. System E "
             "is the full pipeline: Doorman (intent/aspect/persona parsing) -> Librarian (hybrid BM25+BLAIR "
             "retrieval, sentence-level pruning, evidence-sufficiency gate) -> Analyst (positive/negative/"
             "neutral evidence structuring, contradiction flagging) -> Spokesperson (evidence-constrained "
             "generation or deterministic abstention). All three use the same SFT+DPO-tuned Llama-3-8B weights."),
            ("Evaluation set construction", "720 queries across 8 categories (aspect-specific, persona-aware, "
             "multi-aspect, overall-suitability, contradiction-heavy, unsupported-aspect, unsupported-feature, "
             "ambiguous), built entirely from products held out from training. Ground truth (which sentences "
             "are relevant, which products have zero evidence for a given aspect, which have genuinely "
             "contradictory evidence) was labeled by an independent keyword+rating-based method, deliberately "
             "NOT using the system's own semantic retrieval scores -- avoiding a circular evaluation. Split "
             "30/70 into a dev set (used to calibrate thresholds) and this frozen test set (720 -> 504 queries, "
             "touched only for the numbers in this report)."),
            ("What 'hallucination' means here", "Two independent measurements. (1) Objective: on the "
             "unsupported-aspect / unsupported-feature categories, the corpus contains verifiably zero relevant "
             "evidence -- so any response that answers confidently rather than abstaining is fabricating "
             "content by construction, no judgment call required. (2) LLM-judge: a secondary check on "
             "answerable-category responses, asking whether specific claims are supported by the evidence the "
             "system actually retrieved. This is disclosed as a weaker signal -- see Section 8."),
        ], footer="SAT Evaluation Report -- Methodology")

        # ---------------- Retrieval ----------------
        retrieval = metrics.get("retrieval", {})
        if retrieval:
            syss = [s for s in ["B", "E"] if s in retrieval]
            bar_chart_page(pdf, "2. Retrieval Recall", "Mean recall against independently-labeled gold relevant sentences (higher is better)",
                           syss, [retrieval[s]["mean_recall"] for s in syss], "Mean Recall", ylim=(0, 1))
            bar_chart_page(pdf, "2b. Retrieval Precision", "Mean precision against independently-labeled gold relevant sentences (higher is better)",
                           syss, [retrieval[s]["mean_precision"] for s in syss], "Mean Precision", ylim=(0, 1))

        ctx = metrics.get("context_reduction", {})
        if ctx:
            syss = [s for s in ["B", "E"] if s in ctx]
            rows = [[SYSTEM_LABELS.get(s, s), f"{ctx[s]['avg_n_candidates']:.0f}", f"{ctx[s]['avg_n_kept']:.0f}", f"{ctx[s]['avg_reduction_pct']:.1f}%"] for s in syss]
            table_page(pdf, "2c. Context Size / Pruning", headers=["System", "Avg. candidate sentences", "Avg. kept", "Reduction"], rows=rows,
                       subtitle="How much raw retrieved context is discarded before generation")

        # ---------------- Hallucination (core) ----------------
        if halluc:
            syss = [s for s in ["A", "B", "E"] if s in halluc]
            bar_chart_page(pdf, "3. Hallucination Rate on Unsupported Questions",
                           "Objective measurement: corpus verified to contain ZERO relevant evidence -- any confident answer is fabricated (lower is better)",
                           syss, [halluc[s]["hallucination_rate"] for s in syss], "Hallucination Rate", ylim=(0, 1))
            rows = [[SYSTEM_LABELS.get(s, s), halluc[s]["n"], pct_or_na(halluc[s]["abstention_rate"]), pct_or_na(halluc[s]["hallucination_rate"])] for s in syss]
            table_page(pdf, "3b. Abstention Detail", headers=["System", "N questions", "Correctly abstained", "Hallucinated"], rows=rows)

        over_abst = metrics.get("over_abstention_on_answerable", {})
        if over_abst:
            syss = [s for s in ["A", "B", "E"] if s in over_abst]
            bar_chart_page(pdf, "3c. Over-Abstention on Answerable Questions",
                           "Refusing to answer when real evidence WAS available -- a different failure mode (lower is better)",
                           syss, [over_abst[s]["over_abstention_rate"] for s in syss], "Over-Abstention Rate", ylim=(0, 1))

        fab = metrics.get("system_a_fabricated_review_citation_rate")
        if fab:
            table_page(pdf, "3d. Fabricated Review Citations (System A only)",
                       headers=["System", "N questions", "Rate"],
                       rows=[["A: LLM-only", fab["n"], pct_or_na(fab["rate"])]],
                       subtitle="System A is given ZERO review text. Any response that says 'reviewers mention...' / "
                                 "'customers say...' is inventing review content that was never shown to it.")

        brand = metrics.get("brand_hallucination_rate", {})
        if brand:
            syss = [s for s in ["A", "B", "E"] if s in brand]
            rows = [[SYSTEM_LABELS.get(s, s), brand[s]["n"], f"{brand[s]['rate']:.1%} ({round(brand[s]['rate']*brand[s]['n'])})"] for s in syss]
            table_page(pdf, "3e. Wrong-Brand Hallucination Rate", headers=["System", "N checked", "Rate (count)"], rows=rows,
                       subtitle="A brand mention that matches neither the product nor anything in the evidence shown to the "
                                 "model (a review citing a competitor phone for comparison doesn't count -- only a mention "
                                 "appearing from nowhere does). Rates are low across all systems; every one of Full SAT's 4 "
                                 "cases traces to the same Doorman CLARIFY bug in Section 5c, not to the answer-generation path.")

        # ---------------- Judge grounding ----------------
        judge = metrics.get("judge_grounding")
        if judge:
            syss = [s for s in ["B", "E"] if s in judge]
            total_n = {s: judge[s]["n"] + judge[s]["n_judge_parse_failures"] for s in syss}
            parse_fail_rate = {s: judge[s]["n_judge_parse_failures"] / total_n[s] for s in syss}
            max_fail = max(parse_fail_rate.values()) if parse_fail_rate else 0
            max_hall = max((judge[s]["hallucination_rate"] for s in syss), default=0)

            if max_fail > 0.20 or max_hall < 0.05:
                # This IS what happened in this run: high parse-failure rate and
                # near-zero discrimination -- not "no hallucinations found", but
                # the judge failing to discriminate at all. Reporting it as the
                # unreliability finding it actually is, not a clean result.
                rows = [[SYSTEM_LABELS.get(s, s), total_n[s], judge[s]["n_judge_parse_failures"],
                         f"{parse_fail_rate[s]:.0%}", judge[s]["n"], pct_or_na(judge[s]["hallucination_rate"])] for s in syss]
                table_page(pdf, "4. LLM-Judge Grounding Check -- Judge Was Unreliable",
                           headers=["System", "N attempted", "Parse failures", "Fail rate", "N scored", "Flagged hallucinated"],
                           rows=rows,
                           subtitle="This did NOT come back clean -- the judge failed to produce valid structured output "
                                    "20-36% of the time, and among responses it DID score, it almost never flagged anything "
                                    "as hallucinated. That pattern (high failure rate + near-zero discrimination) reads as "
                                    "the judge failing to do useful fact-checking, not as evidence of near-perfect grounding. "
                                    "Treat Section 3's objective measurements as authoritative; do not read this section "
                                    "as '0% hallucination'. See Limitations.")
            else:
                bar_chart_page(pdf, "4. LLM-Judge Grounding Check",
                               "Secondary, weaker signal (see Limitations): judge model flags unsupported specific claims (lower is better)",
                               syss, [judge[s]["hallucination_rate"] for s in syss], "Judged Hallucination Rate", ylim=(0, 1))

        # ---------------- Contradiction / Clarify ----------------
        contra = metrics.get("contradiction_acknowledgment", {})
        if contra:
            syss = [s for s in ["B", "E"] if s in contra]
            bar_chart_page(pdf, "5. Contradiction Acknowledgment Rate",
                           "On queries with verified conflicting evidence (both positive AND negative reviews on the same aspect) -- does the response say so? (higher is better)",
                           syss, [contra[s]["rate"] for s in syss], "Acknowledgment Rate", ylim=(0, 1))

        clarify = metrics.get("clarify_behavior")
        if clarify:
            rows = [
                ["Clarify recall (on genuinely ambiguous queries)", clarify["n_ambiguous"], pct_or_na(clarify["clarify_recall"])],
                ["False-clarify rate (on answerable queries)", clarify["n_non_ambiguous"], pct_or_na(clarify["false_clarify_rate"])],
            ]
            table_page(pdf, "5b. Clarification Behavior (System E only)", headers=["Metric", "N", "Value"], rows=rows,
                       subtitle="Baselines A/B have no clarification mechanism -- they always attempt an answer")

        false_clarify = metrics.get("doorman_false_clarify")
        if false_clarify:
            rows = [
                ["False CLARIFY on non-ambiguous queries", false_clarify["n_non_ambiguous"], f"{false_clarify['n_false_clarify']} ({pct_or_na(false_clarify['false_clarify_rate'])})"],
                ["...of those, hallucinated a brand name in the clarifying question", false_clarify["n_false_clarify"], str(false_clarify["n_with_hallucinated_brand_in_question"])],
            ]
            table_page(pdf, "5c. Bug: Doorman Hallucinates Brand Names", headers=["Metric", "N", "Count"], rows=rows,
                       subtitle="Rare (under 2% of queries) but real: on a handful of unsupported-feature questions, Doorman's "
                                 "CLARIFY branch wrote a clarifying question that named a specific real phone (e.g. 'Samsung "
                                 "Galaxy S21 Ultra') unrelated to the actual product being discussed. Same root cause as the "
                                 "earlier interactive-CLI fix (echoing/rephrasing instead of asking); that fix reduced but did "
                                 "not eliminate this failure mode.")

        # ---------------- Efficiency ----------------
        eff = metrics.get("efficiency", {})
        if eff:
            syss = [s for s in ["A", "B", "E"] if s in eff]
            rows = [[SYSTEM_LABELS.get(s, s), num_or_na(eff[s]["p50_latency_s"]), num_or_na(eff[s]["p95_latency_s"]),
                     num_or_na(eff[s]["mean_input_tokens"], "{:.0f}"), num_or_na(eff[s]["mean_output_tokens"], "{:.0f}")] for s in syss]
            table_page(pdf, "6. Efficiency", headers=["System", "p50 latency (s)", "p95 latency (s)", "Mean input tok.", "Mean output tok."], rows=rows,
                       subtitle="Latency measured under batched generation (batch=8) -- approximate per-item cost, not single-request latency")

        # ---------------- Findings (data-driven, synthesized from metrics.json) ----------------
        findings = []
        if halluc and "A" in halluc and "E" in halluc:
            drop_a = halluc["A"]["hallucination_rate"] - halluc["E"]["hallucination_rate"]
            findings.append(
                f"Hallucination on genuinely unsupported questions dropped from {halluc['A']['hallucination_rate']:.0%} "
                f"(LLM-only) to {halluc['E']['hallucination_rate']:.0%} (Full SAT) -- a {drop_a:.0%}-point reduction. "
                f"This is the report's central claim (structured decomposition + an evidence-sufficiency gate reduces "
                f"hallucination) and it holds on an objective, ground-truth-backed measurement, not a judge's opinion."
            )
        if halluc and "B" in halluc and "E" in halluc:
            drop_b = halluc["B"]["hallucination_rate"] - halluc["E"]["hallucination_rate"]
            verdict = "meaningfully outperforms" if drop_b > 0.05 else ("is roughly comparable to" if abs(drop_b) <= 0.05 else "underperforms")
            findings.append(
                f"Against vanilla RAG specifically (same retrieval, no aspect-pruning/abstention gate), Full SAT "
                f"{verdict} the baseline on unsupported-question hallucination ({halluc['B']['hallucination_rate']:.0%} "
                f"vs {halluc['E']['hallucination_rate']:.0%}). This isolates the value of the sufficiency gate + "
                f"structured evidence, not just having retrieval at all."
            )
        if fab:
            findings.append(
                f"System A fabricated review content it was never shown in {fab['rate']:.0%} of responses -- "
                f"confirming that without retrieval, the model defaults to writing AS IF it had read reviews, "
                f"a distinct and important hallucination mode from simply guessing at specs."
            )
        over_abst = metrics.get("over_abstention_on_answerable", {})
        if "E" in over_abst:
            rate = over_abst["E"]["over_abstention_rate"]
            if rate > 0.10:
                findings.append(
                    f"Full SAT over-abstains (refuses to answer despite real evidence being available) on "
                    f"{rate:.0%} of answerable questions -- worth investigating: is the evidence-sufficiency "
                    f"threshold too conservative? This is the retrieval-gate calibration issue flagged during "
                    f"development (PROJECT_NOTES.md) and should be tuned against the DEV set, not this frozen "
                    f"test set."
                )
            else:
                findings.append(f"Over-abstention on answerable questions stayed low ({rate:.0%}) -- the sufficiency gate isn't costing much recall.")
        contra = metrics.get("contradiction_acknowledgment", {})
        if "B" in contra and "E" in contra:
            e_rate, b_rate = contra["E"]["rate"], contra["B"]["rate"]
            if e_rate > b_rate + 0.05:
                contra_verdict = "the Analyst's positive/negative evidence structuring appears to directly help here"
            elif b_rate > e_rate + 0.05:
                contra_verdict = (
                    "counter-intuitively, vanilla RAG acknowledged it MORE often -- the Analyst's structuring "
                    "or the sufficiency gate may be suppressing contradiction signal here rather than helping; "
                    "worth investigating rather than assumed as a win"
                )
            else:
                contra_verdict = "the two are roughly comparable on this measure"
            findings.append(
                f"On queries with verified conflicting evidence, Full SAT acknowledged the disagreement "
                f"{e_rate:.0%} of the time vs {b_rate:.0%} for vanilla RAG -- {contra_verdict}."
            )
        clarify = metrics.get("clarify_behavior")
        if clarify:
            findings.append(
                f"Doorman correctly triggered clarification on {clarify['clarify_recall']:.0%} of genuinely "
                f"ambiguous queries, with a {clarify['false_clarify_rate']:.0%} false-clarify rate on answerable "
                f"queries -- a capability the baselines don't have at all (they always attempt an answer, "
                f"even to \"is it good?\")."
            )
        judge = metrics.get("judge_grounding")
        if judge and "B" in judge and "E" in judge:
            b_fail = judge["B"]["n_judge_parse_failures"] / (judge["B"]["n"] + judge["B"]["n_judge_parse_failures"])
            e_fail = judge["E"]["n_judge_parse_failures"] / (judge["E"]["n"] + judge["E"]["n_judge_parse_failures"])
            findings.append(
                f"The secondary LLM-judge check did NOT produce a usable result: {b_fail:.0%}/{e_fail:.0%} "
                f"(B/E) of judgments failed to parse as valid structured output, and among the ones that did "
                f"parse, essentially none were flagged as hallucinated ({judge['B']['hallucination_rate']:.0%}/"
                f"{judge['E']['hallucination_rate']:.1%}). That combination reads as the judge failing to "
                f"discriminate at all -- not as confirmation of near-perfect grounding. Using the same model "
                f"family to judge its own outputs was not reliable here; a genuinely independent judge (a "
                f"different model, or human review) is needed before this check is trustworthy. The objective "
                f"measurements in Section 3 remain the credible hallucination numbers from this evaluation."
            )
        eff = metrics.get("efficiency", {})
        if "B" in eff and "E" in eff and eff["B"]["p50_latency_s"] and eff["E"]["p50_latency_s"]:
            findings.append(
                f"Full SAT's extra Doorman routing step costs latency: p50 {eff['E']['p50_latency_s']:.1f}s vs "
                f"{eff['B']['p50_latency_s']:.1f}s for vanilla RAG. Whether that's worth the hallucination "
                f"reduction depends on the deployment's latency budget."
            )

        text_page(pdf, "7. Findings & What Should Be Fixed Next", findings if findings else ["No findings computed -- check metrics.json."],
                   footer="SAT Evaluation Report -- Findings")

        # ---------------- Limitations ----------------
        text_page(pdf, "8. Limitations (1/2)", [
            ("This is not human evaluation", "No human annotators reviewed the gold evidence labels or the "
             "generated responses. The core hallucination metric (Section 3) is objective and doesn't need "
             "human judgment -- it relies on verified absence of evidence, not opinion -- but persona-alignment "
             "quality, answer completeness, and factuality beyond 'is it in the retrieved evidence' were not "
             "human-checked."),
            ("The LLM-judge is not independent", "The grounding judge (Section 4) is the SAME base model "
             "(adapter disabled) that generated the responses being judged, not a different/stronger model or "
             "a human. A model judging outputs from its own weight family is a real source of bias -- treat "
             "Section 4 as a secondary, exploratory signal, and weight the objective measurements in Section 3 "
             "more heavily."),
            ("Retrieval-gate threshold not yet calibrated", "The evidence-sufficiency gate (MIN_SCORE=0.55, "
             "MIN_EVIDENCE_SENTENCES=3) was carried over from training-data generation, not tuned against this "
             "dev/test split. The dev set (eval_dev.json) now exists specifically to do that calibration "
             "properly -- this report's numbers are the BEFORE, not the final, tuned result."),
        ], footer="SAT Evaluation Report -- Limitations (1/2)")

        text_page(pdf, "8. Limitations (2/2)", [
            ("No full 5-tier ablation", "The report's evaluation plan calls for retrieval / +pruning / "
             "+Analyst / +full-pipeline as separate tiers. This evaluation compares 3 systems (LLM-only, "
             "vanilla RAG, full pipeline), not all 5 -- the individual contribution of pruning vs Analyst vs "
             "abstention-gating separately is not isolated here."),
            ("Contradiction/clarify checks are keyword-heuristic", "Contradiction acknowledgment and abstention "
             "detection use regex pattern matching over the response text, not semantic understanding -- a "
             "response that acknowledges disagreement in an unusual phrasing could be undercounted."),
            ("Single hardware run, no repeated trials", "Latency figures are from one run, batched, on a "
             "single A30 GPU -- not averaged over multiple runs, and not representative of single-request "
             "(unbatched) production latency."),
        ], footer="SAT Evaluation Report -- Limitations (2/2)")

        pdf_info = pdf.infodict()
        pdf_info["Title"] = "SAT Evaluation Report"
        pdf_info["Author"] = "SAT Project"

    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
