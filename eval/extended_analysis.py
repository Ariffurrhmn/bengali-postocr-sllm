"""Extended analysis of the eval sweep, run on the existing outputs (no new
model runs). Each analysis follows a published protocol, cited in the comments:

  A. Significance  - Wilcoxon signed-rank per condition with Holm correction
                     across the family of tests, plus median paired deltas
                     (Viana 2026, Information 17(8):722). The paired bootstrap
                     of the mean is kept alongside it.
  B. cMER          - character Match Error Rate, (S+D+I)/(H+S+D+I), bounded in
                     [0,1] unlike CER, which over-generation pushes above 1
                     (Ehrmann et al. 2026, ICDAR HIPE-OCRepair).
  C. Preference    - per page, did correction improve, leave unchanged, or
                     degrade cMER; reported as counts and their mean sign
                     (Ehrmann et al. 2026).
  D. Word fixes vs - each ground-truth word is aligned to the raw OCR and to the
     damage          corrected output; a correct->incorrect transition is a word
                     the engine got right and the model broke, incorrect->correct
                     is a genuine fix (Kumar et al. 2026, EACL).
  E. Change rate   - CCR = CER(corrected, raw OCR) measures how much the model
                     rewrote; change ratio = CCR / original CER; edit-operation
                     counts; long pure-insertion runs (>= 6 chars) relative to
                     the raw OCR as a reference-free hallucination signal
                     (Koynov & Doan 2025, FedCSIS).
  F. Safeguard     - fall back to the raw OCR when output length is outside
                     0.35x-2.8x the input, the thresholds published by a
                     zero-shot HIPE-OCRepair system (Ehrmann et al. 2026). The
                     thresholds are taken as published, not tuned on this data.
  G. Length        - Spearman correlation of input length with CER change
                     (segment length effect: Danilova & Aangenendt 2025).
  H. Difficulty    - CER change by baseline-CER band (thirds of pages per engine).

Usage:
    python extended_analysis.py            # eval split, writes results/extended_eval.{json,txt}
"""
import argparse
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import jiwer
import numpy as np
from scipy import stats

from metrics import cer, normalize, wer

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESAMPLES = 10000
DEFAULT_SEED = 403  # same convention as data/make_split.py and bootstrap.py

MODEL_ORDER = ["gemma-2b", "qwen3-1.7b", "phi4-mini", "llama3.2-1b", "banglat5", "titullm-1b",
               "titullm-1b-instruct", "titullm-3b-instruct"]
ENGINES = ["tesseract", "easyocr"]

SAFEGUARD_MIN_RATIO = 0.35  # published thresholds (HIPE-OCRepair 2026,
SAFEGUARD_MAX_RATIO = 2.8   # Zakaria-ENSIAS zero-shot system), not tuned here
LONG_INSERT_CHARS = 6       # Koynov & Doan 2025 use k = 6


# ---------------------------------------------------------------------------
# Per-page measurements
# ---------------------------------------------------------------------------

def char_counts(reference: str, hypothesis: str) -> dict:
    """Character-level hits/substitutions/deletions/insertions on normalised
    text. jiwer cannot align against an empty string, so those cases are
    filled in directly."""
    ref, hyp = normalize(reference), normalize(hypothesis)
    if not hyp:
        return {"H": 0, "S": 0, "D": len(ref), "I": 0}
    if not ref:
        return {"H": 0, "S": 0, "D": 0, "I": len(hyp)}
    o = jiwer.process_characters(ref, hyp)
    return {"H": o.hits, "S": o.substitutions, "D": o.deletions, "I": o.insertions}


def cmer_from(c: dict) -> float:
    denom = c["H"] + c["S"] + c["D"] + c["I"]
    return (c["S"] + c["D"] + c["I"]) / denom if denom else 0.0


def correct_ref_words(reference: str, hypothesis: str) -> np.ndarray:
    """Boolean per ground-truth word: True if aligned as an exact match."""
    ref_words = normalize(reference).split()
    flags = np.zeros(len(ref_words), dtype=bool)
    hyp = normalize(hypothesis)
    if not ref_words or not hyp:
        return flags
    o = jiwer.process_words(" ".join(ref_words), hyp)
    for chunk in o.alignments[0]:
        if chunk.type == "equal":
            flags[chunk.ref_start_idx:chunk.ref_end_idx] = True
    return flags


def long_insert_runs(ocr_text: str, corrected: str) -> tuple[int, int]:
    """Runs of >= LONG_INSERT_CHARS characters present in the output but with
    no counterpart in the raw OCR (aligned with the OCR as reference). Returns
    (number of runs, total characters in them)."""
    ref, hyp = normalize(ocr_text), normalize(corrected)
    if not ref or not hyp:
        return (0, 0) if not hyp else (1, len(hyp))
    o = jiwer.process_characters(ref, hyp)
    runs = [c.hyp_end_idx - c.hyp_start_idx for c in o.alignments[0]
            if c.type == "insert" and c.hyp_end_idx - c.hyp_start_idx >= LONG_INSERT_CHARS]
    return len(runs), sum(runs)


def safeguarded(ocr_text: str, corrected: str) -> tuple[str, bool]:
    """Keep the raw OCR when the output length is implausible."""
    n_in, n_out = len(normalize(ocr_text)), len(normalize(corrected))
    if n_in == 0:
        return corrected, False
    ratio = n_out / n_in
    if ratio < SAFEGUARD_MIN_RATIO or ratio > SAFEGUARD_MAX_RATIO:
        return ocr_text, True
    return corrected, False


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def bootstrap_ci(deltas: np.ndarray, rng, resamples: int, stat=np.mean):
    idx = rng.integers(0, len(deltas), size=(resamples, len(deltas)))
    boot = stat(deltas[idx], axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


def wilcoxon_p(deltas: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank on paired deltas. All-zero deltas (no
    change at all) have no defined test, so they return p = 1."""
    if np.allclose(deltas, 0):
        return 1.0
    return float(stats.wilcoxon(deltas, zero_method="wilcox", alternative="two-sided").pvalue)


def holm(pvalues: list[float]) -> list[float]:
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    m = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvalues[i])
        adjusted[i] = min(1.0, running)
    return adjusted.tolist()


def paired_summary(deltas: np.ndarray, rng, resamples: int) -> dict:
    lo, hi = bootstrap_ci(deltas, rng, resamples)
    mlo, mhi = bootstrap_ci(deltas, rng, resamples, stat=np.median)
    return {
        "mean_delta": float(deltas.mean()),
        "ci_low": lo,
        "ci_high": hi,
        "median_delta": float(np.median(deltas)),
        "median_ci_low": mlo,
        "median_ci_high": mhi,
        "wilcoxon_p": wilcoxon_p(deltas),
    }


# ---------------------------------------------------------------------------

def load(ocr_path: Path, correction_path: Path):
    pages = {}
    for line in ocr_path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        pages[r["page_id"]] = r
    cells = defaultdict(dict)
    for line in correction_path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        cells[(r["engine"], r["model_key"])][r["page_id"]] = r
    return pages, cells


def page_measures(gt: str, ocr_text: str, corrected: str) -> dict:
    base_c, corr_c = char_counts(gt, ocr_text), char_counts(gt, corrected)
    ocr_ok, corr_ok = correct_ref_words(gt, ocr_text), correct_ref_words(gt, corrected)
    runs, run_chars = long_insert_runs(ocr_text, corrected)
    guarded_text, fell_back = safeguarded(ocr_text, corrected)
    base_cer = cer(gt, ocr_text)
    ccr = cer(ocr_text, corrected) if normalize(ocr_text) else 0.0
    return {
        "base_cer": base_cer,
        "corr_cer": cer(gt, corrected),
        "base_wer": wer(gt, ocr_text),
        "corr_wer": wer(gt, corrected),
        "base_cmer": cmer_from(base_c),
        "corr_cmer": cmer_from(corr_c),
        "base_ops": base_c,
        "corr_ops": corr_c,
        "words_total": int(len(ocr_ok)),
        "kept_correct": int((ocr_ok & corr_ok).sum()),
        "broke_correct": int((ocr_ok & ~corr_ok).sum()),
        "fixed": int((~ocr_ok & corr_ok).sum()),
        "still_wrong": int((~ocr_ok & ~corr_ok).sum()),
        "ccr": ccr,
        "change_ratio": ccr / base_cer if base_cer else float("nan"),
        "long_insert_runs": runs,
        "long_insert_chars": run_chars,
        "corr_len": len(normalize(corrected)),
        "input_len": len(normalize(ocr_text)),
        "fell_back": fell_back,
        "guard_cer": cer(gt, guarded_text),
        "guard_cmer": cmer_from(char_counts(gt, guarded_text)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "eval"], default="eval")
    parser.add_argument("--ocr-path", type=Path, default=None)
    parser.add_argument("--correction-path", type=Path, default=None)
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out-prefix", type=Path, default=None,
                        help="Writes <prefix>.json and <prefix>.txt "
                        "(default: results/extended_<split>)")
    args = parser.parse_args()

    ocr_path = args.ocr_path or (REPO_ROOT / "results" / f"ocr_{args.split}.jsonl")
    correction_path = args.correction_path or (
        REPO_ROOT / "results" / f"correction_{args.split}.jsonl")
    prefix = args.out_prefix or (REPO_ROOT / "results" / f"extended_{args.split}")

    pages, cells = load(ocr_path, correction_path)
    # Only the models this results file has (the guarded runs have no Qwen3).
    global MODEL_ORDER
    MODEL_ORDER = [m for m in MODEL_ORDER if all((e, m) in cells for e in ENGINES)]
    rng = np.random.default_rng(args.seed)

    # Per-page measurements for every engine x model cell.
    measures = {}
    for engine in ENGINES:
        for model in MODEL_ORDER:
            rows = cells.get((engine, model), {})
            measures[(engine, model)] = {
                pid: page_measures(pages[pid]["ground_truth"], pages[pid][engine],
                                   rows[pid]["raw_output"])
                for pid in sorted(rows)
            }

    out = {"split": args.split, "resamples": args.resamples, "seed": args.seed,
           "n_pages": len(pages), "cells": []}
    lines = []
    p = lines.append

    # ---- A + B: significance on CER, WER (the paper's 16 tests) and cMER ----
    tests = []  # (family, engine, model, metric, summary)
    for engine in ENGINES:
        for model in MODEL_ORDER:
            m = measures[(engine, model)]
            for metric in ("cer", "wer", "cmer"):
                d = np.array([v[f"corr_{metric}"] - v[f"base_{metric}"] for v in m.values()])
                summ = paired_summary(d, rng, args.resamples)
                summ["baseline_mean"] = float(np.mean([v[f"base_{metric}"] for v in m.values()]))
                summ["corrected_mean"] = float(np.mean([v[f"corr_{metric}"] for v in m.values()]))
                family = "cer_wer" if metric in ("cer", "wer") else "cmer"
                tests.append((family, engine, model, metric, summ))
    for family in ("cer_wer", "cmer"):
        idx = [i for i, t in enumerate(tests) if t[0] == family]
        adj = holm([tests[i][4]["wilcoxon_p"] for i in idx])
        for i, a in zip(idx, adj):
            tests[i][4]["wilcoxon_p_holm"] = a
            s = tests[i][4]
            s["significant_holm"] = bool(a < 0.05)
            s["significant_bootstrap"] = bool(s["ci_low"] > 0 or s["ci_high"] < 0)

    p("=" * 100)
    p("A/B. SIGNIFICANCE: paired bootstrap of the mean + Wilcoxon signed-rank, Holm-corrected")
    p("     Delta = corrected - raw OCR (positive = correction made it worse). n = 15 pages per cell.")
    n_cw = 2 * len(ENGINES) * len(MODEL_ORDER)
    p(f"     Holm families: the paper's {n_cw} CER/WER tests; the {n_cw // 2} cMER tests separately.")
    p("=" * 100)
    p(f"{'engine':<10}{'model':<13}{'metric':<6}{'base':>7}{'corr':>7}{'mean d':>8}"
      f"{'95% CI (mean)':>19}{'median d':>9}{'p Wilcoxon':>12}{'p Holm':>9}{'sig':>5}")
    for family, engine, model, metric, s in tests:
        p(f"{engine:<10}{model:<13}{metric.upper():<6}{s['baseline_mean']:>7.3f}{s['corrected_mean']:>7.3f}"
          f"{s['mean_delta']:>+8.3f}  [{s['ci_low']:+.3f}, {s['ci_high']:+.3f}]"
          f"{s['median_delta']:>+9.3f}{s['wilcoxon_p']:>12.5f}{s['wilcoxon_p_holm']:>9.4f}"
          f"{('yes' if s['significant_holm'] else 'no'):>5}")
    n_sig = sum(t[4]["significant_holm"] for t in tests if t[0] == "cer_wer")
    n_pos = sum(t[4]["mean_delta"] > 0 for t in tests if t[0] == "cer_wer")
    p(f"\nCER/WER family: {n_pos}/{n_cw} deltas positive; {n_sig}/{n_cw} significant after Holm (Wilcoxon).")
    n_sig_c = sum(t[4]["significant_holm"] for t in tests if t[0] == "cmer")
    p(f"cMER family:    {n_sig_c}/{n_cw // 2} significant after Holm (Wilcoxon).")
    p("Note: with n = 15 and every page in the same direction, the smallest possible exact two-sided")
    p("Wilcoxon p is 2/2^15 = 0.000061; Holm multiplies the smallest p by the family size.")

    # ---- C, D, E per cell ----
    p("\n" + "=" * 100)
    p("C. PREFERENCE (cMER per page): improved / unchanged / degraded, and mean sign (+1 = all improved)")
    p("D. WORD FIXES vs DAMAGE: ground-truth words the raw OCR had right that the model broke, vs words fixed")
    p("E. CHANGE: CCR = CER(output vs raw OCR); change ratio = CCR / original CER; long inserted runs")
    p("=" * 100)
    p(f"{'engine':<10}{'model':<13}{'impr':>5}{'same':>5}{'degr':>5}{'pref':>7}"
      f"{'OCR-correct':>12}{'broken':>8}{'%broken':>8}{'OCR-wrong':>10}{'fixed':>7}{'%fixed':>7}"
      f"{'CCR':>7}{'ratio':>7}{'ins-runs/pg':>12}{'ins-chars%':>11}")
    cell_records = []
    for engine in ENGINES:
        for model in MODEL_ORDER:
            m = list(measures[(engine, model)].values())
            signs = [float(np.sign(v["base_cmer"] - v["corr_cmer"])) for v in m]
            impr, same, degr = (int(sum(s > 0 for s in signs)), int(sum(s == 0 for s in signs)),
                                int(sum(s < 0 for s in signs)))
            cer_signs = [float(np.sign(v["base_cer"] - v["corr_cer"])) for v in m]
            cer_impr = int(sum(s > 0 for s in cer_signs))
            ocr_correct = sum(v["kept_correct"] + v["broke_correct"] for v in m)
            ocr_wrong = sum(v["fixed"] + v["still_wrong"] for v in m)
            broke, fixed = sum(v["broke_correct"] for v in m), sum(v["fixed"] for v in m)
            ccr = float(np.mean([v["ccr"] for v in m]))
            ratio = float(np.nanmean([v["change_ratio"] for v in m]))
            runs = float(np.mean([v["long_insert_runs"] for v in m]))
            ins_share = (sum(v["long_insert_chars"] for v in m) / max(1, sum(v["corr_len"] for v in m)))
            ops_base = {k: float(np.mean([v["base_ops"][k] for v in m])) for k in "SDI"}
            ops_corr = {k: float(np.mean([v["corr_ops"][k] for v in m])) for k in "SDI"}
            rec = {
                "engine": engine, "model": model,
                "pref_improved": impr, "pref_unchanged": same, "pref_degraded": degr,
                "pref_score": float(np.mean(signs)),
                "cer_pages_improved": cer_impr,
                "words_ocr_correct": ocr_correct, "words_broken": broke,
                "pct_broken": broke / ocr_correct if ocr_correct else 0.0,
                "words_ocr_wrong": ocr_wrong, "words_fixed": fixed,
                "pct_fixed": fixed / ocr_wrong if ocr_wrong else 0.0,
                "mean_ccr": ccr, "mean_change_ratio": ratio,
                "long_insert_runs_per_page": runs, "long_insert_char_share": ins_share,
                "edit_ops_vs_gt_raw_ocr": ops_base, "edit_ops_vs_gt_corrected": ops_corr,
            }
            cell_records.append(rec)
            p(f"{engine:<10}{model:<13}{impr:>5}{same:>5}{degr:>5}{rec['pref_score']:>+7.2f}"
              f"{ocr_correct:>12}{broke:>8}{rec['pct_broken']:>8.1%}{ocr_wrong:>10}{fixed:>7}{rec['pct_fixed']:>7.1%}"
              f"{ccr:>7.2f}{ratio:>7.2f}{runs:>12.2f}{ins_share:>11.1%}")
    improved = [f"{r['engine']}/{r['model']} {r['cer_pages_improved']}"
                for r in cell_records if r["cer_pages_improved"]]
    p("\nPages where CER (rather than cMER) improved: " + (", ".join(improved) or "none"))
    p("\nMean edit operations per page against the ground truth (S / D / I), raw OCR -> corrected:")
    for rec in cell_records:
        b, c = rec["edit_ops_vs_gt_raw_ocr"], rec["edit_ops_vs_gt_corrected"]
        p(f"  {rec['engine']:<10}{rec['model']:<13} S {b['S']:6.1f} -> {c['S']:6.1f}   "
          f"D {b['D']:6.1f} -> {c['D']:6.1f}   I {b['I']:6.1f} -> {c['I']:6.1f}")

    # ---- F: safeguard ----
    p("\n" + "=" * 100)
    p(f"F. POST-HOC SAFEGUARD: keep raw OCR if output length is outside "
      f"{SAFEGUARD_MIN_RATIO}x-{SAFEGUARD_MAX_RATIO}x the input (published thresholds, not tuned)")
    p("=" * 100)
    p(f"{'engine':<10}{'model':<13}{'fallbacks':>10}{'CER base':>9}{'CER corr':>9}{'CER guard':>10}"
      f"{'d guard':>8}{'95% CI':>19}{'p Holm':>9}{'cMER guard':>11}")
    guard_tests = []
    for engine in ENGINES:
        for model in MODEL_ORDER:
            m = list(measures[(engine, model)].values())
            d = np.array([v["guard_cer"] - v["base_cer"] for v in m])
            s = paired_summary(d, rng, args.resamples)
            s.update({
                "engine": engine, "model": model,
                "fallbacks": int(sum(v["fell_back"] for v in m)),
                "cer_base": float(np.mean([v["base_cer"] for v in m])),
                "cer_corr": float(np.mean([v["corr_cer"] for v in m])),
                "cer_guard": float(np.mean([v["guard_cer"] for v in m])),
                "cmer_guard": float(np.mean([v["guard_cmer"] for v in m])),
                "cmer_base": float(np.mean([v["base_cmer"] for v in m])),
            })
            guard_tests.append(s)
    for s, a in zip(guard_tests, holm([g["wilcoxon_p"] for g in guard_tests])):
        s["wilcoxon_p_holm"] = a
        p(f"{s['engine']:<10}{s['model']:<13}{s['fallbacks']:>7}/15{s['cer_base']:>9.3f}{s['cer_corr']:>9.3f}"
          f"{s['cer_guard']:>10.3f}{s['mean_delta']:>+8.3f}  [{s['ci_low']:+.3f}, {s['ci_high']:+.3f}]"
          f"{a:>9.4f}{s['cmer_guard']:>11.3f}")
    p("A positive d guard means the safeguarded output is still worse than doing nothing.")

    # ---- G: input length vs damage ----
    p("\n" + "=" * 100)
    p("G. INPUT LENGTH vs DAMAGE: Spearman rho between raw-OCR length (chars) and CER change, per cell")
    p("=" * 100)
    length_records = []
    for engine in ENGINES:
        for model in MODEL_ORDER:
            m = list(measures[(engine, model)].values())
            x = [v["input_len"] for v in m]
            y = [v["corr_cer"] - v["base_cer"] for v in m]
            rho, pv = stats.spearmanr(x, y)
            length_records.append({"engine": engine, "model": model,
                                   "spearman_rho": float(rho), "p": float(pv)})
            p(f"  {engine:<10}{model:<13} rho = {rho:+.2f}  (p = {pv:.3f}, uncorrected)")
    lens = [v["input_len"] for v in measures[("tesseract", MODEL_ORDER[0])].values()]
    lens_e = [v["input_len"] for v in measures[("easyocr", MODEL_ORDER[0])].values()]
    p(f"  Raw-OCR page length, chars: Tesseract median {np.median(lens):.0f} "
      f"(range {min(lens)}-{max(lens)}); EasyOCR median {np.median(lens_e):.0f} "
      f"(range {min(lens_e)}-{max(lens_e)})")

    # ---- H: difficulty bands ----
    p("\n" + "=" * 100)
    p("H. CER CHANGE BY PAGE DIFFICULTY (pages split into thirds by baseline CER, per engine; 5 pages each)")
    p("=" * 100)
    band_records = []
    for engine in ENGINES:
        base = {pid: v["base_cer"] for pid, v in measures[(engine, MODEL_ORDER[0])].items()}
        ordered = sorted(base, key=base.get)
        bands = {"easy": ordered[:5], "medium": ordered[5:10], "hard": ordered[10:]}
        desc = ", ".join(f"{b} {min(base[x] for x in ids):.2f}-{max(base[x] for x in ids):.2f}"
                         for b, ids in bands.items())
        p(f"  {engine}: baseline CER bands -> {desc}")
        for model in MODEL_ORDER:
            m = measures[(engine, model)]
            row = {b: float(np.mean([m[x]["corr_cer"] - m[x]["base_cer"] for x in ids]))
                   for b, ids in bands.items()}
            band_records.append({"engine": engine, "model": model, **row})
            p(f"    {model:<13} mean CER change: easy {row['easy']:+.3f}  medium {row['medium']:+.3f}  "
              f"hard {row['hard']:+.3f}")

    out["significance"] = [
        {"family": f, "engine": e, "model": mo, "metric": me, **s} for f, e, mo, me, s in tests]
    out["cells"] = cell_records
    out["safeguard"] = guard_tests
    out["length_vs_damage"] = length_records
    out["difficulty_bands"] = band_records

    text = "\n".join(lines)
    print(text)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(json.dumps(out, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    prefix.with_suffix(".txt").write_text(text + "\n", encoding="utf-8")
    print(f"\nWrote {prefix.with_suffix('.json')} and {prefix.with_suffix('.txt')}")


if __name__ == "__main__":
    main()
