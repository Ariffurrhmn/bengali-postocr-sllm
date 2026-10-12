"""Analyses added for the revision (reviewer checklist of 2026-10-08), run on
existing outputs; no new model runs.

  M8  micro-averaged (pooled) CER next to the per-page mean
  M7  grapheme-cluster CER (regex \\X, which keeps Bengali conjuncts together)
  M6  cluster bootstrap that resamples books, not pages
  E3  baseline CER against corrected CER and against the change (Spearman)
  E7  error rate by character class (consonant, vowel sign, conjunct, ...)
  E8  edit-distance cap: keep the model output only if it changed the input
      by at most tau (CER against the input); tau grid fixed in advance;
      paired bootstrap CI and Wilcoxon, Holm over all engine x method x tau cells
  E1  clean-text control (ground truth as input), when present
  T1  dataset statistics

Book = page_id without its trailing page number (e.g. 279_2_D_32_0006 and
279_2_D_32_0007 are one book).

Usage:
    python revision_analysis.py --ocr-path ../results/ocr_eval40.jsonl \\
        --correction-path ../results/gpu_eval40.jsonl --out-prefix ../results/revision_gpu40
"""
import argparse
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import jiwer
import numpy as np
import regex
from rapidfuzz.distance import Levenshtein
from scipy import stats

# Imported from one module only: extended_analysis rewraps sys.stdout for
# UTF-8 on import, and a second module doing the same would close it.
from extended_analysis import DEFAULT_RESAMPLES, DEFAULT_SEED, MODEL_ORDER, holm
from metrics import cer, normalize

ENGINES = ["tesseract", "easyocr"]
TAU_GRID = [0.05, 0.10, 0.20, 0.30]

CHAR_CLASSES = [
    # (name, test on a single code point)
    ("consonant", lambda c: 0x0995 <= ord(c) <= 0x09B9 or 0x09DC <= ord(c) <= 0x09DF
     or c == "ৎ"),
    ("independent vowel", lambda c: 0x0985 <= ord(c) <= 0x0994),
    ("vowel sign", lambda c: 0x09BE <= ord(c) <= 0x09CC or c == "ৗ"),
    ("virama", lambda c: c == "্"),
    ("other Bengali sign", lambda c: c in "ঁংঃ়"),
    ("Bengali digit", lambda c: 0x09E6 <= ord(c) <= 0x09EF),
    ("danda", lambda c: c in "।॥"),
    ("space", lambda c: c == " "),
]


def char_class(c: str) -> str:
    for name, test in CHAR_CLASSES:
        if test(c):
            return name
    if "a" <= c.lower() <= "z":
        return "Latin letter"
    if c.isdigit():
        return "other digit"
    return "punctuation/other"


def book_of(page_id: str) -> str:
    return re.sub(r"_\d+$", "", page_id)


def collection_of(page_id: str) -> str:
    """Coarser grouping (first two id fields, e.g. 279_42), as a sensitivity check
    in case pages with the same shelfmark prefix share a printer or typeface."""
    return "_".join(page_id.split("_")[:2])


def graphemes(text: str) -> list:
    return regex.findall(r"\X", normalize(text))


def grapheme_cer(ref: str, hyp: str) -> float:
    r, h = graphemes(ref), graphemes(hyp)
    if not r:
        return 0.0 if not h else 1.0
    return Levenshtein.distance(r, h) / len(r)


def char_edits(ref: str, hyp: str):
    """(S + D + I, reference length) after normalisation, for pooled CER."""
    r, h = normalize(ref), normalize(hyp)
    if not r:
        return len(h), 0
    o = jiwer.process_characters(r, h)
    return o.substitutions + o.deletions + o.insertions, len(r)


def ref_char_errors(ref: str, hyp: str):
    """Per reference character: True if it was substituted or deleted."""
    r, h = normalize(ref), normalize(hyp)
    wrong = [False] * len(r)
    if not r:
        return r, wrong
    for chunk in jiwer.process_characters(r, h).alignments[0]:
        if chunk.type in ("substitute", "delete"):
            for i in range(chunk.ref_start_idx, chunk.ref_end_idx):
                wrong[i] = True
    return r, wrong


def cluster_bootstrap(deltas: dict, resamples: int, rng, group=book_of):
    """deltas: page_id -> delta. Resample groups (books) with replacement."""
    books = defaultdict(list)
    for pid, d in deltas.items():
        books[group(pid)].append(d)
    keys = sorted(books)
    observed = float(np.mean(list(deltas.values())))
    means = []
    for _ in range(resamples):
        pick = rng.integers(0, len(keys), len(keys))
        vals = [d for i in pick for d in books[keys[i]]]
        means.append(np.mean(vals))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return observed, float(lo), float(hi), len(keys)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ocr-path", type=Path, required=True)
    parser.add_argument("--correction-path", type=Path, required=True)
    parser.add_argument("--out-prefix", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    args = parser.parse_args()
    rng = np.random.default_rng(DEFAULT_SEED)

    pages = {}
    for line in args.ocr_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            pages[r["page_id"]] = r
    out_by = defaultdict(dict)  # (engine, model) -> page_id -> output
    for line in args.correction_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out_by[(r["engine"], r["model_key"])][r["page_id"]] = r["raw_output"]
    models = [m for m in MODEL_ORDER if all((e, m) in out_by for e in ENGINES)]

    res = {"n_pages": len(pages), "models": models}
    lines = []
    p = lines.append

    # ---- T1: dataset statistics ----
    gt_len = [len(normalize(v["ground_truth"])) for v in pages.values()]
    gt_words = [len(normalize(v["ground_truth"]).split()) for v in pages.values()]
    books = sorted({book_of(pid) for pid in pages})
    t1 = {"pages": len(pages), "books": len(books),
          "gt_chars_median": float(np.median(gt_len)), "gt_chars_min": min(gt_len),
          "gt_chars_max": max(gt_len), "gt_words_median": float(np.median(gt_words)),
          "pages_per_book_max": max(sum(book_of(pid) == b for pid in pages) for b in books)}
    for e in ENGINES:
        ratio = [len(normalize(v[e])) / max(1, len(normalize(v["ground_truth"]))) for v in pages.values()]
        t1[f"{e}_len_ratio_mean"] = float(np.mean(ratio))
        t1[f"{e}_len_ratio_median"] = float(np.median(ratio))
    res["dataset"] = t1
    p("=" * 90)
    p(f"T1 DATASET: {t1['pages']} pages from {t1['books']} books (at most "
      f"{t1['pages_per_book_max']} pages per book)")
    p(f"   ground truth: median {t1['gt_chars_median']:.0f} characters "
      f"(range {t1['gt_chars_min']}-{t1['gt_chars_max']}), median {t1['gt_words_median']:.0f} words")
    for e in ENGINES:
        p(f"   raw {e} length / reference length: mean {t1[f'{e}_len_ratio_mean']:.2f}, "
          f"median {t1[f'{e}_len_ratio_median']:.2f}")

    # ---- M8 + M7: pooled CER and grapheme CER ----
    p("=" * 90)
    p("M8/M7: per-page mean CER, pooled (micro) CER, per-page mean grapheme CER")
    p(f"{'engine':<10}{'model':<22}{'mean CER':>9}{'pooled CER':>11}{'grapheme CER':>13}")
    res["micro"] = []

    def score_row(engine, label, outputs):
        per, ge, edits, chars = [], [], 0, 0
        for pid, out in outputs.items():
            gt = pages[pid]["ground_truth"]
            per.append(cer(gt, out))
            ge.append(grapheme_cer(gt, out))
            e_, n_ = char_edits(gt, out)
            edits += e_
            chars += n_
        row = {"engine": engine, "model": label, "mean_cer": float(np.mean(per)),
               "pooled_cer": edits / chars, "mean_grapheme_cer": float(np.mean(ge)),
               "pages_exact": sum(c == 0 for c in per), "n": len(per)}
        res["micro"].append(row)
        p(f"{engine:<10}{label:<22}{row['mean_cer']:>9.3f}{row['pooled_cer']:>11.3f}"
          f"{row['mean_grapheme_cer']:>13.3f}")

    for e in ENGINES:
        score_row(e, "(raw OCR)", {pid: v[e] for pid, v in pages.items()})
        for m in models:
            score_row(e, m, out_by[(e, m)])
    if any(("ground_truth", m) in out_by for m in models):
        for m in models:
            if ("ground_truth", m) in out_by:
                score_row("clean", m, out_by[("ground_truth", m)])

    # ---- M7: is the grapheme-CER change significant? ----
    p("=" * 90)
    p("M7: change in grapheme CER (corrected - raw OCR): paired bootstrap 95% CI, "
      "Wilcoxon, Holm over all engine x method tests")
    tests = []
    for e in ENGINES:
        for m in models:
            pids = sorted(out_by[(e, m)])
            d = np.array([grapheme_cer(pages[q]["ground_truth"], out_by[(e, m)][q])
                          - grapheme_cer(pages[q]["ground_truth"], pages[q][e]) for q in pids])
            idx = rng.integers(0, len(d), (args.resamples, len(d)))
            lo, hi = np.percentile(d[idx].mean(axis=1), [2.5, 97.5])
            pv = float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
            tests.append({"engine": e, "model": m, "mean_delta": float(d.mean()),
                          "ci": [float(lo), float(hi)], "wilcoxon_p": pv})
    for t, a in zip(tests, holm([t["wilcoxon_p"] for t in tests])):
        t["wilcoxon_p_holm"] = float(a)
        p(f"  {t['engine']:<10}{t['model']:<22}{t['mean_delta']:+.3f} "
          f"[{t['ci'][0]:+.3f}, {t['ci'][1]:+.3f}]  p Holm {a:.4f}")
    res["grapheme_tests"] = tests

    # ---- M6: cluster bootstrap by book ----
    p("=" * 90)
    p("M6: mean CER change (corrected - raw OCR) with page bootstrap vs book (cluster) bootstrap")
    res["cluster_bootstrap"] = []
    for e in ENGINES:
        for m in models:
            d = {pid: cer(pages[pid]["ground_truth"], out) - cer(pages[pid]["ground_truth"], pages[pid][e])
                 for pid, out in out_by[(e, m)].items()}
            vals = np.array(list(d.values()))
            idx = rng.integers(0, len(vals), (args.resamples, len(vals)))
            plo, phi = np.percentile(vals[idx].mean(axis=1), [2.5, 97.5])
            obs, lo, hi, nb = cluster_bootstrap(d, args.resamples, rng)
            _, clo, chi, nc = cluster_bootstrap(d, args.resamples, rng, collection_of)
            res["cluster_bootstrap"].append({"engine": e, "model": m, "mean_delta": obs,
                                             "page_ci": [float(plo), float(phi)],
                                             "book_ci": [lo, hi], "n_books": nb,
                                             "collection_ci": [clo, chi], "n_collections": nc})
            sig = "excl. 0" if lo > 0 or hi < 0 else "spans 0"
            p(f"  {e:<10}{m:<22}{obs:+.3f}  pages [{plo:+.3f}, {phi:+.3f}]  "
              f"books ({nb}) [{lo:+.3f}, {hi:+.3f}] ({sig})  "
              f"collections ({nc}) [{clo:+.3f}, {chi:+.3f}]")

    # ---- E3: baseline vs corrected / change ----
    p("=" * 90)
    p("E3: Spearman rho of baseline CER with corrected CER and with the change")
    res["correlation"] = []
    for e in ENGINES:
        for m in models:
            pids = sorted(out_by[(e, m)])
            base = [cer(pages[pid]["ground_truth"], pages[pid][e]) for pid in pids]
            corr = [cer(pages[pid]["ground_truth"], out_by[(e, m)][pid]) for pid in pids]
            r1, p1 = stats.spearmanr(base, corr)
            r2, p2 = stats.spearmanr(base, np.array(corr) - np.array(base))
            res["correlation"].append({"engine": e, "model": m, "rho_corrected": float(r1),
                                       "p_corrected": float(p1), "rho_change": float(r2),
                                       "p_change": float(p2)})
            p(f"  {e:<10}{m:<22}corrected rho {r1:+.2f} (p {p1:.3f})   change rho {r2:+.2f} (p {p2:.3f})")

    # ---- E8: edit-distance cap ----
    p("=" * 90)
    p("E8: edit-distance cap: keep output only if CER(output vs input) <= tau, else raw OCR")
    p(f"{'engine':<10}{'model':<22}" + "".join(f"{'tau ' + str(t):>12}" for t in TAU_GRID)
      + "   (mean CER; kept pages)")
    res["edit_cap"] = []
    printed = []
    for e in ENGINES:
        for m in models:
            cells = []
            for tau in TAU_GRID:
                scores, deltas, kept = [], [], 0
                for pid, out in out_by[(e, m)].items():
                    src = pages[pid][e]
                    gt = pages[pid]["ground_truth"]
                    use = out if cer(src, out) <= tau else src
                    kept += use is out
                    scores.append(cer(gt, use))
                    deltas.append(cer(gt, use) - cer(gt, src))
                d = np.array(deltas)
                idx = rng.integers(0, len(d), (args.resamples, len(d)))
                lo, hi = np.percentile(d[idx].mean(axis=1), [2.5, 97.5])
                pv = float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
                cells.append((float(np.mean(scores)), kept))
                res["edit_cap"].append({"engine": e, "model": m, "tau": tau,
                                        "mean_cer": float(np.mean(scores)), "kept": kept,
                                        "n": len(scores), "mean_delta": float(d.mean()),
                                        "ci": [float(lo), float(hi)], "wilcoxon_p": pv})
            printed.append(f"  {e:<10}{m:<22}" + "".join(f"{c:>8.3f} ({k:>2})" for c, k in cells))
    # One Holm family over every engine x method x tau cell of the cap.
    for t, a in zip(res["edit_cap"], holm([t["wilcoxon_p"] for t in res["edit_cap"]])):
        t["wilcoxon_p_holm"] = float(a)
    lines.extend(printed)
    sig = [t for t in res["edit_cap"] if t["wilcoxon_p_holm"] < 0.05]
    p(f"  significant after Holm over {len(res['edit_cap'])} tests: "
      + (", ".join(f"{t['engine']}/{t['model']}/tau {t['tau']} {t['mean_delta']:+.3f}"
                   for t in sig) or "none"))

    # ---- E7: error rate by character class ----
    p("=" * 90)
    p("E7: share of reference characters substituted or deleted, by character class "
      "(pooled over pages); 'conjunct' = characters inside a grapheme cluster with a virama")
    res["char_class"] = []

    def class_rates(engine, label, outputs):
        tot, bad = defaultdict(int), defaultdict(int)
        for pid, out in outputs.items():
            r, wrong = ref_char_errors(pages[pid]["ground_truth"], out)
            in_conj = [False] * len(r)
            pos = 0
            for g in regex.findall(r"\X", r):
                if "্" in g and len(g) > 2:
                    for i in range(pos, pos + len(g)):
                        in_conj[i] = True
                pos += len(g)
            for i, c in enumerate(r):
                k = char_class(c)
                tot[k] += 1
                bad[k] += wrong[i]
                if in_conj[i]:
                    tot["conjunct"] += 1
                    bad["conjunct"] += wrong[i]
        row = {"engine": engine, "model": label,
               "rates": {k: bad[k] / tot[k] for k in tot}, "counts": dict(tot)}
        res["char_class"].append(row)
        return row

    order = ["consonant", "vowel sign", "conjunct", "virama", "independent vowel",
             "other Bengali sign", "Bengali digit", "danda", "punctuation/other", "space"]
    p(f"{'engine':<10}{'model':<22}" + "".join(f"{k[:10]:>11}" for k in order))
    for e in ENGINES:
        for label, outs in [("(raw OCR)", {pid: v[e] for pid, v in pages.items()})] + \
                [(m, out_by[(e, m)]) for m in models]:
            row = class_rates(e, label, outs)
            p(f"  {e:<8}{label:<22}" + "".join(
                f"{row['rates'].get(k, float('nan')):>11.2f}" for k in order))
    counts = res["char_class"][0]["counts"]
    p("  reference character counts: " + ", ".join(f"{k} {counts.get(k, 0)}" for k in order))

    prefix = args.out_prefix
    prefix.with_suffix(".json").write_text(json.dumps(res, indent=1, ensure_ascii=False),
                                           encoding="utf-8")
    prefix.with_suffix(".txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {prefix}.json and {prefix}.txt")


if __name__ == "__main__":
    main()
