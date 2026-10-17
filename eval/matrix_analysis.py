"""Corrected-versus-input matrix for the later third-review runs: one row per
(input, method), every method compared with its own input page by page.

Used for
  C5  region-level OCR (review3c): inputs = Tesseract and EasyOCR region OCR
      (ocr_region40.jsonl); methods = the eight models and SymSpell
  M1  new models (review3b): inputs = Tesseract, EasyOCR and the clean text
      (ocr_eval40.jsonl)
  M3  noise sweep (review3b): inputs = the references noised at CER
      0.02-0.30 (noise40.jsonl, eval/make_noise.py)

Per cell: mean input and output CER, WER and cMER, the mean change with a
paired-bootstrap 95% interval (10,000 resamples, seed 403), the Wilcoxon
signed-rank p (zero differences dropped, as in the main analysis) with Holm
correction over all cells of the file for each metric (an exploratory family),
pages better/worse/unchanged, correct words broken and wrong words fixed
(pooled over pages), the change ratio (median), truncated and empty outputs.
The per-page measures are those of extended_analysis.page_measures.

Usage (from eval/):
    python matrix_analysis.py --ocr-path ../results/region/ocr_region40.jsonl \\
        --corrections ../results/review3c/c5_*.jsonl ../results/region/dict_region40.jsonl \\
        --engines tesseract,easyocr --label "region-level OCR" --out-prefix ../results/region_runs40
"""
import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

# extended_analysis rewraps sys.stdout for UTF-8 on import.
from extended_analysis import DEFAULT_RESAMPLES, DEFAULT_SEED, bootstrap_ci, holm, page_measures, wilcoxon_p
from metrics import normalize

METRICS = [("cer", "base_cer", "corr_cer"), ("wer", "base_wer", "corr_wer"), ("cmer", "base_cmer", "corr_cmer")]


def load(path):
    # Split on newlines only: outputs can contain U+2028/U+2029 (engineering note 12).
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").split("\n") if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocr-path", type=Path, required=True, help="inputs: page_id, ground_truth, one field per input")
    ap.add_argument("--corrections", nargs="+", required=True, help="run_sweep.py outputs (globs allowed)")
    ap.add_argument("--engines", required=True, help="comma-separated input fields")
    ap.add_argument("--label", default="")
    ap.add_argument("--pages-file", type=Path, default=None,
                    help="restrict to these page ids (one per line), e.g. data/m7_pages.txt")
    ap.add_argument("--out-prefix", type=Path, required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(DEFAULT_SEED)
    pages = {r["page_id"]: r for r in load(args.ocr_path)}
    if args.pages_file:
        keep = set(args.pages_file.read_text(encoding="utf-8").split())
        pages = {p: r for p, r in pages.items() if p in keep}
    engines = args.engines.split(",")
    files = sorted({f for pattern in args.corrections for f in glob.glob(pattern)})
    cells = defaultdict(dict)
    for f in files:
        for r in load(f):
            if r["engine"] in engines and r["page_id"] in pages:
                cells[(r["engine"], r["model_key"])][r["page_id"]] = r

    rows = []
    for (e, m), cell in sorted(cells.items(), key=lambda kv: (engines.index(kv[0][0]), kv[0][1])):
        ids = sorted(cell)
        meas = [page_measures(pages[p]["ground_truth"], pages[p][e], cell[p]["raw_output"]) for p in ids]
        row = {"engine": e, "model": m, "n": len(ids), "complete": len(ids) == len(pages),
               "truncated": sum(bool(cell[p].get("truncated")) for p in ids),
               "empty": sum(not normalize(cell[p]["raw_output"]) for p in ids),
               "errors": sum(bool(cell[p].get("error")) for p in ids)}
        for name, b, c in METRICS:
            d = np.array([x[c] - x[b] for x in meas])
            lo, hi = bootstrap_ci(d, rng, DEFAULT_RESAMPLES)
            row[name] = {"input": float(np.mean([x[b] for x in meas])), "output": float(np.mean([x[c] for x in meas])),
                         "delta": float(d.mean()), "ci": [lo, hi], "p": wilcoxon_p(d),
                         "better": int((d < 0).sum()), "worse": int((d > 0).sum()), "same": int((d == 0).sum())}
        ok = sum(x["kept_correct"] + x["broke_correct"] for x in meas)
        wrong = sum(x["fixed"] + x["still_wrong"] for x in meas)
        row["broken_share"] = sum(x["broke_correct"] for x in meas) / ok if ok else float("nan")
        row["fixed_share"] = sum(x["fixed"] for x in meas) / wrong if wrong else float("nan")
        ratios = [x["change_ratio"] for x in meas if not np.isnan(x["change_ratio"])]
        row["change_ratio_median"] = float(np.median(ratios)) if ratios else float("nan")
        rows.append(row)
    for name, _, _ in METRICS:
        for r, ph in zip(rows, holm([r[name]["p"] for r in rows])):
            r[name]["p_holm"] = ph

    lines = [f"{args.label}: {len(rows)} cells, Holm over the {len(rows)} cells per metric (exploratory); "
             f"files: {', '.join(Path(f).name for f in files)}"]
    for r in rows:
        c, w = r["cer"], r["wer"]
        lines.append(
            f"  {r['engine']:<13} {r['model']:<20} n={r['n']:>2}{'' if r['complete'] else ' (INCOMPLETE)'}  "
            f"CER {c['input']:.3f} -> {c['output']:.3f}  {c['delta']:+.3f} [{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}]  "
            f"pHolm {c['p_holm']:.4f}  better/worse {c['better']}/{c['worse']}  |  WER {w['delta']:+.3f} "
            f"pHolm {w['p_holm']:.4f}  |  broken {r['broken_share']:.3f} fixed {r['fixed_share']:.3f}  "
            f"trunc {r['truncated']} empty {r['empty']}" + (f" ERRORS {r['errors']}" if r["errors"] else ""))
    text = "\n".join(lines)
    print(text)
    args.out_prefix.with_suffix(".txt").write_text(text + "\n", encoding="utf-8")
    args.out_prefix.with_suffix(".json").write_text(
        json.dumps({"label": args.label, "files": [Path(f).name for f in files], "cells": rows}, indent=1),
        encoding="utf-8")


if __name__ == "__main__":
    main()
