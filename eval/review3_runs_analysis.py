"""Scores the third-review GPU runs (Kaggle, 2026-10-16; results/review3/).

  C1  precision: first-token check of TituLLMs 3B Instruct in fp16 vs fp32;
      fp32 outputs of TituLLMs 3B Instruct and Phi-4-mini vs the fp16 outputs
      of the main run
  C2  prompt format: TituLLMs 1B with a completion prompt and BanglaT5 with
      normalised input, vs the main run and vs raw OCR
  C4  decoding 2x2: plain (main run), guarded, penalty-only, n-gram-only
  X1  token-level Grad-CAM maps (xai_gradcam.py): shares and source peaks

Paired bootstrap (10,000 resamples, seed 403) and Wilcoxon signed-rank as in
extended_analysis.py. Holm families are stated in the output; all of them
are exploratory (the primary family is the 36 main CER/WER tests).

Usage (from the repo root):
    python eval/review3_runs_analysis.py --runs results/review3 \\
        --ocr-path results/ocr_eval40.jsonl --main results/gpu_eval40.jsonl \\
        --out-prefix results/review3_runs
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

# extended_analysis rewraps sys.stdout for UTF-8 on import.
from extended_analysis import (DEFAULT_RESAMPLES, DEFAULT_SEED, bootstrap_ci, holm, wilcoxon_p)
from metrics import cer, normalize

ENGINES = ["tesseract", "easyocr", "ground_truth"]


def load(path):
    # Split on newlines only: model outputs can contain U+2028/U+2029, which
    # str.splitlines() treats as line breaks inside a JSON string.
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").split("\n") if l.strip()]


def by_key(rows, **filt):
    out = defaultdict(dict)
    for r in rows:
        if all(r.get(k) == v for k, v in filt.items()):
            out[(r["model_key"], r["engine"])][r["page_id"]] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--ocr-path", type=Path, required=True)
    ap.add_argument("--main", type=Path, required=True, help="gpu_eval40.jsonl (plain, main run)")
    ap.add_argument("--out-prefix", type=Path, required=True)
    ap.add_argument("--x1-extra", type=Path, nargs="*", default=[],
                    help="further run folders whose xai_*.jsonl join the X1 rows (review3b)")
    args = ap.parse_args()
    rng = np.random.default_rng(DEFAULT_SEED)
    pages = {r["page_id"]: r for r in load(args.ocr_path)}
    gt = {p: r["ground_truth"] for p, r in pages.items()}
    main_runs = by_key(load(args.main))
    res, lines = {}, []
    p = lines.append

    def score(cell, e):
        """per-page CER of a {page: record} cell against the reference"""
        return {q: cer(gt[q], r["raw_output"]) for q, r in cell.items()}

    def raw_cer(e, q):
        return 0.0 if e == "ground_truth" else cer(gt[q], pages[q][e])

    def compare(a, b, label):
        """paired comparison of two {page: cer} dicts on their common pages (a - b)"""
        common = sorted(set(a) & set(b))
        d = np.array([a[q] - b[q] for q in common])
        lo, hi = bootstrap_ci(d, rng, DEFAULT_RESAMPLES)
        return {"label": label, "n": len(common), "mean_a": float(np.mean([a[q] for q in common])),
                "mean_b": float(np.mean([b[q] for q in common])), "delta": float(d.mean()),
                "ci": [lo, hi], "p": wilcoxon_p(d), "nonzero": int(np.sum(d != 0))}

    def show(c):
        return (f"  {c['label']:<58} n={c['n']:>2}  {c['mean_a']:.3f} vs {c['mean_b']:.3f}  "
                f"delta {c['delta']:+.3f} [{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}]  p {c['p']:.4f}"
                + (f"  p_Holm {c['p_holm']:.4f}" if "p_holm" in c else ""))

    # ---------------- C1 ----------------
    p("C1: precision")
    ft = {prec: load(args.runs / f"first_token_titullm-3b-instruct_{prec}.jsonl") for prec in ("fp16", "fp32")}
    c1 = {"first_token": {}}
    for prec, rows in ft.items():
        mx = max(max(r["hidden_max_abs"]) for r in rows)
        c1["first_token"][prec] = {
            "chunks": len(rows), "logits_nonfinite": sum(not r["logits_finite"] for r in rows),
            "hidden_nonfinite_chunks": sum(not all(r["hidden_finite"]) for r in rows),
            "max_abs_activation": mx, "first_is_eos": sum(r["first_is_eos"] for r in rows),
            "eos_prob_median": float(np.median([r["eos_prob"] for r in rows if r["eos_prob"] is not None]))}
        p(f"  first token {prec}: {c1['first_token'][prec]}")
    k16 = {(r["page_id"], r["engine"], r["chunk"]): r for r in ft["fp16"]}
    agree = [k16[(r["page_id"], r["engine"], r["chunk"])]["first_token"] == r["first_token"]
             for r in ft["fp32"] if (r["page_id"], r["engine"], r["chunk"]) in k16]
    eos_both = [(k16[(r["page_id"], r["engine"], r["chunk"])]["first_is_eos"], r["first_is_eos"])
                for r in ft["fp32"] if (r["page_id"], r["engine"], r["chunk"]) in k16]
    c1["first_token_agreement"] = {"same_first_token": int(sum(agree)), "chunks": len(agree),
                                   "eos_fp16_only": sum(a and not b for a, b in eos_both),
                                   "eos_fp32_only": sum(b and not a for a, b in eos_both),
                                   "eos_both": sum(a and b for a, b in eos_both)}
    p(f"  first-token agreement fp16 vs fp32: {c1['first_token_agreement']}")

    fp32 = by_key(load(args.runs / "c1_fp32.jsonl")) | by_key(load(args.runs / "c1_fp32_phi4.jsonl"))
    c1["outputs"] = []
    for m in ("titullm-3b-instruct", "phi4-mini"):
        for e in ENGINES:
            if (m, e) not in fp32 or not fp32[(m, e)]:
                continue
            a, b = score(fp32[(m, e)], e), score(main_runs[(m, e)], e)
            c = compare(a, b, f"{m} {e}: fp32 - fp16 (CER)")
            c["empty_fp32"] = sum(not r["raw_output"].strip() for r in fp32[(m, e)].values())
            c["empty_fp16"] = sum(not main_runs[(m, e)][q]["raw_output"].strip() for q in fp32[(m, e)])
            c["raw_cer"] = float(np.mean([raw_cer(e, q) for q in a]))
            c["fp32_minus_raw"] = compare(a, {q: raw_cer(e, q) for q in a}, f"{m} {e}: fp32 - raw OCR")
            c1["outputs"].append(c)
            p(show(c) + f"  empty fp32/fp16 {c['empty_fp32']}/{c['empty_fp16']}  raw {c['raw_cer']:.3f}")
            p(show(c["fp32_minus_raw"]))
    res["C1"] = c1

    # ---------------- C2 ----------------
    p("\nC2: prompt format (vs the main run and vs raw OCR)")
    c2 = []
    for fname, m, label in (("c2_completion.jsonl", "titullm-1b", "completion prompt"),
                            ("c2_banglat5_norm.jsonl", "banglat5", "normalised input")):
        runs = by_key(load(args.runs / fname))
        for e in ENGINES:
            a = score(runs[(m, e)], e)
            vs_main = compare(a, score(main_runs[(m, e)], e), f"{m} {e}: {label} - main")
            vs_raw = compare(a, {q: raw_cer(e, q) for q in a}, f"{m} {e}: {label} - raw OCR")
            trunc = sum(r["truncated"] for r in runs[(m, e)].values())
            empty = sum(not r["raw_output"].strip() for r in runs[(m, e)].values())
            c2.append({"model": m, "engine": e, "variant": label, "vs_main": vs_main, "vs_raw": vs_raw,
                       "truncated": trunc, "empty": empty})
            p(show(vs_main) + f"  trunc {trunc}/40 empty {empty}")
            p(show(vs_raw))
    res["C2"] = c2

    # ---------------- C4 ----------------
    p("\nC4: decoding presets vs plain (main run); Holm over all preset-vs-plain tests below")
    presets = ("guarded", "penalty_only", "ngram_only")
    files = sorted(args.runs.glob("c4_*.jsonl"))
    runs = defaultdict(dict)
    for f in files:
        preset = next(pr for pr in presets if f.name.startswith(f"c4_{pr}_"))
        for (m, e), cell in by_key(load(f)).items():
            runs[(m, e, preset)].update(cell)
    c4 = []
    for (m, e, preset) in sorted(runs):
        a = score(runs[(m, e, preset)], e)
        c = compare(a, score(main_runs[(m, e)], e), f"{m} {e}: {preset} - plain")
        c.update({"model": m, "engine": e, "preset": preset,
                  "raw_cer": float(np.mean([raw_cer(e, q) for q in a])),
                  "truncated": sum(r["truncated"] for r in runs[(m, e, preset)].values())})
        c4.append(c)
    for c, ph in zip(c4, holm([c["p"] for c in c4])):
        c["p_holm"] = ph
    for c in c4:
        p(show(c) + f"  raw {c['raw_cer']:.3f}  trunc {c['truncated']}")
    res["C4"] = c4

    # ---------------- X1 ----------------
    p("\nX1: Grad-CAM maps; mean over scored output tokens (mid = 1/2 depth, late = 3/4 depth)")
    x1 = []
    # xai_maps_* files (full maps of two chunks, for a figure) are not part of the table.
    xai_files = [f for d in [args.runs, *args.x1_extra] for f in sorted(d.glob("xai_*.jsonl"))
                 if not f.name.startswith("xai_maps_")]
    for f in xai_files:
        rows = load(f)
        m = rows[0]["model"]
        for e in ("tesseract", "ground_truth"):
            rs = [r for r in rows if r["engine"] == e]
            if not rs:
                continue
            names = dict(zip(rs[0]["layers"], ["embedding", "mid", "late"]))
            # Chance level of the source-peak test: the peak is the argmax over
            # the OCR positions and counts if it is within +-1 of the source,
            # so a random peak hits about 3 / (number of OCR tokens).
            chance = [min(3, r["n_ocr_tokens"]) / r["n_ocr_tokens"]
                      for r in rs for t in r["targets"] if t.get("copy_of") is not None]
            row = {"model": m, "engine": e, "chunks": len(rs),
                   "copy_share": float(np.mean([r["copy_share"] for r in rs])),
                   "tokens": sum(len(r["targets"]) for r in rs),
                   "src_peak_chance": float(np.mean(chance)) if chance else None}
            for li, lname in names.items():
                for kind in ("hires", "literal"):
                    key = f"L{li}_{kind}"
                    vals = [t[key] for r in rs for t in r["targets"] if key in t]
                    if not vals:
                        continue
                    copied = [v for v in vals if "src_peak" in v]
                    row[f"{lname}_{kind}"] = {
                        "ocr": float(np.mean([v["ocr"] for v in vals])),
                        "prompt_other": float(np.mean([v.get("prompt_other", 0.0) for v in vals])),
                        "prefix": float(np.mean([v.get("prefix", 0.0) for v in vals])),
                        "sink": float(np.mean([v["excluded_sink"] for v in vals])),
                        "current": float(np.mean([v["excluded_current"] for v in vals])),
                        "copied_tokens": len(copied),
                        "src_peak": float(np.mean([v["src_peak"] for v in copied])) if copied else None}
            x1.append(row)
            mid = row.get("mid_hires", {})
            p(f"  {m:<22}{e:<13} copy share {row['copy_share']:.2f}  mid-hires: ocr {mid.get('ocr', 0):.2f} "
              f"prompt {mid.get('prompt_other', 0):.2f} own output {mid.get('prefix', 0):.2f}  "
              f"source peak {mid.get('src_peak')}  | literal sink share "
              f"{row.get('mid_literal', {}).get('sink', float('nan')):.2f}, literal source peak "
              f"{row.get('mid_literal', {}).get('src_peak')}")
    res["X1"] = x1

    text = "\n".join(lines)
    print(text)
    args.out_prefix.with_suffix(".txt").write_text(text + "\n", encoding="utf-8")
    args.out_prefix.with_suffix(".json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"\nWrote {args.out_prefix}.txt and .json")


if __name__ == "__main__":
    main()
