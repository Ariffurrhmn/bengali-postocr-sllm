"""Scores the external-review GPU run (Kaggle, results/review4/; review items 4 and 5).

  G   repetition constraints restricted to generated tokens (penalty_gen,
      ngram_gen, guarded_gen) for the four general-purpose models on Tesseract
      text and the clean text, compared with plain decoding (main run) and with
      the same constraint over the whole sequence (results/review3/c4_*).
  P   the two instruction-tuned TituLLMs checkpoints with the plain-text
      completion prompt on Tesseract text, vs the main run and vs raw OCR.

Paired bootstrap (10,000 resamples, seed 403) and Wilcoxon signed-rank as in
review3_runs_analysis.py. Holm families (all exploratory): generated-only vs
plain; generated-only vs whole-sequence; completion prompt vs main.

Usage (from the repo root):
    python eval/review4_runs_analysis.py --runs results/review4 --review3 results/review3 \\
        --ocr-path results/ocr_eval40.jsonl --main results/gpu_eval40.jsonl --out-prefix results/review4_runs
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

# extended_analysis rewraps sys.stdout for UTF-8 on import.
from extended_analysis import DEFAULT_RESAMPLES, DEFAULT_SEED, bootstrap_ci, holm, wilcoxon_p
from metrics import cer

WHOLE = {"guarded_gen": "guarded", "penalty_gen": "penalty_only", "ngram_gen": "ngram_only"}


def load(path):
    # Split on newlines only: outputs can contain U+2028/U+2029 (engineering note 12).
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").split("\n") if l.strip()]


def by_key(rows):
    out = defaultdict(dict)
    for r in rows:
        out[(r["model_key"], r["engine"], r["decoding"], r.get("prompt") or "en")][r["page_id"]] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--review3", type=Path, required=True, help="folder with the whole-sequence c4_*.jsonl runs")
    ap.add_argument("--ocr-path", type=Path, required=True)
    ap.add_argument("--main", type=Path, required=True, help="gpu_eval40.jsonl (plain, main run)")
    ap.add_argument("--out-prefix", type=Path, required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(DEFAULT_SEED)
    pages = {r["page_id"]: r for r in load(args.ocr_path)}
    gt = {p: r["ground_truth"] for p, r in pages.items()}

    def score(cell):
        return {q: cer(gt[q], r["raw_output"]) for q, r in cell.items()}

    def raw(e, qs):
        return {q: 0.0 if e == "ground_truth" else cer(gt[q], pages[q][e]) for q in qs}

    def compare(a, b, label):
        common = sorted(set(a) & set(b))
        d = np.array([a[q] - b[q] for q in common])
        lo, hi = bootstrap_ci(d, rng, DEFAULT_RESAMPLES)
        return {"label": label, "n": len(common), "mean_a": float(np.mean([a[q] for q in common])),
                "mean_b": float(np.mean([b[q] for q in common])), "delta": float(d.mean()),
                "ci": [lo, hi], "p": wilcoxon_p(d), "better": int((d < 0).sum()), "worse": int((d > 0).sum())}

    main_runs = defaultdict(dict)
    for r in load(args.main):
        main_runs[(r["model_key"], r["engine"])][r["page_id"]] = r
    whole = defaultdict(dict)
    for f in sorted(args.review3.glob("c4_*.jsonl")):
        for (m, e, d, _), cell in by_key(load(f)).items():
            whole[(m, e, d)].update(cell)
    gen = defaultdict(dict)
    comp = defaultdict(dict)
    for f in sorted(args.runs.glob("r4_*.jsonl")):
        for (m, e, d, pr), cell in by_key(load(f)).items():
            (comp if pr == "completion" else gen)[(m, e, d) if pr != "completion" else (m, e)].update(cell)

    out = {"G": [], "P": []}
    for (m, e, d) in sorted(gen):
        a = score(gen[(m, e, d)])
        row = {"model": m, "engine": e, "preset": d, "n": len(a),
               "truncated": sum(r["truncated"] for r in gen[(m, e, d)].values()),
               "raw_cer": float(np.mean(list(raw(e, a).values()))),
               "vs_plain": compare(a, score(main_runs[(m, e)]), f"{m} {e}: {d} - plain"),
               "vs_whole": compare(a, score(whole[(m, e, WHOLE[d])]), f"{m} {e}: {d} - {WHOLE[d]}"),
               "vs_raw": compare(a, raw(e, a), f"{m} {e}: {d} - raw OCR")}
        out["G"].append(row)
    for key in ("vs_plain", "vs_whole", "vs_raw"):
        for row, ph in zip(out["G"], holm([r[key]["p"] for r in out["G"]])):
            row[key]["p_holm"] = ph
    for (m, e) in sorted(comp):
        a = score(comp[(m, e)])
        out["P"].append({"model": m, "engine": e, "n": len(a),
                         "truncated": sum(r["truncated"] for r in comp[(m, e)].values()),
                         "empty": sum(not r["raw_output"].strip() for r in comp[(m, e)].values()),
                         "vs_main": compare(a, score(main_runs[(m, e)]), f"{m} {e}: completion - main"),
                         "vs_raw": compare(a, raw(e, a), f"{m} {e}: completion - raw OCR")})
    for key in ("vs_main", "vs_raw"):
        for row, ph in zip(out["P"], holm([r[key]["p"] for r in out["P"]])):
            row[key]["p_holm"] = ph

    lines = ["G: constraints on generated tokens only; Holm within each comparison type over %d tests"
             % len(out["G"])]
    for r in out["G"]:
        lines.append(f"  {r['model']:<12} {r['engine']:<12} {r['preset']:<12} CER {r['vs_plain']['mean_a']:.3f}  "
                     f"plain {r['vs_plain']['mean_b']:.3f} ({r['vs_plain']['delta']:+.3f}, pH {r['vs_plain']['p_holm']:.4f})  "
                     f"whole-seq {r['vs_whole']['mean_b']:.3f} ({r['vs_whole']['delta']:+.3f}, pH {r['vs_whole']['p_holm']:.4f})  "
                     f"trunc {r['truncated']}")
    lines.append("P: completion prompt; Holm over %d tests" % len(out["P"]))
    for r in out["P"]:
        lines.append(f"  {r['model']:<20} {r['engine']:<10} CER {r['vs_main']['mean_a']:.3f}  main "
                     f"{r['vs_main']['mean_b']:.3f} ({r['vs_main']['delta']:+.3f}, pH {r['vs_main']['p_holm']:.4f})  "
                     f"raw {r['vs_raw']['mean_b']:.3f} ({r['vs_raw']['delta']:+.3f}, pH {r['vs_raw']['p_holm']:.4f})  "
                     f"trunc {r['truncated']} empty {r['empty']}")
    print("\n".join(lines))
    args.out_prefix.with_suffix(".txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    args.out_prefix.with_suffix(".json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
