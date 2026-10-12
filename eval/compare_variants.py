"""Prompt and chunking variants (review items M4, M3) against the original
setting (English prompt, ~250-token chunks), page by page on the same
40 pages, plain decoding, same GPU.

For each model, engine and variant: mean CER; change against raw OCR and
against the original setting, each with a paired bootstrap 95% CI; and a
two-sided Wilcoxon signed-rank test of variant against original,
Holm-corrected over all variant comparisons. Also the mean CER of the
output against its own input (how much the model changed).

Usage:
    python compare_variants.py --ocr-path ../results/ocr_eval40.jsonl \\
        --original ../results/gpu_eval40_ocr.jsonl \\
        --variants ../results/gpu_variants40.jsonl --out-prefix ../results/variants_gpu40
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

# extended_analysis rewraps sys.stdout for UTF-8 on import.
from extended_analysis import DEFAULT_RESAMPLES, DEFAULT_SEED, holm
from metrics import cer

ENGINES = ["tesseract", "easyocr"]
VARIANTS = [("en", "chunked", "original"), ("en_preserve", "chunked", "preserve spelling"),
            ("bn", "chunked", "Bengali prompt"), ("fewshot", "chunked", "three-shot"),
            ("en", "chunked_chars", "300-char chunks")]


def load(path: Path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def boot_ci(d, rng, n):
    idx = rng.integers(0, len(d), (n, len(d)))
    lo, hi = np.percentile(d[idx].mean(axis=1), [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ocr-path", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--variants", type=Path, required=True)
    parser.add_argument("--out-prefix", type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(DEFAULT_SEED)

    pages = {r["page_id"]: r for r in load(args.ocr_path)}
    out = defaultdict(dict)  # (engine, model, prompt, approach) -> page -> output
    for r in load(args.original) + load(args.variants):
        out[(r["engine"], r["model_key"], r.get("prompt", "en"), r["approach"])][r["page_id"]] = r["raw_output"]
    models = list(dict.fromkeys(r["model_key"] for r in load(args.variants)))

    rows, tests = [], []
    for e in ENGINES:
        base = {p: cer(v["ground_truth"], v[e]) for p, v in pages.items()}
        for m in models:
            orig = {p: cer(pages[p]["ground_truth"], o) for p, o in out[(e, m, "en", "chunked")].items()}
            for prompt, approach, label in VARIANTS:
                outs = out[(e, m, prompt, approach)]
                pids = sorted(outs)
                c = np.array([cer(pages[p]["ground_truth"], outs[p]) for p in pids])
                b = np.array([base[p] for p in pids])
                o = np.array([orig[p] for p in pids])
                change = np.mean([cer(pages[p][e], outs[p]) for p in pids])
                row = {"engine": e, "model": m, "variant": label, "prompt": prompt,
                       "approach": approach, "n": len(pids), "mean_cer": float(c.mean()),
                       "vs_raw": boot_ci(c - b, rng, DEFAULT_RESAMPLES),
                       "output_vs_input_cer": float(change)}
                if label != "original":
                    row["vs_original"] = boot_ci(c - o, rng, DEFAULT_RESAMPLES)
                    d = c - o
                    p = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
                    row["wilcoxon_p"] = float(p)
                    tests.append(row)
                rows.append(row)
    adjusted = holm([t["wilcoxon_p"] for t in tests])
    for t, a in zip(tests, adjusted):
        t["wilcoxon_p_holm"] = float(a)

    lines = [f"Variants vs original (English prompt, token chunks); 40 pages; Holm over {len(tests)} tests",
             f"{'engine':<10}{'model':<13}{'variant':<19}{'CER':>7}{'vs raw OCR':>26}"
             f"{'vs original':>26}{'p Holm':>8}{'changed':>9}"]
    for r in rows:
        vr = "%+.3f [%+.3f, %+.3f]" % r["vs_raw"]
        vo = "%+.3f [%+.3f, %+.3f]" % r["vs_original"] if "vs_original" in r else "---"
        ph = "%.3f" % r["wilcoxon_p_holm"] if "wilcoxon_p_holm" in r else "---"
        lines.append(f"{r['engine']:<10}{r['model']:<13}{r['variant']:<19}{r['mean_cer']:>7.3f}"
                     f"{vr:>26}{vo:>26}{ph:>8}{r['output_vs_input_cer']:>9.3f}")
    text = "\n".join(lines)
    print(text)
    args.out_prefix.with_suffix(".txt").write_text(text + "\n", encoding="utf-8")
    args.out_prefix.with_suffix(".json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"\nWrote {args.out_prefix}.txt and .json")


if __name__ == "__main__":
    main()
