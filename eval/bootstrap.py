"""Paired bootstrap resampling over per-page CER/WER deltas.

The methodology commits to this test: for each (engine, model, approach) cell,
resample the evaluation pages with replacement ~10,000 times and report the
mean delta (corrected minus raw-OCR baseline) with a 95% percentile CI.

Pairing matters here: the same page contributes both its baseline score and its
corrected score to every resample, so page-to-page difficulty (some pages OCR
far worse than others) cancels out instead of inflating the interval.
"""
import argparse
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from metrics import score_pair

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESAMPLES = 10000
DEFAULT_SEED = 403  # same convention as data/make_split.py


def load_per_page_deltas(ocr_path: Path, correction_path: Path):
    """Returns (baseline, cells) where baseline[engine] maps page_id -> scores
    and cells[(engine, model, approach)] maps page_id -> scores."""
    ground_truth = {}
    raw_ocr = {}
    for line in ocr_path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        ground_truth[r["page_id"]] = r["ground_truth"]
        raw_ocr[(r["page_id"], "tesseract")] = r["tesseract"]
        raw_ocr[(r["page_id"], "easyocr")] = r["easyocr"]

    baseline = defaultdict(dict)
    for (page_id, engine), text in raw_ocr.items():
        baseline[engine][page_id] = score_pair(ground_truth[page_id], text)

    cells = defaultdict(dict)
    for line in correction_path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        key = (r["engine"], r["model_key"], r["approach"])
        cells[key][r["page_id"]] = score_pair(
            ground_truth[r["page_id"]], r["raw_output"]
        )

    return baseline, cells


def paired_bootstrap(deltas: np.ndarray, resamples: int, rng: np.random.Generator):
    """Resample the per-page deltas with replacement; return observed mean and
    the 95% percentile interval of the resampled means."""
    n = len(deltas)
    idx = rng.integers(0, n, size=(resamples, n))
    means = deltas[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(deltas.mean()), float(lo), float(hi)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "eval"], default="eval")
    parser.add_argument("--ocr-path", type=Path, default=None)
    parser.add_argument("--correction-path", type=Path, default=None)
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Optional path to write the results as JSON (for the figures script)",
    )
    args = parser.parse_args()

    ocr_path = args.ocr_path or (REPO_ROOT / "results" / f"ocr_{args.split}.jsonl")
    correction_path = args.correction_path or (
        REPO_ROOT / "results" / f"correction_{args.split}.jsonl"
    )

    baseline, cells = load_per_page_deltas(ocr_path, correction_path)
    rng = np.random.default_rng(args.seed)

    print(
        f"Paired bootstrap: {args.resamples} resamples, seed={args.seed}, "
        f"split={args.split}"
    )
    print("Delta = corrected minus raw-OCR baseline; negative means correction helped.\n")

    header = (
        f"{'engine':<10} {'model':<14} {'metric':<6} {'base':>7} {'corr':>7} "
        f"{'delta':>8} {'95% CI':>18} {'sig':>5}"
    )
    print(header)
    print("-" * len(header))

    results = []
    for (engine, model_key, approach), page_scores in sorted(cells.items()):
        base_pages = baseline[engine]
        # Pair strictly on page_id so every delta is a like-for-like comparison.
        shared = sorted(set(page_scores) & set(base_pages))
        for metric in ("cer", "wer"):
            corrected = np.array([page_scores[p][metric] for p in shared])
            base = np.array([base_pages[p][metric] for p in shared])
            deltas = corrected - base
            mean_delta, lo, hi = paired_bootstrap(deltas, args.resamples, rng)
            # CI excluding zero == significant at the 5% level.
            significant = lo > 0 or hi < 0
            print(
                f"{engine:<10} {model_key:<14} {metric.upper():<6} "
                f"{base.mean():>7.3f} {corrected.mean():>7.3f} {mean_delta:>+8.3f} "
                f"{'[' + format(lo, '+.3f') + ', ' + format(hi, '+.3f') + ']':>18} "
                f"{('yes' if significant else 'no'):>5}"
            )
            results.append(
                {
                    "engine": engine,
                    "model": model_key,
                    "approach": approach,
                    "metric": metric,
                    "n_pages": len(shared),
                    "baseline_mean": float(base.mean()),
                    "corrected_mean": float(corrected.mean()),
                    "mean_delta": mean_delta,
                    "ci_low": lo,
                    "ci_high": hi,
                    "significant": bool(significant),
                }
            )
        print()

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps(
                {
                    "resamples": args.resamples,
                    "seed": args.seed,
                    "split": args.split,
                    "results": results,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
