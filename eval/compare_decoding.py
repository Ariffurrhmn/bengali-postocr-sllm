"""Decoding ablation (review item C1): compares the paper's "guarded" runs
(repetition_penalty 1.3 + no_repeat_ngram_size 4) against "plain" reruns of
the same (page, engine, model, approach), page by page, on whatever pages
the plain file has so far.

Usage:
    python compare_decoding.py --ocr-path results/ocr_eval.jsonl \
        --guarded results/correction_eval.jsonl \
        --plain results/ablation_plain_decoding.jsonl
"""
import argparse
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from metrics import score_pair


def read_jsonl(path: Path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ocr-path", type=Path, required=True)
    parser.add_argument("--guarded", type=Path, required=True)
    parser.add_argument("--plain", type=Path, required=True)
    args = parser.parse_args()

    ocr = {r["page_id"]: r for r in read_jsonl(args.ocr_path)}
    key = lambda r: (r["page_id"], r["engine"], r["model_key"], r["approach"])
    guarded = {key(r): r for r in read_jsonl(args.guarded)}

    by_cell = defaultdict(list)
    for r in read_jsonl(args.plain):
        if key(r) not in guarded:
            continue
        page = ocr[r["page_id"]]
        gt = page["ground_truth"]
        base = score_pair(gt, page[r["engine"]])["cer"]
        old = score_pair(gt, guarded[key(r)]["raw_output"])["cer"]
        new = score_pair(gt, r["raw_output"])["cer"]
        by_cell[(r["engine"], r["model_key"], r["approach"])].append(
            (r["page_id"], base, old, new, r["truncated"])
        )

    for (engine, model, approach), rows in sorted(by_cell.items()):
        print(f"\n{engine} / {model} / {approach}  (n={len(rows)})")
        print(f"  {'page':<20} {'OCR':>6} {'guarded':>8} {'plain':>6}")
        for page_id, base, old, new, trunc in rows:
            print(f"  {page_id:<20} {base:>6.3f} {old:>8.3f} {new:>6.3f}{'  truncated' if trunc else ''}")
        n = len(rows)
        mean = lambda i: sum(r[i] for r in rows) / n
        print(f"  {'MEAN':<20} {mean(1):>6.3f} {mean(2):>8.3f} {mean(3):>6.3f}")
        better_than_ocr = sum(r[3] < r[1] for r in rows)
        print(f"  plain beats raw OCR on {better_than_ocr}/{n} pages; "
              f"plain beats guarded on {sum(r[3] < r[2] for r in rows)}/{n}")


if __name__ == "__main__":
    main()
