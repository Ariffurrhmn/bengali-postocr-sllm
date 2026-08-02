"""Scores corrected output from results/correction_<split>.jsonl against
ground truth, comparing each (engine, model, approach) cell's mean CER/WER
to the raw OCR baseline — i.e. the actual question this project asks."""
import argparse
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from metrics import score_pair

REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "eval"], default="dev")
    parser.add_argument(
        "--correction-path",
        type=Path,
        default=None,
        help="Override for the correction results file (e.g. a Drive-mounted "
        "path when run_sweep.py wrote there instead of the repo's results/ dir)",
    )
    parser.add_argument(
        "--ocr-path",
        type=Path,
        default=None,
        help="Override for the input OCR results file (e.g. a Drive-mounted "
        "path when run_ocr.py wrote there instead of the repo's results/ dir)",
    )
    args = parser.parse_args()

    ocr_path = args.ocr_path or (REPO_ROOT / "results" / f"ocr_{args.split}.jsonl")
    correction_path = args.correction_path or (
        REPO_ROOT / "results" / f"correction_{args.split}.jsonl"
    )

    ground_truth = {}
    raw_ocr = {}
    for line in ocr_path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        ground_truth[r["page_id"]] = r["ground_truth"]
        raw_ocr[(r["page_id"], "tesseract")] = r["tesseract"]
        raw_ocr[(r["page_id"], "easyocr")] = r["easyocr"]

    correction_records = [
        json.loads(l) for l in correction_path.read_text(encoding="utf-8").splitlines()
    ]

    # baseline (raw OCR, no correction) per engine
    baseline = defaultdict(lambda: {"cer": [], "wer": []})
    for (page_id, engine), text in raw_ocr.items():
        gt = ground_truth[page_id]
        scores = score_pair(gt, text)
        baseline[engine]["cer"].append(scores["cer"])
        baseline[engine]["wer"].append(scores["wer"])

    # corrected, grouped by (engine, model, approach)
    cells = defaultdict(lambda: {"cer": [], "wer": [], "skipped": 0, "truncated": 0})
    for r in correction_records:
        gt = ground_truth[r["page_id"]]
        scores = score_pair(gt, r["raw_output"])
        key = (r["engine"], r["model_key"], r["approach"])
        cells[key]["cer"].append(scores["cer"])
        cells[key]["wer"].append(scores["wer"])
        if r.get("skipped"):
            cells[key]["skipped"] += 1
        if r.get("truncated"):
            cells[key]["truncated"] += 1

    print("Baseline (raw OCR, no correction):")
    for engine, vals in baseline.items():
        mean_cer = sum(vals["cer"]) / len(vals["cer"])
        mean_wer = sum(vals["wer"]) / len(vals["wer"])
        print(f"  {engine:<10} mean CER={mean_cer:.3f}  mean WER={mean_wer:.3f}  (n={len(vals['cer'])})")

    print()
    print(f"{'engine':<10} {'model':<14} {'approach':<9} {'CER':>8} {'WER':>8} {'ΔCER':>8}  n  skip trunc")
    for (engine, model_key, approach), vals in sorted(cells.items()):
        mean_cer = sum(vals["cer"]) / len(vals["cer"])
        mean_wer = sum(vals["wer"]) / len(vals["wer"])
        base_cer = sum(baseline[engine]["cer"]) / len(baseline[engine]["cer"])
        delta = mean_cer - base_cer
        print(
            f"{engine:<10} {model_key:<14} {approach:<9} {mean_cer:>8.3f} {mean_wer:>8.3f} "
            f"{delta:>+8.3f}  {len(vals['cer']):<3}{vals['skipped']:<5}{vals['truncated']}"
        )


if __name__ == "__main__":
    main()
