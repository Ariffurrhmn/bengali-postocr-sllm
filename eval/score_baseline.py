"""Scores raw (uncorrected) OCR output from a results/ocr_<split>.jsonl file
against ground truth, reporting per-page and mean CER/WER per engine."""
import argparse
import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from metrics import score_pair

REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "eval"], default="dev")
    args = parser.parse_args()

    path = REPO_ROOT / "results" / f"ocr_{args.split}.jsonl"
    records = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]

    engines = ["tesseract", "easyocr"]
    totals = {e: {"cer": [], "wer": []} for e in engines}

    print(f"{'page_id':<30} {'engine':<10} {'CER':>8} {'WER':>8}")
    for r in records:
        for engine in engines:
            scores = score_pair(r["ground_truth"], r[engine])
            totals[engine]["cer"].append(scores["cer"])
            totals[engine]["wer"].append(scores["wer"])
            print(f"{r['page_id']:<30} {engine:<10} {scores['cer']:>8.3f} {scores['wer']:>8.3f}")

    print()
    print("Mean baseline scores:")
    for engine in engines:
        mean_cer = sum(totals[engine]["cer"]) / len(totals[engine]["cer"])
        mean_wer = sum(totals[engine]["wer"]) / len(totals[engine]["wer"])
        print(f"  {engine:<10} mean CER={mean_cer:.3f}  mean WER={mean_wer:.3f}")


if __name__ == "__main__":
    main()
