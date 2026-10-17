"""Builds the worked examples for the few-shot prompt (review item M4) from
the development pages only, so no evaluation page leaks into the prompt.

Selection is deterministic: development pages in sorted order, engines
alternating Tesseract, EasyOCR, Tesseract; on each page, the first reference
line of 30-90 characters whose best-matching OCR line (lowest CER) has a CER
between 0.05 and 0.30 and contains no Latin letters. One example per page,
three in total. Writes data/fewshot_examples.json, which is gitignored like
the rest of the dataset text.

Usage:
    python data/make_fewshot.py --ocr-dev results/ocr_dev.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
from metrics import cer, normalize  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

N_EXAMPLES = 3
ENGINE_CYCLE = ["tesseract", "easyocr", "tesseract"]


def pick(page: dict, engine: str):
    ocr_lines = [l for l in page[engine].split("\n") if l.strip()]
    for ref in [l for l in page["ground_truth"].split("\n") if l.strip()]:
        ref = normalize(ref)
        if not 30 <= len(ref) <= 90 or not ocr_lines:
            continue
        best = min(ocr_lines, key=lambda o: cer(ref, o))
        score = cer(ref, best)
        if 0.05 <= score <= 0.30 and not any("a" <= c.lower() <= "z" for c in best):
            return {"page_id": page["page_id"], "engine": engine, "ocr": normalize(best),
                    "corrected": ref, "cer": round(score, 3)}
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ocr-dev", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "fewshot_examples.json")
    args = parser.parse_args()

    pages = sorted((json.loads(l) for l in args.ocr_dev.read_text(encoding="utf-8").splitlines()
                    if l.strip()), key=lambda p: p["page_id"])
    examples = []
    for page in pages:
        ex = pick(page, ENGINE_CYCLE[len(examples)])
        if ex:
            examples.append(ex)
        if len(examples) == N_EXAMPLES:
            break
    assert len(examples) == N_EXAMPLES, f"only {len(examples)} examples found"
    args.out.write_text(json.dumps(examples, ensure_ascii=False, indent=1), encoding="utf-8")
    for ex in examples:
        print(f"{ex['page_id']} ({ex['engine']}, CER {ex['cer']}):\n  OCR: {ex['ocr']}\n  GT:  {ex['corrected']}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
