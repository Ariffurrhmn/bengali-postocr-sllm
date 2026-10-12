"""Re-runs both OCR engines with word-level confidences, for confidence-gated
correction (review item E4).

For every page in an existing OCR file (default: the 40 evaluation pages of
results/ocr_eval40.jsonl) this writes:
  tesseract_words - Tesseract words with their confidence (0-100) and line id,
                    from image_to_data (same engine call as run_ocr.py)
  easyocr_boxes   - EasyOCR text boxes with their confidence (0-1); EasyOCR
                    reports confidence per detected box, not per word
  tesseract_text / easyocr_text - the text rebuilt from those units, and
  *_matches       - whether it equals the stored OCR text after whitespace
                    normalisation (the gating step aligns words either way)

Resumable like run_ocr.py: pages already in the output file are skipped.

Usage (Colab, same setup as run_ocr.py):
    python run_ocr_conf.py --dataset-dir /content/Competition_dataset_ImagesPAGEXML \\
        --ocr-path "/content/drive/My Drive/bengali-postocr-results/ocr_eval40.jsonl" \\
        --out "/content/drive/My Drive/bengali-postocr-results/ocr_conf40.jsonl"
"""
import argparse
import json
import sys
from pathlib import Path

from run_ocr import TESSERACT_CMD, resolve_file

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def squash(text: str) -> str:
    return " ".join(text.split())


def tesseract_words(dataset_dir: Path, page_id: str):
    import pytesseract
    from PIL import Image

    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
    img = Image.open(resolve_file(dataset_dir, page_id, ".tif"))
    d = pytesseract.image_to_data(img, lang="ben", output_type=pytesseract.Output.DICT)
    words = []
    for i, text in enumerate(d["text"]):
        if text and text.strip():
            words.append({"text": text, "conf": float(d["conf"][i]),
                          "line": f"{d['block_num'][i]}.{d['par_num'][i]}.{d['line_num'][i]}"})
    lines, current, key = [], [], None
    for w in words:
        if w["line"] != key and current:
            lines.append(" ".join(current))
            current = []
        key = w["line"]
        current.append(w["text"])
    if current:
        lines.append(" ".join(current))
    return words, "\n".join(lines)


def easyocr_boxes(dataset_dir: Path, page_id: str, reader):
    result = reader.readtext(str(resolve_file(dataset_dir, page_id, ".tif")), detail=1)
    boxes = [{"text": text, "conf": float(conf)} for _, text, conf in result]
    return boxes, "\n".join(b["text"] for b in boxes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--ocr-path", type=Path, required=True,
                        help="Existing OCR file: gives the page list and the stored text to compare")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    stored = [json.loads(l) for l in args.ocr_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    done = set()
    if args.out.exists():
        done = {json.loads(l)["page_id"] for l in args.out.read_text(encoding="utf-8").splitlines()
                if l.strip()}
    todo = [p for p in stored if p["page_id"] not in done]
    print(f"{len(stored)} pages, {len(done)} already done, {len(todo)} to run")

    import easyocr
    reader = easyocr.Reader(["bn"], gpu=False)
    with args.out.open("a", encoding="utf-8") as f:
        for i, page in enumerate(todo, 1):
            pid = page["page_id"]
            words, ttext = tesseract_words(args.dataset_dir, pid)
            boxes, etext = easyocr_boxes(args.dataset_dir, pid, reader)
            rec = {"page_id": pid, "tesseract_words": words, "easyocr_boxes": boxes,
                   "tesseract_text": ttext, "easyocr_text": etext,
                   "tesseract_matches": squash(ttext) == squash(page["tesseract"]),
                   "easyocr_matches": squash(etext) == squash(page["easyocr"])}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i}/{len(todo)}] {pid}  tesseract match: {rec['tesseract_matches']}  "
                  f"easyocr match: {rec['easyocr_matches']}")
    print(f"Done. Results in {args.out}")


if __name__ == "__main__":
    main()
