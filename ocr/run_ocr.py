"""Runs Tesseract and EasyOCR over a list of page IDs and extracts ground
truth from each page's PAGE-XML, writing one JSON record per page per engine."""
import argparse
import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = Path(r"D:\Competition_dataset_ImagesPAGEXML")

os.environ["TESSDATA_PREFIX"] = str(REPO_ROOT / ".tessdata")
TESSERACT_CMD = os.environ.get(
    "TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

REGION_RE = re.compile(r"<TextRegion\b.*?</TextRegion>", re.DOTALL)
UNICODE_RE = re.compile(r"<Unicode>([^<]*)</Unicode>")


def ground_truth_text(page_id: str) -> str:
    """Concatenates all TextRegion transcriptions in document order,
    one line per region — mirrors how a page's running text is laid out."""
    xml_text = (DATASET_DIR / f"{page_id}.xml").read_text(encoding="utf-8")
    lines = []
    for region in REGION_RE.findall(xml_text):
        matches = UNICODE_RE.findall(region)
        if matches:
            # A region can have multiple TextLine/Unicode entries; keep only
            # the region-level TextEquiv if present, else join line entries.
            lines.append(matches[-1].strip() if len(matches) == 1 else matches[0].strip())
    return "\n".join(l for l in lines if l)


def run_tesseract(page_id: str) -> str:
    import pytesseract
    from PIL import Image

    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
    img = Image.open(DATASET_DIR / f"{page_id}.tif")
    return pytesseract.image_to_string(img, lang="ben")


def run_easyocr(page_id: str, reader) -> str:
    result = reader.readtext(str(DATASET_DIR / f"{page_id}.tif"), detail=0)
    return "\n".join(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        choices=["dev", "eval"],
        default="dev",
        help="Which frozen split to run OCR over",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSONL path (default: results/ocr_<split>.jsonl)",
    )
    args = parser.parse_args()

    page_ids = (
        (REPO_ROOT / "data" / f"split_{args.split}.txt")
        .read_text(encoding="utf-8")
        .split()
    )

    out_path = args.out or (REPO_ROOT / "results" / f"ocr_{args.split}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Running OCR on {len(page_ids)} pages ({args.split} split)...")

    import easyocr

    reader = easyocr.Reader(["bn"], gpu=False)

    records = []
    for i, page_id in enumerate(page_ids, 1):
        print(f"[{i}/{len(page_ids)}] {page_id}", file=sys.stderr)
        gt = ground_truth_text(page_id)
        tess_out = run_tesseract(page_id)
        easy_out = run_easyocr(page_id, reader)
        records.append(
            {
                "page_id": page_id,
                "ground_truth": gt,
                "tesseract": tess_out,
                "easyocr": easy_out,
            }
        )

    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} records to {out_path}")


if __name__ == "__main__":
    main()
