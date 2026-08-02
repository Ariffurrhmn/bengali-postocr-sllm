"""Runs Tesseract and EasyOCR over a list of page IDs and extracts ground
truth from each page's PAGE-XML, writing one JSON record per page per engine."""
import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Local (Windows) default; override with --dataset-dir on Colab/Linux, e.g.
# /content/Competition_dataset_ImagesPAGEXML after unzipping the dataset.
DEFAULT_DATASET_DIR = r"D:\Competition_dataset_ImagesPAGEXML"

# Same literal value as data/make_split.py's SEED, reused here for the
# separate step of subsampling N pages out of a frozen split (not the same
# operation, but keeping one project-wide default seed for consistency).
DEFAULT_SAMPLE_SEED = 403

os.environ.setdefault("TESSDATA_PREFIX", str(REPO_ROOT / ".tessdata"))
# Local (Windows) default; on Colab, `apt-get install tesseract-ocr` puts
# `tesseract` on PATH, so TESSERACT_CMD=tesseract works with no override.
TESSERACT_CMD = os.environ.get(
    "TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

REGION_RE = re.compile(r"<TextRegion\b.*?</TextRegion>", re.DOTALL)
UNICODE_RE = re.compile(r"<Unicode>([^<]*)</Unicode>")


def resolve_file(dataset_dir: Path, page_id: str, suffix: str) -> Path:
    """Case-insensitive file lookup. The source dataset has at least one page
    (279_2_a_15_0005) where the .tif and .xml stems differ only in case
    (279_2_a_15_0005.tif vs 279_2_A_15_0005.xml) — same quirk data/make_split.py
    already works around when building the split; page_id here is the
    split's canonical (.tif-cased) name, so an exact f"{page_id}{suffix}"
    match can miss the differently-cased file on disk."""
    exact = dataset_dir / f"{page_id}{suffix}"
    if exact.exists():
        return exact
    target = f"{page_id}{suffix}".lower()
    for candidate in dataset_dir.iterdir():
        if candidate.name.lower() == target:
            return candidate
    raise FileNotFoundError(f"No {suffix} file for page_id={page_id!r} in {dataset_dir}")


def ground_truth_text(dataset_dir: Path, page_id: str) -> str:
    """Concatenates all TextRegion transcriptions in document order,
    one line per region — mirrors how a page's running text is laid out."""
    xml_path = resolve_file(dataset_dir, page_id, ".xml")
    xml_text = xml_path.read_text(encoding="utf-8")
    lines = []
    for region in REGION_RE.findall(xml_text):
        matches = UNICODE_RE.findall(region)
        if matches:
            # A region can have multiple TextLine/Unicode entries; keep only
            # the region-level TextEquiv if present, else join line entries.
            lines.append(matches[-1].strip() if len(matches) == 1 else matches[0].strip())
    return "\n".join(l for l in lines if l)


def run_tesseract(dataset_dir: Path, page_id: str) -> str:
    import pytesseract
    from PIL import Image

    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
    img = Image.open(resolve_file(dataset_dir, page_id, ".tif"))
    return pytesseract.image_to_string(img, lang="ben")


def run_easyocr(dataset_dir: Path, page_id: str, reader) -> str:
    result = reader.readtext(str(resolve_file(dataset_dir, page_id, ".tif")), detail=0)
    return "\n".join(result)


def load_done_page_ids(out_path: Path) -> set:
    """Reads whatever's already in the output file so completed pages are
    skipped on resume — mirrors correction/run_sweep.py's resumability."""
    if not out_path.exists():
        return set()
    done = set()
    for line in out_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue  # tolerate a truncated last line from a mid-write interruption
        done.add(r["page_id"])
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        choices=["dev", "eval"],
        default="dev",
        help="Which frozen split to run OCR over",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(DEFAULT_DATASET_DIR),
        help="Directory containing the .tif/.xml page pairs",
    )
    parser.add_argument(
        "--sample-n",
        type=int,
        default=None,
        help="If set, randomly subsample this many pages from the split "
        "(e.g. for a smaller, Colab-feasible eval run) instead of using all of it",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=DEFAULT_SAMPLE_SEED,
        help="Seed for --sample-n subsampling (deterministic/reproducible)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSONL path (default: results/ocr_<split>.jsonl)",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    page_ids = (
        (REPO_ROOT / "data" / f"split_{args.split}.txt")
        .read_text(encoding="utf-8")
        .split()
    )

    if args.sample_n is not None:
        page_ids = sorted(random.Random(args.sample_seed).sample(page_ids, args.sample_n))

    out_path = args.out or (REPO_ROOT / "results" / f"ocr_{args.split}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = load_done_page_ids(out_path)
    remaining = [p for p in page_ids if p not in done]

    print(
        f"Running OCR on {len(page_ids)} pages ({args.split} split) from {dataset_dir} "
        f"— {len(done)} already done, {len(remaining)} remaining..."
    )

    if not remaining:
        print("Nothing to do.")
        return

    import easyocr

    reader = easyocr.Reader(["bn"], gpu=False)

    with out_path.open("a", encoding="utf-8") as f:
        for i, page_id in enumerate(remaining, 1):
            print(f"[{i}/{len(remaining)}] {page_id}", file=sys.stderr)
            gt = ground_truth_text(dataset_dir, page_id)
            tess_out = run_tesseract(dataset_dir, page_id)
            easy_out = run_easyocr(dataset_dir, page_id, reader)
            record = {
                "page_id": page_id,
                "ground_truth": gt,
                "tesseract": tess_out,
                "easyocr": easy_out,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()  # survive an interruption right after this line

    print(f"Wrote {len(remaining)} new records to {out_path}")


if __name__ == "__main__":
    main()
