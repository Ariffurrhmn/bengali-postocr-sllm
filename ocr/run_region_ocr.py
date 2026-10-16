"""Region-level OCR (third review, items C5/M5).

The main OCR reads whole pages, while the ground truth covers only the
annotated text regions, so text outside the regions and a different reading
order count as errors. Here each PAGE-XML text region (version 2 coordinates)
is cut out of the page image and read separately by both engines, so the OCR
and the reference cover the same text in the same order.

  --verify   first re-OCR the whole pages with the code of run_ocr.py and
             compare them with the stored OCR (results/ocr_eval40.jsonl), to
             show that the engine versions match the main run (Tesseract 5.3.4,
             EasyOCR 1.7.2). Writes <out stem>_verify.json and stops if fewer
             than --min-verified pages match exactly for Tesseract.

Region text follows run_ocr.py's rule (region-level TextEquiv). Order: the
reading order declared in the PAGE-XML (regions it lists, by index), then any
other regions in file order; regions without transcribed text are skipped,
since they cannot be scored. Each crop is the region's bounding box plus
--pad pixels (plain bounding-box crop; --mask also paints everything outside the region
polygon white, which cut into glyphs and scored worse in a local test). Tesseract reads each
crop as one block of text (--psm 6, since the whole-page default 3 finds nothing in many short
crops); otherwise both engines use the default settings of run_ocr.py.

Output, one JSON line per page: page_id, ground_truth / tesseract / easyocr
(the regions joined by newlines, in the order above, so run_sweep.py and the
scoring scripts can use the file like ocr_eval40.jsonl), regions (per-region
id, type, text and both OCR outputs) and the engine versions.

Usage (Colab CPU, see notebooks/colab_region_ocr.ipynb):
    python run_region_ocr.py --dataset-dir /content/Competition_dataset_ImagesPAGEXML \\
        --stored ../results/ocr_eval40.jsonl --verify --out <dir>/ocr_region40.jsonl
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from run_ocr import REPO_ROOT, TESSERACT_CMD, resolve_file, run_easyocr, run_tesseract

REGION_RE = re.compile(r"<TextRegion\b([^>]*)>(.*?)</TextRegion>", re.DOTALL)
UNICODE_RE = re.compile(r"<Unicode>([^<]*)</Unicode>")
COORDS_RE = re.compile(r'<Coords\s+points="([^"]+)"')
POINT_RE = re.compile(r'<Point\s+x="(\d+)"\s+y="(\d+)"')


def unescape(text: str) -> str:
    return (text.replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"')
            .replace("&apos;", "'").replace("&amp;", "&"))


def regions(xml: str) -> list[dict]:
    """Text regions with transcriptions, in declared reading order, then file order."""
    out = []
    for attrs, body in REGION_RE.findall(xml):
        matches = UNICODE_RE.findall(body)
        text = (matches[-1] if len(matches) == 1 else matches[0]).strip() if matches else ""
        if not text:
            continue
        # The region's own Coords come before any TextLine, so the first match is it.
        m = COORDS_RE.search(body)
        if m:
            pts = [tuple(int(v) for v in p.split(",")) for p in m.group(1).split()]
        else:
            pts = [(int(x), int(y)) for x, y in POINT_RE.findall(body.split("<TextLine")[0])]
        rtype = re.search(r'\btype="([^"]+)"', attrs)
        out.append({"id": re.search(r'\bid="([^"]+)"', attrs).group(1),
                    "type": rtype.group(1) if rtype else None,
                    "text": unescape(text), "points": pts})
    refs = sorted((int(re.search(r'index="(\d+)"', t).group(1)), re.search(r'regionRef="([^"]+)"', t).group(1))
                  for t in re.findall(r"<RegionRefIndexed[^>]*>", xml))
    rank = {rid: i for i, (_, rid) in enumerate(refs)}
    # sorted() is stable, so unlisted regions keep their file order after the listed ones.
    return sorted(out, key=lambda r: rank.get(r["id"], len(rank)))


def crop_region(img, points, pad: int, mask_outside: bool = True):
    from PIL import Image, ImageDraw

    xs, ys = [p[0] for p in points], [p[1] for p in points]
    box = (max(0, min(xs) - pad), max(0, min(ys) - pad),
           min(img.width, max(xs) + pad), min(img.height, max(ys) + pad))
    if not mask_outside:
        return img.crop(box)
    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).polygon(points, fill=255)
    # Grow the mask by the padding, so glyph edges on the polygon border survive.
    if pad:
        from PIL import ImageFilter
        mask = mask.filter(ImageFilter.MaxFilter(2 * (pad // 2) + 1))
    white = Image.new(img.mode, img.size, "white")
    return Image.composite(img, white, mask).crop(box)


def ocr_crop(crop, reader, psm: int) -> tuple[str, str]:
    import numpy as np
    import pytesseract

    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
    tess = pytesseract.image_to_string(crop, lang="ben", config=f"--psm {psm}").strip()
    easy = ("\n".join(reader.readtext(np.array(crop.convert("RGB")), detail=0)).strip()
            if reader else "")
    return tess, easy


def versions() -> dict:
    import easyocr  # noqa: F401  (version only)

    tv = subprocess.run([TESSERACT_CMD, "--version"], capture_output=True, text=True)
    return {"tesseract": (tv.stdout or tv.stderr).splitlines()[0], "easyocr": easyocr.__version__}


def verify(dataset_dir: Path, stored: dict, reader, out: Path, min_verified: int) -> None:
    sys.path.insert(0, str(REPO_ROOT / "eval"))
    from metrics import cer

    rows = []
    for i, (pid, rec) in enumerate(sorted(stored.items()), 1):
        t = run_tesseract(dataset_dir, pid)
        e = run_easyocr(dataset_dir, pid, reader)
        rows.append({"page_id": pid, "tesseract_identical": t == rec["tesseract"],
                     "easyocr_identical": e == rec["easyocr"],
                     "tesseract_cer_vs_stored": cer(rec["tesseract"], t),
                     "easyocr_cer_vs_stored": cer(rec["easyocr"], e)})
        print(f"[verify {i}/{len(stored)}] {pid} tess same={rows[-1]['tesseract_identical']} "
              f"easy same={rows[-1]['easyocr_identical']}", file=sys.stderr, flush=True)
    summary = {"versions": versions(), "pages": len(rows),
               "tesseract_identical": sum(r["tesseract_identical"] for r in rows),
               "easyocr_identical": sum(r["easyocr_identical"] for r in rows), "per_page": rows}
    out.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"verify: Tesseract identical on {summary['tesseract_identical']}/{len(rows)} pages, "
          f"EasyOCR on {summary['easyocr_identical']}/{len(rows)}; versions {summary['versions']}")
    if summary["tesseract_identical"] < min_verified:
        sys.exit(f"Only {summary['tesseract_identical']} Tesseract pages match the stored OCR "
                 f"(need {min_verified}): this is not the engine of the main run. Stopping.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument("--stored", type=Path, default=REPO_ROOT / "results" / "ocr_eval40.jsonl",
                    help="the main run's OCR; its page list is used, and --verify compares with it")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pad", type=int, default=8, help="pixels around each region")
    ap.add_argument("--psm", type=int, default=6,
                    help="Tesseract page segmentation mode for the crops (6: one block of text; "
                    "the whole-page default 3 finds no text in many short heading crops)")
    ap.add_argument("--mask", action="store_true", help="paint outside the polygon white (worse in a local test)")
    ap.add_argument("--tesseract-only", action="store_true", help="for quick local tests")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--min-verified", type=int, default=38,
                    help="Tesseract pages that must match exactly (the confidence re-run matched 40/40)")
    ap.add_argument("--pages", default=None, help="comma-separated page ids (default: all stored pages)")
    args = ap.parse_args()

    import easyocr
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    stored = {json.loads(l)["page_id"]: json.loads(l)
              for l in args.stored.read_text(encoding="utf-8").split("\n") if l.strip()}
    if args.pages:
        stored = {p: stored[p] for p in args.pages.split(",")}
    reader = None if args.tesseract_only else easyocr.Reader(["bn"], gpu=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    verify_path = args.out.with_name(args.out.stem + "_verify.json")
    if args.verify and not verify_path.exists():
        verify(args.dataset_dir, stored, reader, verify_path, args.min_verified)

    done = set()
    if args.out.exists():
        done = {json.loads(l)["page_id"] for l in args.out.read_text(encoding="utf-8").split("\n") if l.strip()}
    vers = versions()
    with args.out.open("a", encoding="utf-8") as f:
        for i, pid in enumerate(sorted(stored), 1):
            if pid in done:
                continue
            xml = resolve_file(args.dataset_dir, pid, ".xml").read_text(encoding="utf-8")
            img = Image.open(resolve_file(args.dataset_dir, pid, ".tif"))
            img.load()
            regs = []
            for r in regions(xml):
                if len(r["points"]) < 3:
                    print(f"  {pid} {r['id']}: no polygon, skipped", file=sys.stderr)
                    continue
                tess, easy = ocr_crop(crop_region(img, r["points"], args.pad, args.mask),
                                      reader, args.psm)
                regs.append({"id": r["id"], "type": r["type"], "ground_truth": r["text"],
                             "tesseract": tess, "easyocr": easy})
            rec = {"page_id": pid,
                   "ground_truth": "\n".join(r["ground_truth"] for r in regs),
                   "tesseract": "\n".join(r["tesseract"] for r in regs),
                   "easyocr": "\n".join(r["easyocr"] for r in regs),
                   "regions": regs, "pad": args.pad, "psm": args.psm, "mask": args.mask,
                   "versions": vers}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i}/{len(stored)}] {pid}: {len(regs)} regions", file=sys.stderr, flush=True)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
