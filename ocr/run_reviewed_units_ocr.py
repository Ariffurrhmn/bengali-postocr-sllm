"""OCR of the reviewed BL scoring units (tools/build_bl_reviewed_units.py), both engines.

Each unit is a kept text block or a box the author added (D77). The crop is the unit's
reviewed box united with the boxes of the unit's own words (Transkribus sometimes puts a word
slightly outside its box), plus --pad px, no mask. Tesseract --psm 6, EasyOCR defaults, as
ocr/run_revision_ocr.py. ALTO coordinates are scaled by image size / ALTO page size.

Output (stem = --out): <stem>_region.jsonl, one line per page, the schema that
correction/score_trained_corrector.py reads: page_id, source, page_type, ground_truth,
tesseract, easyocr (units joined by newline, in the reviewed order) and "regions" (id, key,
kind, ground_truth, tesseract, easyocr, crop_box); plus <stem>_environment.json.
Resumes: pages already in the file are skipped.

Usage:  python ocr/run_reviewed_units_ocr.py --dataset-dir <dir with the .tif files> \
            --units reviewed_units.json --out /kaggle/working/ocr/reviewed [--limit N]
"""
import argparse
import json
import sys
from pathlib import Path

from run_ocr import resolve_file
from run_region_ocr import crop_region, ocr_crop
from run_revision_ocr import done_ids, environment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument("--units", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pad", type=int, default=8)
    ap.add_argument("--psm", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="first N pages only (smoke test)")
    args = ap.parse_args()

    import easyocr
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    pages = json.loads(args.units.read_text(encoding="utf-8"))["pages"][: args.limit or None]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    region_p = args.out.with_name(args.out.name + "_region.jsonl")
    env = environment()
    args.out.with_name(args.out.name + "_environment.json").write_text(json.dumps(env, indent=1), encoding="utf-8")
    vers = {"tesseract": (env.get("tesseract_version") or [None])[0], "easyocr": env["easyocr"]}
    reader = easyocr.Reader(["bn"], gpu=False)
    done = done_ids(region_p)
    with region_p.open("a", encoding="utf-8") as fr:
        for i, pg in enumerate(pages, 1):
            pid = pg["page_id"]
            if pid in done:
                continue
            img = Image.open(resolve_file(args.dataset_dir, pid, ".tif"))
            img.load()
            sx, sy = img.width / pg["alto_page_size"][0], img.height / pg["alto_page_size"][1]
            regs = []
            for u in pg["units"]:
                x0, y0, x1, y1 = u["crop_union"]
                pts = [(x0 * sx, y0 * sy), (x1 * sx, y0 * sy), (x1 * sx, y1 * sy), (x0 * sx, y1 * sy)]
                crop = crop_region(img, pts, args.pad, mask_outside=False)
                tess, easy = ocr_crop(crop, reader, args.psm)
                xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                regs.append({"id": u["id"], "key": u["key"], "kind": u["kind"], "ground_truth": u["ground_truth"],
                             "tesseract": tess, "easyocr": easy,
                             "crop_box": [round(min(xs)), round(min(ys)), round(max(xs)), round(max(ys))]})
            fr.write(json.dumps({"page_id": pid, "source": "BL", "page_type": pg["page_type"],
                                 "ground_truth": "\n".join(r["ground_truth"] for r in regs),
                                 "tesseract": "\n".join(r["tesseract"] for r in regs),
                                 "easyocr": "\n".join(r["easyocr"] for r in regs),
                                 "regions": regs, "pad": args.pad, "psm": args.psm, "mask": False,
                                 "scale": [sx, sy], "order": "reviewed reading order (D77)", "versions": vers},
                                ensure_ascii=False) + "\n")
            fr.flush()
            print(f"[{i}/{len(pages)}] {pid}: {len(regs)} units, scale {sx:.3f}", file=sys.stderr, flush=True)
    print("done", region_p, flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
