"""Word confidences for the reviewed BL scoring units, for confidence gating on the test set (D84).

Same pages, units, crops and engine builds as ocr/run_reviewed_units_ocr.py (rev 06): the crop is the
unit's reviewed box united with its words' boxes plus --pad px, no mask; Tesseract --psm 6, EasyOCR
defaults. Instead of plain text it saves:
  tesseract_words  Tesseract words with confidence (0-100), from image_to_data on the same crop
  easyocr_boxes    EasyOCR boxes with confidence (0-1); EasyOCR gives one confidence per box
  *_matches        whether the text rebuilt from those units equals the stored rev 06 text of the unit
                   after whitespace normalisation (gating aligns words either way; the rate is reported)

Nothing is scored here and no language model runs. Resumes: pages already in the output are skipped.

Output: <out>.jsonl, one line per page: page_id, units (id, tesseract_words, easyocr_boxes, tesseract_text,
easyocr_text, tesseract_matches, easyocr_matches); plus <out>_environment.json.

Usage:  python ocr/run_reviewed_units_conf.py --dataset-dir <dir with the .tif files> \
            --units reviewed_units.json --stored reviewed_region.jsonl --out /kaggle/working/conf/reviewed_conf
"""
import argparse
import json
import sys
from pathlib import Path

from run_ocr import TESSERACT_CMD, resolve_file
from run_region_ocr import crop_region
from run_revision_ocr import done_ids, environment


def squash(text: str) -> str:
    return " ".join(text.split())


def tesseract_words(crop, psm: int):
    import pytesseract

    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
    d = pytesseract.image_to_data(crop, lang="ben", config=f"--psm {psm}", output_type=pytesseract.Output.DICT)
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


def easyocr_boxes(crop, reader):
    import numpy as np

    result = reader.readtext(np.array(crop.convert("RGB")), detail=1)
    boxes = [{"text": text, "conf": float(conf)} for _, text, conf in result]
    return boxes, "\n".join(b["text"] for b in boxes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument("--units", type=Path, required=True)
    ap.add_argument("--stored", type=Path, required=True, help="rev 06 reviewed_region.jsonl (stored unit text)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pad", type=int, default=8)
    ap.add_argument("--psm", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="first N pages only (smoke test)")
    args = ap.parse_args()

    import easyocr
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    pages = json.loads(args.units.read_text(encoding="utf-8"))["pages"][: args.limit or None]
    stored = {}
    for l in args.stored.read_text(encoding="utf-8").split("\n"):
        if l.strip():
            r = json.loads(l)
            stored[r["page_id"]] = {g["id"]: g for g in r["regions"]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_p = args.out.with_name(args.out.name + ".jsonl")
    args.out.with_name(args.out.name + "_environment.json").write_text(json.dumps(environment(), indent=1),
                                                                       encoding="utf-8")
    reader = easyocr.Reader(["bn"], gpu=False)
    done = done_ids(out_p)
    with out_p.open("a", encoding="utf-8") as fo:
        for i, pg in enumerate(pages, 1):
            pid = pg["page_id"]
            if pid in done:
                continue
            img = Image.open(resolve_file(args.dataset_dir, pid, ".tif"))
            img.load()
            sx, sy = img.width / pg["alto_page_size"][0], img.height / pg["alto_page_size"][1]
            units, n_t, n_e = [], 0, 0
            for u in pg["units"]:
                x0, y0, x1, y1 = u["crop_union"]
                pts = [(x0 * sx, y0 * sy), (x1 * sx, y0 * sy), (x1 * sx, y1 * sy), (x0 * sx, y1 * sy)]
                crop = crop_region(img, pts, args.pad, mask_outside=False)
                tw, tt = tesseract_words(crop, args.psm)
                eb, et = easyocr_boxes(crop, reader)
                old = stored.get(pid, {}).get(u["id"], {})
                tm = squash(tt) == squash(old.get("tesseract", ""))
                em = squash(et) == squash(old.get("easyocr", ""))
                n_t += tm
                n_e += em
                units.append({"id": u["id"], "tesseract_words": tw, "easyocr_boxes": eb, "tesseract_text": tt,
                              "easyocr_text": et, "tesseract_matches": tm, "easyocr_matches": em,
                              "in_stored": bool(old)})
            fo.write(json.dumps({"page_id": pid, "units": units, "pad": args.pad, "psm": args.psm},
                                ensure_ascii=False) + "\n")
            fo.flush()
            print(f"[{i}/{len(pages)}] {pid}: {len(units)} units, text matches tesseract {n_t}, easyocr {n_e}",
                  file=sys.stderr, flush=True)
    print("done", out_p, flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
