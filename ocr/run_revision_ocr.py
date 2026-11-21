"""Fresh OCR for the 2026-11 revision: the BL Transkribus pages (ALTO) and the REID2019
train/validation pages (PAGE), whole page and per text block, in one engine run.

Why one run: Tesseract 5.3.4 output drifted between runs in this project (D44), so the
training pairs, the validation pages and the new test pages are all read in the same
session type with the same engine builds, and the stored October OCR is not mixed in.

Outputs (stem = --out):
  <stem>_wholepage.jsonl  one line per page: page_id, source, ground_truth, tesseract,
                          easyocr. Whole page read exactly as ocr/run_ocr.py does
                          (Tesseract default --psm 3, EasyOCR defaults, image as released).
  <stem>_region.jsonl     one line per page, the same fields joined over text blocks, plus
                          "regions": id, ground_truth, tesseract, easyocr, crop box. Each crop
                          is the block's bounding box plus --pad px (no mask), Tesseract
                          --psm 6, as ocr/run_region_ocr.py.
  <stem>_environment.json Tesseract/Leptonica version lines, CPU, Bengali model SHA-256,
                          EasyOCR, PyTorch, CUDA, Python, Pillow.

Ground truth and block order:
  ALTO (BL)  words joined by one space within a TextLine, lines by "\\n" within a
             TextBlock, blocks by "\\n" in ALTO document order. The ALTO files have no
             explicit reading order; none is inferred here. The author checks the order
             before any test page is scored.
  PAGE (REID) run_region_ocr.regions(): region-level TextEquiv, declared reading order
             first, then file order (as the paper's region run).
Coordinates (ALTO): scaled by image size / ALTO page size where they differ (5 BL pages).
The crop box is the union of the TextBlock box and its lines' and words' boxes
(Transkribus exports sometimes put words slightly outside their block), then padded.

Usage:  python ocr/run_revision_ocr.py --dataset-dir <dir with .tif/.xml> \\
            --pages-file <ids.txt> --out /kaggle/working/ocr/shard_a
"""
import argparse
import json
import platform
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from run_ocr import TESSERACT_CMD, resolve_file, run_easyocr, run_tesseract, tesseract_environment
from run_region_ocr import crop_region, ocr_crop, regions as page_regions


def local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def box_of(el, sx: float, sy: float):
    try:
        x, y = float(el.get("HPOS")), float(el.get("VPOS"))
        w, h = float(el.get("WIDTH")), float(el.get("HEIGHT"))
    except (TypeError, ValueError):
        return None
    return (x * sx, y * sy, (x + w) * sx, (y + h) * sy)


def alto_blocks(xml_path: Path, img_w: int, img_h: int) -> tuple[list, list]:
    root = ET.parse(xml_path).getroot()
    page = next(e for e in root.iter() if local(e.tag) == "Page")
    pw, ph = float(page.get("WIDTH") or img_w), float(page.get("HEIGHT") or img_h)
    sx, sy = img_w / pw, img_h / ph
    out = []
    for blk in (e for e in root.iter() if local(e.tag) == "TextBlock"):
        lines, boxes = [], []
        b0 = box_of(blk, sx, sy)
        if b0:
            boxes.append(b0)
        for ln in (e for e in blk if local(e.tag) == "TextLine"):
            bl = box_of(ln, sx, sy)
            if bl:
                boxes.append(bl)
            words = []
            for s in (e for e in ln if local(e.tag) == "String"):
                words.append(s.get("CONTENT") or "")
                bs = box_of(s, sx, sy)
                if bs:
                    boxes.append(bs)
            text = " ".join(w for w in words if w).strip()
            if text:
                lines.append(text)
        if not lines or not boxes:
            continue
        x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
        x1, y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
        out.append({"id": blk.get("ID"), "ground_truth": "\n".join(lines),
                    "points": [(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
                    "block_box": [round(v) for v in b0] if b0 else None})
    return out, [sx, sy]


def blocks_for(dataset_dir: Path, pid: str, img) -> tuple[str, list, list]:
    xml_path = resolve_file(dataset_dir, pid, ".xml")
    head = xml_path.read_text(encoding="utf-8", errors="replace")[:2000]
    if "<alto" in head:
        blocks, scale = alto_blocks(xml_path, img.width, img.height)
        return "alto", blocks, scale
    regs = [{"id": r["id"], "ground_truth": r["text"], "points": r["points"], "block_box": None}
            for r in page_regions(xml_path.read_text(encoding="utf-8")) if len(r["points"]) >= 3]
    return "page", regs, [1.0, 1.0]


def environment() -> dict:
    import easyocr
    import PIL
    import torch

    env = tesseract_environment()
    env.update({"easyocr": easyocr.__version__, "torch": torch.__version__, "cuda": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(), "python": sys.version,
                "pillow": PIL.__version__, "tesseract_cmd": TESSERACT_CMD, "node": platform.node()})
    return env


def done_ids(path: Path) -> set:
    out = set()
    if path.exists():
        for l in path.read_text(encoding="utf-8").split("\n"):
            try:
                out.add(json.loads(l)["page_id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument("--pages-file", type=Path, required=True, help="one page id per line")
    ap.add_argument("--out", type=Path, required=True, help="output stem")
    ap.add_argument("--pad", type=int, default=8)
    ap.add_argument("--psm", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="first N pages only (smoke test)")
    args = ap.parse_args()

    import easyocr
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    pages = [p.strip() for p in args.pages_file.read_text(encoding="utf-8").split("\n") if p.strip()]
    pages = pages[: args.limit or None]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    whole_p = args.out.with_name(args.out.name + "_wholepage.jsonl")
    region_p = args.out.with_name(args.out.name + "_region.jsonl")
    env = environment()
    args.out.with_name(args.out.name + "_environment.json").write_text(json.dumps(env, indent=1), encoding="utf-8")
    vers = {"tesseract": (env.get("tesseract_version") or [None])[0], "easyocr": env["easyocr"]}
    reader = easyocr.Reader(["bn"], gpu=False)
    dw, dr = done_ids(whole_p), done_ids(region_p)
    with whole_p.open("a", encoding="utf-8") as fw, region_p.open("a", encoding="utf-8") as fr:
        for i, pid in enumerate(pages, 1):
            if pid in dw and pid in dr:
                continue
            img = Image.open(resolve_file(args.dataset_dir, pid, ".tif"))
            img.load()
            fmt, blocks, scale = blocks_for(args.dataset_dir, pid, img)
            source = "BL" if fmt == "alto" else "REID"
            if pid not in dw:
                fw.write(json.dumps({"page_id": pid, "source": source,
                                     "ground_truth": "\n".join(b["ground_truth"] for b in blocks),
                                     "tesseract": run_tesseract(args.dataset_dir, pid),
                                     "easyocr": run_easyocr(args.dataset_dir, pid, reader),
                                     "image_size": [img.width, img.height], "versions": vers},
                                    ensure_ascii=False) + "\n")
                fw.flush()
            if pid not in dr:
                regs = []
                for b in blocks:
                    tess, easy = ocr_crop(crop_region(img, b["points"], args.pad, mask_outside=False),
                                          reader, args.psm)
                    xs, ys = [p[0] for p in b["points"]], [p[1] for p in b["points"]]
                    regs.append({"id": b["id"], "ground_truth": b["ground_truth"], "tesseract": tess,
                                 "easyocr": easy, "block_box": b["block_box"],
                                 "crop_box": [round(min(xs)), round(min(ys)), round(max(xs)), round(max(ys))]})
                fr.write(json.dumps({"page_id": pid, "source": source,
                                     "ground_truth": "\n".join(r["ground_truth"] for r in regs),
                                     "tesseract": "\n".join(r["tesseract"] for r in regs),
                                     "easyocr": "\n".join(r["easyocr"] for r in regs),
                                     "regions": regs, "pad": args.pad, "psm": args.psm, "mask": False,
                                     "scale": scale,
                                     "order": ("ALTO document order (not checked)" if fmt == "alto"
                                               else "PAGE reading order, then file order"),
                                     "versions": vers}, ensure_ascii=False) + "\n")
                fr.flush()
            print(f"[{i}/{len(pages)}] {pid} ({source}): {len(blocks)} blocks, scale {scale[0]:.3f}",
                  file=sys.stderr, flush=True)
    print("done", whole_p, region_p, flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
