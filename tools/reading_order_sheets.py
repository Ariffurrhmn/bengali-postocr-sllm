"""Reading-order check sheets for the BL test pages.

Reads the page list, the ALTO v2 XML and the page images. Draws every
TextBlock box with its file-order number on a downscaled image, and writes
one HTML sheet (sheet.html) that lists number -> block ID -> first characters
of the block text. Also writes a draft CSV with the blocks in file order.
Nothing is scored; text is only displayed, never changed.

Usage:
    python tools/reading_order_sheets.py --group2 "<Group 2 path>" --out results/bl_reading_order [--limit 2]
"""
import argparse
import csv
import datetime
import html
import os
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw, ImageFont

NS = "{http://www.loc.gov/standards/alto/ns-v2#}"
DATA_REL = os.path.join("Research Materials", "Datasets")
ALTO_DIR = os.path.join("BL_Transkribus", "Transkribus_dataset_ImagesALTOXML")
PAGE_LIST = os.path.join("revision_2026-11", "test_bl.txt")
MAX_SIDE = 1400
TEXT_CHARS = 40

COLORS = [(220, 20, 60), (0, 90, 200), (0, 140, 60), (230, 120, 0), (140, 0, 170), (0, 150, 160)]


def parse_page(xml_path):
    root = ET.parse(xml_path).getroot()
    page = root.find(f".//{NS}Page")
    pw, ph = float(page.get("WIDTH")), float(page.get("HEIGHT"))
    blocks = []
    for tb in root.iter(f"{NS}TextBlock"):
        words = []
        for line in tb.iter(f"{NS}TextLine"):
            words.append(" ".join(s.get("CONTENT", "") for s in line.iter(f"{NS}String")))
        blocks.append({
            "id": tb.get("ID"),
            "x": float(tb.get("HPOS")), "y": float(tb.get("VPOS")),
            "w": float(tb.get("WIDTH")), "h": float(tb.get("HEIGHT")),
            "text": " ".join(words).strip(),
        })
    return pw, ph, blocks


def load_font(size):
    for name in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_sheet(img, pw, ph, blocks, png_path):
    iw, ih = img.size
    sx, sy = iw / pw, ih / ph  # ALTO coordinates -> image pixels
    scale = min(1.0, MAX_SIDE / max(iw, ih))
    small = img.convert("RGB").resize((max(1, round(iw * scale)), max(1, round(ih * scale))))
    d = ImageDraw.Draw(small, "RGBA")
    font = load_font(max(18, round(small.size[1] / 45)))
    for i, b in enumerate(blocks, 1):
        c = COLORS[i % len(COLORS)]
        x0, y0 = b["x"] * sx * scale, b["y"] * sy * scale
        x1, y1 = (b["x"] + b["w"]) * sx * scale, (b["y"] + b["h"]) * sy * scale
        d.rectangle([x0, y0, x1, y1], outline=c + (255,), width=3, fill=c + (28,))
        label = str(i)
        tb = d.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        d.rectangle([x0, y0, x0 + tw + 10, y0 + th + 10], fill=c + (235,))
        d.text((x0 + 5 - tb[0], y0 + 5 - tb[1]), label, fill=(255, 255, 255, 255), font=font)
    small.save(png_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group2", required=True, help="path of the 'Group 2' folder")
    ap.add_argument("--out", default=os.path.join("results", "bl_reading_order"))
    ap.add_argument("--limit", type=int, default=0, help="only the first N pages (testing)")
    args = ap.parse_args()

    data = os.path.join(args.group2, DATA_REL)
    with open(os.path.join(data, PAGE_LIST), encoding="utf-8") as f:
        pages = [ln.strip() for ln in f if ln.strip()]
    if args.limit:
        pages = pages[: args.limit]
    img_dir = os.path.join(args.out, "img")
    os.makedirs(img_dir, exist_ok=True)

    today = datetime.date.today().isoformat()
    sections, rows, rescaled = [], [], []
    for n, page in enumerate(pages, 1):
        pw, ph, blocks = parse_page(os.path.join(data, ALTO_DIR, page + ".xml"))
        img = Image.open(os.path.join(data, ALTO_DIR, page + ".tif"))
        iw, ih = img.size
        rw, rh = iw / pw, ih / ph
        resc = abs(rw - 1) > 0.02 or abs(rh - 1) > 0.02
        if resc:
            rescaled.append((page, round(1 / rw, 3)))
        draw_sheet(img, pw, ph, blocks, os.path.join(img_dir, page + ".png"))
        rows.append([page, ";".join(b["id"] for b in blocks), "", ""])
        trs = "".join(
            f"<tr><td>{i}</td><td><code>{html.escape(b['id'])}</code></td>"
            f"<td>{html.escape(b['text'][:TEXT_CHARS]) or '<i>(empty)</i>'}</td></tr>"
            for i, b in enumerate(blocks, 1))
        note = (f" &nbsp;<b style='color:#b00'>boxes rescaled x{rw:.3f}/{rh:.3f}</b>" if resc else "")
        sections.append(
            f"<section><h2>{n}/{len(pages)} &nbsp;{html.escape(page)} &nbsp;({len(blocks)} blocks){note}</h2>"
            f"<div class='row'><img src='img/{html.escape(page)}.png'>"
            f"<table><tr><th>No.</th><th>Block ID</th><th>Text (first {TEXT_CHARS})</th></tr>{trs}</table></div></section>")

    with open(os.path.join(args.out, "test_reading_order_draft.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["page", "ordered_block_ids", "checked_by", "date"])
        w.writerows(rows)

    intro = (
        "<h1>BL test pages: reading-order check</h1>"
        "<p><b>What to look at:</b> the numbers on each image are the blocks in the order they appear in the XML file. "
        "Decide whether that is the order a person would read the page (top to bottom, columns left to right, "
        "headers and page numbers where you would read them).</p>"
        "<p><b>If the order is right:</b> leave the row in <code>test_reading_order_draft.csv</code> as it is and fill "
        "<code>checked_by</code> and <code>date</code>.<br>"
        "<b>If it is wrong:</b> in <code>ordered_block_ids</code> rearrange the block IDs (separated by <code>;</code>) into "
        "reading order. Use the table next to the image to match number to ID. Keep every ID exactly once; do not add or "
        "drop blocks and do not touch any text.</p>"
        "<p>Boxes are already rescaled to the image where the ALTO page size differs from it. Nothing here is scored.</p>")
    css = ("body{font-family:Segoe UI,Nirmala UI,sans-serif;margin:16px}section{margin-bottom:40px;page-break-after:always}"
           ".row{display:flex;gap:16px;align-items:flex-start}img{max-width:62%;border:1px solid #888}"
           "table{border-collapse:collapse;font-size:14px}td,th{border:1px solid #bbb;padding:2px 6px;text-align:left}"
           "code{font-size:12px}")
    with open(os.path.join(args.out, "sheet.html"), "w", encoding="utf-8") as f:
        f.write(f"<!doctype html><meta charset='utf-8'><title>Reading order</title><style>{css}</style>"
                + intro + "".join(sections))
    with open(os.path.join(args.out, "README.txt"), "w", encoding="utf-8") as f:
        f.write("Open sheet.html in a browser and go through the pages. Edit test_reading_order_draft.csv as described\n"
                "at the top of the sheet. Generated by tools/reading_order_sheets.py on " + today + ".\n")

    print(f"{len(pages)} pages written to {args.out}")
    print(f"{len(rescaled)} pages with rescaled boxes (ALTO size != image size): " + ", ".join(f"{p} ({r})" for p, r in rescaled))


if __name__ == "__main__":
    main()
