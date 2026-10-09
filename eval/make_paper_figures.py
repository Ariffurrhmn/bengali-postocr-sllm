"""Paper figures that are diagrams rather than result plots (the result plots
are made by make_figures.py):

  fig_method.pdf/.png  - end-to-end methodology diagram (reviewer item #12)
  fig_dataset.pdf/.png - (a) how the 81 dataset pages become the 15 scored
                         pages, (b) a sample page with its ground-truth text
                         regions drawn on it (reviewer item #36)

Counts in the dataset figure are read from the committed split files, not
typed in, so the figure can't drift from the data.

Usage:
    python make_paper_figures.py --page-image <tif> --page-xml <xml>
    (the paper uses 279_42_B_41_0003, the median-difficulty page also shown
    in examples_*.txt; the dataset itself is not in the repo)
"""
import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "paper" / "figures"

TOTAL_PAGES = 81  # image + PAGE-XML pairs in the REID2019 Bengali release

plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "font.size": 7, "pdf.fonttype": 42})

FILL = {"data": "#e8eef7", "ocr": "#eef5e9", "model": "#fbefe3", "eval": "#f1ecf7", "plain": "#f4f4f4"}
EDGE = "#333333"
TITLE_FS, BODY_FS = 7.2, 6.3
PAD_X, PAD_Y, TITLE_GAP = 5.0, 4.0, 2.5


class Diagram:
    """Boxes laid out in an axes whose data units are typographic points, so
    each box is sized to its measured text instead of a guessed size."""

    def __init__(self, fig, rect):
        self.fig = fig
        self.ax = fig.add_axes(rect)
        w_in, h_in = fig.get_size_inches()
        self.W, self.H = rect[2] * w_in * 72, rect[3] * h_in * 72
        self.ax.set_xlim(0, self.W)
        self.ax.set_ylim(0, self.H)
        self.ax.axis("off")
        self.renderer = fig.canvas.get_renderer()
        self.boxes = {}

    def _measure(self, text, fs, weight="normal"):
        t = self.ax.text(0, 0, text, fontsize=fs, fontweight=weight, linespacing=1.3)
        bb = t.get_window_extent(renderer=self.renderer)
        t.remove()
        return bb.width * 72 / self.fig.dpi, bb.height * 72 / self.fig.dpi

    def size(self, title, body=None, min_w=0):
        tw, th = self._measure(title, TITLE_FS, "bold")
        bw, bh = self._measure(body, BODY_FS) if body else (0, 0)
        w = max(tw, bw, min_w) + 2 * PAD_X
        h = th + (TITLE_GAP + bh if body else 0) + 2 * PAD_Y
        return w, h

    def add(self, key, cx, cy, title, body=None, kind="plain", min_w=0):
        w, h = self.size(title, body, min_w)
        x, y = cx - w / 2, cy - h / 2
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=3",
                                         fc=FILL[kind], ec=EDGE, lw=0.7))
        th = self._measure(title, TITLE_FS, "bold")[1]
        self.ax.text(cx, y + h - PAD_Y, title, ha="center", va="top", fontsize=TITLE_FS,
                     fontweight="bold")
        if body:
            self.ax.text(cx, y + h - PAD_Y - th - TITLE_GAP, body, ha="center", va="top",
                         fontsize=BODY_FS, linespacing=1.3)
        self.boxes[key] = (x, y, w, h)

    def anchor(self, key, side):
        x, y, w, h = self.boxes[key]
        return {"l": (x, y + h / 2), "r": (x + w, y + h / 2),
                "t": (x + w / 2, y + h), "b": (x + w / 2, y)}[side]

    def arrow(self, a, b, ls="-", rad=0.0):
        self.ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=7, lw=0.8,
                                          color=EDGE, linestyle=ls, shrinkA=1.5, shrinkB=1.5,
                                          connectionstyle=f"arc3,rad={rad}"))


def method_figure():
    fig = plt.figure(figsize=(7.16, 1.85))
    d = Diagram(fig, [0, 0, 1, 1])
    ocr_a = ("Tesseract 5", "Bengali, tessdata_best\ndefault settings")
    ocr_b = ("EasyOCR 1.7", "Bengali\ndefault settings")
    nodes = [
        ("img", "Page image", "British Library\nhistorical Bengali print\n(REID2019)\n15 held-out pages", "data"),
        ("ocr", None, None, "ocr"),
        ("raw", "Raw OCR text", "one per page\nper engine", "plain"),
        ("chunk", "Chunking", "~250 tokens,\nsplit at line breaks;\ninputs < 10 chars\nkept unchanged", "plain"),
        ("model", "Zero-shot correction", "one fixed instruction\n(BanglaT5: text only)\ngreedy decoding\nCPU only, bf16\n\nGemma 2B, Llama 3.2 1B,\nTituLLMs 1B, BanglaT5", "model"),
        ("eval", "Evaluation", "vs PAGE-XML ground truth\nNFC + whitespace normalised\nCER, WER, cMER\npaired bootstrap,\nWilcoxon + Holm\nfailure diagnostics", "eval"),
    ]
    widths = [max(d.size(*ocr_a)[0], d.size(*ocr_b)[0]) if key == "ocr" else d.size(title, body)[0]
              for key, title, body, _ in nodes]
    gap = (d.W - 8 - sum(widths)) / (len(widths) - 1)
    cy = d.H * 0.40
    x = 4.0
    for (key, title, body, kind), w in zip(nodes, widths):
        cx = x + w / 2
        if key == "ocr":
            inner = w - 2 * PAD_X
            d.add("tess", cx, cy + d.size(*ocr_a)[1] / 2 + 4, *ocr_a, kind="ocr", min_w=inner)
            d.add("easy", cx, cy - d.size(*ocr_b)[1] / 2 - 4, *ocr_b, kind="ocr", min_w=inner)
        else:
            d.add(key, cx, cy, title, body, kind)
        x += w + gap

    d.arrow(d.anchor("img", "r"), d.anchor("tess", "l"))
    d.arrow(d.anchor("img", "r"), d.anchor("easy", "l"))
    d.arrow(d.anchor("tess", "r"), d.anchor("raw", "l"))
    d.arrow(d.anchor("easy", "r"), d.anchor("raw", "l"))
    d.arrow(d.anchor("raw", "r"), d.anchor("chunk", "l"))
    d.arrow(d.anchor("chunk", "r"), d.anchor("model", "l"))
    d.arrow(d.anchor("model", "r"), d.anchor("eval", "l"))
    # Raw OCR is scored too: it is the no-correction baseline every model is compared against.
    d.arrow(d.anchor("raw", "t"), d.anchor("eval", "t"), ls="--", rad=-0.16)
    (rx, ry), (ex, ey) = d.anchor("raw", "t"), d.anchor("eval", "t")
    # arc3 is a quadratic curve; its apex sits half the control-point offset above the chord.
    apex_y = max(ry, ey) + 0.5 * 0.16 * (ex - rx)
    d.ax.text((rx + ex) / 2, apex_y + 2, "no-correction baseline: raw OCR scored directly",
              ha="center", va="bottom", fontsize=BODY_FS, style="italic")

    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"fig_method.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def read_ids(path: Path) -> list[str]:
    return [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def region_polygons(xml_path: Path):
    """TextRegion polygons from a PAGE-XML file, plus whether each region has
    transcribed text (the ground truth used for scoring)."""
    root = ET.parse(xml_path).getroot()
    ns = re.match(r"\{(.*)\}", root.tag).group(1)
    page = root.find(f"{{{ns}}}Page")
    size = (int(page.get("imageWidth")), int(page.get("imageHeight")))
    regions = []
    for region in page.iter(f"{{{ns}}}TextRegion"):
        coords = region.find(f"{{{ns}}}Coords")
        pts = [tuple(map(int, xy.split(","))) for xy in coords.get("points").split()]
        text = "".join(t.text or "" for t in region.iter(f"{{{ns}}}Unicode"))
        regions.append((pts, bool(text.strip())))
    return size, regions


def dataset_figure(page_image: Path, page_xml: Path):
    excluded = read_ids(REPO_ROOT / "data" / "excluded_no_ground_truth.txt")
    dev = read_ids(REPO_ROOT / "data" / "split_dev.txt")
    eval_ids = read_ids(REPO_ROOT / "data" / "split_eval.txt")
    scored = [json.loads(l)["page_id"] for l in
              (REPO_ROOT / "results" / "ocr_eval.jsonl").read_text(encoding="utf-8").splitlines()]
    usable = TOTAL_PAGES - len(excluded)
    pool = len(dev) + len(eval_ids)

    fig = plt.figure(figsize=(7.16, 3.6))
    d = Diagram(fig, [0.0, 0.0, 0.6, 1.0])
    W, H = d.W, d.H
    main_x, side_x = W * 0.38, W * 0.85
    d.add("all", main_x, H * 0.90, f"{TOTAL_PAGES} page images + PAGE-XML",
          "REID2019 Bengali ground-truth release", "data")
    d.add("usable", main_x, H * 0.73, f"{usable} pages with transcribed text")
    d.add("excl", side_x, H * 0.815, f"{len(excluded)} excluded", "text regions marked,\nbut no transcription")
    d.add("pool", main_x, H * 0.57, f"{pool}-page pool")
    d.add("drop", side_x, H * 0.65, f"{usable - pool} dropped", "thinnest\ntranscriptions")
    d.add("dev", W * 0.19, H * 0.36, f"{len(dev)} development pages",
          "pipeline debugging only;\nnever used for any\nreported number")
    d.add("eval", W * 0.63, H * 0.36, f"{len(eval_ids)} evaluation pages",
          "held out; split frozen and\ncommitted before any\ncorrection model was run", "data")
    d.add("scored", W * 0.63, H * 0.10, f"{len(scored)} evaluation pages scored",
          "random sample, seed 403;\n2 OCR engines x 4 models", "eval")
    d.arrow(d.anchor("all", "b"), d.anchor("usable", "t"))
    d.arrow(d.anchor("usable", "b"), d.anchor("pool", "t"))
    d.arrow(d.anchor("all", "r"), d.anchor("excl", "l"))
    d.arrow(d.anchor("usable", "r"), d.anchor("drop", "l"))
    d.arrow(d.anchor("pool", "b"), d.anchor("dev", "t"))
    d.arrow(d.anchor("pool", "b"), d.anchor("eval", "t"))
    d.arrow(d.anchor("eval", "b"), d.anchor("scored", "t"))
    d.ax.text(2, H - 2, "(a)", fontsize=8, fontweight="bold", va="top")

    # (b) sample page with its ground-truth regions
    (iw, ih), regions = region_polygons(page_xml)
    img = Image.open(page_image).convert("L")
    scale = 900 / max(img.size)
    small = img.resize((int(img.width * scale), int(img.height * scale)))
    axb = fig.add_axes([0.62, 0.06, 0.37, 0.88])
    axb.imshow(small, cmap="gray")
    sx, sy = small.width / iw, small.height / ih
    for pts, has_text in regions:
        axb.add_patch(Polygon([(px * sx, py * sy) for px, py in pts], closed=True, fill=False,
                              ec="#c0392b" if has_text else "#7f8c8d", lw=0.8,
                              ls="-" if has_text else ":"))
    axb.axis("off")
    axb.text(0, -12, "(b)", fontsize=8, fontweight="bold", va="bottom")
    axb.text(small.width / 2, small.height + 8, f"{page_image.stem}: ground-truth text regions (red)",
             ha="center", va="top", fontsize=BODY_FS)

    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"fig_dataset.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--page-image", type=Path, required=True)
    parser.add_argument("--page-xml", type=Path, required=True)
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    method_figure()
    dataset_figure(args.page_image, args.page_xml)
    print(f"Wrote figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
