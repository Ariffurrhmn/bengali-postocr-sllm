"""Grad-CAM map figure (third review, item X1): for one chunk, the element-wise
map of every generated token over the OCR tokens of the prompt, for a copying
model (Gemma-2B-it) and a regenerating one (TituLLMs 1B Instruct). Each row is
one generated token, each column one OCR token; rows are scaled to their own
maximum over the OCR text. Open circles mark, for output tokens that copy an
OCR token, the position of that source token, drawn 3 columns to the right so
the map underneath stays visible: where the dark cells run beside them, the
model relied on the token it copied.

Input: the xai_maps_*.jsonl files of the review3b session (xai_gradcam.py
with --save-maps: full maps of every generated token on two chunks).

Usage (from the repo root):
    python paper/scripts/make_attribution_figure.py --maps results/review3b \\
        --page 279_2_D_32_0006 --out paper/figures/fig_attribution.pdf
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PANELS = [("gemma-2b", "Gemma-2B-it"), ("titullm-1b-instruct", "TituLLMs 1B Instruct")]
MARK = "#eb6834"  # categorical slot 2 of the dataviz palette, on the light surface

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.titlesize": 8})


def load(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").split("\n") if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", type=Path, required=True, help="folder with xai_maps_<model>.jsonl")
    ap.add_argument("--page", required=True)
    ap.add_argument("--engine", default="tesseract")
    ap.add_argument("--rows", type=int, default=120, help="first generated tokens shown")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    fig, axes = plt.subplots(1, len(PANELS), figsize=(7.1, 3.2))
    for ax, (key, name) in zip(axes, PANELS):
        rec = next(r for r in load(args.maps / f"xai_maps_{key}.jsonl")
                   if r["page_id"] == args.page and r["engine"] == args.engine)
        mid = f"L{rec['layers'][1]}_hires"
        ocr = rec["ocr_pos"]
        targets = rec["targets"][: args.rows]
        mat = np.zeros((len(targets), len(ocr)))
        for i, t in enumerate(targets):
            v = np.array(t["maps"][mid])
            row = np.array([v[p] if p < len(v) else 0.0 for p in ocr])
            mat[i] = row / row.max() if row.max() > 0 else row
        ax.imshow(mat, aspect="auto", cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
        copied = [(t["copy_of"], i) for i, t in enumerate(targets) if t.get("copy_of") is not None]
        if copied:
            xs, ys = zip(*copied)
            ax.scatter([x + 3 for x in xs], ys, s=3, facecolors="none", edgecolors=MARK, linewidths=0.5, alpha=0.6,
                       label="source of a copied token (shifted right)")
        share = len(copied) / len(targets)
        ax.set_title(f"{name}: {100 * share:.0f}% of the shown tokens copied")
        ax.set_xlabel("OCR token in the prompt")
        ax.set_ylabel("Generated token")
    axes[0].legend(loc="lower left", fontsize=6.5, frameon=True, framealpha=0.9)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
