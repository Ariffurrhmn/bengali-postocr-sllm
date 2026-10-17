"""Noise-sweep figure (third review, items M3/F7): mean output CER against mean
input CER for the four general-purpose models, from matrix_analysis.py
summaries. Synthetic inputs (noise40.jsonl levels) are joined by lines; the
clean text and the real OCR of the main run, passed with --real, are drawn as
open markers so the synthetic curve can be checked against real OCR noise.
Points below the identity line are inputs the model improved.

Usage (from the repo root):
    python paper/scripts/make_noise_figure.py --sweep results/noise_runs40.json \\
        --real results/main_general40.json --out paper/figures/fig_noise.pdf
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAMES = {"gemma-2b": "Gemma-2B-it", "qwen3-1.7b": "Qwen3 1.7B", "phi4-mini": "Phi-4-mini",
         "llama3.2-1b": "Llama 3.2 1B"}
ORDER = ["gemma-2b", "qwen3-1.7b", "phi4-mini", "llama3.2-1b"]
# Categorical slots 1-4 of the dataviz reference palette, validated on the light
# surface (worst adjacent CVD dE 9.1); marker shapes carry identity as well.
COLORS = dict(zip(ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))
MARKERS = dict(zip(ORDER, ["o", "s", "^", "D"]))
REAL = {"ground_truth": "clean", "tesseract": "Tess.", "easyocr": "Easy."}
INK, MUTED = "#222222", "#888888"

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", type=Path, required=True)
    ap.add_argument("--real", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    sweep = [c for c in json.loads(args.sweep.read_text(encoding="utf-8"))["cells"] if c["complete"]]
    real = ([c for c in json.loads(args.real.read_text(encoding="utf-8"))["cells"]
             if c["engine"] in REAL and c["model"] in ORDER] if args.real else [])
    sweep = [c for c in sweep if c["model"] in ORDER]
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    top = max([c["cer"]["output"] for c in sweep + real] + [c["cer"]["input"] for c in sweep + real]) * 1.08
    ax.plot([0, top], [0, top], color=MUTED, lw=1, ls="--", zorder=1)
    ax.text(top * 0.97, top * 0.86, "no change", color=MUTED, fontsize=7, ha="right", va="top", rotation=45)
    for m in ORDER:
        pts = sorted((c["cer"]["input"], c["cer"]["output"]) for c in sweep if c["model"] == m)
        if pts:
            ax.plot(*zip(*pts), color=COLORS[m], lw=2, marker=MARKERS[m], ms=5, mec="white", mew=1,
                    label=NAMES[m], zorder=3)
        for c in (c for c in real if c["model"] == m):
            ax.scatter(c["cer"]["input"], c["cer"]["output"], s=30, marker=MARKERS[m], facecolors="white",
                       edgecolors=COLORS[m], linewidths=1.2, zorder=4)
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_aspect("equal")
    ax.set_xlabel("Mean CER of the input")
    ax.set_ylabel("Mean CER after correction")
    ax.grid(color="#e5e5e5", lw=0.6)
    ax.set_axisbelow(True)
    handles, labels = ax.get_legend_handles_labels()
    if real:
        handles.append(plt.Line2D([], [], marker="o", ls="", mfc="white", mec=INK, ms=5))
        labels.append("clean text / real OCR (main run)")
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
