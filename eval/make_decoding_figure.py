"""Figure for the decoding pitfall (review item F1, third review).

(a) Schematic: in Transformers generate(), repetition_penalty and
    no_repeat_ngram_size act on the whole token sequence, which for a
    decoder-only model includes the prompt and so the OCR text being copied.
(b) Mean CER per model and engine under guarded and plain greedy decoding,
    with the raw OCR as reference, from results/decoding_ablation.json
    (compare_decoding.py). When the 2x2 ablation on the 40-page setup exists,
    point --ablation at its summary instead.

Usage (from eval/):
    python make_decoding_figure.py --ablation ../results/decoding_ablation.json \\
        --out ../paper/figures/fig_decoding.pdf
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

NAMES = {"gemma-2b": "Gemma-2B-it", "llama3.2-1b": "Llama 3.2 1B", "titullm-1b": "TituLLMs 1B",
         "banglat5": "BanglaT5", "qwen3-1.7b": "Qwen3 1.7B"}
ORDER = ["gemma-2b", "llama3.2-1b", "titullm-1b", "banglat5"]
ENGINES = [("tesseract", "Tesseract"), ("easyocr", "EasyOCR")]
# Validated with the dataviz palette checker (light surface): CVD dE 20.
PLAIN, GUARDED, INK, MUTED = "#1f6fb2", "#c4502a", "#222222", "#888888"

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False})


def schematic(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0.3, 3.2)
    ax.axis("off")
    boxes = [(0.1, 2.2, "Instruction", "#eeeeee"), (2.5, 4.3, "OCR text (to be copied)", "#dbe8f5"),
             (6.95, 2.95, "Generated output", "#f7e1d8")]
    for x, w, label, fc in boxes:
        ax.add_patch(FancyBboxPatch((x, 1.25), w, 0.7, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc=fc, ec=MUTED, lw=0.6))
        ax.text(x + w / 2, 1.6, label, ha="center", va="center", color=INK)

    def bracket(x0, x1, y, text, color, up):
        d = 0.12 if up else -0.12
        ax.plot([x0, x0, x1, x1], [y, y + d, y + d, y], color=color, lw=1.2)
        ax.text((x0 + x1) / 2, y + d * 2.2, text, ha="center", va="bottom" if up else "top",
                color=INK, fontsize=7)

    bracket(0.1, 9.9, 2.5, "Scope of repetition penalty and n-gram ban in generate(): whole sequence",
            GUARDED, True)
    bracket(0.1, 6.8, 1.05, "Prompt", MUTED, False)
    bracket(6.95, 9.9, 1.05, "Scope a correction task needs", PLAIN, False)
    ax.text(-0.05, 3.2, "(a)", fontweight="bold", va="top", ha="right")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = {(r["engine"], r["model"]): r for r in json.loads(args.ablation.read_text(encoding="utf-8"))}
    models = [m for m in ORDER if all((e, m) in rows for e, _ in ENGINES)]

    fig = plt.figure(figsize=(7.1, 3.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.75, 1.6], hspace=0.12, wspace=0.08)
    schematic(fig.add_subplot(gs[0, :]))
    xmax = max(max(r["guarded_mean"], r["plain_mean"]) for r in rows.values()) * 1.08
    for k, (e, name) in enumerate(ENGINES):
        ax = fig.add_subplot(gs[1, k])
        raw = rows[(e, models[0])]["ocr_mean"]
        ax.axvline(raw, color=MUTED, lw=1, ls="--")
        ax.text(raw, len(models) - 0.35, " raw OCR %.3f" % raw, color=MUTED, fontsize=7, va="bottom")
        for i, m in enumerate(models):
            y = len(models) - 1 - i
            r = rows[(e, m)]
            ax.plot([r["plain_mean"], r["guarded_mean"]], [y, y], color="#bbbbbb", lw=2, zorder=1)
            ax.scatter(r["guarded_mean"], y, s=36, marker="s", color=GUARDED, zorder=3,
                       edgecolor="white", linewidth=1, label="Guarded" if i == 0 else None)
            ax.scatter(r["plain_mean"], y, s=36, marker="o", color=PLAIN, zorder=3,
                       edgecolor="white", linewidth=1, label="Plain greedy" if i == 0 else None)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels([NAMES[m] for m in reversed(models)] if k == 0 else [])
        ax.set_ylim(-0.6, len(models) - 0.1)
        ax.set_xlim(0, xmax)
        ax.set_xlabel("Mean CER (%s input, 15 pages)" % name)
        ax.grid(axis="x", color="#e5e5e5", lw=0.6)
        ax.set_axisbelow(True)
        if k == 0:
            ax.text(-0.32, 1.02, "(b)", transform=ax.transAxes, fontweight="bold")
        if k == 1:
            ax.legend(loc="lower right", frameon=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
