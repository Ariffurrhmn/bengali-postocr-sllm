"""Figure for the decoding pitfall (review item F1, third review).

(a) Schematic: in Transformers generate(), repetition_penalty and
    no_repeat_ngram_size act on the whole token sequence, which for a
    decoder-only model includes the prompt and so the OCR text being copied.
(b) Mean CER per model on Tesseract, EasyOCR and clean-text input under plain
    greedy decoding, the penalty alone, the n-gram ban alone and both (the 2x2
    of review item C4, 40 pages), from results/review3_runs.json
    (review3_runs_analysis.py), with the raw OCR as reference.

Usage (from eval/):
    python make_decoding_figure.py --runs ../results/review3_runs.json \\
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
         "banglat5": "BanglaT5", "qwen3-1.7b": "Qwen3 1.7B",
         "phi4-mini": "Phi-4-mini"}
ORDER = ["gemma-2b", "qwen3-1.7b", "phi4-mini", "llama3.2-1b"]
INPUTS = [("tesseract", "Tesseract"), ("easyocr", "EasyOCR"), ("ground_truth", "clean text")]
# Categorical slots 1-4 of the dataviz reference palette, validated on the light
# surface (worst adjacent CVD dE 9.1); marker shapes carry identity as well.
PLAIN, GUARDED, INK, MUTED = "#2a78d6", "#eb6834", "#222222", "#888888"
PRESETS = [("plain", "Plain greedy", PLAIN, "o"), ("penalty_only", "Penalty only", "#1baf7a", "^"),
           ("ngram_only", "n-gram ban only", "#eda100", "D"), ("guarded", "Both (guarded)", GUARDED, "s")]

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
    parser.add_argument("--runs", type=Path, required=True,
                        help="review3_runs.json (review3_runs_analysis.py): the 40-page 2x2 (C4)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    c4 = {(c["model"], c["engine"], c["preset"]): c
          for c in json.loads(args.runs.read_text(encoding="utf-8"))["C4"]}
    models = [m for m in ORDER if any(k[0] == m for k in c4)]

    fig = plt.figure(figsize=(7.1, 3.1))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.7, 1.6], hspace=0.14, wspace=0.08)
    schematic(fig.add_subplot(gs[0, :]))
    xmax = max(c["mean_a"] for c in c4.values()) * 1.05
    for k, (e, name) in enumerate(INPUTS):
        ax = fig.add_subplot(gs[1, k])
        raws = [c["raw_cer"] for (m, e2, _), c in c4.items() if e2 == e]
        if e != "ground_truth":
            ax.axvline(raws[0], color=MUTED, lw=1, ls="--")
            ax.text(raws[0], len(models) - 0.45, " raw OCR %.3f" % raws[0], color=MUTED, fontsize=6.5,
                    va="bottom")
        for i, m in enumerate(models):
            y = len(models) - 1 - i
            if not all((m, e, p) in c4 for p, _, _, _ in PRESETS[1:]):
                ax.text(0.5 * xmax, y, "not run", color=MUTED, fontsize=6.5, va="center")
                continue
            plain = c4[(m, e, "guarded")]["mean_b"]
            vals = [plain] + [c4[(m, e, p)]["mean_a"] for p, _, _, _ in PRESETS[1:]]
            ax.plot([min(vals), max(vals)], [y, y], color="#d0d0d0", lw=1.5, zorder=1)
            # Small vertical offsets keep coinciding markers visible.
            for j, ((p, label, color, marker), v) in enumerate(zip(PRESETS, vals)):
                ax.scatter(v, y + 0.27 - 0.18 * j, s=30, marker=marker, color=color, zorder=3, edgecolor="white",
                           linewidth=1, label=label if i == 0 else None)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels([NAMES[m] for m in reversed(models)] if k == 0 else [])
        ax.set_ylim(-0.6, len(models) - 0.1)
        ax.set_xlim(0, xmax)
        ax.set_xlabel("Mean CER, %s" % name)
        ax.grid(axis="x", color="#e5e5e5", lw=0.6)
        ax.set_axisbelow(True)
        if k == 0:
            ax.text(-0.42, 1.02, "(b)", transform=ax.transAxes, fontweight="bold")
    handles, labels = fig.axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
