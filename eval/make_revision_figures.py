"""Result figures for the revised paper, from the 40-page GPU run.

  fig_delta.png    - mean change in CER per model and engine with paired
                     bootstrap 95% CIs; left: all models, right: the models
                     near zero on a finer scale (replaces old Figs. 3-4,
                     review item F1)
  fig_scatter.png  - per-page change in CER against baseline CER, one panel
                     per model (review items F2, F5)
  fig_corrected.png - per-page corrected CER against baseline CER with the
                     identity line, one panel per model (review item E3)

Usage:
    python make_revision_figures.py --ocr-path ../results/ocr_eval40.jsonl \\
        --correction-path ../results/gpu_eval40_ocr.jsonl \\
        --bootstrap-json ../results/bootstrap_gpu40.json --out-dir ../paper/figures
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from metrics import cer

# Fixed order best-to-worst so every figure tells the story the same way.
_ORDER = ["symspell", "gemma-2b", "qwen3-1.7b", "phi4-mini", "llama3.2-1b", "banglat5",
          "titullm-1b", "titullm-1b-instruct", "titullm-3b-instruct"]
MODEL_LABELS = {
    "symspell": "SymSpell (dictionary)",
    "gemma-2b": "Gemma-2B-it",
    "qwen3-1.7b": "Qwen3 1.7B",
    "phi4-mini": "Phi-4-mini",
    "titullm-1b-instruct": "TituLLMs 1B Inst.",
    "titullm-3b-instruct": "TituLLMs 3B Inst.",
    "llama3.2-1b": "Llama 3.2 1B",
    "banglat5": "BanglaT5",
    "titullm-1b": "TituLLMs 1B",
}

ENGINES = [("tesseract", "Tesseract", "#4C72B0", "o"), ("easyocr", "EasyOCR", "#DD8452", "s")]
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8})


def fig_delta(boot, models, out_dir):
    near = [m for m in models
            if max(abs(boot[(e, m)]["mean_delta"]) for e, *_ in ENGINES) < 0.1]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.9),
                             gridspec_kw={"width_ratios": [1.25, 1]})
    for ax, subset, title in [(axes[0], models, "All models"),
                              (axes[1], near, "Models near zero (finer scale)")]:
        y = np.arange(len(subset))[::-1]
        for k, (e, name, color, marker) in enumerate(ENGINES):
            off = 0.15 if k == 0 else -0.15
            for yi, m in zip(y, subset):
                r = boot[(e, m)]
                ax.errorbar(r["mean_delta"], yi + off,
                            xerr=[[r["mean_delta"] - r["ci_low"]], [r["ci_high"] - r["mean_delta"]]],
                            fmt=marker, color=color, ms=4, capsize=2, lw=1,
                            label=name if yi == y[0] else None)
        ax.axvline(0, color="#333333", lw=1)
        ax.set_yticks(y)
        ax.set_yticklabels([MODEL_LABELS[m] for m in subset])
        ax.set_xlabel("Change in CER (corrected $-$ raw OCR)")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.3)
    axes[0].legend(loc="upper right", frameon=True)
    fig.tight_layout()
    path = out_dir / "fig_delta.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_scatter(pages, outputs, models, out_dir):
    cols = 4
    rows_n = int(np.ceil(len(models) / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(7.16, 1.9 * rows_n), sharex=True)
    axes = np.atleast_2d(axes)
    for ax, m in zip(axes.flat, models):
        for e, name, color, marker in ENGINES:
            pids = sorted(outputs[(e, m)])
            base = np.array([cer(pages[p]["ground_truth"], pages[p][e]) for p in pids])
            corr = np.array([cer(pages[p]["ground_truth"], outputs[(e, m)][p]) for p in pids])
            ax.scatter(base, corr - base, s=9, color=color, marker=marker, alpha=0.75,
                       linewidths=0, label=name)
        ax.axhline(0, color="#333333", lw=0.8)
        ax.set_title(MODEL_LABELS[m])
        ax.grid(alpha=0.3)
    for ax in axes.flat[len(models):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("Baseline CER")
    for ax in axes[:, 0]:
        ax.set_ylabel("Change in CER")
    axes.flat[0].legend(loc="upper right", markerscale=1.5, handletextpad=0.2)
    fig.tight_layout()
    path = out_dir / "fig_scatter.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def fig_corrected(pages, outputs, models, out_dir):
    """Corrected CER against baseline CER, with the identity line: points on it
    are pages the method left as they were, below it pages it improved
    (review item E3; added after the second review)."""
    cols = 4
    rows_n = int(np.ceil(len(models) / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(7.16, 1.9 * rows_n))
    axes = np.atleast_2d(axes)
    for ax, m in zip(axes.flat, models):
        for e, name, color, marker in ENGINES:
            pids = sorted(outputs[(e, m)])
            base = np.array([cer(pages[p]["ground_truth"], pages[p][e]) for p in pids])
            corr = np.array([cer(pages[p]["ground_truth"], outputs[(e, m)][p]) for p in pids])
            ax.scatter(base, corr, s=9, color=color, marker=marker, alpha=0.75,
                       linewidths=0, label=name)
        ax.plot([0, 1.05], [0, 1.05], color="#333333", lw=0.8)
        ax.set_xlim(0, 1.05)
        ax.set_title(MODEL_LABELS[m])
        ax.grid(alpha=0.3)
    for ax in axes.flat[len(models):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("Baseline CER")
    for ax in axes[:, 0]:
        ax.set_ylabel("Corrected CER")
    axes.flat[0].legend(loc="upper left", markerscale=1.5, handletextpad=0.2)
    fig.tight_layout()
    path = out_dir / "fig_corrected.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ocr-path", type=Path, required=True)
    parser.add_argument("--correction-path", type=Path, required=True)
    parser.add_argument("--bootstrap-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    pages = {}
    for line in args.ocr_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            pages[r["page_id"]] = r
    outputs = {}
    for line in args.correction_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            outputs.setdefault((r["engine"], r["model_key"]), {})[r["page_id"]] = r["raw_output"]
    boot = {(r["engine"], r["model"]): r
            for r in json.loads(args.bootstrap_json.read_text(encoding="utf-8"))["results"]
            if r["metric"] == "cer"}
    models = [m for m in _ORDER if all((e, m) in boot for e, *_ in ENGINES)]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print("wrote", fig_delta(boot, models, args.out_dir))
    print("wrote", fig_scatter(pages, outputs, models, args.out_dir))
    print("wrote", fig_corrected(pages, outputs, models, args.out_dir))


if __name__ == "__main__":
    main()
