"""Generates the evaluation matrix and curves for the Results section.

Produces four figures plus a CSV of the evaluation matrix:
  fig1_cer_by_model     - mean CER per model/engine against the baseline
  fig2_bootstrap_ci     - mean CER delta with 95% bootstrap CIs (the key figure)
  fig3_per_page         - per-page CER, showing the effect holds page by page
  fig4_truncation       - truncation rate against CER (the secondary finding)
  evaluation_matrix.csv - the full numeric table
"""
import argparse
import csv
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: no display needed, just write files
import matplotlib.pyplot as plt
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from metrics import score_pair

REPO_ROOT = Path(__file__).resolve().parent.parent

# Fixed order best-to-worst so every figure tells the story the same way.
MODEL_ORDER = ["gemma-2b", "llama3.2-1b", "banglat5", "titullm-1b"]
MODEL_LABELS = {
    "gemma-2b": "Gemma 2B",
    "llama3.2-1b": "Llama 3.2 1B",
    "banglat5": "BanglaT5",
    "titullm-1b": "TituLLM 1B",
}
ENGINE_LABELS = {"tesseract": "Tesseract", "easyocr": "EasyOCR"}
ENGINE_COLORS = {"tesseract": "#4C72B0", "easyocr": "#DD8452"}
BASELINE_COLORS = {"tesseract": "#2A4D7A", "easyocr": "#A85B32"}


def load(ocr_path: Path, correction_path: Path):
    ground_truth, raw_ocr = {}, {}
    for line in ocr_path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        ground_truth[r["page_id"]] = r["ground_truth"]
        raw_ocr[(r["page_id"], "tesseract")] = r["tesseract"]
        raw_ocr[(r["page_id"], "easyocr")] = r["easyocr"]

    baseline = defaultdict(dict)
    for (page_id, engine), text in raw_ocr.items():
        baseline[engine][page_id] = score_pair(ground_truth[page_id], text)

    cells = defaultdict(dict)
    flags = defaultdict(lambda: {"truncated": 0, "skipped": 0, "n": 0})
    for line in correction_path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        key = (r["engine"], r["model_key"])
        cells[key][r["page_id"]] = score_pair(
            ground_truth[r["page_id"]], r["raw_output"]
        )
        flags[key]["n"] += 1
        if r.get("truncated"):
            flags[key]["truncated"] += 1
        if r.get("skipped"):
            flags[key]["skipped"] += 1

    return baseline, cells, flags


def fig1_cer_by_model(baseline, cells, out_dir):
    """Grouped bars: absolute CER per model, with baseline as a reference line."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(MODEL_ORDER))
    width = 0.38

    for i, engine in enumerate(("tesseract", "easyocr")):
        means = [np.mean(list(
            s["cer"] for s in cells[(engine, m)].values()
        )) for m in MODEL_ORDER]
        offset = (i - 0.5) * width
        bars = ax.bar(
            x + offset, means, width,
            label=f"{ENGINE_LABELS[engine]} + correction",
            color=ENGINE_COLORS[engine], edgecolor="white", linewidth=0.8,
        )
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)

        base = np.mean([s["cer"] for s in baseline[engine].values()])
        ax.axhline(
            base, color=BASELINE_COLORS[engine], linestyle="--", linewidth=1.6,
            label=f"{ENGINE_LABELS[engine]} baseline (no correction) = {base:.3f}",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER])
    ax.set_ylabel("Mean CER (lower is better)")
    ax.set_title(
        "Post-OCR correction CER vs. uncorrected baseline\n"
        "Bengali historical print, 15 evaluation pages, zero-shot chunked correction",
        fontsize=11,
    )
    ax.legend(fontsize=8.5, framealpha=0.95)
    ax.grid(axis="y", alpha=0.3, linewidth=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    path = out_dir / "fig1_cer_by_model.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def fig2_bootstrap_ci(bootstrap_results, out_dir):
    """The key figure: mean CER delta with 95% CIs. Zero line = no effect."""
    rows = [r for r in bootstrap_results if r["metric"] == "cer"]
    order = [(e, m) for m in MODEL_ORDER for e in ("tesseract", "easyocr")]
    lookup = {(r["engine"], r["model"]): r for r in rows}

    fig, ax = plt.subplots(figsize=(9, 5.5))
    y = np.arange(len(order))
    for i, key in enumerate(order):
        r = lookup[key]
        engine = r["engine"]
        # Error bar spans the CI; the marker is the observed mean delta.
        ax.errorbar(
            r["mean_delta"], i,
            xerr=[[r["mean_delta"] - r["ci_low"]], [r["ci_high"] - r["mean_delta"]]],
            fmt="o", color=ENGINE_COLORS[engine], markersize=7,
            capsize=4, capthick=1.6, elinewidth=1.8,
        )

    ax.axvline(0, color="#333333", linewidth=1.4)
    # Annotate the zero line near the top, clear of the x-axis tick labels.
    ax.text(
        0.02, -0.42, "no effect",
        fontsize=8.5, color="#333333", va="center", ha="left",
    )
    ax.set_yticks(y)
    ax.set_yticklabels([
        f"{MODEL_LABELS[m]} · {ENGINE_LABELS[e]}" for e, m in order
    ])
    ax.set_xlabel("Mean ΔCER (corrected − baseline); positive = correction made it worse")
    ax.set_title(
        "Paired bootstrap: every model significantly increased CER\n"
        "10,000 resamples over 15 pages, 95% percentile intervals",
        fontsize=11,
    )
    handles = [
        plt.Line2D([], [], color=ENGINE_COLORS[e], marker="o", linestyle="",
                   label=ENGINE_LABELS[e])
        for e in ("tesseract", "easyocr")
    ]
    # Place the legend outside the data area so it can't overlap an error bar.
    ax.legend(
        handles=handles, fontsize=8.5, loc="upper left",
        bbox_to_anchor=(1.01, 1.0), borderaxespad=0,
    )
    ax.grid(axis="x", alpha=0.3, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_ylim(len(order) - 0.4, -0.7)  # headroom for the zero-line label
    fig.tight_layout()
    path = out_dir / "fig2_bootstrap_ci.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def fig3_per_page(baseline, cells, out_dir):
    """Per-page CER curves: shows the effect isn't driven by a few bad pages."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    for ax, engine in zip(axes, ("tesseract", "easyocr")):
        pages = sorted(baseline[engine])
        # Sort by baseline difficulty so the x-axis is interpretable.
        pages.sort(key=lambda p: baseline[engine][p]["cer"])
        x = np.arange(len(pages))

        ax.plot(
            x, [baseline[engine][p]["cer"] for p in pages],
            "o-", color="#333333", linewidth=2.2, markersize=5,
            label="Baseline (no correction)", zorder=5,
        )
        for m in MODEL_ORDER:
            ax.plot(
                x, [cells[(engine, m)][p]["cer"] for p in pages],
                "o--", markersize=4, linewidth=1.3, alpha=0.85,
                label=MODEL_LABELS[m],
            )

        ax.set_title(f"{ENGINE_LABELS[engine]}", fontsize=11)
        ax.set_xlabel("Evaluation page (sorted by baseline CER)")
        ax.set_xticks(x)
        ax.set_xticklabels([str(i + 1) for i in x], fontsize=8)
        ax.grid(alpha=0.3, linewidth=0.7)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("CER (lower is better)")
    axes[0].legend(fontsize=8.5, framealpha=0.95)
    fig.suptitle(
        "Per-page CER: correction sits above the baseline on essentially every page",
        fontsize=12,
    )
    fig.tight_layout()
    path = out_dir / "fig3_per_page.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def fig4_truncation(cells, flags, out_dir):
    """Secondary finding: truncation rate tracks CER almost perfectly."""
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    xs, ys = [], []
    for m in MODEL_ORDER:
        for engine in ("tesseract", "easyocr"):
            f = flags[(engine, m)]
            rate = 100.0 * f["truncated"] / f["n"]
            cer_mean = np.mean([s["cer"] for s in cells[(engine, m)].values()])
            xs.append(rate)
            ys.append(cer_mean)
            ax.scatter(
                rate, cer_mean, s=95, color=ENGINE_COLORS[engine],
                edgecolor="white", linewidth=1.2, zorder=5,
            )
            # The two engines often land near-identically; push their labels to
            # opposite sides so they never overlap each other.
            if engine == "tesseract":
                xytext, ha = (9, 6), "left"
            else:
                xytext, ha = (-9, -10), "right"
            ax.annotate(
                f"{MODEL_LABELS[m]} ({ENGINE_LABELS[engine]})",
                (rate, cer_mean), textcoords="offset points", xytext=xytext,
                fontsize=7.5, color="#333333", ha=ha,
            )

    r = np.corrcoef(xs, ys)[0, 1]
    ax.set_xlabel("Truncation rate (% of pages hitting the generation cap)")
    ax.set_ylabel("Mean CER")
    ax.set_title(
        f"Truncation predicts degenerate output (Pearson r = {r:.2f})\n"
        "Models that run to the token cap are the ones that score worst",
        fontsize=11,
    )
    ax.grid(alpha=0.3, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.margins(0.18)
    handles = [
        plt.Line2D([], [], color=ENGINE_COLORS[e], marker="o", linestyle="",
                   label=ENGINE_LABELS[e])
        for e in ("tesseract", "easyocr")
    ]
    ax.legend(handles=handles, fontsize=8.5, loc="upper left")
    fig.tight_layout()
    path = out_dir / "fig4_truncation.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path, r


def write_matrix_csv(baseline, cells, flags, bootstrap_results, out_dir):
    """The evaluation matrix as a CSV, for pasting into the Results section."""
    lookup = {
        (r["engine"], r["model"], r["metric"]): r for r in bootstrap_results
    }
    path = out_dir / "evaluation_matrix.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([
            "engine", "model", "n_pages",
            "baseline_cer", "corrected_cer", "delta_cer",
            "cer_ci_low", "cer_ci_high", "cer_significant",
            "baseline_wer", "corrected_wer", "delta_wer",
            "wer_ci_low", "wer_ci_high", "wer_significant",
            "truncated", "skipped", "truncation_rate_pct",
        ])
        for engine in ("tesseract", "easyocr"):
            base_cer = np.mean([s["cer"] for s in baseline[engine].values()])
            base_wer = np.mean([s["wer"] for s in baseline[engine].values()])
            w.writerow([
                engine, "BASELINE (no correction)", len(baseline[engine]),
                f"{base_cer:.4f}", "", "", "", "", "",
                f"{base_wer:.4f}", "", "", "", "", "", "", "", "",
            ])
            for m in MODEL_ORDER:
                c = lookup[(engine, m, "cer")]
                v = lookup[(engine, m, "wer")]
                f = flags[(engine, m)]
                w.writerow([
                    engine, m, c["n_pages"],
                    f"{c['baseline_mean']:.4f}", f"{c['corrected_mean']:.4f}",
                    f"{c['mean_delta']:+.4f}",
                    f"{c['ci_low']:+.4f}", f"{c['ci_high']:+.4f}",
                    "yes" if c["significant"] else "no",
                    f"{v['baseline_mean']:.4f}", f"{v['corrected_mean']:.4f}",
                    f"{v['mean_delta']:+.4f}",
                    f"{v['ci_low']:+.4f}", f"{v['ci_high']:+.4f}",
                    "yes" if v["significant"] else "no",
                    f["truncated"], f["skipped"],
                    f"{100.0 * f['truncated'] / f['n']:.1f}",
                ])
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "eval"], default="eval")
    parser.add_argument("--ocr-path", type=Path, default=None)
    parser.add_argument("--correction-path", type=Path, default=None)
    parser.add_argument("--bootstrap-json", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    ocr_path = args.ocr_path or (REPO_ROOT / "results" / f"ocr_{args.split}.jsonl")
    correction_path = args.correction_path or (
        REPO_ROOT / "results" / f"correction_{args.split}.jsonl"
    )
    bootstrap_json = args.bootstrap_json or (
        REPO_ROOT / "results" / f"bootstrap_{args.split}.json"
    )
    out_dir = args.out_dir or (REPO_ROOT / "results" / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline, cells, flags = load(ocr_path, correction_path)
    bootstrap_results = json.loads(
        bootstrap_json.read_text(encoding="utf-8")
    )["results"]

    print(f"Writing figures to {out_dir}\n")
    print(f"  {fig1_cer_by_model(baseline, cells, out_dir).name}")
    print(f"  {fig2_bootstrap_ci(bootstrap_results, out_dir).name}")
    print(f"  {fig3_per_page(baseline, cells, out_dir).name}")
    path4, r = fig4_truncation(cells, flags, out_dir)
    print(f"  {path4.name}   (truncation/CER correlation r={r:.3f})")
    print(f"  {write_matrix_csv(baseline, cells, flags, bootstrap_results, out_dir).name}")


if __name__ == "__main__":
    main()
