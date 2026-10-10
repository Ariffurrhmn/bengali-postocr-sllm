"""LaTeX tables for the revised paper, generated from the result files so no
number is typed by hand.

  paper/tables/baseline.tex     - uncorrected OCR baselines (CER, WER, cMER)
  paper/tables/eval_matrix.tex  - full evaluation matrix: deltas, bootstrap CIs,
                                  Holm-corrected Wilcoxon p, truncation
                                  (full text width, so no column is cut off;
                                  reviewer item #43)
  paper/tables/diagnostics.tex  - output diagnostics (length, script, retention)
  paper/tables/damage.tex       - extended analysis: cMER, words broken/fixed,
                                  change ratio, invented runs, length safeguard

The best corrected model per engine and column is marked with \\best{}
(reviewer item #42). Macros are in paper/tables/preamble.tex.

  paper/tables/decoding.tex     - decoding ablation: guarded vs plain decoding
                                  (review item C1)

Main tables use the plain-decoding runs (repetition_penalty 1.0, no n-gram
ban); the original guarded runs appear only in decoding.tex.

Inputs (defaults): results/bootstrap_plain.json, results/extended_plain.json
(run bootstrap.py and extended_analysis.py on ablation_plain_decoding.jsonl
first), results/ocr_eval.jsonl, results/ablation_plain_decoding.jsonl,
results/decoding_ablation.json (from compare_decoding.py).
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

# Imported before anything prints: inspect_examples re-wraps stdout for UTF-8.
from inspect_examples import MODEL_ORDER, script_profile, token_retention

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"
OUT_DIR = REPO_ROOT / "paper" / "tables"

ENGINES = [("tesseract", "Tesseract"), ("easyocr", "EasyOCR")]
MODEL_NAMES = {"gemma-2b": "Gemma 2B", "llama3.2-1b": "Llama 3.2 1B",
               "banglat5": "BanglaT5", "titullm-1b": "TituLLMs 1B",
               "qwen3-1.7b": "Qwen3 1.7B", "phi4-mini": "Phi-4-mini",
               "titullm-1b-instruct": "TituLLMs 1B Instruct",
               "titullm-3b-instruct": "TituLLMs 3B Instruct"}


def mark(values: dict, key, text: str, lower_is_better=True) -> str:
    """Wrap text in \\best{} if values[key] is the best of the dict."""
    target = min(values.values()) if lower_is_better else max(values.values())
    return f"\\best{{{text}}}" if abs(values[key] - target) < 1e-12 else text


def fmt_p(p: float) -> str:
    if p < 0.001:
        return "$<$0.001"
    return f"{p:.3f}" if p < 0.05 else f"{p:.2f}\\textsuperscript{{ns}}"


def load_jsonl(path: Path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--boot", type=Path, default=RESULTS / "bootstrap_plain.json")
    parser.add_argument("--ext", type=Path, default=RESULTS / "extended_plain.json")
    parser.add_argument("--corrections", type=Path,
                        default=RESULTS / "ablation_plain_decoding.jsonl")
    parser.add_argument("--decoding", type=Path, default=RESULTS / "decoding_ablation.json")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    boot = json.loads(args.boot.read_text(encoding="utf-8"))
    ext = json.loads(args.ext.read_text(encoding="utf-8"))
    ocr = {r["page_id"]: r for r in load_jsonl(RESULTS / "ocr_eval.jsonl")}
    corrections = load_jsonl(args.corrections)

    b = {(r["engine"], r["model"], r["metric"]): r for r in boot["results"]}
    models = [m for m in MODEL_ORDER if all((e, m, "cer") in b for e, _ in ENGINES)]
    n_tests = 2 * len(ENGINES) * len(models)
    sig = {(r["engine"], r["model"], r["metric"]): r for r in ext["significance"]}
    cells = {(r["engine"], r["model"]): r for r in ext["cells"]}
    guard = {(r["engine"], r["model"]): r for r in ext["safeguard"]}
    trunc = defaultdict(int)
    for r in corrections:
        trunc[(r["engine"], r["model_key"])] += int(r["truncated"])
    n_pages = ext["n_pages"]

    # ---- baseline ----
    rows = []
    for e, name in ENGINES:
        base = b[(e, models[0], "cer")]["baseline_mean"]
        wbase = b[(e, models[0], "wer")]["baseline_mean"]
        cbase = sig[(e, models[0], "cmer")]["baseline_mean"]
        rows.append(f"{name} & {base:.3f} & {wbase:.3f} & {cbase:.3f} \\\\")
    (OUT_DIR / "baseline.tex").write_text(
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Uncorrected OCR Baselines (" + str(n_pages) + " Evaluation Pages)}\n"
        "\\label{tab:baseline}\n"
        "\\begin{tabular}{lccc}\n\\toprule\n"
        "Engine & CER & WER & cMER \\\\\n\\midrule\n" + "\n".join(rows) + "\n"
        "\\bottomrule\n\\end{tabular}\n\\end{table}\n", encoding="utf-8")

    # ---- full evaluation matrix ----
    body = []
    for i, (e, name) in enumerate(ENGINES):
        cer_abs = {m: b[(e, m, "cer")]["corrected_mean"] for m in models}
        wer_abs = {m: b[(e, m, "wer")]["corrected_mean"] for m in models}
        cer_d = {m: b[(e, m, "cer")]["mean_delta"] for m in models}
        wer_d = {m: b[(e, m, "wer")]["mean_delta"] for m in models}
        tr = {m: trunc[(e, m)] / n_pages for m in models}
        first = b[(e, models[0], "cer")]
        body.append(f"{name} & (no correction) & {first['baseline_mean']:.3f} & --- & --- & "
                    f"{b[(e, models[0], 'wer')]['baseline_mean']:.3f} & --- & --- & --- \\\\")
        for m in models:
            c, w = b[(e, m, "cer")], b[(e, m, "wer")]
            cols = [
                name, MODEL_NAMES[m],
                mark(cer_abs, m, "%.3f" % cer_abs[m]),
                mark(cer_d, m, "%+.3f" % c["mean_delta"]) + " [%+.3f, %+.3f]" % (c["ci_low"], c["ci_high"]),
                fmt_p(sig[(e, m, "cer")]["wilcoxon_p_holm"]),
                mark(wer_abs, m, "%.3f" % wer_abs[m]),
                mark(wer_d, m, "%+.3f" % w["mean_delta"]) + " [%+.3f, %+.3f]" % (w["ci_low"], w["ci_high"]),
                fmt_p(sig[(e, m, "wer")]["wilcoxon_p_holm"]),
                mark(tr, m, "%d\\%%" % round(tr[m] * 100)),
            ]
            body.append(" & ".join(cols) + " \\\\")
        if i == 0:
            body.append("\\midrule")
    (OUT_DIR / "eval_matrix.tex").write_text(
        "\\begin{table*}[t]\n\\centering\n"
        "\\caption{Full Evaluation Matrix: " + str(n_pages) + " Pages, 2 Engines, "
        + str(len(models)) + " Models}\n"
        "\\label{tab:matrix}\n"
        "\\setlength{\\tabcolsep}{4pt}\n\\footnotesize\n"
        "\\begin{tabular}{llccccccc}\n\\toprule\n"
        "Engine & Model & CER & $\\Delta$CER [95\\% CI] & $p_{\\mathrm{Holm}}$ & WER & "
        "$\\Delta$WER [95\\% CI] & $p_{\\mathrm{Holm}}$ & Trunc. \\\\\n\\midrule\n"
        + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n"
        "\\par\\smallskip\n\\begin{minipage}{\\linewidth}\\footnotesize $\\Delta$ = corrected $-$ uncorrected (positive = worse). "
        "CI: paired bootstrap, 10{,}000 resamples, seed 403. "
        "$p_{\\mathrm{Holm}}$: two-sided Wilcoxon signed-rank, Holm-corrected over the "
        + str(n_tests) + " tests. "
        "ns: not significant. Trunc.: pages whose output hit the token cap. "
        "\\colorbox{bestbg}{\\textbf{Shaded}}: best corrected model per engine.\\end{minipage}\n"
        "\\end{table*}\n", encoding="utf-8")

    # ---- diagnostics (pooled over both engines, as in the original Table IV) ----
    stats = defaultdict(lambda: defaultdict(list))
    for r in corrections:
        gt, out = ocr[r["page_id"]]["ground_truth"], r["raw_output"]
        s = stats[r["model_key"]]
        s["len"].append(len(out) / len(gt))
        prof = script_profile(out)
        for k in ("bengali", "latin", "other"):
            s[k].append(prof[k])
        s["ret"].append(token_retention(gt, out))
    mean = {m: {k: sum(v) / len(v) for k, v in stats[m].items()} for m in models}
    ref_rows = []
    for e, name in ENGINES:
        prof = [script_profile(v[e]) for v in ocr.values()]
        ret = [token_retention(v["ground_truth"], v[e]) for v in ocr.values()]
        avg = lambda k: sum(p[k] for p in prof) / len(prof)
        ref_rows.append(f"Raw {name} & --- & {avg('bengali'):.0%} & {avg('latin'):.0%} & "
                        f"{avg('other'):.0%} & {sum(ret) / len(ret):.0%} \\\\".replace("%", "\\%"))
    len_dev = {m: abs(mean[m]["len"] - 1) for m in models}
    model_rows = []
    for m in models:
        v = mean[m]
        cells_ = [
            mark(len_dev, m, f"{v['len']:.2f}$\\times$"),
            mark({k: mean[k]["bengali"] for k in models}, m, f"{v['bengali']:.0%}", False),
            mark({k: mean[k]["latin"] for k in models}, m, f"{v['latin']:.0%}"),
            mark({k: mean[k]["other"] for k in models}, m, f"{v['other']:.0%}"),
            mark({k: mean[k]["ret"] for k in models}, m, f"{v['ret']:.0%}", False),
        ]
        model_rows.append(f"{MODEL_NAMES[m]} & " + " & ".join(cells_).replace("%", "\\%") + " \\\\")
    (OUT_DIR / "diagnostics.tex").write_text(
        "\\begin{table}[t]\n\\centering\n\\caption{Output Diagnostics by Model}\n"
        "\\label{tab:diagnostics}\n\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabular}{lccccc}\n\\toprule\n"
        "Source & Len. & Bn & Latin & Other & GT ret. \\\\\n\\midrule\n"
        + "\n".join(ref_rows) + "\n\\midrule\n" + "\n".join(model_rows) + "\n"
        "\\bottomrule\n\\end{tabular}\n\\par\\smallskip\n\\begin{minipage}{\\linewidth}\\footnotesize Len.: output length / reference length. "
        "Bn / Latin / Other: share of characters by script. GT ret.: share of distinct ground-truth "
        "words present in the output. Pooled over both engines.\\end{minipage}\n\\end{table}\n", encoding="utf-8")

    # ---- extended: damage vs fixes, change, safeguard ----
    rows = []
    for i, (e, name) in enumerate(ENGINES):
        cm = {m: sig[(e, m, "cmer")]["corrected_mean"] for m in models}
        broken = {m: cells[(e, m)]["pct_broken"] for m in models}
        fixed = {m: cells[(e, m)]["pct_fixed"] for m in models}
        ratio = {m: cells[(e, m)]["mean_change_ratio"] for m in models}
        ins = {m: cells[(e, m)]["long_insert_char_share"] for m in models}
        gcer = {m: guard[(e, m)]["cer_guard"] for m in models}
        rows.append(f"{name} & (no correction) & {sig[(e, models[0], 'cmer')]['baseline_mean']:.3f} "
                    f"& --- & --- & --- & --- & {guard[(e, models[0])]['cer_base']:.3f} \\\\")
        for m in models:
            g = guard[(e, m)]
            cols = [
                name, MODEL_NAMES[m],
                mark(cm, m, "%.3f" % cm[m]),
                mark(broken, m, "%.0f\\%%" % (broken[m] * 100)),
                mark(fixed, m, "%.1f\\%%" % (fixed[m] * 100), lower_is_better=False),
                mark(ratio, m, "%.1f" % ratio[m]),
                mark(ins, m, "%.0f\\%%" % (ins[m] * 100)),
                mark(gcer, m, "%.3f" % gcer[m]) + " (%d/%d)" % (g["fallbacks"], n_pages),
            ]
            rows.append(" & ".join(cols) + " \\\\")
        if i == 0:
            rows.append("\\midrule")
    (OUT_DIR / "damage.tex").write_text(
        "\\begin{table*}[t]\n\\centering\n"
        "\\caption{What Correction Changed: Bounded Error, Word-Level Damage and Fixes, "
        "Change Ratio, and a Length Safeguard}\n\\label{tab:damage}\n"
        "\\setlength{\\tabcolsep}{4pt}\n\\footnotesize\n"
        "\\begin{tabular}{llcccccc}\n\\toprule\n"
        "Engine & Model & cMER & \\shortstack{Correct words\\\\broken} & \\shortstack{Wrong words\\\\fixed} & \\shortstack{Change\\\\ratio} & "
        "\\shortstack{Invented\\\\runs} & \\shortstack{CER with safeguard\\\\(fallbacks)} \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\par\\smallskip\n"
        "\\begin{minipage}{\\linewidth}\\footnotesize cMER $=(S{+}D{+}I)/(H{+}S{+}D{+}I)$, bounded in $[0,1]$. "
        "Correct words broken: share of ground-truth words the raw OCR had right that the output gets wrong; "
        "wrong words fixed: share the raw OCR had wrong that the output gets right (word alignment against "
        "the ground truth). Change ratio: CER of the output against the raw OCR, divided by the raw OCR's CER. "
        "Invented runs: share of output characters in runs of $\\geq$6 characters with no counterpart in "
        "the raw OCR. Safeguard: raw OCR kept when output length is outside 0.35--2.8$\\times$ the input. "
        "\\colorbox{bestbg}{\\textbf{Shaded}}: best corrected model per engine.\\end{minipage}\n\\end{table*}\n", encoding="utf-8")

    # ---- decoding ablation (review item C1) ----
    if args.decoding.exists():
        abl = {(r["engine"], r["model"]): r
               for r in json.loads(args.decoding.read_text(encoding="utf-8"))}
        rows = []
        for i, (e, name) in enumerate(ENGINES):
            base = b[(e, models[0], "cer")]["baseline_mean"]
            for m in [m for m in models if (e, m) in abl]:
                r = abl[(e, m)]
                rows.append(" & ".join([
                    name, MODEL_NAMES[m], "%.3f" % base, "%.3f" % r["guarded_mean"],
                    "%.3f" % r["plain_mean"],
                    "%+.3f [%+.3f, %+.3f]" % (r["plain_minus_guarded"], r["ci_low"], r["ci_high"]),
                    "%.3f" % r["plain_vs_input_median_cer"],
                ]) + " \\\\")
            if i == 0:
                rows.append("\\midrule")
        (OUT_DIR / "decoding.tex").write_text(
            "\\begin{table*}[t]\n\\centering\n"
            "\\caption{Decoding Ablation: Repetition Guards versus Plain Greedy Decoding (Mean CER)}\n"
            "\\label{tab:decoding}\n\\setlength{\\tabcolsep}{4pt}\n\\footnotesize\n"
            "\\begin{tabular}{llccccc}\n\\toprule\n"
            "Engine & Model & Raw OCR & Guarded & Plain & Plain $-$ guarded [95\\% CI] & "
            "\\shortstack{Plain output vs.\\\\its input (median CER)} \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\par\\smallskip\n"
            "\\begin{minipage}{\\linewidth}\\footnotesize Guarded: \\texttt{repetition\\_penalty}=1.3 and "
            "\\texttt{no\\_repeat\\_ngram\\_size}=4, as in the original runs; for decoder-only models both "
            "also act on the prompt, so they penalise copying the OCR text. Plain: neither; greedy decoding "
            "with only the length cap. CI: paired bootstrap, 10{,}000 resamples, seed 403. Last column: CER "
            "of the plain output against its own OCR input (near 0 = the model returned its input almost "
            "unchanged).\\end{minipage}\n\\end{table*}\n", encoding="utf-8")

    print(f"Wrote tables to {OUT_DIR}")


if __name__ == "__main__":
    main()
