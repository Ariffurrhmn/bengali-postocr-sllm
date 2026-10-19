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

  paper/tables/decoding.tex     - decoding ablation on the 40 pages: plain,
                                  penalty only, n-gram ban only, both
                                  (third review C4; review3_runs.json); falls
                                  back to the 15-page CPU guarded-vs-plain
                                  table when review3_runs.json is missing
  paper/tables/checks.tex       - precision (float32) and prompt-format checks
                                  (third review C1, C2)
  paper/tables/attribution.tex  - token-level Grad-CAM maps (third review X1)

Main tables use the plain-decoding runs (repetition_penalty 1.0, no n-gram
ban); the guarded runs appear only in decoding.tex.

  paper/tables/dataset.tex      - evaluation pages (review item T1)
  paper/tables/models.tex       - models and configuration (T2)
  paper/tables/editops.tex      - edit operations per page (T3)
  paper/tables/clean.tex        - clean-text control (E1)

Main tables use the 40-page GPU run with plain decoding. The decoding
ablation (decoding.tex) uses the 40-page GPU runs of 2026-10-16; its note
quotes the 15-page CPU comparison for the two Bengali-pretrained checkpoints.

Inputs (defaults, all in results/): ocr_eval40.jsonl, gpu_eval40_ocr.jsonl,
bootstrap_gpu40.json (bootstrap.py), extended_gpu40.json (extended_analysis.py),
revision_gpu40.json (revision_analysis.py), model_profile.json
(model_profile.py), decoding_ablation.json (compare_decoding.py),
review3_runs.json (review3_runs_analysis.py).
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "eval"))

# Imported before anything prints: inspect_examples re-wraps stdout for UTF-8.
from inspect_examples import MODEL_ORDER, script_profile, token_retention  # noqa: E402
from metrics import normalize  # noqa: E402

RESULTS = REPO_ROOT / "results"
OUT_DIR = REPO_ROOT / "paper" / "tables"

ENGINES = [("tesseract", "Tesseract"), ("easyocr", "EasyOCR")]
MODEL_NAMES = {"symspell": "SymSpell (dictionary)", "gemma-2b": "Gemma-2B-it", "llama3.2-1b": "Llama 3.2 1B",
               "banglat5": "BanglaT5", "titullm-1b": "TituLLMs 1B",
               "qwen3-1.7b": "Qwen3 1.7B", "phi4-mini": "Phi-4-mini",
               "titullm-1b-instruct": "TituLLMs 1B Instruct",
               "titullm-3b-instruct": "TituLLMs 3B Instruct",
               # Third review: new models (M1, M7) and the fine-tuning pilot (M2).
               "qwen3-0.6b": "Qwen3 0.6B", "qwen3-4b": "Qwen3 4B", "qwen3-8b": "Qwen3 8B",
               "qwen3-32b": "Qwen3 32B (4-bit)", "gemma3-1b": "Gemma 3 1B", "gemma3-4b": "Gemma 3 4B",
               "llama3.2-3b": "Llama 3.2 3B", "byt5-small-ft": "ByT5-small, fine-tuned"}
# Order of the scale table: families, smallest first.
SCALE_ORDER = ["qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "qwen3-8b", "qwen3-32b", "gemma-2b", "gemma3-1b",
               "gemma3-4b", "llama3.2-1b", "llama3.2-3b", "phi4-mini"]
NOISE_ORDER = ["gemma-2b", "qwen3-1.7b", "phi4-mini", "llama3.2-1b"]
# Computed from each checkpoint's configuration, tied embeddings counted once
# (BanglaT5's input and output embeddings are not tied in the checkpoint).
PARAMS_B = {"gemma-2b": 2.51, "qwen3-1.7b": 1.72, "phi4-mini": 3.84, "llama3.2-1b": 1.24,
            "banglat5": 0.25, "titullm-1b": 1.24, "titullm-1b-instruct": 1.32,
            "titullm-3b-instruct": 3.34}
# Hugging Face revision of each checkpoint used in the runs (each was last
# modified before the runs, 10-12 October 2026, so the default branch then
# was this commit).
REVISIONS = {"gemma-2b": "9698841", "qwen3-1.7b": "70d244c", "phi4-mini": "cfbefac",
             "llama3.2-1b": "9213176", "banglat5": "76b135c", "titullm-1b": "521da71",
             "titullm-1b-instruct": "039dbd2", "titullm-3b-instruct": "6a96004"}
# Precision on the T4 (correction/models.py: GPU_FLOAT16_MODELS).
GPU_DTYPE = {m: "fp16" if m in ("phi4-mini", "titullm-3b-instruct") else "fp32"
             for m in PARAMS_B}


def mark(values: dict, key, text: str, lower_is_better=True) -> str:
    """Wrap text in \\best{} if values[key] is the best of the dict."""
    target = min(values.values()) if lower_is_better else max(values.values())
    return f"\\best{{{text}}}" if abs(values[key] - target) < 1e-12 else text


def fmt_p(p: float) -> str:
    if p < 0.001:
        return "$<$0.001"
    return f"{p:.3f}" if p < 0.05 else f"{p:.2f}\\textsuperscript{{ns}}"


def sigmark(r: dict) -> str:
    """$^{+}$ / $^{-}$ when a test is significant after Holm correction."""
    if r["wilcoxon_p_holm"] >= 0.05:
        return ""
    return "$^{-}$" if r["mean_delta"] < 0 else "$^{+}$"


def fmt_tost(p: float) -> str:
    return "$<$0.001" if p < 0.001 else "%.3f" % p


def ci(lo: float, hi: float) -> str:
    return "[%+.3f, %+.3f]" % (lo, hi)


def load_jsonl(path: Path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ocr", type=Path, default=RESULTS / "ocr_eval40.jsonl")
    parser.add_argument("--boot", type=Path, default=RESULTS / "bootstrap_gpu40.json")
    parser.add_argument("--ext", type=Path, default=RESULTS / "extended_gpu40.json")
    parser.add_argument("--corrections", type=Path,
                        default=RESULTS / "eval40_ocr.jsonl")
    parser.add_argument("--revision", type=Path, default=RESULTS / "revision_gpu40.json")
    parser.add_argument("--profile", type=Path, default=RESULTS / "model_profile.json")
    parser.add_argument("--decoding", type=Path, default=RESULTS / "decoding_ablation.json")
    parser.add_argument("--variants", type=Path, default=RESULTS / "variants_gpu40.json")
    parser.add_argument("--gated", type=Path, default=RESULTS / "gated_gpu40.json")
    parser.add_argument("--sensitivity", type=Path, default=RESULTS / "sensitivity_gpu40.json")
    parser.add_argument("--review3", type=Path, default=RESULTS / "review3_gpu40.json")
    parser.add_argument("--region-ocr", type=Path, default=RESULTS / "region_ocr40.json")
    parser.add_argument("--region-runs", type=Path, default=RESULTS / "region_runs40.json")
    parser.add_argument("--m1-runs", type=Path, default=RESULTS / "m1_runs40.json")
    parser.add_argument("--main-matrix", type=Path, default=RESULTS / "main_general40.json")
    parser.add_argument("--noise-runs", type=Path, default=RESULTS / "noise_runs40.json")
    parser.add_argument("--m2-runs", type=Path, default=RESULTS / "m2_runs40.json")
    parser.add_argument("--m2-log", type=Path, default=RESULTS / "review3e" / "m2_train_log.json")
    parser.add_argument("--m4-runs", type=Path, default=RESULTS / "m4_dict40.json")
    parser.add_argument("--fusion-runs", type=Path, default=RESULTS / "m6_fusion40.json")
    parser.add_argument("--edit-analysis", type=Path, default=RESULTS / "edit_analysis40.json")
    parser.add_argument("--review4-runs", type=Path, default=RESULTS / "review4_runs.json")
    parser.add_argument("--output-filter", type=Path, default=RESULTS / "output_filter40.json")
    parser.add_argument("--fusion-family4", type=Path, default=RESULTS / "m6_fusion40_family4.json",
                        help="first scoring of fusion against both engines (4-cell Holm family, D51/D60)")
    parser.add_argument("--m7-runs", type=Path, default=RESULTS / "m7_runs10.json")
    parser.add_argument("--review3-runs", type=Path, default=RESULTS / "review3_runs.json",
                        help="third-review GPU runs (review3_runs_analysis.py): C1, C2, C4, X1")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    boot = json.loads(args.boot.read_text(encoding="utf-8"))
    ext = json.loads(args.ext.read_text(encoding="utf-8"))
    ocr = {r["page_id"]: r for r in load_jsonl(args.ocr)}
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
    # Third-review analyses (review3_analysis.py): wMER, equivalence, page counts.
    r3 = json.loads(args.review3.read_text(encoding="utf-8"))
    wm = {(r["engine"], r["model"]): r for r in r3["wmer_tests"]}
    eqv = {(r["engine"], r["model"]): r for r in r3["equivalence"]}

    # ---- baseline ----
    rows = []
    for e, name in ENGINES:
        base = b[(e, models[0], "cer")]["baseline_mean"]
        wbase = b[(e, models[0], "wer")]["baseline_mean"]
        cbase = sig[(e, models[0], "cmer")]["baseline_mean"]
        rows.append(f"{name} & {base:.3f} & {wbase:.3f} & {cbase:.3f} & "
                    f"{r3['wmer_baseline'][e]:.3f} \\\\")
    (OUT_DIR / "baseline.tex").write_text(
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Uncorrected OCR Baselines (" + str(n_pages) + " Evaluation Pages)}\n"
        "\\label{tab:baseline}\n"
        "\\begin{tabular}{lcccc}\n\\toprule\n"
        "Engine & CER & WER & cMER & wMER \\\\\n\\midrule\n" + "\n".join(rows) + "\n"
        "\\bottomrule\n\\end{tabular}\n\\par\\smallskip\n\\begin{minipage}{\\linewidth}\\footnotesize "
        "cMER, wMER: character and word match error rates, $(S{+}D{+}I)/(H{+}S{+}D{+}I)$, "
        "bounded in $[0,1]$.\\end{minipage}\n\\end{table}\n", encoding="utf-8")

    # ---- full evaluation matrix ----
    body = []
    for i, (e, name) in enumerate(ENGINES):
        cer_abs = {m: b[(e, m, "cer")]["corrected_mean"] for m in models}
        wer_abs = {m: b[(e, m, "wer")]["corrected_mean"] for m in models}
        cer_d = {m: b[(e, m, "cer")]["mean_delta"] for m in models}
        wer_d = {m: b[(e, m, "wer")]["mean_delta"] for m in models}
        tr = {m: trunc[(e, m)] / n_pages for m in models}
        first = b[(e, models[0], "cer")]
        wm_abs = {m: wm[(e, m)]["corrected"] for m in models}
        better = {m: eqv[(e, m)]["better"] for m in models}
        body.append(f"{name} & (no correction) & {first['baseline_mean']:.3f} & --- & --- & "
                    f"{b[(e, models[0], 'wer')]['baseline_mean']:.3f} & --- & --- & "
                    f"{r3['wmer_baseline'][e]:.3f} & --- & --- \\\\")
        for m in models:
            c, w = b[(e, m, "cer")], b[(e, m, "wer")]
            q = eqv[(e, m)]
            cols = [
                name, MODEL_NAMES[m],
                mark(cer_abs, m, "%.3f" % cer_abs[m]),
                mark(cer_d, m, "%+.3f" % c["mean_delta"]) + " [%+.3f, %+.3f]" % (c["ci_low"], c["ci_high"]),
                fmt_p(sig[(e, m, "cer")]["wilcoxon_p_holm"]),
                mark(wer_abs, m, "%.3f" % wer_abs[m]),
                mark(wer_d, m, "%+.3f" % w["mean_delta"]) + " [%+.3f, %+.3f]" % (w["ci_low"], w["ci_high"]),
                fmt_p(sig[(e, m, "wer")]["wilcoxon_p_holm"]),
                mark(wm_abs, m, "%.3f" % wm_abs[m]) + sigmark(wm[(e, m)]),
                mark(better, m, "%d/%d" % (q["better"], q["worse"]), lower_is_better=False),
                mark(tr, m, "%d\\%%" % round(tr[m] * 100)),
            ]
            body.append(" & ".join(cols) + " \\\\")
        if i == 0:
            body.append("\\midrule")
    (OUT_DIR / "eval_matrix.tex").write_text(
        "\\begin{table*}[t]\n\\centering\n"
        "\\caption{Full Evaluation Matrix: " + str(n_pages) + " Pages, 2 Engines, "
        + str(len(models)) + " Correction Methods}\n"
        "\\label{tab:matrix}\n"
        "\\setlength{\\tabcolsep}{3pt}\n\\footnotesize\n"
        "\\begin{tabular}{llccccccccc}\n\\toprule\n"
        "Engine & Model & CER & $\\Delta$CER [95\\% CI] & $p_{\\mathrm{Holm}}$ & WER & "
        "$\\Delta$WER [95\\% CI] & $p_{\\mathrm{Holm}}$ & wMER & \\shortstack{Pages\\\\better/worse} & "
        "Trunc. \\\\\n\\midrule\n"
        + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n"
        "\\par\\smallskip\n\\begin{minipage}{\\linewidth}\\footnotesize $\\Delta$ = corrected $-$ uncorrected (positive = worse). "
        "CI: paired bootstrap, 10{,}000 resamples, seed 403. "
        "$p_{\\mathrm{Holm}}$: two-sided Wilcoxon signed-rank test, Holm-corrected over the "
        + str(n_tests) + " CER and WER tests, the primary family of the study; "
        "pages whose error rate did not change are dropped by the test (zero differences). "
        "ns: not significant. wMER: word match error rate; $^{+}$/$^{-}$: significantly higher/lower "
        "than the raw OCR (exploratory family of " + str(len(wm)) + " tests). "
        "Pages better/worse: pages whose CER fell/rose, of " + str(n_pages) + ". "
        "Trunc.: pages whose output hit the token cap. "
        "\\colorbox{bestbg}{\\textbf{Shaded}}: best value in the column among the corrected methods "
        "of that engine (lowest error, most pages better, least truncation); it is not compared "
        "with the raw OCR.\\end{minipage}\n"
        "\\end{table*}\n", encoding="utf-8")

    # ---- equivalence tests (review item E2) ----
    gp = [m for m in ("gemma-2b", "qwen3-1.7b", "phi4-mini", "llama3.2-1b") if m in models]
    rows = []
    for i, (e, name) in enumerate(ENGINES):
        for m in gp:
            q = eqv[(e, m)]
            if m == gp[0]:
                rows.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{name}}}}} \\\\")
            rows.append(" & ".join([
                MODEL_NAMES[m], "%+.3f" % q["mean_delta"],
                "[%+.3f, %+.3f]" % tuple(q["ci90"]),
                fmt_tost(q["tost_p_0.01"]), fmt_tost(q["tost_p_0.02"]),
                "%d/%d/%d" % (q["better"], q["worse"], q["same"])]) + " \\\\")
        if i == 0:
            rows.append("\\midrule")
    (OUT_DIR / "equivalence.tex").write_text(
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Equivalence Tests for the Models Closest to the Raw OCR}\n\\label{tab:equivalence}\n"
        "\\setlength{\\tabcolsep}{3pt}\n\\footnotesize\n"
        "\\begin{tabular}{lccccc}\n\\toprule\n"
        " & & & \\multicolumn{2}{c}{TOST $p$} & \\\\\n\\cmidrule(lr){4-5}\n"
        "Model & $\\Delta$CER & 90\\% CI & $\\pm$0.01 & $\\pm$0.02 & \\shortstack{Better/\\\\worse/same} \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\par\\smallskip\n"
        "\\begin{minipage}{\\linewidth}\\footnotesize $\\Delta$CER: corrected $-$ raw OCR. 90\\% CI: "
        "paired bootstrap. TOST: two one-sided paired $t$-tests against an equivalence margin of "
        "$\\pm$0.01 or $\\pm$0.02 CER; $p<0.05$ means the change is within the margin. Not adjusted for "
        "multiplicity (exploratory). Last column: pages whose CER fell, rose or stayed the same, "
        "of " + str(n_pages) + ".\\end{minipage}\n\\end{table}\n", encoding="utf-8")

    # ---- diagnostics (both engines together) ----
    # Length ratio and retention are means over outputs (as Table II's raw
    # length ratio); script shares are pooled over all non-space characters,
    # so empty outputs do not count as 0% of every script (second review).
    def diag(pairs):
        """pairs: list of (ground truth, output text)."""
        chars = {"bengali": 0, "latin": 0, "other": 0}
        for _, out in pairs:
            prof = script_profile(out)
            for k in chars:
                chars[k] += prof[k] * prof["n"]
        total = sum(chars.values()) or 1
        return {"len": sum(len(normalize(o)) / len(normalize(g)) for g, o in pairs) / len(pairs),
                **{k: v / total for k, v in chars.items()},
                "ret": sum(token_retention(g, o) for g, o in pairs) / len(pairs),
                "empty": sum(not o.strip() for _, o in pairs), "n": len(pairs)}

    by_model = defaultdict(list)
    for r in corrections:
        by_model[r["model_key"]].append((ocr[r["page_id"]]["ground_truth"], r["raw_output"]))
    mean = {m: diag(by_model[m]) for m in models}
    pct = lambda x: f"{x:.0%}".replace("%", "\\%")
    ref_rows = []
    for e, name in ENGINES:
        v = diag([(p_["ground_truth"], p_[e]) for p_ in ocr.values()])
        ref_rows.append(f"Raw {name} & {v['len']:.2f}$\\times$ & {pct(v['bengali'])} & "
                        f"{pct(v['latin'])} & {pct(v['other'])} & {pct(v['ret'])} & "
                        f"{v['empty']}/{v['n']} \\\\")
    len_dev = {m: abs(mean[m]["len"] - 1) for m in models}
    model_rows = []
    for m in models:
        v = mean[m]
        cells_ = [
            mark(len_dev, m, f"{v['len']:.2f}$\\times$"),
            mark({k: mean[k]["bengali"] for k in models}, m, pct(v["bengali"]), False),
            mark({k: mean[k]["latin"] for k in models}, m, pct(v["latin"])),
            mark({k: mean[k]["other"] for k in models}, m, pct(v["other"])),
            mark({k: mean[k]["ret"] for k in models}, m, pct(v["ret"]), False),
            f"{v['empty']}/{v['n']}",
        ]
        model_rows.append(f"{MODEL_NAMES[m]} & " + " & ".join(cells_) + " \\\\")
    (OUT_DIR / "diagnostics.tex").write_text(
        "\\begin{table}[t]\n\\centering\n\\caption{Output Diagnostics by Model}\n"
        "\\label{tab:diagnostics}\n\\setlength{\\tabcolsep}{3pt}\n\\footnotesize\n"
        "\\begin{tabular}{lcccccc}\n\\toprule\n"
        "Source & Len. & Bn & Latin & Other & GT ret. & Empty \\\\\n\\midrule\n"
        + "\n".join(ref_rows) + "\n\\midrule\n" + "\n".join(model_rows) + "\n"
        "\\bottomrule\n\\end{tabular}\n\\par\\smallskip\n\\begin{minipage}{\\linewidth}\\footnotesize "
        "Both engines together (80 outputs per model; 40 per raw OCR row). Len.: output length / "
        "reference length, mean over outputs (the raw OCR values are those of Table~\\ref{tab:dataset}). "
        "Bn / Latin / Other: share of all non-space output characters, pooled over outputs, so "
        "empty outputs do not enter. GT ret.: share of distinct ground-truth words present in the "
        "output, mean over outputs (an empty output retains none). Empty: outputs with no "
        "text.\\end{minipage}\n\\end{table}\n", encoding="utf-8")

    # ---- extended: damage vs fixes, change, safeguard ----
    rows = []
    for i, (e, name) in enumerate(ENGINES):
        cm = {m: sig[(e, m, "cmer")]["corrected_mean"] for m in models}
        broken = {m: cells[(e, m)]["pct_broken"] for m in models}
        fixed = {m: cells[(e, m)]["pct_fixed"] for m in models}
        ratio = {m: cells[(e, m)]["mean_change_ratio"] for m in models}
        ins = {m: cells[(e, m)]["long_insert_char_share"] for m in models}
        rows.append(f"{name} & (no correction) & {sig[(e, models[0], 'cmer')]['baseline_mean']:.3f} "
                    f"& --- & --- & --- & --- \\\\")
        for m in models:
            cols = [
                name, MODEL_NAMES[m],
                mark(cm, m, "%.3f" % cm[m]) + sigmark(sig[(e, m, "cmer")]),
                mark(broken, m, "%.0f\\%%" % (broken[m] * 100)),
                mark(fixed, m, "%.1f\\%%" % (fixed[m] * 100), lower_is_better=False),
                mark(ratio, m, "%.1f" % ratio[m]),
                mark(ins, m, "%.0f\\%%" % (ins[m] * 100)),
            ]
            rows.append(" & ".join(cols) + " \\\\")
        if i == 0:
            rows.append("\\midrule")
    (OUT_DIR / "damage.tex").write_text(
        "\\begin{table*}[t]\n\\centering\n"
        "\\caption{What Correction Changed: Bounded Error, Word-Level Damage and Fixes, "
        "and Change Ratio}\n\\label{tab:damage}\n"
        "\\setlength{\\tabcolsep}{4pt}\n\\footnotesize\n"
        "\\begin{tabular}{llccccc}\n\\toprule\n"
        "Engine & Model & cMER & \\shortstack{Correct words\\\\broken} & \\shortstack{Wrong words\\\\fixed} & \\shortstack{Change\\\\ratio} & "
        "\\shortstack{Invented\\\\runs} \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\par\\smallskip\n"
        "\\begin{minipage}{\\linewidth}\\footnotesize cMER $=(S{+}D{+}I)/(H{+}S{+}D{+}I)$, bounded in $[0,1]$. "
        "Correct words broken: share of ground-truth words the raw OCR had right that the output gets wrong; "
        "wrong words fixed: share the raw OCR had wrong that the output gets right (word alignment against "
        "the ground truth). Change ratio: CER of the output against the raw OCR, divided by the raw OCR's CER. "
        "Invented runs: share of output characters in runs of $\\geq$6 characters with no counterpart in "
        "the raw OCR. $^{+}$/$^{-}$: cMER significantly higher/lower than the raw OCR, Wilcoxon "
        "signed-rank test, Holm-corrected over the 18 cMER tests (exploratory family). "
        "\\colorbox{bestbg}{\\textbf{Shaded}}: best corrected model per engine.\\end{minipage}\n\\end{table*}\n", encoding="utf-8")

    # ---- dataset statistics (review item T1) ----
    if args.revision.exists():
        rev = json.loads(args.revision.read_text(encoding="utf-8"))
        d = rev["dataset"]
        rows = [
            f"Pages & {d['pages']} \\\\",
            f"Books & {d['books']} (at most {d['pages_per_book_max']} pages per book) \\\\",
            f"Reference characters per page & median {d['gt_chars_median']:.0f} "
            f"(range {d['gt_chars_min']}--{d['gt_chars_max']}) \\\\",
            f"Reference words per page & median {d['gt_words_median']:.0f} \\\\",
        ] + [f"Raw {name} length / reference & mean {d[f'{e}_len_ratio_mean']:.2f}, "
             f"median {d[f'{e}_len_ratio_median']:.2f} \\\\" for e, name in ENGINES]
        rows.insert(2, "Printed & 1713--1914 (no per-book dates) \\\\")
        if args.sensitivity.exists():
            n = json.loads(args.sensitivity.read_text(encoding="utf-8"))["page_type_counts"]
            rows.insert(3, "Page types & " + ", ".join(f"{n[t]} {t}" for t in
                                                       ("title", "verse", "prose", "list"))
                        + " \\\\")
        (OUT_DIR / "dataset.tex").write_text(
            "\\begin{table}[t]\n\\centering\n\\caption{Evaluation Pages}\n\\label{tab:dataset}\n"
            "\\setlength{\\tabcolsep}{4pt}\n\\footnotesize\n\\begin{tabular}{ll}\n\\toprule\n" + "\n".join(rows) + "\n\\bottomrule\n"
            "\\end{tabular}\n\\par\\smallskip\n\\begin{minipage}{\\linewidth}\\footnotesize "
            "Book: page identifier without its page number. Lengths after Unicode NFC "
            "normalisation and whitespace collapsing.\\end{minipage}\n\\end{table}\n",
            encoding="utf-8")

        # ---- clean-text control (review item E1) ----
        clean = {r["model"]: r for r in rev["micro"] if r["engine"] == "clean"}
        if clean:
            rows = []
            for m in [m for m in models if m in clean]:
                r = clean[m]
                cw = r3["clean_word_rates"][m]
                rows.append(f"{MODEL_NAMES[m]} & {r['mean_cer']:.3f} & {r['pooled_cer']:.3f} & "
                            f"{cw['wer']:.3f} & {cw['wmer']:.3f} & {r['pages_exact']}/{r['n']} \\\\")
            (OUT_DIR / "clean.tex").write_text(
                "\\begin{table}[t]\n\\centering\n"
                "\\caption{Clean-Text Control: Ground Truth Given as Input}\n\\label{tab:clean}\n"
                "\\setlength{\\tabcolsep}{3pt}\n\\footnotesize\n\\begin{tabular}{lccccc}\n\\toprule\n"
                "Model & \\shortstack{Mean\\\\CER} & \\shortstack{Pooled\\\\CER} & WER & wMER & "
                "\\shortstack{Pages\\\\unchanged} \\\\\n\\midrule\n"
                + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\par\\smallskip\n"
                "\\begin{minipage}{\\linewidth}\\footnotesize Each model receives the reference "
                "text itself, with the same prompt and plain decoding. Error rates are measured "
                "against that reference, so any value above 0 is damage to correct text; WER and "
                "wMER are per-page means. Pages unchanged: CER exactly 0 after "
                "normalisation.\\end{minipage}\n\\end{table}\n",
                encoding="utf-8")

    # ---- edit operations (review item T3) ----
    rows = []
    for i, (e, name) in enumerate(ENGINES):
        raw = cells[(e, models[0])]["edit_ops_vs_gt_raw_ocr"]
        rows.append(f"{name} & (no correction) & {raw['S']:.0f} & {raw['D']:.0f} & {raw['I']:.0f} \\\\")
        for m in models:
            c = cells[(e, m)]["edit_ops_vs_gt_corrected"]
            rows.append(f"{name} & {MODEL_NAMES[m]} & {c['S']:.0f} & {c['D']:.0f} & {c['I']:.0f} \\\\")
        if i == 0:
            rows.append("\\midrule")
    (OUT_DIR / "editops.tex").write_text(
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Mean Edit Operations per Page Against the Reference}\n\\label{tab:editops}\n"
        "\\setlength{\\tabcolsep}{4pt}\n\\footnotesize\n"
        "\\begin{tabular}{llccc}\n\\toprule\n"
        "Engine & Model & Subst. & Del. & Ins. \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n", encoding="utf-8")

    # ---- model and configuration details (review item T2) ----
    if args.profile.exists():
        prof = {r["model"]: r for r in json.loads(args.profile.read_text(encoding="utf-8"))}
        conf_rows, cost_rows = [], []
        for m in [m for m in models if m in prof]:
            r = prof[m]
            cpu = "%.1f" % r["cpu_min_per_page"] if r["cpu_min_per_page"] is not None else "---"
            gpu = "%.2f" % r["gpu_min_per_page"]
            # Break the checkpoint ID after "/" and "-" so it wraps inside its column
            # (the first version of this table ran off the page; second review).
            ckpt = (r["checkpoint"].replace("_", "\\_").replace("/", "/\\allowbreak ")
                    .replace("-", "-\\allowbreak ") + "@\\allowbreak " + REVISIONS[m])
            weights_gb = PARAMS_B[m] * (2 if GPU_DTYPE[m] == "fp16" else 4)
            fmt = "text" if m == "banglat5" else "template"
            # Group headings: general-purpose models vs Bengali-pretrained diagnostic checkpoints.
            if m == "gemma-2b":
                conf_rows.append("\\multicolumn{6}{l}{\\textit{General-purpose, instruction-tuned}} \\\\")
            if m == "banglat5":
                conf_rows.append("\\multicolumn{6}{l}{\\textit{Bengali-pretrained (diagnostic)}} \\\\")
            conf_rows.append(f"{MODEL_NAMES[m]} & \\texttt{{{ckpt}}} & {PARAMS_B[m]:.2f} & "
                             f"{GPU_DTYPE[m]} & {fmt} & {r['mean_chars_per_chunk']:.0f} \\\\")
            cost_rows.append(f"{MODEL_NAMES[m]} & {r['tokens_per_char']:.2f} & "
                             f"{r['tokens_per_word']:.1f} & {weights_gb:.1f} & {cpu} & {gpu} \\\\")
        (OUT_DIR / "models.tex").write_text(
            "\\begin{table}[t]\n\\centering\n\\caption{Models and Configuration}\n"
            "\\label{tab:models}\n\\setlength{\\tabcolsep}{2.5pt}\n\\footnotesize\n"
            "\\begin{tabular}{>{\\raggedright\\arraybackslash}p{1.85cm}"
            ">{\\raggedright\\arraybackslash\\scriptsize}p{2.75cm}cccc}\n\\toprule\n"
            "Model & Checkpoint@revision & \\shortstack{Params\\\\(B)} & \\shortstack{GPU\\\\prec.} & "
            "Input & \\shortstack{Chars/\\\\chunk} \\\\\n\\midrule\n"
            + "\n".join(conf_rows) + "\n"
            "\\bottomrule\n\\end{tabular}\n\\par\\smallskip\n\\begin{minipage}{\\linewidth}"
            "\\footnotesize Checkpoint: Hugging Face model and revision. Parameters from each "
            "checkpoint's configuration, tied embeddings counted once. GPU prec.: precision on the "
            "T4 (fp16 for the two models whose fp32 weights exceed its memory). Input: the prompt "
            "through the chat template shipped with the checkpoint, or the OCR text alone. Chars/chunk: mean "
            "characters per chunk of about 250 tokens on raw Tesseract text. Decoding for all "
            "models: greedy, at most $\\min(512,\\max(64,1.5n))$ new tokens, no repetition penalty "
            "or n-gram ban.\\end{minipage}\n\\end{table}\n", encoding="utf-8")
        (OUT_DIR / "cost.tex").write_text(
            "\\begin{table}[t]\n\\centering\n\\caption{Tokenisation and Computational Cost}\n"
            "\\label{tab:cost}\n\\setlength{\\tabcolsep}{2pt}\n\\footnotesize\n"
            "\\begin{tabular}{lccccc}\n\\toprule\n"
            "Model & \\shortstack{Tokens/\\\\char} & \\shortstack{Tokens/\\\\word} & "
            "\\shortstack{GPU weights\\\\(GB)} & \\shortstack{CPU\\\\min/page} & "
            "\\shortstack{GPU\\\\min/page} \\\\\n\\midrule\n"
            + "\n".join(cost_rows) + "\n"
            "\\bottomrule\n\\end{tabular}\n\\par\\smallskip\n\\begin{minipage}{\\linewidth}"
            "\\footnotesize Tokens/char and tokens/word: each tokenizer on the 40 reference texts. "
            "GPU weights: parameters $\\times$ 4 bytes in fp32 or 2 in fp16; peak memory during "
            "generation was not recorded. GPU: NVIDIA T4, plain decoding, the 40 pages of the main "
            "setting. CPU: free Colab CPU runtime in bfloat16, plain decoding, on the 15-page "
            "sample with its August 2026 OCR, a different input from the main setting "
            "(---: not run on CPU). Times are per page and engine.\\end{minipage}\n"
            "\\end{table}\n", encoding="utf-8")

    # ---- confidence-gated correction (review item E4) ----
    if args.gated.exists():
        gated = {(r["engine"], r["model"], r["threshold"]): r
                 for r in json.loads(args.gated.read_text(encoding="utf-8"))}
        thr = {e: sorted({t for (ee, _, t) in gated if ee == e}) for e, _ in ENGINES}
        rows = [" & ".join(["Raw OCR"] + [f"{b[(e, models[0], 'cer')]['baseline_mean']:.3f}"
                                          for e, _ in ENGINES for _t in thr[e]]) + " \\\\",
                "\\midrule"]
        for m in [m for m in models if (ENGINES[0][0], m, thr[ENGINES[0][0]][0]) in gated]:
            cols = [MODEL_NAMES[m]]
            for e, _ in ENGINES:
                for t in thr[e]:
                    r = gated[(e, m, t)]
                    star = ""
                    if r["wilcoxon_p_holm"] < 0.05:
                        star = "$^{-}$" if r["mean_delta"] < 0 else "$^{+}$"
                    cols.append("%.3f%s" % (r["mean_cer"], star))
            rows.append(" & ".join(cols) + " \\\\")
        head = " & ".join(["Method"] + [str(t) for e, _ in ENGINES for t in thr[e]])
        (OUT_DIR / "gated.tex").write_text(
            "\\begin{table}[t]\n\\centering\n"
            "\\caption{Confidence-Gated Correction (Mean CER, 40 Pages)}\n\\label{tab:gated}\n"
            "\\setlength{\\tabcolsep}{2pt}\n\\footnotesize\n"
            "\\begin{tabular}{lcccccc}\n\\toprule\n"
            " & \\multicolumn{3}{c}{Tesseract (word conf.)} & \\multicolumn{3}{c}{EasyOCR (box conf.)} \\\\\n"
            "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n" + head + " \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\par\\smallskip\n"
            "\\begin{minipage}{\\linewidth}\\footnotesize A model's edit is kept only where every "
            "OCR word it changes has a confidence below the threshold (column header); elsewhere "
            "the raw OCR is kept. $^{+}$/$^{-}$: significantly higher/lower than the raw OCR, "
            "Wilcoxon signed-rank test, Holm-corrected over the 54 gated tests (exploratory "
            "family); pages left identical to the raw OCR are dropped by the test.\\end{minipage}\n"
            "\\end{table}\n", encoding="utf-8")

    # ---- prompt and chunking variants (review items M3, M4) ----
    if args.variants.exists():
        var = json.loads(args.variants.read_text(encoding="utf-8"))
        cell = {(r["engine"], r["model"], r["variant"]): r for r in var}
        labels = list(dict.fromkeys(r["variant"] for r in var))
        vmodels = list(dict.fromkeys(r["model"] for r in var))
        rows = []
        for m in vmodels:
            for i, v in enumerate(labels):
                cols = [MODEL_NAMES[m] if i == 0 else "", v]
                for e, _ in ENGINES:
                    r = cell[(e, m, v)]
                    star = ("$^{*}$" if r.get("wilcoxon_p_holm", 1) < 0.05 else "")
                    cols.append("%.3f%s" % (r["mean_cer"], star))
                rows.append(" & ".join(cols) + " \\\\")
            if m != vmodels[-1]:
                rows.append("\\midrule")
        # Entries of one variant that coincide after rounding come from different outputs;
        # print their unrounded values so they are not mistaken for copy errors.
        ties = []
        for e, ename in ENGINES:
            vals = {(m, v): cell[(e, m, v)]["mean_cer"] for m in vmodels for v in labels}
            for x in sorted({"%.3f" % y for y in vals.values()}):
                same = [k for k in vals if "%.3f" % vals[k] == x]
                if len(same) > 1:
                    ties.append("%s: %s" % (ename, ", ".join(
                        "%s %s %.4f" % (MODEL_NAMES[m], v, vals[(m, v)]) for m, v in same)))
        tie_note = (" Equal entries in a column come from different outputs (unrounded: "
                    + "; ".join(ties) + ").") if ties else ""
        base = {e: b[(e, models[0], "cer")]["baseline_mean"] for e, _ in ENGINES}
        (OUT_DIR / "variants.tex").write_text(
            "\\begin{table}[t]\n\\centering\n"
            "\\caption{Prompt and Chunking Variants (Mean CER, 40 Pages)}\n\\label{tab:variants}\n"
            "\\setlength{\\tabcolsep}{4pt}\n\\footnotesize\n"
            "\\begin{tabular}{llcc}\n\\toprule\n"
            "Model & Variant & Tesseract & EasyOCR \\\\\n\\midrule\n"
            f" & Raw OCR & {base['tesseract']:.3f} & {base['easyocr']:.3f} \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\par\\smallskip\n"
            "\\begin{minipage}{\\linewidth}\\footnotesize Original: English prompt, chunks of about "
            "250 tokens. Each variant changes one thing: the preserve-spelling instruction, the "
            "prompt in Bengali, three worked examples, or chunks of at most 300 characters. "
            "$^{*}$: differs from the original setting, two-sided Wilcoxon signed-rank test, "
            "Holm-corrected over the 32 variant comparisons (exploratory family)." + tie_note
            + "\\end{minipage}\n"
            "\\end{table}\n",
            encoding="utf-8")

    runs3 = json.loads(args.review3_runs.read_text(encoding="utf-8")) if args.review3_runs.exists() else None
    INPUTS = ENGINES + [("ground_truth", "Clean text")]

    def star(c):
        """$^{*}$ when an unadjusted Wilcoxon test is significant (exploratory checks)."""
        return "$^{*}$" if c["p"] < 0.05 else ""

    # ---- decoding 2x2 on the 40 pages (third review, item C4) ----
    if runs3 and args.decoding.exists():
        abl = {(r["engine"], r["model"]): r
               for r in json.loads(args.decoding.read_text(encoding="utf-8"))}
        c4 = {(c["model"], c["engine"], c["preset"]): c for c in runs3["C4"]}
        presets = ["penalty_only", "ngram_only", "guarded"]
        dec_models = [m for m in MODEL_ORDER if any(k[0] == m for k in c4)]
        rows = []
        for i, (e, name) in enumerate(INPUTS):
            ms = [m for m in dec_models if all((m, e, p) in c4 for p in presets)]
            if i:
                rows.append("\\midrule")
            for m in ms:
                cs = [c4[(m, e, p)] for p in presets]
                raw = "--" if e == "ground_truth" else "%.3f" % cs[0]["raw_cer"]
                cells = ["%.3f%s" % (c["mean_a"], "$^{+}$" if c["p_holm"] < 0.05 and c["delta"] > 0
                                     else "$^{-}$" if c["p_holm"] < 0.05 else "") for c in cs]
                rows.append(" & ".join([name, MODEL_NAMES[m], raw, "%.3f" % cs[0]["mean_b"]] + cells
                                       + ["%+.3f %s" % (cs[2]["delta"], ci(*cs[2]["ci"])),
                                          "/".join(str(c["truncated"]) for c in cs)]) + " \\\\")
        n_tests = len(c4)
        old = "; ".join("%s %s" % (MODEL_NAMES[m], " and ".join(
            "%+.2f" % (-abl[(e, m)]["plain_minus_guarded"]) for e, _ in ENGINES))
                        for m in ("titullm-1b", "banglat5") if all((e, m) in abl for e, _ in ENGINES))
        (OUT_DIR / "decoding.tex").write_text(
            "\\begin{table*}[t]\n\\centering\n"
            "\\caption{Decoding Ablation on the 40 Pages: Repetition Penalty and $n$-Gram Ban, "
            "Alone and Together (Mean CER)}\n"
            "\\label{tab:decoding}\n\\setlength{\\tabcolsep}{4pt}\n\\footnotesize\n"
            "\\begin{tabular}{llcccccll}\n\\toprule\n"
            "Input & Model & Raw OCR & Plain & Penalty only & $n$-gram ban only & Both (guarded) & "
            "Guarded $-$ plain [95\\% CI] & Truncated \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\par\\smallskip\n"
            "\\begin{minipage}{\\linewidth}\\footnotesize Plain: greedy decoding with only the length "
            "cap (the main runs). Penalty: \\texttt{repetition\\_penalty}=1.3; $n$-gram ban: "
            "\\texttt{no\\_repeat\\_ngram\\_size}=4; in a decoder-only model both also act on the "
            "prompt, so they penalise copying the OCR text. Clean text: the reference given as input, "
            "so any change is damage. $^{+}$/$^{-}$: higher/lower than plain decoding, Wilcoxon "
            "signed-rank test, Holm-corrected over the " + str(n_tests) + " preset-versus-plain tests "
            "(exploratory family). CI: paired bootstrap, 10{,}000 resamples, seed 403. Truncated: "
            "pages that reached the token limit (penalty/$n$-gram/both), of 40. Phi-4-mini was not run "
            "on EasyOCR text (time limit of the GPU session). An earlier ablation of guarded against "
            "plain decoding on " + str(len(load_jsonl(RESULTS / "ocr_eval.jsonl")))
            + " pages (August OCR, CPU, bfloat16) also covered two Bengali-pretrained checkpoints; "
            "guarded minus plain CER on Tesseract and EasyOCR text: " + old + ".\\end{minipage}\n"
            "\\end{table*}\n", encoding="utf-8")

    # ---- precision and prompt-format checks (third review, items C1, C2) ----
    if runs3:
        c1 = {(c["label"].split()[0], c["label"].split()[1].rstrip(":")): c for c in runs3["C1"]["outputs"]}
        c2 = {(c["model"], c["engine"]): c for c in runs3["C2"]}
        checks = [("Float32", "titullm-3b-instruct", c1),
                  ("Float32", "phi4-mini", c1),
                  ("Completion prompt", "titullm-1b", {k: v["vs_main"] for k, v in c2.items()}),
                  ("Normalised input", "banglat5", {k: v["vs_main"] for k, v in c2.items()})]
        rows = []
        for label, m, src in checks:
            cols = [label, MODEL_NAMES[m]]
            for e, _ in INPUTS:
                c = src.get((m, e))
                if c is None or c["n"] < 40:
                    cols += ["--", "--"]
                else:
                    cols += ["%.3f / %.3f" % (c["mean_b"], c["mean_a"]),
                             "%+.3f%s %s" % (c["delta"], star(c), ci(*c["ci"]))]
            rows.append(" & ".join(cols) + " \\\\")
        ft = runs3["C1"]["first_token"]
        ag = runs3["C1"]["first_token_agreement"]
        empty = ", ".join("%d/%d" % (c["empty_fp16"], c["empty_fp32"])
                          for e, _ in INPUTS for c in [c1[("titullm-3b-instruct", e)]])
        # Checks of the Bengali-pretrained checkpoints against the raw OCR, OCR inputs only.
        vs_raw = ([c["vs_raw"] for c in runs3["C2"] if c["engine"] != "ground_truth"]
                  + [c["fp32_minus_raw"] for (m, e), c in c1.items()
                     if m == "titullm-3b-instruct" and e != "ground_truth"])
        assert all(c["p"] < 0.001 and c["delta"] > 0 for c in vs_raw)
        best, worst = min(c["delta"] for c in vs_raw), max(c["delta"] for c in vs_raw)
        # Completion-prompt changes: the text reports them as unadjusted p < 0.001.
        compl = [c["vs_main"] for c in runs3["C2"] if c["model"] == "titullm-1b"]
        assert len(compl) == 3 and all(c["p"] < 0.001 for c in compl)
        (OUT_DIR / "checks.tex").write_text(
            "\\begin{table*}[t]\n\\centering\n"
            "\\caption{Precision and Prompt-Format Checks (Mean CER, 40 Pages)}\n"
            "\\label{tab:checks}\n\\setlength{\\tabcolsep}{3pt}\n\\scriptsize\n"
            "\\begin{tabular}{llcccccc}\n\\toprule\n"
            " & & \\multicolumn{2}{c}{Tesseract} & \\multicolumn{2}{c}{EasyOCR} & "
            "\\multicolumn{2}{c}{Clean text} \\\\\n"
            "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\n"
            "Check & Model & Main / check & $\\Delta$ [95\\% CI] & Main / check & $\\Delta$ [95\\% CI] & "
            "Main / check & $\\Delta$ [95\\% CI] \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\par\\smallskip\n"
            "\\begin{minipage}{\\linewidth}\\footnotesize Float32: instead of float16. Main: the setting "
            "of the main runs (float16 "
            "for the two models above 3B parameters; each model's shipped chat template; BanglaT5 on "
            "unnormalised text). Check: the same model rerun with one change. $\\Delta$: check $-$ "
            "main; CI: paired bootstrap, 10{,}000 resamples; $^{*}$: Wilcoxon signed-rank $p<0.05$, "
            "unadjusted (exploratory); the three completion-prompt changes have $p<0.001$. --: Phi-4-mini in "
            "float32 was not run on EasyOCR text, and on the clean text only " + "%d of 40 pages" % c1[("phi4-mini", "ground_truth")]["n"]
            + " finished within the session's time limit, so neither is reported. Float32 reruns: two Kaggle T4 GPUs. First-token "
            "check of TituLLMs 3B Instruct on all " + str(ag["chunks"]) + " chunks: the same first "
            "token in both precisions on " + str(ag["same_first_token"]) + ", end-of-sequence first on "
            + str(ag["eos_both"]) + " in both, no non-finite logits or hidden states, largest "
            "activation %.0f (float16 limit 65{,}504). Empty pages (float16/float32, Tesseract, EasyOCR, "
            "clean text): %s. Every check of a Bengali-pretrained checkpoint stays significantly worse "
            "than the raw OCR ($+$%.2f to $+$%.2f CER on OCR text, $p<0.001$).\\end{minipage}\n"
            "\\end{table*}\n" % (max(ft[p]["max_abs_activation"] for p in ft), empty, best, worst),
            encoding="utf-8")

    # ---- token-level Grad-CAM (third review, item X1) ----
    if runs3 and runs3.get("X1"):
        x1 = {(r["model"], r["engine"]): r for r in runs3["X1"]}
        xm = [m for m in MODEL_ORDER if any(k[0] == m for k in x1)]
        missing = [m for m in MODEL_ORDER if m != "symspell" and m not in xm]
        rows = []
        for i, (e, name) in enumerate([("tesseract", "Tesseract text"), ("ground_truth", "Clean text")]):
            rows.append("%s\\multicolumn{9}{l}{\\textit{%s}} \\\\" % ("\\midrule\n" if i else "", name))
            for m in xm:
                r = x1[(m, e)]
                h, lit, late = r["mid_hires"], r["mid_literal"], r["late_hires"]
                seq2seq = m == "banglat5"
                shares = (["%.2f" % h["ocr"], "--", "--"] if seq2seq else
                          ["%.2f" % h["ocr"], "%.2f" % h["prompt_other"], "%.2f" % h["prefix"]])
                rows.append(" & ".join(
                    [MODEL_NAMES[m], "%.0f" % (100 * r["copy_share"])] + shares
                    + ["%.0f" % (100 * h["src_peak"]), "%.0f" % (100 * late["src_peak"]),
                       "%.0f" % (100 * lit["src_peak"]), "%.1f" % (100 * r["src_peak_chance"])]) + " \\\\")
        chunks = {r["chunks"] for r in runs3["X1"]}
        toks = {r["tokens"] for r in runs3["X1"]}
        (OUT_DIR / "attribution.tex").write_text(
            "\\begin{table*}[t]\n\\centering\n"
            "\\caption{Gradient-Based Attribution of Generated Tokens: Token-Level Grad-CAM Maps}\n"
            "\\label{tab:attribution}\n\\setlength{\\tabcolsep}{4pt}\n\\footnotesize\n"
            "\\begin{tabular}{lcccccccc}\n\\toprule\n"
            " & & \\multicolumn{3}{c}{Share of the map (element-wise, mid)} & "
            "\\multicolumn{4}{c}{Map peaks on the copied source token (\\%)} \\\\\n"
            "\\cmidrule(lr){3-5}\\cmidrule(lr){6-9}\n"
            "Model & \\shortstack{Copied\\\\tokens (\\%)} & OCR text & \\shortstack{Rest of\\\\prompt} & "
            "\\shortstack{Own earlier\\\\output} & \\shortstack{Element-wise\\\\mid} & "
            "\\shortstack{Element-wise\\\\late} & \\shortstack{Literal\\\\mid} & Chance \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\par\\smallskip\n"
            "\\begin{minipage}{\\linewidth}\\footnotesize " + "/".join(map(str, sorted(chunks)))
            + " chunks per input (one from each of 10 pages, seed 403), up to 24 generated tokens scored "
            "per chunk (" + str(max(toks)) + " per model and input" + "".join(
                "; %s: %s, because of empty outputs" % (MODEL_NAMES[m], " and ".join(
                    "%d on %s" % (x1[(m, e)]["tokens"], n) for e, n in (("tesseract", "Tesseract"),
                                                                          ("ground_truth", "clean text"))))
                for m in xm if min(x1[(m, e)]["tokens"] for e in ("tesseract", "ground_truth")) < max(toks))
            + "); plain greedy decoding, the main prompt. "
            "Score: the logit of each generated token. Literal: Grad-CAM, gradients averaged over "
            "positions; element-wise: gradient $\\times$ hidden state per position, summed over the hidden dimensions. "
            "Mid/late: hidden states at 1/2 and 3/4 of the depth. The first token, special tokens and "
            "the predicting position are left out of every map. Copied tokens: output tokens aligned "
            "with an identical OCR token. Peak: the largest map value over the OCR text lies within one "
            "position of the copied token's source; chance: the rate for a random peak. BanglaT5 is an "
            "encoder-decoder model, so its map covers only the encoder input, which is the OCR text."
            + ("" if not missing else " Not run: " + " and ".join(MODEL_NAMES[m] for m in missing) + ".")
            + "\\end{minipage}\n\\end{table*}\n", encoding="utf-8")

    # ---- decoding ablation, 15-page CPU version (used only without the 40-page runs) ----
    if args.decoding.exists() and not runs3:
        abl = {(r["engine"], r["model"]): r
               for r in json.loads(args.decoding.read_text(encoding="utf-8"))}
        # The ablation's August OCR against the October OCR of the same pages.
        from metrics import cer
        aug = {r["page_id"]: r for r in load_jsonl(RESULTS / "ocr_eval.jsonl")}
        aug_cer = sum(cer(r["ground_truth"], r["tesseract"]) for r in aug.values()) / len(aug)
        oct_cer = sum(cer(ocr[p]["ground_truth"], ocr[p]["tesseract"]) for p in aug) / len(aug)
        rows = []
        for i, (e, name) in enumerate(ENGINES):
            for m in [m for m in models if (e, m) in abl]:
                r = abl[(e, m)]
                rows.append(" & ".join([
                    name, MODEL_NAMES[m], "%.3f" % r["ocr_mean"], "%.3f" % r["guarded_mean"],
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
            "\\texttt{no\\_repeat\\_ngram\\_size}=4, a common setting against repetition loops; for decoder-only models both "
            "also act on the prompt, so they penalise copying the OCR text. Plain: neither; greedy decoding "
            "with only the length cap. Setting: the " + str(len(aug)) + "-page random sample, OCR run "
            "in August 2026 (Tesseract build not recorded; mean raw Tesseract CER " + "%.3f" % aug_cer
            + " against " + "%.3f" % oct_cer + " for the same pages in the main setting), Colab CPU in "
            "bfloat16; the four models run there. "
            "CI: paired bootstrap, 10{,}000 resamples, seed 403. Last column: CER "
            "of the plain output against its own OCR input (near 0 = the model returned its input almost "
            "unchanged).\\end{minipage}\n\\end{table*}\n", encoding="utf-8")

    rev = json.loads(args.revision.read_text(encoding="utf-8")) if args.revision.exists() else None
    sens = (json.loads(args.sensitivity.read_text(encoding="utf-8"))
            if args.sensitivity.exists() else None)

    # ---- safeguards with their tests (review item E8, second review) ----
    if rev:
        cap = {(r["engine"], r["model"], r["tau"]): r for r in rev["edit_cap"]}
        taus = sorted({t for (_, _, t) in cap})
        rows = []
        for i, (e, name) in enumerate(ENGINES):
            rows.append(f"\\multicolumn{{{4 + len(taus)}}}{{l}}{{\\textit{{{name}: raw OCR "
                        f"CER {b[(e, models[0], 'cer')]['baseline_mean']:.3f}}}}} \\\\")
            for m in models:
                g = guard[(e, m)]
                cols = [MODEL_NAMES[m],
                        "%.3f (%d/%d)" % (g["cer_guard"], g["fallbacks"], g["n"]),
                        "%+.3f %s" % (g["mean_delta"], ci(g["ci_low"], g["ci_high"])),
                        fmt_p(g["wilcoxon_p_holm"])]
                for t in taus:
                    r = cap[(e, m, t)]
                    cols.append("%.3f%s" % (r["mean_cer"], sigmark(r)))
                rows.append(" & ".join(cols) + " \\\\")
            if i == 0:
                rows.append("\\midrule")
        (OUT_DIR / "safeguards.tex").write_text(
            "\\begin{table*}[t]\n\\centering\n"
            "\\caption{Post-Hoc Safeguards: Length Rule and Edit-Distance Cap (Mean CER, 40 Pages)}\n"
            "\\label{tab:safeguards}\n\\setlength{\\tabcolsep}{4pt}\n\\footnotesize\n"
            "\\begin{tabular}{lccc" + "c" * len(taus) + "}\n\\toprule\n"
            " & \\multicolumn{3}{c}{Length rule} & \\multicolumn{" + str(len(taus))
            + "}{c}{Edit-distance cap $\\tau$} \\\\\n"
            "\\cmidrule(lr){2-4}\\cmidrule(lr){5-" + str(4 + len(taus)) + "}\n"
            "Method & CER (fallbacks) & $\\Delta$CER [95\\% CI] & $p_{\\mathrm{Holm}}$ & "
            + " & ".join(str(t) for t in taus) + " \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\par\\smallskip\n"
            "\\begin{minipage}{\\linewidth}\\footnotesize Length rule: the raw OCR replaces an output "
            "whose length is outside 0.35--2.8 times the input; fallbacks: pages replaced. "
            "$\\Delta$: safeguarded minus raw OCR (positive = worse); CI: paired bootstrap, 10{,}000 "
            "resamples; $p_{\\mathrm{Holm}}$: Wilcoxon signed-rank test, Holm-corrected over the 18 "
            "length-rule tests. Edit-distance cap: the output is kept only if its CER against the raw "
            "OCR is at most $\\tau$; $^{+}$/$^{-}$: significantly higher/lower than the raw OCR, "
            "Holm-corrected over the " + str(len(cap)) + " cap tests. Both families are "
            "exploratory.\\end{minipage}\n"
            "\\end{table*}\n", encoding="utf-8")

    # ---- robustness: pooled CER, grapheme CER, book bootstrap, reading order,
    #      word recall (review items M2, M6, M7, M8; second review) ----
    if rev and sens:
        micro = {(r["engine"], r["model"]): r for r in rev["micro"]}
        graph = {(r["engine"], r["model"]): r for r in rev["grapheme_tests"]}
        clus = {(r["engine"], r["model"]): r for r in rev["cluster_bootstrap"]}
        ro = {(r["engine"], r["model"]): r for r in sens["reading_order_tests"]}
        rec = {(r["engine"], r["model"]): r for r in sens["word_recall_tests"]}
        rows = []
        for i, (e, name) in enumerate(ENGINES):
            raw = micro[(e, "(raw OCR)")]
            rows.append(f"\\multicolumn{{9}}{{l}}{{\\textit{{{name}: raw OCR pooled CER "
                        f"{raw['pooled_cer']:.3f}, grapheme CER {raw['mean_grapheme_cer']:.3f}, "
                        f"word recall {sens[e + '_raw_word_recall']:.3f} "
                        f"{ci(*r3['recall_vs_grapheme'][e]['recall_ci']).replace('+', '')}}}}} \\\\")
            for m in models:
                g, c, r_, w = graph[(e, m)], clus[(e, m)], ro[(e, m)], rec[(e, m)]
                rows.append(" & ".join([
                    MODEL_NAMES[m], "%.3f" % micro[(e, m)]["pooled_cer"],
                    "%+.3f %s" % (g["mean_delta"], ci(*g["ci"])), fmt_p(g["wilcoxon_p_holm"]),
                    ci(*c["book_ci"]),
                    "%+.3f" % r_["mean_delta"], fmt_p(r_["wilcoxon_p_holm"]),
                    "%+.3f" % w["mean_delta"], fmt_p(w["wilcoxon_p_holm"]),
                ]) + " \\\\")
            if i == 0:
                rows.append("\\midrule")
        (OUT_DIR / "robustness.tex").write_text(
            "\\begin{table*}[t]\n\\centering\n"
            "\\caption{Robustness of the Main Comparison to the Metric, the Resampling Unit and "
            "the Reference}\n\\label{tab:robustness}\n\\setlength{\\tabcolsep}{4pt}\n\\footnotesize\n"
            "\\begin{tabular}{lcccccccc}\n\\toprule\n"
            " & & \\multicolumn{2}{c}{Grapheme CER} & Books & "
            "\\multicolumn{2}{c}{Reading order} & \\multicolumn{2}{c}{Word recall} \\\\\n"
            "\\cmidrule(lr){3-4}\\cmidrule(lr){5-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\n"
            "Method & \\shortstack{Pooled\\\\CER} & $\\Delta$ [95\\% CI] & $p_{\\mathrm{Holm}}$ & "
            "$\\Delta$CER 95\\% CI & $\\Delta$CER & $p_{\\mathrm{Holm}}$ & $\\Delta$ & "
            "$p_{\\mathrm{Holm}}$ \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\par\\smallskip\n"
            "\\begin{minipage}{\\linewidth}\\footnotesize Pooled CER: total character errors over "
            "total reference characters. Grapheme CER: CER over extended grapheme clusters; "
            "$\\Delta$ = corrected $-$ raw OCR. Books: 95\\% interval of the mean change in CER "
            "when whole books (37) are resampled instead of pages. Reading order: change in CER "
            "against a reference whose regions follow the reading order declared in the PAGE-XML "
            "file (" + str(sens["reading_order"]["pages_reference_changed"]) + " of 40 references "
            "change). Word recall: share of reference word tokens found in the output, ignoring "
            "order and extra text; positive $\\Delta$ = better; the raw value is followed by its 95\\% "
            "bootstrap interval. Each $p_{\\mathrm{Holm}}$ column is its own exploratory family of 18 "
            "tests.\\end{minipage}\n\\end{table*}\n", encoding="utf-8")

    # ---- page type (review item E7) ----
    if sens:
        pt = {(r["engine"], r["model"]): r for r in sens["page_type"]}
        types = ["title", "verse", "prose", "list"]
        n = sens["page_type_counts"]
        rows = [" & ".join(["Raw OCR (CER)"] + ["%.3f" % pt[(e, "(raw OCR)")][t]
                                                for e, _ in ENGINES for t in types]) + " \\\\",
                "\\midrule"]
        for m in models:
            rows.append(" & ".join([MODEL_NAMES[m]] + ["%+.3f" % pt[(e, m)][t]
                                                       for e, _ in ENGINES for t in types]) + " \\\\")
        head = " & ".join(["Method"] + [f"{t} ({n[t]})" for _e in ENGINES for t in types])
        (OUT_DIR / "pagetypes.tex").write_text(
            "\\begin{table*}[t]\n\\centering\n"
            "\\caption{Mean Change in CER by Page Type}\n\\label{tab:pagetypes}\n"
            "\\setlength{\\tabcolsep}{4pt}\n\\footnotesize\n"
            "\\begin{tabular}{lcccccccc}\n\\toprule\n"
            " & \\multicolumn{4}{c}{Tesseract} & \\multicolumn{4}{c}{EasyOCR} \\\\\n"
            "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\n" + head + " \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\par\\smallskip\n"
            "\\begin{minipage}{\\linewidth}\\footnotesize First row: mean CER of the raw OCR; other "
            "rows: mean change in CER (corrected $-$ raw OCR, positive = worse). Page counts in "
            "parentheses. Title: title pages; verse: pages whose main text is metrical, including "
            "verse with a commentary; prose: running prose; list: contents, indexes, tables and "
            "dictionary entries. Descriptive only: the groups are too small for separate "
            "tests.\\end{minipage}\n\\end{table*}\n", encoding="utf-8")

    # ---- later third-review runs (matrix_analysis.py summaries) ----
    def matrix(path):
        if not path.exists():
            return None
        return {(c["engine"], c["model"]): c for c in json.loads(path.read_text(encoding="utf-8"))["cells"]}

    def hmark(m):
        """$^{+}$/$^{-}$ for a Holm-significant increase/decrease in one metric of a matrix cell."""
        if m["p_holm"] >= 0.05:
            return ""
        return "$^{+}$" if m["delta"] > 0 else "$^{-}$"

    def pct(x):
        return "--" if x != x else "%.0f" % (100 * x)

    def bf(c):
        """Broken / fixed as word counts: the shares have different denominators
        (words the raw OCR had right vs. wrong), so only counts can be compared."""
        w = c["words"]
        return "%d / %s" % (w["broken"], w["fixed"] if w["raw_wrong"] else "--")

    def nref(m):
        c = next(iter(m.values()))["words"]
        return format(c["raw_correct"] + c["raw_wrong"], ",").replace(",", "{,}")

    BF_NOTE = ("Broken / fixed: reference words the raw OCR had right that the output gets wrong, and words "
               "it had wrong that the output gets right, as counts of the %s aligned reference words, so the "
               "two can be compared directly.")

    # ---- region-level OCR (item C5) ----
    reg_ocr = json.loads(args.region_ocr.read_text(encoding="utf-8")) if args.region_ocr.exists() else None
    reg = matrix(args.region_runs)
    if reg and reg_ocr:
        methods = [m for m in MODEL_ORDER if all((e, m) in reg for e, _ in ENGINES)]
        methods += sorted({m for (_, m) in reg} - set(methods), key=lambda m: SCALE_ORDER.index(m)
                          if m in SCALE_ORDER else 99)
        raw = {e: reg[(e, methods[0])]["cer"]["input"] for e, _ in ENGINES}
        rows = [" & ".join(["Raw region OCR"] + sum([["%.3f" % raw[e], "", "", ""] for e, _ in ENGINES], []))
                + " \\\\", "\\midrule"]
        for m in methods:
            cols = [MODEL_NAMES[m]]
            for e, _ in ENGINES:
                c = reg.get((e, m))
                if c is None:
                    cols += ["--"] * 4
                    continue
                cols += ["%.3f%s" % (c["cer"]["output"], hmark(c["cer"])),
                         "%+.3f %s" % (c["cer"]["delta"], ci(*c["cer"]["ci"])),
                         "%+.3f%s" % (c["wer"]["delta"], hmark(c["wer"])),
                         bf(c)]
            rows.append(" & ".join(cols) + " \\\\")
        t, ez = reg_ocr["engines"]["tesseract"], reg_ocr["engines"]["easyocr"]
        (OUT_DIR / "region.tex").write_text(
            "\\begin{table*}[t]\n\\centering\n"
            "\\caption{Correction of Region-Level OCR (Mean CER and Change, 40 Pages)}\n"
            "\\label{tab:region}\n\\setlength{\\tabcolsep}{4pt}\n\\footnotesize\n"
            "\\begin{tabular}{lcccccccc}\n\\toprule\n"
            " & \\multicolumn{4}{c}{Tesseract} & \\multicolumn{4}{c}{EasyOCR} \\\\\n"
            "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\n"
            "Method & CER & $\\Delta$CER [95\\% CI] & $\\Delta$WER & \\shortstack{Broken /\\\\fixed (words)} & "
            "CER & $\\Delta$CER [95\\% CI] & $\\Delta$WER & \\shortstack{Broken /\\\\fixed (words)} \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\par\\smallskip\n"
            "\\begin{minipage}{\\linewidth}\\footnotesize Each PAGE-XML text region read separately (" +
            str(reg_ocr["regions"]) + " regions with text) and scored against the regions in reading "
            "order. Whole-page OCR from the same run: Tesseract %.3f, EasyOCR %.3f; region OCR changes "
            "them by %+.3f %s and %+.3f %s (paired bootstrap). The Tesseract build of this run does not "
            "reproduce the main run's text (CER %.2f between them), so Tesseract region results are "
            "compared only within this run; EasyOCR reproduces the main run (CER %.3f between them). "
            "$\\Delta$: corrected $-$ raw region OCR (positive = worse). $^{+}$/$^{-}$: significantly "
            "higher/lower, Wilcoxon signed-rank, Holm-corrected over the %d cells of the table for each "
            "metric (exploratory). " % (
                t["region_vs_same_run_wholepage"]["mean_b"], ez["region_vs_same_run_wholepage"]["mean_b"],
                t["region_vs_same_run_wholepage"]["delta"], ci(*t["region_vs_same_run_wholepage"]["ci"]),
                ez["region_vs_same_run_wholepage"]["delta"], ci(*ez["region_vs_same_run_wholepage"]["ci"]),
                t["rerun_vs_main_cer_mean"], ez["rerun_vs_main_cer_mean"], len(reg))
            + BF_NOTE % nref(reg) + "\\end{minipage}\n\\end{table*}\n",
            encoding="utf-8")

    # ---- model scale and family (item M1; M7 separately on 10 pages) ----
    m1, mainm = matrix(args.m1_runs), matrix(args.main_matrix)
    if m1 and mainm:
        src = {**mainm, **m1}
        rows = []
        for m in [m for m in SCALE_ORDER if ("tesseract", m) in src]:
            new = ("tesseract", m) in m1
            cols = [MODEL_NAMES[m] + ("" if new else "$^{\\dagger}$")]
            for e, _ in ENGINES:
                c = src[(e, m)]
                cols.append("%+.3f%s" % (c["cer"]["delta"], hmark(c["cer"])))
            g = src[("ground_truth", m)]
            cols.append("%.3f" % g["cer"]["output"])
            for e, _ in ENGINES:
                c = src[(e, m)]
                cols.append("%+.3f%s" % (c["wer"]["delta"], hmark(c["wer"])))
            for e, _ in ENGINES:
                c = src[(e, m)]
                cols.append(bf(c))
            rows.append(" & ".join(cols) + " \\\\")
        (OUT_DIR / "scale.tex").write_text(
            "\\begin{table*}[t]\n\\centering\n"
            "\\caption{General-Purpose Models by Family and Size (Change Against Raw OCR, 40 Pages)}\n"
            "\\label{tab:scale}\n\\setlength{\\tabcolsep}{4pt}\n\\footnotesize\n"
            "\\begin{tabular}{lccccccc}\n\\toprule\n"
            " & \\multicolumn{2}{c}{$\\Delta$CER} & Clean text & \\multicolumn{2}{c}{$\\Delta$WER} & "
            "\\multicolumn{2}{c}{Broken / fixed (words)} \\\\\n"
            "\\cmidrule(lr){2-3}\\cmidrule(lr){4-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\n"
            "Model & Tesseract & EasyOCR & CER & Tesseract & EasyOCR & Tesseract & EasyOCR \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\par\\smallskip\n"
            "\\begin{minipage}{\\linewidth}\\footnotesize $\\Delta$: corrected $-$ raw OCR (positive = "
            "worse); clean text: CER of the output when the reference itself is the input. "
            "$^{\\dagger}$: core models of the main runs; the others extend the comparison to newer "
            "checkpoints and to size within the Qwen3, Gemma and Llama families (same "
            "prompt, chunking and plain decoding; float16 above 3B parameters, Gemma 3 in float32). "
            "$^{+}$/$^{-}$: Holm-significant, Wilcoxon signed-rank, within each run's exploratory family "
            "(main run %d cells, added models %d cells, per metric). " % (len(mainm), len(m1))
            + BF_NOTE % nref(m1) + "\\end{minipage}\n\\end{table*}\n",
            encoding="utf-8")

    # ---- noise sweep (item M3) ----
    noise = matrix(args.noise_runs)
    if noise:
        levels = sorted({e for (e, _) in noise}, key=lambda e: float(e.split("_")[1]))
        inp = {e: next(c["cer"]["input"] for (ee, _), c in noise.items() if ee == e) for e in levels}
        rows = []
        for m in [m for m in NOISE_ORDER if any(mm == m for (_, mm) in noise)]:
            rows.append(" & ".join([MODEL_NAMES[m]] + ["%.3f%s" % (noise[(e, m)]["cer"]["output"],
                                                                    hmark(noise[(e, m)]["cer"]))
                                                       for e in levels]) + " \\\\")
        (OUT_DIR / "noise.tex").write_text(
            "\\begin{table}[t]\n\\centering\n"
            "\\caption{Correction of Synthetic Noise (Mean CER After Correction)}\n"
            "\\label{tab:noise}\n\\setlength{\\tabcolsep}{4pt}\n\\footnotesize\n"
            "\\begin{tabular}{l" + "c" * len(levels) + "}\n\\toprule\n"
            "Input CER & " + " & ".join("%.3f" % inp[e] for e in levels) + " \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\par\\smallskip\n"
            "\\begin{minipage}{\\linewidth}\\footnotesize The 40 references corrupted with a "
            "character-level noise model estimated from the OCR of the 10 development pages only, "
            "scaled to each target CER. $^{+}$/$^{-}$: higher/lower than the input, Wilcoxon "
            "signed-rank, Holm-corrected over the %d cells (exploratory).\\end{minipage}\n"
            "\\end{table}\n" % len(noise), encoding="utf-8")

    # ---- fine-tuning pilot (item M2) ----
    m2 = matrix(args.m2_runs)
    if m2:
        log = json.loads(args.m2_log.read_text(encoding="utf-8")) if args.m2_log.exists() else None
        rows = []
        for e, name in INPUTS:
            c = m2.get((e, "byt5-small-ft"))
            if c is None:
                continue
            rows.append(" & ".join([name, "%.3f" % c["cer"]["input"], "%.3f%s" % (c["cer"]["output"], hmark(c["cer"])),
                                    "%+.3f %s" % (c["cer"]["delta"], ci(*c["cer"]["ci"])),
                                    "%+.3f%s" % (c["wer"]["delta"], hmark(c["wer"])),
                                    bf(c),
                                    "%d/%d" % (c["cer"]["better"], c["cer"]["worse"]),
                                    str(c["truncated"])]) + " \\\\")
        note = ""
        if log:
            note = (" Training: %s training pairs, %s updates of %d pairs (%.0f min on one T4 GPU, including "
                    "the development evaluations); development-page CER %.3f after correction against %.3f raw "
                    "(checkpoint chosen on it)." %
                    (format(log["pairs"], ",").replace(",", "{,}"), format(log["steps"], ",").replace(",", "{,}"),
                     log.get("effective_batch", 8), log["train_minutes"],
                     log["best_dev_cer"], log["raw_dev_cer"]))
        (OUT_DIR / "finetune.tex").write_text(
            "\\begin{table*}[t]\n\\centering\n"
            "\\caption{Fine-Tuning Pilot: ByT5-Small Trained on Synthetic OCR Noise (40 Pages)}\n"
            "\\label{tab:finetune}\n\\setlength{\\tabcolsep}{4pt}\n\\footnotesize\n"
            "\\begin{tabular}{lccccccc}\n\\toprule\n"
            "Input & Raw CER & Corrected CER & $\\Delta$CER [95\\% CI] & $\\Delta$WER & "
            "\\shortstack{Broken /\\\\fixed (words)} & \\shortstack{Pages better /\\\\worse} & Truncated \\\\\n"
            "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\par\\smallskip\n"
            "\\begin{minipage}{\\linewidth}\\footnotesize google/byt5-small fine-tuned on Bengali "
            "Wikipedia segments and development-page reference lines, corrupted with the development-page "
            "noise model (no evaluation text in training)." + note + " $^{+}$/$^{-}$: Holm-significant "
            "over the %d cells (exploratory). " % len(m2) + BF_NOTE % nref(m2)
            + "\\end{minipage}\n\\end{table*}\n", encoding="utf-8")

    # ---- dictionary variants (item M4) and engine fusion (item M6) ----
    m4, fus = matrix(args.m4_runs), matrix(args.fusion_runs)
    if m4 and fus:
        names = {"symspell": "SymSpell, subtitles, $d\\le2$ (main)", "symspell-ed1": "SymSpell, subtitles, $d\\le1$",
                 "symspell-wiki": "SymSpell, Wikipedia, $d\\le2$", "symspell-wiki-ed1": "SymSpell, Wikipedia, $d\\le1$",
                 "fusion-conf": "Fusion, higher confidence", "fusion-margin": "Fusion, confidence margin"}
        rows = []
        for src, keys in ((m4, ["symspell", "symspell-ed1", "symspell-wiki", "symspell-wiki-ed1"]),
                          (fus, ["fusion-conf", "fusion-margin"])):
            if src is fus:
                rows.append("\\midrule")
            for m in [k for k in keys if ("easyocr", k) in src]:
                cols = [names[m]]
                for e, _ in ENGINES:
                    c = None if src is fus and e == "tesseract" else src.get((e, m))
                    # Fusion CER changes are descriptive: their family was chosen after a first scoring.
                    cols.append("--" if c is None else "%+.3f%s %s" % (
                        c["cer"]["delta"], "" if src is fus else hmark(c["cer"]), ci(*c["cer"]["ci"])))
                g = src.get(("ground_truth", m))
                cols.append("--" if g is None else "%.3f" % g["cer"]["output"])
                for e, _ in ENGINES:
                    c = None if src is fus and e == "tesseract" else src.get((e, m))
                    cols.append("--" if c is None else "%+.3f%s" % (c["wer"]["delta"], hmark(c["wer"])))
                for e, _ in ENGINES:
                    c = None if src is fus and e == "tesseract" else src.get((e, m))
                    cols.append("--" if c is None else bf(c))
                rows.append(" & ".join(cols) + " \\\\")
        n_fus = sum(1 for (e, _) in fus if e == "easyocr")
        fus4 = matrix(args.fusion_family4)
        p_fus = "; ".join("%s rule %.3f / %.3f" % (
            lbl, fus[("easyocr", k)]["cer"]["p_holm"], fus4[("easyocr", k)]["cer"]["p_holm"])
            for k, lbl in (("fusion-conf", "higher-confidence"), ("fusion-margin", "margin"))) if fus4 else ""
        fus_note = ("Fusion CER changes are descriptive, not significant. Holm-adjusted Wilcoxon $p$ over the "
                    "%d fusion cells / over the %d cells of the first scoring, which added the comparisons "
                    "with Tesseract output (the smaller family was chosen after it): " % (n_fus, len(fus4))
                    + p_fus + ". ") if fus4 else ""
        (OUT_DIR / "classical.tex").write_text(
            "\\begin{table*}[t]\n\\centering\n"
            "\\caption{Methods Without a Language Model: Dictionary Variants and Engine Fusion (40 Pages)}\n"
            "\\label{tab:classical}\n\\setlength{\\tabcolsep}{4pt}\n\\footnotesize\n"
            "\\begin{tabular}{lccccccc}\n\\toprule\n"
            " & \\multicolumn{2}{c}{$\\Delta$CER [95\\% CI]} & Clean text & \\multicolumn{2}{c}{$\\Delta$WER} & "
            "\\multicolumn{2}{c}{Broken / fixed (words)} \\\\\n"
            "\\cmidrule(lr){2-3}\\cmidrule(lr){4-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\n"
            "Method & Tesseract & EasyOCR & CER & Tesseract & EasyOCR & Tesseract & EasyOCR \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\par\\smallskip\n"
            "\\begin{minipage}{\\linewidth}\\footnotesize $\\Delta$: output $-$ raw OCR of that engine "
            "(positive = worse). SymSpell: lexicon of the 50{,}000 most frequent OpenSubtitles words "
            "(main baseline) or of the most frequent words of Bengali Wikipedia; $d$: maximum edit "
            "distance. Fusion: EasyOCR text in which short stretches where the engines disagree take "
            "Tesseract's reading if its confidence ranks higher within the page (margin: by at least "
            "0.25); it is compared with EasyOCR, its base, only, and needs two OCR readings, so it has "
            "no clean-text control. $^{+}$/$^{-}$: Holm-significant, Wilcoxon signed-rank, over the "
            "%d dictionary cells and, separately, the %d fusion cells, per metric (exploratory). "
            % (len(m4), n_fus) + fus_note + BF_NOTE % nref(m4) + "\\end{minipage}\n\\end{table*}\n", encoding="utf-8")

    # ---- constraints on generated tokens only (external review item 4) ----
    if args.review4_runs.exists():
        r4 = json.loads(args.review4_runs.read_text(encoding="utf-8"))
        g = {(r["model"], r["engine"], r["preset"]): r for r in r4["G"]}
        rows = []
        for i, (e, ename) in enumerate([("tesseract", "Tesseract"), ("ground_truth", "Clean text")]):
            ms = [m for m in MODEL_ORDER if (m, e, "guarded_gen") in g]
            rows.append("%s\\multicolumn{9}{l}{\\textit{%s}} \\\\" % ("\\midrule\n" if i else "", ename))
            for m in ms:
                cells = [MODEL_NAMES[m], "%.3f" % g[(m, e, "guarded_gen")]["vs_plain"]["mean_b"]]
                for pr in ("penalty_gen", "ngram_gen", "guarded_gen"):
                    r = g[(m, e, pr)]
                    marks = (("+" if r["vs_plain"]["p_holm"] < 0.05 and r["vs_plain"]["delta"] > 0 else "")
                             + ("\\circ" if r["vs_whole"]["p_holm"] >= 0.05 else ""))
                    cells += ["%.3f" % r["vs_whole"]["mean_b"],
                              "%.3f%s" % (r["vs_whole"]["mean_a"], "$^{%s}$" % marks if marks else "")]
                r = g[(m, e, "guarded_gen")]
                plain, whole_, gen_ = r["vs_plain"]["mean_b"], r["vs_whole"]["mean_b"], r["vs_whole"]["mean_a"]
                cells.append("%.0f\\%%" % (100 * (whole_ - gen_) / (whole_ - plain)))
                rows.append(" & ".join(cells) + " \\\\")
        n_up = sum(r["vs_plain"]["p_holm"] < 0.05 and r["vs_plain"]["delta"] > 0 for r in r4["G"])
        n_down = sum(r["vs_whole"]["p_holm"] < 0.05 and r["vs_whole"]["delta"] < 0 for r in r4["G"])
        (OUT_DIR / "decoding_gen.tex").write_text(
            "\\begin{table*}[t]\n\\centering\n"
            "\\caption{Repetition Constraints on the Whole Sequence and on Generated Tokens Only "
            "(Mean CER, 40 Pages)}\n\\label{tab:decodinggen}\n\\setlength{\\tabcolsep}{4pt}\n\\footnotesize\n"
            "\\begin{tabular}{lcccccccc}\n\\toprule\n"
            " & & \\multicolumn{2}{c}{Penalty} & \\multicolumn{2}{c}{$n$-gram ban} & "
            "\\multicolumn{2}{c}{Both (guarded)} & \\\\\n"
            "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\n"
            "Model & Plain & Whole & Generated & Whole & Generated & Whole & Generated & "
            "\\shortstack{Excess removed\\\\(both)} \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\par\\smallskip\n"
            "\\begin{minipage}{\\linewidth}\\footnotesize Whole: the constraint acts on the whole token "
            "sequence, prompt included, as \\texttt{generate} applies it (Table~\\ref{tab:decoding}); "
            "Generated: the same constraint restricted to the generated tokens (penalty 1.3 with "
            "\\texttt{prompt\\_ignore\\_length}, four-gram ban on the output only). $^{+}$: significantly "
            "higher than plain decoding (%d of %d); $^{\\circ}$: not significantly lower than the whole-"
            "sequence constraint (%d of %d are lower). Wilcoxon signed-rank, Holm-corrected within each "
            "comparison type over the %d settings (exploratory). Excess removed: the share of guarded "
            "decoding's increase over plain decoding that disappears when the constraints act on generated "
            "tokens only.\\end{minipage}\n\\end{table*}\n"
            % (n_up, len(r4["G"]), n_down, len(r4["G"]), len(r4["G"])), encoding="utf-8")

    # ---- output filter for commentary (external review item 5) ----
    if args.output_filter.exists():
        of = {(r["engine"], r["model"]): r for r in json.loads(args.output_filter.read_text(encoding="utf-8"))}
        rows = []
        for m in [m for m in MODEL_ORDER if ("tesseract", m) in of]:
            cols = [MODEL_NAMES[m]]
            for e, _ in ENGINES:
                r = of[(e, m)]
                cols += ["%.3f" % r["output_cer"], "%.3f" % r["filtered_cer"], "%d / %d" % (r["lines_dropped"], r["lines"])]
            rows.append(" & ".join(cols) + " \\\\")
        any_r = next(iter(of.values()))
        (OUT_DIR / "filter.tex").write_text(
            "\\begin{table*}[t]\n\\centering\n"
            "\\caption{Mean CER With a Generous Commentary Filter (40 Pages)}\n\\label{tab:filter}\n"
            "\\setlength{\\tabcolsep}{5pt}\n\\footnotesize\n"
            "\\begin{tabular}{lcccccc}\n\\toprule\n"
            " & \\multicolumn{3}{c}{Tesseract (raw %.3f)} & \\multicolumn{3}{c}{EasyOCR (raw %.3f)} \\\\\n"
            "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n"
            "Model & Output & Filtered & Lines dropped & Output & Filtered & Lines dropped \\\\\n\\midrule\n"
            % tuple(next(r["raw_cer"] for (ee, _), r in of.items() if ee == e) for e, _ in ENGINES)
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\par\\smallskip\n"
            "\\begin{minipage}{\\linewidth}\\footnotesize Filter: an output line is dropped when fewer than "
            "%.0f\\%% of its character trigrams occur in the page's OCR input. It removes preambles and "
            "descriptions of the text, but also invented lines and occasionally correct ones, so it can "
            "raise CER as well as lower it. A rough diagnostic of how much of the error commentary accounts "
            "for, not a bound and not a method we evaluate. Descriptive, no test.\\end{minipage}\n"
            "\\end{table*}\n" % (100 * any_r["min_overlap"]), encoding="utf-8")

    # ---- where the character errors go (external review item 7) ----
    if args.edit_analysis.exists():
        ea = json.loads(args.edit_analysis.read_text(encoding="utf-8"))
        names = {**MODEL_NAMES, "fusion-conf": "Fusion, higher confidence", "fusion-margin": "Fusion, margin"}
        order = ["qwen3-1.7b", "qwen3-4b", "qwen3-8b", "llama3.2-3b", "byt5-small-ft", "fusion-conf"]
        rows = []
        for i, (e, ename) in enumerate(ENGINES):
            rs = [r for m in order for r in ea if r["model"] == m and r["engine"] == e]
            if not rs:
                continue
            rows.append("%s\\multicolumn{8}{l}{\\textit{%s}} \\\\" % ("\\midrule\n" if i else "", ename))
            for r in rs:
                d, w = r["delta_by_group"], r["words"]
                rows.append(" & ".join([
                    names[r["model"]], "%d / %d" % (w.get("fixed", 0), w.get("broken", 0)),
                    "%+d" % d["fixed"], "%+d" % d["broken"],
                    "%+d (%d)" % (d["still_wrong"], r["still_wrong_changed"]),
                    "%+d" % (d["boundary"] + d["kept"]),
                    "%+d" % (r["char_errors_corrected"] - r["char_errors_raw"]),
                    "%.3f / %.3f" % (r["nospace_cer_raw"], r["nospace_cer_corrected"])]) + " \\\\")
        ref_chars = ea[0]["ref_chars"]
        (OUT_DIR / "editlevel.tex").write_text(
            "\\begin{table*}[t]\n\\centering\n"
            "\\caption{Where the Character Errors Go: Change in Character Errors by Reference-Word Group "
            "(40 Pages, Pooled)}\n\\label{tab:editlevel}\n\\setlength{\\tabcolsep}{4pt}\n\\footnotesize\n"
            "\\begin{tabular}{lccccccc}\n\\toprule\n"
            "Method & \\shortstack{Words\\\\fixed / broken} & \\shortstack{$\\Delta$ errors,\\\\fixed words} & "
            "\\shortstack{$\\Delta$ errors,\\\\broken words} & \\shortstack{$\\Delta$ errors, still-wrong\\\\words "
            "(words edited)} & \\shortstack{$\\Delta$ errors,\\\\spaces and other} & \\shortstack{$\\Delta$ errors,\\\\"
            "total} & \\shortstack{CER without spaces\\\\raw / corrected} \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\par\\smallskip\n"
            "\\begin{minipage}{\\linewidth}\\footnotesize Each character error of the alignment with the "
            "reference is assigned to a reference word (insertions to the word they follow); $\\Delta$: errors "
            "in the corrected text minus errors in the raw OCR, summed over the 40 pages (%s reference "
            "characters per engine), so the groups add up to the total. Fixed / broken: reference words the "
            "output gets right that the raw OCR had wrong, and the reverse. Still wrong: wrong in both; in "
            "brackets, how many of them the method edited. Spaces and other: errors on the spaces between "
            "words (merged or split words) and insertions next to words that stay right. CER without "
            "spaces: mean page CER with all spaces removed. Descriptive, no test.\\end{minipage}\n"
            "\\end{table*}\n" % "{:,}".format(ref_chars).replace(",", "{,}"), encoding="utf-8")

    # ---- large model on 10 pages (item M7) ----
    m7 = matrix(args.m7_runs)
    if m7 and any(m == "qwen3-32b" for (_, m) in m7):
        sizes = [m for m in SCALE_ORDER if m.startswith("qwen3") and ("tesseract", m) in m7]
        rows = []
        for m in sizes:
            cols = [MODEL_NAMES[m]]
            for e, _ in INPUTS:
                c = m7.get((e, m))
                cols.append("--" if c is None else "%+.3f%s" % (c["cer"]["delta"], hmark(c["cer"])))
            for e, _ in ENGINES:
                c = m7.get((e, m))
                cols.append("--" if c is None else "%+.3f%s" % (c["wer"]["delta"], hmark(c["wer"])))
            rows.append(" & ".join(cols) + " \\\\")
        n = next(iter(m7.values()))["n"]
        raw = {e: next(c["cer"]["input"] for (ee, _), c in m7.items() if ee == e) for e, _ in ENGINES}
        # The Holm family is every model scored on these pages, not only the Qwen3 rows shown.
        m7_models = sorted({m for (_, m) in m7})
        core = [m for m in m7_models if m in MODEL_ORDER]
        extra = [MODEL_NAMES[m] for m in SCALE_ORDER if m in m7_models and m not in MODEL_ORDER]
        m7_desc = "the %d core models and %s" % (len(core), ", ".join(extra))
        (OUT_DIR / "large.tex").write_text(
            "\\begin{table}[t]\n\\centering\n"
            "\\caption{Qwen3 from 0.6B to 32B Parameters on %d Pages (Change Against the Input)}\n"
            "\\label{tab:large}\n\\setlength{\\tabcolsep}{3pt}\n\\footnotesize\n"
            "\\begin{tabular}{lccccc}\n\\toprule\n"
            " & \\multicolumn{3}{c}{$\\Delta$CER} & \\multicolumn{2}{c}{$\\Delta$WER} \\\\\n"
            "\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}\n"
            "Model & Tess. & Easy. & Clean & Tess. & Easy. \\\\\n\\midrule\n" % n
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\par\\smallskip\n"
            "\\begin{minipage}{\\linewidth}\\footnotesize The %d pages of the attribution analysis "
            "(raw CER %.3f Tesseract, %.3f EasyOCR). Qwen3 32B in 4-bit (NF4) on two T4 GPUs; the others "
            "as in Table~\\ref{tab:scale}, restricted to the same pages. Tests: Wilcoxon signed-rank, "
            "Holm-corrected over the %d cells of each metric, that is, all %d models rescored on these pages "
            "(%s) on the %d inputs; the table shows the Qwen3 rows. With %d pages the smallest possible "
            "two-sided Wilcoxon $p$ is $2/2^{%d}$, so after this correction no change can reach significance "
            "(smallest adjusted $p$ %.3f): the table gives descriptive estimates, and the text gives 95\\%% "
            "intervals.\\end{minipage}\n\\end{table}\n"
            % (n, raw["tesseract"], raw["easyocr"], len(m7), len(m7_models), m7_desc,
               len({e for (e, _) in m7}), n, n, min(1.0, len(m7) * 2 / 2 ** n)),
            encoding="utf-8")

    print(f"Wrote tables to {OUT_DIR}")


if __name__ == "__main__":
    main()
