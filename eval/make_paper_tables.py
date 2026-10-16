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
from collections import defaultdict
from pathlib import Path

# Imported before anything prints: inspect_examples re-wraps stdout for UTF-8.
from inspect_examples import MODEL_ORDER, script_profile, token_retention
from metrics import normalize

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"
OUT_DIR = REPO_ROOT / "paper" / "tables"

ENGINES = [("tesseract", "Tesseract"), ("easyocr", "EasyOCR")]
MODEL_NAMES = {"symspell": "SymSpell (dictionary)", "gemma-2b": "Gemma-2B-it", "llama3.2-1b": "Llama 3.2 1B",
               "banglat5": "BanglaT5", "titullm-1b": "TituLLMs 1B",
               "qwen3-1.7b": "Qwen3 1.7B", "phi4-mini": "Phi-4-mini",
               "titullm-1b-instruct": "TituLLMs 1B Instruct",
               "titullm-3b-instruct": "TituLLMs 3B Instruct"}
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
        + str(len(models)) + " Models}\n"
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
            "Holm-corrected over the 32 variant comparisons (exploratory family).\\end{minipage}\n"
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
            "unadjusted (exploratory). --: not run (Phi-4-mini in float32 on EasyOCR text) or "
            "incomplete (on clean text, 1 of 40 pages within the session's time limit). First-token "
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
            "\\caption{What the Models Attend To: Token-Level Grad-CAM Maps}\n"
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
            + " chunks per input (one from each of 10 pages, seed 403), " + "/".join(map(str, sorted(toks)))
            + " generated tokens scored per model and input; plain greedy decoding, the main prompt. "
            "Score: the logit of each generated token. Literal: Grad-CAM, gradients averaged over "
            "positions; element-wise: gradient $\\times$ hidden state per position (as in HiResCAM). "
            "Mid/late: hidden states at 1/2 and 3/4 of the depth. The first token, special tokens and "
            "the predicting position are left out of every map. Copied tokens: output tokens aligned "
            "with an identical OCR token. Peak: the largest map value over the OCR text lies within one "
            "position of the copied token's source; chance: the rate for a random peak. BanglaT5 is an "
            "encoder-decoder model, so its map covers only the encoder input, which is the OCR text. Not "
            "run: Phi-4-mini and TituLLMs 3B Instruct (time limit of the GPU session).\\end{minipage}\n"
            "\\end{table*}\n", encoding="utf-8")

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

    print(f"Wrote tables to {OUT_DIR}")


if __name__ == "__main__":
    main()
