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

  paper/tables/dataset.tex      - evaluation pages (review item T1)
  paper/tables/models.tex       - models and configuration (T2)
  paper/tables/editops.tex      - edit operations per page (T3)
  paper/tables/clean.tex        - clean-text control (E1)

Main tables use the 40-page GPU run with plain decoding. The decoding
ablation (decoding.tex) uses the 15-page CPU runs, where both settings exist.

Inputs (defaults, all in results/): ocr_eval40.jsonl, gpu_eval40_ocr.jsonl,
bootstrap_gpu40.json (bootstrap.py), extended_gpu40.json (extended_analysis.py),
revision_gpu40.json (revision_analysis.py), model_profile.json
(model_profile.py), decoding_ablation.json (compare_decoding.py).
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
MODEL_NAMES = {"symspell": "SymSpell (dictionary)", "gemma-2b": "Gemma 2B", "llama3.2-1b": "Llama 3.2 1B",
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
        (OUT_DIR / "dataset.tex").write_text(
            "\\begin{table}[t]\n\\centering\n\\caption{Evaluation Pages}\n\\label{tab:dataset}\n"
            "\\begin{tabular}{ll}\n\\toprule\n" + "\n".join(rows) + "\n\\bottomrule\n"
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
                rows.append(f"{MODEL_NAMES[m]} & {r['mean_cer']:.3f} & {r['pooled_cer']:.3f} & "
                            f"{r['pages_exact']}/{r['n']} \\\\")
            (OUT_DIR / "clean.tex").write_text(
                "\\begin{table}[t]\n\\centering\n"
                "\\caption{Clean-Text Control: Ground Truth Given as Input}\n\\label{tab:clean}\n"
                "\\begin{tabular}{lccc}\n\\toprule\n"
                "Model & Mean CER & Pooled CER & Pages unchanged \\\\\n\\midrule\n"
                + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\par\\smallskip\n"
                "\\begin{minipage}{\\linewidth}\\footnotesize Each model receives the reference "
                "text itself, with the same prompt and decoding. CER is measured against that "
                "reference, so any value above 0 is damage to correct text. Pages unchanged: "
                "CER exactly 0 after normalisation.\\end{minipage}\n\\end{table}\n",
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
        rows = []
        for m in [m for m in models if m in prof]:
            r = prof[m]
            cpu = "%.1f" % r["cpu_min_per_page"] if r["cpu_min_per_page"] is not None else "---"
            gpu = "%.2f" % r["gpu_min_per_page"]
            rows.append(f"{MODEL_NAMES[m]} & \\texttt{{{r['checkpoint'].replace('_', chr(92) + '_')}@{REVISIONS[m]}}} & "
                        f"{PARAMS_B[m]:.2f} & {GPU_DTYPE[m]} & {r['tokens_per_char']:.2f} & "
                        f"{r['mean_chars_per_chunk']:.0f} & {cpu} & {gpu} \\\\")
        (OUT_DIR / "models.tex").write_text(
            "\\begin{table*}[t]\n\\centering\n\\caption{Models and Configuration}\n"
            "\\label{tab:models}\n\\setlength{\\tabcolsep}{4pt}\n\\footnotesize\n"
            "\\begin{tabular}{llcccccc}\n\\toprule\n"
            "Model & Checkpoint & Params (B) & GPU precision & Tokens/char & Chars/chunk & "
            "CPU min/page & GPU min/page \\\\\n\\midrule\n" + "\n".join(rows) + "\n"
            "\\bottomrule\n\\end{tabular}\n\\par\\smallskip\n\\begin{minipage}{\\linewidth}"
            "\\footnotesize Checkpoint: Hugging Face model and revision. "
            "Parameters from each checkpoint's configuration, tied embeddings "
            "counted once. Tokens/char: each tokenizer on the 40 reference texts. Chars/chunk: "
            "mean characters per chunk of about 250 tokens on raw Tesseract text. Decoding for "
            "all models: greedy, at most $\\min(512,\\max(64,1.5n))$ new tokens, no repetition "
            "penalty or n-gram ban. CPU: free Colab runtime, bfloat16, 15 pages (---: not run on "
            "CPU). GPU: NVIDIA T4, 40 pages. Times are per page and engine.\\end{minipage}\n"
            "\\end{table*}\n", encoding="utf-8")

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
            "Holm-corrected over the 32 variant comparisons.\\end{minipage}\n\\end{table}\n",
            encoding="utf-8")

    # ---- decoding ablation (review item C1) ----
    if args.decoding.exists():
        abl = {(r["engine"], r["model"]): r
               for r in json.loads(args.decoding.read_text(encoding="utf-8"))}
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
            "\\texttt{no\\_repeat\\_ngram\\_size}=4, as in the original runs; for decoder-only models both "
            "also act on the prompt, so they penalise copying the OCR text. Plain: neither; greedy decoding "
            "with only the length cap. CI: paired bootstrap, 10{,}000 resamples, seed 403. Last column: CER "
            "of the plain output against its own OCR input (near 0 = the model returned its input almost "
            "unchanged).\\end{minipage}\n\\end{table*}\n", encoding="utf-8")

    print(f"Wrote tables to {OUT_DIR}")


if __name__ == "__main__":
    main()
