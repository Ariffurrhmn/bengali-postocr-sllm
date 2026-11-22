"""Tables of the held-out BL test set (revision 2026-11): split, zero-shot methods, trained correctors,
confidence gating. Every number comes from a result file; nothing is typed by hand.

Inputs (local, results/ is gitignored):
  data/split_v2/split_manifest.csv                     split v2 (D83)
  results/rev08_score/summary.json                     zero-shot methods and SymSpell (D82)
  results/bengali-rev-07-score-reviewed/rev07_score/   trained correctors (D79)
  results/rev07_holm.json                              Holm over the trained family (D85)
  results/rev09_gated/summary.json                     confidence gating (D84), if it exists
Outputs: paper/tables/test_split.tex, test_zeroshot.tex, test_trained.tex, test_gating.tex

Usage: python paper/scripts/make_test_tables.py
"""
import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "results"
OUT = REPO / "paper" / "tables"
ENGINES = [("tesseract", "Tesseract"), ("easyocr", "EasyOCR")]
NAMES = {"symspell": "SymSpell (dictionary)", "gemma-2b": "Gemma-2B-it", "qwen3-1.7b": "Qwen3 1.7B",
         "phi4-mini": "Phi-4-mini", "llama3.2-1b": "Llama 3.2 1B", "banglat5": "BanglaT5",
         "titullm-1b": "TituLLMs 1B", "titullm-1b-instruct": "TituLLMs 1B Instruct",
         "titullm-3b-instruct": "TituLLMs 3B Instruct", "byt5-small": "ByT5-small", "banglabyt5": "BanglaByT5"}
GENERAL = ["symspell", "gemma-2b", "qwen3-1.7b", "phi4-mini", "llama3.2-1b"]
DIAGNOSTIC = ["banglat5", "titullm-1b", "titullm-1b-instruct", "titullm-3b-instruct"]
BLOCKS = 1250


def fmt_p(p):
    if p is None:
        return "--"
    if p < 0.001:
        return "$<$0.001"
    return f"{p:.3f}" if p < 0.05 else f"{p:.2f}\\textsuperscript{{ns}}"


def ci(c):
    return f"[{c[0]:+.3f}, {c[1]:+.3f}]"


def num(x):
    return f"{x:,}".replace(",", "{,}")


def table(label, caption, cols, header, rows, note, wide=True, size="\\footnotesize", sep="3pt"):
    env = "table*" if wide else "table"
    return "\n".join([f"\\begin{{{env}}}[t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
                      f"\\setlength{{\\tabcolsep}}{{{sep}}}", size, f"\\begin{{tabular}}{{{cols}}}", "\\toprule",
                      header + " \\\\", "\\midrule", *rows, "\\bottomrule", "\\end{tabular}", "\\par\\smallskip",
                      f"\\begin{{minipage}}{{\\linewidth}}\\footnotesize {note}\\end{{minipage}}", f"\\end{{{env}}}", ""])


def split_table():
    rows = list(csv.DictReader((REPO / "data/split_v2/split_manifest.csv").open(encoding="utf-8")))
    out = []
    for s, name in (("train", "Training"), ("val", "Validation"), ("test", "Test (held out)"), ("unused", "Not used")):
        sel = [r for r in rows if r["split"] == s]
        groups = len({r["group_id"] for r in sel})
        reid = sum(r["source"] == "REID" for r in sel)
        bl = sum(r["source"] == "BL" for r in sel)
        chars = sum(int(r["gt_chars"]) for r in sel)
        out.append(f"{name} & {len(sel)} & {groups} & {reid} & {bl} & {num(chars)} \\\\")
    note = ("Split v2, frozen on 20 November 2026 before any model saw a test page. REID: pages of the REID2019 release "
            "(PAGE-XML); BL: pages of the British Library Transkribus set (ALTO). Books are merged into groups by "
            "known aliases and by shelfmark siblings (the same shelfmark without its last part); no group appears in "
            "two of training, validation and test, and no test group contains a REID page. Test pages need at least "
            "300 reference characters, at least half of them Bengali. Not used: REID pages without a transcription, "
            "short BL pages and one duplicate page. Characters: reference characters per split.")
    return table("tab:testsplit", "Revision Split: Training, Validation and Held-Out Test Pages", "lccccc",
                 "Split & Pages & Book groups & REID & BL & Characters", out, note, wide=False, sep="4pt")


def zeroshot_table():
    s = json.loads((RES / "rev08_score/summary.json").read_text(encoding="utf-8"))
    holm = {(r["method"], r["engine"], r["metric"]): r["holm_p"] for r in s["primary_family"]["results"]}
    rows = []
    for e, en in ENGINES:
        first = s["methods"]["symspell"][e]
        rows.append(f"{en} & (no correction) & {first['blockcer']['mean_raw']:.3f} & --- & --- & "
                    f"{first['blockwer']['mean_raw']:.3f} & --- & --- & --- & --- \\\\")
        for group in (GENERAL, DIAGNOSTIC):
            for m in group:
                d = s["methods"][m][e]
                c, w = d["blockcer"], d["blockwer"]
                rows.append(f"{en} & {NAMES[m]} & {c['mean_corrected']:.3f} & {c['delta']:+.3f} {ci(c['ci95'])} & "
                            f"{fmt_p(holm[(m, e, 'blockcer')])} & {w['mean_corrected']:.3f} & {w['delta']:+.3f} {ci(w['ci95'])} & "
                            f"{fmt_p(holm[(m, e, 'blockwer')])} & {c['better']}/{c['worse']} & {d['truncated']} \\\\")
            if group is GENERAL:
                rows.append("\\cmidrule(lr){2-10}")
        if e == "tesseract":
            rows.append("\\midrule")
    clean = "; ".join(f"{NAMES[m]} {s['methods'][m]['clean_text']['changed']}" for m in GENERAL + DIAGNOSTIC)
    note = (f"41 held-out pages, {num(BLOCKS)} reviewed text blocks, each corrected on its own (plain greedy decoding, "
            "main prompt). Block CER/WER: errors summed over the blocks of a page divided by its reference characters "
            "(words), then averaged over pages. $\\Delta$ = corrected $-$ uncorrected (positive = worse); CI: paired "
            "bootstrap over pages, 10{,}000 resamples, seed 403. $p_{\\mathrm{Holm}}$: two-sided Wilcoxon signed-rank "
            "test, Holm-corrected over these 36 tests, the primary family of the held-out evaluation. ns: not "
            "significant. Pages: pages whose block CER fell/rose, of 41. Trunc.: blocks whose output hit the token cap. "
            "Below the rule: the four Bengali-pretrained checkpoints, unadapted and diagnostic. Blocks under 10 "
            "characters (463 Tesseract, 459 EasyOCR) are passed through unchanged by the language models. "
            f"Clean-text control (block references as input), blocks changed of {num(BLOCKS)}: {clean}.")
    return table("tab:testzeroshot", "Held-Out Test: Zero-Shot Methods on 41 Book-Disjoint Pages", "llcccccccc",
                 "Engine & Method & CER & $\\Delta$CER [95\\% CI] & $p_{\\mathrm{Holm}}$ & WER & $\\Delta$WER [95\\% CI] & "
                 "$p_{\\mathrm{Holm}}$ & \\shortstack{Pages\\\\better/worse} & Trunc.", rows, note)


def trained_table():
    h = json.loads((RES / "rev07_holm.json").read_text(encoding="utf-8"))
    holm = {(r["model"], r["engine"], r["metric"]): r["holm_p"] for r in h["rows"]}
    rows = []
    for e, en in ENGINES:
        for m in ("byt5-small", "banglabyt5"):
            s = json.loads((RES / f"bengali-rev-07-score-reviewed/rev07_score/{m}_test_bl/summary.json").read_text(encoding="utf-8"))
            d = s["engines"][e]
            pc, pw, gc = d["plain_blockcer"], d["plain_blockwer"], d["guarded_blockcer"]
            rows.append(f"{en} & {NAMES[m]} & {pc['mean_raw']:.3f} & {pc['mean_corrected']:.3f} & {pc['delta']:+.3f} {ci(pc['ci95'])} & "
                        f"{fmt_p(holm[(m, e, 'blockcer')])} & {pw['delta']:+.3f} {ci(pw['ci95'])} & {fmt_p(holm[(m, e, 'blockwer')])} & "
                        f"{gc['mean_corrected']:.3f} & {gc['delta']:+.3f} {ci(gc['ci95'])} & {d['truncated_blocks']} \\\\")
        if e == "tesseract":
            rows.append("\\midrule")
    cl = []
    for m in ("byt5-small", "banglabyt5"):
        s = json.loads((RES / f"bengali-rev-07-score-reviewed/rev07_score/{m}_test_bl/summary.json").read_text(encoding="utf-8"))
        c = s["clean_text"]
        cl.append(f"{NAMES[m]} changed {c['changed']} of {num(c['blocks'])} blocks (CER {c['mean_cer_vs_reference']:.3f})")
    note = ("Fully trained correctors on the 41 held-out pages, block by block. Both were trained on aligned OCR and "
            "reference pairs from the 62 training pages and synthetic pairs, and their checkpoints were chosen on the 14 "
            "validation pages; the reviewed test units were scored once. Plain: the model output, the result of record; "
            "$p_{\\mathrm{Holm}}$ over the 8 plain CER and WER tests, a family fixed before any test score. Guarded: "
            "a block falls back to the raw OCR when the output hits the length cap, changes length by more than a "
            "factor of two, or contains a run of six identical characters absent from the input; the rule was written "
            "after reading validation outputs, so it is secondary and untested. Trunc.: blocks that hit the length cap. "
            "Clean-text control: " + "; ".join(cl) + ".")
    return table("tab:testtrained", "Held-Out Test: Fully Trained Correctors (ByT5-Small and BanglaByT5)", "llccccccccc",
                 "Engine & Model & Raw CER & CER & $\\Delta$CER [95\\% CI] & $p_{\\mathrm{Holm}}$ & $\\Delta$WER [95\\% CI] & "
                 "$p_{\\mathrm{Holm}}$ & \\shortstack{Guarded\\\\CER} & \\shortstack{Guarded\\\\$\\Delta$CER [95\\% CI]} & Trunc.",
                 rows, note, sep="2pt")


def gating_table():
    p = RES / "rev09_gated/summary.json"
    if not p.exists():
        return None
    s = json.loads(p.read_text(encoding="utf-8"))
    by = {(r["engine"], r["method"], r["threshold"]): r for r in s["rows"]}
    rows = []
    for e, en in ENGINES:
        ths = sorted({r["threshold"] for r in s["rows"] if r["engine"] == e})
        rows.append(f"\\multicolumn{{2}}{{l}}{{\\textit{{{en}}}}} & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{$<${t:g}}}" for t in ths) + " \\\\")
        for m in GENERAL + DIAGNOSTIC:
            if (e, m, ths[0]) not in by:
                continue
            r0 = by[(e, m, ths[0])]
            cells = []
            for t in ths:
                r = by[(e, m, t)]
                d = r["blockcer_vs_raw"]
                # direction of a significant Wilcoxon test: which way most pages moved (rank-based),
                # not the sign of the mean, which outliers can flip
                star = "" if r["holm_p"] >= 0.05 else ("$^{-}$" if d["better"] > d["worse"] else "$^{+}$")
                cells.append(f"{r['gated_blockcer']:.3f} & {d['delta']:+.3f}{star}")
            rows.append(f"& {NAMES[m]} ({r0['ungated_blockcer']:.3f}) & " + " & ".join(cells) + " \\\\")
        if e == "tesseract":
            rows.append("\\midrule")
    raw = {e: next(r["raw_blockcer"] for r in s["rows"] if r["engine"] == e) for e, _ in ENGINES}
    stats = " ".join(x.replace("%", "\\%").replace("<", "$<$") for x in s["confidence_stats"])
    note = (f"Confidence gating on the 41 held-out pages with the thresholds fixed for the 40-page study: a model's "
            f"change to a block is kept only where every OCR word it touches has a confidence below the threshold. "
            f"Raw block CER: Tesseract {raw['tesseract']:.3f}, EasyOCR {raw['easyocr']:.3f}; in brackets, the ungated "
            "block CER. $\\Delta$: gated $-$ raw OCR, mean over pages. $^{+}$/$^{-}$: Holm-significant Wilcoxon test over "
            "the 54 tests (exploratory family, as in the 40-page study), marked by the direction most pages moved"
            + "".join(f"; for {NAMES[r['method']]} on {dict(ENGINES)[r['engine']]} at {r['threshold']:g} the mean is "
                      f"{'negative' if r['blockcer_vs_raw']['delta'] < 0 else 'positive'} but "
                      f"{r['blockcer_vs_raw']['worse']} pages got worse and {r['blockcer_vs_raw']['better']} better"
                      for r in s["rows"] if r["holm_p"] < 0.05
                      and (r["blockcer_vs_raw"]["delta"] < 0) != (r["blockcer_vs_raw"]["better"] > r["blockcer_vs_raw"]["worse"]))
            + ". Confidence run: " + stats)
    return table("tab:testgating", "Held-Out Test: Confidence-Gated Correction (Block CER)", "llcccccc",
                 "& Method (ungated) & Gated & $\\Delta$ & Gated & $\\Delta$ & Gated & $\\Delta$", rows, note)


def main():
    made = []
    for name, fn in (("test_split", split_table), ("test_zeroshot", zeroshot_table),
                     ("test_trained", trained_table), ("test_gating", gating_table)):
        tex = fn()
        if tex is None:
            print(f"{name}: input missing, not written")
            continue
        (OUT / f"{name}.tex").write_text(tex, encoding="utf-8")
        made.append(name)
    print("wrote", ", ".join(made))


if __name__ == "__main__":
    main()
