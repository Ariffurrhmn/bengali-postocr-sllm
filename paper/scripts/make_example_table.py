"""Qualitative example table for the paper (review item T4): two reference
lines with the raw OCR and the output of each method on Tesseract input,
words that differ from the reference shown in red.

For each output page, the span aligned to the reference line is found with
find_examples.best_span; words outside the line at either end (context from
the neighbouring lines) are trimmed, and the remaining words are aligned with
the reference word by word. Bengali text is set with \\bn{} (preamble.tex),
which needs XeLaTeX and the bundled Noto Serif Bengali font.

Usage:
    python paper/scripts/make_example_table.py --ocr-path results/ocr_eval40.jsonl \\
        --correction-path results/eval40_ocr.jsonl --out paper/tables/examples.tex
"""
import argparse
import difflib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
from find_examples import best_span  # noqa: E402
from metrics import cer, normalize  # noqa: E402

EXAMPLES = [
    ("VT_961_k_0172", "শাখায় ঝুলিয়া থাকে, রাত্রি কালে আহার নিমিত্ত বহির্গত",
     "``[they] hang from branches, and at night go out for food''"),
    ("279_2_D_32_0007", "যুদ্ধেস্থিত কৌরবেরে দেখিয়া অর্জুন। অস্ত্রপাত",
     "``seeing the Kauravas arrayed for battle, Arjuna \\ldots''"),
]
ROWS = [("Tesseract (raw)", "tesseract", None), ("EasyOCR (raw)", "easyocr", None),
        ("SymSpell", "tesseract", "symspell"), ("Gemma-2B-it", "tesseract", "gemma-2b"),
        ("Qwen3 1.7B", "tesseract", "qwen3-1.7b"), ("Phi-4-mini", "tesseract", "phi4-mini"),
        ("Llama 3.2 1B", "tesseract", "llama3.2-1b"), ("BanglaT5", "tesseract", "banglat5"),
        ("TituLLMs 1B", "tesseract", "titullm-1b"),
        ("TituLLMs 1B Inst.", "tesseract", "titullm-1b-instruct"),
        ("TituLLMs 3B Inst.", "tesseract", "titullm-3b-instruct")]
LATEX_SPECIAL = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
                 "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
                 "\\": r"\textbackslash{}"}


def esc(text: str) -> str:
    return "".join(LATEX_SPECIAL.get(c, c) for c in text)


def trim(span_words, ref_words):
    """Drop a word at either end while that lowers the span's CER against the
    reference line: removes context picked up from neighbouring lines, by the
    same rule for every row."""
    ref = " ".join(ref_words)
    words = list(span_words)
    changed = True
    while changed and len(words) > 1:
        changed = False
        for cut in (words[1:], words[:-1]):
            if cer(ref, " ".join(cut)) < cer(ref, " ".join(words)):
                words, changed = cut, True
                break
    return words


def highlight(span_words, ref_words) -> str:
    parts = []
    sm = difflib.SequenceMatcher(a=ref_words, b=span_words, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        for w in span_words[j1:j2]:
            parts.append(esc(w) if tag == "equal" else "\\textcolor{red}{" + esc(w) + "}")
    return " ".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ocr-path", type=Path, required=True)
    parser.add_argument("--correction-path", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    pages = {json.loads(l)["page_id"]: json.loads(l) for l in
             args.ocr_path.read_text(encoding="utf-8").splitlines() if l.strip()}
    out = {}
    for l in args.correction_path.read_text(encoding="utf-8").splitlines():
        if l.strip():
            r = json.loads(l)
            out[(r["engine"], r["model_key"], r["page_id"])] = r["raw_output"]

    cells = {}
    for k, (pid, ref, _) in enumerate(EXAMPLES):
        ref = normalize(ref)
        ref_words = ref.split()
        cells[("Reference", k)] = "\\bn{%s}" % esc(ref)
        for label, engine, model in ROWS:
            text = pages[pid][engine] if model is None else out[(engine, model, pid)]
            span = best_span(ref, text) if normalize(text) else ""
            words = trim(span.split(), ref_words) if span else []
            # Only the Bengali goes in \bn{}: the Bengali font has no Latin letters.
            shown = "\\bn{%s}" % highlight(words, ref_words) if words else "\\textit{(empty)}"
            cells[(label, k)] = shown + " \\hfill{\\scriptsize %.2f}" % cer(ref, " ".join(words))

    rows = ["Reference & " + " & ".join(cells[("Reference", k)] for k in range(len(EXAMPLES))) + " \\\\",
            "Gloss & " + " & ".join("\\textit{%s}" % EXAMPLES[k][2] for k in range(len(EXAMPLES))) + " \\\\",
            "\\midrule"]
    for label, _, model in ROWS:
        rows.append(label + " & " + " & ".join(cells[(label, k)]
                                               for k in range(len(EXAMPLES))) + " \\\\")
        if label == "EasyOCR (raw)":
            rows.append("\\midrule")
    ids = ", ".join(pid.replace("_", "\\_") for pid, _, _ in EXAMPLES)
    # The table body is set twice: live under XeLaTeX, and as a PDF rendered by
    # XeLaTeX that pdfLaTeX includes as an image. Overleaf's default compiler
    # is pdfLaTeX, which cannot set Bengali, so the paper builds either way.
    body = args.out.with_name(args.out.stem + "_body.tex")
    body.write_text(
        "\\footnotesize\n\\renewcommand{\\arraystretch}{1.15}\n"
        "\\begin{tabular}{p{2.3cm}p{7.2cm}p{7.2cm}}\n\\toprule\n"
        " & Example 1 & Example 2 \\\\\n\\midrule\n" + "\n".join(rows) + "\n"
        "\\bottomrule\n\\end{tabular}\n\\par\\smallskip\n\\begin{minipage}{\\linewidth}\\footnotesize "
        f"Pages {ids}. Each row shows the span of the output aligned to the reference line, "
        "with words at either end dropped while that lowers its CER; "
        "words that differ from the reference are in \\textcolor{red}{red}, and the number at the "
        "right is the span's CER against the reference. Glosses are approximate. A dotted circle "
        "marks a sign with no letter to attach to (for example a doubled virama) in the text "
        "itself.\\end{minipage}\n",
        encoding="utf-8")
    args.out.write_text(
        "\\begin{table*}[t]\n\\centering\n"
        "\\caption{Two Reference Lines with the Raw OCR and Each Method's Output on Tesseract Input}\n"
        "\\label{tab:examples}\n"
        "\\ifXeTeX\n  \\input{tables/" + body.stem + "}\n"
        "\\else\n  % pdfLaTeX: the same table, rendered by XeLaTeX (render_examples_pdf.py)\n"
        "  \\includegraphics[width=\\textwidth]{tables/" + body.stem + ".pdf}\n"
        "\\fi\n\\end{table*}\n", encoding="utf-8")
    print(f"Wrote {args.out} and {body}; now run paper/scripts/render_examples_pdf.py")


if __name__ == "__main__":
    main()
