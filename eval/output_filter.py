"""How much of a model's error is commentary rather than correction? (external review item 5)

A documented, generous output filter in the spirit of Kanerva et al.: an output
line is dropped when fewer than --min-overlap of its character trigrams (after
normalisation, spaces removed) occur anywhere in the OCR input of that page.
This removes preambles, descriptions of the text and other commentary, but also
invented lines and occasionally correct ones, so the filtered CER is a rough
diagnostic, not a bound on what a filter could reach and not a corrected method. Every core model is rescored on both engines;
the result is descriptive (no test).

Usage:
    python eval/output_filter.py --ocr-path results/ocr_eval40.jsonl \
        --corrections results/gpu_eval40.jsonl --out-json results/output_filter40.json
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

# extended_analysis rewraps sys.stdout for UTF-8 on import.
import extended_analysis  # noqa: F401
from metrics import cer, normalize


def trigrams(text: str) -> set:
    t = normalize(text).replace(" ", "")
    return {t[i:i + 3] for i in range(len(t) - 2)}


def filter_output(ocr_text: str, output: str, min_overlap: float) -> tuple[str, int]:
    src = trigrams(ocr_text)
    kept, dropped = [], 0
    for line in output.split("\n"):
        grams = trigrams(line)
        if grams and len(grams & src) / len(grams) < min_overlap:
            dropped += 1
            continue
        kept.append(line)
    return "\n".join(kept), dropped


def load(path: Path):
    # Split on newlines only: outputs can contain U+2028/U+2029 (engineering note 12).
    return [json.loads(l) for l in path.read_text(encoding="utf-8").split("\n") if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocr-path", type=Path, required=True)
    ap.add_argument("--corrections", type=Path, required=True)
    ap.add_argument("--min-overlap", type=float, default=0.3)
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()
    pages = {r["page_id"]: r for r in load(args.ocr_path)}
    cells = defaultdict(list)
    for r in load(args.corrections):
        if r["engine"] in ("tesseract", "easyocr"):
            cells[(r["engine"], r["model_key"])].append(r)
    out = []
    for (e, m), rows in sorted(cells.items()):
        raw, unf, fil, dropped, lines = [], [], [], 0, 0
        for r in rows:
            p = pages[r["page_id"]]
            text, d = filter_output(p[e], r["raw_output"], args.min_overlap)
            dropped += d
            lines += len(r["raw_output"].split("\n"))
            raw.append(cer(p["ground_truth"], p[e]))
            unf.append(cer(p["ground_truth"], r["raw_output"]))
            fil.append(cer(p["ground_truth"], text))
        res = {"engine": e, "model": m, "n": len(rows), "min_overlap": args.min_overlap,
               "raw_cer": float(np.mean(raw)), "output_cer": float(np.mean(unf)),
               "filtered_cer": float(np.mean(fil)), "lines_dropped": dropped, "lines": lines}
        out.append(res)
        print(f"{e:<10} {m:<20} raw {res['raw_cer']:.3f}  output {res['output_cer']:.3f}  "
              f"filtered {res['filtered_cer']:.3f}  lines dropped {dropped}/{lines}")
    args.out_json.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
