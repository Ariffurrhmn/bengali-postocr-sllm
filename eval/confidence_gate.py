"""Confidence-gated correction (review item E4), applied to existing model
outputs; no new model run.

The model still corrects whole chunks, so it sees the context, but a change
is kept only where the OCR engine was unsure. Per page, the raw OCR words are
aligned with the model output (word-level, difflib). An edit that replaces or
deletes OCR words is accepted only if every OCR word it touches has a
confidence below the threshold; an inserted word is accepted only if an
adjacent OCR word is below it. Everything else stays as the OCR read it.

Confidences come from run_ocr_conf.py: Tesseract gives one per word (0-100),
EasyOCR one per detected text box (0-1), which every word in that box takes.
They are mapped onto the stored OCR words by word alignment; a stored word
with no counterpart counts as confident, so it is never changed.

Thresholds are fixed in advance: Tesseract 50/70/90, EasyOCR 0.5/0.7/0.9.

Usage:
    python eval/confidence_gate.py --ocr-path results/ocr_eval40.jsonl \\
        --conf-path results/ocr_conf40.jsonl \\
        --correction-path results/eval40_ocr.jsonl --out-prefix results/gated_gpu40
"""
import argparse
import difflib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

# extended_analysis rewraps sys.stdout for UTF-8 on import.
from extended_analysis import DEFAULT_RESAMPLES, DEFAULT_SEED, MODEL_ORDER, holm
from metrics import cer, normalize

THRESHOLDS = {"tesseract": [50, 70, 90], "easyocr": [0.5, 0.7, 0.9]}


def load(path: Path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def word_confidences(stored_text: str, units: list) -> list:
    """Confidence for each whitespace token of the stored OCR text. units are
    (text, conf) pairs; a unit's text may hold several words."""
    conf_words, confs = [], []
    for text, conf in units:
        for w in normalize(text).split():
            conf_words.append(w)
            confs.append(conf)
    stored = normalize(stored_text).split()
    out = [float("inf")] * len(stored)  # unmatched: treated as confident
    sm = difflib.SequenceMatcher(a=stored, b=conf_words, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal" or (tag == "replace" and i2 - i1 == j2 - j1):
            for k in range(i2 - i1):
                out[i1 + k] = confs[j1 + k]
    return out


def gate(stored_text: str, output: str, confs: list, threshold: float) -> str:
    ocr = normalize(stored_text).split()
    out = normalize(output).split()
    low = [c < threshold for c in confs]
    result = []
    sm = difflib.SequenceMatcher(a=ocr, b=out, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            result += ocr[i1:i2]
        elif tag in ("replace", "delete"):
            result += out[j1:j2] if all(low[i1:i2]) else ocr[i1:i2]
        else:  # insert before ocr[i1]
            near = [k for k in (i1 - 1, i1) if 0 <= k < len(ocr)]
            if any(low[k] for k in near):
                result += out[j1:j2]
    return " ".join(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ocr-path", type=Path, required=True)
    parser.add_argument("--conf-path", type=Path, required=True)
    parser.add_argument("--correction-path", type=Path, required=True)
    parser.add_argument("--out-prefix", type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(DEFAULT_SEED)

    pages = {r["page_id"]: r for r in load(args.ocr_path)}
    conf = {r["page_id"]: r for r in load(args.conf_path)}
    outputs = defaultdict(dict)
    for r in load(args.correction_path):
        outputs[(r["engine"], r["model_key"])][r["page_id"]] = r["raw_output"]
    models = [m for m in MODEL_ORDER if all((e, m) in outputs for e in THRESHOLDS)]

    word_conf = {}
    lines = []
    for e in THRESHOLDS:
        matched = sum(conf[p][f"{e}_matches"] for p in pages)
        for p in pages:
            units = ([(w["text"], w["conf"]) for w in conf[p]["tesseract_words"]] if e == "tesseract"
                     else [(b["text"], b["conf"]) for b in conf[p]["easyocr_boxes"]])
            word_conf[(e, p)] = word_confidences(pages[p][e], units)
        allc = [c for p in pages for c in word_conf[(e, p)]]
        unmatched = sum(c == float("inf") for c in allc)
        shares = ", ".join(f"<{t}: {np.mean([c < t for c in allc]):.0%}" for t in THRESHOLDS[e])
        lines.append(f"{e}: confidence run reproduces the stored text on {matched}/{len(pages)} pages; "
                     f"{unmatched}/{len(allc)} stored words without a confidence; words below threshold {shares}")

    rows, tests = [], []
    for e in THRESHOLDS:
        base = {p: cer(v["ground_truth"], v[e]) for p, v in pages.items()}
        for m in models:
            for t in THRESHOLDS[e]:
                pids = sorted(outputs[(e, m)])
                g = np.array([cer(pages[p]["ground_truth"],
                                  gate(pages[p][e], outputs[(e, m)][p], word_conf[(e, p)], t))
                              for p in pids])
                d = g - np.array([base[p] for p in pids])
                idx = rng.integers(0, len(d), (DEFAULT_RESAMPLES, len(d)))
                lo, hi = np.percentile(d[idx].mean(axis=1), [2.5, 97.5])
                pv = float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
                row = {"engine": e, "model": m, "threshold": t, "mean_cer": float(g.mean()),
                       "ungated_cer": float(np.mean([cer(pages[p]["ground_truth"], outputs[(e, m)][p])
                                                     for p in pids])),
                       "mean_delta": float(d.mean()), "ci": [float(lo), float(hi)], "wilcoxon_p": pv}
                rows.append(row)
    for r, a in zip(rows, holm([r["wilcoxon_p"] for r in rows])):
        r["wilcoxon_p_holm"] = float(a)

    lines.append(f"\nGated CER vs raw OCR (Holm over {len(rows)} tests)")
    lines.append(f"{'engine':<10}{'model':<22}{'thr':>5}{'ungated':>9}{'gated':>8}"
                 f"{'change vs raw OCR':>28}{'p Holm':>8}")
    for r in rows:
        lines.append(f"{r['engine']:<10}{r['model']:<22}{r['threshold']:>5}{r['ungated_cer']:>9.3f}"
                     f"{r['mean_cer']:>8.3f}   {r['mean_delta']:+.3f} [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]"
                     f"{r['wilcoxon_p_holm']:>8.3f}")
    text = "\n".join(lines)
    print(text)
    args.out_prefix.with_suffix(".txt").write_text(text + "\n", encoding="utf-8")
    args.out_prefix.with_suffix(".json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"\nWrote {args.out_prefix}.txt and .json")


if __name__ == "__main__":
    main()
