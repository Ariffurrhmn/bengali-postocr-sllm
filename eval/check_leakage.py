"""Data leakage checks for the study, written out as one JSON report.

The correction models are used zero-shot, so the usual train/test leakage only
applies to the parts that use data: the frozen page split, the few-shot prompt
examples, the synthetic noise model and the fine-tuning pilot (ByT5-small).
This script checks each of them against the 40 evaluation references and
prints what it found. It needs results/ocr_dev.jsonl and
results/ocr_eval40.jsonl (local run outputs). Text overlap is measured with
character shingles, with a positive control so that a score of 0 means
something.

Usage (from the repository root):
    python eval/check_leakage.py --out-json logs/leakage/leakage_report.json
"""
import argparse
import io
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from metrics import normalize

REPO = Path(__file__).resolve().parent.parent
NGRAM = 25  # same n-gram length as the overlap filter in correction/finetune_byt5.py


def read_ids(path):
    return [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def book_of(page_id):  # same rule as eval/revision_analysis.py
    return re.sub(r"_\d+$", "", page_id)


def load_jsonl(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def grams(text, n):
    t = normalize(text)
    return {t[i:i + n] for i in range(max(1, len(t) - n + 1))} if t else set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocr-dev", type=Path, default=REPO / "results/ocr_dev.jsonl")
    ap.add_argument("--ocr-eval", type=Path, default=REPO / "results/ocr_eval40.jsonl")
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()

    report = {}
    dev = read_ids(REPO / "data/split_dev.txt")
    ev = read_ids(REPO / "data/split_eval.txt")
    excl = read_ids(REPO / "data/excluded_no_ground_truth.txt")

    # 1. Page split
    report["split"] = {
        "dev_pages": len(dev), "eval_pages": len(ev), "excluded_pages": len(excl),
        "duplicate_ids_within_lists": len(dev) + len(ev) - len(set(dev) | set(ev)),
        "dev_eval_page_overlap": sorted(set(dev) & set(ev)),
        "excluded_in_dev_or_eval": sorted(set(excl) & (set(dev) | set(ev))),
        "books_shared_dev_eval": sorted({book_of(p) for p in dev} & {book_of(p) for p in ev}),
    }

    # 2. Reference text: duplicates and near-duplicates (25-character shingles)
    dev_rows = {r["page_id"]: r for r in load_jsonl(args.ocr_dev)}
    ev_rows = {r["page_id"]: r for r in load_jsonl(args.ocr_eval)}
    report["files_match_split"] = {
        "ocr_dev_pages_equal_split": sorted(dev_rows) == sorted(dev),
        "ocr_eval_pages_equal_split": sorted(ev_rows) == sorted(ev),
    }
    g_dev = {p: grams(r["ground_truth"], NGRAM) for p, r in dev_rows.items()}
    g_ev = {p: grams(r["ground_truth"], NGRAM) for p, r in ev_rows.items()}
    eval_union = set().union(*g_ev.values())

    def max_share(src, others):
        """Largest share of src's shingles found in any single other page."""
        best = (0.0, None)
        for q, g in others.items():
            if src and g:
                s = len(src & g) / len(src)
                if s > best[0]:
                    best = (s, q)
        return best

    cross = []
    for p, g in g_ev.items():
        s, q = max_share(g, g_dev)
        cross.append({"eval_page": p, "most_similar_dev_page": q, "share_of_eval_shingles_in_dev_page": round(s, 4)})
    within = []
    for p, g in g_ev.items():
        s, q = max_share(g, {k: v for k, v in g_ev.items() if k != p})
        within.append({"page": p, "most_similar_other_eval_page": q, "share": round(s, 4)})
    exact_dupes = [k for k, c in Counter(normalize(r["ground_truth"]) for r in ev_rows.values()).items() if c > 1]
    # Shorter shingles catch partial reuse (a shared heading or verse line).
    g12_dev = {p: grams(r["ground_truth"], 12) for p, r in dev_rows.items()}
    g12_ev = {p: grams(r["ground_truth"], 12) for p, r in ev_rows.items()}
    short = sorted(({"eval_page": p, "most_similar_dev_page": max_share(g, g12_dev)[1],
                     "share": round(max_share(g, g12_dev)[0], 4)} for p, g in g12_ev.items()),
                   key=lambda c: -c["share"])
    # Positive control: the same measure must find a page in itself, and a page
    # with half of another page pasted in must score close to 0.5. A short page
    # scores lower, because the shingles that cross the cut are lost (0.426 for
    # the first evaluation page, a short title page).
    p0 = next(iter(ev_rows))
    half = normalize(ev_rows[p0]["ground_truth"])[: len(normalize(ev_rows[p0]["ground_truth"])) // 2]
    planted = grams(half + " " + normalize(next(iter(dev_rows.values()))["ground_truth"]), NGRAM)
    report["positive_control"] = {
        "page_vs_itself": max_share(g_ev[p0], {p0: g_ev[p0]})[0],
        "half_of_eval_page_planted_in_dev_text": round(len(g_ev[p0] & planted) / len(g_ev[p0]), 3),
    }
    report["reference_text"] = {
        "eval_vs_dev_12char_top5": short[:5],
        "shingle_length": NGRAM,
        "eval_pages_with_identical_reference": len(exact_dupes),
        "max_eval_vs_dev_share": max(c["share_of_eval_shingles_in_dev_page"] for c in cross),
        "eval_vs_dev_top5": sorted(cross, key=lambda c: -c["share_of_eval_shingles_in_dev_page"])[:5],
        "max_eval_vs_eval_share": max(w["share"] for w in within),
        "eval_vs_eval_top5": sorted(within, key=lambda w: -w["share"])[:5],
    }

    # 3. Few-shot examples used in one prompt variant
    shots = json.loads((REPO / "data/fewshot_examples.json").read_text(encoding="utf-8"))
    report["fewshot"] = [{
        "page_id": s["page_id"], "page_in_dev": s["page_id"] in dev, "page_in_eval": s["page_id"] in ev,
        "shingles_shared_with_eval_refs": len(grams(s["corrected"], NGRAM) & eval_union)
                                          + len(grams(s["ocr"], NGRAM) & eval_union),
    } for s in shots]

    # 4. Noise model (estimated from development OCR only)
    noise = json.loads((REPO / "data/noise_model_dev.json").read_text(encoding="utf-8"))
    report["noise_model"] = {"source": noise.get("source"), "engines": noise.get("engines")}

    # 5. Fine-tuning pilot: historical training lines from dev pages (finetune_byt5.py
    # keeps lines of >= 20 characters and drops any that share a 25-gram with eval).
    dev_lines = [l.strip() for r in dev_rows.values() for l in r["ground_truth"].split("\n") if len(l.strip()) >= 20]
    hit = [l for l in dev_lines if grams(l, NGRAM) & eval_union]
    report["finetune_dev_lines"] = {"dev_lines": len(dev_lines), "lines_sharing_a_25gram_with_eval": len(hit),
                                    "note": "finetune_byt5.py drops these before training; the Wikipedia part was "
                                            "filtered the same way (Kaggle logs: 0 segments dropped)."}

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
