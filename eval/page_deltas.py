"""Page-level CER changes of one cell (engine x model), largest first, to see
whether a few pages drive a cell's mean (draft-04 correction C10).

Usage:
    python eval/page_deltas.py --ocr-path results/region/ocr_region40.jsonl \
        --corrections results/review3c/c5_phi4-mini.jsonl --engine easyocr \
        --out-json results/region_phi4-mini_easyocr_pages.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

# extended_analysis rewraps sys.stdout for UTF-8 on import.
from extended_analysis import page_measures


def load(path):
    # Split on newlines only: outputs can contain U+2028/U+2029 (engineering note 12).
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").split("\n") if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocr-path", type=Path, required=True)
    ap.add_argument("--corrections", type=Path, required=True)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()
    pages = {r["page_id"]: r for r in load(args.ocr_path)}
    rows = []
    for r in load(args.corrections):
        if r["engine"] != args.engine or r["page_id"] not in pages:
            continue
        p = pages[r["page_id"]]
        m = page_measures(p["ground_truth"], p[args.engine], r["raw_output"])
        rows.append({"page_id": r["page_id"], "delta": m["corr_cer"] - m["base_cer"], "raw_cer": m["base_cer"],
                     "corrected_cer": m["corr_cer"], "truncated": bool(r.get("truncated")),
                     "long_insert_chars": m["long_insert_chars"]})
    rows.sort(key=lambda x: -x["delta"])
    d = np.array([x["delta"] for x in rows])
    trunc = [x["delta"] for x in rows if x["truncated"]]
    summary = {"engine": args.engine, "file": args.corrections.name, "n": len(rows),
               "mean_delta": float(d.mean()), "median_delta": float(np.median(d)),
               "truncated_pages": len(trunc), "truncated_share_of_total_delta": float(sum(trunc) / d.sum()),
               "mean_delta_without_largest": float(d[1:].mean()), "pages": rows}
    print(json.dumps({k: v for k, v in summary.items() if k != "pages"}, indent=1))
    for x in rows[:5]:
        print("%+.3f  %s  raw %.3f  corrected %.3f  truncated %s" % (
            x["delta"], x["page_id"], x["raw_cer"], x["corrected_cer"], x["truncated"]))
    args.out_json.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
