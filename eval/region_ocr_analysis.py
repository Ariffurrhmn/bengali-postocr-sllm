"""Region-level OCR (third review C5/M5): how much does reading each PAGE-XML
region separately lower the OCR error rate?

Inputs (ocr/run_region_ocr.py, Kaggle CPU run of 2026-10-17):
  --region     ocr_region40.jsonl            region OCR, reference = regions in reading order
  --wholepage  ocr_region40_wholepage.jsonl  whole-page re-OCR from the same run (same engine build)
  --verify     ocr_region40_verify.json      whether that re-OCR matches the main run
  --main       ocr_eval40.jsonl              the main run's whole-page OCR

Each whole-page text is scored against the main reference (file order), each
region text against the region reference (reading order); the paired
difference per page is tested with the paired bootstrap (10,000 resamples,
seed 403) and the Wilcoxon signed-rank test of extended_analysis.py. Also
reported: region-level CER against the reading-order whole-page comparison of
the sensitivity analysis (same reference order), empty region outputs, and
how far the same-run whole-page OCR is from the main run (if far, the
main-run comparison is not valid for that engine and only the same-engine
comparison is reported as a result).

Usage (from the repo root):
    python eval/region_ocr_analysis.py --region results/region/ocr_region40.jsonl \\
        --wholepage results/region/ocr_region40_wholepage.jsonl \\
        --verify results/region/ocr_region40_verify.json \\
        --main results/ocr_eval40.jsonl --out-prefix results/region_ocr40
"""
import argparse
import json
from pathlib import Path

import numpy as np

# extended_analysis rewraps sys.stdout for UTF-8 on import.
from extended_analysis import DEFAULT_RESAMPLES, DEFAULT_SEED, bootstrap_ci, wilcoxon_p
from metrics import cer

ENGINES = ["tesseract", "easyocr"]


def load(path):
    return {r["page_id"]: r for r in
            (json.loads(l) for l in Path(path).read_text(encoding="utf-8").split("\n") if l.strip())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", type=Path, required=True)
    ap.add_argument("--wholepage", type=Path, required=True)
    ap.add_argument("--verify", type=Path, required=True)
    ap.add_argument("--main", type=Path, required=True)
    ap.add_argument("--out-prefix", type=Path, required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(DEFAULT_SEED)
    region, whole, main_ocr = load(args.region), load(args.wholepage), load(args.main)
    verify = json.loads(args.verify.read_text(encoding="utf-8"))
    pages = sorted(main_ocr)
    assert set(region) == set(whole) == set(pages)
    res = {"versions": verify["versions"], "pages": len(pages),
           "regions": sum(len(region[p]["regions"]) for p in pages), "engines": {}}
    lines = [f"Region-level OCR, {len(pages)} pages, {res['regions']} regions with text; "
             f"engines {verify['versions']}"]

    def compare(a, b):
        d = np.array([a[p] - b[p] for p in pages])
        lo, hi = bootstrap_ci(d, rng, DEFAULT_RESAMPLES)
        return {"mean_a": float(np.mean(list(a.values()))), "mean_b": float(np.mean(list(b.values()))),
                "delta": float(d.mean()), "ci": [lo, hi], "p": wilcoxon_p(d),
                "better": int((d < 0).sum()), "worse": int((d > 0).sum())}

    for e in ENGINES:
        main_cer = {p: cer(main_ocr[p]["ground_truth"], main_ocr[p][e]) for p in pages}
        rerun_cer = {p: cer(whole[p]["ground_truth"], whole[p][e]) for p in pages}
        region_cer = {p: cer(region[p]["ground_truth"], region[p][e]) for p in pages}
        drift = [cer(main_ocr[p][e], whole[p][e]) for p in pages]
        identical = sum(r[f"{e}_identical"] for r in verify["per_page"])
        n_reg = sum(len(region[p]["regions"]) for p in pages)
        empty = sum(1 for p in pages for g in region[p]["regions"] if not g[e].strip())
        r = {"identical_to_main": identical, "rerun_vs_main_cer_mean": float(np.mean(drift)),
             "rerun_vs_main_cer_max": float(np.max(drift)),
             "main_wholepage_cer": float(np.mean(list(main_cer.values()))),
             "region_vs_same_run_wholepage": compare(region_cer, rerun_cer),
             "region_vs_main_wholepage": compare(region_cer, main_cer),
             "empty_regions": empty, "regions": n_reg}
        # The main-run comparison is valid only if the re-run reproduces the main OCR.
        r["main_comparison_valid"] = r["rerun_vs_main_cer_mean"] < 0.01
        res["engines"][e] = r
        s, m = r["region_vs_same_run_wholepage"], r["region_vs_main_wholepage"]
        lines += [f"\n{e}: same-run whole-page OCR vs main run: identical on {identical}/{len(pages)} pages, "
                  f"CER between them mean {r['rerun_vs_main_cer_mean']:.3f} (max {r['rerun_vs_main_cer_max']:.3f}); "
                  f"main-run comparison {'valid' if r['main_comparison_valid'] else 'NOT valid'}",
                  f"  region vs same-run whole page: {s['mean_a']:.3f} vs {s['mean_b']:.3f}  "
                  f"delta {s['delta']:+.3f} [{s['ci'][0]:+.3f}, {s['ci'][1]:+.3f}]  p {s['p']:.4f}  "
                  f"pages better/worse {s['better']}/{s['worse']}",
                  f"  region vs main-run whole page: {m['mean_a']:.3f} vs {m['mean_b']:.3f}  "
                  f"delta {m['delta']:+.3f} [{m['ci'][0]:+.3f}, {m['ci'][1]:+.3f}]  p {m['p']:.4f}",
                  f"  empty region outputs: {empty}/{n_reg}"]
    text = "\n".join(lines)
    print(text)
    args.out_prefix.with_suffix(".txt").write_text(text + "\n", encoding="utf-8")
    args.out_prefix.with_suffix(".json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
