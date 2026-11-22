"""Holm correction over the trained-corrector test family (D85; family fixed on 2026-11-21 in
plan-remaining-2026-11-21.md as "BL-test-trained": 2 models x 2 engines x CER and WER = 8 tests).

Reads the rev 07 summaries (correction/score_trained_corrector.py on the reviewed BL test units, D79)
and adjusts their Wilcoxon p-values for the plain output, the result of record. The guarded variant
was written after reading validation logs (D76) and stays outside the family. No rescoring: the
p-values are read as stored.

Usage:
    python eval/trained_family_holm.py --summaries results/bengali-rev-07-score-reviewed/rev07_score \\
        --out results/rev07_holm.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extended_analysis import holm  # noqa: E402

MODELS = ("byt5-small", "banglabyt5")
ENGINES = ("tesseract", "easyocr")
METRICS = ("blockcer", "blockwer")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summaries", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows = []
    for m in MODELS:
        s = json.loads((args.summaries / f"{m}_test_bl" / "summary.json").read_text(encoding="utf-8"))
        for e in ENGINES:
            for k in METRICS:
                r = s["engines"][e][f"plain_{k}"]
                rows.append({"model": m, "engine": e, "metric": k, "mean_raw": r["mean_raw"],
                             "mean_corrected": r["mean_corrected"], "delta": r["delta"], "ci95": r["ci95"],
                             "wilcoxon_p": r["wilcoxon_p"], "better": r["better"], "worse": r["worse"]})
    for r, a in zip(rows, holm([r["wilcoxon_p"] for r in rows])):
        r["holm_p"] = float(a)
    args.out.write_text(json.dumps({"family": "BL-test-trained", "tests": len(rows), "variant": "plain",
                                    "rows": rows}, indent=1), encoding="utf-8")
    for r in rows:
        print(f"{r['model']:<11}{r['engine']:<10}{r['metric']:<9} {r['mean_raw']:.4f} -> {r['mean_corrected']:.4f} "
              f"delta {r['delta']:+.4f} [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]  p {r['wilcoxon_p']:.4f}  Holm {r['holm_p']:.4f}")


if __name__ == "__main__":
    main()
