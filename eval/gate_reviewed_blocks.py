"""Confidence gating on the reviewed BL test set (D84). Run once, after rev 09 is checked for completeness.

Applies the paper's gating rule (eval/confidence_gate.py: word_confidences and gate, unchanged) block by
block to the saved zero-shot outputs (rev 08 and SymSpell), with the rev 09 confidences. No model runs.
Thresholds fixed in advance: Tesseract 50/70/90, EasyOCR 0.5/0.7/0.9.

Per method, engine and threshold: per-block CER (main) and WER of the gated output, character-weighted
within a page, mean over pages; change against raw OCR with the paired 10,000-resample bootstrap (seed 403)
and the Wilcoxon test; Holm over the 54 block-CER tests (9 methods x 3 thresholds x 2 engines), as the
paper's gating family. Also, without tests: gated against ungated block CER, the share of stored words
below each threshold, words without a confidence, and how often the confidence run's text equals the
stored text. The script refuses to score if a page or unit lacks confidences or outputs, unless
--allow-incomplete is given.

Usage:
    python eval/gate_reviewed_blocks.py --ocr-region <rev06 reviewed_region.jsonl> \\
        --conf <rev09 reviewed_conf.jsonl> --split-dir data/split_v2 \\
        --outputs results/rev08/*.jsonl results/rev08_symspell/zs_symspell.jsonl --out-dir results/rev09_gated
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "correction"))

from confidence_gate import THRESHOLDS, gate, word_confidences  # noqa: E402  (extended_analysis rewraps stdout)
from extended_analysis import holm  # noqa: E402
from metrics import block_scores  # noqa: E402
from score_trained_corrector import SEED, paired  # noqa: E402

ENGINES = ("tesseract", "easyocr")
METHOD_ORDER = ["symspell", "gemma-2b", "qwen3-1.7b", "phi4-mini", "llama3.2-1b", "banglat5", "titullm-1b",
                "titullm-1b-instruct", "titullm-3b-instruct"]


def load_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").split("\n") if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocr-region", type=Path, required=True)
    ap.add_argument("--conf", type=Path, required=True)
    ap.add_argument("--split-dir", type=Path, required=True)
    ap.add_argument("--split", default="test_bl")
    ap.add_argument("--outputs", type=Path, nargs="+", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--allow-incomplete", action="store_true")
    args = ap.parse_args()

    ids = {l.strip() for l in (args.split_dir / f"{args.split}.txt").read_text(encoding="utf-8").split("\n") if l.strip()}
    pages = sorted((r for r in load_jsonl(args.ocr_region) if r["page_id"] in ids), key=lambda r: r["page_id"])
    conf = {r["page_id"]: {u["id"]: u for u in r["units"]} for r in load_jsonl(args.conf)}
    outs = {}
    for path in args.outputs:
        for r in load_jsonl(path):
            if r.get("error") is None:
                outs[(r["model_key"], r["engine"], r["page_id"])] = r
    methods = sorted({k[0] for k in outs}, key=lambda m: (METHOD_ORDER.index(m) if m in METHOD_ORDER else 99, m))

    problems = []
    for p in pages:
        for g in p["regions"]:
            if g["id"] not in conf.get(p["page_id"], {}):
                problems.append(f"no confidences: {p['page_id']} unit {g['id']}")
        for m in methods:
            for e in ENGINES:
                r = outs.get((m, e, p["page_id"]))
                if r is None or r["block_ids"] != [g["id"] for g in p["regions"]]:
                    problems.append(f"no outputs or different blocks: {m} {e} {p['page_id']}")
    if problems and not args.allow_incomplete:
        sys.exit("refusing to score, incomplete inputs:\n  " + "\n  ".join(problems[:40])
                 + (f"\n  ... {len(problems)} in all" if len(problems) > 40 else ""))

    # confidence of every stored word, per page, engine and unit
    wc, stats_lines = {}, []
    for e in ENGINES:
        allc, match, n_units = [], 0, 0
        for p in pages:
            for g in p["regions"]:
                u = conf.get(p["page_id"], {}).get(g["id"])
                if u is None:
                    continue
                units = ([(w["text"], w["conf"]) for w in u["tesseract_words"]] if e == "tesseract"
                         else [(b["text"], b["conf"]) for b in u["easyocr_boxes"]])
                wc[(e, p["page_id"], g["id"])] = word_confidences(g[e], units)
                allc += wc[(e, p["page_id"], g["id"])]
                match += u[f"{e}_matches"]
                n_units += 1
        unmatched = sum(c == float("inf") for c in allc)
        shares = ", ".join(f"<{t}: {np.mean([c < t for c in allc]):.1%}" for t in THRESHOLDS[e])
        stats_lines.append(f"{e}: confidence-run text equals stored text on {match}/{n_units} units; "
                           f"{unmatched}/{len(allc)} stored words without a confidence; words below threshold {shares}")

    rng = np.random.default_rng(SEED)
    rows, page_rows = [], []
    for e in ENGINES:
        for m in methods:
            for t in THRESHOLDS[e]:
                per = []
                for p in pages:
                    r = outs.get((m, e, p["page_id"]))
                    if r is None or r["block_ids"] != [g["id"] for g in p["regions"]]:
                        continue
                    refs = [g["ground_truth"] for g in p["regions"]]
                    src = [g[e] for g in p["regions"]]
                    gated = [gate(g[e], o, wc.get((e, p["page_id"], g["id"]), [float("inf")] * len(g[e].split())), t)
                             if (e, p["page_id"], g["id"]) in wc else g[e]
                             for g, o in zip(p["regions"], r["outputs"])]
                    raw_b, ung_b, gat_b = block_scores(refs, src), block_scores(refs, r["outputs"]), block_scores(refs, gated)
                    per.append({"page_id": p["page_id"], "page_type": p.get("page_type", ""),
                                "raw_blockcer": raw_b["cer"], "ungated_blockcer": ung_b["cer"], "gated_blockcer": gat_b["cer"],
                                "raw_blockwer": raw_b["wer"], "gated_blockwer": gat_b["wer"]})
                if len(per) < 2:
                    continue
                cer_vs_raw = paired([x["raw_blockcer"] for x in per], [x["gated_blockcer"] for x in per], rng)
                wer_vs_raw = paired([x["raw_blockwer"] for x in per], [x["gated_blockwer"] for x in per], rng)
                rows.append({"engine": e, "method": m, "threshold": t, "pages": len(per),
                             "raw_blockcer": cer_vs_raw["mean_raw"], "ungated_blockcer": float(np.mean([x["ungated_blockcer"] for x in per])),
                             "gated_blockcer": cer_vs_raw["mean_corrected"], "blockcer_vs_raw": cer_vs_raw,
                             "gated_minus_ungated": float(np.mean([x["gated_blockcer"] - x["ungated_blockcer"] for x in per])),
                             "blockwer_vs_raw": wer_vs_raw})
                page_rows += [{"engine": e, "method": m, "threshold": t, **x} for x in per]
    adj = holm([r["blockcer_vs_raw"]["wilcoxon_p"] if r["blockcer_vs_raw"].get("wilcoxon_p") is not None else 1.0
                for r in rows]) if rows else []
    for r, a in zip(rows, adj):
        r["holm_p"] = float(a)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"split": args.split, "pages": len(pages), "incomplete": bool(problems), "problems": problems,
               "confidence_stats": stats_lines, "family": {"tests": len(rows), "correction": "Holm"}, "rows": rows}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    with (args.out_dir / "pages.jsonl").open("w", encoding="utf-8") as f:
        for x in page_rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    tag = " [INCOMPLETE]" if problems else ""
    lines = [f"Confidence gating on {args.split}: {len(pages)} pages; gated block CER against raw OCR, "
             f"Holm over {len(rows)} tests{tag}", *stats_lines, "",
             f"{'engine':<10}{'method':<21}{'thr':>5}{'raw':>8}{'ungated':>9}{'gated':>8}  "
             f"{'change vs raw [95% CI]':<28}{'p':>8}{'Holm p':>8}{'gated-ungated':>15}"]
    for r in rows:
        s = r["blockcer_vs_raw"]
        p = s.get("wilcoxon_p")
        lines.append(f"{r['engine']:<10}{r['method']:<21}{r['threshold']:>5}{r['raw_blockcer']:>8.4f}{r['ungated_blockcer']:>9.4f}"
                     f"{r['gated_blockcer']:>8.4f}  {s['delta']:+.4f} [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}]  "
                     f"{(f'{p:.4f}' if p is not None else 'n/a'):>8}{r['holm_p']:>8.4f}{r['gated_minus_ungated']:>+15.4f}")
    (args.out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
