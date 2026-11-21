"""Score a trained corrector (train_corrector.py output) on held-out pages.

Reads the region-level OCR of the chosen split, corrects every text block of every page
with the checkpoint in --weights (usually <train out-dir>/best), and compares with the raw
OCR of the same blocks against the page reference. Nothing is tuned here: run it once on
the validation pages as a check of the training log, then once on test_bl for the number
that goes in the paper.

Per engine and per variant it reports mean CER and WER of raw OCR and of the corrected text,
the mean paired difference (corrected minus raw) with a 95% paired bootstrap interval over
pages (10,000 resamples, seed 403, as eval/bootstrap.py), a Wilcoxon signed-rank p-value
and the number of pages that got better and worse.

Variants:
  plain    the corrector's output as it is. This is the result.
  guarded  a fallback chosen AFTER seeing the validation logs (D76), reported as a secondary
           analysis only: a block goes back to its raw OCR when the model hit the length cap
           without an end token, when the output is shorter than half or longer than twice the
           input, or when it contains a run of 6 or more identical characters that the input
           does not contain (the repeated danda seen in the training logs).
Clean text (--clean, default on): the reference text of each block goes through the
corrector too; reports the share of blocks changed and the CER of the output against the
reference (should be 0 for a model that copies clean text).

Usage (Kaggle or local GPU, CPU also works but is slow):
    python correction/score_trained_corrector.py --model banglabyt5 \\
        --weights results/bengali-rev-03-train-both/train/banglabyt5/best \\
        --ocr-region <shard_a_region.jsonl> --split-dir <folder with val.txt, test_bl.txt> \\
        --split test_bl --out-dir results/rev03_test_banglabyt5
Smoke test (random tiny model, no weights): add --tiny and leave out --weights.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

from corrector_common import MODELS, REPO, correct_texts, flat, load_jsonl, load_model, read_ids

sys.path.insert(0, str(REPO / "eval"))
from metrics import block_scores, cer, wer  # noqa: E402

SEED = 403
RESAMPLES = 10000
ENGINES = ("tesseract", "easyocr")
RUN = re.compile(r"(.)\1{5,}")


def degenerate_runs(text: str) -> set:
    return {m.group(1) for m in RUN.finditer(text)}


def guard(src: str, out: str, truncated: bool) -> bool:
    """True when the corrected block should be replaced by its raw OCR."""
    if truncated:
        return True
    n_src, n_out = len(flat(src)), len(flat(out))
    if n_src and not (0.5 * n_src <= n_out <= 2 * n_src):
        return True
    return bool(degenerate_runs(out) - degenerate_runs(src))


def paired(raw, new, rng):
    d = np.asarray(new, float) - np.asarray(raw, float)
    idx = rng.integers(0, len(d), size=(RESAMPLES, len(d)))
    means = d[idx].mean(1)
    out = {"mean_raw": float(np.mean(raw)), "mean_corrected": float(np.mean(new)),
           "delta": float(d.mean()), "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
           "better": int((d < 0).sum()), "worse": int((d > 0).sum()), "same": int((d == 0).sum()), "pages": len(d)}
    try:
        from scipy.stats import wilcoxon

        out["wilcoxon_p"] = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    except Exception as e:  # scipy missing, or all-zero differences
        out["wilcoxon_p"] = None
        out["wilcoxon_note"] = f"{type(e).__name__}: {e}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--weights", type=Path, default=None, help="folder with the fine-tuned weights (train out-dir/best)")
    ap.add_argument("--ocr-region", type=Path, nargs="+", required=True, help="region OCR files holding the split's pages")
    ap.add_argument("--split-dir", type=Path, required=True)
    ap.add_argument("--split", default="test_bl", help="name of the page list in --split-dir (val, test_bl)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-bytes", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="only the first N pages (smoke test)")
    ap.add_argument("--no-clean", action="store_true", help="skip the clean-text (reference through the corrector) check")
    ap.add_argument("--tiny", action="store_true", help="random tiny model (local smoke test; no weights)")
    args = ap.parse_args()
    if not args.tiny and args.weights is None:
        ap.error("--weights is required unless --tiny")

    import torch

    ids = read_ids(args.split_dir / f"{args.split}.txt")
    recs = {}
    for p in args.ocr_region:
        for r in load_jsonl(p):
            recs[r["page_id"]] = r
    missing = [i for i in ids if i not in recs]
    pages = [recs[i] for i in sorted(ids) if i in recs]
    if args.limit:
        pages = pages[:args.limit]
    assert pages, "no pages of the split found in the OCR files"
    print(f"{args.model} on {args.split}: {len(pages)} pages, missing {len(missing)}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok, model = load_model(args.model, args.tiny, path=None if args.tiny else args.weights)
    model.to(device).eval()

    t0 = time.time()
    per_page = []
    for rec in pages:
        row = {"page_id": rec["page_id"], "page_type": rec.get("page_type", ""), "ground_truth": rec["ground_truth"],
               "blocks": len(rec["regions"])}
        for eng in ENGINES:
            src = [reg[eng] for reg in rec["regions"]]
            outs, trunc = correct_texts(model, tok, src, device, args.max_bytes, args.batch)
            fallback = [guard(s, o, t) for s, o, t in zip(src, outs, trunc)]
            kept = [s if f else o for s, o, f in zip(src, outs, fallback)]
            raw_region = "\n".join(src)
            texts = {"raw": raw_region, "plain": "\n".join(outs), "guarded": "\n".join(kept)}
            row[eng] = {
                "raw_toplevel_cer": cer(rec["ground_truth"], rec[eng]) if eng in rec else None,
                "truncated_blocks": int(sum(trunc)), "fallback_blocks": int(sum(fallback)),
                "degenerate_blocks": int(sum(bool(degenerate_runs(o) - degenerate_runs(s)) for s, o in zip(src, outs))),
                "changed_blocks": int(sum(flat(s) != flat(o) for s, o in zip(src, outs))),
                "text": texts,
            }
            refs = [reg["ground_truth"] for reg in rec["regions"]]
            per_block = {"raw": src, "plain": outs, "guarded": kept}
            for name, t in texts.items():
                row[eng][name + "_cer"] = cer(rec["ground_truth"], t)
                row[eng][name + "_wer"] = wer(rec["ground_truth"], t)
                # main test metric (D78): per-block, character-weighted within the page
                bs = block_scores(refs, per_block[name])
                row[eng][name + "_blockcer"] = bs["cer"]
                row[eng][name + "_blockwer"] = bs["wer"]
                row[eng]["scored_blocks"] = bs["blocks"]
        if not args.no_clean:
            refs = [reg["ground_truth"] for reg in rec["regions"] if flat(reg.get("ground_truth", ""))]
            outs, trunc = correct_texts(model, tok, refs, device, args.max_bytes, args.batch)
            row["clean"] = {"blocks": len(refs), "changed": int(sum(flat(r) != flat(o) for r, o in zip(refs, outs))),
                            "truncated": int(sum(trunc)),
                            "cer_vs_reference": float(np.mean([cer(r, o) for r, o in zip(refs, outs)])) if refs else 0.0}
        per_page.append(row)
        print(f"  {rec['page_id']}: " + ", ".join(
            f"{e} raw {row[e]['raw_blockcer']:.3f} -> {row[e]['plain_blockcer']:.3f}" for e in ENGINES) + " (per-block CER)", flush=True)

    rng = np.random.default_rng(SEED)
    summary = {"model": args.model, "weights": str(args.weights), "split": args.split, "pages": len(pages),
               "missing_pages": missing, "seconds": round(time.time() - t0), "device": device,
               "engines": {}}
    for eng in ENGINES:
        e = {"raw_toplevel_cer_mean": float(np.mean([r[eng]["raw_toplevel_cer"] for r in per_page
                                                    if r[eng]["raw_toplevel_cer"] is not None] or [np.nan])),
             "truncated_blocks": sum(r[eng]["truncated_blocks"] for r in per_page),
             "degenerate_blocks": sum(r[eng]["degenerate_blocks"] for r in per_page),
             "fallback_blocks": sum(r[eng]["fallback_blocks"] for r in per_page),
             "changed_blocks": sum(r[eng]["changed_blocks"] for r in per_page),
             "blocks": sum(r["blocks"] for r in per_page)}
        for variant in ("plain", "guarded"):
            for metric in ("blockcer", "blockwer", "cer", "wer"):
                e[f"{variant}_{metric}"] = paired([r[eng][f"raw_{metric}"] for r in per_page],
                                                  [r[eng][f"{variant}_{metric}"] for r in per_page], rng)
        # layout breakdown (D78): table-of-contents pages against the others, per-block metric
        e["by_layout"] = {}
        for label, keep in (("toc", lambda r: r.get("page_type") == "toc"), ("other", lambda r: r.get("page_type") != "toc")):
            sel = [r for r in per_page if keep(r)]
            if len(sel) >= 2:
                e["by_layout"][label] = {f"{v}_blockcer": paired([r[eng]["raw_blockcer"] for r in sel],
                                                                 [r[eng][f"{v}_blockcer"] for r in sel], rng)
                                         for v in ("plain", "guarded")}
        summary["engines"][eng] = e
    if not args.no_clean:
        c = [r["clean"] for r in per_page]
        summary["clean_text"] = {"blocks": sum(x["blocks"] for x in c), "changed": sum(x["changed"] for x in c),
                                 "truncated": sum(x["truncated"] for x in c),
                                 "mean_cer_vs_reference": float(np.mean([x["cer_vs_reference"] for x in c]))}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    with (args.out_dir / "pages.jsonl").open("w", encoding="utf-8") as f:
        for r in per_page:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    lines = [f"{args.model} on {args.split} ({len(pages)} pages, {device}, {summary['seconds']} s)"]
    for eng, e in summary["engines"].items():
        lines.append(f"{eng}: top-level raw CER {e['raw_toplevel_cer_mean']:.4f}; blocks {e['blocks']}, changed "
                     f"{e['changed_blocks']}, truncated {e['truncated_blocks']}, degenerate {e['degenerate_blocks']}")
        for variant in ("plain", "guarded"):
            for metric in ("blockcer", "blockwer", "cer", "wer"):
                s = e[f"{variant}_{metric}"]
                p = f"{s['wilcoxon_p']:.4f}" if s.get("wilcoxon_p") is not None else "n/a"
                lines.append(f"  {variant:7s} {metric.upper()}{' (main, per-block)' if metric == 'blockcer' else ''} {s['mean_raw']:.4f} -> {s['mean_corrected']:.4f}  "
                             f"delta {s['delta']:+.4f} [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}]  "
                             f"better {s['better']} worse {s['worse']}  Wilcoxon p {p}")
        for label, d in e["by_layout"].items():
            for variant, s in d.items():
                lines.append(f"  layout {label:5s} {variant}: {s['mean_raw']:.4f} -> {s['mean_corrected']:.4f} "
                             f"delta {s['delta']:+.4f} [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}] ({s['pages']} pages)")
    if "clean_text" in summary:
        c = summary["clean_text"]
        lines.append(f"clean text: {c['changed']}/{c['blocks']} blocks changed, mean CER vs reference "
                     f"{c['mean_cer_vs_reference']:.4f}, truncated {c['truncated']}")
    (args.out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
