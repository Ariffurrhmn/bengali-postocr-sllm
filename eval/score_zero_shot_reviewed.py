"""Score the zero-shot methods on the reviewed BL test units (D81). Run once, after all outputs are in.

Inputs: the reviewed units with their OCR (reviewed_region.jsonl of rev 06) and the output files of
correction/run_zero_shot_blocks.py (any number; one line per page, engine and method).

Per method and engine, as for the trained correctors (D78, D79): per-block CER (main) and WER,
character-weighted within a page, joined-page CER and WER, each as the change against raw OCR with
the paired 10,000-resample bootstrap (seed 403) and the Wilcoxon test over pages. The primary
family is block CER and block WER for every method and engine (9 x 2 x 2 = 36 tests with the
full set), Holm-corrected (D73, D81). Also: table-of-contents / other pages (exploratory), blocks
truncated, skipped and changed, and the clean-text check (engine ground_truth).

The script refuses to score when a method lacks a page or an engine, or its block ids differ from
the units file, unless --allow-incomplete is given (then the output says so on every line).

Usage:
    python eval/score_zero_shot_reviewed.py --ocr-region <reviewed_region.jsonl> \\
        --split-dir data/split_v2 --outputs results/rev08/*.jsonl --out-dir results/rev08_score
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "correction"))

from extended_analysis import holm  # noqa: E402  (also rewraps stdout for UTF-8)
from metrics import block_scores, cer, normalize, wer  # noqa: E402
from score_trained_corrector import SEED, paired  # noqa: E402

ENGINES = ("tesseract", "easyocr")
METHOD_ORDER = ["symspell", "gemma-2b", "qwen3-1.7b", "phi4-mini", "llama3.2-1b", "banglat5", "titullm-1b",
                "titullm-1b-instruct", "titullm-3b-instruct"]


def load_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").split("\n") if l.strip()]


def flat(text):
    return " ".join(normalize(text).split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocr-region", type=Path, required=True)
    ap.add_argument("--split-dir", type=Path, required=True)
    ap.add_argument("--split", default="test_bl")
    ap.add_argument("--outputs", type=Path, nargs="+", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--allow-incomplete", action="store_true")
    args = ap.parse_args()

    ids = {l.strip() for l in (args.split_dir / f"{args.split}.txt").read_text(encoding="utf-8").split("\n") if l.strip()}
    pages = sorted((r for r in load_jsonl(args.ocr_region) if r["page_id"] in ids), key=lambda r: r["page_id"])
    by_id = {p["page_id"]: p for p in pages}

    outs = {}  # (method, engine, page) -> record; a later line replaces an earlier one
    for path in args.outputs:
        for r in load_jsonl(path):
            if r.get("error") is None and r["page_id"] in by_id:
                outs[(r["model_key"], r["engine"], r["page_id"])] = r
    methods = sorted({k[0] for k in outs}, key=lambda m: (METHOD_ORDER.index(m) if m in METHOD_ORDER else 99, m))

    problems = []
    for m in methods:
        for eng in ENGINES + ("ground_truth",):
            for p in pages:
                r = outs.get((m, eng, p["page_id"]))
                if r is None:
                    problems.append(f"{m} {eng} {p['page_id']}: missing")
                elif r["block_ids"] != [g["id"] for g in p["regions"]]:
                    problems.append(f"{m} {eng} {p['page_id']}: block ids differ from the units file")
    if problems and not args.allow_incomplete:
        sys.exit("refusing to score, incomplete outputs:\n  " + "\n  ".join(problems[:40])
                 + (f"\n  ... {len(problems)} in all" if len(problems) > 40 else ""))
    incomplete = bool(problems)

    rng = np.random.default_rng(SEED)
    summary = {"split": args.split, "pages": len(pages), "blocks": sum(len(p["regions"]) for p in pages),
               "incomplete": incomplete, "problems": problems, "methods": {}}
    page_rows = []
    tests = []  # primary family: (method, engine, metric) -> Wilcoxon p
    for m in methods:
        summary["methods"][m] = {}
        for eng in ENGINES:
            rows = []
            for p in pages:
                r = outs.get((m, eng, p["page_id"]))
                if r is None or r["block_ids"] != [g["id"] for g in p["regions"]]:
                    continue
                refs = [g["ground_truth"] for g in p["regions"]]
                src = [g[eng] for g in p["regions"]]
                raw_b, new_b = block_scores(refs, src), block_scores(refs, r["outputs"])
                raw_j, new_j = "\n".join(src), "\n".join(r["outputs"])
                rows.append({"page_id": p["page_id"], "page_type": p.get("page_type", ""),
                             "raw_blockcer": raw_b["cer"], "new_blockcer": new_b["cer"],
                             "raw_blockwer": raw_b["wer"], "new_blockwer": new_b["wer"],
                             "raw_cer": cer(p["ground_truth"], raw_j), "new_cer": cer(p["ground_truth"], new_j),
                             "raw_wer": wer(p["ground_truth"], raw_j), "new_wer": wer(p["ground_truth"], new_j),
                             "truncated": sum(r["truncated"]), "skipped": sum(r["skipped"]),
                             "changed": sum(flat(s) != flat(o) for s, o in zip(src, r["outputs"])),
                             "blocks": len(src)})
            if len(rows) < 2:
                continue
            e = {"pages": len(rows), **{k: sum(x[k] for x in rows) for k in ("blocks", "truncated", "skipped", "changed")}}
            for metric in ("blockcer", "blockwer", "cer", "wer"):
                e[metric] = paired([x[f"raw_{metric}"] for x in rows], [x[f"new_{metric}"] for x in rows], rng)
            for metric in ("blockcer", "blockwer"):
                tests.append((m, eng, metric, e[metric].get("wilcoxon_p")))
            e["by_layout"] = {}
            for label, keep in (("toc", lambda x: x["page_type"] == "toc"), ("other", lambda x: x["page_type"] != "toc")):
                sel = [x for x in rows if keep(x)]
                if len(sel) >= 2:
                    e["by_layout"][label] = paired([x["raw_blockcer"] for x in sel], [x["new_blockcer"] for x in sel], rng)
            summary["methods"][m][eng] = e
            page_rows += [{"method": m, "engine": eng, **x} for x in rows]

        # clean-text check: the block references through the method
        cl = []
        for p in pages:
            r = outs.get((m, "ground_truth", p["page_id"]))
            if r is None or r["block_ids"] != [g["id"] for g in p["regions"]]:
                continue
            for g, o in zip(p["regions"], r["outputs"]):
                if flat(g["ground_truth"]):
                    cl.append((flat(g["ground_truth"]) != flat(o), cer(g["ground_truth"], o)))
        if cl:
            summary["methods"][m]["clean_text"] = {"blocks": len(cl), "changed": int(sum(c for c, _ in cl)),
                                                    "mean_cer_vs_reference": float(np.mean([x for _, x in cl]))}

    ps = [t[3] if t[3] is not None else 1.0 for t in tests]
    adj = holm(ps) if ps else []
    summary["primary_family"] = {"tests": len(tests), "correction": "Holm",
                                 "results": [{"method": m, "engine": e, "metric": k, "wilcoxon_p": p, "holm_p": float(a)}
                                             for (m, e, k, p), a in zip(tests, adj)]}
    holm_of = {(r["method"], r["engine"], r["metric"]): r["holm_p"] for r in summary["primary_family"]["results"]}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    with (args.out_dir / "pages.jsonl").open("w", encoding="utf-8") as f:
        for x in page_rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    tag = " [INCOMPLETE]" if incomplete else ""
    lines = [f"Zero-shot methods on {args.split}: {len(pages)} pages, {summary['blocks']} reviewed blocks; "
             f"Holm over {len(tests)} tests (block CER and block WER){tag}"]
    for m, d in summary["methods"].items():
        for eng in ENGINES:
            if eng not in d:
                continue
            e = d[eng]
            lines.append(f"{m} / {eng}: {e['pages']} pages, blocks {e['blocks']}, changed {e['changed']}, "
                         f"skipped {e['skipped']}, truncated {e['truncated']}{tag}")
            for metric in ("blockcer", "blockwer", "cer", "wer"):
                s = e[metric]
                p = f"{s['wilcoxon_p']:.4f}" if s.get("wilcoxon_p") is not None else "n/a"
                h = holm_of.get((m, eng, metric))
                lines.append(f"  {metric:8s} {s['mean_raw']:.4f} -> {s['mean_corrected']:.4f}  delta {s['delta']:+.4f} "
                             f"[{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}]  better {s['better']} worse {s['worse']}  "
                             f"Wilcoxon p {p}" + (f"  Holm p {h:.4f}" if h is not None else ""))
            for label, s in e["by_layout"].items():
                lines.append(f"  layout {label:5s} blockcer {s['mean_raw']:.4f} -> {s['mean_corrected']:.4f} "
                             f"delta {s['delta']:+.4f} [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}] ({s['pages']} pages)")
        if "clean_text" in d:
            c = d["clean_text"]
            lines.append(f"{m} clean text: {c['changed']}/{c['blocks']} blocks changed, mean CER vs reference "
                         f"{c['mean_cer_vs_reference']:.4f}")
    (args.out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
