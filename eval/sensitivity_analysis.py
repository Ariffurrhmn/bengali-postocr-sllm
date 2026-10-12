"""Sensitivity checks for the second review (2026-10-12), on existing outputs;
no new model runs.

  M2  reading order: the reference concatenates the TextRegion transcriptions
      in the order they appear in the PAGE-XML file. Many files also declare a
      ReadingOrder. Re-score every method against a reference built in the
      declared reading order (regions it lists first, by index, then any
      others in file order) and repeat the paired tests.
  M2  region and order: reference-word recall, the share of reference word
      tokens that also occur in the output (multiset match). It ignores word
      order and does not penalise extra text, so neither a different reading
      order nor OCR text from outside the annotated regions lowers it.
  E7  page type: mean change in CER by page type (title, verse, prose, list),
      from data/page_types.tsv; descriptive, the groups are small.

Usage:
    python sensitivity_analysis.py --ocr-path ../results/ocr_eval40.jsonl \\
        --correction-path ../results/eval40_ocr.jsonl \\
        --dataset-dir "G:/New folder/Competition_dataset_ImagesPAGEXML" \\
        --out-prefix ../results/sensitivity_gpu40
"""
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# Imported from one module only: extended_analysis rewraps sys.stdout for UTF-8.
from extended_analysis import (DEFAULT_RESAMPLES, DEFAULT_SEED, MODEL_ORDER, holm,
                               paired_summary)
from metrics import cer, normalize

REPO_ROOT = Path(__file__).resolve().parent.parent
ENGINES = ["tesseract", "easyocr"]
REGION_RE = re.compile(r"<TextRegion\b(.*?)>(.*?)</TextRegion>", re.DOTALL)
UNICODE_RE = re.compile(r"<Unicode>([^<]*)</Unicode>")
TYPE_ORDER = ["title", "verse", "prose", "list"]


def find_xml(dataset_dir: Path, page_id: str) -> Path:
    target = f"{page_id}.xml".lower()
    for p in dataset_dir.iterdir():
        if p.name.lower() == target:
            return p
    raise FileNotFoundError(page_id)


def region_texts(xml: str):
    """(region id, text) in file order; same text rule as ocr/run_ocr.py."""
    out = []
    for attrs, body in REGION_RE.findall(xml):
        matches = UNICODE_RE.findall(body)
        if matches:
            text = matches[-1].strip() if len(matches) == 1 else matches[0].strip()
            if text:
                out.append((re.search(r'\bid="([^"]+)"', attrs).group(1), text))
    return out


def reading_order(xml: str) -> list:
    refs = []
    for tag in re.findall(r"<RegionRefIndexed[^>]*>", xml):
        refs.append((int(re.search(r'index="(\d+)"', tag).group(1)),
                     re.search(r'regionRef="([^"]+)"', tag).group(1)))
    return [r for _, r in sorted(refs)]


def word_recall(reference: str, hypothesis: str) -> float:
    ref = Counter(normalize(reference).split())
    hyp = Counter(normalize(hypothesis).split())
    total = sum(ref.values())
    return sum(min(n, hyp[w]) for w, n in ref.items()) / total if total else 1.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ocr-path", type=Path, required=True)
    parser.add_argument("--correction-path", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--page-types", type=Path, default=REPO_ROOT / "data" / "page_types.tsv")
    parser.add_argument("--out-prefix", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    args = parser.parse_args()
    rng = np.random.default_rng(DEFAULT_SEED)

    pages = {}
    for line in args.ocr_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            pages[r["page_id"]] = r
    out_by = defaultdict(dict)
    for line in args.correction_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out_by[(r["engine"], r["model_key"])][r["page_id"]] = r["raw_output"]
    models = [m for m in MODEL_ORDER if all((e, m) in out_by for e in ENGINES)]
    pids = sorted(pages)
    res = {"n_pages": len(pids), "models": models}
    lines = []
    p = lines.append

    # ---- reading-order reference ----
    ro_gt, n_changed, n_with_ro, n_mismatch = {}, 0, 0, 0
    for pid in pids:
        xml = find_xml(args.dataset_dir, pid).read_text(encoding="utf-8")
        regions = region_texts(xml)
        file_order = "\n".join(t for _, t in regions)
        if normalize(file_order) != normalize(pages[pid]["ground_truth"]):
            n_mismatch += 1  # stored reference not rebuilt; keep it unchanged
            ro_gt[pid] = pages[pid]["ground_truth"]
            continue
        order = reading_order(xml)
        n_with_ro += bool(order)
        rank = {rid: i for i, rid in enumerate(order)}
        ranked = sorted(range(len(regions)),
                        key=lambda i: (0, rank[regions[i][0]]) if regions[i][0] in rank
                        else (1, i))
        ro_text = "\n".join(regions[i][1] for i in ranked)
        n_changed += normalize(ro_text) != normalize(file_order)
        ro_gt[pid] = ro_text
    res["reading_order"] = {"pages_with_reading_order": n_with_ro,
                            "pages_reference_changed": n_changed,
                            "pages_not_rebuilt": n_mismatch}
    p("=" * 90)
    p(f"M2 reading order: {n_with_ro} of {len(pids)} pages declare a reading order; "
      f"the reference changes on {n_changed}; {n_mismatch} not rebuilt")
    tests = []
    for e in ENGINES:
        base_file = np.mean([cer(pages[q]["ground_truth"], pages[q][e]) for q in pids])
        base_ro = np.mean([cer(ro_gt[q], pages[q][e]) for q in pids])
        p(f"  raw {e}: mean CER {base_file:.3f} (file order) vs {base_ro:.3f} (reading order)")
        res["reading_order"][f"{e}_raw_cer_file"] = float(base_file)
        res["reading_order"][f"{e}_raw_cer_ro"] = float(base_ro)
        for m in models:
            d = np.array([cer(ro_gt[q], out_by[(e, m)][q]) - cer(ro_gt[q], pages[q][e])
                          for q in pids])
            s = paired_summary(d, rng, args.resamples)
            s.update({"engine": e, "model": m,
                      "corrected_mean": float(np.mean([cer(ro_gt[q], out_by[(e, m)][q])
                                                       for q in pids]))})
            tests.append(s)
    for s, a in zip(tests, holm([t["wilcoxon_p"] for t in tests])):
        s["wilcoxon_p_holm"] = float(a)
        p(f"  {s['engine']:<10}{s['model']:<22}CER {s['corrected_mean']:.3f}  "
          f"d {s['mean_delta']:+.3f} [{s['ci_low']:+.3f}, {s['ci_high']:+.3f}]  "
          f"p Holm {a:.4f}")
    res["reading_order_tests"] = tests

    # ---- reference-word recall ----
    p("=" * 90)
    p("M2 reference-word recall (order-free, extra text not penalised); "
      "d = corrected - raw, positive = better")
    tests = []
    for e in ENGINES:
        base = [word_recall(pages[q]["ground_truth"], pages[q][e]) for q in pids]
        p(f"  raw {e}: mean recall {np.mean(base):.3f}")
        res[f"{e}_raw_word_recall"] = float(np.mean(base))
        for m in models:
            corr = [word_recall(pages[q]["ground_truth"], out_by[(e, m)][q]) for q in pids]
            s = paired_summary(np.array(corr) - np.array(base), rng, args.resamples)
            s.update({"engine": e, "model": m, "baseline_mean": float(np.mean(base)),
                      "corrected_mean": float(np.mean(corr))})
            tests.append(s)
    for s, a in zip(tests, holm([t["wilcoxon_p"] for t in tests])):
        s["wilcoxon_p_holm"] = float(a)
        p(f"  {s['engine']:<10}{s['model']:<22}recall {s['corrected_mean']:.3f}  "
          f"d {s['mean_delta']:+.3f} [{s['ci_low']:+.3f}, {s['ci_high']:+.3f}]  "
          f"p Holm {a:.4f}")
    res["word_recall_tests"] = tests

    # ---- page type ----
    types = {}
    for line in args.page_types.read_text(encoding="utf-8").splitlines()[1:]:
        if line.strip():
            pid, t = line.split("\t")[:2]
            types[pid] = t
    missing = [q for q in pids if q not in types]
    if missing:
        raise SystemExit(f"page types missing for {missing}")
    counts = Counter(types[q] for q in pids)
    res["page_type_counts"] = {t: counts[t] for t in TYPE_ORDER}
    p("=" * 90)
    p("E7 page type: mean raw CER and mean change in CER (corrected - raw) by page type; "
      + ", ".join(f"{t} {counts[t]}" for t in TYPE_ORDER))
    res["page_type"] = []
    for e in ENGINES:
        for label, outs in [("(raw OCR)", None)] + [(m, out_by[(e, m)]) for m in models]:
            row = {"engine": e, "model": label}
            for t in TYPE_ORDER:
                group = [q for q in pids if types[q] == t]
                raw = [cer(pages[q]["ground_truth"], pages[q][e]) for q in group]
                if outs is None:
                    row[t] = float(np.mean(raw))
                else:
                    row[t] = float(np.mean([cer(pages[q]["ground_truth"], outs[q]) - r
                                            for q, r in zip(group, raw)]))
            res["page_type"].append(row)
            p(f"  {e:<10}{label:<22}" + "".join(f"{t} {row[t]:+.3f}  " for t in TYPE_ORDER))

    prefix = args.out_prefix
    prefix.with_suffix(".json").write_text(json.dumps(res, indent=1, ensure_ascii=False),
                                           encoding="utf-8")
    prefix.with_suffix(".txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {prefix}.json and {prefix}.txt")


if __name__ == "__main__":
    main()
