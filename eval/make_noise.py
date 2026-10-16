"""Synthetic OCR noise for the noise sweep (third review, items M3/F7) and the
fine-tuning pilot (M2).

  fit    Estimate a character-level noise model from the 10 development pages
         only (never the evaluation pages): align each engine's OCR with the
         reference (Levenshtein edit operations, rapidfuzz) and count, for every
         reference character, how often it was kept, replaced (and by what) or
         deleted, and which characters were inserted. Both engines are pooled.
         Writes a small JSON of counts (no page text), committed to the repo.

  apply  Corrupt the evaluation references at target CERs (default 0.02,
         0.05, 0.10, 0.20, 0.30). Each character keeps the dev pages' relative
         error profile: its substitution and deletion rates and the insertion
         rate are multiplied by one factor per level, found by bisection so
         that the mean page CER (metrics.cer) matches the target. Characters
         never seen on the dev pages use the pooled rates. Newlines are kept,
         so chunking still follows lines. Deterministic (seed 403 and the
         level). Output: one JSON line per page with ground_truth and one
         field per level, "noise_0.02" ..., usable as engines by run_sweep.py.

Usage:
    python make_noise.py fit --dev ../results/ocr_dev.jsonl --out ../data/noise_model_dev.json
    python make_noise.py apply --model ../data/noise_model_dev.json \\
        --ocr-path ../results/ocr_eval40.jsonl --out ../results/noise40.jsonl
"""
import argparse
import json
import random
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from metrics import cer

LEVELS = [0.02, 0.05, 0.10, 0.20, 0.30]
SEED = 403


def load(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").split("\n") if l.strip()]


def prep(text: str) -> str:
    """NFC and single spaces inside lines, as metrics.normalize does, but newlines kept."""
    text = unicodedata.normalize("NFC", text)
    return "\n".join(" ".join(line.split()) for line in text.split("\n") if line.strip())


def fit(dev_path: Path, out: Path):
    from rapidfuzz.distance import Levenshtein

    kept, sub, dele, ins = Counter(), defaultdict(Counter), Counter(), Counter()
    n_ref = 0
    for page in load(dev_path):
        ref = prep(page["ground_truth"]).replace("\n", " ")
        for engine in ("tesseract", "easyocr"):
            hyp = prep(page[engine]).replace("\n", " ")
            touched = set()
            for op in Levenshtein.editops(ref, hyp):
                if op.tag == "replace":
                    sub[ref[op.src_pos]][hyp[op.dest_pos]] += 1
                    touched.add(op.src_pos)
                elif op.tag == "delete":
                    dele[ref[op.src_pos]] += 1
                    touched.add(op.src_pos)
                else:
                    ins[hyp[op.dest_pos]] += 1
            for i, c in enumerate(ref):
                if i not in touched:
                    kept[c] += 1
            n_ref += len(ref)
    model = {"source": str(dev_path.name), "engines": ["tesseract", "easyocr"], "reference_chars": n_ref,
             "kept": dict(kept), "substituted": {c: dict(v) for c, v in sub.items()},
             "deleted": dict(dele), "inserted": dict(ins)}
    out.write_text(json.dumps(model, ensure_ascii=False, indent=0), encoding="utf-8")
    n_sub, n_del, n_ins = sum(sum(v.values()) for v in sub.values()), sum(dele.values()), sum(ins.values())
    print(f"dev reference characters {n_ref}: substituted {n_sub / n_ref:.3f}, deleted {n_del / n_ref:.3f}, "
          f"inserted {n_ins / n_ref:.3f} per reference character; wrote {out}")


class Noiser:
    def __init__(self, model: dict):
        self.rates, self.subs = {}, {}
        tot_kept = sum(model["kept"].values())
        tot_sub = sum(sum(v.values()) for v in model["substituted"].values())
        tot_del = sum(model["deleted"].values())
        tot = tot_kept + tot_sub + tot_del
        self.default = (tot_sub / tot, tot_del / tot)
        self.ins_rate = sum(model["inserted"].values()) / model["reference_chars"]
        pooled = Counter()
        for v in model["substituted"].values():
            pooled.update(v)
        self.pooled_sub = (list(pooled), list(pooled.values()))
        self.ins = (list(model["inserted"]), list(model["inserted"].values()))
        for c in set(model["kept"]) | set(model["substituted"]) | set(model["deleted"]):
            s = sum(model["substituted"].get(c, {}).values())
            d = model["deleted"].get(c, 0)
            n = model["kept"].get(c, 0) + s + d
            self.rates[c] = (s / n, d / n)
            if s:
                self.subs[c] = (list(model["substituted"][c]), list(model["substituted"][c].values()))

    def corrupt(self, text: str, k: float, rng: random.Random) -> str:
        out = []
        for c in text:
            if c == "\n":
                out.append(c)
                continue
            if rng.random() < min(0.9, k * self.ins_rate):
                out.append(rng.choices(*self.ins)[0])
            ps, pd = self.rates.get(c, self.default)
            ps, pd = k * ps, k * pd
            if ps + pd > 0.95:
                ps, pd = 0.95 * ps / (ps + pd), 0.95 * pd / (ps + pd)
            u = rng.random()
            if u < ps:
                out.append(rng.choices(*self.subs.get(c, self.pooled_sub))[0])
            elif u < ps + pd:
                continue
            else:
                out.append(c)
        return "".join(out)


def apply(model_path: Path, ocr_path: Path, out: Path, levels):
    noiser = Noiser(json.loads(model_path.read_text(encoding="utf-8")))
    pages = load(ocr_path)
    refs = {p["page_id"]: prep(p["ground_truth"]) for p in pages}

    def run(level, k):
        rng = random.Random(f"{SEED}-{level}")
        return {pid: noiser.corrupt(ref, k, rng) for pid, ref in sorted(refs.items())}

    def mean_cer(noised):
        return sum(cer(refs[p], t) for p, t in noised.items()) / len(noised)

    fields, report = {}, {}
    for level in levels:
        lo, hi = 0.0, 50.0
        for _ in range(30):
            mid = (lo + hi) / 2
            if mean_cer(run(level, mid)) < level:
                lo = mid
            else:
                hi = mid
        k = (lo + hi) / 2
        fields[level] = run(level, k)
        report[level] = {"factor": k, "mean_cer": mean_cer(fields[level])}
        print(f"level {level:.2f}: factor {k:.3f}, mean CER {report[level]['mean_cer']:.4f}", flush=True)
    with out.open("w", encoding="utf-8") as f:
        for p in pages:
            rec = {"page_id": p["page_id"], "ground_truth": p["ground_truth"]}
            rec.update({f"noise_{lv:.2f}": fields[lv][p["page_id"]] for lv in levels})
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    out.with_name(out.stem + "_levels.json").write_text(
        json.dumps({f"noise_{lv:.2f}": v for lv, v in report.items()}, indent=1), encoding="utf-8")
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    f = sp.add_parser("fit")
    f.add_argument("--dev", type=Path, required=True)
    f.add_argument("--out", type=Path, required=True)
    a = sp.add_parser("apply")
    a.add_argument("--model", type=Path, required=True)
    a.add_argument("--ocr-path", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--levels", default=",".join(map(str, LEVELS)))
    args = ap.parse_args()
    if args.cmd == "fit":
        fit(args.dev, args.out)
    else:
        apply(args.model, args.ocr_path, args.out, [float(x) for x in args.levels.split(",")])


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
