"""Side-by-side qualitative comparison of ground truth, raw OCR, and corrected
output — the evidence behind *why* correction made CER worse, not just *that*
it did.

Also computes per-model diagnostics that separate the distinct failure modes:
  length ratio      - overgeneration (>1) vs. dropped content (<1)
  script mix        - fraction of Bengali vs. Latin vs. other characters, which
                      catches models drifting out of the target script
  GT-token retention- how much of the ground-truth vocabulary survives, i.e.
                      whether the model preserved real content or replaced it
"""
import argparse
import io
import json
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from metrics import normalize, score_pair

REPO_ROOT = Path(__file__).resolve().parent.parent

MODEL_ORDER = ["gemma-2b", "llama3.2-1b", "banglat5", "titullm-1b"]

BENGALI_RANGE = (0x0980, 0x09FF)


def script_profile(text: str) -> dict:
    """Fraction of non-space characters that are Bengali, Latin, or other."""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return {"bengali": 0.0, "latin": 0.0, "other": 0.0, "n": 0}
    bengali = latin = other = 0
    for c in chars:
        cp = ord(c)
        if BENGALI_RANGE[0] <= cp <= BENGALI_RANGE[1]:
            bengali += 1
        elif ("a" <= c.lower() <= "z"):
            latin += 1
        else:
            other += 1
    n = len(chars)
    return {
        "bengali": bengali / n,
        "latin": latin / n,
        "other": other / n,
        "n": n,
    }


def token_retention(reference: str, hypothesis: str) -> float:
    """Fraction of distinct ground-truth tokens that still appear in the output.
    High CER with high retention means noise added around real content; high CER
    with low retention means the content itself was replaced."""
    ref_tokens = set(normalize(reference).split())
    if not ref_tokens:
        return 0.0
    hyp_tokens = set(normalize(hypothesis).split())
    return len(ref_tokens & hyp_tokens) / len(ref_tokens)


def load(ocr_path: Path, correction_path: Path):
    ocr = {}
    for line in ocr_path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        ocr[r["page_id"]] = r
    corrections = [
        json.loads(l)
        for l in correction_path.read_text(encoding="utf-8").splitlines()
    ]
    return ocr, corrections


def clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit] + f" […+{len(text) - limit} more chars]"


def print_diagnostics(ocr, corrections):
    """Per-model aggregate diagnostics — the quantitative backing for the
    qualitative examples below."""
    stats = defaultdict(lambda: defaultdict(list))
    for r in corrections:
        gt = ocr[r["page_id"]]["ground_truth"]
        out = r["raw_output"]
        m = r["model_key"]
        if gt:
            stats[m]["len_ratio"].append(len(out) / len(gt))
        prof = script_profile(out)
        stats[m]["bengali"].append(prof["bengali"])
        stats[m]["latin"].append(prof["latin"])
        stats[m]["other"].append(prof["other"])
        stats[m]["retention"].append(token_retention(gt, out))

    # Reference rows: what the ground truth and raw OCR themselves look like.
    gt_prof = [script_profile(v["ground_truth"]) for v in ocr.values()]
    print("Reference (ground truth): "
          f"Bengali={sum(p['bengali'] for p in gt_prof) / len(gt_prof):.0%}  "
          f"Latin={sum(p['latin'] for p in gt_prof) / len(gt_prof):.0%}  "
          f"other={sum(p['other'] for p in gt_prof) / len(gt_prof):.0%}")
    for engine in ("tesseract", "easyocr"):
        prof = [script_profile(v[engine]) for v in ocr.values()]
        ret = [token_retention(v["ground_truth"], v[engine]) for v in ocr.values()]
        print(f"Reference (raw {engine:<9}): "
              f"Bengali={sum(p['bengali'] for p in prof) / len(prof):.0%}  "
              f"Latin={sum(p['latin'] for p in prof) / len(prof):.0%}  "
              f"other={sum(p['other'] for p in prof) / len(prof):.0%}  "
              f"GT-token retention={sum(ret) / len(ret):.0%}")

    print()
    header = (f"{'model':<14} {'len ratio':>10} {'Bengali':>9} {'Latin':>7} "
              f"{'other':>7} {'GT retention':>13}")
    print(header)
    print("-" * len(header))
    for m in MODEL_ORDER:
        s = stats[m]
        mean = lambda k: sum(s[k]) / len(s[k])
        print(f"{m:<14} {mean('len_ratio'):>9.2f}x {mean('bengali'):>8.0%} "
              f"{mean('latin'):>6.0%} {mean('other'):>6.0%} {mean('retention'):>12.0%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "eval"], default="eval")
    parser.add_argument("--ocr-path", type=Path, default=None)
    parser.add_argument("--correction-path", type=Path, default=None)
    parser.add_argument(
        "--engine", default="tesseract", choices=["tesseract", "easyocr"]
    )
    parser.add_argument(
        "--page", default=None,
        help="Specific page_id to show; default picks a mid-difficulty page",
    )
    parser.add_argument(
        "--chars", type=int, default=420,
        help="Characters of each text to display",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Optional path to write the full report as UTF-8 text",
    )
    args = parser.parse_args()

    ocr_path = args.ocr_path or (REPO_ROOT / "results" / f"ocr_{args.split}.jsonl")
    correction_path = args.correction_path or (
        REPO_ROOT / "results" / f"correction_{args.split}.jsonl"
    )
    ocr, corrections = load(ocr_path, correction_path)

    buf = io.StringIO()

    def emit(line=""):
        print(line)
        buf.write(line + "\n")

    emit("=" * 78)
    emit("PER-MODEL FAILURE-MODE DIAGNOSTICS (all 15 pages, both engines)")
    emit("=" * 78)
    saved = sys.stdout
    sys.stdout = buf_capture = io.StringIO()
    print_diagnostics(ocr, corrections)
    sys.stdout = saved
    emit(buf_capture.getvalue().rstrip())
    emit()

    # Choose a representative page: median baseline CER for the chosen engine,
    # so the example isn't cherry-picked from either extreme.
    if args.page:
        page_id = args.page
    else:
        ranked = sorted(
            ocr.values(),
            key=lambda r: score_pair(r["ground_truth"], r[args.engine])["cer"],
        )
        page_id = ranked[len(ranked) // 2]["page_id"]

    rec = ocr[page_id]
    base_cer = score_pair(rec["ground_truth"], rec[args.engine])["cer"]

    emit("=" * 78)
    emit(f"SIDE-BY-SIDE — page {page_id}, engine {args.engine}")
    emit(f"(median-difficulty page; baseline CER = {base_cer:.3f})")
    emit("=" * 78)
    emit()
    emit(f"--- GROUND TRUTH ({len(rec['ground_truth'])} chars) " + "-" * 24)
    emit(clip(rec["ground_truth"], args.chars))
    emit()
    emit(f"--- RAW OCR / {args.engine} (CER {base_cer:.3f}) " + "-" * 24)
    emit(clip(rec[args.engine], args.chars))
    emit()

    by_model = {
        r["model_key"]: r
        for r in corrections
        if r["page_id"] == page_id and r["engine"] == args.engine
    }
    for m in MODEL_ORDER:
        r = by_model.get(m)
        if not r:
            continue
        sc = score_pair(rec["ground_truth"], r["raw_output"])
        prof = script_profile(r["raw_output"])
        ret = token_retention(rec["ground_truth"], r["raw_output"])
        emit(f"--- {m} (CER {sc['cer']:.3f}, {'TRUNCATED' if r['truncated'] else 'complete'}, "
             f"{len(r['raw_output'])} chars, Bengali {prof['bengali']:.0%}, "
             f"GT retention {ret:.0%}) " + "-" * 8)
        emit(clip(r["raw_output"], args.chars))
        emit()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(buf.getvalue(), encoding="utf-8")
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
