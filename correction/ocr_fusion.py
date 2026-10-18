"""Engine fusion (third review, item M6): combine Tesseract and EasyOCR word by
word, with no language model, as a do-no-harm alternative to correction.

EasyOCR (the engine with the lower CER) is the base text. Its words are aligned
to Tesseract's (difflib on whitespace tokens, as in eval/confidence_gate.py).
Where the two disagree over a short stretch (a "replace" block of at most
MAX_BLOCK words on each side), Tesseract's words replace EasyOCR's if their
confidence ranks higher. Confidences come from results/ocr_conf40.jsonl
(Tesseract 0-100 per word, EasyOCR 0-1 per box), so each engine's confidences
are turned into ranks within the page (0 lowest, 1 highest); a word with no
confidence counts as confident (rank 1), as in confidence gating. Longer
blocks, and words only one engine has, keep EasyOCR. Rules fixed in advance:
  fusion-conf    Tesseract's mean rank in the block is higher than EasyOCR's
  fusion-margin  higher by at least MARGIN
Output records follow run_sweep.py, one per page under engine "easyocr" and
again under "tesseract" with the same fused text, so eval/matrix_analysis.py
scores the fusion against each engine's raw OCR. No clean-text control: fusion
needs two OCR readings.

Usage:
    python correction/ocr_fusion.py --ocr-path results/ocr_eval40.jsonl \\
        --conf-path results/ocr_conf40.jsonl --out results/m6_fusion40.jsonl
"""
import argparse
import difflib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
from confidence_gate import word_confidences  # noqa: E402
from metrics import normalize  # noqa: E402

MAX_BLOCK = 3
MARGIN = 0.25
RULES = {"fusion-conf": 0.0, "fusion-margin": MARGIN}


def load(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").split("\n") if l.strip()]


def ranks(confs):
    """Rank of each confidence within the page, 0-1; missing (inf) -> 1."""
    finite = sorted(c for c in confs if c != float("inf"))
    if len(finite) < 2:
        return [1.0] * len(confs)
    pos = {}
    for i, c in enumerate(finite):
        pos.setdefault(c, []).append(i)
    return [1.0 if c == float("inf") else (sum(pos[c]) / len(pos[c])) / (len(finite) - 1) for c in confs]


def fuse(easy_text, easy_rank, tess_text, tess_rank, margin):
    easy, tess = normalize(easy_text).split(), normalize(tess_text).split()
    out, swapped = [], 0
    sm = difflib.SequenceMatcher(a=easy, b=tess, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if (tag == "replace" and i2 - i1 <= MAX_BLOCK and j2 - j1 <= MAX_BLOCK
                and sum(tess_rank[j1:j2]) / (j2 - j1) > sum(easy_rank[i1:i2]) / (i2 - i1) + margin):
            out += tess[j1:j2]
            swapped += i2 - i1
        elif tag != "insert":
            out += easy[i1:i2]
    return " ".join(out), swapped, len(easy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocr-path", type=Path, required=True)
    ap.add_argument("--conf-path", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    conf = {r["page_id"]: r for r in load(args.conf_path)}
    pages = load(args.ocr_path)
    with args.out.open("w", encoding="utf-8") as f:
        for key, margin in RULES.items():
            n_swapped = n_words = 0
            for p in pages:
                c = conf[p["page_id"]]
                er = ranks(word_confidences(p["easyocr"], [(b["text"], b["conf"]) for b in c["easyocr_boxes"]]))
                tr = ranks(word_confidences(p["tesseract"], [(w["text"], w["conf"]) for w in c["tesseract_words"]]))
                text, swapped, words = fuse(p["easyocr"], er, p["tesseract"], tr, margin)
                n_swapped, n_words = n_swapped + swapped, n_words + words
                for engine in ("easyocr", "tesseract"):
                    f.write(json.dumps({
                        "page_id": p["page_id"], "engine": engine, "model_key": key, "approach": "word",
                        "decoding": "none", "device": "cpu", "raw_output": text, "truncated": False,
                        "skipped": False, "error": None, "words_swapped": swapped,
                        "conf_matches": bool(c["easyocr_matches"] and c["tesseract_matches"]),
                    }, ensure_ascii=False) + "\n")
            print(f"{key}: {n_swapped} of {n_words} EasyOCR words replaced by Tesseract's ({n_swapped / n_words:.1%})")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
