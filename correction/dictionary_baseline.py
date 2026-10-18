"""Classical baseline for the revision (review item C4): dictionary-based
spelling correction with SymSpell, no language model.

Every Bengali word in the OCR text that is not in the lexicon is replaced by
the lexicon word at the smallest edit distance (ties broken by frequency), if
one exists within max_edit_distance; everything else (punctuation, digits,
Latin text, whitespace, known words) is left as it is. Settings are SymSpell's
defaults, fixed in advance: max_edit_distance 2, prefix_length 7.

Lexicon: FrequencyWords bn_50k.txt (hermitdave/FrequencyWords, 2018 release,
OpenSubtitles-derived, CC BY-SA 4.0), saved as data/lexicon/bn_50k.txt
(not committed). sha256 2fd6c08f29335f26af983c26821537e5ed5e42dfdf43275a0c646a15d18d636e.
Only entries made of Bengali letters and signs are used.

Output records follow run_sweep.py's format (model_key "symspell"), so the
evaluation scripts read them unchanged. "ground_truth" as an engine gives the
clean-text control.

Usage:
    python correction/dictionary_baseline.py --ocr-path results/ocr_eval40.jsonl \\
        --engines tesseract,easyocr,ground_truth --out results/dict_eval40.jsonl
"""
import argparse
import io
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from symspellpy import SymSpell, Verbosity  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
LEXICON = REPO_ROOT / "data" / "lexicon" / "bn_50k.txt"
MAX_EDIT_DISTANCE = 2
PREFIX_LENGTH = 7

# A Bengali word: letters, vowel signs, virama and other signs of the Bengali
# block (U+0980-09FF) without its digits (U+09E6-09EF), plus ZWNJ/ZWJ. The
# danda (U+0964-5) lies outside the block, so it is never part of a word.
WORD = re.compile(r"[ঀ-৥ৰ-৿‌‍]+")


def load_symspell(path: Path, max_edit: int = MAX_EDIT_DISTANCE) -> SymSpell:
    sym = SymSpell(max_dictionary_edit_distance=max_edit, prefix_length=PREFIX_LENGTH)
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.rsplit(" ", 1)
        if len(parts) != 2 or not parts[1].isdigit():
            continue
        word = unicodedata.normalize("NFC", parts[0])
        if WORD.fullmatch(word):
            sym.create_dictionary_entry(word, int(parts[1]))
    return sym


def correct(text: str, sym: SymSpell, max_edit: int = MAX_EDIT_DISTANCE) -> tuple[str, int]:
    text = unicodedata.normalize("NFC", text)
    changed = 0

    def fix(m):
        nonlocal changed
        w = m.group(0)
        if w in sym.words:
            return w
        hits = sym.lookup(w, Verbosity.TOP, max_edit_distance=max_edit)
        if hits and hits[0].term != w:
            changed += 1
            return hits[0].term
        return w

    return WORD.sub(fix, text), changed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ocr-path", type=Path, required=True)
    parser.add_argument("--engines", default="tesseract,easyocr,ground_truth")
    parser.add_argument("--out", type=Path, required=True)
    # Third review (M4): another lexicon (data/build_wiki_lexicon.py) or edit distance 1.
    parser.add_argument("--lexicon", type=Path, default=LEXICON)
    parser.add_argument("--max-edit-distance", type=int, default=MAX_EDIT_DISTANCE)
    parser.add_argument("--model-key", default="symspell")
    args = parser.parse_args()

    sym = load_symspell(args.lexicon, args.max_edit_distance)
    print(f"Lexicon: {len(sym.words)} Bengali words")
    pages = [json.loads(l) for l in args.ocr_path.read_text(encoding="utf-8").split("\n") if l.strip()]
    n_changed = 0
    with args.out.open("w", encoding="utf-8") as f:
        for engine in args.engines.split(","):
            for page in pages:
                t0 = time.time()
                out, changed = correct(page[engine], sym, args.max_edit_distance)
                n_changed += changed
                f.write(json.dumps({
                    "page_id": page["page_id"], "engine": engine, "model_key": args.model_key,
                    "approach": "word", "decoding": "none", "device": "cpu",
                    "raw_output": out, "truncated": False, "skipped": False, "error": None,
                    "words_changed": changed, "elapsed_sec": round(time.time() - t0, 3),
                }, ensure_ascii=False) + "\n")
    print(f"Wrote {args.out} ({n_changed} words replaced in total)")


if __name__ == "__main__":
    main()
