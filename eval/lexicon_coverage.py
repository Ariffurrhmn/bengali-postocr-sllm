"""Share of reference Bengali word tokens missing from each SymSpell lexicon
(third review, items E3 and M4), with the dictionary baseline's own word
pattern and lexicon filter.

Usage:
    python eval/lexicon_coverage.py --ocr-path results/ocr_eval40.jsonl \\
        --lexicons data/lexicon/bn_50k.txt data/lexicon/bn_wiki100k.txt --out results/m4_lexicon_coverage.json
"""
import argparse
import json
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "correction"))
from dictionary_baseline import WORD, load_symspell  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocr-path", type=Path, required=True)
    ap.add_argument("--lexicons", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    pages = [json.loads(l) for l in args.ocr_path.read_text(encoding="utf-8").split("\n") if l.strip()]
    words = [w for p in pages for w in WORD.findall(unicodedata.normalize("NFC", p["ground_truth"]))]
    res = {"reference_bengali_words": len(words), "lexicons": {}}
    for path in args.lexicons:
        lex = set(load_symspell(path, 1).words)
        missing = sum(w not in lex for w in words)
        res["lexicons"][path.name] = {"size": len(lex), "missing": missing, "share_missing": missing / len(words)}
        print(f"{path.name}: {len(lex)} words, {missing}/{len(words)} reference tokens missing "
              f"({missing / len(words):.1%})")
    args.out.write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
