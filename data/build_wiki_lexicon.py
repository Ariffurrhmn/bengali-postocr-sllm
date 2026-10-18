"""Stronger lexicon for the dictionary baseline (third review, item M4): word
frequencies from Bengali Wikipedia (wikimedia/wikipedia, 20231101.bn, streamed),
replacing the modern, colloquial OpenSubtitles list as a test of whether the
lexicon explains SymSpell's damage.

Words are NFC-normalised runs of Bengali letters and signs, cut out with the
same pattern as correction/dictionary_baseline.py. The most frequent --top
words that occur at least --min-count times are written as "word count"
lines, the format of the FrequencyWords list. Settings fixed in advance.

Usage (from the repository root; streams about 150k articles):
    python data/build_wiki_lexicon.py --out data/lexicon/bn_wiki100k.txt
"""
import argparse
import hashlib
import sys
import unicodedata
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "correction"))
from dictionary_baseline import WORD  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--top", type=int, default=100_000)
    ap.add_argument("--min-count", type=int, default=5)
    args = ap.parse_args()

    from datasets import load_dataset

    counts, n_art = Counter(), 0
    for art in load_dataset("wikimedia/wikipedia", "20231101.bn", split="train", streaming=True):
        counts.update(WORD.findall(unicodedata.normalize("NFC", art["text"])))
        n_art += 1
        if n_art % 20_000 == 0:
            print(f"{n_art} articles, {len(counts)} distinct words", flush=True)
    kept = [(w, c) for w, c in counts.most_common(args.top) if c >= args.min_count]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(f"{w} {c}\n" for w, c in kept), encoding="utf-8")
    sha = hashlib.sha256(args.out.read_bytes()).hexdigest()
    print(f"{n_art} articles, {sum(counts.values())} word tokens, {len(counts)} distinct; "
          f"kept {len(kept)} (lowest count {kept[-1][1]}); sha256 {sha}")


if __name__ == "__main__":
    main()
