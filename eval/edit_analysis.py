"""Why do some methods lower WER without lowering CER? (external review item 7)

For each page, the character alignment of the reference with a hypothesis (raw
OCR or corrected output) assigns every character error to a reference word:
substitutions and deletions to the word of the reference character, insertions
to the word they follow (or precede, at a word start); errors on the spaces
between words go to a separate "word boundary" bucket. The word alignment of
extended_analysis.py marks each reference word as kept, fixed, broken or still
wrong. The change in character errors (corrected minus raw OCR) is then split
exactly into these groups, so the gain on fixed words can be set against the
cost on broken and still-wrong words. Errors are also classified by the
Bengali character class involved (consonant, vowel sign, virama, other sign,
independent vowel, digit, punctuation and danda, space, other), and a
whitespace-insensitive CER (all spaces removed) is reported.

Pooled over the 40 pages; character errors are counts, so they add up to the
pooled CER numerator.

Usage:
    python eval/edit_analysis.py --ocr-path results/ocr_eval40.jsonl \
        --cell gpu_eval40.jsonl:qwen3-1.7b:tesseract --cell ... \
        --out-json results/edit_analysis40.json
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import jiwer
import numpy as np

# extended_analysis rewraps sys.stdout for UTF-8 on import.
from extended_analysis import correct_ref_words
from metrics import cer, normalize

RESULTS = Path(__file__).resolve().parents[1] / "results"
GROUPS = ["fixed", "broken", "still_wrong", "kept", "boundary"]
CLASSES = ["consonant", "vowel_sign", "virama", "other_sign", "indep_vowel", "digit", "punct", "space", "other"]


def char_class(ch: str) -> str:
    o = ord(ch)
    if ch == " ":
        return "space"
    if 0x0995 <= o <= 0x09B9 or 0x09DC <= o <= 0x09DF or o == 0x09CE:
        return "consonant"
    if 0x09BE <= o <= 0x09CC or o in (0x09D7, 0x09E2, 0x09E3):
        return "vowel_sign"
    if o == 0x09CD:
        return "virama"
    if 0x0981 <= o <= 0x0983 or o == 0x09BC:
        return "other_sign"
    if 0x0985 <= o <= 0x0994 or o in (0x09E0, 0x09E1):
        return "indep_vowel"
    if 0x09E6 <= o <= 0x09EF or ch.isdigit():
        return "digit"
    if o in (0x0964, 0x0965) or (not ch.isalnum() and not ch.isspace()):
        return "punct"
    return "other"


def word_errors(ref: str, hyp: str):
    """Per reference word: Counter of error classes; plus the boundary bucket."""
    words = defaultdict(Counter)
    boundary = Counter()
    word_of = []
    w = 0
    for ch in ref:
        if ch == " ":
            word_of.append(-1)
            w += 1
        else:
            word_of.append(w)
    if not hyp:
        for i, ch in enumerate(ref):
            (boundary if word_of[i] < 0 else words[word_of[i]])[char_class(ch)] += 1
        return words, boundary
    o = jiwer.process_characters(ref, hyp)
    for c in o.alignments[0]:
        if c.type in ("substitute", "delete"):
            for i in range(c.ref_start_idx, c.ref_end_idx):
                (boundary if word_of[i] < 0 else words[word_of[i]])[char_class(ref[i])] += 1
        elif c.type == "insert":
            k = c.ref_start_idx
            owner = word_of[k - 1] if k > 0 and word_of[k - 1] >= 0 else (
                word_of[k] if k < len(ref) and word_of[k] >= 0 else -1)
            for j in range(c.hyp_start_idx, c.hyp_end_idx):
                (boundary if owner < 0 else words[owner])[char_class(hyp[j])] += 1
    return words, boundary


def nospace_cer(ref: str, hyp: str) -> float:
    r, h = normalize(ref).replace(" ", ""), normalize(hyp).replace(" ", "")
    if not r:
        return 0.0 if not h else 1.0
    return jiwer.cer(r, h) if h else 1.0


def analyse(pages: dict, rows: dict, engine: str) -> dict:
    delta = {g: Counter() for g in GROUPS}
    n_words = Counter()
    changed_still_wrong = 0
    raw_err = corr_err = ref_chars = 0
    ns_raw, ns_corr = [], []
    for pid, r in rows.items():
        gt, ocr, out = normalize(pages[pid]["ground_truth"]), normalize(pages[pid][engine]), normalize(r["raw_output"])
        ok_o, ok_c = correct_ref_words(gt, ocr), correct_ref_words(gt, out)
        we_o, b_o = word_errors(gt, ocr)
        we_c, b_c = word_errors(gt, out)
        for i in range(len(ok_o)):
            g = ("kept" if ok_o[i] and ok_c[i] else "broken" if ok_o[i] else
                 "fixed" if ok_c[i] else "still_wrong")
            n_words[g] += 1
            d = Counter(we_c.get(i, {}))
            d.subtract(we_o.get(i, {}))
            delta[g].update(d)
            if g == "still_wrong" and we_c.get(i, Counter()) != we_o.get(i, Counter()):
                changed_still_wrong += 1
        d = Counter(b_c)
        d.subtract(b_o)
        delta["boundary"].update(d)
        raw_err += sum(sum(c.values()) for c in we_o.values()) + sum(b_o.values())
        corr_err += sum(sum(c.values()) for c in we_c.values()) + sum(b_c.values())
        ref_chars += len(gt)
        ns_raw.append(nospace_cer(gt, ocr))
        ns_corr.append(nospace_cer(gt, out))
    return {
        "pages": len(rows), "ref_chars": ref_chars,
        "char_errors_raw": raw_err, "char_errors_corrected": corr_err,
        "words": dict(n_words), "still_wrong_changed": changed_still_wrong,
        "delta_by_group": {g: sum(delta[g].values()) for g in GROUPS},
        "delta_by_group_class": {g: {k: delta[g][k] for k in CLASSES if delta[g][k]} for g in GROUPS},
        "nospace_cer_raw": float(np.mean(ns_raw)), "nospace_cer_corrected": float(np.mean(ns_corr)),
    }


def load(path: Path):
    # Split on newlines only: outputs can contain U+2028/U+2029 (engineering note 12).
    return [json.loads(l) for l in path.read_text(encoding="utf-8").split("\n") if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocr-path", type=Path, required=True)
    ap.add_argument("--cell", action="append", required=True,
                    help="file:model:engine, file relative to results/ (decoding plain, main prompt)")
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()
    pages = {r["page_id"]: r for r in load(args.ocr_path)}
    out = []
    for spec in args.cell:
        f, model, engine = spec.split(":")
        # Each of these files holds one output per page, model and engine.
        rows = {r["page_id"]: r for r in load(RESULTS / f)
                if r.get("model_key") == model and r["engine"] == engine}
        res = {"file": f, "model": model, "engine": engine, **analyse(pages, rows, engine)}
        # The split is exact: the group deltas add up to the change in total errors.
        assert sum(res["delta_by_group"].values()) == res["char_errors_corrected"] - res["char_errors_raw"]
        out.append(res)
        d = res["delta_by_group"]
        print(f"{model:<16} {engine:<9} n={res['pages']}  errors {res['char_errors_raw']} -> "
              f"{res['char_errors_corrected']} ({res['char_errors_corrected'] - res['char_errors_raw']:+d}): "
              + "  ".join(f"{g} {d[g]:+d}" for g in GROUPS)
              + f"  | words {res['words']}  still-wrong changed {res['still_wrong_changed']}"
              + f"  | no-space CER {res['nospace_cer_raw']:.3f} -> {res['nospace_cer_corrected']:.3f}")
    args.out_json.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
