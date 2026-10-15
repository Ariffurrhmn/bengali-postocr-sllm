"""Analyses for the third review (Draft 03), run on existing outputs; no new
model runs. Item codes refer to context/review-draft-03.md.

  E1  bounded word match error rate, wMER = (S+D+I)/(H+S+D+I) over words,
      per page; paired bootstrap and Wilcoxon (exploratory, own Holm family)
  E2  equivalence (TOST, paired t) of the CER change against margins 0.01
      and 0.02, with the 90% bootstrap interval; pages improved / worse /
      unchanged for every method
  C8  number of non-zero page differences behind each Wilcoxon test (scipy's
      default zero_method "wilcox" drops zero differences), for the main
      CER tests and for confidence gating
  E3  ASCII digits in the outputs; CER with digits normalised (ASCII digits
      mapped to Bengali digits on both sides); CER without ZWJ/ZWNJ; CER with
      khanda-ta normalised (TA + virama + ZWJ -> U+09CE); per-page share of
      reference words missing from the SymSpell lexicon against the
      dictionary's damage on the clean-text control (Spearman)
  E4  clean-text control: which pages dominate a method's pooled CER
  E5  reference-word recall of the raw OCR with bootstrap intervals, next to
      grapheme-cluster CER (the reviewer flagged that both read 0.431)
  C1  empty outputs per model, with the generation time of empty and
      non-empty pages

Usage (from eval/):
    python review3_analysis.py --ocr-path ../results/ocr_eval40.jsonl \\
        --correction-path ../results/eval40_all.jsonl \\
        --gpu-runs ../results/gpu_eval40.jsonl --conf-path ../results/ocr_conf40.jsonl \\
        --out-prefix ../results/review3_gpu40
"""
import argparse
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import jiwer
import numpy as np
from scipy import stats

# extended_analysis rewraps sys.stdout for UTF-8 on import.
from extended_analysis import (DEFAULT_RESAMPLES, DEFAULT_SEED, MODEL_ORDER, bootstrap_ci,
                               correct_ref_words, holm, wilcoxon_p)
from confidence_gate import THRESHOLDS, gate, word_confidences
from metrics import cer, normalize
from revision_analysis import grapheme_cer
from sensitivity_analysis import word_recall

REPO_ROOT = Path(__file__).resolve().parent.parent
LEXICON = REPO_ROOT / "data" / "lexicon" / "bn_50k.txt"
ENGINES = ["tesseract", "easyocr"]
MARGINS = [0.01, 0.02]

ASCII_TO_BN = str.maketrans("0123456789", "".join(chr(0x09E6 + i) for i in range(10)))
ZW = dict.fromkeys(map(ord, "‌‍"))
KHANDA_TA = ("ত্‍", "ৎ")
# The dictionary baseline's word pattern (correction/dictionary_baseline.py: WORD):
# Bengali letters and signs, no digits, plus ZWNJ/ZWJ.
SYMSPELL_WORD = re.compile("[ঀ-৥ৰ-৿‌‍]+")


def load(path: Path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def wmer(reference: str, hypothesis: str) -> float:
    ref, hyp = normalize(reference), normalize(hypothesis)
    if not ref or not hyp:
        return 0.0 if ref == hyp else 1.0
    o = jiwer.process_words(ref, hyp)
    errors = o.substitutions + o.deletions + o.insertions
    return errors / (o.hits + errors)


def tost_p(d: np.ndarray, margin: float) -> float:
    """Paired TOST: the larger of the two one-sided t-test p-values for
    H0: mean <= -margin and H0: mean >= +margin."""
    n, m, s = len(d), d.mean(), d.std(ddof=1)
    if s == 0:
        return 0.0 if abs(m) < margin else 1.0
    se = s / np.sqrt(n)
    p_low = 1 - stats.t.cdf((m + margin) / se, n - 1)
    p_high = stats.t.cdf((m - margin) / se, n - 1)
    return float(max(p_low, p_high))


def khanda(t: str) -> str:
    return unicodedata.normalize("NFC", t).replace(*KHANDA_TA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ocr-path", type=Path, required=True)
    parser.add_argument("--correction-path", type=Path, required=True)
    parser.add_argument("--gpu-runs", type=Path, required=True)
    parser.add_argument("--conf-path", type=Path, required=True)
    parser.add_argument("--out-prefix", type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(DEFAULT_SEED)

    pages = {r["page_id"]: r for r in load(args.ocr_path)}
    pids = sorted(pages)
    gt = {p: pages[p]["ground_truth"] for p in pids}
    out = defaultdict(dict)
    for r in load(args.correction_path):
        out[(r["engine"], r["model_key"])][r["page_id"]] = r["raw_output"]
    models = [m for m in MODEL_ORDER if all((e, m) in out for e in ENGINES)]
    res = {"n_pages": len(pids), "models": models}
    lines = []
    p = lines.append

    def deltas(metric, e, m, src=None):
        src = src or (lambda q: out[(e, m)][q])
        return np.array([metric(gt[q], src(q)) - metric(gt[q], pages[q][e]) for q in pids])

    # ---- E1: wMER ----
    p("E1: word match error rate (wMER), per-page mean; change vs raw OCR, "
      "paired bootstrap 95% CI, Wilcoxon, Holm over the tests below (exploratory)")
    res["wmer_baseline"] = {e: float(np.mean([wmer(gt[q], pages[q][e]) for q in pids])) for e in ENGINES}
    tests = []
    for e in ENGINES:
        for m in models:
            d = deltas(wmer, e, m)
            lo, hi = bootstrap_ci(d, rng, DEFAULT_RESAMPLES)
            tests.append({"engine": e, "model": m, "baseline": res["wmer_baseline"][e],
                          "corrected": float(res["wmer_baseline"][e] + d.mean()),
                          "mean_delta": float(d.mean()), "ci_low": lo, "ci_high": hi,
                          "wilcoxon_p": wilcoxon_p(d), "nonzero": int(np.sum(d != 0))})
    for t, a in zip(tests, holm([t["wilcoxon_p"] for t in tests])):
        t["wilcoxon_p_holm"] = a
    clean = {}
    for m in models:
        if ("ground_truth", m) in out:
            clean[m] = {"wer": float(np.mean([jiwer.wer(normalize(gt[q]), normalize(out[("ground_truth", m)][q]))
                                              if normalize(out[("ground_truth", m)][q]) else 1.0 for q in pids])),
                        "wmer": float(np.mean([wmer(gt[q], out[("ground_truth", m)][q]) for q in pids]))}
    res["wmer_tests"], res["clean_word_rates"] = tests, clean
    p(f"baseline wMER: " + ", ".join(f"{e} {v:.3f}" for e, v in res["wmer_baseline"].items()))
    for t in tests:
        p(f"  {t['engine']:<10}{t['model']:<22}{t['corrected']:.3f}  {t['mean_delta']:+.3f} "
          f"[{t['ci_low']:+.3f}, {t['ci_high']:+.3f}]  p_Holm {t['wilcoxon_p_holm']:.4f}  nonzero {t['nonzero']}")
    p("clean-text control, mean WER / wMER against the reference:")
    for m, v in clean.items():
        p(f"  {m:<22}WER {v['wer']:.3f}  wMER {v['wmer']:.3f}")

    # ---- E2 + C8: equivalence, page counts, non-zero pairs (CER) ----
    p("\nE2/C8: CER change vs raw OCR; pages better/worse/same; non-zero pairs used by Wilcoxon; "
      "TOST p (paired t) and 90% bootstrap CI against margins " + ", ".join(map(str, MARGINS)))
    eq = []
    for e in ENGINES:
        for m in models:
            d = deltas(cer, e, m)
            lo90, hi90 = np.percentile(d[rng.integers(0, len(d), (DEFAULT_RESAMPLES, len(d)))].mean(axis=1),
                                       [5, 95])
            row = {"engine": e, "model": m, "mean_delta": float(d.mean()),
                   "ci90": [float(lo90), float(hi90)],
                   "better": int(np.sum(d < 0)), "worse": int(np.sum(d > 0)), "same": int(np.sum(d == 0)),
                   "nonzero": int(np.sum(d != 0)),
                   "baseline_mean": float(np.mean([cer(gt[q], pages[q][e]) for q in pids]))}
            row["relative_change"] = row["mean_delta"] / row["baseline_mean"]
            for mg in MARGINS:
                row[f"tost_p_{mg}"] = tost_p(d, mg)
                row[f"equivalent_{mg}"] = bool(row[f"tost_p_{mg}"] < 0.05)
            eq.append(row)
            p(f"  {e:<10}{m:<22}{row['mean_delta']:+.4f} ({row['relative_change']:+.1%})  90% CI "
              f"[{lo90:+.4f}, {hi90:+.4f}]  better/worse/same {row['better']}/{row['worse']}/{row['same']}  "
              + "  ".join(f"TOST{mg} p={row[f'tost_p_{mg}']:.3f}" for mg in MARGINS))
    res["equivalence"] = eq

    # ---- C8: confidence gating, non-zero pairs ----
    p("\nC8: confidence gating, CER change vs raw OCR; non-zero pairs; Wilcoxon p (unadjusted)")
    conf = {r["page_id"]: r for r in load(args.conf_path)}
    gated = []
    for e in ENGINES:
        wc = {}
        for q in pids:
            units = ([(w["text"], w["conf"]) for w in conf[q]["tesseract_words"]] if e == "tesseract"
                     else [(b["text"], b["conf"]) for b in conf[q]["easyocr_boxes"]])
            wc[q] = word_confidences(pages[q][e], units)
        res[f"{e}_conf_pages_reproduced"] = int(sum(conf[q][f"{e}_matches"] for q in pids))
        res[f"{e}_conf_pages_not_reproduced"] = [q for q in pids if not conf[q][f"{e}_matches"]]
        for m in models:
            for t in THRESHOLDS[e]:
                d = deltas(cer, e, m, lambda q: gate(pages[q][e], out[(e, m)][q], wc[q], t))
                row = {"engine": e, "model": m, "threshold": t, "mean_delta": float(d.mean()),
                       "better": int(np.sum(d < 0)), "worse": int(np.sum(d > 0)),
                       "same": int(np.sum(d == 0)), "nonzero": int(np.sum(d != 0)),
                       "wilcoxon_p": wilcoxon_p(d)}
                gated.append(row)
                if m in ("qwen3-1.7b", "gemma-2b"):
                    p(f"  {e:<10}{m:<14}{t:>5}  {row['mean_delta']:+.4f}  better/worse/same "
                      f"{row['better']}/{row['worse']}/{row['same']}  p {row['wilcoxon_p']:.4f}")
        if res[f"{e}_conf_pages_not_reproduced"]:
            p(f"  {e}: confidence run did not reproduce the stored text on "
              + ", ".join(res[f"{e}_conf_pages_not_reproduced"]))
    res["gated_pairs"] = gated

    # ---- E3: digits, ZWJ/ZWNJ, khanda-ta ----
    p("\nE3: ASCII digits per output (total over pages); CER under three normalisations "
      "(per-page mean): digits mapped to Bengali, ZWJ/ZWNJ removed, khanda-ta unified")
    norms = {"plain": lambda t: t,
             "digits": lambda t: t.translate(ASCII_TO_BN),
             "no_zw": lambda t: unicodedata.normalize("NFC", t).translate(ZW),
             "khanda": khanda}
    count_ascii = lambda t: sum(c in "0123456789" for c in t)
    count_bn = lambda t: sum(0x09E6 <= ord(c) <= 0x09EF for c in t)
    e3 = []
    res["reference_digits"] = {"ascii": sum(count_ascii(gt[q]) for q in pids),
                               "bengali": sum(count_bn(gt[q]) for q in pids)}
    res["reference_zw"] = sum(gt[q].count("‌") + gt[q].count("‍") for q in pids)
    res["reference_khanda_ta"] = {"u09ce": sum(gt[q].count("ৎ") for q in pids),
                                  "ta_virama_zwj": sum(unicodedata.normalize("NFC", gt[q]).count(KHANDA_TA[0])
                                                       for q in pids)}
    p(f"reference: digits {res['reference_digits']}, ZWJ/ZWNJ {res['reference_zw']}, "
      f"khanda-ta {res['reference_khanda_ta']}")
    for e in ENGINES:
        for m in [None] + models:
            src = (lambda q: pages[q][e]) if m is None else (lambda q, m=m: out[(e, m)][q])
            row = {"engine": e, "model": m or "(raw OCR)",
                   "ascii_digits": sum(count_ascii(src(q)) for q in pids),
                   "bengali_digits": sum(count_bn(src(q)) for q in pids)}
            for k, f in norms.items():
                row[f"cer_{k}"] = float(np.mean([cer(f(gt[q]), f(src(q))) for q in pids]))
            e3.append(row)
            p(f"  {e:<10}{row['model']:<22}ASCII {row['ascii_digits']:>4} BN {row['bengali_digits']:>4}  "
              + "  ".join(f"{k} {row[f'cer_{k}']:.4f}" for k in norms))
    res["normalisation"] = e3

    # E3: lexicon coverage vs dictionary damage
    # Same lexicon filter and word pattern as the baseline itself.
    lex = set()
    for line in LEXICON.read_text(encoding="utf-8").splitlines():
        parts = line.rsplit(" ", 1)
        if len(parts) == 2 and parts[1].isdigit():
            w = unicodedata.normalize("NFC", parts[0])
            if SYMSPELL_WORD.fullmatch(w):
                lex.add(w)
    oov, dmg_clean, dmg = [], [], {e: [] for e in ENGINES}
    total_words = total_oov = 0
    for q in pids:
        words = SYMSPELL_WORD.findall(unicodedata.normalize("NFC", gt[q]))
        n_oov = sum(w not in lex for w in words)
        total_words += len(words)
        total_oov += n_oov
        oov.append(n_oov / len(words) if words else 0.0)
        if ("ground_truth", "symspell") in out:
            dmg_clean.append(cer(gt[q], out[("ground_truth", "symspell")][q]))
        for e in ENGINES:
            ok = correct_ref_words(gt[q], pages[q][e])
            corr = correct_ref_words(gt[q], out[(e, "symspell")][q])
            dmg[e].append(float((ok & ~corr).sum() / ok.sum()) if ok.sum() else 0.0)
    rho_clean = stats.spearmanr(oov, dmg_clean) if dmg_clean else None
    res["lexicon"] = {"size": len(lex), "reference_bengali_words": total_words,
                      "reference_words_missing": total_oov,
                      "share_missing": total_oov / total_words,
                      "page_share_missing_mean": float(np.mean(oov)),
                      "spearman_oov_vs_clean_cer": [float(rho_clean.statistic), float(rho_clean.pvalue)]
                      if rho_clean else None,
                      "spearman_oov_vs_broken": {e: [float(stats.spearmanr(oov, dmg[e]).statistic),
                                                     float(stats.spearmanr(oov, dmg[e]).pvalue)]
                                                 for e in ENGINES}}
    L = res["lexicon"]
    p(f"\nE3/M4: {L['reference_words_missing']}/{L['reference_bengali_words']} reference Bengali word "
      f"tokens ({L['share_missing']:.1%}) are missing from the {L['size']}-word lexicon; per-page mean "
      f"{L['page_share_missing_mean']:.1%}. Spearman (per-page missing share vs SymSpell clean-text CER): "
      f"{L['spearman_oov_vs_clean_cer']}; vs share of correct OCR words broken: {L['spearman_oov_vs_broken']}")

    # ---- E4: which pages dominate the clean-text pooled CER ----
    p("\nE4: clean-text control, mean vs pooled CER, and the pages with the most character errors")
    e4 = []
    for m in models:
        if ("ground_truth", m) not in out:
            continue
        errs, lens = [], []
        for q in pids:
            ref, hyp = normalize(gt[q]), normalize(out[("ground_truth", m)][q])
            errs.append(cer(gt[q], out[("ground_truth", m)][q]) * len(ref))
            lens.append(len(ref))
        errs, lens = np.array(errs), np.array(lens)
        order = np.argsort(-errs)
        top = [{"page": pids[i], "ref_len": int(lens[i]), "cer": float(errs[i] / lens[i]),
                "share_of_errors": float(errs[i] / errs.sum()),
                "out_len_ratio": len(normalize(out[("ground_truth", m)][pids[i]])) / lens[i]}
               for i in order[:3]]
        row = {"model": m, "mean_cer": float(np.mean(errs / lens)), "pooled_cer": float(errs.sum() / lens.sum()),
               "spearman_len_cer": float(stats.spearmanr(lens, errs / lens).statistic), "top": top}
        e4.append(row)
        p(f"  {m:<22}mean {row['mean_cer']:.3f} pooled {row['pooled_cer']:.3f}  rho(length, CER) "
          f"{row['spearman_len_cer']:+.2f}  top pages: "
          + "; ".join(f"{t['page']} len {t['ref_len']} CER {t['cer']:.2f} out/ref {t['out_len_ratio']:.2f} "
                      f"({t['share_of_errors']:.0%} of errors)" for t in top))
    res["clean_pooled"] = e4

    # ---- E5: word recall CI and grapheme CER of raw OCR ----
    p("\nE5: raw OCR, reference-word recall and grapheme-cluster CER (per-page means, 95% bootstrap CI)")
    e5 = {}
    for e in ENGINES:
        rec = np.array([word_recall(gt[q], pages[q][e]) for q in pids])
        gc = np.array([grapheme_cer(gt[q], pages[q][e]) for q in pids])
        e5[e] = {"recall": float(rec.mean()), "recall_ci": bootstrap_ci(rec, rng, DEFAULT_RESAMPLES),
                 "grapheme_cer": float(gc.mean()), "grapheme_cer_ci": bootstrap_ci(gc, rng, DEFAULT_RESAMPLES),
                 "pages_equal": int(np.sum(np.isclose(rec, gc)))}
        p(f"  {e:<10}recall {rec.mean():.5f} {e5[e]['recall_ci']}  grapheme CER {gc.mean():.5f} "
          f"{e5[e]['grapheme_cer_ci']}  pages where the two are equal: {e5[e]['pages_equal']}")
    res["recall_vs_grapheme"] = e5

    # ---- C1: empty outputs and their generation time ----
    p("\nC1: empty outputs (whitespace only) per model and input, with generation time per page")
    c1 = []
    runs = load(args.gpu_runs)
    for m in MODEL_ORDER:
        rows = [r for r in runs if r["model_key"] == m]
        if not rows:
            continue
        emp = [r for r in rows if not r["raw_output"].strip()]
        full = [r for r in rows if r["raw_output"].strip()]
        row = {"model": m, "n": len(rows), "empty": len(emp),
               "empty_by_input": {e: sum(r["engine"] == e for r in emp) for e in ENGINES + ["ground_truth"]},
               "empty_sec_median": float(np.median([r["elapsed_sec"] for r in emp])) if emp else None,
               "empty_sec_max": float(max(r["elapsed_sec"] for r in emp)) if emp else None,
               "nonempty_sec_median": float(np.median([r["elapsed_sec"] for r in full])),
               "outputs_with_exclamation_runs": sum(bool(re.search(r"!{5,}", r["raw_output"])) for r in rows)}
        c1.append(row)
        p(f"  {m:<22}empty {row['empty']}/{row['n']} {row['empty_by_input']}  time empty median "
          f"{row['empty_sec_median']} max {row['empty_sec_max']}  non-empty median {row['nonempty_sec_median']:.1f}"
          f"  '!!!!!' runs {row['outputs_with_exclamation_runs']}")
    res["empty_outputs"] = c1

    text = "\n".join(lines)
    print(text)
    args.out_prefix.with_suffix(".txt").write_text(text + "\n", encoding="utf-8")
    args.out_prefix.with_suffix(".json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"\nWrote {args.out_prefix}.txt and .json")


if __name__ == "__main__":
    main()
