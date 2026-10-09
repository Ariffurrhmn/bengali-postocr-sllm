# Open items

## Post-submission feedback (Sep 2026)

Reviewed 24 Aug 2026; verdict "not acceptable at current stage". All 50 items are transcribed in `reviewer-feedback/feedback.md`. Related work:
- `reviewer-feedback/published-references.md`: published versions of the circled preprints (6 of 8 published).
- `reviewer-feedback/candidate-references.md` and `paper-reviews.md`: 14 candidate papers read in full, with cite/redo verdicts.
- `results/extended_eval.txt`: new analyses answering the stats/validation/XAI comments (D27).
- **Still to do (user):** ask the reviewer what "XAI", "validation methods" and "not right claim" mean; fix the references; rewrite the paper.

**Draft errors found:** Section IV-A misreads bbOCR (bbOCR never tested EasyOCR, so our ordering agrees with it rather than contradicting it); Beshirov is vol. 26 no. 1 art. 4, not no. 4; Levchenko is missing pages 75–85 and the location.

## Unfinished work (from the paper's own limitations)

- **Alignment-based overgeneration filter** (D23). Specified, not built. Kanerva's method: character-level local alignment (biopython `PairwiseAligner`) to strip leading and trailing text that doesn't align with the input. Would show how much of the damage is preamble or commentary rather than real rewriting. Needs no new model runs: it can be applied to the existing `correction_eval.jsonl`.
- **Remaining 25 eval pages.** Only 15 of the 40 were run. It would cost roughly 24 h × 25/15 ≈ 40 h of Colab CPU for the chunked approach.
- **Phi-3 Mini** (D20). Never completed.
- **Whole-page approach on eval.** The code supports it; it only ran on dev. Dev suggested whole-page may do less damage for TituLLMs (D19).
- **Prompt variation.** One fixed prompt only (D10).
- **Future directions named in the paper's conclusion:** constrained or edit-distance-bounded decoding; line- or word-level correction; supervised fine-tuning (the approach that worked for Bulgarian in Beshirov et al.); simple guards on output length and Bengali-character share.

## Known inconsistencies and loose ends

- **Proposal vs paper hypothesis.** The proposal states ≥10% CER reduction; methodology dropped it (D03). Make sure the final paper doesn't reintroduce it.
- **Zenodo DOI.** The proposal promised to archive splits and code on Zenodo with a DOI. The paper only cites GitHub.
- **Table III layout.** In the submitted PDF, the rightmost column ("Tr", truncation %) is cut off at the page edge.
- **Paper source.** The current source is `paper/main.tex` (restored 2026-10-09 to the version that compiled to Draft_Paper_02). Older versions are in `paper/archive/`.
- **Dev baseline numbers differ between runs.** Tesseract dev CER was 0.371 in the first local scoring and ~0.350 in the Colab sweep scoring. Probably a different Tesseract build (local 5.4.0 against Colab's apt package); not verified. Dev numbers are never reported, so this is low priority.
- **Result files aren't versioned** (D26). The eval outputs exist only in local `results/` and on Drive. Consider committing the small summary files (`bootstrap_eval.json`, `figures/evaluation_matrix.csv`), or archiving the full outputs with the Zenodo deposit.

## Decoding ablation (review item C1), started 2026-10-09

`correct_text` now takes `decoding="guarded"` (the paper's runs: repetition_penalty 1.3 + no_repeat_ngram_size 4) or `"plain"` (neither; only the max_new_tokens guard). `run_sweep.py` exposes this as `--decoding` and adds `--pages`.

Smoke test, TituLLMs 1B, Tesseract, chunked, 3 eval pages, run on the local PC (not Colab, so not for the paper; D17):

| Page | OCR CER | guarded | plain |
|---|---|---|---|
| 279_34_D_26_0007 | 0.113 | 1.330 | 0.956 |
| 279_41_D_19_0004 | 0.319 | 1.825 | 1.003 |
| 279_42_B_41_0003 | 0.493 | 1.392 | 0.760 |

Plain decoding cuts the damage but the output is still far worse than raw OCR, and all 3 pages hit max_new_tokens. The failure changes from English commentary to Bengali sentence-repetition loops. Outputs: `results/ablation_plain_decoding.jsonl`. **Next:** the same test on Gemma 2B (needs an HF token; gated repo), then the full ablation on Colab.
