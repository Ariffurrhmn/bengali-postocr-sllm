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

**Gemma 2B result (Colab CPU, same 3 pages, 2026-10-09):** the guards caused nearly all of Gemma's damage.

| Page | OCR CER | guarded | plain |
|---|---|---|---|
| 279_34_D_26_0007 | 0.113 | 0.298 | 0.120 |
| 279_41_D_19_0004 | 0.319 | 0.626 | 0.310 |
| 279_42_B_41_0003 | 0.493 | 0.892 | 0.500 |
| mean | 0.308 | 0.605 | 0.310 |

With plain decoding Gemma ends up about level with raw OCR (it neither helps nor hurts). The paper's headline "correction makes OCR worse" is therefore, at least for Gemma, mostly a decoding artefact. Not yet checked: whether plain Gemma is just echoing its input. **Next:** Gemma plain on all 15 pages × both engines, then Llama 3.2 1B; then rewrite the result.

**Gemma 2B, full 15 pages × both engines (Colab CPU, 2026-10-09):**

| Engine | raw OCR CER | guarded | plain | plain beats OCR | plain beats guarded |
|---|---|---|---|---|---|
| Tesseract | 0.364 | 0.583 | 0.360 | 7/15 | 14/15 |
| EasyOCR | 0.297 | 0.517 | 0.301 | 2/15 | 15/15 |

How much plain Gemma changed its input (CER of output vs OCR input): mostly 0.00–0.03, max 0.12. So with plain decoding Gemma essentially **echoes the OCR text**: no harm, no help. All of the paper's Gemma damage came from the decoding guards. Candidate reframing: zero-shot small LMs don't correct Bengali OCR; with plain decoding they copy, and with common anti-repetition guards (which also see the prompt) they are forced to rewrite and do damage. Plain outputs are on Drive: `bengali-postocr-results/ablation_plain_decoding.jsonl`. **Next:** Llama 3.2 1B, TituLLMs, BanglaT5 plain on the same 15 × 2.

**All four models, 15 pages × both engines (Colab CPU, finished 2026-10-10).** Mean per-page CER:

| Model | Tess OCR | Tess guarded | Tess plain | Easy OCR | Easy guarded | Easy plain |
|---|---|---|---|---|---|---|
| Gemma 2B | 0.364 | 0.583 | 0.360 | 0.297 | 0.517 | 0.301 |
| Llama 3.2 1B | 0.364 | 0.844 | 0.430 | 0.297 | 0.897 | 0.337 |
| TituLLMs 1B | 0.364 | 1.996 | 0.929 | 0.297 | 2.588 | 1.029 |
| BanglaT5 | 0.364 | 0.889 | 0.900 | 0.297 | 0.817 | 0.837 |

Reading: the guards caused most of the damage for all three decoder-only models. With plain decoding, no model beats raw OCR on average: Gemma ties it (it echoes the input), Llama is somewhat worse, and TituLLMs is still far worse (repetition loops; every page hits max_new_tokens). BanglaT5 is unchanged, as expected, because for an encoder-decoder the guards don't see the input; it fails because it isn't trained for correction. **The negative result survives, but in a different form:** "zero-shot small LMs don't improve Bengali OCR, and a common anti-repetition setting turns them from harmless to harmful." The paper's damage numbers (Tables IV/V, 76–100% of correct words broken) are mostly a decoding artefact and must be redone.

Note: TituLLMs plain on 279_34_D_26_0007 scored 0.956 in the local smoke test and 0.905 on Colab (different library versions). Only the Colab numbers count.

**Significance (paired bootstrap, 10k resamples, seed 403; `results/bootstrap_plain.json`, `results/decoding_ablation.json`).**
- Plain vs raw OCR, CER: Gemma n.s. on both engines (Tess −0.004 [−0.013, +0.002]; Easy +0.004 [−0.014, +0.017]). Llama significantly worse (Tess +0.066 [+0.019, +0.114]; Easy +0.040 [+0.005, +0.082]). TituLLMs and BanglaT5 significantly worse (+0.54 to +0.73). No cell significantly better.
- Plain minus guarded, CER: significant improvement for all decoder-only cells (Gemma −0.22, Llama −0.41/−0.56, TituLLMs −1.07/−1.56). BanglaT5 slightly worse without guards (Easy +0.020, sig; Tess +0.011, n.s.).
- Median CER of plain output against its own OCR input: Gemma 0.005/0.012 (echoes), Llama 0.065/0.163 (light edits that hurt), TituLLMs 0.96, BanglaT5 0.84–0.88 (regenerates).

**Newer model (E-a), 2026-10-10:** added `qwen3-1.7b` (Qwen/Qwen3-1.7B, ungated, thinking disabled via the chat template) to `correction/models.py`. To run on Colab with plain decoding, all 15 pages × both engines.

**Qwen3 1.7B, plain, 15 pages × both engines (Colab CPU, 2026-10-10):** CER Tess 0.380 vs OCR 0.364 (+0.016 [−0.007, +0.047], n.s.); Easy 0.310 vs 0.297 (+0.013 [−0.005, +0.035], n.s.). WER slightly lower, n.s. Sample output is a near-copy of the OCR input, Latin-script garbage included. So a 2025 model behaves like Gemma: it mostly echoes and neither helps nor hurts. The negative result is not specific to 2024-era models.

**Paper tables regenerated from the plain runs (2026-10-10).** `make_paper_tables.py` now defaults to the plain-decoding files and also writes `paper/tables/decoding.tex` (guarded vs plain; no Qwen row since Qwen has no guarded run). Correction to the note above: Llama's CER increase is significant by bootstrap CI but **not** after Holm-corrected Wilcoxon (Tess p=0.16, Easy p=0.36), the paper's primary test. Under that test only BanglaT5 and TituLLMs are significantly worse; Gemma, Qwen and Llama are not significantly different from raw OCR. `main.tex` text and the figures still describe the old guarded results; they need rewriting next.

**main.tex partly rewritten (2026-10-10).** Done: Methods (Decoding paragraph now discloses that the guards were added after seeing dev output and explains why they penalise copying; Choice of models adds Qwen3 1.7B and measured parameter counts: Gemma 2.51B, Qwen3 1.72B, Llama/TituLLMs 1.24B, BanglaT5 ~248M; Holm family now 20 tests), the whole Results section (new "Effect of the decoding setting" subsection with `tables/decoding.tex`) and the whole Failure Analysis. Figures 1–3 regenerated from the plain runs; the truncation figure and its r=0.84 claim are dropped (reviewer F3). Qwen3 citation added (arXiv:2505.09388, checked).
**Still stale (describe the old result):** title, abstract, introduction (RQ1 "at most two billion", "Correction increased error rates for both engines", contributions), Discussion (RQ answers, relation to prior work, explanations, practical implications, the "No output filter" limitation that quotes Gemma 0.364→0.583), Conclusion.
**Open question:** the paper says TituLLMs has an extended tokenizer, but the v1.1 checkpoint config has the same vocab size as Llama 3.2 1B (128,256). Check the TituLLMs paper/model card before keeping that sentence.
