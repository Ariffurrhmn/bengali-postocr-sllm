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

**40-page GPU run (planned 2026-10-10/11).** Decision: main results come from one consistent setup, all 40 eval pages on a free T4 GPU (Kaggle, D30) in float32 (float16 above ~3B) with plain decoding, five models, both engines, plus the clean-text control (ground truth as input, E1). The 15-page CPU runs become a CPU-vs-GPU cross-check; the decoding ablation stays on the 15 CPU pages. Colab Tesseract is 5.3.4. OCR runs on a CPU runtime (EasyOCR is set to CPU anyway) to save GPU quota: `ocr_check15.jsonl` re-OCRs the original 15 pages to test whether today's engines reproduce the August OCR; `ocr_eval40.jsonl` = the original 15 pages copied unchanged plus 25 new. Corrections go to `gpu_eval40.jsonl` (cell 3a: tesseract+easyocr first; cell 3b: ground_truth).

**OCR drift found (2026-10-11).** Re-OCR of the original 15 pages on today's Colab (Tesseract 5.3.4, Python 3.13) does not reproduce the August OCR: Tesseract text differs on 15/15 pages, EasyOCR on 10/15; ground truth identical. So the 40-page set must not mix August OCR (15 pages) with today's (25). Plan: build the 40-page OCR file from today's engines only (`ocr_check15.jsonl` + the 25 new pages from `ocr_eval40.jsonl`) and quantify the baseline-CER shift on the 15 overlapping pages. The 15-page CPU results stay internally consistent (August OCR throughout).

**OCR drift measured (2026-10-11).** On the 15 original pages, today's Tesseract (5.3.4) scores mean CER 0.320 against 0.364 for the August OCR, and the two Tesseract outputs differ by mean CER 0.278 (max 0.785) against each other; EasyOCR is essentially unchanged (0.297 both; outputs differ by 0.001). The August Tesseract version is not recorded. `results/ocr_eval40.jsonl` now holds all 40 pages from today's engines (15 from `ocr_check15.jsonl` + 25 new); the mixed version is kept as `ocr_eval40_mixed.jsonl` (locally and on Drive) and must not be used; the Drive `ocr_eval40.jsonl` was replaced with the single-version file on 2026-10-11. 40-page baselines: Tesseract 0.341, EasyOCR 0.297.

**Kaggle run 1 started (2026-10-11).** Notebook `kaggle_gpu_run`, Version #2 ("run1-gpu-40pages"), GPU T4 x2, background commit; input dataset `arifurrhmn/bengali-ocr-40` (private) holding the single-version `ocr_eval40.jsonl`. Repo at 640c6d2. Output `gpu_eval40.jsonl` and `sweep.log` in the version's Output tab. If it stops early: new version with the previous version's output added as input; the notebook resumes.

**Kaggle run 1 finished (2026-10-11): 960/960 results, 0 errors, all on cuda:0** (8 models × 40 pages × {tesseract, easyocr, ground_truth}). The user copied the files from Kaggle's text view (download didn't work); validated: 960 unique keys, no U+FFFD or mojibake, size matches Kaggle's 2.16 MB. Files: `results/gpu_eval40.jsonl` (all), `gpu_eval40_ocr.jsonl` (OCR inputs), `gpu_eval40_clean.jsonl` (clean control), `gpu_sweep40.log`; copies on Drive. Analyses: `results/bootstrap_gpu40.json`, `results/extended_gpu40.{json,txt}`.

40-page results (mean CER; baselines Tess 0.341, Easy 0.297; Holm-corrected Wilcoxon over 32 CER/WER tests):
- Not significantly different from raw OCR on CER: Gemma 2B (0.353 / 0.301), Qwen3 1.7B (0.337 / 0.315), Phi-4-mini (0.354 / 0.315).
- Significantly worse: Llama 3.2 1B (0.380 / 0.330; Holm p 0.0023 / 0.017, now significant with 40 pages), BanglaT5 (0.860 / 0.831), TituLLMs 1B base (1.084 / 1.033), TituLLMs 1B Instruct (1.642 / 1.633, the worst), TituLLMs 3B Instruct (0.958 / 0.886).
- The only significant improvement anywhere: Qwen3 1.7B WER on Tesseract text, 0.827 → 0.799 (−0.028, CI [−0.047, −0.013], Holm p 0.0016). Its CER change there is −0.004, n.s.
- 19/32 CER/WER tests significant after Holm, all increases except Qwen's Tesseract WER.

Clean-text control (ground truth as input; CER of output vs ground truth; perfect = 0): Gemma 0.032 [0.018, 0.049], Qwen3 0.035, Phi-4-mini 0.060, Llama 0.119, BanglaT5 0.833, TituLLMs 1B 0.988, TituLLMs 3B Instruct 0.967, TituLLMs 1B Instruct 1.565. Pages left exactly correct: Gemma 6/40, Phi-4-mini 4, Llama 3, Qwen3 2, others 0. So every model damages already-correct text; the general models only slightly.

CPU-vs-GPU cross-check (EasyOCR, the 5 pages whose input is byte-identical across OCR versions): Gemma outputs identical 5/5; Qwen3 3/5 and Llama 2/5 with outputs differing by CER ≤0.004; TituLLMs and BanglaT5 outputs differ (CER 0.20 / 0.44 between runs) but their scores agree (0.945 vs 0.954; 0.806 vs 0.813). Hardware changes individual degenerate outputs, not the conclusions.

**Revision analyses on the 40-page GPU run (2026-10-11)** — `eval/revision_analysis.py` → `results/revision_gpu40.{json,txt}`; `eval/model_profile.py` → `results/model_profile.json`.
- T1: 40 pages from 37 books (book = page id minus page number; at most 2 pages per book; 28 groups at the coarser "collection" level, e.g. 279_42). GT median 577 characters (193–2715), median 89 words. Raw OCR length / reference ≈ 1.0 for both engines.
- M8 pooled CER is below the per-page mean (raw Tesseract 0.298 vs 0.341; EasyOCR 0.286 vs 0.297): short pages have higher CER. Model ranking unchanged.
- M7 grapheme-cluster CER (regex \X, Unicode 14 rules: conjuncts stay one cluster, ZWNJ splits) is *higher* than code-point CER (raw Tesseract 0.431, EasyOCR 0.387), because one wrong code point fails the whole cluster. Ranking unchanged.
- M6: book-level and collection-level cluster bootstraps give practically the same intervals as the page bootstrap; no conclusion changes.
- E3: for Gemma, Qwen3, Phi-4-mini and Llama, corrected CER tracks baseline CER almost perfectly (Spearman 0.88–0.99), consistent with copying; the change is negatively correlated with baseline for most cells (larger increases on easier pages). For BanglaT5/TituLLMs corrected CER is unrelated to input quality (rho −0.00 to 0.30, except TituLLMs 1B on Tesseract 0.55), i.e. regeneration rather than editing (reviewer C5).
- E8 edit-distance cap (output kept only if CER vs input ≤ tau, tau grid 0.05/0.1/0.2/0.3 fixed in advance): no setting brings any model meaningfully below raw OCR (best: Qwen3 Tesseract 0.337–0.339 vs 0.341); for BanglaT5/TituLLMs the cap rejects (almost) every page, i.e. it reduces to raw OCR.
- E7 by character class: raw OCR is worst on Bengali digits (Tess 0.65, Easy 0.53), danda (EasyOCR 1.00: it never outputs a danda) and punctuation; conjunct characters 0.29 / 0.20, consonants 0.19 / 0.14. The general models raise every class slightly; Gemma raises digit errors most (Tess 0.65 → 0.75, Easy 0.53 → 0.73). Page-type breakdown (verse/title/prose) not done: no page-type labels.
- E5/T2: tokens per character on the 40-page ground truth: Llama 3.2 1B and TituLLMs 1B v1.1 1.22, Qwen3 1.08, Gemma 0.60, Phi-4-mini 0.45, TituLLMs Instruct 1B/3B 0.38, BanglaT5 0.29 (SentencePiece). Mean characters per ~250-token chunk therefore range 168 (Llama) to 483 (BanglaT5): the tokenizer-dependent chunking confound of M3 is real. Fertility does not predict outcome (TituLLMs Instruct has low fertility and is worst). Run time per page: CPU (Colab, 15 pages) 2.2–14.3 min; GPU (T4) mostly 0.3–0.6 min, BanglaT5 ~3 s, TituLLMs 1B Instruct 2.4 min (long degenerate outputs).

**Tables and figures switched to the 40-page GPU run (2026-10-11).** `make_paper_tables.py` now defaults to `ocr_eval40.jsonl` / `gpu_eval40_ocr.jsonl` / `bootstrap_gpu40.json` / `extended_gpu40.json` and adds `dataset.tex` (T1), `models.tex` (T2), `editops.tex` (T3) and `clean.tex` (E1); `decoding.tex` keeps the 15-page CPU ablation with its own baseline (0.364/0.297). New figures from `eval/make_revision_figures.py`: `fig_delta.png` (F1: ΔCER with CIs, all models plus a finer-scale panel) and `fig_scatter.png` (F2/F5/E3: per-page ΔCER vs baseline, one panel per model). The old fig1–3 and all of `main.tex`'s numbers still describe the 15-page plain run and must be replaced in the rewrite. BanglaT5's falling line in fig_scatter is mechanical: its output CER is near-constant (~0.85), so ΔCER ≈ 0.85 − baseline.
**M1 still open:** Tesseract is called with defaults (`image_to_string(img, lang="ben")`, so PSM 3, LSTM engine, tessdata_best `ben`), no preprocessing; Colab version 5.3.4 for the 40-page OCR, August version unknown. Image DPI not yet read: the dataset zip on Drive is online-only and the C: drive was full (117 MB free) when Drive tried to cache it.

**Classical baseline (C4) done (2026-10-11).** `correction/dictionary_baseline.py`: SymSpell (symspellpy, defaults max edit distance 2, prefix length 7) with the FrequencyWords `bn_50k` lexicon (OpenSubtitles 2018, CC BY-SA 4.0; 44,335 Bengali entries; `data/lexicon/`, not committed, sha256 in the script). Output `results/dict_eval40.jsonl` (model_key `symspell`). It also makes the text worse: CER Tess 0.341 → 0.378 (+0.037 [+0.031, +0.042]), Easy 0.297 → 0.337 (+0.040); WER +0.09 on both; on clean text it replaces ~31 words per page (CER 0.057, WER 0.268), because the modern subtitle lexicon lacks the historical spellings. So the failure is not specific to language models. SymSpell is now part of all 40-page analyses (combined files `results/eval40_all.jsonl` / `eval40_ocr.jsonl`; Holm family 36 CER/WER tests, 23 significant, all increases except Qwen3's Tesseract WER), the tables and the figures.

**Kaggle variants run (2026-10-11).** Notebook `notebook31c350ef4f` (imported `kaggle_variants_run.ipynb`), dataset `bengali-ocr-40` version 2 (adds `fewshot_examples.json`), repo at 34b5e2d. Versions #1 and #2 were cancelled within minutes, cause not checked (the user stopped the draft session around then); Version #3 ("variants-run1") queued. Output: `gpu_variants40.jsonl` (4 models × 4 variants × 80 = 1,280 records).

**Variants run finished (2026-10-12): 1,280/1,280, 0 errors, all cuda:0** (Version #3; copied from Kaggle's text view and validated). Files `results/gpu_variants40.jsonl`, `gpu_variants40.log`; Drive copies. Analysis `eval/compare_variants.py` → `results/variants_gpu40.{txt,json}` (paired bootstrap vs raw OCR and vs original prompt; Wilcoxon vs original, Holm over 32 tests).
- No prompt or chunking variant makes any model better than raw OCR. Closest: Qwen3 + preserve-spelling on Tesseract, 0.335 vs 0.341 (−0.006 [−0.015, −0.000]).
- Preserve-spelling prompt: fewer changes and slightly less damage for every model (Llama: Tess 0.380 → 0.362, Easy 0.330 → 0.304), none significant against the original after Holm.
- Bengali prompt: more damage; significant against the original for Llama (Tess +0.136, Easy +0.129, Holm p < 0.001 / 0.001) and Phi-4-mini on EasyOCR (+0.044, Holm p 0.027).
- Three-shot prompt: more damage for Gemma, Phi-4-mini and Llama (Llama Tess +0.100, Holm p 0.011); Qwen3 on EasyOCR slightly better than original (−0.012, n.s.).
- 300-character chunks (same for every model): no significant change for any model; e.g. Gemma Tess 0.343 vs 0.353, Llama 0.380 vs 0.380.
So the negative result is robust to prompt language, the preserve-spelling instruction, few-shot examples and model-independent chunking (M3, M4 done).

**M1 and F4 done; Methods rewritten (2026-10-12).** After the user freed C: (7.5 GB), the 40 evaluation pages (TIFF + PAGE-XML) were extracted from the Drive zip to `data/REID2019/` (gitignored; 79 files: page 279_2_a_15_0005 has only one of the two, name-case issue, not yet checked). All 40 images: 400 dpi, RGB, uncompressed, median 1808 × 3020 px. `make_paper_figures.py` updated: method diagram with larger fonts and the current setup (F4); dataset figure now ends in "all 40 pages scored". `main.tex` Methodology fully rewritten for the 40-page GPU study (dataset, OCR versions and scope, prompts and variants, chunking, decoding disclosure, eight models, SymSpell, computing setting, metrics incl. pooled/grapheme CER and ZWJ/ZWNJ/danda handling, 36-test Holm family, book bootstrap, clean-text control, safeguards, reproducibility with checkpoint revisions). New references: tesseract5, abouelenin2025phi4mini, garbe2025symspell, dave2018frequencywords. Results, Failure Analysis, Discussion, Introduction, Abstract and Conclusion still describe the 15-page run.

**Bug fixed in `extended_analysis.py` (2026-10-12):** the difficulty bands were hard-coded as pages [:5], [5:10], [10:] — correct for the 15-page sample, wrong for 40 pages (5/5/30). Now thirds of the page count (13/13/14 for 40; still 5/5/5 for 15, verified byte-identical against the old 15-page output). Also replaced the hard-coded "n = 15" and "/15" labels. No reported 40-page number had used the faulty bands yet.

**Qwen3 on Tesseract slightly improves word- and grapheme-level error (2026-10-12).** Grapheme-cluster CER change, paired bootstrap + Wilcoxon with Holm over 18 tests (`revision_analysis.py`, section M7): Qwen3/Tesseract −0.015 [−0.032, −0.004], Holm p 0.021; together with its WER change −0.028 [−0.047, −0.013], Holm p 0.002 (36-test family). Its code-point CER change (−0.004) is not significant. No other model–engine pair improves on any metric. The paper's claim must be "no model significantly lowered CER; one pair slightly lowered WER and grapheme CER", not "nothing improved anything". Also measured on the 40-page GPU outputs: TituLLMs 3B Instruct returned an empty page on 27/80 and a Bengali description of the text instead of a correction on 37/80; TituLLMs 1B loops (a ≥20-character segment recurring ≥3 times) on 51/80 pages, TituLLMs 1B Instruct on 34/80, no other model on more than 2.

**Results and Failure Analysis rewritten for the 40-page study (2026-10-12).** New Results subsections: baseline; effect of correction (two groups, Llama significant, Qwen's WER/grapheme improvement); clean-text control; decoding setting (15-page CPU); prompt and chunking variants (new `tables/variants.tex`); engine agreement; per-page analysis (copy vs regenerate, difficulty thirds); robustness (pooled, grapheme, cMER, book bootstrap); CPU vs GPU. Failure Analysis: word retention, edit profiles (incl. TituLLMs failure modes and character classes), truncation, both safeguards. Old fig1–fig3 removed (replaced by fig_delta, fig_scatter). Still stale: Discussion, Introduction, Abstract, title, Conclusion, Related Work.

**Discussion and Conclusion rewritten (2026-10-12).** Four research questions answered (RQ3 = what correction does to the text, RQ4 = whether decoding, prompt, chunking or safeguards change the outcome; reviewer W4). "Possible explanations" now separates tested from untested: input noise contradicted by the clean-text control (C5), tokenisation contradicted by fertility (E5), historical spelling and instruction-following kept as labelled hypotheses (W6). New "Threats to validity and limitations" subsection (W8) lists the skipped checklist items (larger models, fine-tuning, stronger OCR, confidence gating, page-type breakdown) as out of scope. Practical warning about repetition penalty/n-gram ban acting on the prompt in HF generate. Still stale: title, abstract, introduction (RQ list and contributions), Related Work.

**Confidence-gated correction (E4) done (2026-10-12).** `ocr_conf40.jsonl` (Colab CPU, Tesseract 5.3.4 / EasyOCR 1.7.2) reproduces the stored OCR text on 40/40 Tesseract and 37/40 EasyOCR pages; 0/5015 and 1/4952 stored words lack a confidence. Low-confidence shares are large: Tesseract words below 50/70/90 = 41/57/77%; EasyOCR boxes below 0.5/0.7/0.9 = 52/81/93% of words. Results `results/gated_gpu40.{txt,json}` (Holm over 54 tests):
- Gating cuts the damage of the harmful methods sharply but never below raw OCR: e.g. TituLLMs 1B Tess 1.084 → 0.454 (θ=50), BanglaT5 Tess 0.860 → 0.440, SymSpell Tess 0.378 → 0.348; most remain significantly worse.
- General models end at or just below raw OCR on Tesseract: Qwen3 θ=70 −0.004 [−0.006, −0.002], Holm p 0.038 (0.341 → 0.337) — the only significant CER improvement in the study, about 1% relative; Phi-4-mini −0.003 (θ 50/70, CI excludes 0, Holm n.s.); Gemma ≈0. On EasyOCR nothing improves (best Phi-4-mini θ=0.5 −0.001, n.s.).
- Lower thresholds (stricter gating) are better throughout; the EasyOCR gate is coarse because confidences are per box.

**Title, abstract and introduction rewritten (2026-10-12).** Title: "Does Zero-Shot Small-Model Post-Correction Help Bengali OCR? A Negative Result Across Eight Models and Two OCR Engines" (reviewer C2, extended). Abstract 224 words, structured problem → setup/corpus → key numbers → finding → limitation (W1). Introduction groups prior evidence into helped / mixed / worse, then the gap (W2), states four RQs (W4) and five contributions; novelty claim qualified with "to our knowledge" (V6 still needs a literature search). "Held-out" and "at most two billion" removed everywhere. Related-work gap sentence corrected to "none applies open models below seven billion parameters to OCR correction or uses input with a CER above 0.25" (checked against tables/literature.tex); "This study" row of the literature table updated. Remaining: Related Work synthesis sentences (W3), new references (R1), reference/model-name checks (V5, e.g. "GPT-5.5" in Viana), novelty search (V6), Bengali examples (T4).

**Literature items done (2026-10-12): R1, V5, V6, W3.** Related Work now has a new subsection "Post-OCR correction before language models" (Nguyen et al. 2021 survey; ICDAR 2017 and 2019 post-OCR competitions; Rijhwani et al. 2020 low-resource post-correction with a copy mechanism; Pal, Kundu & Chaudhuri 2000 Bangla OCR correction by morphological parsing), Chaudhury et al. 2022 (degraded Bangla OCR, TALLIP), BSpell (BLP-2023) and the DPCSpell journal version (Computer Speech & Language 89, 2025), and a synthesis sentence closing each subsection (W3). The Witte poster (old [15]) is replaced by Do et al., AAAI 2025 (spell correction underperforms on historical documents because modern and historical corpora differ; verified from the abstract). Verified (V5): Vershinin et al. exists in the Russian Digital Libraries Journal 28(5):1036–1056, in Russian with an English abstract (old [4]; numbers come from its Table 1, read in full earlier); Viana 2026 matches Crossref (Information 17(8):722); "GPT-5.5" is a real OpenAI model (April 2026). Novelty (V6): no published LLM post-correction study for Bengali OCR found (only this project's repo); the claim keeps "to our knowledge" and now cites the pre-LLM Bangla work. Not used: the claim that degraded-Bangla OCR text is "linguistically older" (only in the paper body, not verifiable from the abstract). All new entries checked against Crossref/ACL Anthology/publisher. Remaining checklist item: T4 (Bengali example lines).

**T4 (Bengali examples) set up (2026-10-12).** `eval/find_examples.py` picks candidate lines; `eval/make_example_table.py` writes `paper/tables/examples.tex` (two lines: VT_961_k_0172, a natural-history page where Qwen3 repairs a word and the Bengali models regenerate; 279_2_D_32_0007, verse, where Qwen3 repairs one word and breaks another). Each row is the output span aligned to the reference line, trimmed at the ends only while that lowers its CER (same rule for every row), words differing from the reference in red. `tables/preamble.tex` loads fontspec + Noto Serif Bengali under XeLaTeX, with a placeholder under pdfLaTeX. Needs `paper/fonts/NotoSerifBengali-Regular.ttf` (notofonts.github.io, 336,632 bytes, OFL) and the Overleaf compiler set to XeLaTeX.
