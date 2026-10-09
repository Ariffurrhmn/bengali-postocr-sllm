# Results

Numbers here are copied from `results/bootstrap_eval.json` and `results/correction_eval.jsonl` (checked 2026-09-26). If they disagree with this file, the result files win.

## Final eval: 15 held-out pages, chunked, bf16, multithreaded

**Eval pages (15, sampled from `data/split_eval.txt` with seed 403):**
`14076_a_2_0006`, `14090_d_15_0006`, `14090_d_15_0007`, `14131_c_2_2_0005`, `279_21_B_2_0004`, `279_23_D_6_0013`, `279_2_D_32_0006`, `279_2_a_15_0005`, `279_34_D_26_0007`, `279_35_B_1_0062`, `279_41_D_19_0004`, `279_42_A_60_0011`, `279_42_B_41_0003`, `279_42_K_4_0125`, `VT_1625_f_0003`

120/120 runs completed: 0 errors, 0 skipped.

**Uncorrected baselines:** Tesseract CER 0.3642, WER 0.8099. EasyOCR CER 0.2971, WER 0.7191.

Δ = corrected − baseline (positive = correction made it worse). CI = 95% paired bootstrap, 10,000 resamples, seed 403.

| Engine | Model | CER | ΔCER [95% CI] | WER | ΔWER [95% CI] | Trunc. |
|---|---|---|---|---|---|---|
| Tesseract | Gemma 2B | 0.5833 | +0.2190 [+0.1458, +0.2924] | 1.1070 | +0.2970 [+0.2012, +0.3912] | 0/15 |
| Tesseract | Llama 3.2 1B | 0.8439 | +0.4796 [+0.3620, +0.5843] | 1.1027 | +0.2928 [+0.0916, +0.4625] | 10/15 |
| Tesseract | BanglaT5 | 0.8892 | +0.5249 [+0.3939, +0.6384] | 0.9774 | +0.1675 [−0.1009, +0.3975] **n.s.** | 1/15 |
| Tesseract | TituLLMs 1B | 1.9958 | +1.6316 [+1.2051, +2.0455] | 2.3444 | +1.5345 [+0.9641, +2.0515] | 15/15 |
| EasyOCR | Gemma 2B | 0.5171 | +0.2200 [+0.1593, +0.3025] | 0.9780 | +0.2589 [+0.0844, +0.3898] | 0/15 |
| EasyOCR | Llama 3.2 1B | 0.8973 | +0.6001 [+0.4701, +0.7256] | 1.2146 | +0.4955 [+0.3310, +0.6805] | 9/15 |
| EasyOCR | BanglaT5 | 0.8165 | +0.5194 [+0.4122, +0.6123] | 1.0467 | +0.3276 [+0.1380, +0.4975] | 1/15 |
| EasyOCR | TituLLMs 1B | 2.5881 | +2.2910 [+1.7081, +2.8863] | 3.0266 | +2.3075 [+1.7231, +3.0201] | 15/15 |

All 16 deltas are positive; 15 are significant. The only non-significant one is Tesseract + BanglaT5 on WER.

**Runtime per page (seconds, mean / max):** BanglaT5 137–154 / 272. Llama 625–651 / 1456. Gemma 654–733 / 1338. TituLLMs 671–837 / 1568.

## Headline findings

1. **Correction makes things worse, everywhere.** Even the least-bad model (Gemma 2B) increases CER by ~0.22 on both engines. The worst (TituLLMs, EasyOCR) goes from 0.297 to 2.588, about 9× the baseline.
2. **Engine-agnostic in the bad sense.** Per-model deltas are similar across engines (within 0.13 CER for Gemma, BanglaT5 and Llama).
3. **Per page:** correction reduced CER in only 3 of 120 page-level cells (2.5%). All 3 are on the same page, the worst Tesseract page (baseline 0.882). The best improvement there was only to 0.803.
4. **Mechanism: content destruction.** Ground-truth token retention is 46% for raw Tesseract and 55% for raw EasyOCR. After correction it is 13% (Gemma), 2% (Llama), 3% (BanglaT5) and 4% (TituLLMs).
5. **Two opposite failure modes.** BanglaT5 collapses (0.34× reference length, degenerate punctuation and digits). TituLLMs runs away (2.58× length, 40% Bengali and 34% Latin, English literary commentary). Gemma stays closest to the reference length (0.88×) and does least damage. Length fidelity tracks damage.
6. **Truncation predicts failure.** Across the 8 engine × model conditions, truncation rate correlates with mean CER at r = 0.84.
7. **The Bengali-specific model did worst.** At this scale, Bengali pretraining did not help and plausibly hurt instruction-following.

## Output diagnostics (pooled over 15 pages × 2 engines)

| Source | Length / ref | Bengali | Latin | Other | GT token retention |
|---|---|---|---|---|---|
| Ground truth | — | 95% | 0% | 5% | — |
| Raw Tesseract | — | 79% | 12% | 9% | 46% |
| Raw EasyOCR | — | 92% | 0% | 8% | 55% |
| Gemma 2B | 0.88× | 84% | 7% | 9% | 13% |
| Llama 3.2 1B | 0.84× | 78% | 9% | 13% | 2% |
| BanglaT5 | 0.34× | 63% | 10% | 27% | 3% |
| TituLLMs 1B | 2.58× | 40% | 34% | 26% | 4% |

Source: `results/examples_*.txt` from `eval/inspect_examples.py`.

## Figures (`results/figures/`, produced by `eval/make_figures.py`)

- `fig1_cer_by_model.png`: mean CER per model and engine against the dashed baselines.
- `fig2_bootstrap_ci.png`: ΔCER with 95% CIs; all clear of zero.
- `fig3_per_page.png`: per-page CER sorted by baseline difficulty.
- `fig4_truncation.png`: truncation rate against mean CER (r = 0.84).
- `evaluation_matrix.csv`: Table III in CSV form.

## Dev set (10 pages): pipeline debugging only, never reported

These were run under mixed conditions: TituLLMs and BanglaT5 at fp32 on a single thread; Llama and Gemma at bf16 and multithreaded, whole-page only. Baselines: Tesseract ~0.35–0.37, EasyOCR ~0.35.

- TituLLMs whole-page: CER 1.61 (Tess), 3.36 (Easy); chunked 2.78 / 6.70. 10/10 truncated.
- BanglaT5 whole-page: 0.92 / 0.89; chunked 0.88 / 0.83.
- Gemma 2B whole-page: ~0.65–0.68.
- Llama 3.2 1B whole-page: ~2.5× baseline.
- Phi-3 Mini: 1 page only, CER 9.06, then abandoned.

The dev set already showed every model worse than baseline; the eval set confirmed it.

## Extended analysis (2026-10-04): no new model runs

Script: `eval/extended_analysis.py` → `results/extended_eval.txt` / `.json`. Each analysis follows a published protocol (see `reviewer-feedback/paper-reviews.md`).

- **Significance survives a stricter test.** Wilcoxon signed-rank with Holm correction: **15/16 CER/WER deltas significant**, the same 15 as the bootstrap. The only non-significant one is again Tesseract + BanglaT5 on WER (Holm p = 0.15). Every one of the 15 pages got worse in every CER cell (exact p = 0.00006, the smallest possible at n = 15). cMER: 8/8 significant.
- **The bounded metric (cMER) keeps the conclusion and removes the >1 artefact.** TituLLMs: CER 2.00/2.59 → cMER 0.88/0.88 (baseline 0.34/0.26). Every model roughly doubles to triples cMER.
- **Preference: 0 of 120 page-cells improve on cMER.** On CER, 3 of 120 improve (the 3 already reported, all on the worst Tesseract page); those vanish under cMER.
- **Damage vs fixes (word level).** Of the ground-truth words the raw OCR already had right, models broke **76–82% (Gemma)** and **96–100% (the other three)**. Of the words the OCR had wrong, they fixed **1–6%**.
- **Mechanism, by edit type** (mean per page vs ground truth, raw OCR → corrected): BanglaT5 deletions 72 → 667 (Tess) and 64 → 524 (Easy), i.e. collapse. TituLLMs insertions 37 → 1,106 and 72 → 1,604, i.e. runaway; 44–52% of its output characters sit in long runs with no counterpart in the input (hallucination). Llama substitutions roughly triple to quadruple, i.e. rewriting.
- **Change ratio** (characters changed ÷ characters that were wrong): Gemma 1.7–1.9; Llama/BanglaT5 3.9–4.6; TituLLMs 10–14. Every model changed more than there was to fix.
- **Length safeguard** (fall back to raw OCR if output < 0.35× or > 2.8× input; published thresholds): it reduces damage but **no model beats doing nothing**. BanglaT5 + Tesseract gets within +0.07 CER (n.s.) only because 13/15 pages fall back to raw OCR. Gemma triggers 0–1 fallbacks: its damage is same-length rewriting, which length checks can't catch.
- **Difficulty:** damage is **largest on the easiest pages** and smallest on the hardest (e.g. Llama + Tesseract +0.69 easy vs +0.23 hard). This matches the over-correction-of-low-noise-input finding in HIPE-OCRepair 2026.
- **Input length:** weak positive correlation between page length and damage (Spearman 0.03–0.65; significant only for Llama/BanglaT5 + Tesseract, uncorrected). Raw-OCR pages are a median 743–790 characters, all above the ~400-character range where Danilova et al. report degradation. Chunk lengths in characters are **not yet measured** (needs the model tokenizers).
