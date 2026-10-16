# Results

Numbers here are copied from the result files in `results/` (local only, gitignored; copies on Google Drive in `bengali-postocr-results/`). If they disagree with this file, the result files win. The paper's tables are generated from the same files by `eval/make_paper_tables.py`.

## Current study (revision, October 2026)

**Setup.** All 40 evaluation pages (37 books), OCR by Tesseract 5.3.4 and EasyOCR 1.7.2 (October 2026, `results/ocr_eval40.jsonl`), plain greedy decoding, eight models on a Kaggle T4 GPU plus the SymSpell dictionary baseline (`results/eval40_all.jsonl`). Paired bootstrap (10,000 resamples, seed 403) and Wilcoxon signed-rank with Holm correction over the 36 CER/WER tests.

**Baselines.** Tesseract CER 0.341, WER 0.827, cMER 0.290. EasyOCR CER 0.297, WER 0.757, cMER 0.260.

| Method | Tesseract CER | EasyOCR CER | Verdict |
|---|---|---|---|
| SymSpell (dictionary) | 0.378 | 0.337 | significantly worse |
| Gemma 2B | 0.353 | 0.301 | n.s. |
| Qwen3 1.7B | 0.337 | 0.315 | n.s. (WER on Tesseract significantly better: 0.827 → 0.799) |
| Phi-4-mini | 0.354 | 0.315 | n.s. |
| Llama 3.2 1B | 0.380 | 0.330 | significantly worse |
| BanglaT5 | 0.860 | 0.831 | significantly worse |
| TituLLMs 1B (base, v1.1) | 1.084 | 1.033 | significantly worse |
| TituLLMs 1B Instruct | 1.642 | 1.633 | significantly worse |
| TituLLMs 3B Instruct | 0.958 | 0.886 | significantly worse |

**Headline findings**
1. No method significantly lowers CER on either engine. 23 of 36 CER/WER tests are significant; 22 are increases. The one improvement is Qwen3 1.7B on Tesseract output (WER −0.028, grapheme CER −0.015, both significant; CER −0.004 n.s.).
2. General-purpose models copy: corrected CER tracks baseline CER page by page (Spearman 0.88–0.99), change ratio 0.1–0.4. Bengali-specific models regenerate: CER unrelated to input quality, 74–99% of correct words broken. TituLLMs 3B Instruct returned an empty page on 27/80 and a description of the text on 37/80; TituLLMs 1B loops on 51/80 pages.
3. Clean-text control (reference text as input): every method damages correct text. CER against the reference: Gemma 0.032, Qwen3 0.035, SymSpell 0.057, Phi-4-mini 0.060, Llama 0.119, Bengali-specific models 0.83–1.57.
4. Engines agree: CER changes differ by at most 0.035 between engines; same groups and same order of the four Bengali-specific models.
5. Robustness: pooled CER, grapheme-cluster CER, cMER (13/18 significant, all increases) and book-level bootstrap give the same conclusions.
6. CPU vs GPU: on identical inputs Gemma's outputs are identical, Qwen3/Llama differ by CER ≤ 0.004; degenerate models differ but score the same.

**Decoding ablation** (`results/decoding_ablation.json`; 15-page CPU sample, August OCR, the only setting where both decodings exist). The original repetition penalty 1.3 + no-repeat-4-gram also act on the prompt. Plain minus guarded CER: Gemma −0.224/−0.216, Llama −0.413/−0.560, TituLLMs −1.067/−1.559 (all significant); BanglaT5 +0.011/+0.020. Under the guards Gemma raised CER 0.364 → 0.583; under plain decoding it echoes its input (median CER vs input 0.012/0.005). Most of the first version's reported damage was caused by the decoding constraints.

**Prompt and chunking variants** (`results/variants_gpu40.txt`; four general-purpose models). No variant beats raw OCR. Preserve-spelling instruction: slightly less damage in 7 of 8 pairs, none significant. Bengali prompt: worse (significant for Llama on both engines and Phi-4-mini on EasyOCR). Three-shot (examples from dev pages): worse (significant for Llama on Tesseract). 300-character chunks: changes ≤ 0.016, none significant.

**Safeguards** (`results/extended_gpu40.txt`, `results/revision_gpu40.txt`, `results/gated_gpu40.txt`). Length safeguard triggers only for Bengali-specific models; all remain significantly worse. Edit-distance cap: no τ gets below raw OCR. Confidence gating (edits kept only on words below an OCR-confidence threshold): removes most of the damage (TituLLMs 1B Tesseract 1.084 → 0.454) but never gives a meaningful gain; the one significant CER reduction in the study is Qwen3 on Tesseract at threshold 70, 0.341 → 0.337.

**Safeguard tests (added 2026-10-12, second review).** Length rule p-values (Holm over 18): every triggering model stays significantly worse (e.g. TituLLMs 3B Instruct Tesseract +0.319 [+0.193, +0.460], p 0.001). Edit-distance cap (Holm over 72): nothing significantly better; 7 significantly worse (SymSpell τ ≥ 0.1 both engines, Llama τ 0.3 Tesseract). Table `safeguards.tex`.

**Sensitivity checks** (`results/sensitivity_gpu40.txt`, second review). Reading order: 30/40 PAGE files declare one; it changes the reference on 17 pages; raw CER 0.335/0.294 instead of 0.341/0.297; the same 12/18 CER comparisons significant, all increases. Reference-word recall (order-free, ignores extra text): raw 0.431/0.484; no method raises it significantly (Qwen3 +0.003/+0.002 n.s.), all others lower it significantly (Gemma −0.017/−0.019). Page types (`data/page_types.tsv`, assigned from transcriptions): 10 title, 8 verse, 13 prose, 9 list; raw CER highest on lists (0.576/0.511), lowest on prose (0.199/0.184); no method improves any type consistently.

**Script shares fixed.** Table X now pools script shares over all output characters (the first version averaged per-output shares and counted empty outputs as 0%, so TituLLMs 3B Instruct showed 63/0/3%). Now: TituLLMs 3B Instruct 97% Bengali with 27/80 empty outputs; BanglaT5 28% "other" (was reported as about a third).

**Other measurements.** Tokens per character 0.29 (BanglaT5) to 1.22 (Llama/TituLLMs 1B), per word 1.9 to 8.0; chunk sizes 168–483 characters. CPU time per page 2.2–14.3 min (Colab); GPU 0.05–2.4 min (T4). Raw OCR worst on Bengali digits and the danda (EasyOCR never outputs a danda). Dataset: GT median 577 characters (193–2715), raw OCR ≈ 1.0× reference length.

**Third-review analyses** (`results/review3_gpu40.txt`, 2026-10-15; existing outputs). wMER baseline 0.652 / 0.629; Phi-4-mini significantly higher on both engines, Llama on Tesseract (exploratory Holm over 18). Equivalence (TOST, paired t): within ±0.02 CER only Qwen3/Tesseract (p 0.001) and Gemma/EasyOCR (p < 0.001); none within ±0.01. Pages better/worse: Gemma 16/16 (Tess), Qwen3 22/13 (Tess), Llama 7/29 and 8/29. Gated Qwen3 Tesseract @70: 25 better, 11 worse, 4 unchanged. Gemma converts Bengali to ASCII digits (ASCII 88→336 Tess, 420→640 Easy); digit-normalised CER moves it by 0.002/0.006. ZWJ/ZWNJ removal ≤ 0.003; khanda-ta no effect. 52% of reference word tokens are missing from the SymSpell lexicon. Phi-4-mini clean-text pooled CER 0.097 vs mean 0.060: page 279_23_D_6_0013 gives 70% of its errors (switches to English, runs to the cap). Raw Tesseract word recall 0.4314 vs grapheme CER 0.4311 (coincidence). TituLLMs 3B Instruct: 43/120 outputs empty (27 OCR + 16 clean), each in 0.2–0.5 s.

**Third-review GPU runs** (Kaggle T4 x2, 2026-10-16, transformers 5.19.0, torch 2.11.0+cu128; outputs `results/review3/`, scored by `eval/review3_runs_analysis.py` → `results/review3_runs.txt`).
- **C1 precision: not an artifact.** TituLLMs 3B Instruct in fp32 gives the same first token as fp16 on all 208 chunks (87 end-of-sequence in both), no inf/NaN, largest activation about 1000 (fp16 limit 65504), the same empty pages (13/14/16) and the same CER (fp32 − fp16: +0.000 Tess, +0.002 Easy). Phi-4-mini fp32 on Tesseract: CER identical (0.354). Phi-4 fp32 clean text: only 1 of 40 pages ran (phase-1 time cap).
- **C2 prompt format:** completion prompt for TituLLMs 1B lowers CER from 1.084 to 0.877 (Tess) and 1.033 to 0.861 (Easy), clean text 0.988 to 0.605 (all p < 0.001), but it stays far worse than raw OCR (+0.54/+0.56). BanglaT5 with the normaliser: no change (0.858 vs 0.860).
- **C4 2x2 decoding (40 pages, vs plain):** guarded raises CER for all four models on both engines and on clean text (Gemma Tess 0.353→0.545, Easy 0.301→0.520, clean 0.032→0.360; Llama Tess 0.380→0.972; Qwen3 Tess 0.337→0.925, clean 0.035→0.940; Phi-4 Tess 0.354→1.740, clean 0.060→1.406). Components differ by model: for Gemma the n-gram ban does almost all the damage (+0.18 Tess) and the penalty little (+0.004); for Phi-4 the penalty is catastrophic (+1.40) and the n-gram ban moderate (+0.16); for Llama and Qwen3 both hurt (+0.35 to +0.49). All 33 tests significant after Holm except Gemma penalty-only on Tesseract (p 0.020, still significant).
- **X1 Grad-CAM (element-wise variant, mid layer, 10 pages):** general models copy 93–99% of output tokens and their maps peak on the copied source token for 58–78% of them; TituLLMs 1B / 1B Instruct / BanglaT5 copy 11–23%, and TituLLMs 1B Instruct relies more on its own earlier output (0.55 of the map) than on the OCR text (0.28). Literal Grad-CAM peaks on the source for only 0–8% of copied tokens. Not run: TituLLMs 3B Instruct and Phi-4-mini (phase-1 time cap).

## First version (August 2026, superseded)

15 randomly sampled evaluation pages (seed 403), CPU (Colab, bf16), four models, decoding with repetition penalty 1.3 and no-repeat-4-gram, August OCR (Tesseract version unrecorded). Baselines: Tesseract CER 0.364, EasyOCR 0.297. All 16 CER/WER deltas positive, 15 significant; Gemma 2B 0.583/0.517, Llama 0.844/0.897, BanglaT5 0.889/0.817, TituLLMs 1B 1.996/2.588. Files: `results/correction_eval.jsonl`, `bootstrap_eval.json`, `extended_eval.{txt,json}`. The decoding ablation above shows that most of this damage came from the decoding constraints.

## Dev set (10 pages): pipeline debugging only, never reported

Run under mixed early settings; every model was already worse than the baseline. Also the source of the three few-shot examples (`data/make_fewshot.py`).
