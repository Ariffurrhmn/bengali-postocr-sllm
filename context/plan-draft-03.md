# Plan for the third review (Draft 03)

Written 2026-10-15. Item codes refer to [review-draft-03.md](review-draft-03.md). Status of each item is ticked there; this file holds the cross-check and the order of work.

## What the reviewer wants, in short
- The paper is now judged as a finished study, not a revision. The guarded-versus-plain decoding result is the lead contribution, and it needs its own evidence (2x2 ablation, guarded clean-text control, a main figure).
- The negative result must be fair to each model before it can be explained: rule out a float16 artifact (C1) and a prompt-format mismatch for the Bengali checkpoints (C2). Until then, claims are about "these checkpoints with this prompt", not about Bengali pretraining.
- The disclosed omissions from draft 02 (fine-tuning, noise sweep, cleaner OCR, upper-bound model, bounded WER, equivalence tests) are no longer acceptable as limitations. They are asks: M2, M3, C5/M5, M7, E1, E2.
- Draft-history language goes (C3); the one significant CER gain is "negligible and not robust" (C8).

## Cross-check: top priorities and critical items

| Item | What exists | What is missing | Needs |
|---|---|---|---|
| C1 fp16 | `results/gpu_eval40.jsonl` has the outputs and per-page times. TituLLMs 3B Instruct has 43 empty outputs of 120 (27 on OCR input, 16 on the clean-text control). Every empty page took 0.2 to 0.5 s for all its chunks, against a median of 14.5 s for the non-empty ones, so the model emits end-of-sequence at or near the first token. No output contains a run of "!" (token 0, which greedy decoding picks when logits are NaN). Phi-4-mini has no empty outputs. | Logits were not logged, so a NaN/inf or fp16-specific EOS cannot be ruled out from files. No fp32/bf16 run of either model. | GPU: rerun both fp16 models in fp32 on Kaggle "GPU T4 x2" (fits split over two cards), plus a first-token diagnostic. |
| C2 prompt formats | Each model's shipped chat template (`apply_chat_template`); BanglaT5 gets raw text. | Model-card recommended prompts not checked; no completion-style prompt for base TituLLMs 1B; BanglaT5 not labelled as a control. | Local: read model cards and tokenizer configs. GPU: rerun the three TituLLMs checkpoints with their recommended format (and a completion prompt for the base model). Paper: rename group, narrow claims. |
| C3 history wording | n/a | Abstract, III-B/C/D/E, IV-D, VI-C still narrate the first version. | Paper only. |
| C4 decoding | `results/decoding_ablation.json`: guarded vs plain, 15 pages, August OCR, CPU bf16, 4 models. | Penalty-only and n-gram-only presets; guarded on the clean-text control; guarded on the 40-page October setup; Transformers class names and version. | GPU (2x2 + guarded on 40 pages and clean text). Paper: (iii) and (iv) now. |
| C5 region OCR | PAGE-XML v2 region coordinates in the dataset; `ocr/run_ocr.py`. | Crop-and-OCR script, region-level scoring, and correction runs on region text. | CPU OCR run (Colab/Kaggle, to keep Tesseract 5.3.4; local Tesseract is 5.5.3), then GPU runs. |
| C6 OCR stability | `results/ocr_eval40.jsonl`, `ocr_conf40.jsonl`; versions in `ENVIRONMENT.md`. | Release of raw OCR text; statement of how the 3 non-reproduced EasyOCR pages were gated. | Paper only (the gating code already treats unmatched words as confident, so they are never changed). Release of texts is your call (they contain dataset text, which is public domain). |
| C7 timing | `model_profile.json` CPU minutes come from `ablation_plain_decoding.jsonl`, which is plain decoding, not guarded, but on the 15-page sample, August OCR, Colab CPU bf16. | The paper does not say this. | Paper only (state the setting). Re-timing on the October OCR would need a CPU run; not planned. |
| C8 Qwen3 gating | `gated_gpu40.json`; Wilcoxon uses `zero_method="wilcox"` (zeros dropped). | Non-zero pair counts; relative size; robustness statement. | Local analysis, then paper. |
| P2 F1 figure | Table VII data. | Figure. | Local now (from existing ablation); update after the 2x2 run. |
| P5 M3 / M2 | Clean-text control code path (`engine=ground_truth`). | Noise model from dev-page confusions; noised inputs; fine-tuning script. | Local: noise script. GPU: sweep and pilot. |
| P6 W2 | ~16 tables. | Venue page limit. | Question for you. |

## Groups

### (a) Paper-only edits (now)
C3, C6 (gating statement, hardware), C7, C8 wording, C4 (iii)-(iv), C2 wording (rename group, BanglaT5 as control, narrow abstract), W1, W3-W9, E5 (after the check), E6 labelling, E8 statement, F2-F6, R1, R3, R4, R2 (versions known so far). W2 waits for the page limit.

### (b) Analyses on existing results (now)
E1 wMER (all tables with WER), E2 TOST and improved/worse counts, E3 digit-normalised CER, ZWJ/ZWNJ and khanda-ta sensitivity, lexicon-coverage correlation, E4 Phi-4-mini pooled vs mean, E5 recall CIs, E7 (what the existing CPU/GPU pairs allow), C8 non-zero pairs, C1 empty-output timing table, M4 lexicon coverage and edit distance 1 (SymSpell is local and cheap), F1 preliminary figure.

### (c) New runs on your GPU (scripts to be written; nothing here exists yet)
The Kaggle notebooks clone the public GitHub repo, so each new script has to be committed and pushed before its run. I will prepare them and ask before committing.

Order and rough runtime (Kaggle free tier, about 30 GPU hours a week):
1. **Session A, about 6 h (C1, C2, E7).** fp32 rerun of Phi-4-mini and TituLLMs 3B Instruct on "GPU T4 x2" (40 pages x Tesseract, EasyOCR, clean text; about 3 h), with a first-token diagnostic on every chunk in fp16 and fp32 (NaN/inf, EOS probability). Then the three TituLLMs checkpoints with their model-card prompt format, the base model also with a completion prompt (about 3 h). Should come first: it can change the Bengali-checkpoint numbers in every main table.
   Add X1 (Grad-CAM-style attribution, about 1.5 h): `python xai_gradcam.py --model <key> --engines tesseract,easyocr,ground_truth --n-pages 10 --max-targets 24 --out xai_<key>.jsonl` for each of the eight models.
2. **Session B, about 8-10 h (C4).** Presets `penalty_only`, `ngram_only`, `guarded` on the 40-page October OCR and the clean-text control, for Gemma 2B, Qwen3 1.7B, Phi-4-mini, Llama 3.2 1B, and TituLLMs 1B on Tesseract and clean text only.
3. **CPU job, about 2 h (C5/M5 OCR).** Crop each PAGE-XML region and OCR it with both engines, on a Colab or Kaggle CPU runtime; the notebook first re-OCRs the whole pages and checks they match `ocr_eval40.jsonl`, so the engine versions are proven equal.
4. **Session C, about 6 h (C5).** All eight models on the region-level OCR text (640 runs; the original 960 fitted in one session).
5. **Session D, about 5 h (M3).** Noise sweep: 40 references noised at CER 0.02, 0.05, 0.10, 0.20, 0.30 with the dev-page confusion model, four general-purpose models (800 runs).
6. **Session E, about 6 h (M1).** Qwen3 0.6B, 4B, 8B (8B in 8-bit or on two T4s), Gemma 3 1B and 4B (fp32 on two T4s, since Gemma overflows in fp16), Llama 3.2 3B Instruct. SmolLM3 3B is optional: its card lists no Bengali.
7. **Session F, about 4 h (M2).** ByT5-small fine-tuned on dev-confusion-noised Bengali text, then the 40 pages.
8. **Session G, about 3-4 h (M7).** One 30B-class model in 4-bit on two T4s (Qwen3-30B-A3B is the default), 10-page subset.

Total about 40-45 GPU hours, so about two weeks of free quota.

**Dependencies.**
- C1 and C2 can change TituLLMs numbers in Tables V, VI, X, XI, XII and the abstract. Finish them before the final table pass.
- C5 changes the OCR input. If region-level OCR became the main setting, every downstream table and every model run would have to be redone. Default: keep whole-page OCR as the main, frozen setting, and report region-level as a second setting with its own table.
- M2 and M3 share the noise model; write it once from the 10 dev pages (never the eval pages).
- F1 and F7 depend on sessions B and D; F1 can be drafted now from Table VII.

## Defaults chosen (change any of them)
- Region-level evaluation is a secondary setting, not a replacement (see above).
- M1 list as in session E; SmolLM3 only if time allows.
- M5 stronger engine: Surya (free, open, lists Bengali). Google Cloud Vision is paid, so not used.
- M7: Qwen3-30B-A3B in 4-bit on Kaggle T4 x2, 10 pages.
- E2 equivalence margin: 0.01 CER (stricter end of the reviewer's range), also reported at 0.02.
- E6 primary family: the 36 main CER/WER tests; everything else labelled exploratory.
- E8: state plainly that one annotator labelled page types; a second annotator only if you can find one.
- The handwritten "Grad Cam" is an XAI request (confirmed by the author): item X1, a token-level Grad-CAM adaptation for the correction models.
