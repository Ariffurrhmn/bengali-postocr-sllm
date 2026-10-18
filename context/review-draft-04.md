# Bijon Sir's corrections on draft 04 (received 2026-10-18)

Source: `C:/Users/Windows 11/Documents/Bengali_PostOCR_draft04_corrections.pdf` (2 pages). Decisions in D60.

## A. Corrections
- [x] **C1** Float32 TituLLMs 3B: text now says at most 0.002 on OCR text, -0.040 (n.s., CI [-0.132, +0.032]) on clean text.
- [x] **C2** Repetition constraints checked on Transformers 5.19.0: `eval/check_prompt_constraints.py`, `results/prompt_constraints_5.19.0.json` (Qwen3 0.6B, CPU): penalty changes all 42 distinct prompt ids and no other; n-gram ban gives -inf to the token completing a prompt 4-gram. 5.19.0 source: only `RepetitionPenaltyLogitsProcessor(prompt_ignore_length=...)` built by hand excludes the prompt; the n-gram ban has no such option. Same result on 5.14.1.
- [x] **C3** Fusion CER gain called descriptive, not significant (text IV-N, RQ1, gating paragraph); table note gives both Holm p-values from `results/m6_fusion40.json` and the new `results/m6_fusion40_family4.json`; the significance mark on fusion CER removed. Abstract unchanged ("about 1% at best").
- [x] **C4** Deferred by Sir (D61): no extra GPU now; he will arrange GPU runs after the external review if needed. The paper already calls it a pilot that saw about 14,300 of 43,312 pairs once (IV-L).
- [x] **C5** Region crop settings: the methods now state that the 8 evaluation pages influenced the choice (Sir's second option); no rerun (D61).
- [x] **C6** Region EasyOCR CI: text now uses the same-run comparison, [-0.120, -0.033], as the table.
- [x] **C7** Completion-prompt p-values: text says unadjusted p < 0.001 in each case (1.9e-6, 4.5e-6, 8.1e-10); table note adds the same.
- [x] **C8** "Decoding." starts a new paragraph.
- [x] **C9** Abstract: "by up to 1.4 (absolute CER)"; trimmed to 208 words.
- [x] **C10** Phi-4-mini region EasyOCR: three truncated pages give 87% of its increase (one page +1.35); median +0.007; worse on 30/40 (`eval/page_deltas.py`, `results/region_phi4-mini_easyocr_pages.json`). Stated in IV-K.
- [x] **C11** Gemma noise range: unrounded minimum is 0.0158, so 0.016 is right; text now says the ranges come from unrounded values.
- [x] **C12** Leptonica/CPU flags of the drifting runs were not recorded (stated); `ocr/run_ocr.py` and `ocr/run_region_ocr.py` now record the full `tesseract --version`, CPU model and flags, and the model's sha256.

## B. Suggested additions (deferred by Sir, D61: no new experiments or GPU before the external review)
- [ ] S1 vision-language baseline (Gemma 3 4B direct transcription)
- [ ] S2 second collection
- [ ] S3 word-search (retrieval) test
- [ ] S4 trim / supplement (paper is now 29 pages)

## C. Target journal
- [ ] J1 LRE (Springer), J2 IPM (needs S3), J3 fallbacks. Author to confirm which index the university counts.
