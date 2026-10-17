# Review feedback: Draft 03

Source: `Review feedback Draft 03.pdf` (Google Drive, Group 2 / Review Feedback), a scanned hardcopy with handwritten marks. Transcribed on 2026-10-15. The reviewer is Bijon Sir. Paper: "Does Zero-Shot Small-Model Post-Correction Help Bengali OCR? A Negative Result Across Eight Models and Two OCR Engines" (Arifur Rahman, CSE403).

Items are numbered: C = critical new issues, W = writing, M = models/experiments, E = metrics/statistics, F = figures/tables, R = references. Locations are section, table and figure labels from the printed draft. The hardcopy has a Tick column to mark each item when done. A handwritten "Grad Cam" appears next to the title. The author confirmed on 2026-10-15 that the reviewer asked verbally for an explainability (XAI) analysis, something like Grad-CAM; it is item X1 below. A handwritten tick is next to E1.

## Current situation (reviewer's summary)
Draft 03 is much stronger than draft 02. Most earlier points are fixed, and the decoding-guard confound is now handled correctly and is the lead contribution. The reviewer cross-checked Tables V, IX, X, XI, XIV and XV against the text (23 of 36 significant tests with 22 increases; 12 and 13 of 18 in the sensitivity checks; the safeguard counts; page-type means reproduce Table V) and found them consistent.

Remaining risks:
1. Possible precision and prompt-format artifacts in the Bengali checkpoints.
2. The whole-page versus region ground-truth mismatch.
3. Draft-history wording.
4. The fine-tuning pilot and noise sweep that would turn a negative result into an explained one.

## Top priorities
1. Check the float16 runs (C1) and the prompt formats of the Bengali checkpoints (C2).
2. Strengthen the decoding-pitfall evidence: 2x2 ablation and the guarded clean-text control (C4), plus a main figure (F1).
3. Region-level OCR and scoring using the PAGE-XML boxes (C5).
4. Remove draft-history language and soften the Qwen3 gating claim (C3, C8).
5. Run the noise sweep (M3) and the small fine-tuning pilot (M2) before submission.
6. Trim to the venue page limit (W2).

## Status of the draft-02 feedback
| Draft-02 item | Status | Note |
|---|---|---|
| Decoding guards (repetition penalty, 4-gram ban) | Done | Main runs use plain greedy decoding; guarded vs plain reported as an ablation (Sec III-D, Table VII). |
| Page-to-ground-truth alignment and reading order | Done | Sec III-C; sensitivity checks (declared reading order, order-free word recall; Sec IV-H, Table IX). Region-level scoring still not done (see C5). |
| Chunk-length contradiction; tokenisation hypothesis | Done | Page-length argument removed; tokens per character and per word measured (Table III). |
| Number mismatch (99.5% vs 100%) | Done | Table XI and Sec V-A agree (74% to 99%). |
| Title versus evidence | Done | Retitled to the actual finding. |
| "First evaluation" claim; Bengali OCR correction literature | Done | Claim narrowed; Pal, Ahmed, BnMMLU, BSpell and others cited (Sec II-B, II-E). |
| Newer models | Partly | Qwen3 1.7B, Phi-4-mini, TituLLMs 1B/3B Instruct added. No Gemma 3, Llama 3.2 3B or larger Qwen3 (see M1). |
| Prompt variants | Done | Preserve-spelling, Bengali and three-shot prompts plus 300-character chunks (Table VIII). |
| Classical baseline | Partly | SymSpell added, but with a modern subtitle lexicon (see M4). |
| Safeguards | Done | Length rule, edit-distance cap, confidence gating (Tables XV, XVI). |
| All 40 pages; book-level bootstrap; grapheme CER; cMER tests | Done | 40 pages scored; books resampled; grapheme-cluster CER and cMER inference added (Tables IX, XI). |
| Runtime, character-class and page-type analyses; qualitative examples | Done | Tables III, XII, XIV and Sec V-B added. |
| Preamble/commentary filter | Partly | Not implemented. Now argued with data (Table X: no Latin text for the copying models), which is acceptable for them but not for the Instruct checkpoints. |
| Model naming (Gemma) | Partly | Checkpoint named in Table III, but text still says "Gemma 2B" (Gemma 1 2B-it). |
| Reproducibility | Partly | Revisions and versions listed. PyTorch/CUDA versions missing; no archive DOI or commit hash. |
| References and Table I layout | Partly | Accent, URL and poster issues fixed. Table I rows still collide; model-card URLs still missing. |
| Fine-tuning pilot; synthetic noise sweep; stronger OCR; upper-bound model | Open | Listed as not performed (Sec VI-E). See M2, M3, M5, M7. |
| Bounded WER; equivalence tests; effect sizes; spelling-variant sensitivity | Open | Not added. See E1-E3. |

## A. Critical: new issues to check first
- [x] **C1** (Sec III-G; Table III; Table X; Sec V-B, TituLLMs 3B Instruct). Possible precision artifact. Phi-4-mini and TituLLMs 3B Instruct ran in float16 on the T4, the others in float32. TituLLMs 3B returned empty output in 27 of 80 cases. float16 overflow can give empty or garbled output. Check for NaN/inf and immediate end-of-sequence, rerun a subset in bf16 or float32 (CPU bf16, 8-bit, or a larger-memory GPU), and report the comparison. Otherwise the "empty page" finding may be an artifact.
  - *Status:* Partly done 2026-10-15: every empty TituLLMs 3B page took 0.2-0.5 s for all chunks (median 14.5 s otherwise), so generation ended at the first token; no NaN-style token runs; Phi-4-mini has no empty output. Disclosed in new Sec IV-J. fp32 rerun still needed (GPU session A).
  - *Status 2026-10-16:* fp32 rerun done. Not a precision artifact: identical first tokens on 208/208 chunks, no overflow (max activation about 1000), same empty pages and CER. Paper text still to update (Sec IV-J).
  - *Status 2026-10-16 (paper):* Done. New Sec IV-J "Precision and prompt format" and Table checks.tex report the fp32 rerun and first-token check; precision caveats removed from III-G, VI-C and the threats.
- [x] **C2** (Sec III-E; Abstract; Sec VI-C, "instruction-following"). Fairness of the Bengali checkpoints. The base TituLLMs 1B was given an instruction chat template it was not tuned for, and the Instruct models often describe the text or return nothing, which may be a prompt-format mismatch. Check each model card's recommended prompt format and run each model with it (and a completion-style prompt for the base model). BanglaT5 without a task prefix is a denoiser, so label it a control. Rename the group "Bengali-pretrained checkpoints" and keep the abstract to what was tested (these checkpoints, this prompt), not Bengali pretraining in general.
  - *Status:* Partly done 2026-10-15: model cards checked. Base TituLLMs template only concatenates text (no answer cue); Instruct cards give no format and the shipped Alpaca template is what we used; BanglaT5 card asks for the csebuetnlp normaliser, which we did not apply. Group renamed, BanglaT5 labelled a control, abstract narrowed. Re-runs with a completion prompt and the normaliser still needed (session A).
  - *Status 2026-10-16:* runs done. Completion prompt helps TituLLMs 1B (CER 1.084→0.877) but it stays far above raw OCR; BanglaT5 normaliser changes nothing. Paper text still to update.
  - *Status 2026-10-16 (paper):* Done. Completion prompt and normaliser reruns in Sec IV-J and checks.tex as a sensitivity result; main numbers unchanged; discussion bullet rewritten. The Instruct checkpoints have no card format, so only their shipped template was tested (stated).
- [x] **C3** (Abstract; Sec III-B, III-C, III-D, III-E, IV-D, VI-C: "our first runs overstated", "first version", "first runs", "our earlier suggestion", "replaces Phi-3"). Draft-history language. A reader sees one paper. Present guarded versus plain decoding as a designed ablation of a common default, delete "our first runs overstated the harm" from the abstract, "our earlier suggestion" from VI-C and the Phi-3 sentence, and shorten the August/October Tesseract story to a reproducibility note. Keep it only if an earlier version is public (then cite it).
  - *Status:* Done 2026-10-15: history sentences removed (abstract, III-B, III-C, III-D, III-E, IV-D, VI-C); guarded decoding presented as an ablation; the August/October Tesseract story is a two-sentence reproducibility note.
- [x] **C4** (Sec III-D "Decoding"; Table VII; Contribution 2). Strengthen the lead contribution. (i) Run a 2x2 ablation (penalty only; n-gram ban only). (ii) Run the guarded setting on the clean-text control (Table VI): any change there is damage, so it demonstrates the artifact most cleanly. (iii) Cite the exact Transformers classes (RepetitionPenaltyLogitsProcessor, NoRepeatNGramLogitsProcessor) and version. (iv) Table VII uses the old Tesseract text and CPU bf16: say so or rerun in the 40-page setup.
  - *Status:* Partly done 2026-10-15: (iii) class names and version cited; (iv) Table VII note and Sec IV-D state the 15-page, August-OCR, CPU bf16 setting. (i) 2x2 and (ii) guarded clean-text control need GPU session B.
  - *Status 2026-10-16:* (i) 2x2 and (ii) guarded clean-text control done on 40 pages for Gemma, Llama, Qwen3, Phi-4 (Tess, Easy for three, clean text). Guarded damages every model; component effects differ by model. Paper Table VII, Fig 4 and Sec IV-D still to update.
  - *Status 2026-10-16 (paper):* Done. Table VII replaced by the 40-page 2x2 (decoding.tex: plain/penalty/n-gram/both on Tesseract, EasyOCR, clean text; 33 Holm tests); 15-page CPU numbers kept only in the table note; Sec IV-D, contribution 1, abstract ("up to 1.4", error-free input), RQ4 and conclusion rewritten.
- [ ] **C5** (Sec III-C; Sec IV-H; Sec VI-E, construct validity). Region-level evaluation is the cleanest fix for the ground-truth mismatch. The PAGE-XML (version 2, improved coordinates) gives region boxes: crop each region, OCR the crops, and score region by region in the declared order. This removes out-of-region text and reading-order errors at the source and lowers input CER, closer to the 0.15 regime of prior work.
  - *Status 2026-10-16:* `ocr/run_region_ocr.py` written and tested locally (8 pages, Tesseract 5.5.3: region CER 0.379 vs 0.547 whole-page with the same engine; psm 6, plain crops). Full run on Kaggle CPU with Tesseract 5.3.4 pending (D41, D42). Region level will be a full second setting (D42).
  - *Status 2026-10-17:* Kaggle CPU run done (D43). EasyOCR region CER 0.225 vs 0.297 whole-page (p 0.002). Tesseract build on Kaggle differs from the main run, so its region OCR is being redone on Colab.
  - *Status 2026-10-17 (later):* Colab gives the same text as Kaggle today, neither reproduces the 11 October main run (D44). Region OCR final: `results/region/ocr_region40.jsonl`; Tesseract compared with its same-run whole-page OCR (0.425 → 0.333). Next: all models on region text (GPU).
  - *Status 2026-10-17:* All eight models and SymSpell run on region text (review3c, D46): no method lowers CER; all significantly worse (`results/region_runs40.txt`). Paper text to do.
- [ ] **C6** (Sec III-C; Sec III-H "Confidence-gated correction"; Sec IV-I). OCR output is not stable: Tesseract changed (CER 0.364 to 0.320) and EasyOCR text matched between CPU and GPU runs on only 5 of 15 pages. Release the exact raw OCR text files and state the OCR hardware. The confidence re-run reproduced only 37 of 40 EasyOCR pages: say how the other 3 pages were gated.
  - *Status:* Partly done 2026-10-15: OCR hardware stated; how the 3 non-reproduced EasyOCR pages were gated is stated (unmatched words count as confident and are never changed). Release of the raw OCR text needs the author's decision.
- [x] **C7** (Table III footnote versus Sec III-G). The note says decoding had no repetition penalty, but the CPU minutes per page come from the earlier guarded runs on the 15-page sample (different Tesseract text, bf16). Re-time under plain decoding or state this; the claim "saves two to fourteen minutes per page" (VI-D) depends on it.
  - *Status:* Done 2026-10-15: checked `model_profile.json`. The CPU times come from plain decoding (`ablation_plain_decoding.jsonl`), not the guarded runs, but on the 15-page sample with August OCR in bf16. Stated in Sec III-G, the new cost table and VI-D. Not re-timed.
- [x] **C8** (Abstract; Sec V-D; Sec VI-A RQ1; Sec VII). The one significant CER gain (Qwen3 1.7B, gating, -0.004, about 1%) holds at one of three thresholds (70), with p = 0.038 after Holm over 54 tests, but not across all analyses. Call it "negligible and not robust", give the relative size, and state how zero differences were handled in the Wilcoxon tests (many gated or capped pages equal the raw OCR). Report the number of non-zero pairs.
  - *Status:* Done 2026-10-15: VI-D/V-F call the gain negligible and not robust; relative size 1.1%; Wilcoxon drops zero differences (stated in III-H and table notes); 36 non-zero pairs (25 better, 11 worse, 4 same); thresholds 50/90 not significant after Holm.

## B. Writing and structure
- [x] **W1** (Abstract). Remove the history sentence (C3). Define "confidence gating" in one clause. Replace "Bengali-specific models generated replacement text" with checkpoint wording (C2). Mention the clean-text control. Keep about 200 words.
  - *Status:* Done 2026-10-15: about 190 words; history sentence removed; gating defined; checkpoint wording; clean-text control mentioned.
- [ ] **W2** (Whole paper). About 16 tables and many pages. Check the venue page limit. Move Tables IX, XIII, XIV and parts of XV/XVI to an appendix; keep Tables I-III, V-VIII, XI and XII in the main text.
  - *Status 2026-10-16:* The author was given no page limit. Paper now 23 pages; trimming waits for a venue.
- [x] **W3** (Sec III-D "Zero-shot"). "Here, zero-shot evaluation..." is still awkward. Define zero-shot in one plain sentence.
  - *Status:* Done 2026-10-15.
- [x] **W4** (Sec III-H). More than a page with nine run-in headings. Split into Metrics, Statistics, Analyses, Safeguards and gating. Define each symbol where it is first used.
  - *Status:* Done 2026-10-15: Sec III-H split into Metrics, Statistics, Analyses, Safeguards and confidence gating; S, D, I, H, N defined at first use.
- [x] **W5** (Intro, Contributions). Merge items 3 and 4 into one analysis contribution. Lead with the decoding pitfall (it is the lead of your conclusion). Trim item 5.
  - *Status:* Done 2026-10-15: decoding pitfall first; items 3 and 4 merged; code item trimmed.
- [x] **W6** (Sec VI-A, RQ1-RQ4). Tie each answer to its table (e.g. RQ1 to Table V and IX, RQ4 to Tables VII and VIII). Shorten RQ4.
  - *Status:* Done 2026-10-15: each RQ cites its tables; RQ4 shortened.
- [x] **W7** (Sec VII Conclusion). Repeats the abstract. Cut it and add one practitioner sentence: keep raw OCR, or gate a copying model's edits by OCR confidence; do not apply repetition penalties to the prompt.
  - *Status:* Done 2026-10-15: conclusion cut to four sentences with the practitioner sentence.
- [x] **W8** (Terminology). "Bengali-specific models" to "Bengali-pretrained checkpoints". "Gemma 2B" to "Gemma-2B-it (Gemma 1)". Use "confidence gating" consistently.
  - *Status:* Done 2026-10-15: "Bengali-pretrained checkpoints" throughout; Gemma-2B-it (Gemma 1) in text, tables and figures.
- [x] **W9** (Sec IV-C). "WER 0.268" for the dictionary is not in any table. Add WER (and wMER) columns to Table VI.
  - *Status:* Done 2026-10-15: WER and wMER columns in the clean-text table.

## C. Models and experiments
- [ ] **M1** (Sec III-E, more models). Add Gemma 3 (1B, 4B), Llama 3.2 3B-Instruct and SmolLM3 3B. Qwen3 gave the only gains, so a Qwen3 size series (0.6B, 1.7B, 4B, 8B) gives a clean scale trend on a T4 (4B in fp16 is about 8 GB). Check Hugging Face for newer releases at submission; the reviewer's knowledge of releases ends mid-2026.
  - *Status 2026-10-17:* Run (review3b, D46): Qwen3 0.6B/4B/8B, Gemma 3 1B/4B, Llama 3.2 3B on 40 pages x 3 inputs. No CER gain; Qwen3 4B and 8B lower WER significantly. Qwen3-32B (M7) and Qwen3 4B on region text in review3d. Paper text to do.
- [ ] **M2** (Sec VI-E (ii), fine-tuning). Do the pilot before submission. Fine-tune BanglaT5, ByT5-small or mT5-small on synthetic noised Bengali text, with noise built from the Tesseract/EasyOCR confusions measured on the 10 development pages. Test on the same 40 pages with the same statistics. A small T4 job answers the "fine-tuning untested" caveat.
  - *Status 2026-10-17:* Code ready (D47): ByT5-small fine-tuned on dev-noise-corrupted Wikipedia + dev text; runs in review3d.
- [ ] **M3** (Sec VI-E (v), noise sweep). Reuse the clean-text control code: corrupt the 40 references at CER 0.02, 0.05, 0.10, 0.20, 0.30, run the four general-purpose models, and plot corrected versus input CER. It directly tests the "input noise" explanation that VI-C says it cannot quantify, and compares with the ~0.15 break-even in [7].
  - *Status 2026-10-17:* Run (review3b, D46): four models x five noise levels (0.02-0.30): every model raises CER at every level; no break-even. Figure `paper/figures/fig_noise.pdf` (F7). Paper text to do.
- [ ] **M4** (Sec IV-F, classical baseline). The OpenSubtitles lexicon is modern and colloquial, a poor match for 1713-1914 literary text. Report the share of reference words missing from the lexicon (a proxy for historical spelling), try edit distance 1, and add a stronger lexicon (e.g. Bengali Wikipedia text). Optional: n-gram or masked-LM rescoring.
  - *Status:* Partly done 2026-10-15: share of reference words missing from the lexicon reported (52%). Edit distance 1 and a stronger lexicon not yet run (local, cheap).
- [ ] **M5** (Sec III-C, OCR input). Add region-crop OCR (C5) and one stronger engine or VLM transcription (bbOCR, Surya, Google Cloud Vision; check Bengali support) to test the findings at lower input CER.
- [ ] **M6** (new, optional). Edit-list or constrained prompting (the model outputs only word replacements), and Tesseract+EasyOCR fusion, as do-no-harm alternatives to free regeneration.
- [ ] **M7** (Sec VI-E (i), upper bound). Run one large open model (30-70B, 4-bit, on a larger GPU) on a page subset to show how much scale alone changes the outcome.

  - *Status 2026-10-17:* Code ready (D47): Qwen3-32B 4-bit on 10 pages; runs in review3d.
## D. Metrics and statistics
- [x] **E1** (Sec III-H; Tables V, IX). Add a bounded word match error rate (wMER); WER reaches 3.03 in Table V. *(Handwritten tick beside this item on the hardcopy.)*
  - *Status:* Done 2026-10-15: wMER in the baseline table, the main matrix (with exploratory Holm marks) and the clean-text table (`eval/review3_analysis.py`).
- [x] **E2** (Sec III-H; Table V). Add equivalence tests (TOST, margin about 0.01-0.02 CER) for Gemma, Qwen3 and Phi-4-mini versus raw OCR, because the claim for them is "no change". Report pages improved/worse out of 40 for every method, not only Qwen3 and Gemma.
  - *Status:* Done 2026-10-15: new equivalence table (TOST at 0.01 and 0.02, 90% CI). Equivalent at 0.02 only for Qwen3/Tesseract and Gemma/EasyOCR; none at 0.01. Pages better/worse added for every method in the main matrix.
- [x] **E3** (Sec V-B "Character classes"; Sec VI-C "Historical spelling"). Gemma's digit error rises from 65% to 75% and 53% to 73% while CER barely moves: check whether it converts Bengali to ASCII digits and add a digit-normalised CER. Add ZWJ/ZWNJ and khanda-ta sensitivity, and correlate per-page out-of-lexicon rate with dictionary-vs-historical-spelling damage.
  - *Status:* Done 2026-10-15: Gemma rewrites Bengali digits as ASCII (88 to 336 on Tesseract, 420 to 640 on EasyOCR); digit-normalised CER changes it by 0.002/0.006. ZWJ/ZWNJ removal changes any CER by at most 0.003; khanda-ta normalisation changes nothing. 52% of reference word tokens are missing from the lexicon; per-page share correlates with words broken (rho 0.38) but not with clean-text CER (rho 0.08).
- [x] **E4** (Table VI). Phi-4-mini pooled CER (0.097) is far above its mean (0.060), unlike the other models. Inspect which pages dominate (truncation, degenerate output on long pages). Also report the guarded setting (C4).
  - *Status:* Partly done 2026-10-15: one page (279_23_D_6_0013) gives 70% of Phi-4-mini's clean-text errors: mid-chunk it switches to English prose and runs to the token limit (1.94x length). In Sec IV-C. Guarded setting on clean text waits for C4.
  - *Status 2026-10-16 (paper):* Done. Guarded setting on clean text in decoding.tex (Phi-4-mini 0.060 to 1.406).
- [x] **E5** (Table IX). Raw Tesseract word recall (0.431) equals its grapheme CER (0.431): check it is not a copy error. Add CIs for word recall.
  - *Status:* Done 2026-10-15: not a copy error. Raw Tesseract word recall 0.4314, grapheme CER 0.4311; equal on no page. Recall CIs added to the robustness table.
- [x] **E6** (Sec III-H, Holm families). You say Holm controls error only within families. Name one primary family (the 36 main tests) and label everything else exploratory in the tables.
  - *Status:* Done 2026-10-15: primary family = 36 CER/WER tests (III-H, Table V note); every other table note says exploratory.
- [ ] **E7** (Sec IV-I, CPU/GPU). Only five EasyOCR pages and only the general-purpose models are compared, and precision also differs (bf16 versus float32). Extend to the Tesseract pages and add the fp16 check from C1.
  - *Status:* Partly done 2026-10-15: the limits are stated (5 EasyOCR pages, no Tesseract pairs exist, precision differs). Extension needs new runs.
- [x] **E8** (Sec III-B, page types). Labels come from one annotator (the author). Add a second annotator for a sample, or state this plainly.
  - *Status:* Done 2026-10-15: stated in III-B and limitations (single annotator). Note: the labels were assigned by the assistant and still need the author's check.

## E. Figures and tables
- [x] **F1** (new figure). Show the repetition-penalty pitfall: guarded versus plain CER per model and engine (from Table VII), plus a small schematic of the penalty acting on prompt tokens. It is your lead contribution and deserves a main figure.
  - *Status:* Draft done 2026-10-15: `paper/figures/fig_decoding.pdf` (schematic + guarded vs plain, 15 pages), `eval/make_decoding_figure.py`. Update after the 2x2 run.
  - *Status 2026-10-16 (paper):* Done. fig_decoding.pdf redrawn from review3_runs.json: four settings per model on Tesseract, EasyOCR and clean text.
- [x] **F2** (Table I). Last rows still collide ("Bhattacharyya & Bhattacharya [31] Bengali grammar correction 1.1B-8B", "correctorsupervised"). Widen or wrap. Verify "GPT-5.5" against [21].
  - *Status:* Done 2026-10-15: ragged-right columns and row spacing; rows no longer collide. GPT-5.5 matches the full-text notes in `reviewer-feedback/paper-reviews.md` (Sec 6); worth one look at the paper itself.
- [x] **F3** (Tables V and XI). Caption "best corrected model per engine" is inaccurate where shading marks the best value per column (including truncation). Reword. Add a note on zero differences in Wilcoxon.
  - *Status:* Done 2026-10-15: shading note reworded (best value per column among corrected methods, not compared with raw OCR); zero-difference note added.
- [x] **F4** (Table III). Too wide. Split into configuration and cost (tokenisation, runtime). Add peak memory if you can record it.
  - *Status:* Done 2026-10-15: split into Table III (configuration) and Table IV (tokenisation and cost). Peak memory still not recorded.
- [ ] **F5** (Figs 4 and 5). Largely redundant: keep Fig 4 in the main text and move Fig 5 to the appendix, or merge.
- [x] **F6** (Table XII). Check that Bengali conjuncts and vowel signs print correctly (embedded Bengali font). Show page type and a longer chunk if space allows.
  - *Status:* Done 2026-10-15 for rendering: conjuncts and vowel signs print correctly under Tectonic; dotted circles come from the text itself (doubled virama), now explained in the note. Page type and a longer chunk not added (space).
- [ ] **F7** (new, from M3). Noise-sweep line plot (corrected versus input CER): likely the most informative new figure.

  - *Status 2026-10-17:* Figure drawn: `eval/make_noise_figure.py` → `paper/figures/fig_noise.pdf`. Not yet in the paper.
## G. Explainability (verbal request)
- [x] **X1** (title note "Grad Cam"; Sec VI-E "Scope of the analysis", which says no attribution method was applied). Add an explainability analysis in the spirit of Grad-CAM. Grad-CAM was built for CNN feature maps; here it is adapted to token positions of the correction models (gradient-weighted hidden states), to show which input tokens each output token relies on: the aligned OCR token (copying), the instruction, or the model's own earlier output (regeneration).
  - *Status:* Script ready and piloted 2026-10-15 (`eval/xai_gradcam.py`; Qwen3 0.6B on CPU, 3 chunks, `results/xai_pilot_qwen3-0.6b.jsonl`). The element-wise variant (HiResCAM-style) at mid/late layers peaks on the copied OCR token for 43-87% of copied tokens (chance about 1-2%). Literal Grad-CAM does not localise: about half its mass is on the attention-sink token. Maps exclude the first/special tokens and the predicting position. Full run for all eight models needs GPU (session A).
  - *Status 2026-10-16:* run for six models (not TituLLMs 3B, Phi-4). Copying models' maps peak on the copied source token 58-78%; Bengali checkpoints copy 11-23%. Literal Grad-CAM fails (0-8%). Table, figure and paper text still to do.
  - *Status 2026-10-16 (paper):* Done for six models. New Sec V-E "What the models attend to", attribution.tex, Attribution paragraph in III-H, Grad-CAM and HiResCAM cited (checked). The full run does not support the pilot's attention-sink explanation for literal Grad-CAM (sink share 0-19%), so the paper gives the averaging argument of HiResCAM instead. Missing: Phi-4-mini, TituLLMs 3B Instruct (next GPU session); no map figure yet (needs --save-maps).

  - *Status 2026-10-17:* All eight models now (Phi-4-mini and TituLLMs 3B Instruct from review3b); full maps of two chunks saved for a figure (`xai_maps_*`), figure not yet drawn.
## F. References and reproducibility
- [x] **R1** ([40], [43]). Model cards: add Hugging Face URLs and revisions.
  - *Status:* Done 2026-10-15: Hugging Face URLs and revisions for Gemma, Llama, Qwen3, Phi-4-mini.
- [ ] **R2** ([49] and Sec III-I). Add a commit hash or Zenodo DOI; record PyTorch/CUDA versions; release the raw OCR texts and all model outputs.
  - *Status:* Partly: versions known so far are listed. Commit hash or Zenodo DOI and the output release need the author.
- [x] **R3** ([24]). Hyphen lost ("languageunderstanding"). Re-run the bibliography for similar breaks.
  - *Status:* Checked 2026-10-15: in the current Tectonic build [24] (BanglaBERT) reads "low-resource language understanding" and no other joined words were found in the extracted text. The break was likely in the reviewer's pdfLaTeX build.
- [x] **R4** (Sec III-D). Cite the Transformers documentation or source for the repetition-penalty and n-gram scope.
  - *Status:* Done 2026-10-15: new reference to the Transformers logits-processor documentation; scope verified in the library source (5.14.1).
