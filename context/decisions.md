# Decision log

Each entry gives the decision, why it was made, what else was considered, and what follows from it. Status is **current** unless marked otherwise. Dates are 2026.

---

## Framing

### D01. Research question: zero-shot, engine-agnostic, CPU-only
**Decision:** Test whether a small open-weight LM, used purely as a post-processing layer with no training or fine-tuning, reduces CER/WER on Bengali OCR output, and whether the effect is consistent across two OCR engines.
**Why:** The target users (archives, government offices, NGOs in Bangladesh) typically lack GPUs. A correction layer that bolts onto any engine without retraining would be the cheapest possible fix if it worked. No prior work tested small-model post-OCR correction for Bengali, or engine-independence for any language.
**Consequence:** Every later choice (free Colab CPU, no fine-tuning, one fixed prompt) follows from keeping the setup this cheap and portable.

### D02. Base paper: Kanerva et al. 2025, "No Free Lunches"
**Decision:** Use Kanerva, Ledins, Käpyaho & Ginter (RESOURCEFUL-2025, arXiv:2502.01205) as the methodological template. Their code: github.com/TurkuNLP/ocr-postcorrection-lm.
**Why:** It is the closest precedent: zero-shot open-weight LLMs, historical OCR, CER/WER, and a cross-lingual comparison. Its key finding is directly relevant: open models improved English but made Finnish (the lower-resource language) *worse*. That makes Bengali a genuinely open question.
**What we took from it:** NFC normalisation before scoring (D21); ~200–300 subword segments (D19); character-level alignment to strip overgeneration (D23, **not implemented**).
**Verified facts from the paper (don't re-guess):** dev/test are 200 *segments* per language (~300 subwords each), not whole pages. English improvements 7.3% (Llama-3-8B) to 58.1% (GPT-4o). Finnish was negative for every open-weight model (−19% to −76%); only GPT-4o improved it (+11.9%).

### D03. No pass/fail threshold; report the effect as measured
**Decision:** Drop the proposal's "≥10% CER reduction" hypothesis (decided 2026-07-25) and measure how much, if any, improvement occurs.
**Why:** The 10% figure was an uncited assertion with no grounding. Small models aren't comparable to the larger models in prior work, so any improvement would be meaningful, and a null or negative result is equally reportable.
**Note:** The proposal PDF still states the 10% hypothesis. The paper doesn't use it.

### D25. Report as a negative result
**Decision:** Frame the paper around the finding that correction hurts, with a failure analysis explaining how.
**Why:** The result is decisive (16/16 worse, 15 significant) and consistent with the base paper's Finnish finding and with Levchenko 2025. A clean negative result with a mechanism is a contribution; hiding it or chasing a positive result by tuning on the eval set would not be.

---

## Data

### D04. Dataset: British Library historical Bengali print (ICDAR 2019 REID)
**Decision:** Use "Ground truth transcriptions for training OCR of historical Bengali printed texts" (British Library / Two Centuries of Indian Print, with PRImA and Jadavpur University; REID2019 competition). Books printed 1713–1914. https://bl.iro.bl.uk/concern/datasets/bb125e50-4a09-46e8-904b-6e46cf308d91
**Why:** Mirrors the base paper's historical-print design. Page images plus PAGE-XML ground truth give page-level CER/WER. Images are out of copyright and transcriptions are public domain.
**Alternatives:** bbOCR / BCD3 (~88.5k words) as fallback. Rejected BaDLAD (layout annotation only, no transcriptions) and BN-HTRd (handwritten, different problem).
**Consequence:** Results describe old printed material (ornate type, verse, title pages) and may not transfer to modern documents. This is a stated limitation.

### D05. Exclude the 29 pages with no ground-truth text
**Decision:** Of the 81 image + PAGE-XML pairs, 29 (36%) have region boxes but empty `<Unicode>` transcriptions. Exclude them, leaving 52 usable pages.
**Why:** You can't compute CER/WER without a reference. Found with `data/check_transcriptions.py`; list in `data/excluded_no_ground_truth.txt`.
**Consequence:** The eval pool is smaller than planned, so confidence intervals are wider.

### D06. 50-page pool: drop the 2 pages with the worst region coverage
**Decision:** From the 52 usable pages, drop `14053_C_15_0026` (83% of text regions transcribed) and `14123_c_2_0169` (67%).
**Why:** A page where many regions are untranscribed is a weaker reference than a short but complete one. Region coverage measures this directly; see `data/analyze_thin_transcriptions.py`.
**Superseded alternative:** The first version dropped the two *shortest* pages (`14028_D_2_0004`, `14048_D_11_0003`). Those are complete, just short, so they were put back.

### D07. 10 dev / 40 eval split, frozen before any model ran
**Decision:** Split the 50 pages 10/40 with `data/make_split.py` (seed 403). Committed as b445580 before any correction model was run.
**Why:** Freezing the split and committing it first means no eval result could have influenced which pages were used. The dev set is only for debugging the pipeline and is never used for reported numbers.
**Superseded:** An earlier plan of 15 dev / 66 eval assumed all 81 pages had ground truth (see D05).

---

## OCR stage

### D08. Two engines: Tesseract and EasyOCR, default settings
**Decision:** Run both engines on every page with default settings. Tesseract uses the `tessdata_best` Bengali model, kept in a project-local `.tessdata/` directory.
**Why:** Two engines with different error profiles are the minimum needed to test engine-independence. Default settings avoid engine-specific tuning, which would contradict the engine-agnostic claim. `tessdata_best` was chosen for accuracy over speed. The local `.tessdata/` exists because the local machine has no admin rights to add langpacks system-wide; it also makes setup reproducible (SETUP.md).
**Consequence:** EasyOCR beat Tesseract here (eval CER 0.297 vs 0.364), the reverse of bbOCR's modern-document comparison. The likely reason is historical type.

---

## Correction stage

### D09. The five models and why each
| Key | HF ID | Role |
|---|---|---|
| phi3-mini | microsoft/Phi-3-mini-4k-instruct | largest general sLLM (3.8B). **Not run**, see D20 |
| llama3.2-1b | meta-llama/Llama-3.2-1B-Instruct | general sLLM, same scale as TituLLMs |
| gemma-2b | google/gemma-2b-it | general sLLM, 2B |
| titullm-1b | hishab/titulm-llama-3.2-1b-v1.1 | Bengali-pretrained Llama 3.2 1B |
| banglat5 | csebuetnlp/banglat5 | Bengali seq2seq (~248M), "natural" text-to-text baseline |

**Why:** A mix of general multilingual models and Bengali-specific ones. TituLLMs 1B sits at the same parameter count as Llama 3.2 1B, so comparing them isolates the effect of Bengali pretraining. Llama and Gemma are gated on HuggingFace (licence accepted, read token used).
**Note:** The TituLLMs ID was checked with a HuggingFace search; the first guessed ID was wrong.

### D10. One fixed zero-shot prompt, identical everywhere
**Decision:** The same instruction for every model and engine (`CORRECTION_INSTRUCTION` in `correction/models.py`). It asks for corrected Bengali text only, with no explanation or preamble. It was specified in advance and never revised in response to output.
**Why:** Any difference in outcome is then attributable to the model or engine, not to prompt engineering. Tuning the prompt against eval output would also leak eval information.
**Trade-off:** The results describe *this* prompt, not the best achievable prompt per model. Kanerva et al. show that prompt and segmentation matter. This is a stated limitation.

### D11. BanglaT5 gets raw text, no instruction
**Decision:** BanglaT5 is fed the OCR text directly (no chat template, no instruction).
**Why:** It is not instruction-tuned and has no chat template, so an English instruction would just be more input text.
**Consequence:** Its numbers describe this integration, not the model's ceiling on the task. It was never fine-tuned for correction. This is a stated limitation.

### D12. Greedy decoding with repetition guards
**Decision:** `do_sample=False`, `repetition_penalty=1.3`, `no_repeat_ngram_size=4`.
**Why:** Correction isn't creative generation, and determinism is needed for reproducibility. The guards were added after TituLLMs looped one phrase ~40 times until it hit the token cap (2026-07-26). They are deterministic, so decoding is still greedy.
**Caveat:** A repetition penalty can in principle discourage correctly copying repeated words from the input. This was not measured.

### D13. max_new_tokens scales with input length
**Decision:** `min(512, max(64, 1.5 × input tokens))`.
**Why:** A flat 512 cap let instruction models run long on every page. This was first blamed for the slowness, but the real cause turned out to be D15. It was kept anyway because a correction should be about as long as its input, and the cap bounds worst-case runtime.
**Consequence:** "Truncated" in the results means the output hit this cap. Truncation rate correlates with CER at r = 0.84 across the 8 conditions.

### D14. bfloat16 for all models
**Decision:** Load every model in bf16.
**Why:** Phi-3 at fp32 (~7.6 GB of weights) was killed at ~70% of weight loading on free Colab RAM (~12–13 GB). bf16 halves the memory and, unlike fp16, is numerically stable on CPU. It was applied to all models so precision isn't a confound.
**Caveat:** The early *dev-set* runs of TituLLMs and BanglaT5 were fp32 and single-threaded. All *eval-set* numbers are consistently bf16 and multithreaded.

### D15. Force PyTorch to use all CPU cores
**Decision:** `torch.set_num_threads(os.cpu_count())` at import, in `correction/models.py`.
**Why:** On Colab, `torch.get_num_threads()` returned 1, giving ~1 s/token. This was the real root cause of the "stuck" and slow pages, found with `correction/diagnose_speed.py` (removed 2026-10-12; in git history). After the fix: 2 threads.

### D16. Skip near-empty OCR input
**Decision:** If OCR text is under 10 characters, return it unchanged and mark `skipped`.
**Why:** With almost no signal, the model confabulates. No page in the eval sweep triggered it (0 skipped).

### D17. All model runs on free Colab CPU; local machine for code only
**Decision:** Develop and test locally, but run every model on a free Colab CPU runtime.
**Why:** CPU-only matches the deployment claim (D01). The local C: drive also had under 1 GB free, not enough for even one model download.
**Consequence:** Runs took 2–26 minutes per page. This drove D19 and D20.

### D18. Crash-proof sweeps: per-line flush, resume by skip, results on Drive
**Decision:** `run_sweep.py` and `run_ocr.py` append and flush one JSON line per result. On restart they skip any (page, engine, model, approach) already written. Colab writes these files to Google Drive, not the VM disk.
**Why:** Free Colab disconnects (idle timeout, session caps). Flushing plus resume covers disconnects and per-page errors; writing to Drive covers a full runtime reset. The eval sweep did disconnect at 84/120 and resumed with nothing lost.

---

## Evaluation scope

### D19. Final eval: 15 of the 40 eval pages, chunked, 4 models, both engines
**Decision:** Randomly sample 15 eval pages (seed 403, `run_ocr.py --sample-n 15`). Run the chunked approach only. Chunks are ~250 tokens of the model's own tokenizer, split on line boundaries so no chunk cuts a line (`correction/chunking.py`). That gives 15 × 2 engines × 4 models = 120 runs.
**Why 15 pages:** Using the dev-set effect sizes (+0.29 CER at the smallest), the estimated 95% CI half-width at n = 15 was about ±0.11–0.13. That is far too small to overturn "correction makes it worse", so 15 is decisive, not just convenient. The estimated runtime was ~24 h for chunked, against ~7 h for whole-page.
**Why chunked:** To stay faithful to the base paper's segment-level method.
**Known tension:** On the dev set, chunked did *worse* than whole-page for TituLLMs (CER 2.8–6.7 against 1.6–3.4). Possible reasons: our chunking is simpler than Kanerva's (no boundary-context methods), and our pages may already be close to or below one chunk in length. The whole-page code path still exists but was not run on eval.
**Eval page IDs:** see [results.md](results.md).

### D20. Phi-3 Mini excluded
**Decision:** Leave Phi-3 out of the reported results and state this as a compute limitation.
**Why:** It never completed a run: once killed for lack of memory (before D14), then too slow to finish a page. The four completed models averaged 137–837 s per page per engine, with a maximum of 1568 s. A 3.8B model would have cost days before the deadline, for a data point unlikely to change a conclusion that held across all four models. The user decided on 2026-07-27.

---

## Scoring and analysis

### D21. CER/WER: NFC + whitespace collapse, macro-averaged per page
**Decision:** Normalise reference and hypothesis to Unicode NFC and collapse all whitespace runs before jiwer CER/WER (`eval/metrics.py`). Compute per page, then average across pages so each page counts equally.
**Why:** Bengali vowel signs and nukta letters can be encoded as different codepoint sequences that look identical. Line-break differences between OCR and ground truth aren't transcription errors.
**Note:** Only vowel signs O (U+09CB) and AU (U+09CC) and the nukta letters RRA/RHA/YYA actually decompose in Bengali. The tests (`eval/test_metrics.py`) use these, built from `chr()` codepoints.

### D22. Paired bootstrap, 10,000 resamples, 95% percentile CI
**Decision:** For each engine × model × metric, resample pages with replacement 10,000 times (seed 403) and report the mean delta (corrected − baseline) with a 95% percentile interval. Significant = the interval excludes 0 (`eval/bootstrap.py`).
**Why:** Pairing cancels page-to-page difficulty. It was chosen over Wilcoxon because it gives an effect size with an interval, not just a p-value.

### D23. Overgeneration alignment filter: specified, not implemented
**Decision:** Kanerva-style character-level local alignment to strip leading and trailing non-corrective text (their Section 4.3, biopython pairwise aligner) was planned but not built. The paper discloses it as a protocol deviation (Section III-F) and a limitation (Section VI-D). The README claim was removed in 839a772.
**Why disclosed rather than hidden:** It affects models unequally. TituLLMs emits long English commentary and would benefit most.
**Counter-argument recorded in the paper:** Gemma 2B emits almost no preamble and still degrades CER from 0.364 to 0.583, so the omission doesn't explain the overall result.

### D24. Failure diagnostics beyond CER
**Decision:** For each model, measure output length relative to the reference, script composition (Bengali / Latin / other), ground-truth token retention (share of reference word tokens present in the output), truncation rate, and side-by-side examples (`eval/inspect_examples.py`).
**Why:** CER says correction fails but not how. These show the mechanism: models discard text the engines had already got right (retention falls from 46–55% to 2–13%). BanglaT5 collapses (0.34× length); TituLLMs runs away (2.58×, 34% Latin, English commentary).

### D26. Run outputs are not committed
**Decision:** `results/` is gitignored apart from `.gitkeep`.
**Why:** The outputs are generated and regenerable, and the raw output JSONL is large.
**Risk:** Regenerating the eval sweep costs ~24 h of Colab time, so it is not cheap. The only copies are the local `results/` and Drive. See open-items.

---

## Post-feedback analysis (October 2026)

### D27. Extended analysis on existing outputs instead of new model runs
**Decision:** Answer the reviewer's "statistical test / validation / XAI / contribution" comments first with analyses on the existing eval outputs (`eval/extended_analysis.py`): Wilcoxon + Holm, cMER, preference score, word fix/damage transitions, change ratio and edit operations, a post-hoc length safeguard with published thresholds, and difficulty bands.
**Why:** Every method comes from a 2024–2026 published paper reviewed in `reviewer-feedback/paper-reviews.md`. They cost minutes, need no Colab, and add evidence without touching the frozen protocol. New model runs (conservative prompt, few-shot, shorter chunks, bigger quantized model) stay optional.
**Rules kept:** safeguard thresholds were taken as published (0.35×–2.8×), not tuned on eval pages. Holm is applied per family (16 CER/WER tests; 8 cMER tests; 8 safeguard tests).
**Result:** the conclusion holds under every added test (see results.md).

### D28. Paper figures and tables are generated, not hand-made
**Decision:** The methodology diagram, dataset figure and all result tables are produced by scripts (`eval/make_paper_figures.py`, `eval/make_paper_tables.py`) into `paper/`. The literature table (`paper/tables/literature.tex`) is hand-written but every cell was checked against the source paper.
**Why:** No number can drift from the result files. Regenerating after any re-analysis is one command. Captions and all paper prose stay with the author (AI-similarity rule).
**Notes:** The dataset figure reads its counts from `data/*.txt` and `results/ocr_eval.jsonl`. The sample page is 279_42_B_41_0003 (median difficulty; the same page as `examples_*.txt`). The "best" highlight marks the best *corrected* model per engine; the uncorrected baseline still beats every model.

### D29. Revised paper text written by the assistant at the author's request (2026-10-05)
**Decision:** `paper/main.tex` is a full revised draft that responds to the reviewer's comments (`reviewer-feedback/feedback.md`): new abstract, section openings, contribution list, numbered research questions, explanation of zero-shot, justified model choice, new figures and tables, 2025-2026 references, shortened limitations and conclusion.
**Why:** The author asked for the text to be written after the assistant had flagged the course's AI-similarity limit. This is the author's decision and risk.
**How it was kept honest:** every number was cross-checked against `results/`; claims about cited papers were checked against full-text reviews (`reviewer-feedback/paper-reviews.md`); unverifiable items are listed in `paper/CHECK-BEFORE-SUBMITTING.md`.

### D30. Revision experiments run on a free T4 GPU, on Kaggle (2026-10-10)
**Decision:** The 40-page revision runs use a free T4 GPU, on Kaggle (`notebooks/kaggle_gpu_run.ipynb`; Colab T4 is the fallback): float32 for models that fit, float16 for those above ~3B (Phi-4-mini, TituLLMs 3B Instruct). This changes D17 (all model runs on free Colab CPU). OCR still runs on a CPU runtime.
**Why:** Bijon Sir's review item M5: CPU-only is a deployment claim, not a reason to limit the experiment; use a free GPU and report CPU latency separately. On CPU the remaining runs would take about 40 hours. Kaggle was chosen over Colab because its free GPU quota (about 30 h a week) is more predictable and its background runs (up to 12 h) need no open browser; the notebook stops starting new pages after 11 h so the output is saved. Float16 is avoided where possible because Gemma is known to overflow in it, and the T4 has no native bfloat16.
**Consequence:** Main results come from one consistent GPU setup on all 40 pages. The original 15 CPU pages become a CPU-vs-GPU cross-check, and CPU per-page times from those runs are reported separately. The decoding ablation stays on the 15 CPU pages.

### D31. Models added in revision: Phi-4-mini and the TituLLMs Instruct checkpoints (2026-10-10)
**Decision:** Add `microsoft/Phi-4-mini-instruct` (3.84B), `hishab/titulm-llama-3.2-1b-v2.0-Instruct-v1.0` (1.32B) and `hishab/titulm-llama-3.2-3b-v2.0-Instruct-v1.0` (3.34B) to the 40-page GPU run, alongside the five existing models (including Qwen3 1.7B, added on 2026-10-10).
**Why:** Review items M5 (run Phi-3 or Phi-4-mini; Phi-4-mini is the successor, so Phi-3 Mini is not run) and E-b (TituLLMs 3B, Bengali-specific alternatives). While checking, we found that the TituLLMs checkpoint used so far, `titulm-llama-3.2-1b-v1.1`, is a continually pretrained **base** model (base_model: meta-llama/Llama-3.2-1B), not instruction-tuned, and its vocabulary (128,256) equals Llama's, so the paper's "extended tokenizer" claim is wrong for it. The v2.0 Instruct checkpoints are instruction-tuned, use an extended vocabulary (170,497) and an Alpaca-style chat template. Adding them in the same run avoids a second GPU session.
**Consequence:** The paper must describe TituLLMs 1B v1.1 as a base model, and the Llama-vs-TituLLMs comparison as base-vs-instruct, not a clean test of Bengali pretraining.

### D32. Prompt and chunking variants (2026-10-11)
**Decision:** Test three prompt variants and one chunking variant on the four general-purpose models (Gemma 2B, Qwen3 1.7B, Phi-4-mini, Llama 3.2 1B), 40 pages, both engines, plain decoding, on a Kaggle T4 (`notebooks/kaggle_variants_run.ipynb`): `en_preserve` (the original prompt plus "preserve the historical spelling; if unsure, leave it unchanged"), `bn` (the original instruction in Bengali), `fewshot` (the original prompt plus three worked examples), and `chunked_chars` (the original prompt with chunks of at most 300 characters, identical for every model). The original prompt (`en`) is unchanged, so earlier results stay valid.
**Why:** Review items M4 (only one English prompt tested; try a Bengali prompt, the preserve-spelling instruction, 1–3 shots) and M3 (token-based chunking gives each model different chunks: 168 to 483 characters on average). BanglaT5 and the TituLLMs models are left out: they fail by regenerating or looping, which a prompt wording is unlikely to fix, and leaving them out keeps the run within one background session.
**How the few-shot examples were chosen:** `data/make_fewshot.py`, deterministically from the development pages only (no evaluation page): one line per page, engines alternating Tesseract/EasyOCR/Tesseract, reference line of 30–90 characters whose best-matching OCR line has CER 0.05–0.30 and no Latin letters. Pages 14048_c_24_0185, 14123_h_3_0005, 14129_b_1-12_00028. The examples file is dataset text, so it is gitignored and uploaded to the private Kaggle dataset.
**Note:** the Bengali prompt was written by the assistant and checked by the author (a native speaker) on 2026-10-11.

### D33. Confidence-gated correction from existing outputs (2026-10-12)
**Decision:** Implement review item E4 without new model runs. Re-run both OCR engines on the 40 pages with confidences (`ocr/run_ocr_conf.py`, Colab CPU, same Tesseract 5.3.4 / EasyOCR 1.7.2 setup as `ocr_eval40.jsonl`), then gate the existing plain-decoding outputs (`eval/confidence_gate.py`): a model edit is kept only if every OCR word it replaces or deletes has confidence below the threshold, and an insertion only next to such a word. Thresholds fixed in advance: Tesseract 50/70/90 (per word), EasyOCR 0.5/0.7/0.9 (per text box, the finest EasyOCR reports). Confidences are mapped onto the stored OCR words by word alignment; stored words without a counterpart are treated as confident (never changed).
**Why:** E4 asks to correct only low-confidence words. The models already corrected whole chunks with context, so gating their edits tests the idea at no GPU cost. The alternative, prompting the model with only the uncertain words, would need new runs and changes what the model sees.

### D34. Second review: fixes without new model runs (2026-10-12)
**Decision:** Respond to the second review of the compiled PDF (`After feedback.pdf`) with existing outputs only:
- Fail the build loudly under pdfLaTeX (the submitted PDF had placeholders instead of Bengali).
- Load the Times font by file name so Overleaf, TeX Live and Tectonic all compile.
- Redesign Table III. Wrap the checkpoint IDs, add tokens/word, and add GPU weight memory computed from parameters × bytes. Peak memory was not recorded, and the paper says so.
- Pool script shares over characters, and report empty outputs separately.
- Add p-values to the length rule and the edit-distance cap (new safeguards table).
- Add a robustness table: pooled CER, grapheme CER, book CI, reading-order reference, word recall.
- Label page types from the transcriptions (`data/page_types.tsv`) for E7/T1.
- Add a corrected-vs-baseline scatter with the identity line (E3).
- Narrow the novelty claim and cite Ahmed et al. 2016 and BnMMLU (Findings ACL 2026).
- Drop the claim that the region mismatch cancels in paired differences.
- Qualify the noise, tokenisation and "at least as good" statements.
- Name the five experiments not performed in the limitations.
- Commit numeric summaries (`summaries/`) and an environment record (`ENVIRONMENT.md`).

**Why:** Every item in the review that could be done without a GPU. The remaining items (E-c, E-d, E-e, E-f, E-g, E2, region-level evaluation, peak-memory measurement) need new runs, so they stay deferred and are disclosed.

**Note:** this changes D26 only for numeric summaries. Per-page outputs, which contain transcribed text, stay out of git. The page-type labels were assigned by the assistant from the transcriptions and need the author's check.

### D35. Repo cleaned to the current study; compiled paper committed (2026-10-12)
**Decision:**
- Remove superseded files: the first-version figure script (`eval/make_figures.py`; its labels moved into `make_revision_figures.py`), the one-off `correction/diagnose_speed.py`, `paper/archive/`, and `SETUP.md`, whose setup steps are folded into the README.
- Move the reviewer-checklist status to `context/checklist-status.md`, so it is public with the rest of the log.
- Commit the compiled `paper/main.pdf`, so readers need not compile the paper.

**Why:** The author asked for a repo that holds only up-to-date material, and people will check it. Everything removed is still in git history.

### D36. Third review: paper edits and analyses on existing outputs first (2026-10-15)
**Decision:** Answer the Draft 03 review (`context/review-draft-03.md`, plan in `context/plan-draft-03.md`) in three groups: paper-only edits and analyses on existing outputs now; new GPU runs (C1, C2, C4 2x2, C5, M1-M3, M7) prepared next and run by the author on Kaggle.
- New analyses in `eval/review3_analysis.py` (output `results/review3_gpu40.{json,txt}`, copied to `summaries/`): wMER, TOST equivalence, page counts, non-zero Wilcoxon pairs, digit/ZWJ/khanda-ta normalisation, lexicon coverage, clean-text pooled-CER pages, recall CIs, empty-output timing.
- Primary statistical family: the 36 CER/WER tests. Every other family is labelled exploratory.
- Equivalence margins 0.01 and 0.02 CER, fixed before looking at the TOST results.
- Table III split into configuration (`models.tex`) and cost (`cost.tex`); new `equivalence.tex`; new decoding figure (`eval/make_decoding_figure.py`).
- Gemma is named Gemma-2B-it (Gemma 1); the TituLLMs/BanglaT5 group is "Bengali-pretrained checkpoints"; BanglaT5 is reported as a control.
- Draft-history wording removed from the paper; the August OCR is mentioned only as a reproducibility note and as the input of the decoding ablation.

**Why:** These items need no GPU and several of them (wording, C8, E6) change how the existing numbers are read. The GPU runs can change the Bengali-checkpoint numbers (C1, C2), so the final table pass waits for them.

**Findings that changed the text:** "no change" for the general-purpose models is supported by equivalence only for Qwen3/Tesseract and Gemma/EasyOCR at 0.02; Phi-4-mini's wMER rises significantly on both engines; the TituLLMs 3B empty outputs end at the first token (precision not ruled out); the base TituLLMs template only concatenates text; the BanglaT5 card requires a normaliser that was not applied; CPU times are plain decoding but on the August 15-page input (the reviewer's C7 assumed guarded runs).

### D37. Explainability (X1): Grad-CAM adapted to token positions (2026-10-15)
**Decision:** Answer the reviewer's "Grad Cam" request with gradient-weighted maps over token positions of the correction models (`eval/xai_gradcam.py`), not Grad-CAM on the OCR engines. Score = logit of each generated token (greedy, plain decoding, same prompts and chunks as the main runs); maps at the embeddings and at 1/2 and 3/4 depth, both literal Grad-CAM (gradients averaged over positions) and the element-wise variant (gradient x activation per position, as in HiResCAM). The first token, special tokens and the predicting position are excluded and their shares recorded. Measures: share of the map on the OCR text, the rest of the prompt and the model's own earlier output; for copied tokens, whether the map peaks on the source token (+-1).
**Why:** The paper's question is what correction does to the text, and its central contrast is copying versus regeneration; attribution over input tokens tests that directly. Grad-CAM on the OCR CNN would explain recognition, which the paper does not study.
**Pilot finding:** literal Grad-CAM is dominated by the attention-sink token and does not localise; the element-wise variant does (peaks on the copied source token for 43-87% of copied tokens on three Qwen3 0.6B chunks). Local CPU runs crashed for lack of memory until the score was computed from one position instead of full logits.


### D38. First GPU session after the third review: 10.5 h of quota, two GPUs (2026-10-15)
**Decision:** Spend the remaining 10 h 34 min of the week's Kaggle quota (reset 2026-10-17) on one "GPU T4 x2" session (`notebooks/kaggle_review3_run.ipynb` → `correction/run_review3_kaggle.py`), with 9.25 h of jobs. Phase 1 uses both GPUs for the float32 runs (C1: first-token check in fp16 and fp32, TituLLMs 3B Instruct on all inputs, Phi-4-mini on Tesseract and clean text, X1 for both). Phases 2-3 run two lanes, one per GPU: C2 (TituLLMs 1B completion prompt, BanglaT5 with the csebuetnlp normaliser), X1 for the other models, and C4 (guarded, penalty-only, n-gram-only on Tesseract and clean text; EasyOCR if time is left).
**Why:** Kaggle counts session time, not GPUs, so two lanes roughly double the work per quota hour. C1 and C2 can change the main tables and come first; X1 is the reviewer's explicit request; C4 fills the rest. Region-level OCR, the noise sweep, new models, fine-tuning and the large model wait for the reset.
**Settings:** transformers pinned to 5.19.0 (as in the main runs); precision and placement set through `POSTOCR_DTYPE` / `POSTOCR_DEVICE_MAP` (`models.load_model`); every output line now records its dtype; PyTorch/CUDA versions are written to `environment.txt` (R2).
**Not tested locally:** float32 placement across two GPUs and the BanglaT5 normaliser (no GPU, and the BanglaT5 tokenizer does not load on the local machine). Everything else was smoke-tested on CPU with Qwen3 0.6B.

### D39. First third-review GPU session: outcome (2026-10-16)
**What ran:** the full plan of D38 except the fp32 clean-text run of Phi-4-mini (1/40 pages) and X1 for TituLLMs 3B Instruct and Phi-4-mini: phase 1 hit its 3.5 h cap, because fp32 split over two T4s is slow. Kaggle quota left afterwards: about 40 min.
**Consequences for the paper:** (1) C1 is settled: the empty TituLLMs 3B outputs are the model's behaviour, not float16; the precision caveat in Sec IV-J, III-G and the limitations can be replaced by the check. (2) C2: the prompt format matters for the base TituLLMs 1B (CER lower by about 0.2) but does not change any conclusion; report it as a sensitivity result. (3) C4 strengthens the lead contribution on the 40-page setup: guarded decoding damages every general model on OCR and on clean text, and which half does the damage depends on the model. (4) X1 supports copy vs regenerate with model-internal evidence; literal Grad-CAM is reported as failing in transformers.
**Engineering:** Qwen3 under guarded decoding emits U+2028/U+2029; `str.splitlines()` splits JSON lines there. New analysis code splits on "
" only.

### D40. Paper updated with the first third-review GPU session (2026-10-16)
**Decision:** Put C1, C2, C4 and X1 into the paper from `results/review3_runs.json`, with every table and figure generated:
- `decoding.tex` is now the 40-page 2x2 (plain / penalty only / n-gram ban only / both, Tesseract, EasyOCR and clean text; Holm over 33 tests). The 15-page CPU ablation survives only as one sentence in its note (TituLLMs 1B and BanglaT5, which the 40-page run did not cover). `fig_decoding.pdf` panel (b) redrawn from the same data (dataviz palette slots 1-4, validated; marker shapes for identity).
- New `checks.tex` (C1 float32 and C2 prompt format) and Sec IV-J "Precision and prompt format", replacing the precision caveat in III-G, VI-C and the threats.
- New `attribution.tex`, Sec V-E "What the models attend to" and an Attribution paragraph in III-H; Grad-CAM (Selvaraju et al., ICCV 2017) and HiResCAM (Draelos and Carin, arXiv 2011.08891) added after checking both. `review3_runs_analysis.py` now also writes the chance rate of the source-peak test (all other numbers reproduced exactly).
- Abstract, contributions 1 and 3, RQ3, RQ4, conclusion rewritten backwards from the new results; the gap paragraph in Sec I reframed as a risk (WRITING-GUIDE Sec 1). Title unchanged until the remaining runs and Bijon Sir's view.
**Why:** The task list of the handoff; the reviewer's C1, C2, C4, F1 and X1.
**Findings that changed the text:** (1) The pilot's explanation for literal Grad-CAM (attention sink) is not supported by the full run (sink share 0-19% at mid depth); the paper gives the HiResCAM averaging argument instead. (2) The TituLLMs checkpoints' few copied tokens also peak on their source (41-63%), so copy vs regenerate is a difference in how much is copied, not how; stated. (3) Phi-4-mini float32 is checked on Tesseract only, so "same results" is limited to that. The paper grew from 21 to 23 pages, which W2 must absorb.

### D41. Next GPU session (review3b) and region-level OCR prepared (2026-10-16)
**Decision:** Split the remaining runs into (a) a Colab CPU job for region-level OCR (C5/M5, `ocr/run_region_ocr.py`, `notebooks/colab_region_ocr.ipynb`), which costs no GPU quota; (b) Kaggle GPU session "review3b" (`correction/run_review3b_kaggle.py`, `notebooks/kaggle_review3b_run.ipynb`, about 11 h on T4 x2): X1 for Phi-4-mini and TituLLMs 3B Instruct (float16, since C1 showed float16 = float32), the M3 noise sweep for the four general-purpose models, M1 (Qwen3 0.6B, 4B, 8B; Gemma 3 1B, 4B; Llama 3.2 3B-Instruct) on Tesseract, clean and EasyOCR text, and full Grad-CAM maps of two chunks for a figure; (c) later sessions: all models on region-level OCR text (C5), the fine-tuning pilot (M2) and the large model (M7).
**Noise model (M3):** `eval/make_noise.py fit` on the 10 development pages only (both engines pooled; per-character substitution, deletion and insertion rates; `data/noise_model_dev.json`, counts only, no text). `apply` multiplies all rates by one factor per level, found by bisection on mean page CER; achieved 0.020, 0.048, 0.100, 0.199, 0.297 for targets 0.02-0.30. The dev OCR (`ocr_dev.jsonl`) predates the October OCR run, so its engine build differs; it only shapes the error profile, the level is set by the target CER.
**Precision for M1:** Qwen3 4B and Llama 3.2 3B in float16 on one GPU; Qwen3 8B in float16 and Gemma 3 4B in float32 across both GPUs (Gemma overflows in float16). Gemma 3 4B loads as `Gemma3ForConditionalGeneration` and receives text only. All chat templates checked locally (Qwen3 thinking disabled).
**Region OCR settings:** crops = region bounding box + 8 px; reading order as in the sensitivity analysis; Tesseract `--psm 6` on crops. Chosen from a local comparison on 8 evaluation pages (every 5th; local Tesseract 5.5.3, so plumbing and relative settings only): mean region CER 0.475 (psm 3, polygon mask), 0.474 (psm 4, mask), 0.436 (psm 6, mask), 0.423 (psm 6, mask, 25 px), **0.379 (psm 6, no mask)**, against 0.547 for whole-page OCR with the same local engine (stored 5.3.4: 0.541). The mask cut into glyphs, so it is off by default (`--mask`). Tesseract only; EasyOCR was not compared locally; the Colab job first proves that its whole-page OCR matches the main run (at least 38/40 Tesseract pages identical) and stops otherwise.
**Why:** Region-level OCR is the reviewer's third priority and needs no GPU; the noise sweep (M3) and the scale series (M1) need no new data and fit one session; M2 and M7 need more new code and are better tested separately.

### D42. Author's answers of 2026-10-16 (night); region OCR moved to Kaggle CPU
**Answers:** no page limit has been given (W2: no trimming target yet; the paper is 23 pages); region-level OCR to follow the reviewer's suggestion; release of OCR text and outputs undecided (C6, R2 stay open); page-type labels not yet checked (E8 open); no news from Bijon Sir; 41 min of Kaggle GPU quota left before the reset.
**Decision (region level):** the reviewer calls region-level evaluation "the cleanest fix" for the ground-truth mismatch (C5) and asks to *add* region-crop OCR to test the findings at lower input CER (M5). So region level becomes a full second setting: both engines, all eight models and the dictionary, its own table and tests; whole-page OCR stays the main, frozen setting, on which every other analysis is built. If the region-level results change a conclusion, the paper says so in the abstract.
**Decision (where):** the region OCR runs on a Kaggle CPU session (`notebooks/kaggle_region_ocr.ipynb`), not Colab, so it runs unattended overnight and uses no GPU quota. Tesseract 5.3.4 comes from conda-forge (micromamba, own prefix); if that fails, the system Tesseract is used. Either way the verify step records whether the whole-page OCR matches the main run, and `run_region_ocr.py` now saves those whole-page texts (`ocr_region40_wholepage.jsonl`), so region and whole-page OCR can be compared with the same engine build even if it is not 5.3.4. Page images, XML and `ocr_eval40.jsonl` go to Kaggle as one dataset (`G:/kaggle-upload/reid2019_eval40.zip`, local only).

### D43. Region OCR on Kaggle CPU: outcome; Tesseract part to be redone on Colab (2026-10-17)
**What ran:** `notebooks/kaggle_region_ocr.ipynb`, 1 h 38 min, 40 pages, 576 regions with text (`results/region/`, scored by `eval/region_ocr_analysis.py` → `results/region_ocr40.{txt,json}`).
**Check failed for Tesseract:** the conda-forge build of Tesseract 5.3.4 does not reproduce the main run: identical on 0/40 whole pages, CER between its text and the stored text 0.224 on average, and its whole-page CER is 0.425 against 0.341. Same version number, different build (the main run used Colab's apt build). EasyOCR 1.7.2 reproduces the main run closely (identical on 17/40 pages, CER between them 0.002).
**Results:** EasyOCR region OCR lowers CER from 0.297 to 0.225 (Δ −0.072 [−0.121, −0.033], p 0.002; 26 pages better, 14 worse), valid against the main run. Tesseract (conda build) region OCR: 0.333 against 0.425 for the same build's whole pages (Δ −0.092 [−0.152, −0.029]); not comparable with the main run. Empty region outputs: Tesseract 21/576, EasyOCR 93/576.
**Decision:** keep the EasyOCR region text; redo only the Tesseract region OCR on Colab, whose apt build produced the main run (`notebooks/colab_region_ocr.ipynb`, `--tesseract-only --keep-easyocr`, stops unless at least 38/40 whole pages match). If Colab no longer reproduces the main run either, use the conda-build Tesseract region text with its same-engine whole-page baseline and say so in the paper.
**Why:** the region setting is meant to test the findings at lower input CER with the same engines; a different Tesseract build would mix an engine change into the comparison.

### D44. Tesseract output of the main run is no longer reproducible; region setting uses today's Tesseract (2026-10-17)
**Finding:** the Colab rerun (`notebooks/colab_region_ocr.ipynb`, apt Tesseract 5.3.4) also matches the main run's Tesseract text on 0/40 pages (CER between them 0.224, whole-page CER 0.425 against 0.341), and its whole-page text is byte-identical to the Kaggle conda-forge build (CER between them 0.000). On 12 October the Colab confidence re-run still reproduced the main run on 40/40 pages, so Colab's environment changed in between (most likely a system library under Tesseract). Ruled out: the Bengali `tessdata_best` model (unchanged on GitHub since 2017; same file as ours), TIFF decoding (uncompressed RGB; Pillow and OpenCV decode identically) and the resolution tag (pytesseract passes 400 dpi; same output as the command line). The changes go both ways per page (e.g. 279_21_B_2_0004: 0.996 in the main run, about 0.59 now), so it is not a uniform degradation. EasyOCR is unaffected.
**Decision:** the region-level setting uses Tesseract as it runs today (the Kaggle file `results/region/ocr_region40.jsonl`; no Colab rerun needed, since Colab gives the same text), compared with today's whole-page Tesseract from the same run (`ocr_region40_wholepage.jsonl`, CER 0.425 → 0.333). EasyOCR region text is compared with the main run directly (0.297 → 0.225). The main whole-page setting keeps its frozen, stored OCR. The paper will state that Tesseract 5.3.4 output changed between 12 and 17 October 2026 with the same version and model, which strengthens the case for releasing the raw OCR text (C6, R2).
**Why:** two independent environments agree today, so today's output is reproducible; the old output cannot be regenerated, and a region comparison needs both texts from one engine build.

### D45. GPU session "review3c": all eight models on region-level OCR text (2026-10-17)
**Decision:** a separate Kaggle GPU session (`correction/run_review3c_kaggle.py`, `notebooks/kaggle_review3c_run.ipynb`, about 4 h on T4 x2, `HOURS = 5`) runs the eight models on `ocr_region40.jsonl` (both engines, main settings: plain greedy decoding, main prompt, about 250-token chunks, float16 above 3B), uploaded to Kaggle as its own small dataset `region-ocr40`. The dictionary baseline was run locally (`results/region/dict_region40.jsonl`: CER 0.333 → 0.369 Tesseract, 0.225 → 0.269 EasyOCR, descriptive until tested). No clean-text control on region text: the main run's control already covers it.
**Order:** review3c first (C5 is the reviewer's third priority and the session is short), then review3b (M3, M1, X1 leftovers, about 11 h, suits an overnight run). Together about 16 h of the weekly 30 h, leaving room for M2 and M7.
**Scoring (to write):** region-level tables mirroring the main matrix (corrected vs raw region OCR, paired bootstrap, Wilcoxon, Holm within the region family, which is exploratory); Tesseract against today's region OCR only (D44).

### D46. GPU runs review3b and review3c scored (2026-10-17)
**What ran:** both sessions in parallel, every job complete, no errors (precision as planned: Qwen3 4B/8B and Llama 3.2 3B float16, Gemma 3 float32). The Kaggle noise file is identical in content to the local one (only line endings differ), so `make_noise.py apply` is deterministic.
**Findings that change the paper:** (1) Region-level OCR lowers the input CER, but correction still never helps; the general-purpose models become significantly harmful on region text. (2) The noise sweep finds no level at which correction helps, down to input CER 0.02, so input noise does not explain the failure (VI-C "input noise" can be answered). (3) Within Qwen3, scale lowers WER significantly at 4B and 8B and raises the share of wrong words fixed, without lowering CER: the first trend towards help in the study, to report as such, not as a gain. (4) Grad-CAM now covers all eight models.
**Scoring:** `matrix_analysis.py` (each run file its own exploratory Holm family); summaries copied to `summaries/`.

### D47. Last GPU session (review3d): fine-tuning pilot, large model, Qwen3 4B on region text (2026-10-17)
**Decision:** `correction/run_review3d_kaggle.py` + `notebooks/kaggle_review3d_run.ipynb`, about 5 h on T4 x2 (`HOURS = 6`).
- **M2:** `correction/finetune_byt5.py`. google/byt5-small (bytes as tokens, so no Bengali tokenisation issue), float32, AdamW 3e-4, batch 8 (halved automatically on out-of-memory), 100 min. Training text: up to 40,000 segments of 60-300 characters from Bengali Wikipedia (20231101.bn, at least 90% Bengali letters) plus the dev-page reference lines x20; any segment sharing a 25-character string with an evaluation reference is dropped. Pairs: 10% clean (to learn copying), the rest corrupted with the dev-page noise model at CER 0.02-0.40. Checkpoint chosen by CER on the real OCR of the 10 development pages (never scored in the paper). Test: 40 pages x Tesseract, EasyOCR, clean text, chunks of whole lines up to 300 characters, greedy.
- **M7:** Qwen3-32B (dense) in 4-bit NF4 (bitsandbytes, float16 compute) across two T4s, plain decoding and the main prompt, on the 10 pages of the attribution analysis (`data/m7_pages.txt`, the seed-403 sample), Tesseract, clean and EasyOCR text. Chosen instead of Qwen3-30B-A3B because 4-bit loading of mixture-of-experts weights in Transformers 5 is uncertain, and because it extends the Qwen3 size series (0.6-32B) in which 4B and 8B showed the study's only trend towards help (WER).
- **C5 extra:** Qwen3 4B on the region text (one GPU, float16).
**Tested locally:** the whole fine-tuning pipeline with a tiny random model, Wikipedia streaming and the overlap filter, the noise corruption; the dry run of the orchestrator. Not testable locally: the 4-bit 32B load and the real ByT5 training on a T4.
**Environment note:** installing `datasets` locally upgraded `huggingface-hub` to 2.2.0, which Transformers 5.14 rejects; restored to 1.33. The notebook pins `huggingface-hub<2` for the same reason.

### D48. Repository reorganized; DOI metadata added (2026-10-17)
**Decision:** clean the repo to the current study (the author's request), with one rule for commands: everything runs from the repository root (`paper/README.md`).
- **Removed** (in git history): `STATUS.md` (superseded on 2026-10-15), `context/open-items.md` and `context/checklist-status.md` (second review, replaced by `review-draft-03.md`), `context/plan-draft-03.md` (third-review plan, now done and recorded in D36-D47), `ocr/test_env.py` (unused), `notebooks/colab_region_ocr.ipynb` (made unnecessary by D44).
- **Moved** (older entries in this file and in `timeline.md` use the old paths):

| Old path | New path |
|---|---|
| `eval/make_paper_tables.py`, `make_decoding_figure.py`, `make_noise_figure.py`, `make_example_table.py`, `make_paper_figures.py`, `make_revision_figures.py` | `paper/scripts/` (same names) |
| `paper/check_refs.py`, `make_overleaf_zip.py`, `render_examples_pdf.py` | `paper/scripts/` |
| `correction/run_review3_kaggle.py`, `run_review3b_kaggle.py`, `run_review3c_kaggle.py`, `run_review3d_kaggle.py` | `correction/sessions/` (same names; still run with `correction/` as working directory) |
| `notebooks/kaggle_gpu_run.ipynb` | `notebooks/kaggle/main_run_40pages.ipynb` |
| `notebooks/kaggle_variants_run.ipynb` | `notebooks/kaggle/prompt_variants.ipynb` |
| `notebooks/kaggle_region_ocr.ipynb` | `notebooks/kaggle/region_ocr_cpu.ipynb` |
| `notebooks/kaggle_review3_run.ipynb` | `notebooks/kaggle/review3a_precision_prompt_decoding_xai.ipynb` |
| `notebooks/kaggle_review3b_run.ipynb`, `_review3c_`, `_review3d_` | `notebooks/kaggle/review3b_noise_newmodels.ipynb`, `review3c_region_text.ipynb`, `review3d_finetune_largemodel.ipynb` |
| `notebooks/setup_and_dry_run.ipynb` | `notebooks/colab/first_version_cpu_15pages.ipynb` |

- **Verified after the move:** every script compiles; all 22 tables regenerate byte-identical (except `attribution.tex`, which gained the two models of review3b and a data-driven note); `compare_decoding.py`, `bootstrap.py` and `region_ocr_analysis.py` run from the root reproduce their stored outputs exactly; all figures redraw; the four session drivers dry-run from `correction/`; the paper builds with no overfull boxes and all references resolve; the Overleaf zip builds. The running review3d session was unaffected (it had already cloned the code).
- **DOI (C6/R2):** `CITATION.cff` (valid against CFF 1.2.0) and `.zenodo.json` added. Minting the DOI needs the author: link Zenodo to the GitHub account, enable the repository, then publish a GitHub release; Zenodo archives that release and assigns a DOI, which then replaces the plain repository URL in the paper's reference `rahman2026repo`. Two decisions are the author's: a licence for the code (none chosen yet; Zenodo needs one) and whether the per-page OCR text and model outputs are released (they are not in the repo; a separate Zenodo dataset deposit would hold them; the Tesseract drift of D44 makes the OCR text impossible to regenerate, which argues for releasing it).

### D49. Licences: MIT for the code, CC BY 4.0 for the released data (2026-10-17)
**Decision (author):** the code is under the MIT licence (`LICENSE`; also in `CITATION.cff` and `.zenodo.json`). The per-page OCR text and model outputs will be released separately, under CC BY 4.0, as a Zenodo dataset deposit together with the final paper (C6, R2), not before: a Zenodo record cannot be deleted, only versioned.
**Why:** MIT lets others reuse the code with attribution and keeps all rights with the author; CC BY 4.0 is the usual licence for research data and fits the public-domain source. Releasing the OCR text matters because the main run's Tesseract output can no longer be regenerated (D44).
**Still to do for the DOI:** the author links Zenodo to GitHub and enables the repository; after the final paper pass a GitHub release is made (with the author's go-ahead), Zenodo assigns the DOI, and the DOI replaces the repository URL in the reference `rahman2026repo`. Check with Bijon Sir whether the venue reviews anonymously (the repository is public under the author's name).
**Update 2026-10-17:** the author linked Zenodo to GitHub; the repository has an active Zenodo webhook for release events.

### D50. review3d scored; M2 pilot flawed and rerun as review3e (2026-10-18)
**What ran:** all three review3d jobs, no errors (`results/review3d/`). M7 and the region run of Qwen3 4B are valid and scored (`results/m7_runs10.txt`, `results/region_runs40.txt`; summaries copied).
**Findings:** (1) Qwen3-32B in 4-bit on 10 pages does not lower CER (Tesseract −0.003, EasyOCR −0.006, both intervals include zero) but lowers WER by about 0.1 and fixes the largest share of wrong words (15%/24%) of any model; on clean text it does more damage than Qwen3 1.7B-8B (0.073). Scale moves Qwen3 towards fixing words, not towards a lower CER. (2) Qwen3 4B on region text: significantly worse CER on EasyOCR (+0.014) with significantly lower WER (−0.030); n.s. on Tesseract. (3) The 10-page M7 file cannot reach significance after Holm over 36 (minimum possible p 0.070); the paper reports its intervals, not tests.
**M2 is not a valid result.** Training segments never contain a line break, but `correct()` fed chunks of whole lines joined by "\n" (also in the development evaluation that picks the checkpoint). On clean text the model deletes and repeats whole lines (168/678 lines missing; deletions 0.153 and insertions 0.108 per reference character, substitutions 0.056). Out-of-memory halvings also left it at batch 2 (about 7,900 pairs seen). Reporting this as "fine-tuning does not help" would be wrong.
**Decision:** fix and rerun only M2 (`notebooks/kaggle/review3e_finetune_rerun.ipynb`, one T4, about 3 h): chunk lines joined by spaces, as in training (CER ignores whitespace, so scoring is unchanged); effective batch 8 with gradient accumulation when the micro-batch is halved; 150 training minutes; development evaluation every 200 updates. Everything else as D47. The flawed scores are kept as `results/m2_runs40_review3d.*` (not in `summaries/`, not in the paper; the methods may mention the failed attempt in one sentence). Tested locally with the tiny random model; the out-of-memory branch can only run on the GPU. The notebook clones from GitHub, so the fix must be pushed before the run.
**Consequence:** the paper rewrite waits for review3e unless the author decides otherwise.

### D51. Local runs: dictionary variants (M4) and engine fusion (M6); edit-list prompting not run (2026-10-18)
**Decision (author):** run M4 and the fusion part of M6 locally; skip edit-list prompting (needs a GPU session with a real risk of a rerun before the planned finish; confidence gating already tests the closest constraint). It goes to future work.
**M4:** `correction/dictionary_baseline.py` gained `--lexicon`, `--max-edit-distance`, `--model-key` (defaults reproduce `dict_eval40.jsonl` exactly, 120/120 outputs). New lexicon `data/lexicon/bn_wiki100k.txt` from `data/build_wiki_lexicon.py` (not committed; sha256 recorded in results.md). Four settings (2 lexicons x edit distance 1/2) fixed in advance, one Holm family of 12 cells. Result: a better lexicon and a smaller edit distance cut the damage by about four-fifths but every setting stays significantly worse; the lexicon is not the whole explanation.
**M6 fusion:** `correction/ocr_fusion.py`, rules fixed in advance (EasyOCR base; replace blocks of at most 3 words; Tesseract wins if its mean within-page confidence rank is higher, or higher by 0.25). Uses the stored confidences (`ocr_conf40.jsonl`: 40/40 Tesseract and 37/40 EasyOCR pages reproduce the main OCR; unmatched words count as confident).
**Test family, stated openly:** the first scoring put the comparisons with both engines in one Holm family of 4. The comparison with Tesseract mostly measures that EasyOCR is the better engine, not fusion, so fusion is scored against EasyOCR only (2 cells). This choice was made after seeing the 4-cell run. In the 2-cell family the margin rule's CER change (−0.002) is significant (p 0.045), in the 4-cell family it is not (p 0.09). The paper reports both and calls the gain negligible, like the gating gain; the WER decrease (−0.014) is significant in both.
**Consequence:** new table `classical.tex` (generated), summaries `m4_dict40.json`, `m6_fusion40.json`, `m4_lexicon_coverage.json`.

### D52. Paper rewritten with all third-review results except M2 (2026-10-18)
**Decision:** one rewrite, written backwards (WRITING-GUIDE): new Results subsections IV-K region-level OCR, IV-L synthetic noise (Fig. 6, `fig_noise.pdf`), IV-M model size (Tables `scale.tex`, `large.tex`), IV-N methods without a language model (`classical.tex`); Sec V-C attribution now covers all eight models with the map figure (Fig. 7, `fig_attribution.pdf`); new Methods subsection III-L "Further experiments" (region OCR, noise, newer/larger models, fine-tuning, engine fusion); RQ5 added; contributions, discussion (input-noise explanation answered; model size; historical spelling partly tested), practical implications, threats and "experiments not performed" rewritten; Tesseract drift and the CC BY 4.0 release of OCR text stated in III-C/III-M; stale sentences fixed (Phi-4-mini and TituLLMs 3B Instruct now analysed; "nothing is trained"; "region-matched CER untested").
**Length / F5 / W2:** robustness, edit-operation, page-type and safeguard tables and the change-in-CER scatter (old Fig. 6; F5) moved to an appendix, as the reviewer suggested. The paper is 27 pages (no page limit given).
**Title (working, for Bijon Sir):** "Small Language Models Do Not Correct Historical Bengali OCR, and a Decoding Default Makes Them Look Worse". Alternatives: "Leave the OCR Alone: Zero-Shot Language Models Copy or Rewrite Historical Bengali Print but Do Not Correct It"; "Zero-Shot Language Models Do Not Correct Historical Bengali OCR at Any Noise Level or Size up to 32B". `CITATION.cff` and `.zenodo.json` keep the old title until the author fixes the title.
**Left for review3e:** the fine-tuning subsection (`finetune.tex`), one sentence each in RQ5, practical implications, conclusion and abstract (marked `% M2` in `main.tex`); the abstract is 216 words and must be trimmed to about 200 when that sentence goes in.

### D53. Literature added after a Consensus check (2026-10-18)
**Decision:** cite seven papers found with Consensus, each checked against its primary record (ACL Anthology, Princeton/ACM DocEng record, arXiv, ACM DOI, TACL): Boros et al. 2024 (LaTeCH-CLfL; 14 models zero/few-shot, "anything but efficient"), Zhang et al. 2024 (DocEng; GPT models over-correct good OCR), Maheshwari et al. 2022 (Findings of EMNLP; Sanskrit post-OCR, ByT5 best), Bhandari & Harit 2026 (TALLIP; supervised Indic post-OCR with context), Bhandari & Harit 2026 (arXiv 2609.21595; Hindi/Marathi, 3-32B with retrieved examples, below 8B little or worse), Bourne 2024 (arXiv; fine-tuning on synthetic corruption), Wiher et al. 2022 (TACL; decoding properties do not transfer across tasks). Placed in Sec II-C, II-E, the research-gap paragraph (now limited to what Table I shows), III-D, III-L (fine-tuning) and VI-B; two rows added to Table I and its "This study" row updated.
**Not cited:** Huang et al. 2025 (RAP; no venue found), Santos et al., Greif et al., CLOCR-C, Guan et al., Soper et al. (optional; see `context/literature-leads.md`).
**Novelty:** neither the search nor the Consensus research report (20 papers) found a paper stating that repetition penalties and n-gram bans act on the prompt and distort correction evaluations; the paper keeps "we show", not "first".
**Caveat:** the TALLIP paper's ACM page blocked automated reading; its details come from the DOI record, a citing paper and the abstract indexed by Consensus. Check the PDF if you have ACM access.

### D54. review3e scored; fine-tuning in the paper; paper complete (2026-10-18)
**Result:** the fixed M2 run is valid (format bug gone: 14/678 clean-text lines missing instead of 168). ByT5-small fine-tuned on synthetic noise lowers WER significantly on both engines and does the least damage to error-free text of any method (CER 0.015), but CER rises (+0.012 significant on Tesseract, +0.016 n.s. on EasyOCR). It behaves like the larger zero-shot Qwen3 models: copy-and-edit, word-level gains, no CER gain.
**Paper:** new Sec IV (Fine-tuning) with `finetune.tex`; the failed first run is stated in one sentence in III-L; RQ5, a "Task-specific training" explanation, practical implications, conclusion, contribution 3, Table I ("one fine-tuned pilot") and the abstract (now 200 words) updated. The title still holds (a small fine-tuned model does not correct CER either).
**Correction:** the claim that fusion is the only method fixing more words than it breaks (D51, results.md) was false; corrected in the paper (IV-N, VI-B RQ5, VI-D) and README to "breaks fewer correct words than any ungated language model on the 40 pages".
**Generator:** `finetune.tex` note now reports updates of the effective batch and thousands separators; default `--m2-log` reads `results/review3e/m2_train_log.json`.

### D55. No preprints; single-file final version (2026-10-18)
**Decision (author):** only published papers are cited, no preprints. The final paper goes to Bijon Sir, who sends it for publication.
**Changes:** removed bbOCR (arXiv 2308.10647; its Tesseract CER figures left the introduction and Sec II-A), HiResCAM (arXiv 2011.08891, still "under review"; the element-wise Grad-CAM variant is now described in our own words, with the averaging argument stated without citation, also in the generated `attribution.tex` note), and Bhandari & Harit's Devanagari study (arXiv 2609.21595; its Table I row and three sentences removed; the research gap again says no post-OCR study in Table I prompts open models below 7B, which holds for every row). Bourne is now cited by its published version (IJDAR 28(4):741-755, 2025, doi 10.1007/s10032-025-00522-0). The arXiv technical reports of Qwen3, Phi-4-mini, Gemma and Gemma 3 are replaced by their Hugging Face model cards (software documentation, not papers; Llama 3.2 was already cited this way). 60 references, none a preprint.
**Single file:** `paper/scripts/make_single_file.py` writes `paper/main-single.tex` (all tables and the compiled bibliography inlined) and `paper/overleaf-final.zip`. Tested: the zip, unpacked into an empty folder, compiles with Tectonic (XeLaTeX) to the same 28 pages, no errors, no undefined references, no overfull boxes. pdfLaTeX could not be tested locally (not installed); the example table then falls back to the pre-rendered PDF, as before.
**Also:** Results introduction now lists the further experiments.

### D56. Revision after the editorial review (2026-10-18)
**Source:** `G:/Codex docs/2026-10-18/.../editor-feedback.md` (external editorial review).
**Done:** fixed/broken now reported as absolute word counts (matrix_analysis.py stores `words`; all matrix results rerun, numbers unchanged; four tables switched to counts); "decoding default" claim removed (Transformers neutral values 1.0/0; our 1.3/4-gram were added after loops); new working title "Small Language Models for Historical Bengali OCR Correction: Word-Level Gains, Character-Level Damage, and a Decoding Pitfall"; abstract, introduction (document-processing motivation, 3 RQs, 3 contributions), Discussion (3 RQ answers, new "word-level gains without character-level gains"), conclusion rewritten; experiment map (hand-written `tables/experiments.tex`); section-3 inconsistencies fixed (development-page uses, "no model was trained", CPU claim, 33 completed comparisons, Table V caption, "third review" note, reproducibility wording, metric-scope limitation); "confirm"/"no break-even"/"does not explain"/"less than 1%" qualified (exact gains 1.1% and 0.8%); added the missed WER exceptions (Llama 3.2 3B, Qwen3 4B region, two noise cells); appendix tables flushed before references; Ethnologue dropped; prose cut from 11,464 to 10,814 words (-6%).
**Not done:** the review's 15-25% cut (only 6%); interpreting the Bengali examples (which form is an OCR error vs a historical form) needs the author; student-ID line in the author block is the author's call.

### D57. Second editing pass after the editorial review; full read-through (2026-10-18)
**Cuts:** body prose from 13,899 words (before the review, same counter) to about 12,150 (-12.6%): Related Work on language-model correction regrouped by what decides success (training, input quality, language) instead of repeating Table I; the engine-comparison subsection folded into the main result; CPU/GPU check and Truncation moved/merged (appendix, BanglaT5 paragraph); edit profiles, character classes, page types, safeguards, precision, robustness, tokenisation and prompt-format bullets shortened; threats rewritten in the review's priority order with an ordered future-work sentence (replaces "experiments not performed").
**Read-through fixes:** Fig. 2 development-page box now lists all uses (debugging, few-shot, noise model, fine-tuning; "not in reported scores"); appendix references no longer print "Section A-A"; "relies on" wording for Grad-CAM replaced by "map peaks at/beside" (text, Fig. 7 caption, Table XXII caption "Gradient-Based Attribution of Generated Tokens"); "least damage" in the clean-text section limited to the core methods (ByT5, best dictionary and Gemma 3 1B do less); region "wrong edits weigh more" marked as a possible reason; fine-tuning budget and pair counts made consistent (150-minute budget incl. evaluations, ended at 155; ~14,300 of 43,312 pairs; ~40,000 Wikipedia segments).
**Still for the author:** interpreting the two Bengali examples (OCR error vs valid historical form); the student-ID line; the title (working title in D56).

### D58. Seven corrections from the reviewer's second reading (2026-10-18)
1. "Code-point CER" in the abstract, conclusion and RQ1; RQ1 names the grapheme-cluster exception (Qwen3 1.7B, Tesseract).
2. Methods overview: "receive every analysis" replaced; the main evaluation covers the eight core models and the dictionary, diagnostics and extensions cover the subsets of Table II.
3. RQ3: "no tested prompt variant, chunking scheme or page-level safeguard produced a statistically significant CER reduction", with the example 0.335 vs 0.341 (Qwen3 1.7B, preserve-spelling, Tesseract).
4. 32B table: the Holm family was always 36 cells per metric = all 12 models rescored on the 10 pages (8 core + Qwen3 0.6B/4B/8B/32B) x 3 inputs (checked in `results/m7_runs10.json`); the table shows only the Qwen3 rows. The note (generated from the data) and the text now say so; the family is unchanged. "Estimate, not a test" replaced by the reviewer's wording.
5. "Decoding is greedy, so output is deterministic" -> "We use greedy decoding, without token sampling"; the reproducibility caveat stays in Statistics.
6. Related work on language-model correction split into three paragraphs (training, input quality, language/register); future work reduced to three extensions (validate gating/fusion; task-specific training; measure downstream value) plus one clause.
7. Abstract: "up to 1.4 CER units over plain decoding (Phi-4-mini with the repetition penalty alone)" (+1.396 on Tesseract text, Table X). Abstract 216 words.

### D59. Draft title restored; abstract in the writing guide's order (2026-10-18)
**Decision (author):** use the title of the draft Bijon Sir reviewed, "Does Zero-Shot Small-Model Post-Correction Help Bengali OCR? A Negative Result Across Eight Models and Two OCR Engines" (still accurate: the main evaluation is eight models on two engines). This departs from WRITING-GUIDE Sec 3 (statement titles); the author's choice.
**Abstract:** content unchanged, reordered to the guide's narrative order (background, problem, what we did, what we found, what to do); 223 words, acronyms only CER and OCR; "we found no study that measures this" instead of an absolute novelty claim.

### D60. Bijon Sir's draft-04 corrections: text and number fixes (2026-10-18)
**Source:** `Bengali_PostOCR_draft04_corrections.pdf` (12 corrections, 4 suggestions, journal list); item status in `context/review-draft-04.md`.
**Done:** C1, C2, C3, C6-C12 (details in review-draft-04.md). C2 is now tested on the version the runs used (5.19.0, Qwen3 0.6B, one generation step, CPU), not only read in 5.14.1 source. C3: fusion's CER gain is called descriptive, not significant, everywhere; its table mark is removed and both Holm p-values come from result files (the 4-cell family rerun with `eval/matrix_analysis.py` on the fusion outputs duplicated as engine "tesseract" reproduces the D51 values 0.090 and 0.205). C5: stated, not redone.
**New result files (new outputs, none edited):** `results/prompt_constraints_5.19.0.json`, `results/m6_fusion40_family4.{json,txt}`, `results/region_phi4-mini_easyocr_pages.json`.
**Not done (need the author):** C4 longer ByT5 training (Kaggle GPU); C5 redoing the crop choice on development pages (Tesseract run); S1-S4 and the journal choice.
**Build:** 29 pages (was 28), no overfull boxes, references resolve, Bengali table renders.
