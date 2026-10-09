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
**Why:** On Colab, `torch.get_num_threads()` returned 1, giving ~1 s/token. This was the real root cause of the "stuck" and slow pages, found with `correction/diagnose_speed.py`. After the fix: 2 threads.

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
