# Next session: start here

Handoff updated 2026-10-16 (night), at the end of the session that put the first third-review GPU session into the paper and prepared the next runs. A new session should read this file first, then the files it points to. This session's work was committed and pushed to `master` on 2026-10-16 (see `git log`; the commit after `93b323e`).

## 1. The project in five lines
- Paper: "Does Zero-Shot Small-Model Post-Correction Help Bengali OCR? A Negative Result Across Eight Models and Two OCR Engines" (`paper/main.tex`, author Arifur Rahman, course CSE403, reviewer **Bijon Sir**).
- 40 historical Bengali pages (REID2019, British Library), Tesseract 5.3.4 and EasyOCR 1.7.2, eight models (0.25-3.8B) plus a SymSpell dictionary baseline, zero-shot, plain greedy decoding.
- Finding: no method significantly lowers CER. General-purpose models copy their input (now also shown with token-level Grad-CAM maps); the Bengali-pretrained checkpoints regenerate or drop it, also in float32 and with a completion prompt; every method damages clean text.
- Lead contribution: a repetition penalty and n-gram ban (common decoding defaults) also act on the prompt in Transformers; on the 40 pages every constraint, alone or together, makes every copying model harmful, even on error-free input.
- The reviewer's third round is 43 items plus a verbal Grad-CAM request (item X1).

## 2. Read these, in order
1. `context/review-draft-03.md`: the items, each with *Status* notes. Ticked = done.
2. `context/decisions.md`: D39 (first GPU session), **D40** (paper update), **D41** (next runs), **D42** (author's answers; region OCR on Kaggle CPU, region level = full second setting), **D43** (region OCR outcome: EasyOCR valid, Tesseract to be redone on Colab). Earlier entries explain older choices.
3. `context/results.md`: current numbers ("Third-review GPU runs").
4. `context/engineering-notes.md`: gotchas. Most important here: about 4 GB free RAM; `HF_HUB_CACHE=G:/hf-cache`; **Bash heredocs mangle backslashes** (bit this session twice: use the Edit tool or a script file for LaTeX/Python edits); `PYTHONIOENCODING=utf-8`; locally, `TESSDATA_PREFIX` must point at the repo's `.tessdata` (scoop sets its own).
5. `paper/WRITING-GUIDE.md` and `paper/README.md` (regeneration commands now include `review3_runs_analysis.py` and the new `make_decoding_figure.py --runs`).

## 3. Where things stand

### This session (2026-10-16, committed and pushed)
- **Paper updated with the first GPU session (D40):** new tables `decoding.tex` (40-page 2x2, replaces the 15-page CPU table), `checks.tex` (C1 float32, C2 prompt format), `attribution.tex` (X1); Fig 4 redrawn; Sec IV-D rewritten, new Sec IV-J "Precision and prompt format", new Sec V-E "What the models attend to", Attribution paragraph in III-H; abstract, contributions, gap paragraph (risk framing), RQ3/RQ4, discussion, threats and conclusion updated; Grad-CAM and HiResCAM references added (checked). Built with Tectonic: no overfull boxes, all refs resolve, Bengali table renders. **23 pages** (was 21).
- `eval/review3_runs_analysis.py` also writes the source-peak chance rate; regenerated `results/review3_runs.*` and `summaries/review3_runs.json` (all other numbers identical).
- Review items ticked: C1, C2, C4, E4, F1, X1.
- **Prepared, not yet run:** `ocr/run_region_ocr.py` + `notebooks/kaggle_region_ocr.ipynb` (C5, Kaggle CPU overnight, Tesseract 5.3.4 from conda-forge; the Colab notebook is a fallback); `eval/make_noise.py` + `data/noise_model_dev.json` (M3); new model keys in `correction/models.py` (M1); `correction/run_review3b_kaggle.py` + `notebooks/kaggle_review3b_run.ipynb` (M3, M1, X1 leftovers, map figure). See D41.
- **Region OCR local check (D41):** on 8 pages, region crops with `--psm 6` and no polygon mask gave Tesseract CER 0.379 against 0.547 for whole pages with the same local engine (5.5.3); the real run must use 5.3.4 on Colab.
- **Finding to remember:** the pilot's explanation of why literal Grad-CAM fails (attention sink) is not supported by the full run; the paper uses the HiResCAM averaging argument.

### Not done
- Runs: Kaggle CPU region OCR done (D43, `results/region/`, scored in `results/region_ocr40.txt`); Tesseract part to be redone on Colab (`notebooks/colab_region_ocr.ipynb` → `ocr_region40_colab.jsonl`; if its verify fails, use the Kaggle file and its same-engine baseline); Kaggle review3b (M3, M1, X1 leftovers). Then scoring scripts for them (not written yet: write `eval/review3b_runs_analysis.py` once outputs exist, following `review3_runs_analysis.py`; M3 plot = F7).
- Later sessions: correction runs on region OCR text (C5), fine-tuning pilot (M2), Qwen3-30B-A3B 4-bit (M7), M4 edit distance 1 / stronger lexicon (local, cheap), E7.
- Waiting on the author: W2 page limit (paper is 23 pages), F5, C6/R2 release, E8 labels, title change (WRITING-GUIDE Sec 3; check with Bijon Sir).

## 4. Task list for the next session (in order)
1. If the author has run them: score the region OCR (`region.zip` → `ocr_region40.jsonl`; check `ocr_region40_verify.json` first: if Tesseract is not identical to the main run, compare region OCR with `ocr_region40_wholepage.jsonl`, the same-engine whole-page baseline, and say so) and the Kaggle `review3b.zip` (unpack to `results/review3b/`, gitignored). Write `eval/review3b_runs_analysis.py` (paired bootstrap + Wilcoxon as in `review3_runs_analysis.py`; M3: corrected vs input CER per level and model; M1: same tables as the main matrix for the new models; X1: add Phi-4-mini and TituLLMs 3B Instruct to `attribution.tex`; map figure from the `maps_*` files).
2. Put the results in the paper (write backwards, WRITING-GUIDE), regenerate tables/figures, rebuild, tick items, decision entry.
3. Prepare the next GPU session: all models on region OCR text (`run_sweep.py --ocr-path ocr_region40.jsonl`), M2 (ByT5-small fine-tuned on text corrupted with `make_noise.py`'s model; clean training text must not include eval pages), M7.
4. Ask before every commit/push; the Kaggle and Colab notebooks need the code on GitHub.

## 5. Questions to ask the author at the start of every session
Ask only those still open, together:
1. How much Kaggle GPU quota is left, and when does it reset?
2. Has the Kaggle region-OCR run (CPU) or the review3b run (GPU) finished? Where are the outputs?
3. Any venue or page limit yet? (None given as of 2026-10-16; W2, F5; the paper is 23 pages.)
4. May the raw OCR text and model outputs be released (Zenodo or the repo)? (C6, R2)
5. Have you checked the page-type labels in `data/page_types.tsv`? (E8)
6. Any news from Bijon Sir (Grad-CAM section, title)?
7. May I commit and push?

## 6. Rules carried over from the author
- Do not discard uncommitted changes. Commit or push only when the author says so.
- Never edit result files; regenerate tables and figures from them. No number in a table is typed by hand.
- Do not invent numbers. If a check fails or a claim is not supported, say so.
- Keep decisions in `context/decisions.md` (decision, why, consequence). Tick items in `context/review-draft-03.md` with a *Status* note.
- Rebuild the paper with Tectonic after edits and check that the Bengali table renders.
- Heavy model runs happen on Kaggle, run by the author; local runs are CPU only and limited by RAM.
- The author does not want to babysit runs: give clear, short instructions for each Kaggle/Colab step.
