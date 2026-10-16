# Next session: start here

Handoff written 2026-10-15 (evening), updated 2026-10-16 after the first GPU session, at the end of the session that answered the third review ("Draft 03"). A new session should read this file first, then the files it points to. Everything below is current as of commit `b73f20f` on `master`.

## 1. The project in five lines
- Paper: "Does Zero-Shot Small-Model Post-Correction Help Bengali OCR? A Negative Result Across Eight Models and Two OCR Engines" (`paper/main.tex`, author Arifur Rahman, course CSE403, reviewer **Bijon Sir**).
- 40 historical Bengali pages (REID2019, British Library), Tesseract 5.3.4 and EasyOCR 1.7.2, eight models (0.25-3.8B) plus a SymSpell dictionary baseline, zero-shot, plain greedy decoding.
- Finding: no method significantly lowers CER. General-purpose models copy their input; the Bengali-pretrained checkpoints regenerate or drop it; every method damages clean text.
- Lead contribution: a repetition penalty and n-gram ban (common decoding defaults) also act on the prompt in Transformers and make copying models look harmful.
- The reviewer's third round is 43 items plus a verbal Grad-CAM request (item X1).

## 2. Read these, in order
1. `context/review-draft-03.md`: the 43 items (C1-C8, W1-W9, M1-M7, E1-E8, F1-F7, R1-R4) plus X1, each with a *Status* note under it. Ticked = done.
2. `context/plan-draft-03.md`: cross-check of each critical item against the repo, the GPU run plan and the defaults chosen.
3. `context/decisions.md`: D36 (paper edits and analyses), D37 (Grad-CAM design), D38 (first GPU session). Earlier entries D01-D35 explain older choices.
4. `context/results.md`: current numbers, including the "Third-review analyses" paragraph.
5. `context/engineering-notes.md`: gotchas. Most important for this machine: about 4 GB of free RAM (models above about 0.6B do not run locally), the C: drive is full (set `HF_HUB_CACHE=G:/hf-cache`), Bash heredocs mangle backslashes (write edit scripts to a file first), and Bengali output needs `PYTHONIOENCODING=utf-8`.
6. `paper/WRITING-GUIDE.md`: how to write and revise the paper (narrative, write backwards, a title that states the finding, plain words, no hype). Follow it whenever paper text changes.
7. `paper/README.md`: how tables and figures are regenerated and how the paper is built (Tectonic, `tectonic -X compile main.tex` in `paper/`).

## 3. Where things stand (updated 2026-10-16)

### Done
- 2026-10-15 (commit b73f20f): paper-only edits C3, C7, C8, W1, W3-W9, E5, E6, E8, F2, F3, F4, F6, R1, R3, R4; analyses on existing outputs (`eval/review3_analysis.py` → `results/review3_gpu40.*`); tables `equivalence.tex`, `cost.tex`; draft figure `fig_decoding.pdf`; Grad-CAM script `eval/xai_gradcam.py`.
- 2026-10-16: **first GPU session finished** (Kaggle T4 x2, about 9.5 h; quota left afterwards about 40 min). Outputs unpacked to `results/review3/` (gitignored), scored by `eval/review3_runs_analysis.py` → `results/review3_runs.{txt,json}` (copy in `summaries/`). Findings are in `context/results.md` ("Third-review GPU runs") and D39:
  - C1: float16 is **not** the cause of the empty TituLLMs 3B outputs (identical first tokens fp16/fp32 on 208/208 chunks, no overflow, same CER).
  - C2: completion prompt lowers TituLLMs 1B CER by about 0.2 but it stays far above raw OCR; BanglaT5 normaliser changes nothing.
  - C4: 2x2 on 40 pages done for Gemma, Llama, Qwen3, Phi-4 (Tesseract, clean text; EasyOCR for three). Guarded decoding damages every model; which half does the damage depends on the model.
  - X1: six models done. Copying models' maps peak on the copied source token 58-78%; literal Grad-CAM fails (0-8%).
  - Missing from that session: X1 for TituLLMs 3B Instruct and Phi-4-mini; Phi-4-mini fp32 on clean text (1/40). Both are low priority.

### Not done
- **Paper not yet updated with the 2026-10-16 results** (next task).
- GPU, after the next quota reset: C5/M5 region-level OCR, M3 noise sweep, M1 new models, M2 fine-tuning pilot, M7 large model, plus the two missing pieces above.
- Waiting on the author: W2 (page limit), F5; C6/R2 release of OCR text and outputs; E8 page-type labels; commit/push of everything since b73f20f.

## 4. Task list for the next session (in order)

1. **Update the paper with the 2026-10-16 results**, following `paper/WRITING-GUIDE.md` (write backwards). Never type numbers by hand: extend `eval/make_paper_tables.py` / `eval/make_decoding_figure.py` to read `results/review3_runs.json`.
   - C1: replace the precision caveat (Sec III-G, IV-J, threats to validity, discussion "prompt format and instruction-following") with the check and its result.
   - C2: add the completion-prompt and normaliser results as a sensitivity result (short paragraph + a row block in a table or a small table); keep the main numbers as they are.
   - C4: replace Table VII (15-page CPU ablation) with the 40-page 2x2 (plain / penalty only / n-gram only / both, per model, OCR and clean text); keep the 15-page CPU numbers only as a one-line note or drop them; redraw Fig 4 from the 40-page data; rewrite Sec IV-D; update contribution 1, the abstract sentence ("raising CER by up to …") and the conclusion.
   - X1: new subsection in Sec V ("What the models attend to"), one table (copy share, map shares on OCR text / instruction / own output, source-peak rate, literal vs element-wise) and, if space allows, one figure; one sentence on why literal Grad-CAM fails (attention-sink token); cite Grad-CAM (Selvaraju et al., ICCV 2017) and HiResCAM (Draelos and Carin) after checking the references; remove "Scope of the analysis: no attribution" from the limitations.
   - Rebuild with Tectonic, check overflows and the Bengali table, tick items in `context/review-draft-03.md`, add a decision entry, update `timeline.md`.
2. **Ask to commit and push** (nothing since b73f20f is committed).
3. **Prepare the next GPU session** (see the proposal "Prepare post-reset GPU runs and scoring"; check `context/decisions.md` for D40+ in case another session did it):
   - C5/M5: region-crop OCR from the PAGE-XML v2 boxes, both engines, on a Colab or Kaggle CPU (local Tesseract is 5.5.3, not 5.3.4; first re-OCR whole pages and check they match `results/ocr_eval40.jsonl`); region-level scoring in the declared reading order; then all models on region text.
   - M3: noise model estimated from the 10 dev pages only; references noised to CER 0.02/0.05/0.10/0.20/0.30; four general-purpose models; plot corrected vs input CER (F7).
   - M1: Qwen3 0.6B/4B/8B, Gemma 3 1B/4B (fp32; Gemma overflows in fp16), Llama 3.2 3B-Instruct.
   - M2: ByT5-small (or mT5-small/BanglaT5) fine-tuned on dev-noise-corrupted Bengali text; test on the 40 pages.
   - M7: Qwen3-30B-A3B in 4-bit on two T4s, 10 pages.
   - Leftovers: X1 for TituLLMs 3B Instruct and Phi-4-mini (fp16 on one GPU is fine now that C1 showed fp16 = fp32).
   - One notebook + orchestrator like `run_review3_kaggle.py` (two lanes on T4 x2). fp32 across two T4s is slow (engineering note 14); new notebooks need `HF_TOKEN` ticked under Add-ons → Secrets (note 13).
   - New code must be committed and pushed before a Kaggle run. Ask first.
4. **After those runs:** score, update the paper, then W2 trimming once the page limit is known.

## 5. Questions to ask the author at the start of every session
Ask only those that are still open (check this file and `review-draft-03.md` first), and ask them together at the start:
1. How much Kaggle GPU quota is left, and when does it reset? (About 40 min were left on 2026-10-16.)
2. Is there a new Kaggle output to score? If so, where is the zip?
3. What is the venue and its page limit, and what is the submission deadline? (Blocks W2, F5.)
4. Should region-level OCR (C5) be a second setting (default) or replace whole-page OCR as the main setting?
5. May the raw OCR text and model outputs be released (Zenodo or the repo)? (C6, R2.)
6. Have you checked the page-type labels in `data/page_types.tsv`? (They were assigned by the assistant; E8.)
7. Has Bijon Sir said anything more about Grad-CAM, or any new feedback?
8. May I commit and push the changes (needed before any Kaggle run)?

## 6. Rules carried over from the author
- Do not discard uncommitted changes. Commit or push only when the author says so.
- Never edit result files; regenerate tables and figures from them. No number in the paper is typed by hand.
- Do not invent numbers. If a check fails or a claim is not supported, say so.
- Keep decisions in `context/decisions.md` in the existing style (decision, why, consequence). Tick items in `context/review-draft-03.md` with a *Status* note.
- Rebuild the paper with Tectonic after edits and check that the Bengali table renders.
- Heavy model runs happen on Kaggle, run by the author; local runs are CPU only and limited by RAM.
- The author does not want to babysit runs: give clear, short instructions for each Kaggle step.
