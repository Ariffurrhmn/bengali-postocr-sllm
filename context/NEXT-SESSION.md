# Next session: start here

Handoff updated 2026-10-17 (evening). Read this file first, then the files in section 2. Everything up to the repository reorganization (D48) is committed and pushed to `master` unless `git status` says otherwise.

## 1. The project in five lines
- Paper: "Does Zero-Shot Small-Model Post-Correction Help Bengali OCR? ..." (`paper/main.tex`, author Arifur Rahman, course CSE403, reviewer **Bijon Sir**). The title is to become a statement of the finding once all results are in (WRITING-GUIDE Sec 3; check with Bijon Sir).
- 40 historical Bengali pages (REID2019), Tesseract 5.3.4 and EasyOCR 1.7.2, zero-shot correction by open models (0.25-8B, and 32B in review3d), a SymSpell baseline, plain greedy decoding.
- Finding so far: no method lowers CER, on whole-page OCR, on region-level OCR, or on synthetic noise at any level (0.02-0.30); general models copy (shown also by Grad-CAM maps), Bengali-pretrained checkpoints regenerate; every method damages clean text. Qwen3 4B/8B lower WER significantly without lowering CER.
- Lead contribution: repetition penalty and n-gram ban also act on the prompt in Transformers and make copying models look harmful (40-page 2x2, including clean text).
- The reviewer's third round: `context/review-draft-03.md` (43 items + X1, with status notes).

## 2. Read these, in order
1. `context/review-draft-03.md`: item status.
2. `context/decisions.md`: D39-D48 (third-review runs, region OCR, Tesseract drift, reorganization with the old→new path map).
3. `context/results.md`: all numbers, including "Third-review GPU runs 2 and 3".
4. `context/engineering-notes.md`: gotchas. Most important: about 4 GB free RAM; `HF_HUB_CACHE=G:/hf-cache`; **Bash heredocs mangle backslashes** (use the Edit tool or a script file); `PYTHONIOENCODING=utf-8`; local `TESSDATA_PREFIX` must point at `.tessdata`; **do not let pip upgrade `huggingface-hub` to 2.x** (Transformers 5 refuses it; D47); Tesseract output drifted within 5.3.4 (note 15).
5. `paper/WRITING-GUIDE.md` and `paper/README.md` (every command runs from the repository root).

## 3. Where things stand

### Done
- Paper updated with the first third-review GPU session (C1, C2, C4, X1; D40). 23 pages.
- Region-level OCR (Kaggle CPU; D42-D44) and all models on it (review3c); noise sweep and new models (review3b); Grad-CAM for all eight models. Scored (D46): `results/{region_ocr40,region_runs40,m1_runs40,noise_runs40,main_general40}.txt`, summaries in `summaries/`. Noise figure `paper/figures/fig_noise.pdf` (F7), not yet in the paper.
- Repository reorganized (D48); `CITATION.cff` and `.zenodo.json` added.

### Running
- **review3d** on Kaggle (started 2026-10-17): M2 ByT5-small fine-tuning pilot, M7 Qwen3-32B 4-bit on 10 pages, Qwen3 4B on region text (D47). Output `review3d.zip` → unpack to `results/review3d/`.

### Not done
- Score review3d (commands at the end of the "Third-review GPU runs" block in `paper/README.md`; check `results/review3d/m2_train_log.json` for the development CER per checkpoint and `phase2.log` for the 4-bit load).
- Paper tables for C5, M1, M2, M3, M7 and the Grad-CAM map figure (generators in `paper/scripts/`; check `git log` for what is done), then the full paper rewrite, written backwards (WRITING-GUIDE): conclusions, results, methods, discussion, introduction, abstract, title. The text still says Phi-4-mini and TituLLMs 3B Instruct were not analysed (now they are) and lists the noise sweep, fine-tuning, stronger OCR and large model as "not performed".
- Local and cheap: M4 (SymSpell edit distance 1, stronger lexicon), E7.
- Author decisions: code licence and DOI (Zenodo link + GitHub release, D48), release of the OCR text and outputs (C6, R2), page-type labels (E8), title, F5 (Figs 5/6), W2 (no page limit given; 23 pages).

## 4. Task list for the next session (in order)
1. Score review3d; record results (results.md, decision entry, review status for M2, M7, C5).
2. Finish the table and figure generators for C5, M1, M2, M3, M7 and the Grad-CAM maps; regenerate; check they render.
3. Rewrite the paper with all results (write backwards); rebuild with Tectonic; check the Bengali table; tick items.
4. Ask the author about the licence and the DOI release; if they agree, create the GitHub release with `gh` (Zenodo must be linked first).
5. Ask before every commit/push.

## 5. Questions to ask the author at the start of every session
Ask only those still open, together:
1. Has review3d finished? Where is `review3d.zip`?
2. Which licence for the code (for example MIT or Apache-2.0), and should I make the GitHub release for the DOI once Zenodo is linked?
3. May the raw OCR text and model outputs be released (a Zenodo dataset deposit)? (C6, R2)
4. Have you checked the page-type labels in `data/page_types.tsv`? (E8)
5. Any news from Bijon Sir (Grad-CAM section, title)?
6. May I commit and push?

## 6. Rules carried over from the author
- Do not discard uncommitted changes. Commit or push only when the author says so.
- Never edit result files; regenerate tables and figures from them. No number in a table is typed by hand.
- Do not invent numbers. If a check fails or a claim is not supported, say so.
- Keep decisions in `context/decisions.md` (decision, why, consequence). Tick items in `context/review-draft-03.md` with a *Status* note.
- Rebuild the paper with Tectonic after edits and check that the Bengali table renders.
- Heavy model runs happen on Kaggle, run by the author; local runs are CPU only and limited by RAM.
- The author does not want to babysit runs: give clear, short instructions for each Kaggle step.
