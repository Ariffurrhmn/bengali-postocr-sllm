# Next session: start here

Handoff updated 2026-10-19 (after the external review by Ahmed Shafkat, D63). Read this file first, then the files in section 2. Claude commits and pushes itself (author, 2026-10-18), with plain messages: no Co-Authored-By line and no Claude email; never force-push.

## 1. The project in five lines
- Paper: "Does Zero-Shot Small-Model Post-Correction Help Bengali OCR? A Negative Result Across Eight Models and Two OCR Engines" (Sir's draft title, kept by the author, D59). Builds to paper/main.pdf, paper/main-single.tex and paper/overleaf-final.zip (local).
- 40 historical Bengali pages (REID2019), Tesseract 5.3.4 and EasyOCR 1.7.2; 14 zero-shot open models (0.25-8B), Qwen3-32B 4-bit on 10 pages, a fine-tuned ByT5-small, SymSpell variants, engine fusion; plain greedy decoding.
- Finding: no language model lowers CER (whole page, region level, synthetic noise down to 0.02); general models copy (Grad-CAM), Bengali-pretrained checkpoints regenerate; larger Qwen3 and the fine-tuned model fix words (WER down) but not characters; every method changes clean text; gating and fusion reduce damage and gain about 1% of CER at best; findings are stated for code-point CER (one grapheme-cluster exception).
- Lead contribution: repetition penalty and n-gram ban also act on the prompt in Transformers and make copying models look harmful (no prior paper found, D53).
- **All third-review experiments are done and in the paper** (D50-D54). Builds clean with Tectonic; all references resolve.

## 2. Read these, in order
0. `context/review-external-ahmed.md`: **external review (15 items) and status; item 4 waits for the Kaggle run.**
1. `context/review-draft-04.md`: **Bijon Sir's draft-04 corrections** and their status (all A items closed; S1-S4 and journal deferred, D61).
1b. `context/review-draft-03.md`: older item status (still open: C6, W2, M5, E7, R2).
2. `context/decisions.md`: D50-D63 (D60: draft-04 corrections; D62: plain abstract; D63: external review) (D56-D58: revisions after the external editorial review and the reviewer's second reading; D59: draft title kept, abstract in the guide's order).
3. `context/results.md`: all numbers (review3d, local runs, review3e, and the fusion-claim correction).
4. `context/engineering-notes.md`: gotchas (heredocs mangle backslashes, also in `python -c` and `python - <<EOF`: use the Edit tool or a script file).
5. `paper/WRITING-GUIDE.md`, `paper/README.md`.
6. `context/literature-leads.md` and `context/consensus-novelty-check-2026-10-18.md`: literature check; seven papers cited (D53), others optional.

## 2b. Final version (D55)
- **Send to Bijon Sir:** `paper/main.pdf`, or the single LaTeX file `paper/main-single.tex` / `paper/overleaf-final.zip` (Overleaf, compiler XeLaTeX). After ANY edit: rebuild with Tectonic, then rerun `python paper/scripts/make_single_file.py`.
- **References: published work only, no preprints** (author's rule). Models are cited by model cards. Check every new citation for a published version.

## 3. What is left (needs the author)
0. **Kaggle run for review item 4** (`notebooks/kaggle/review4_decoding_generated_only.ipynb`, HOURS 6, dataset with `ocr_eval40.jsonl`, HF_TOKEN): unzip `review4.zip` into `results/review4/`, score with `eval/matrix_analysis.py` against the plain runs (`results/gpu_eval40.jsonl`; engines tesseract,ground_truth; decoding presets penalty_gen, ngram_gen, guarded_gen) and the completion-prompt runs against the main TituLLMs Instruct runs; then add a generated table, rewrite Sec IV-D1, the claims table row and contribution 3, and update D63.
1. **Bengali examples (Table XXI):** say which changed forms are OCR errors and which are valid historical spellings (editorial review); the student-ID line in the author block.
2. **Bijon Sir's draft-04 corrections** (`context/review-draft-04.md`): C1-C3 and C6-C12 done (D60). Sir (WhatsApp, D61): only minor LaTeX corrections, no code or extra GPU for now; when the paper is ready he sends it for an external review and arranges GPU runs then if needed. So C4 and S1-S3 wait for that review; C5 is closed by the added statement; S4 trim and the journal wait for the journal choice. Author has about 8 h of Kaggle GPU left, not needed now.
3. **Page-type labels** (`data/page_types.tsv`, E8): the author has not checked them.
4. **Release (C6, R2), only with the author's go-ahead after the final pass:** GitHub release → Zenodo DOI → put the DOI into reference `rahman2026repo`; separate Zenodo dataset deposit of the per-page OCR text and model outputs under CC BY 4.0 (the paper already says they are released with it). `results/` is local only (gitignored); copies on Google Drive.
5. Optional: check the TALLIP paper's PDF if the author has ACM access (D53 caveat); arXiv preprint (WRITING-GUIDE Sec 6).

## 4. Questions to ask the author at the start of every session
1. Has the Kaggle run for review item 4 finished (`review4.zip`)?
2. Have you checked `data/page_types.tsv` (E8)?
3. Go-ahead for the GitHub release / DOI and the data deposit?

## 5. Rules carried over from the author
- Do not discard uncommitted changes. Commit and push after each coherent piece of work (no attribution lines).
- Never edit result files; regenerate tables and figures from them. No number in a table is typed by hand.
- Do not invent numbers. If a check fails or a claim is not supported, say so. Verify every new citation against its primary record.
- Keep decisions in `context/decisions.md` (decision, why, consequence). Tick items in `context/review-draft-03.md` with a *Status* note.
- Rebuild the paper with Tectonic after edits and check that the Bengali table renders.
- Heavy model runs happen on Kaggle, run by the author (about 8 h of weekly GPU quota left; reset about 2026-10-24); local runs are CPU only (about 4 GB free RAM).
