# Next session: start here

Handoff updated 2026-10-18 (evening). Read this file first, then the files in section 2. Claude commits and pushes itself (author, 2026-10-18), with plain messages: no Co-Authored-By line and no Claude email; never force-push.

## 1. The project in five lines
- Paper: working title "Small Language Models for Historical Bengali OCR Correction: Word-Level Gains, Character-Level Damage, and a Decoding Pitfall" (D56; the old one claimed a decoding "default", which the editorial review showed is wrong) (`paper/main.tex`, author Arifur Rahman, course CSE403, reviewer **Bijon Sir**). Alternatives in D52; the author checks with Bijon Sir. `CITATION.cff` and `.zenodo.json` still carry the old title.
- 40 historical Bengali pages (REID2019), Tesseract 5.3.4 and EasyOCR 1.7.2; 14 zero-shot open models (0.25-8B), Qwen3-32B 4-bit on 10 pages, a fine-tuned ByT5-small, SymSpell variants, engine fusion; plain greedy decoding.
- Finding: no language model lowers CER (whole page, region level, synthetic noise down to 0.02); general models copy (Grad-CAM), Bengali-pretrained checkpoints regenerate; larger Qwen3 and the fine-tuned model fix words (WER down) but not characters; every method changes clean text; gating and fusion avoid damage but gain under 1%.
- Lead contribution: repetition penalty and n-gram ban also act on the prompt in Transformers and make copying models look harmful (no prior paper found, D53).
- **All third-review experiments are done and in the paper** (D50-D54). Builds clean with Tectonic; all references resolve.

## 2. Read these, in order
1. `context/review-draft-03.md`: item status (still open: C6, W2, M5, E7, R2).
2. `context/decisions.md`: D50-D57 (D56-D57: revision after the external editorial review).
3. `context/results.md`: all numbers (review3d, local runs, review3e, and the fusion-claim correction).
4. `context/engineering-notes.md`: gotchas (heredocs mangle backslashes, also in `python -c` and `python - <<EOF`: use the Edit tool or a script file).
5. `paper/WRITING-GUIDE.md`, `paper/README.md`.
6. `context/literature-leads.md` and `context/consensus-novelty-check-2026-10-18.md`: literature check; seven papers cited (D53), others optional.

## 2b. Final version (D55)
- **Send to Bijon Sir:** `paper/main.pdf`, or the single LaTeX file `paper/main-single.tex` / `paper/overleaf-final.zip` (Overleaf, compiler XeLaTeX). After ANY edit: rebuild with Tectonic, then rerun `python paper/scripts/make_single_file.py`.
- **References: published work only, no preprints** (author's rule). Models are cited by model cards. Check every new citation for a published version.

## 3. What is left (needs the author)
0. **Bengali examples (Table XXI):** say which changed forms are OCR errors and which are valid historical spellings (editorial review); the student-ID line in the author block.
1. **Title:** the author picks one (D52) with Bijon Sir; then update `main.tex`, README, `CITATION.cff`, `.zenodo.json`.
2. **Bijon Sir's reading** of the new draft (`paper/main.pdf`): Grad-CAM section, title, anonymous review (the repo is public under the author's name).
3. **Page-type labels** (`data/page_types.tsv`, E8): the author has not checked them.
4. **Release (C6, R2), only with the author's go-ahead after the final pass:** GitHub release → Zenodo DOI → put the DOI into reference `rahman2026repo`; separate Zenodo dataset deposit of the per-page OCR text and model outputs under CC BY 4.0 (the paper already says they are released with it). `results/` is local only (gitignored); copies on Google Drive.
5. Optional: Overleaf zip (`python paper/scripts/make_overleaf_zip.py`); check the TALLIP paper's PDF if the author has ACM access (D53 caveat); arXiv preprint (WRITING-GUIDE Sec 6).

## 4. Questions to ask the author at the start of every session
1. Which title? Any news from Bijon Sir?
2. Have you checked `data/page_types.tsv` (E8)?
3. Go-ahead for the GitHub release / DOI and the data deposit?

## 5. Rules carried over from the author
- Do not discard uncommitted changes. Commit and push after each coherent piece of work (no attribution lines).
- Never edit result files; regenerate tables and figures from them. No number in a table is typed by hand.
- Do not invent numbers. If a check fails or a claim is not supported, say so. Verify every new citation against its primary record.
- Keep decisions in `context/decisions.md` (decision, why, consequence). Tick items in `context/review-draft-03.md` with a *Status* note.
- Rebuild the paper with Tectonic after edits and check that the Bengali table renders.
- Heavy model runs happen on Kaggle, run by the author (about 8 h of weekly GPU quota left; reset about 2026-10-24); local runs are CPU only (about 4 GB free RAM).
