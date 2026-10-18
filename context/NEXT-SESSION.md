# Next session: start here

Handoff updated 2026-10-18. Read this file first, then the files in section 2. Claude commits and pushes itself (author, 2026-10-18), with plain messages: no Co-Authored-By line and no Claude email.

## 1. The project in five lines
- Paper: working title "Small Language Models Do Not Correct Historical Bengali OCR, and a Decoding Default Makes Them Look Worse" (`paper/main.tex`, author Arifur Rahman, course CSE403, reviewer **Bijon Sir**). Two alternative titles in D52; the author checks with Bijon Sir. `CITATION.cff` and `.zenodo.json` still carry the old title.
- 40 historical Bengali pages (REID2019), Tesseract 5.3.4 and EasyOCR 1.7.2, zero-shot correction by 14 open models (0.25-8B) plus Qwen3-32B 4-bit on 10 pages, SymSpell baselines, engine fusion, plain greedy decoding.
- Finding: no language model lowers CER (whole page, region level, synthetic noise down to 0.02); general models copy (Grad-CAM), Bengali-pretrained checkpoints regenerate; larger Qwen3 fix more words (WER down) but not CER; every method damages clean text; fusion and gating avoid damage but gain under 1%.
- Lead contribution: repetition penalty and n-gram ban also act on the prompt in Transformers and make copying models look harmful.
- Reviewer's third round: `context/review-draft-03.md` (status notes; open: C6, W2, M2, M5, E7, R2).

## 2. Read these, in order
1. `context/review-draft-03.md`: item status.
2. `context/decisions.md`: D50-D52 (review3d, M2 flaw and rerun, local M4/fusion, paper rewrite).
3. `context/results.md`: all numbers, including "Third-review GPU run 4: review3d" and "Local runs, 2026-10-18".
4. `context/engineering-notes.md`: gotchas (Bash heredocs mangle backslashes, also inside `python - <<EOF`: write scripts to a file; `PYTHONIOENCODING=utf-8`; `HF_HUB_CACHE=G:/hf-cache`; keep `huggingface-hub<2`).
5. `paper/WRITING-GUIDE.md` and `paper/README.md` (every command runs from the repository root).
6. `context/literature-leads.md`: papers found with Consensus (2026-10-18), to verify and cite; Boros et al. 2024 is a must-cite.

## 3. Where things stand

### Done (2026-10-18)
- review3d scored (D50): M7 Qwen3-32B and Qwen3 4B on region text valid; summaries copied. The M2 run was invalid (test chunks contained line breaks never seen in training) and its scores are kept apart as `results/m2_runs40_review3d.*`.
- M2 fix pushed (`33091c1`): `correction/finetune_byt5.py` joins chunk lines with spaces and accumulates gradients (effective batch 8); notebook `notebooks/kaggle/review3e_finetune_rerun.ipynb` (one T4, 150 training minutes, about 3 h).
- Local M4 (Wikipedia lexicon, edit distance 1) and M6 engine fusion (D51); table `classical.tex`. Edit-list prompting dropped (author).
- Paper rewritten with everything except M2 (D52); builds with Tectonic, no overfull boxes, all references resolve, Bengali example table renders; 27 pages; secondary tables in an appendix.

### Waiting
- **review3e** on Kaggle (the author starts it; output `review3e.zip`, normally in `C:\Users\Windows 11\Downloads\`).

### Committed
Everything up to the paper rewrite is committed and pushed (`d93ed48`).

## 4. Task list for the next session (in order)
1. Unpack `review3e.zip` to `results/review3e/` (own new folder; check names for `..` first). Check `lane0.log` for errors and out-of-memory lines, `m2_train_log.json` (development CER per checkpoint against raw 0.362; `micro_batch`, `effective_batch`), and `m2_byt5.jsonl` (120 rows, no errors; truncation count). If the development CER is again above raw OCR at every checkpoint, inspect outputs as in D50 (deletions/insertions vs substitutions, missing lines) before believing it.
2. Score: the first M2 command in `paper/README.md` (`--corrections results/review3e/m2_byt5.jsonl ... --out-prefix results/m2_runs40`); copy `results/m2_runs40.json` and `results/review3e/environment.txt` (as `summaries/review3e_environment.txt`) to `summaries/`; add `m2_runs40.json` to `summaries/README.md`.
3. `python paper/scripts/make_paper_tables.py` writes `finetune.tex` (reads `results/review3e/m2_train_log.json`; its note says "steps", which are now optimizer steps of effective batch 8: check the wording). Test-render it.
4. Paper: fill the places marked `% M2` in `main.tex` (Results subsection after the large-model paragraph with `\input{tables/finetune}`; RQ5; practical implications; conclusion; abstract) and the closing sentence of "Possible explanations" ("fine-tuned versions of these models"). Mention the failed first run in one sentence in Methods (III-L, fine-tuning). Trim the abstract from 216 to about 200 words. Revisit the title if fine-tuning helps.
5. Rebuild (`cd paper; tectonic -X compile main.tex`), `python paper/scripts/check_refs.py`, check overfull boxes and the Bengali table; `python paper/scripts/make_overleaf_zip.py` if the author wants Overleaf.
6. Record: results.md, a decision entry (D54), review-draft-03 M2 status, timeline, README (last bullet), this file.
7. Commit and push (no attribution trailer). Later, with their go-ahead: update the title in `CITATION.cff` and `.zenodo.json`, GitHub release for the DOI (Zenodo is linked), DOI into `rahman2026repo`, CC BY 4.0 data deposit of the OCR text and outputs (the paper already says they are released with it).

## 5. Questions to ask the author at the start of every session
Ask only those still open, together:
1. Has review3e finished? Where is `review3e.zip`?
2. Which title (D52 has three)? Any news from Bijon Sir (title, Grad-CAM section, anonymous review)?
3. Have you checked the page-type labels in `data/page_types.tsv`? (E8)

## 6. Rules carried over from the author
- Do not discard uncommitted changes. Claude commits and pushes after each coherent piece of work, plain messages without Co-Authored-By or Claude email; never force-push.
- Never edit result files; regenerate tables and figures from them. No number in a table is typed by hand.
- Do not invent numbers. If a check fails or a claim is not supported, say so.
- Keep decisions in `context/decisions.md` (decision, why, consequence). Tick items in `context/review-draft-03.md` with a *Status* note.
- Rebuild the paper with Tectonic after edits and check that the Bengali table renders.
- Heavy model runs happen on Kaggle, run by the author (about 8 h of weekly GPU quota left after review3e; reset about 2026-10-24); local runs are CPU only (about 4 GB free RAM).
- The author does not want to babysit runs: give clear, short instructions for each Kaggle step.
