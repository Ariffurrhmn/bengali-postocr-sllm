# Timeline

All dates 2026. Commit hashes refer to `master`.

## Planning
- **by Jun 9:** Project proposal written (hypothesis: ≥10% CER reduction; 4–5 models; two engines).
- **Jun 10:** Literature review.
- **~Jul 21:** Base paper (Kanerva et al. 2025) and dataset (British Library REID2019) chosen. D02, D04.
- **Jul 23:** Course meeting 1: final methodology and dataset.
- **Jul 25:** Methodology gaps resolved in a grilling session: 10% threshold dropped (D03), paired bootstrap chosen (D22), Gemma included. The 29 pages without ground truth were found (D05).

## Build (Jul 25–26)
- `651d602` Initial scaffolding. Public GitHub repo created.
- `b445580` 10/40 split frozen (D07). `ed73163` Pool selection switched to region coverage (D06).
- `f1daa54` Local environment verified (project-local tessdata, UTF-8 console).
- `7a424e4` OCR runner, CER/WER module with tests, baseline scoring.
- `9c9dfc2` Correction model wrappers plus dry run. Local run impossible (C: drive full), so moved to Colab (D17).
- `d1ab1b9`, `b3bf75a` Colab notebook, dataset path made configurable.
- `0ccc144` Fix: chat template input missing attention_mask.
- `730dc09`, `9503f0c` Colab auto-committed the notebook; merged.
- `cf8644b` Fix: notebook re-runs created a nested clone.
- `0fad60c` Repetition guards (D12).
- `6ac582a` Resumable sweep, chunking, skip guard (D16, D18, D19).
- First dev sweep (TituLLMs + BanglaT5): every run worse than baseline.
- `2bb8331` Correction scoring script.
- `47f9111` bf16 (D14). `cc978bf` Input-scaled max_new_tokens (D13).
- `d050db1` Speed diagnostic. `e79cbec` Fix: 1 CPU thread on Colab (D15).
- `4c181d3` Live progress heartbeat.
- Dev results for 4 models, all worse. Phi-3 abandoned (D20).
- Eval scope set: 15 pages, chunked (D19).

## Eval run (Jul 26–28)
- `63d861f` 15-page eval subsample, Drive-backed resumable sweep, scoring overrides.
- `5934a74` Fix: case-mismatched .tif/.xml crashed eval OCR on Linux.
- **Jul 27–28:** Eval sweep completed, 120/120 runs, after one disconnect at 84/120.

## Analysis and paper (Aug)
- **~Aug 11:** Course schedule slipped one week (draft ~Aug 20, final ~Aug 27).
- `45fd1cb` Paired bootstrap plus figures (D22).
- `2953ddb` Qualitative side-by-side and failure diagnostics (D24).
- `839a772` README corrected: alignment claim removed (D23), dataset provenance added.
- **~Aug 16:** 9-page draft paper finished in Overleaf and taken to the instructor. Table III was checked against `bootstrap_eval.json` and matched exactly.
- **~Aug 27:** Final paper submitted.

## After submission
- **Sep:** Paper returned with corrections and suggestions (to be recorded in open-items.md).
- **Sep 26:** `context/` folder created from the session history and result files.
- **Oct 4:** Feedback transcribed; references checked; 14 candidate papers reviewed; extended analysis run (D27).
- **Oct 4:** Paper assets generated in `paper/` (method + dataset figures, 5 LaTeX tables, references.bib) (D28).
- **Oct 5:** Full revised draft written (`paper/main.tex`, D29); verification checklist in `paper/CHECK-BEFORE-SUBMITTING.md`.
