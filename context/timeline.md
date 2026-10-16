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
- **Oct 8:** Second review returned as a 54-item checklist (Bijon Sir): decoding confound, title, model sizes, coverage, baselines, methodology gaps, new models and evaluations, tables, figures, checks, writing and references.

## Revision (October 2026)
- **Oct 9:** Decoding ablation switch added (`a07c202`). Smoke test showed the repetition penalty and n-gram ban, which act on the prompt, caused most of the damage.
- **Oct 9–10:** Plain-decoding reruns of the four original models on the 15-page sample on Colab CPU; Qwen3 1.7B added (`b0c75c9`, `d1e4baa`). Methods/Results first rewritten for plain decoding (`df32339`).
- **Oct 10:** GPU support (`845d0ba`); decision to run the 40-page study on a free GPU (D30). OCR of the 25 new pages on Colab CPU; OCR drift found (August Tesseract output not reproducible), so all 40 pages use October OCR.
- **Oct 11:** Phi-4-mini and the TituLLMs Instruct checkpoints added; the original TituLLMs found to be a base model (D31) (`0fe934f`, `640c6d2`). Kaggle T4 run: 960/960 results in one background session. 40-page analyses, SymSpell baseline, new tables and figures (`e0341ec`). Prompt and chunking variants set up (`34b5e2d`, D32).
- **Oct 12:** Variants run (1,280 results) and comparison (`f9bfabb`). Methodology, Results, Failure Analysis, Discussion and Conclusion rewritten (`66399c6`, `35ea653`, `aafb615`); difficulty-band bug fixed. Confidence-gated correction from a re-OCR with confidences (D33; `bcef0ac`, `787ca72`). Title, abstract and introduction (`36ab1ed`), Related Work and references (`5f9779f`), Bengali example table and font (`26c8cf9`). Repo and context docs cleaned up.
- **Oct 12 (evening):** Second review of the compiled PDF (`After feedback.pdf`) found that it had been built with pdfLaTeX (Bengali placeholders), that Table III ran off the page, a script-share aggregation bug, an unsupported claim about the region mismatch, a broad novelty claim, and missing safeguard p-values. All were fixed without new model runs (D34). The fixes added sensitivity checks (reading order, word recall, page types), a corrected-vs-baseline figure, `ENVIRONMENT.md`, `summaries/`, and `context/checklist-status.md`. Tectonic was installed so the paper can be compiled and checked locally.
- **Oct 12 (late):** Repo cleaned to the current study (D35); compiled paper added; Overleaf upload zip added after an Overleaf build failure (the old zip lacked the new tables). Overleaf then compiled with pdfLaTeX despite a `latexmkrc`, so the Bengali example table is now also pre-rendered by XeLaTeX as a PDF that pdfLaTeX includes.
- **Oct 15:** Third review (Draft 03) transcribed: 43 items. Cross-check and run plan written (`context/plan-draft-03.md`). Paper-only edits and analyses on existing outputs done (D36): history wording removed, Bengali-pretrained wording, wMER, equivalence tests, Table III split, decoding figure, digit/normalisation and lexicon checks, empty-output timing. GPU runs for C1, C2, C4, C5 and the M items not yet started.
- **Oct 16:** First third-review GPU session (Kaggle T4 x2, about 9.5 h; the first attempt died after one minute because HF_TOKEN was not attached). Scored (D39): fp16 not the cause of empty outputs (C1); completion prompt helps TituLLMs 1B but not enough (C2); 40-page 2x2 decoding ablation (C4); Grad-CAM maps for six models (X1). Paper not yet updated.
