# Figures and tables for the revised paper

Everything here is generated from the result files, or hand-checked against the source papers. **Captions and paper text are yours to write.** These are only the floats.

## How to use in Overleaf
1. Upload `figures/`, `tables/` and `references.bib` into your Overleaf project.
2. Add `\input{tables/preamble}` after `\documentclass{...}` (it loads booktabs, xcolor and graphicx, and defines `\best`).
3. Put each `\input{tables/...}` or figure where the plan says (see `reviewer-feedback/rewrite-plan.md`).
4. To test-compile everything first: upload this whole folder to a new Overleaf project and compile `preview.tex`.

## Files

| File | What it is | Fixes reviewer item | Goes in |
|---|---|---|---|
| `figures/fig_method.pdf` | Methodology diagram | #12 "Where is methodology diagram?" | §III, full width (`figure*`) |
| `figures/fig_dataset.pdf` | (a) 81 → 52 → 50 → 10/40 → 15 pages; (b) sample page with ground-truth regions | #36 "Add demo/relation fig of dataset" | §III-B, full width |
| `tables/literature.tex` | Prior work vs this study (language, models, setting, input error, did it help) | #10 "Add tables if possible" | §II, end of related work |
| `tables/baseline.tex` | Uncorrected baselines incl. cMER | #40 "baselines of what?" | §IV-A |
| `tables/eval_matrix.tex` | Full matrix with bootstrap CIs, **Holm-corrected Wilcoxon p**, truncation; full width, so nothing is cut off | #42 color the best, #43 cut-off column, #3 statistical test | §IV-B |
| `tables/diagnostics.tex` | Length, script share, GT retention (old Table IV, best values marked) | #42 | §V |
| `tables/damage.tex` | **New:** cMER, correct words broken, wrong words fixed, change ratio, invented runs, length safeguard | #3 statistical/"XAI", #2 contribution | §V |
| `references.bib` | Corrected and new references (the single canonical bib) | #11, #50 | bibliography |

## Still for you to write
- Every caption's descriptive sentence (what the figure or table shows and what to notice). The reviewer asked for definitions and descriptions on Fig. 2 and Fig. 4 (#41, #46); the definitions are in each table's footnote and in `context/results.md`.
- Fill in the Rabby et al. author list in `references.bib`.

## Regenerate
```
cd eval
python extended_analysis.py            # results/extended_eval.{json,txt}
python make_paper_tables.py            # paper/tables/*.tex (except literature.tex, preamble.tex)
python make_paper_figures.py --page-image <279_42_B_41_0003.tif> --page-xml <279_42_B_41_0003.xml>
```
The dataset isn't in the repo. The sample page is in `REID2019.zip` on Google Drive (`My Drive/Dataset/`).

## main.tex (added 2026-10-05)
A complete revised draft that uses everything above: `main.tex` + `tables/` + `figures/` + `references.bib`. Compile with pdfLaTeX (Overleaf). Read `CHECK-BEFORE-SUBMITTING.md` first. `python check_refs.py` verifies all citations and cross-references.

## Which version is current (corrected 2026-10-09)
- **`main.tex` is the current source.** It is the version that compiled to `Draft_Paper_02.pdf` (Overleaf, 5 Oct 2026, 09:23). It was restored from the v2 zip, because the earlier `main.tex` here was an older wording pass.
- `Draft_Paper_02.pdf`: local copy of the compiled PDF. Gitignored.
- `archive/`: older source and the two Overleaf zips. Not current. `main_v1_...tex` is the pre-v2 wording; `EDITORIAL-NOTES.md` describes the pass that produced v1, so its quotes may not match `main.tex`.
- `CHECK-BEFORE-SUBMITTING.md`: item 1 (never compiled) is out of date; it did compile.
