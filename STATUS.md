# Status (2026-10-09)

**Read this first.** One page on where the project is and which file is current.

## Where we are
Experiments are finished and the paper is a complete draft. Result: zero-shot correction with four small models (Gemma 2B, Llama 3.2 1B, TituLLMs 1B, BanglaT5) made Bengali OCR output worse on both Tesseract and EasyOCR (15 of 16 comparisons significant). The first submission (Aug 2026) was rejected by the reviewer; the revised draft responds to all 50 comments.

## What is current
| Thing | Current file |
|---|---|
| Paper source | `paper/main.tex` (+ `paper/tables/`, `paper/figures/`, `paper/references.bib`) |
| Compiled paper | `paper/Draft_Paper_02.pdf` (also Drive: Group 2 / Draft Paper / `222_CSE403_Draft_Paper_02.pdf`) |
| Numbers | `results/` (local only; `extended_eval.txt` is the latest analysis) |
| Decisions and reasons | `context/decisions.md` |
| What is left to do | `context/open-items.md` |
| Reviewer comments and responses | `reviewer-feedback/` |

## What is NOT current (do not edit)
- `paper/archive/`: older sources and Overleaf zips.
- `222_CSE403_Draft_Paper.pdf` in Drive (the August submission).

## Repo map
- `ocr/`, `correction/`, `eval/`, `notebooks/`: code. `data/`: split lists only (dataset not in repo).
- `results/`: run outputs, gitignored.
- `paper/`: the paper. `reviewer-feedback/`: the review and our response material.
- `context/`: the decision log. `Group 2.lnk`: shortcut to the Drive folder.
