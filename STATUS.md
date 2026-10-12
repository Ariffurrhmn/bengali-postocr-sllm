# Status (2026-10-12)

**Read this first.** One page on where the project is and which file is current.

## Where we are
The revision is complete. All experiments for the reviewer's 54-item checklist are done (48 addressed, 6 deliberately out of scope), and the whole paper has been rewritten around the 40-page results. What remains is for the author: compile on Overleaf with XeLaTeX and read it through (see `context/open-items.md`).

Result in one line: none of eight small models, prompted zero-shot, nor a dictionary spell-checker significantly lowered the character error rate of historical Bengali OCR; the general models mostly copy their input, the Bengali-specific ones regenerate it.

## What is current
| Thing | Current file |
|---|---|
| Paper source | `paper/main.tex` (+ `paper/tables/`, `paper/figures/`, `paper/fonts/`, `paper/references.bib`) |
| How to compile and regenerate | `paper/README.md` |
| Numbers | `context/results.md` (from local `results/`) |
| Decisions and reasons | `context/decisions.md` |
| What is left to do | `context/open-items.md` |

## What is NOT current (do not edit)
- `paper/archive/`: older source and Overleaf zips.
- `paper/Draft_Paper_02.pdf` (local): the draft the reviewer marked up.
- `results/correction_eval.jsonl`, `bootstrap_eval.json`, `extended_eval.*`: the first version's 15-page runs, kept for the decoding ablation and the record.

## Repo map
- `ocr/`, `correction/`, `eval/`, `notebooks/`: code. `data/`: split lists only (dataset not in repo).
- `results/`: run outputs, gitignored.
- `paper/`: the paper.
- `context/`: the decision log.
