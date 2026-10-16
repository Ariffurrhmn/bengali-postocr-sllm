# Status (2026-10-12)

> **Superseded on 2026-10-15:** a third review (Draft 03) arrived and work on it is under way. Read [context/NEXT-SESSION.md](context/NEXT-SESSION.md) first; the text below describes the state before that review.

**Read this first.** One page on where the project is and which file is current.

## Where we are
The revision has been through a second review of the compiled PDF and fixed (2026-10-12, D34). Of the reviewer's 54 checklist items, 44 are done, 4 are partial and 6 are deferred (not performed, disclosed in the paper). The item-by-item status is in `context/checklist-status.md`. What remains is for the author: submit a XeLaTeX build, check the page-type labels, read the paper through, and agree the deferred experiments with the teacher (see `context/open-items.md`).

Result in one line: none of eight small models, prompted zero-shot, nor a dictionary spell-checker significantly lowered the character error rate of historical Bengali OCR; the general models mostly copy their input, the Bengali-specific ones regenerate it.

## What is current
| Thing | Current file |
|---|---|
| Paper source | `paper/main.tex` (+ `paper/tables/`, `paper/figures/`, `paper/fonts/`, `paper/references.bib`); local XeLaTeX build `paper/main.pdf` |
| Checklist status for the teacher | `context/checklist-status.md` |
| Versions and environment | `ENVIRONMENT.md`; numeric summaries in `summaries/` |
| How to compile and regenerate | `paper/README.md` |
| Numbers | `context/results.md` (from local `results/`) |
| Decisions and reasons | `context/decisions.md` |
| What is left to do | `context/open-items.md` |

## What is NOT current (do not edit)
- `paper/Draft_Paper_02.pdf` (local): the draft the reviewer marked up.
- `results/correction_eval.jsonl`, `bootstrap_eval.json`, `extended_eval.*`: the first version's 15-page runs, kept for the decoding ablation and the record.

## Repo map
- `ocr/`, `correction/`, `eval/`, `notebooks/`: code. `data/`: split lists and page-type labels (dataset not in repo).
- `summaries/`: numeric analysis outputs behind the tables.
- `results/`: run outputs, gitignored.
- `paper/`: the paper, source and compiled PDF.
- `context/`: decision log, results, open items, checklist status.
