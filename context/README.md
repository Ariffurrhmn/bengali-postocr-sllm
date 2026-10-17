# Project context

Working memory for this project, so decisions and the reasons behind them survive between sessions (human or AI). Code comments say *what* the code does; these files say *why* the project is shaped the way it is.

**New session? Read [NEXT-SESSION.md](NEXT-SESSION.md) first.**

## Files
- [NEXT-SESSION.md](NEXT-SESSION.md): handoff: current state, task list, questions to ask the author.
- [review-draft-03.md](review-draft-03.md): the third review (43 items + X1), each with dated status notes.
- [decisions.md](decisions.md): every design decision with its reasons, alternatives and consequences (D01-D48). D36 onwards cover the third review; D48 maps the old file paths to the reorganized layout.
- [results.md](results.md): the numbers, current study first, the superseded first version after.
- [timeline.md](timeline.md): dated history with commit hashes.
- [engineering-notes.md](engineering-notes.md): bugs found and fixed, and environment gotchas that will bite again.

The second-review checklist status, the old open-items list and the third-review run plan were removed on 2026-10-17 (D48); they are in git history and their content is covered by the files above.

## Where things live
- **Code:** this repo, public at https://github.com/Ariffurrhmn/bengali-postocr-sllm (branch `master`).
- **Dataset:** not in the repo. On Google Drive as `My Drive/Dataset/REID2019.zip`; the full release is extracted at `G:\New folder\Competition_dataset_ImagesPAGEXML`, the 40 evaluation pages in `data/REID2019/` (gitignored). On Kaggle: private datasets `reid2019-eval40` (pages, XML, `ocr_eval40.jsonl`), `region-ocr40`, `postocr-dev10`.
- **Frozen split:** `data/split_dev.txt` (10), `data/split_eval.txt` (40), `data/excluded_no_ground_truth.txt` (29).
- **Run outputs:** `results/` (gitignored), with copies of the Colab outputs in `My Drive/bengali-postocr-results/`. Kaggle session outputs are unpacked into `results/review3/`, `review3b/`, `review3c/`, `review3d/`, the region OCR into `results/region/`.
- **Paper:** `paper/main.tex`, built locally with Tectonic (see `paper/README.md`).
- **Compute:** model runs on Kaggle T4 x2 (`notebooks/kaggle/`); the first-version CPU runs on Colab (`notebooks/colab/`); analysis on the local machine (about 4 GB free RAM, so no models above about 0.6B locally).

## Conventions
- Random seed **403** everywhere (split, samples, bootstrap, noise).
- Scoring always goes through `eval/metrics.py` (NFC + whitespace collapse).
- Every sweep writes one JSON line per result and flushes immediately; re-running the same command resumes.
- Every command in `paper/README.md` runs from the repository root.
- Scripts that print Bengali re-wrap stdout as UTF-8 (Windows console).
- Result files are never edited; tables and figures are regenerated from them.
