# Project context

Working memory for this project, so decisions and the reasons behind them survive between sessions (human or AI). Code comments say *what* the code does; these files say *why* the project is shaped the way it is.

Last updated: 2026-10-09. See also `../STATUS.md`.

## Files

- [decisions.md](decisions.md): every design decision, with the reasoning, the alternatives considered, and its consequences. **Start here.**
- [results.md](results.md): the numbers (dev set and the final 15-page eval), taken from the result files.
- [timeline.md](timeline.md): dated history with commit hashes.
- [engineering-notes.md](engineering-notes.md): bugs found and fixed, plus environment gotchas that will bite again.
- [open-items.md](open-items.md): what is unfinished, known inconsistencies, and the post-submission feedback.

## Status in one paragraph

The study asked whether a small open-weight language model, used zero-shot on CPU as a post-processing layer, can reduce CER/WER on Bengali OCR output, and whether the effect holds across OCR engines. It can't. On 15 held-out pages of historical Bengali print, all 4 evaluated models made both CER and WER worse on both engines (16 of 16 conditions worse, 15 significant by paired bootstrap). The effect was similar across Tesseract and EasyOCR. The first draft paper (9 pages, IEEE two-column) was submitted in August 2026 and came back with corrections and suggestions (see [open-items.md](open-items.md)).

## Where things live

- **Code:** this repo, public at https://github.com/Ariffurrhmn/bengali-postocr-sllm (branch `master`).
- **Dataset:** not in the repo. Locally at `D:\Competition_dataset_ImagesPAGEXML`; on Google Drive as `My Drive/Dataset/REID2019.zip` (identical content, wrapped in one subfolder).
- **Frozen split:** `data/split_dev.txt` (10), `data/split_eval.txt` (40), `data/excluded_no_ground_truth.txt` (29).
- **Run outputs:** `results/` (gitignored, so NOT on GitHub). Colab runs wrote to `My Drive/bengali-postocr-results/`. Local copies: `results/ocr_eval.jsonl`, `results/correction_eval.jsonl`, `results/bootstrap_eval.json`, `results/figures/`, `results/examples_*.txt`.
- **Paper, proposal, literature, related papers:** kept outside this repo. The current paper source is `paper/main.tex`; it was written in Overleaf.
- **Compute:** free Google Colab, CPU runtime, via `notebooks/setup_and_dry_run.ipynb`. The local machine is for code and tests only (see decisions D17).

## Conventions

- Random seed **403** everywhere (split, eval subsample, bootstrap).
- Scoring always goes through `eval/metrics.py` (NFC + whitespace collapse).
- Every sweep writes one JSON line per result and flushes immediately; re-running the same command resumes.
- Scripts that print Bengali re-wrap stdout as UTF-8 (Windows console).
