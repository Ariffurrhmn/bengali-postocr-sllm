# Project context

Working memory for this project, so decisions and the reasons behind them survive between sessions (human or AI). Code comments say *what* the code does; these files say *why* the project is shaped the way it is.

Last updated: 2026-10-15. **New session? Read [NEXT-SESSION.md](NEXT-SESSION.md) first.** See also `../STATUS.md`.

## Files

- [NEXT-SESSION.md](NEXT-SESSION.md): handoff for the next session: state, task list, questions to ask the author.
- [review-draft-03.md](review-draft-03.md) and [plan-draft-03.md](plan-draft-03.md): the third review (43 items + X1) with status notes, and the plan.
- [decisions.md](decisions.md): every design decision, with the reasoning, the alternatives considered, and its consequences. **Start here.** D30–D35 cover the October revision; D36–D38 the third review.
- [results.md](results.md): the numbers, current study first, the superseded first version after.
- [timeline.md](timeline.md): dated history with commit hashes.
- [engineering-notes.md](engineering-notes.md): bugs found and fixed, plus environment gotchas that will bite again.
- [open-items.md](open-items.md): review-checklist status and what is left.

## Status in one paragraph

The study asks whether small open language models, used zero-shot as a post-processing layer, can reduce the error rate of OCR output for historical Bengali print, and whether the effect is the same across OCR engines. On all 40 evaluation pages, none of eight models (0.25–3.8B parameters) nor a dictionary spell-checker significantly lowered CER on Tesseract or EasyOCR output. General-purpose models mostly copy their input; Bengali-specific models regenerate it; every method damages correct text. A repetition penalty that also acts on the prompt had made the first version's models look far more harmful than they are. The first draft (August 2026, 15 pages) was rejected; the revised paper (`paper/main.tex`) answers the reviewer's 54-item checklist.

## Where things live

- **Code:** this repo, public at https://github.com/Ariffurrhmn/bengali-postocr-sllm (branch `master`).
- **Dataset:** not in the repo. On Google Drive as `My Drive/Dataset/REID2019.zip`; the 40 evaluation pages are extracted locally in `data/REID2019/` (gitignored).
- **Frozen split:** `data/split_dev.txt` (10), `data/split_eval.txt` (40), `data/excluded_no_ground_truth.txt` (29).
- **Run outputs:** `results/` (gitignored). Colab and Kaggle runs write to, or are copied into, `My Drive/bengali-postocr-results/`. Current files: `ocr_eval40.jsonl`, `gpu_eval40.jsonl`, `gpu_variants40.jsonl`, `dict_eval40.jsonl`, `ocr_conf40.jsonl`, and the analyses derived from them.
- **Paper:** `paper/main.tex`, compiled in Overleaf with XeLaTeX (see `paper/README.md`).
- **Compute:** model runs on a free Kaggle T4 GPU (`notebooks/kaggle_*.ipynb`); OCR on a free Colab CPU runtime; analysis on the local machine (D30).

## Conventions

- Random seed **403** everywhere (split, sample, bootstrap).
- Scoring always goes through `eval/metrics.py` (NFC + whitespace collapse).
- Every sweep writes one JSON line per result and flushes immediately; re-running the same command resumes.
- Scripts that print Bengali re-wrap stdout as UTF-8 (Windows console).
