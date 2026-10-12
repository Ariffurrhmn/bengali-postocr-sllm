# Checklist status (Bijon Sir's review of 2026-10-08)

Status after the second round of fixes (2026-10-12). Statuses:
- **Done:** the change is in the paper.
- **Partial:** part of the item remains.
- **Deferred:** not performed. The paper says so, but it is not complete.

Section and table numbers refer to the XeLaTeX build of `paper/main.tex`.

## Done (44)
C1, C2, C3, C5, M1, M3, M4, M5, M6, M7, M8, E-a, E-b, E1, E3, E4, E5, E6, E7, E8, T1, T3, T4, T5, F1–F5, V1–V6, W1–W8, R1

Notes on items that were partial after the first round:
- **C5:** "Input noise is not the main cause" became "Input noise alone does not explain the failure". The clean-text control shows that noise is not needed for the damage; how much noise adds is not measured.
- **E3:** new Fig. 4 plots corrected CER against baseline CER, with the identity line.
- **E5:** tokens per word added to Table III.
- **E7:** page types are labelled (10 title, 8 verse, 13 prose, 9 list) and the CER change is broken down by type (Table XIV). The labels come from the transcriptions and the author still needs to check them.
- **E8:** length-rule and edit-distance-cap p-values are now shown (Table XV). The cap brings nothing significantly below raw OCR.
- **T1:** page types added. Per-book years are not in the release, so the table gives the 1713–1914 range and says so.
- **T4:** the Bengali examples render under XeLaTeX. The source now refuses to build under pdfLaTeX.
- **T5:** raw OCR length ratios are filled in Table X.
- **V5:** the script shares in Table X were averaged per output with empty outputs counted as 0% (TituLLMs 3B Instruct showed 63/0/3%). They are now pooled over characters (97/0/3%), with a separate "Empty" column (27/80).
- **V6 / R1:** "first" was removed. The novelty claim is narrowed to a controlled evaluation of zero-shot open models below 4B on historical Bengali print across two engines. The paper now cites Ahmed et al. (2016, dictionary and edit-distance correction of Bangla OCR) and BnMMLU (Findings ACL 2026, GPT-3.5 copy-editing of EasyOCR text of modern textbooks, not measured against a reference).
- **W1:** the abstract was cut to its setup and three numbers.
- **W6 / W8:** the tokenisation and "at least as good" statements were qualified. The gated Qwen3 result is named as the one significant CER reduction.

## Partial (4)
- **C4:** all 40 pages and SymSpell are evaluated. The development pages were used for debugging and the few-shot examples. Prompt and decoding were not tuned on them as a separate step.
- **M2:** whole-page OCR, the PAGE-XML reference order and the absence of order matching are now stated. The claim that the mismatch cancels in paired differences was removed. Two sensitivity checks agree with the main results: a reference in the declared reading order, and an order-free word recall (Table IX). A region-level evaluation was not run.
- **T2:** the table now fits the page. It has checkpoint@revision, parameters, tokens/char, tokens/word, chunk size, decoding, CPU and GPU latency, and the GPU weight memory implied by parameters × precision. Measured peak memory was not recorded.
- **W9:** the paper and `ENVIRONMENT.md` list the dataset record, versions, checkpoint revisions, chat template and generation settings. `summaries/` holds the numeric outputs behind every table. The PyTorch and CUDA versions of the Kaggle image were not recorded.

## Deferred: not performed (6)
Each is named in the paper's limitations, under "Experiments not performed".

| Code | Experiment | Why not done |
|---|---|---|
| E-c | A 27–70B or frontier model on the same pages | Outside the zero-shot small-model scope and the free-compute budget |
| E-d | Fine-tuned BanglaT5 / mT5 / ByT5 | Needs aligned training data and training compute |
| E-e | A stronger or cleaner OCR input | Not run. It would test whether correction helps once input CER falls towards 0.15 |
| E-f | Direct transcription by a multimodal model (optional) | Optional in the review. Not run |
| E-g | Synthetic-pair fine-tuning pilot (strongly recommended) | Not run. Named as the next planned experiment |
| E2 | Synthetic noise sweep at CER 0.02–0.30 | Not run. The clean-text control covers only the 0.00 end |

**Question for Sir:** is the zero-shot scope acceptable for this paper, without E-e and E-g?
