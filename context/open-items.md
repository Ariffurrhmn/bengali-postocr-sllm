# Open items

Last updated 2026-10-12, after the second review of the compiled PDF. The item-by-item status for the teacher is in [checklist-status.md](checklist-status.md). The history is in [timeline.md](timeline.md), the numbers in [results.md](results.md), and the reasons in [decisions.md](decisions.md) (D30–D34).

## Next tasks: third review (Draft 03, received 2026-10-15)
The teacher's 43-item checklist (C1–C8, W1–W9, M1–M7, E1–E8, F1–F7, R1–R4) is transcribed in [review-draft-03.md](review-draft-03.md), with the six top priorities. In order: (1) float16 check on Phi-4-mini and TituLLMs 3B Instruct, and model-card prompt formats (C1, C2); (2) 2x2 decoding ablation, guarded clean-text control and a main figure (C4, F1); (3) region-level OCR and scoring from the PAGE-XML boxes (C5); (4) remove draft-history wording and soften the Qwen3 gating claim (C3, C8); (5) noise sweep and a small fine-tuning pilot (M3, M2); (6) trim to the venue page limit (W2). The items marked "Open" in the draft-02 status table (fine-tuning, noise sweep, stronger OCR, upper-bound model, bounded WER and equivalence tests) are now explicit asks, so the "disclosed omission" route in the section below no longer covers them.

### Progress (2026-10-15)
Paper-only edits and analyses on existing outputs are done; each item's status is noted under it in the review file, and the cross-check and run plan are in [plan-draft-03.md](plan-draft-03.md). Open: C1, C2, C4 (i)-(ii), C5 and the M items need GPU runs; W2 and F5 wait for the venue page limit; C6/R2 need a decision on releasing the OCR text and outputs. The paper is now 21 pages (was 19).

## Left for the author
1. **Submit a XeLaTeX PDF.** The earlier `After feedback.pdf` was compiled with pdfLaTeX, so its Bengali table showed placeholders. Either use `paper/main.pdf` (compiled with Tectonic, a XeTeX engine), or upload `paper/overleaf-upload.zip` (built by `paper/make_overleaf_zip.py`) as a new Overleaf project. It builds under either compiler, because the Bengali table is also included as a PDF pre-rendered with XeLaTeX.
2. **Check the page-type labels** in `data/page_types.tsv` (title / verse / prose / list). The assistant assigned them from the transcriptions. Fix any you disagree with, then re-run `eval/sensitivity_analysis.py` and `eval/make_paper_tables.py`.
3. **Read the paper through once**, and check the Ethnologue figures (see `paper/CHECK-BEFORE-SUBMITTING.md`).
4. **Discuss the deferred experiments with Bijon Sir**, especially E-e (cleaner OCR) and E-g (synthetic-pair fine-tuning). The paper names all five in its limitations. A disclosed omission is a scope decision for him to accept, not a completed item.

## Not done (needs new runs)
- E-c, E-d, E-e, E-f, E-g and E2 (see the checklist status file).
- Region-level evaluation (M2).
- Measured peak GPU/CPU memory per model (T2). The table gives the weight memory implied by the parameter count and precision.
- Per-book publication years (T1). The release does not give them, and looking them up would mean searching the British Library catalogue for 37 books.

## Known loose ends
- **Result files:** the per-page outputs are not versioned (D26). Numeric summaries are now in `summaries/` (D34). A Zenodo deposit was promised in the proposal and has not been made.
- **Alignment-based preamble filter** (D23): specified but not built. The paper discloses this.
- **Unrecorded versions:** the August 2026 Tesseract build, and the PyTorch/CUDA versions of the Kaggle image (`ENVIRONMENT.md`).
- **Page 279_2_a_15_0005:** its file names differ only in case. The full dataset is at `G:\New folder\Competition_dataset_ImagesPAGEXML`, which `eval/sensitivity_analysis.py` reads.
