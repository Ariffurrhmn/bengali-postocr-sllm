# Open items

Last updated 2026-10-12. The detailed history of the revision is in [timeline.md](timeline.md), the numbers in [results.md](results.md), and the reasons in [decisions.md](decisions.md) (D30–D33 for the revision).

## Review checklist (Bijon Sir, 2026-10-08)

The checklist has 54 items. Status:
- **Done (48):** C1–C5, M1–M8, E-a, E-b, E1, E3–E8, T1–T5, F1–F5, V1–V6, W1–W9, R1. Two are partly done and say so in the paper's limitations: M2 (no separate region-level evaluation) and E7 (no breakdown by page type, because pages are not labelled verse/title/prose).
- **Skipped on purpose (6), explained in Threats to validity:** E-c (27–70B or frontier model), E-d and E-g (fine-tuning), E-e (stronger OCR engine), E-f (direct transcription by a multimodal model), E2 (synthetic noise sweep). E-g is the one to name as planned future work.

## Left for the author

1. **Compile on Overleaf with XeLaTeX** (Menu → Compiler → XeLaTeX). The paper needs `paper/fonts/NotoSerifBengali-Regular.ttf` for the example table; under pdfLaTeX it still compiles but shows a placeholder there. LaTeX is not installed locally, so the current source has never been compiled. Check that the wide tables fit.
2. **Read the whole paper once.** Every number and every claim about a cited paper was checked against the result files or the source, but the text has not been read end to end by the author.
3. **Items in `paper/CHECK-BEFORE-SUBMITTING.md`** (local only): the Ethnologue figures, and an eyeball check of the literature table.
4. **Optional:** label the 40 pages by type (verse, title, prose) to complete E7.

## Known loose ends

- **Result files are not versioned** (D26). The run outputs live in local `results/` and on Drive. Consider committing small summaries or archiving everything with a Zenodo deposit; the proposal promised a Zenodo DOI and the paper only cites GitHub.
- **Proposal hypothesis.** The proposal states a ≥10% CER reduction; the methodology dropped it (D03). The paper does not reintroduce it.
- **Alignment-based preamble filter** (D23) was specified but not built; the paper discloses this. It matters mainly for the TituLLMs Instruct models, which add commentary.
- **August Tesseract version unknown.** The 15-page CPU runs (used only for the decoding ablation and CPU timing) used Colab's Tesseract of August 2026, whose output differs from 5.3.4.
- **Page 279_2_a_15_0005** has file names that differ only in case (`279_2_a_15_0005.tif` vs `279_2_A_15_0005.xml`). `ocr/run_ocr.py` handles it; the local extraction in `data/REID2019/` holds only its `.tif`.
