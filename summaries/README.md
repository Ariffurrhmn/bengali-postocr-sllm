# Summary statistics

Copies of the numeric analysis outputs in `results/` from which the paper's tables and figures are generated (`paper/scripts/`). They contain no page text, only numbers. The script and command that produce each file are listed in `paper/README.md`; if an analysis is re-run, copy the new file here.

| File | Produced by | Covers |
|---|---|---|
| `bootstrap_gpu40.json`, `extended_gpu40.json`, `revision_gpu40.json` | `eval/bootstrap.py`, `extended_analysis.py`, `revision_analysis.py` | main 40-page comparison, damage measures, robustness |
| `sensitivity_gpu40.json`, `variants_gpu40.json`, `gated_gpu40.json` | `sensitivity_analysis.py`, `compare_variants.py`, `confidence_gate.py` | reading order and word recall, prompt variants, confidence gating |
| `model_profile.json`, `decoding_ablation.json` | `model_profile.py`, `compare_decoding.py` | tokenisation and cost, the 15-page decoding comparison |
| `review3_gpu40.json` | `review3_analysis.py` | third-review analyses on existing outputs |
| `review3_runs.json` | `review3_runs_analysis.py` | precision, prompt format, 2x2 decoding, Grad-CAM |
| `region_ocr40.json`, `region_runs40.json` | `region_ocr_analysis.py`, `matrix_analysis.py` | region-level OCR and correction on it |
| `m1_runs40.json`, `noise_runs40.json`, `main_general40.json` | `matrix_analysis.py` | new models, noise sweep, main run in the same format |
| `m4_dict40.json`, `m4_lexicon_coverage.json` | `matrix_analysis.py`, `lexicon_coverage.py` | dictionary variants (edit distance, Wikipedia lexicon) and lexicon coverage |
| `m6_fusion40.json` | `matrix_analysis.py` | Tesseract and EasyOCR fused by word confidence |
| `m2_runs40.json` | `matrix_analysis.py` | fine-tuned ByT5-small (review3e) on both engines and clean text |
| `m7_runs10.json` | `matrix_analysis.py` | Qwen3 size series up to 32B (4-bit) and the main models on 10 pages |
| `review4_runs.json` | `review4_runs_analysis.py` | constraints on generated tokens only; completion prompt for the TituLLMs Instruct checkpoints |
| `edit_analysis40.json`, `output_filter40.json` | `edit_analysis.py`, `output_filter.py` | character errors per word group; the commentary filter |
| `m6_fusion40_family4.json` | `matrix_analysis.py` | fusion scored in the first 4-cell Holm family (D51, D60) |
| `region_phi4-mini_easyocr_pages.json` | `page_deltas.py` | per-page CER changes of Phi-4-mini on EasyOCR region text |
| `*_environment.txt` | the Kaggle notebooks | package versions of each GPU session |
