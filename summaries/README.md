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
| `*_environment.txt` | the Kaggle notebooks | package versions of each GPU session |
