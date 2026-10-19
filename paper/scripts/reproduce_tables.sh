#!/usr/bin/env bash
# Rescore the released per-page outputs and regenerate every generated table
# (external review item 15). Run from the repository root with the outputs in
# results/ (the CC BY 4.0 data release unpacked there). It does not rerun OCR or
# any model, and leaves out the analyses that need the page images
# (sensitivity_analysis.py, region_ocr_analysis.py) or a GPU; their summary
# files are part of the release. Set PY to the project's Python if needed.
set -euo pipefail
PY=${PY:-python}
export PYTHONIOENCODING=utf-8
R=results

# Main study (40 pages, eight models, two engines)
$PY eval/bootstrap.py --ocr-path $R/ocr_eval40.jsonl --correction-path $R/eval40_ocr.jsonl --out-json $R/bootstrap_gpu40.json
$PY eval/extended_analysis.py --ocr-path $R/ocr_eval40.jsonl --correction-path $R/eval40_ocr.jsonl --out-prefix $R/extended_gpu40
$PY eval/revision_analysis.py --ocr-path $R/ocr_eval40.jsonl --correction-path $R/eval40_all.jsonl --out-prefix $R/revision_gpu40
$PY eval/compare_variants.py --ocr-path $R/ocr_eval40.jsonl --original $R/gpu_eval40_ocr.jsonl --variants $R/gpu_variants40.jsonl --out-prefix $R/variants_gpu40
$PY eval/confidence_gate.py --ocr-path $R/ocr_eval40.jsonl --conf-path $R/ocr_conf40.jsonl --correction-path $R/eval40_ocr.jsonl --out-prefix $R/gated_gpu40
$PY eval/review3_analysis.py --ocr-path $R/ocr_eval40.jsonl --correction-path $R/eval40_all.jsonl --gpu-runs $R/gpu_eval40.jsonl --conf-path $R/ocr_conf40.jsonl --out-prefix $R/review3_gpu40

# Extensions
$PY eval/review3_runs_analysis.py --runs $R/review3 --x1-extra $R/review3b --ocr-path $R/ocr_eval40.jsonl --main $R/gpu_eval40.jsonl --out-prefix $R/review3_runs
$PY eval/matrix_analysis.py --ocr-path $R/region/ocr_region40.jsonl --corrections "$R/review3c/c5_*.jsonl" $R/region/dict_region40.jsonl $R/review3d/c5_qwen3-4b.jsonl --engines tesseract,easyocr --label "region-level OCR" --out-prefix $R/region_runs40
$PY eval/matrix_analysis.py --ocr-path $R/ocr_eval40.jsonl --corrections "$R/review3b/m1_*.jsonl" --engines tesseract,easyocr,ground_truth --label "new models (M1)" --out-prefix $R/m1_runs40
$PY eval/matrix_analysis.py --ocr-path $R/review3b/noise40.jsonl --corrections "$R/review3b/m3_*.jsonl" --engines noise_0.02,noise_0.05,noise_0.10,noise_0.20,noise_0.30 --label "noise sweep (M3)" --out-prefix $R/noise_runs40
$PY eval/matrix_analysis.py --ocr-path $R/ocr_eval40.jsonl --corrections $R/gpu_eval40.jsonl --engines ground_truth,tesseract,easyocr --label "main run, general-purpose models" --out-prefix $R/main_general40
$PY eval/matrix_analysis.py --ocr-path $R/ocr_eval40.jsonl --corrections $R/review3e/m2_byt5.jsonl --engines tesseract,easyocr,ground_truth --label "fine-tuned ByT5-small (M2)" --out-prefix $R/m2_runs40
$PY eval/matrix_analysis.py --ocr-path $R/ocr_eval40.jsonl --corrections $R/review3d/m7_qwen3-32b.jsonl $R/gpu_eval40.jsonl "$R/review3b/m1_qwen3-*.jsonl" --pages-file data/m7_pages.txt --engines tesseract,easyocr,ground_truth --label "Qwen3 size series on the 10 M7 pages" --out-prefix $R/m7_runs10
$PY eval/matrix_analysis.py --ocr-path $R/ocr_eval40.jsonl --corrections $R/dict_eval40.jsonl $R/m4_dict_ed1.jsonl $R/m4_dict_wiki.jsonl $R/m4_dict_wiki_ed1.jsonl --engines tesseract,easyocr,ground_truth --label "dictionary variants (M4)" --out-prefix $R/m4_dict40
$PY eval/matrix_analysis.py --ocr-path $R/ocr_eval40.jsonl --corrections $R/m6_fusion40.jsonl --engines easyocr --label "engine fusion (M6), against EasyOCR, its base" --out-prefix $R/m6_fusion40

# External-review analyses (descriptive)
$PY eval/edit_analysis.py --ocr-path $R/ocr_eval40.jsonl --out-json $R/edit_analysis40.json \
  $(for e in tesseract easyocr; do for c in gpu_eval40.jsonl:gemma-2b gpu_eval40.jsonl:qwen3-1.7b review3b/m1_qwen3-4b.jsonl:qwen3-4b \
      review3b/m1_qwen3-8b.jsonl:qwen3-8b review3b/m1_llama3.2-3b.jsonl:llama3.2-3b review3e/m2_byt5.jsonl:byt5-small-ft; do
      echo --cell $c:$e; done; done) --cell m6_fusion40.jsonl:fusion-conf:easyocr --cell m6_fusion40.jsonl:fusion-margin:easyocr
$PY eval/output_filter.py --ocr-path $R/ocr_eval40.jsonl --corrections $R/gpu_eval40.jsonl --out-json $R/output_filter40.json
$PY eval/page_deltas.py --ocr-path $R/region/ocr_region40.jsonl --corrections $R/review3c/c5_phi4-mini.jsonl --engine easyocr --out-json $R/region_phi4-mini_easyocr_pages.json

# Tables
$PY paper/scripts/make_paper_tables.py
git status --short paper/tables
