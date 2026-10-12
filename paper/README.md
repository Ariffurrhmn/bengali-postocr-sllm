# Paper

`main.tex` is the current source of the revised paper. Every table and figure in it is generated from the result files, except `tables/literature.tex` and `tables/preamble.tex`, which are hand-written.

## Compile (Overleaf)
1. Upload this folder: `main.tex`, `references.bib`, `tables/`, `figures/` and `fonts/`.
2. Set the compiler to **XeLaTeX** (Menu → Compiler). The example table (`tables/examples.tex`) sets Bengali text in `fonts/NotoSerifBengali-Regular.ttf` (Noto, SIL Open Font License). Under pdfLaTeX the paper still compiles, but that table shows a placeholder.
3. `python check_refs.py` checks that every citation and cross-reference resolves.

## Regenerate tables and figures
Run from `eval/` after the analyses (each script's docstring gives its exact inputs):
```
python bootstrap.py --ocr-path ../results/ocr_eval40.jsonl --correction-path ../results/eval40_ocr.jsonl --out-json ../results/bootstrap_gpu40.json
python extended_analysis.py --ocr-path ../results/ocr_eval40.jsonl --correction-path ../results/eval40_ocr.jsonl --out-prefix ../results/extended_gpu40
python revision_analysis.py --ocr-path ../results/ocr_eval40.jsonl --correction-path ../results/eval40_all.jsonl --out-prefix ../results/revision_gpu40
python compare_decoding.py --ocr-path ../results/ocr_eval.jsonl --guarded ../results/correction_eval.jsonl --plain ../results/ablation_plain_decoding.jsonl --out-json ../results/decoding_ablation.json
python compare_variants.py --ocr-path ../results/ocr_eval40.jsonl --original ../results/gpu_eval40_ocr.jsonl --variants ../results/gpu_variants40.jsonl --out-prefix ../results/variants_gpu40
python confidence_gate.py --ocr-path ../results/ocr_eval40.jsonl --conf-path ../results/ocr_conf40.jsonl --correction-path ../results/eval40_ocr.jsonl --out-prefix ../results/gated_gpu40
python model_profile.py --ocr-path ../results/ocr_eval40.jsonl --cpu-runs ../results/ablation_plain_decoding.jsonl --gpu-runs ../results/gpu_eval40.jsonl --out ../results/model_profile.json
python make_paper_tables.py
python make_revision_figures.py --ocr-path ../results/ocr_eval40.jsonl --correction-path ../results/eval40_ocr.jsonl --bootstrap-json ../results/bootstrap_gpu40.json --out-dir ../paper/figures
python make_paper_figures.py --page-image ../data/REID2019/279_42_B_41_0003.tif --page-xml ../data/REID2019/279_42_B_41_0003.xml
python make_example_table.py --ocr-path ../results/ocr_eval40.jsonl --correction-path ../results/eval40_ocr.jsonl --out ../paper/tables/examples.tex
```

## Files
- `tables/`: one `.tex` per table (`\input` in `main.tex`); `preamble.tex` holds the packages, the `\best` macro and the Bengali font setup.
- `figures/`: method and dataset diagrams, the change-in-CER plot (`fig_delta.png`) and the per-page scatter (`fig_scatter.png`).
- `archive/`: superseded source and Overleaf zips. Not current.
- `Draft_Paper_02.pdf` (local, gitignored): the draft the reviewer marked up.
- `CHECK-BEFORE-SUBMITTING.md` (local, gitignored): what still needs a human check.
