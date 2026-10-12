# Paper

`main.tex` is the current source of the revised paper. Every table and figure in it is generated from the result files, except `tables/literature.tex` and `tables/preamble.tex`, which are hand-written.

## Compile
The paper needs **XeLaTeX**. The example table (`tables/examples.tex`) sets Bengali text in `fonts/NotoSerifBengali-Regular.ttf` (Noto, SIL Open Font License). Under pdfLaTeX the build stops with an error that says so.
- **Overleaf:** run `python make_overleaf_zip.py`, then on Overleaf choose New Project → Upload Project and upload `overleaf-upload.zip`. `latexmkrc` makes Overleaf build with XeLaTeX whatever the menu says. Don't zip the folder with Windows' "Send to → Compressed folder" or PowerShell: they write backslashes, and Overleaf then loses `tables/` and `figures/`.
- **Locally:** Tectonic, a XeTeX engine, is installed through Scoop. Run `tectonic -X compile main.tex` in this folder; it writes `main.pdf` (gitignored).

`python check_refs.py` checks that every citation and cross-reference resolves.

## Regenerate tables and figures
Run from `eval/` with the project venv (`../.venv/Scripts/python.exe`) after the analyses (each script's docstring gives its exact inputs). Copy the updated JSON outputs to `summaries/`.
```
python bootstrap.py --ocr-path ../results/ocr_eval40.jsonl --correction-path ../results/eval40_ocr.jsonl --out-json ../results/bootstrap_gpu40.json
python extended_analysis.py --ocr-path ../results/ocr_eval40.jsonl --correction-path ../results/eval40_ocr.jsonl --out-prefix ../results/extended_gpu40
python revision_analysis.py --ocr-path ../results/ocr_eval40.jsonl --correction-path ../results/eval40_all.jsonl --out-prefix ../results/revision_gpu40
python compare_decoding.py --ocr-path ../results/ocr_eval.jsonl --guarded ../results/correction_eval.jsonl --plain ../results/ablation_plain_decoding.jsonl --out-json ../results/decoding_ablation.json
python compare_variants.py --ocr-path ../results/ocr_eval40.jsonl --original ../results/gpu_eval40_ocr.jsonl --variants ../results/gpu_variants40.jsonl --out-prefix ../results/variants_gpu40
python confidence_gate.py --ocr-path ../results/ocr_eval40.jsonl --conf-path ../results/ocr_conf40.jsonl --correction-path ../results/eval40_ocr.jsonl --out-prefix ../results/gated_gpu40
python model_profile.py --ocr-path ../results/ocr_eval40.jsonl --cpu-runs ../results/ablation_plain_decoding.jsonl --gpu-runs ../results/gpu_eval40.jsonl --out ../results/model_profile.json
python sensitivity_analysis.py --ocr-path ../results/ocr_eval40.jsonl --correction-path ../results/eval40_ocr.jsonl --dataset-dir "G:/New folder/Competition_dataset_ImagesPAGEXML" --out-prefix ../results/sensitivity_gpu40
python make_paper_tables.py
python make_revision_figures.py --ocr-path ../results/ocr_eval40.jsonl --correction-path ../results/eval40_ocr.jsonl --bootstrap-json ../results/bootstrap_gpu40.json --out-dir ../paper/figures
python make_paper_figures.py --page-image ../data/REID2019/279_42_B_41_0003.tif --page-xml ../data/REID2019/279_42_B_41_0003.xml
python make_example_table.py --ocr-path ../results/ocr_eval40.jsonl --correction-path ../results/eval40_ocr.jsonl --out ../paper/tables/examples.tex
```

## Files
- `tables/`: one `.tex` per table (`\input` in `main.tex`); `preamble.tex` holds the packages, the `\best` macro and the Bengali font setup.
- `figures/`: method and dataset diagrams, the change-in-CER plot (`fig_delta.png`), and the per-page scatters (`fig_corrected.png`: corrected vs baseline CER; `fig_scatter.png`: change vs baseline CER).
- `Draft_Paper_02.pdf` (local, gitignored): the draft the reviewer marked up.
- `CHECK-BEFORE-SUBMITTING.md` (local, gitignored): what still needs a human check.
- `main.pdf`: the compiled paper (Tectonic/XeLaTeX), committed so readers need not compile it.
- `latexmkrc`, `make_overleaf_zip.py`: make Overleaf build with XeLaTeX, and build a correct upload zip.
