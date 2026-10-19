# Paper

`main.tex` is the current source of the paper. Every table and figure in it is generated from the result files by the scripts in `scripts/`, except `tables/literature.tex`, `tables/preamble.tex` and the two design tables `tables/experiments.tex` (experiment map with test families) and `tables/claims.tex` (claims and their evidence, no numbers), which are hand-written. No number in a table is typed by hand.

## Compile
The paper needs **XeLaTeX**. The example table (`tables/examples.tex`) sets Bengali text in `fonts/NotoSerifBengali-Regular.ttf` (Noto, SIL Open Font License). Under pdfLaTeX the build stops with an error that says so.
- **Locally:** Tectonic, a XeTeX engine (installed through Scoop). In `paper/`: `tectonic -X compile main.tex`. It writes `main.pdf`, which is committed so readers need not compile it.
- **Overleaf:** `python paper/scripts/make_overleaf_zip.py`, then on Overleaf New Project → Upload Project with `paper/overleaf-upload.zip`. It builds with either compiler: XeLaTeX sets the Bengali table live, pdfLaTeX (Overleaf's default) includes `tables/examples_body.pdf`, the same table pre-rendered with XeLaTeX. Do not zip the folder with Windows' "Send to → Compressed folder" or PowerShell: they write backslashes, and Overleaf then loses `tables/` and `figures/`.
- **Check:** `python paper/scripts/check_refs.py` confirms that every citation and cross-reference resolves.
- **Single file (final version for the reviewer and Overleaf):** `python paper/scripts/make_single_file.py` writes `paper/main-single.tex`, one self-contained file with every table and the reference list inlined (no `references.bib` needed), and `paper/overleaf-final.zip` (that file as `main.tex`, plus the figures, the font and the pre-rendered Bengali table). Upload the zip on Overleaf (New Project → Upload Project) and set Menu → Compiler → XeLaTeX. Rerun the script after any edit to `main.tex`, a table or `references.bib`; never edit `main-single.tex` by hand.
- **References:** published work only, no preprints (author's rule, 2026-10-18). Models are cited by their Hugging Face model cards, not by arXiv technical reports.

## Regenerate analyses, tables and figures
**One command:** `PY=.venv/Scripts/python.exe bash paper/scripts/reproduce_tables.sh` rescoring the released outputs and regenerating every generated table (verified 2026-10-19: all 47 summary files and every table identical; back up `results/` first, since it rewrites the summaries). The commands below are the same steps one by one.

Run everything **from the repository root** with the project venv (`.venv/Scripts/python.exe` on Windows). The per-page run outputs live in `results/` (local, gitignored; see the root README). After an analysis is re-run, copy its JSON output to `summaries/`.

**Main study (40 pages, eight models, two engines)**
```
python eval/bootstrap.py --ocr-path results/ocr_eval40.jsonl --correction-path results/eval40_ocr.jsonl --out-json results/bootstrap_gpu40.json
python eval/extended_analysis.py --ocr-path results/ocr_eval40.jsonl --correction-path results/eval40_ocr.jsonl --out-prefix results/extended_gpu40
python eval/revision_analysis.py --ocr-path results/ocr_eval40.jsonl --correction-path results/eval40_all.jsonl --out-prefix results/revision_gpu40
python eval/compare_decoding.py --ocr-path results/ocr_eval.jsonl --guarded results/correction_eval.jsonl --plain results/ablation_plain_decoding.jsonl --out-json results/decoding_ablation.json
python eval/compare_variants.py --ocr-path results/ocr_eval40.jsonl --original results/gpu_eval40_ocr.jsonl --variants results/gpu_variants40.jsonl --out-prefix results/variants_gpu40
python eval/confidence_gate.py --ocr-path results/ocr_eval40.jsonl --conf-path results/ocr_conf40.jsonl --correction-path results/eval40_ocr.jsonl --out-prefix results/gated_gpu40
python eval/model_profile.py --ocr-path results/ocr_eval40.jsonl --cpu-runs results/ablation_plain_decoding.jsonl --gpu-runs results/gpu_eval40.jsonl --out results/model_profile.json
python eval/sensitivity_analysis.py --ocr-path results/ocr_eval40.jsonl --correction-path results/eval40_ocr.jsonl --dataset-dir "G:/New folder/Competition_dataset_ImagesPAGEXML" --out-prefix results/sensitivity_gpu40
python eval/review3_analysis.py --ocr-path results/ocr_eval40.jsonl --correction-path results/eval40_all.jsonl --gpu-runs results/gpu_eval40.jsonl --conf-path results/ocr_conf40.jsonl --out-prefix results/review3_gpu40
```

**Third-review GPU runs** (unpack each Kaggle zip into `results/review3*/`)
```
python eval/review3_runs_analysis.py --runs results/review3 --x1-extra results/review3b --ocr-path results/ocr_eval40.jsonl --main results/gpu_eval40.jsonl --out-prefix results/review3_runs
python eval/region_ocr_analysis.py --region results/region/ocr_region40.jsonl --wholepage results/region/ocr_region40_wholepage.jsonl --verify results/region/ocr_region40_verify.json --main results/ocr_eval40.jsonl --out-prefix results/region_ocr40
python eval/matrix_analysis.py --ocr-path results/region/ocr_region40.jsonl --corrections "results/review3c/c5_*.jsonl" results/region/dict_region40.jsonl results/review3d/c5_qwen3-4b.jsonl --engines tesseract,easyocr --label "region-level OCR" --out-prefix results/region_runs40
python eval/matrix_analysis.py --ocr-path results/ocr_eval40.jsonl --corrections "results/review3b/m1_*.jsonl" --engines tesseract,easyocr,ground_truth --label "new models (M1)" --out-prefix results/m1_runs40
python eval/matrix_analysis.py --ocr-path results/review3b/noise40.jsonl --corrections "results/review3b/m3_*.jsonl" --engines noise_0.02,noise_0.05,noise_0.10,noise_0.20,noise_0.30 --label "noise sweep (M3)" --out-prefix results/noise_runs40
python eval/matrix_analysis.py --ocr-path results/ocr_eval40.jsonl --corrections results/gpu_eval40.jsonl --engines ground_truth,tesseract,easyocr --label "main run, general-purpose models" --out-prefix results/main_general40
python eval/matrix_analysis.py --ocr-path results/ocr_eval40.jsonl --corrections results/review3e/m2_byt5.jsonl --engines tesseract,easyocr,ground_truth --label "fine-tuned ByT5-small (M2)" --out-prefix results/m2_runs40
python eval/matrix_analysis.py --ocr-path results/ocr_eval40.jsonl --corrections results/review3d/m2_byt5.jsonl --engines tesseract,easyocr,ground_truth --label "fine-tuned ByT5-small, review3d (flawed test format, D50)" --out-prefix results/m2_runs40_review3d
python eval/matrix_analysis.py --ocr-path results/ocr_eval40.jsonl --corrections results/review3d/m7_qwen3-32b.jsonl results/gpu_eval40.jsonl "results/review3b/m1_qwen3-*.jsonl" --pages-file data/m7_pages.txt --engines tesseract,easyocr,ground_truth --label "Qwen3 size series on the 10 M7 pages" --out-prefix results/m7_runs10
```

**Local runs without a language model** (dictionary variants, M4; engine fusion, M6)
```
python data/build_wiki_lexicon.py --out data/lexicon/bn_wiki100k.txt
python correction/dictionary_baseline.py --ocr-path results/ocr_eval40.jsonl --max-edit-distance 1 --model-key symspell-ed1 --out results/m4_dict_ed1.jsonl
python correction/dictionary_baseline.py --ocr-path results/ocr_eval40.jsonl --lexicon data/lexicon/bn_wiki100k.txt --model-key symspell-wiki --out results/m4_dict_wiki.jsonl
python correction/dictionary_baseline.py --ocr-path results/ocr_eval40.jsonl --lexicon data/lexicon/bn_wiki100k.txt --max-edit-distance 1 --model-key symspell-wiki-ed1 --out results/m4_dict_wiki_ed1.jsonl
python eval/matrix_analysis.py --ocr-path results/ocr_eval40.jsonl --corrections results/dict_eval40.jsonl results/m4_dict_ed1.jsonl results/m4_dict_wiki.jsonl results/m4_dict_wiki_ed1.jsonl --engines tesseract,easyocr,ground_truth --label "dictionary variants (M4)" --out-prefix results/m4_dict40
python eval/lexicon_coverage.py --ocr-path results/ocr_eval40.jsonl --lexicons data/lexicon/bn_50k.txt data/lexicon/bn_wiki100k.txt --out results/m4_lexicon_coverage.json
python correction/ocr_fusion.py --ocr-path results/ocr_eval40.jsonl --conf-path results/ocr_conf40.jsonl --out results/m6_fusion40.jsonl
python eval/matrix_analysis.py --ocr-path results/ocr_eval40.jsonl --corrections results/m6_fusion40.jsonl --engines easyocr --label "engine fusion (M6), against EasyOCR, its base" --out-prefix results/m6_fusion40
```

**External-review analyses (2026-10-19, descriptive)**
```
python eval/edit_analysis.py --ocr-path results/ocr_eval40.jsonl --out-json results/edit_analysis40.json --cell <file>:<model>:<engine> ...   # full list in reproduce_tables.sh
python eval/output_filter.py --ocr-path results/ocr_eval40.jsonl --corrections results/gpu_eval40.jsonl --out-json results/output_filter40.json
python eval/page_deltas.py --ocr-path results/region/ocr_region40.jsonl --corrections results/review3c/c5_phi4-mini.jsonl --engine easyocr --out-json results/region_phi4-mini_easyocr_pages.json
python eval/check_prompt_constraints.py --model Qwen/Qwen3-0.6B --out-json results/prompt_constraints_5.19.0.json   # with Transformers 5.19.0
```

**Tables and figures**
```
python paper/scripts/make_paper_tables.py
python paper/scripts/make_decoding_figure.py --runs results/review3_runs.json --out paper/figures/fig_decoding.pdf
python paper/scripts/make_attribution_figure.py --maps results/review3b --page 279_2_D_32_0006 --out paper/figures/fig_attribution.pdf
python paper/scripts/make_noise_figure.py --sweep results/noise_runs40.json --real results/main_general40.json --out paper/figures/fig_noise.pdf
python paper/scripts/make_revision_figures.py --ocr-path results/ocr_eval40.jsonl --correction-path results/eval40_ocr.jsonl --bootstrap-json results/bootstrap_gpu40.json --out-dir paper/figures
python paper/scripts/make_paper_figures.py --page-image data/REID2019/279_42_B_41_0003.tif --page-xml data/REID2019/279_42_B_41_0003.xml
python paper/scripts/make_example_table.py --ocr-path results/ocr_eval40.jsonl --correction-path results/eval40_ocr.jsonl --out paper/tables/examples.tex
python paper/scripts/render_examples_pdf.py
```

## Files
- `main.tex`, `references.bib`: source and bibliography. `main.pdf`: the compiled paper.
- `tables/`: one `.tex` per table (`\input` in `main.tex`); `preamble.tex` holds the packages, the `\best` macro and the Bengali font setup.
- `figures/`: method and dataset diagrams, the change-in-CER plot, the per-page scatters, the decoding-pitfall figure and the noise-sweep figure.
- `fonts/`: Noto Serif Bengali for the example table.
- `scripts/`: the generators above, `check_refs.py`, `make_overleaf_zip.py` and `render_examples_pdf.py`.
- `WRITING-GUIDE.md`: how the paper is written and revised.
- Local only (gitignored): `Draft_Paper_02.pdf` (the draft the reviewer marked up), `CHECK-BEFORE-SUBMITTING.md`, `overleaf-upload.zip`.
