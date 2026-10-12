# Environment record

What the reported runs used, as far as it was recorded (review item W9). Where a version was not recorded, this file says so rather than guessing.

## Data
- **Dataset:** "Ground Truth transcriptions for training OCR of historical Bengali printed texts – Recognition of Early Indian Printed Documents competition – updated with improved XML coordinates", version 2 (2020). Creators: Tom Derrick and the British Library. DOI [10.23636/1168](https://doi.org/10.23636/1168), record at https://bl.iro.bl.uk/concern/datasets/bb125e50-4a09-46e8-904b-6e46cf308d91. It has 81 TIFF pages at 400 dpi with PAGE-XML. The licence is the Public Domain Mark 1.0.
- **Split:** `data/split_dev.txt` and `data/split_eval.txt`. They were frozen before any model ran. `data/excluded_no_ground_truth.txt` lists the pages without transcriptions.
- **Page types:** `data/page_types.tsv`. These were assigned from the transcriptions after the runs and are used only for the descriptive page-type table.

## OCR (October 2026, Google Colab CPU runtime)
- Tesseract 5.3.4 with the LSTM engine and the Bengali `tessdata_best` model. It ran in the default page-segmentation mode (3) with no preprocessing.
- EasyOCR 1.7.2 with Bengali enabled, on default settings.
- Both used the page image as released.
- The August 2026 Tesseract build (15-page CPU runs) was not recorded. Its output differs from 5.3.4.

## Correction runs
- **GPU (main results, 10–12 October 2026):**
  - Hardware and image: Kaggle, NVIDIA T4 (15 GB), the free Kaggle image of those dates.
  - Recorded versions: Transformers 5.19.0 and Python 3.11.
  - Not recorded: the PyTorch and CUDA versions of the image.
  - Precision: float32, or float16 for Phi-4-mini and TituLLMs 3B Instruct.
  - Driver: notebooks `notebooks/kaggle_gpu_run.ipynb` and `notebooks/kaggle_variants_run.ipynb`.
- **CPU (decoding ablation and CPU timing, 15 pages, August 2026):** free Google Colab CPU runtime in bfloat16. The package versions were not recorded.
- **Checkpoints and revisions:** see `REVISIONS` in `eval/make_paper_tables.py` and Table III of the paper.
- **Chat template:** each model's own template, as shipped with that revision, applied with `apply_chat_template(add_generation_prompt=True)`. Qwen3 additionally gets `enable_thinking=False`. BanglaT5 gets the OCR text alone, with no template. The prompts are in `correction/models.py` (`PROMPTS`).
- **Generation:**
  - Greedy decoding (`do_sample=False`).
  - `max_new_tokens = min(512, max(64, 1.5 × input tokens))`.
  - Main runs (`plain`): no repetition penalty and no n-gram ban.
  - Ablation (`guarded`): `repetition_penalty=1.3` and `no_repeat_ngram_size=4`.
  - The settings are defined in `DECODING_PRESETS` in `correction/models.py`.

## Analysis (local, Windows, `.venv`)
Python 3.11.0, numpy 2.4.6, scipy 1.17.1, jiwer 4.0.0, RapidFuzz 3.14.5, regex 2026.7.19, matplotlib 3.11.1, symspellpy 6.10.0, sentencepiece 0.2.2, tokenizers 0.22.2, transformers 5.14.1 (tokenizer counts only), torch 2.13.0. Bootstrap: 10,000 resamples, seed 403.

The paper compiles with XeLaTeX. Tectonic 0.17.0 (a XeTeX engine) was used to check the layout locally.

## Summary statistics
`summaries/` holds the numeric outputs of the analysis scripts that every table and figure is built from: bootstrap, extended analysis, revision analysis, sensitivity checks, variants, gating, model profile and decoding ablation. They contain no page text. The per-page model outputs stay outside version control (they contain the transcribed text and are large); see `context/decisions.md` D26 and D34.
