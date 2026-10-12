# bengali-postocr-sllm

Code, data splits and paper for **"Does Zero-Shot Small-Model Post-Correction Help Bengali OCR? A Negative Result Across Eight Models and Two OCR Engines"** (Arifur Rahman, CSE403, Bangladesh University of Business and Technology, 2026).

**Paper:** [paper/main.pdf](paper/main.pdf) (source: [paper/main.tex](paper/main.tex))

## In short
We OCR 40 pages of Bengali books printed 1713–1914 with Tesseract 5.3.4 and EasyOCR 1.7.2. Each output is then corrected zero-shot by eight open models of 0.25–3.8B parameters (Gemma 2B, Qwen3 1.7B, Phi-4-mini, Llama 3.2 1B, BanglaT5, and three TituLLMs checkpoints), plus a SymSpell dictionary baseline.

- **No ungated method significantly lowers the character error rate** on either engine. The general-purpose models mostly copy their input. The Bengali-specific models regenerate it (CER above 0.8).
- **Every method damages error-free text** when given the reference itself.
- **A common anti-repetition setting** (repetition penalty, n-gram ban) also acts on the prompt in Hugging Face `generate`. It made copying models look harmful in our first runs.
- **The only significant gains** are small, from Qwen3 1.7B on Tesseract output: WER 0.827 → 0.799, and CER −0.004 when edits are restricted to words the OCR engine was unsure of.

The full numbers are in [context/results.md](context/results.md) and the paper.

## Repository layout
| Folder | Contents |
|---|---|
| `paper/` | The paper: `main.tex`, generated `tables/` and `figures/`, `references.bib`, the bundled Bengali font, and the compiled `main.pdf`. See [paper/README.md](paper/README.md). |
| `data/` | Frozen page lists (`split_dev.txt`, `split_eval.txt`), page-type labels, and the scripts that made them. The dataset itself is not committed. |
| `ocr/` | Tesseract and EasyOCR runs (`run_ocr.py`) and word/box confidences (`run_ocr_conf.py`). |
| `correction/` | Correction harness: model loading, prompts, chunking and decoding presets (`models.py`, `chunking.py`), the resumable sweep (`run_sweep.py`), and the SymSpell baseline. |
| `eval/` | Scoring (NFC-normalised CER/WER/cMER), paired bootstrap, Holm-corrected Wilcoxon tests, all analyses, and the generators for every paper table and figure. |
| `notebooks/` | `kaggle_gpu_run.ipynb` and `kaggle_variants_run.ipynb` (40-page T4 runs), and `setup_and_dry_run.ipynb` (first-version Colab CPU pipeline, used for the decoding ablation and CPU timing). |
| `summaries/` | Numeric outputs of the analyses that the tables and figures are built from. They contain no page text. |
| `context/` | Decision log, results, timeline, open items, and the reviewer-checklist status. |
| `results/` | Run outputs (local, gitignored). |

[STATUS.md](STATUS.md) says what is current. [ENVIRONMENT.md](ENVIRONMENT.md) records versions, checkpoints and settings.

## Reproducing
1. **Dataset:** download the REID2019 Bengali ground truth from the British Library, version 2 ([doi:10.23636/1168](https://doi.org/10.23636/1168); public domain). Then unzip it locally or into the Colab/Kaggle runtime.
2. **Python:** `python -m venv .venv`, activate it, then `pip install -r requirements.txt`.
3. **Tesseract:** install the binary (`apt install tesseract-ocr`, or the UB-Mannheim build on Windows). Put `ben.traineddata` from `tessdata_best` in `.tessdata/` and set `TESSDATA_PREFIX` to that folder (see `ocr/run_ocr.py`).
4. **Hugging Face:** Gemma and Llama are gated. Accept their licences and provide `HF_TOKEN` as a Colab/Kaggle secret.
5. **Run:**
   - OCR: `ocr/run_ocr.py`
   - Correction: `notebooks/kaggle_gpu_run.ipynb`, which calls `correction/run_sweep.py`
   - Baseline: `correction/dictionary_baseline.py`
   - Analyses, tables and figures: the commands in [paper/README.md](paper/README.md)

On Windows, scripts that print Bengali re-wrap stdout as UTF-8 so the console does not crash.

## Licence and data
The page images are out of copyright and the transcriptions carry the Public Domain Mark. The bundled font, Noto Serif Bengali, is under the SIL Open Font License. Model checkpoints are under their own licences and are not redistributed here.
