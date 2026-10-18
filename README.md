# bengali-postocr-sllm

Code, data splits and paper for **"Small Language Models Do Not Correct Historical Bengali OCR, and a Decoding Default Makes Them Look Worse"** (working title; Arifur Rahman, CSE403, Bangladesh University of Business and Technology, 2026). The paper is being revised after a third review.

**Paper:** [paper/main.pdf](paper/main.pdf) (source: [paper/main.tex](paper/main.tex)). **Cite:** see [CITATION.cff](CITATION.cff).

## In short
40 pages of Bengali books printed 1713–1914, read by Tesseract 5.3.4 and EasyOCR 1.7.2, corrected zero-shot by 14 open language models of 0.25–8 billion parameters (and Qwen3 32B in 4-bit on 10 pages), by SymSpell dictionary baselines and by a word-level fusion of the two engines.

- **No language model lowers the character error rate (CER)** on either engine. General-purpose models mostly copy their input; the Bengali-pretrained checkpoints regenerate it (CER above 0.8). Every method damages error-free text.
- **The failure is not explained by noisy input:** correction also hurts on region-level OCR with lower CER, and on reference text corrupted to any CER from 0.02 to 0.30.
- **A common decoding default misleads evaluations:** a repetition penalty or n-gram ban also acts on the prompt in Hugging Face `generate` and makes copying models look harmful.
- **Larger Qwen3 models** (4B, 8B, 32B) fix more wrong words and lower the word error rate, but break correct ones, so CER does not fall.
- **Without a language model:** a better (Wikipedia) lexicon cuts the dictionary's damage by four-fifths but it stays harmful; fusing the two engines by word confidence fixes more words than it breaks, but lowers CER by under 1%.
- Token-level Grad-CAM maps show copying from inside the models: copied tokens rely on their own source token.
- A fine-tuning pilot (ByT5-small on synthetic noise) is being rerun after a test-format error (decision D50).

Current numbers: [context/results.md](context/results.md).

## Repository layout
| Folder | Contents |
|---|---|
| `paper/` | `main.tex`, `references.bib`, generated `tables/` and `figures/`, the Bengali font, the compiled `main.pdf`, and `scripts/` that generate every table and figure. See [paper/README.md](paper/README.md). |
| `data/` | Frozen page lists (`split_dev.txt`, `split_eval.txt`, `excluded_no_ground_truth.txt`), page-type labels, the development-page noise model, the 10-page list of the large-model run, and the scripts that made them. The dataset itself is not committed. |
| `ocr/` | Whole-page OCR (`run_ocr.py`), word and box confidences (`run_ocr_conf.py`), region-level OCR from the PAGE-XML regions (`run_region_ocr.py`). |
| `correction/` | The correction harness: models, prompts, decoding presets (`models.py`), chunking, the resumable sweep (`run_sweep.py`), the SymSpell baseline, the fine-tuning pilot (`finetune_byt5.py`), the float16 check. `sessions/` holds the drivers of the Kaggle GPU sessions. |
| `eval/` | Scoring (`metrics.py`: NFC-normalised CER, WER, cMER), paired bootstrap and Holm-corrected Wilcoxon tests, every analysis, the synthetic-noise model (`make_noise.py`) and the Grad-CAM attribution (`xai_gradcam.py`). |
| `notebooks/` | `kaggle/`: the GPU and CPU sessions that produced the results (main run, prompt variants, region OCR, third-review runs a-d). `colab/`: the first-version CPU pipeline on 15 pages (decoding comparison and CPU timings). |
| `summaries/` | Numeric outputs of the analyses that the tables and figures are built from. No page text. |
| `context/` | The project's memory: decisions with reasons, results, timeline, engineering notes, the third review with status, and the handoff for the next session. |
| `results/` | Run outputs (local, gitignored). |

[ENVIRONMENT.md](ENVIRONMENT.md) records versions, checkpoints and settings.

## Reproducing
1. **Dataset:** the REID2019 Bengali ground truth from the British Library, version 2 ([doi:10.23636/1168](https://doi.org/10.23636/1168); public domain). Unzip it locally or in the Colab/Kaggle runtime.
2. **Python:** `python -m venv .venv`, activate it, `pip install -r requirements.txt`.
3. **Tesseract:** install the binary and put `ben.traineddata` from `tessdata_best` in `.tessdata/`, with `TESSDATA_PREFIX` pointing there. Note that Tesseract 5.3.4 no longer reproduces the main run's text (ENVIRONMENT.md), so the main OCR files are kept as released data.
4. **Hugging Face:** Gemma, Gemma 3 and Llama are gated; accept their licences and provide `HF_TOKEN` as a Kaggle/Colab secret.
5. **Run:** OCR with `ocr/`; correction with the notebooks in `notebooks/kaggle/` (they clone this repo and call `correction/run_sweep.py` or a driver in `correction/sessions/`); the dictionary baseline with `correction/dictionary_baseline.py`; analyses, tables and figures with the commands in [paper/README.md](paper/README.md), from the repository root.

On Windows, scripts that print Bengali re-wrap stdout as UTF-8 so the console does not crash.

## Licence and data
The code is released under the [MIT licence](LICENSE). The per-page OCR text and model outputs are not in this repository; they will be released separately under CC BY 4.0 together with the final paper. The page images are out of copyright and the transcriptions carry the Public Domain Mark. The bundled font, Noto Serif Bengali, is under the SIL Open Font License. Model checkpoints are under their own licences and are not redistributed here.
