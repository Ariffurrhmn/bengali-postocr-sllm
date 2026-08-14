# bengali-postocr-sllm

OCR-engine-agnostic post-OCR correction for Bengali text using small open-weight language models (zero-shot, CPU-only).

See the project methodology in the Drive folder: `Draft Paper/Methodology_PostOCR.docx`.

## Layout

- `data/` — dataset lives outside the repo (`D:\Competition_dataset_ImagesPAGEXML`, British Library historical Bengali corpus, 81 image+PAGE-XML page pairs). This folder is a placeholder for local staging only; large files are not committed.
- `ocr/` — Tesseract / EasyOCR wrappers.
- `correction/` — per-model zero-shot correction harness (Phi-3 Mini, Llama 3.2 1B, Gemma 2B, TituLLMs 1B, BanglaT5).
- `eval/` — CER/WER scoring, NFC normalization, paired bootstrap significance testing, figures, qualitative failure-mode diagnostics.
- `results/` — logged per-page outcome data (CSV/JSON), not committed except for structure.
- `notebooks/` — Colab notebook(s).

## Dataset

Ground-truth transcriptions for OCR of historical Bengali printed texts, released for the **ICDAR 2019
Competition on Recognition of Early Indian Printed Documents (REID2019)** — organised by the British
Library's *Two Centuries of Indian Print* project with PRImA Research Lab and Jadavpur University.
Source books date from 1713–1914. Images are out of copyright; transcriptions are public domain.

Not included in this repo. Supply locally at a fixed path, or mount/copy into the Colab runtime at the start of each session.
