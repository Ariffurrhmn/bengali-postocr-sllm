# bengali-postocr-sllm

Does zero-shot post-correction with small open language models help OCR of historical Bengali print? This repository holds the code, frozen page lists, prompts and paper source for an evaluation of eight models (0.25–3.8B parameters) and a dictionary baseline on the output of Tesseract and EasyOCR. Short answer: no model significantly lowers the character error rate; see [context/results.md](context/results.md).

Current status and which file is latest: [STATUS.md](STATUS.md). Decision log: [context/](context/).

## Layout

- `data/`: frozen split lists and the scripts that made them; the dataset itself is not committed.
- `ocr/`: Tesseract / EasyOCR runs, including word-level confidences.
- `correction/`: zero-shot correction harness (models, prompts, chunking, decoding), the resumable sweep, and the SymSpell baseline.
- `eval/`: scoring (NFC-normalised CER/WER/cMER), paired bootstrap and Holm-corrected Wilcoxon tests, the revision analyses, and the generators for every paper table and figure.
- `notebooks/`: the original Colab CPU notebook and the Kaggle GPU notebooks used for the 40-page runs.
- `results/`: run outputs, gitignored.
- `paper/`: paper source (`main.tex`), generated tables and figures, bibliography, and the bundled Bengali font; `archive/` holds a superseded version.
- `context/`: decision log, results, timeline, engineering notes, open items.

## Dataset

Ground-truth transcriptions for OCR of historical Bengali printed texts, released for the **ICDAR 2019
Competition on Recognition of Early Indian Printed Documents (REID2019)**, organised by the British
Library's *Two Centuries of Indian Print* project with PRImA Research Lab and Jadavpur University.
Source books date from 1713–1914. Images are out of copyright; transcriptions are public domain.

Not included in this repo. Supply it locally, or unzip it into the Colab/Kaggle runtime at the start of each session (see the notebooks and [SETUP.md](SETUP.md)).
