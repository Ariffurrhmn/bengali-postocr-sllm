# Environment setup

## Python deps

```
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

## Tesseract + Bengali langpack

`pytesseract` wraps the Tesseract binary — it does not install it. Tesseract itself must be
installed separately (e.g. from https://github.com/UB-Mannheim/tesseract/wiki on Windows,
or `apt install tesseract-ocr` on Colab/Linux).

The Bengali language data (`ben.traineddata`) usually is **not** bundled with the base
Tesseract install and must be fetched separately. Rather than writing into the system
Tesseract install location (which needs admin rights on Windows), this project keeps its
own local tessdata directory so the setup is self-contained and reproducible:

```
mkdir .tessdata
curl -L -o .tessdata/ben.traineddata https://github.com/tesseract-ocr/tessdata_best/raw/main/ben.traineddata
curl -L -o .tessdata/eng.traineddata https://github.com/tesseract-ocr/tessdata_best/raw/main/eng.traineddata
curl -L -o .tessdata/osd.traineddata https://github.com/tesseract-ocr/tessdata_best/raw/main/osd.traineddata
```

(`tessdata_best` = highest-accuracy models, appropriate for a research evaluation over
raw inference speed.)

Set `TESSDATA_PREFIX` to this directory before running any Tesseract calls — see
`ocr/test_env.py` for the pattern:

```python
os.environ["TESSDATA_PREFIX"] = str(REPO_ROOT / ".tessdata")
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # or wherever installed
```

`.tessdata/` is gitignored (binary langpacks, not project code) — each environment (local
machine, Colab) regenerates it with the commands above.

## Windows console encoding

Printing Bengali text to a Windows terminal can crash with `UnicodeEncodeError` because the
default console codepage can't encode it. Scripts that print Bengali text should force UTF-8
stdout, e.g.:

```python
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
```

## Verifying the environment

```
python ocr/test_env.py
```

Confirms pytesseract (+ Bengali langpack), jiwer (CER scoring), transformers (model loading),
and easyocr (Bengali detection/recognition model, downloaded and cached on first run to
`~/.EasyOCR/model/`) all work end-to-end against a real dev-set page.

## Gated HuggingFace models — not yet done

Llama 3.2 1B and Gemma 2B require accepting a license on huggingface.co and an auth token
(`huggingface-cli login` or `HF_TOKEN` env var) before they can be loaded. Not set up yet —
needed before the correction stage can run against those two models.
