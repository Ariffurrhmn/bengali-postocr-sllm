"""Environment smoke test: confirms pytesseract, easyocr, transformers, and
jiwer are installed and functional against real dev-set data. Not part of the
pipeline itself — a one-off verification script."""
import io
import os
import sys
from pathlib import Path

# Windows consoles default to a codepage that can't encode Bengali/box-drawing
# characters; force UTF-8 stdout so printing OCR output doesn't crash.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = Path(r"D:\Competition_dataset_ImagesPAGEXML")
TEST_PAGE = "14048_D_11_0003"

os.environ["TESSDATA_PREFIX"] = str(REPO_ROOT / ".tessdata")
os.environ.setdefault(
    "TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


def test_pytesseract():
    import pytesseract
    from PIL import Image

    pytesseract.pytesseract.tesseract_cmd = os.environ["TESSERACT_CMD"]
    img = Image.open(DATASET_DIR / f"{TEST_PAGE}.tif")
    text = pytesseract.image_to_string(img, lang="ben")
    assert text.strip(), "pytesseract returned empty output"
    print("[pytesseract] OK — sample output:")
    print(text.strip()[:200])


def test_jiwer():
    import jiwer

    ref = "উত্তর মীমাংসা ।"
    hyp = "উত্তর মীমাংসা |"
    cer = jiwer.cer(ref, hyp)
    print(f"[jiwer] OK — CER({ref!r}, {hyp!r}) = {cer:.4f}")


def test_easyocr():
    import easyocr

    print("[easyocr] loading Bengali reader (downloads model on first run)...")
    reader = easyocr.Reader(["bn"], gpu=False)
    result = reader.readtext(str(DATASET_DIR / f"{TEST_PAGE}.tif"), detail=0)
    assert result, "easyocr returned no text regions"
    print(f"[easyocr] OK — {len(result)} text regions detected, sample:")
    print(result[:5])


def test_transformers():
    from transformers import AutoTokenizer

    print("[transformers] loading a small public tokenizer (gpt2, no gating)...")
    tok = AutoTokenizer.from_pretrained("gpt2")
    ids = tok.encode("hello world")
    assert ids, "tokenizer produced no tokens"
    print(f"[transformers] OK — tokenizer loaded, encoded {len(ids)} tokens")


if __name__ == "__main__":
    steps = [
        ("pytesseract", test_pytesseract),
        ("jiwer", test_jiwer),
        ("transformers", test_transformers),
        ("easyocr", test_easyocr),  # slowest / largest download — last
    ]
    failures = []
    for name, fn in steps:
        try:
            fn()
        except Exception as e:
            print(f"[{name}] FAILED: {e}")
            failures.append(name)
        print()

    if failures:
        print(f"FAILED: {failures}")
        sys.exit(1)
    print("All environment checks passed.")
