"""Step 3 dry run: run one correction model on one dev page's OCR output,
to sanity-check the prompt format and catch obvious failure modes
(refusals, echoing, wrong script, truncation) before running the full sweep."""
import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from models import correct_text, load_model

REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
    model_key = sys.argv[1] if len(sys.argv) > 1 else "titullm-1b"
    # Page defaults to the best (lowest-CER) Tesseract dev result, not just
    # the first record — a near-empty/garbage OCR input (e.g. 14048_D_11_0003,
    # CER=0.640) gives the model almost nothing to work with and produces
    # uninformative confabulation rather than testing actual correction
    # behavior. Pass a page_id explicitly as the 2nd CLI arg to override.
    page_id_arg = sys.argv[2] if len(sys.argv) > 2 else None

    ocr_path = REPO_ROOT / "results" / "ocr_dev.jsonl"
    records = [json.loads(l) for l in ocr_path.read_text(encoding="utf-8").splitlines()]

    default_page_id = "279_36_C_32_0016"  # best (lowest-CER) Tesseract dev result
    if page_id_arg:
        page = next(r for r in records if r["page_id"] == page_id_arg)
    else:
        page = next((r for r in records if r["page_id"] == default_page_id), records[0])

    print(f"Page: {page['page_id']}")
    print(f"Model: {model_key}")
    print()
    print("Ground truth:")
    print(page["ground_truth"])
    print()
    print("Tesseract OCR (input to correction):")
    print(page["tesseract"])
    print()

    print(f"Loading {model_key}...")
    tokenizer, model = load_model(model_key)

    print("Running correction...")
    result = correct_text(model_key, tokenizer, model, page["tesseract"])

    print()
    print("Corrected output:")
    print(result.raw_output)
    print()
    print(f"Truncated (hit max_new_tokens): {result.truncated}")


if __name__ == "__main__":
    main()
