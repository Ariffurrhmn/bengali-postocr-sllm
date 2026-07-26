"""Correction sweep: runs one or more models, over one or more approaches
(whole-page vs chunked prompting), over the dev (or eval) OCR results.

Designed to survive a Colab disconnect: every individual result is appended
to the output JSONL file immediately after being computed (not batched), and
on start the script reads whatever is already in the output file and skips
any (page, engine, model, approach) combination already done. Re-running the
same command after a disconnect resumes rather than restarts.

Usage:
    python run_sweep.py --split dev --models titullm-1b,phi3-mini \
        --approaches whole,chunked --engines tesseract,easyocr
"""
import argparse
import io
import json
import sys
import time
import traceback
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from chunking import chunk_text
from models import MODEL_IDS, correct_text, load_model  # noqa: E402 (also sets CPU threads)

REPO_ROOT = Path(__file__).resolve().parent.parent

ALL_MODELS = list(MODEL_IDS.keys())
ALL_ENGINES = ["tesseract", "easyocr"]
ALL_APPROACHES = ["whole", "chunked"]


def load_done_keys(out_path: Path) -> set:
    """Reads whatever's already in the output file so completed
    (page, engine, model, approach) combos are skipped on resume."""
    if not out_path.exists():
        return set()
    done = set()
    for line in out_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue  # tolerate a truncated last line from a mid-write disconnect
        done.add((r["page_id"], r["engine"], r["model_key"], r["approach"]))
    return done


def correct_whole(model_key, tokenizer, model, ocr_text):
    result = correct_text(model_key, tokenizer, model, ocr_text)
    return result.raw_output, result.truncated, result.skipped


def correct_chunked(model_key, tokenizer, model, ocr_text):
    chunks = chunk_text(ocr_text, tokenizer)
    outputs = []
    any_truncated = False
    all_skipped = True
    for chunk in chunks:
        result = correct_text(model_key, tokenizer, model, chunk)
        outputs.append(result.raw_output)
        any_truncated = any_truncated or result.truncated
        all_skipped = all_skipped and result.skipped
    return "\n".join(outputs), any_truncated, all_skipped


def main():
    import torch

    print(f"torch.get_num_threads() = {torch.get_num_threads()}", file=sys.stderr)

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "eval"], default="dev")
    parser.add_argument(
        "--models", default=",".join(ALL_MODELS), help="Comma-separated model keys"
    )
    parser.add_argument(
        "--engines", default=",".join(ALL_ENGINES), help="Comma-separated OCR engines"
    )
    parser.add_argument(
        "--approaches",
        default=",".join(ALL_APPROACHES),
        help="Comma-separated: whole,chunked",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    models = args.models.split(",")
    engines = args.engines.split(",")
    approaches = args.approaches.split(",")

    ocr_path = REPO_ROOT / "results" / f"ocr_{args.split}.jsonl"
    pages = [json.loads(l) for l in ocr_path.read_text(encoding="utf-8").splitlines()]

    out_path = args.out or (REPO_ROOT / "results" / f"correction_{args.split}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = load_done_keys(out_path)
    print(f"{len(done)} (page, engine, model, approach) combos already done — will skip.")

    total = len(pages) * len(engines) * len(models) * len(approaches)
    print(f"Total combos to consider: {total}")

    with out_path.open("a", encoding="utf-8") as out_f:
        for model_key in models:
            print(f"\n=== Loading model: {model_key} ===")
            try:
                tokenizer, model = load_model(model_key)
            except Exception as e:
                print(f"FAILED to load {model_key}: {e}", file=sys.stderr)
                traceback.print_exc()
                continue

            for engine in engines:
                for approach in approaches:
                    for page in pages:
                        key = (page["page_id"], engine, model_key, approach)
                        if key in done:
                            continue

                        print(
                            f"  -> starting {page['page_id']} {engine}/{model_key}/{approach}...",
                            file=sys.stderr,
                        )
                        t0 = time.time()
                        try:
                            fn = correct_whole if approach == "whole" else correct_chunked
                            raw_output, truncated, skipped = fn(
                                model_key, tokenizer, model, page[engine]
                            )
                            error = None
                        except Exception as e:
                            raw_output, truncated, skipped = "", False, False
                            error = f"{type(e).__name__}: {e}"
                            traceback.print_exc()

                        elapsed = time.time() - t0
                        record = {
                            "page_id": page["page_id"],
                            "engine": engine,
                            "model_key": model_key,
                            "approach": approach,
                            "raw_output": raw_output,
                            "truncated": truncated,
                            "skipped": skipped,
                            "error": error,
                            "elapsed_sec": round(elapsed, 1),
                        }
                        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        out_f.flush()  # survive a disconnect right after this line
                        done.add(key)

                        status = "ERROR" if error else ("SKIPPED" if skipped else "OK")
                        print(
                            f"[{len(done)}/{total}] {page['page_id']} "
                            f"{engine}/{model_key}/{approach} {status} ({elapsed:.1f}s)",
                            file=sys.stderr,
                        )

            del model, tokenizer  # free memory before loading the next model

    print(f"\nDone. Results in {out_path}")


if __name__ == "__main__":
    main()
