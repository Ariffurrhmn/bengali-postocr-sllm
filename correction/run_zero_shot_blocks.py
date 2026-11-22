"""Zero-shot correction of the reviewed BL test units, block by block (D81). Writes outputs only.

Each reviewed block (tools/build_bl_reviewed_units.py, OCR by ocr/run_reviewed_units_ocr.py) is
corrected on its own with the settings of the main 40-page runs: plain greedy decoding, the "en"
prompt, models.py's dtype and max-new-tokens rules. A block longer than about 250 tokens of the
model's tokenizer is split on line breaks (chunking.chunk_text) and the chunk outputs are joined.
Inputs under 10 characters are passed through unchanged by models.correct_text (main-run rule).
"symspell" is the dictionary baseline (CPU, no model).

Engines: tesseract, easyocr, and ground_truth (the block reference as input: clean-text check).
One output line per (page, engine, method) with the block outputs in the reviewed order. The run
resumes from an existing output file, so a time-limited session can be restarted.

Nothing is scored here, on purpose: eval/score_zero_shot_reviewed.py scores all methods once,
after every output is in.

Usage:
    python correction/run_zero_shot_blocks.py --ocr-region <reviewed_region.jsonl> \\
        --split-dir data/split_v2 --split test_bl --models gemma-2b,qwen3-1.7b \\
        --engines tesseract,easyocr,ground_truth --out results/rev08/zs_gemma-2b.jsonl
Smoke test (validation pages, CPU): --split val --ocr-region <val region file> --limit 2 --models qwen3-0.6b
"""
import argparse

import json
import os
import sys
import time
import traceback
from pathlib import Path

# reconfigure, not a new wrapper: dictionary_baseline and run_sweep wrap sys.stdout again on
# import, and a dropped wrapper would close the shared buffer.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

DECODING = "plain"
PROMPT = "en"


def load_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").split("\n") if l.strip()]


def read_ids(path):
    return [l.strip() for l in Path(path).read_text(encoding="utf-8").split("\n") if l.strip()]


def done_keys(path: Path) -> set:
    if not path.exists():
        return set()
    keys = set()
    for line in path.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue  # a cut last line from a stopped session
        if r.get("error") is None:
            keys.add((r["page_id"], r["engine"], r["model_key"]))
    return keys


def make_corrector(model_key: str):
    """Returns (correct_block, info). correct_block(text) -> (output, truncated, skipped, chunks)."""
    if model_key == "symspell":
        from dictionary_baseline import LEXICON, MAX_EDIT_DISTANCE, correct, load_symspell

        sym = load_symspell(LEXICON, MAX_EDIT_DISTANCE)

        def run(text):
            out, _ = correct(text, sym, MAX_EDIT_DISTANCE)
            return out, False, False, 1

        return run, {"device": "cpu", "dtype": "none", "lexicon": LEXICON.name}

    from chunking import chunk_text
    from models import correct_text, load_model

    tok, model = load_model(model_key, hf_token=os.environ.get("HF_TOKEN"))

    def run(text):
        # chunk_text drops blank lines; a block with no text keeps its input (as correct_text would)
        chunks = chunk_text(text, tok) if text.strip() else [text]
        outs, trunc, skipped = [], False, True
        for c in chunks:
            r = correct_text(model_key, tok, model, c, decoding=DECODING, prompt=PROMPT)
            outs.append(r.raw_output)
            trunc = trunc or r.truncated
            skipped = skipped and r.skipped
        return "\n".join(outs), trunc, skipped, len(chunks)

    return run, {"device": str(model.device), "dtype": str(model.dtype)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocr-region", type=Path, required=True)
    ap.add_argument("--split-dir", type=Path, required=True)
    ap.add_argument("--split", default="test_bl")
    ap.add_argument("--models", required=True, help="comma-separated model keys (models.MODEL_IDS) or symspell")
    ap.add_argument("--engines", default="tesseract,easyocr,ground_truth")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0, help="first N pages only (smoke test)")
    ap.add_argument("--stop-after-minutes", type=float, default=None,
                    help="start no new page after this many minutes; rerun to resume")
    args = ap.parse_args()

    ids = set(read_ids(args.split_dir / f"{args.split}.txt"))
    pages = sorted((r for r in load_jsonl(args.ocr_region) if r["page_id"] in ids), key=lambda r: r["page_id"])
    if args.limit:
        pages = pages[:args.limit]
    assert pages, "no pages of the split in the OCR file"
    engines = args.engines.split(",")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = done_keys(args.out)
    print(f"{len(pages)} pages of {args.split}, {sum(len(p['regions']) for p in pages)} blocks; "
          f"{len(done)} (page, engine, method) already done", flush=True)

    t_start = time.time()
    with args.out.open("a", encoding="utf-8") as f:
        for model_key in args.models.split(","):
            todo = [(p, e) for e in engines for p in pages if (p["page_id"], e, model_key) not in done]
            if not todo:
                continue
            print(f"=== {model_key}: {len(todo)} (page, engine) to run", flush=True)
            try:
                run, info = make_corrector(model_key)
            except Exception:
                traceback.print_exc()
                print(f"FAILED to load {model_key}", flush=True)
                continue
            for page, eng in todo:
                if args.stop_after_minutes is not None and time.time() - t_start > args.stop_after_minutes * 60:
                    print(f"time budget of {args.stop_after_minutes:.0f} min reached; rerun to resume", flush=True)
                    return
                t0 = time.time()
                outs, trunc, skip, nch, error = [], [], [], [], None
                try:
                    for reg in page["regions"]:
                        o, t, s, n = run(reg[eng])
                        outs.append(o)
                        trunc.append(t)
                        skip.append(s)
                        nch.append(n)
                except Exception as e:
                    traceback.print_exc()
                    error = f"{type(e).__name__}: {e}"
                rec = {"page_id": page["page_id"], "engine": eng, "model_key": model_key,
                       "unit": "reviewed_block", "decoding": DECODING if model_key != "symspell" else "none",
                       "prompt": PROMPT if model_key != "symspell" else "none", **info,
                       "block_ids": [reg["id"] for reg in page["regions"]],
                       "outputs": outs, "truncated": trunc, "skipped": skip, "chunks": nch,
                       "error": error, "elapsed_sec": round(time.time() - t0, 1)}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                print(f"  {page['page_id']} {eng}/{model_key}: {len(outs)} blocks, truncated {sum(trunc)}, "
                      f"skipped {sum(skip)}{' ERROR' if error else ''} ({rec['elapsed_sec']} s)", flush=True)
            if model_key != "symspell":
                import gc

                import torch

                del run
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    print("done:", args.out, flush=True)


if __name__ == "__main__":
    main()
