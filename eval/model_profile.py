"""Per-model profile for the revision (review items E5 and T2): tokenizer
fertility on the evaluation pages, resulting chunk sizes, and per-page run
time on CPU and GPU.

  tokens/char, tokens/word - each model's tokenizer on the normalised ground
                             truth of the 40 evaluation pages
  chars/chunk              - mean characters per chunk when the raw Tesseract
                             text is cut into ~250-token chunks (chunking.py)
  CPU / GPU min per page   - mean wall time per page and engine from the run
                             files (CPU: 15-page Colab runs; GPU: 40-page T4 run)

Needs a Hugging Face login for the gated tokenizers (Gemma, Llama).

Usage:
    python model_profile.py --ocr-path ../results/ocr_eval40.jsonl \\
        --cpu-runs ../results/ablation_plain_decoding.jsonl \\
        --gpu-runs ../results/gpu_eval40.jsonl --out ../results/model_profile.json
"""
import argparse
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "correction"))

from chunking import chunk_text  # noqa: E402
from metrics import normalize  # noqa: E402


def load_jsonl(path: Path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


class _SentencePieceTokenizer:
    """Minimal stand-in exposing encode(text, add_special_tokens=False)."""

    def __init__(self, path):
        import sentencepiece as spm
        self.sp = spm.SentencePieceProcessor(model_file=path)

    def encode(self, text, add_special_tokens=False):
        return self.sp.encode(text)


def load_tokenizer(model_key, model_id):
    # BanglaT5's tokenizer is a plain SentencePiece model. Loading it through
    # transformers needs protobuf for the fast conversion, so read it directly;
    # the piece counts are the same tokens the model receives.
    if model_key == "banglat5":
        from huggingface_hub import hf_hub_download
        return _SentencePieceTokenizer(hf_hub_download(model_id, "spiece.model"))
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_id)


def mean_minutes(rows, model, engines):
    t = [r["elapsed_sec"] for r in rows if r["model_key"] == model and r["engine"] in engines]
    return sum(t) / len(t) / 60 if t else None


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "correction"))
    from models import MODEL_IDS

    parser = argparse.ArgumentParser()
    parser.add_argument("--ocr-path", type=Path, required=True)
    parser.add_argument("--cpu-runs", type=Path, required=True)
    parser.add_argument("--gpu-runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    pages = load_jsonl(args.ocr_path)
    gt = [normalize(p["ground_truth"]) for p in pages]
    n_chars = sum(len(t) for t in gt)
    n_words = sum(len(t.split()) for t in gt)
    cpu, gpu = load_jsonl(args.cpu_runs), load_jsonl(args.gpu_runs)
    models = list(dict.fromkeys(r["model_key"] for r in gpu))

    out = []
    print(f"{'model':<22}{'tok/char':>9}{'tok/word':>9}{'chars/chunk':>12}"
          f"{'CPU min/pg':>11}{'GPU min/pg':>11}")
    for m in models:
        tok = load_tokenizer(m, MODEL_IDS[m])
        n_tok = sum(len(tok.encode(t, add_special_tokens=False)) for t in gt)
        chunk_chars = [len(c) for p in pages for c in chunk_text(p["tesseract"], tok)]
        row = {"model": m, "checkpoint": MODEL_IDS[m],
               "tokens_per_char": n_tok / n_chars, "tokens_per_word": n_tok / n_words,
               "mean_chars_per_chunk": sum(chunk_chars) / len(chunk_chars),
               "cpu_min_per_page": mean_minutes(cpu, m, {"tesseract", "easyocr"}),
               "gpu_min_per_page": mean_minutes(gpu, m, {"tesseract", "easyocr"})}
        out.append(row)
        f = lambda v: f"{v:.1f}" if v is not None else "---"
        print(f"{m:<22}{row['tokens_per_char']:>9.2f}{row['tokens_per_word']:>9.1f}"
              f"{row['mean_chars_per_chunk']:>12.0f}{f(row['cpu_min_per_page']):>11}"
              f"{f(row['gpu_min_per_page']):>11}")
    args.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
