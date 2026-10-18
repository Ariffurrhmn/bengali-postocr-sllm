"""Fine-tuning pilot (third review, item M2): a small byte-level model trained
to correct synthetic OCR noise, then tested on the 40 evaluation pages.

  data   Clean Bengali text: segments of 60-300 characters from Bengali
         Wikipedia (wikimedia/wikipedia 20231101.bn, streamed) plus the lines
         of the 10 development-page references (repeated, to add historical
         spelling). Segments sharing any 25-character string with an evaluation
         reference are dropped, so no evaluation text is trained on. Each
         segment is corrupted with the development-page noise model
         (eval/make_noise.py) at a random level: 10% of pairs are left clean,
         so the model also learns to copy, the rest get a CER of 0.02-0.40.
  train  google/byt5-small (about 300M parameters, bytes as tokens, so no
         Bengali tokenisation problem), AdamW, float32, for a fixed time
         budget. Every --eval-every steps the model corrects the real OCR of
         the development pages; the checkpoint with the lowest development
         CER is kept. Development pages are never scored in the paper.
  test   The kept checkpoint corrects the 40 evaluation pages (Tesseract,
         EasyOCR and the clean text) in chunks of at most 300 characters of
         whole lines (chunking.chunk_text_chars), the lines of a chunk joined
         by spaces as in training, greedy decoding. Output
         records follow run_sweep.py (model_key "byt5-small-ft"), so
         eval/matrix_analysis.py scores them like any other method.

Usage (Kaggle GPU, see sessions/run_review3d_kaggle.py):
    python correction/finetune_byt5.py --ocr-path <ocr_eval40.jsonl> --dev-path <ocr_dev.jsonl> \\
        --out-dir /kaggle/working/review3d --train-minutes 100
Local smoke test (tiny random model, no download of the real weights):
    python correction/finetune_byt5.py --ocr-path results/ocr_eval40.jsonl --dev-path results/ocr_dev.jsonl \\
        --out-dir <tmp> --tiny --no-wiki --train-steps 3 --test-pages 1
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "eval"))

from chunking import chunk_text_chars  # noqa: E402
from make_noise import Noiser, prep  # noqa: E402
from metrics import cer  # noqa: E402

MODEL_ID = "google/byt5-small"
MODEL_KEY = "byt5-small-ft"
SEED = 403
NGRAM = 25
# Noise factor per target CER, from the calibration of make_noise.py apply
# (factor 1.018 gives CER 0.297 on the evaluation references).
FACTOR_PER_CER = 3.4


def load(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").split("\n") if l.strip()]


def bengali_share(text):
    letters = [c for c in text if c.isalpha()]
    return sum(0x0980 <= ord(c) <= 0x09FF for c in letters) / len(letters) if letters else 0.0


def segments_from(text, lo=60, hi=300):
    """Split a paragraph at dandas into segments of lo-hi characters."""
    out, cur = [], ""
    for part in text.replace("\n", " ").split("।"):
        part = part.strip()
        if not part:
            continue
        piece = part + "।"
        if len(cur) + len(piece) + 1 > hi and len(cur) >= lo:
            out.append(cur)
            cur = ""
        cur = (cur + " " + piece).strip()
    if lo <= len(cur) <= hi:
        out.append(cur)
    return [s for s in out if lo <= len(s) <= hi]


def build_data(args, eval_refs, dev_pages, rng):
    banned = set()
    for ref in eval_refs:
        t = " ".join(prep(ref).split())
        banned.update(t[i:i + NGRAM] for i in range(max(1, len(t) - NGRAM + 1)))

    def clean(seg):
        t = " ".join(prep(seg).split())
        return not any(t[i:i + NGRAM] in banned for i in range(max(1, len(t) - NGRAM + 1)))

    segs, dropped = [], 0
    if not args.no_wiki:
        from datasets import load_dataset

        ds = load_dataset("wikimedia/wikipedia", "20231101.bn", split="train", streaming=True)
        for art in ds:
            for s in segments_from(art["text"]):
                if bengali_share(s) < 0.9:
                    continue
                if clean(s):
                    segs.append(s)
                else:
                    dropped += 1
            if len(segs) >= args.wiki_segments:
                break
    dev_lines = [l.strip() for p in dev_pages for l in prep(p["ground_truth"]).split("\n") if len(l.strip()) >= 20]
    dev_lines = [l for l in dev_lines if clean(l)]
    segs += dev_lines * args.dev_repeat
    rng.shuffle(segs)
    print(f"training segments: {len(segs)} ({len(dev_lines)} dev lines x{args.dev_repeat}); "
          f"dropped for overlap with evaluation text: {dropped}", flush=True)
    return segs


def corrupt_pairs(segs, noiser, rng):
    pairs = []
    for s in segs:
        level = 0.0 if rng.random() < 0.1 else rng.uniform(0.02, 0.40)
        noisy = noiser.corrupt(s, level * FACTOR_PER_CER, rng) if level else s
        pairs.append((noisy, s))
    return pairs


def correct(model, tok, text, device, max_bytes):
    import torch

    outs, truncated = [], False
    for chunk in chunk_text_chars(text):
        # Lines joined by spaces, as in training (segments_from removes newlines);
        # review3d fed "\n" the model had never seen and it skipped and repeated
        # lines (D50). CER ignores whitespace, so scoring is unaffected.
        chunk = " ".join(chunk.split())
        enc = tok(chunk, return_tensors="pt", truncation=True, max_length=max_bytes).to(device)
        n_in = enc["input_ids"].shape[1]
        cap = min(max_bytes, int(n_in * 1.3) + 16)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=cap, do_sample=False)[0]
        truncated |= gen.shape[0] - 1 >= cap
        outs.append(tok.decode(gen, skip_special_tokens=True))
    return "\n".join(outs), truncated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocr-path", type=Path, required=True)
    ap.add_argument("--dev-path", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--wiki-segments", type=int, default=40000)
    ap.add_argument("--dev-repeat", type=int, default=20)
    ap.add_argument("--no-wiki", action="store_true")
    ap.add_argument("--train-minutes", type=float, default=100)
    ap.add_argument("--train-steps", type=int, default=0, help="stop after this many steps (0: time only)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-bytes", type=int, default=1024)
    ap.add_argument("--eval-every", type=int, default=200, help="optimizer steps between development evaluations")
    ap.add_argument("--test-pages", type=int, default=0, help="limit the test pages (smoke test)")
    ap.add_argument("--tiny", action="store_true", help="tiny random model (local smoke test)")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5Config, T5ForConditionalGeneration

    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    pages = load(args.ocr_path)
    dev_pages = load(args.dev_path)
    noiser = Noiser(json.loads((REPO / "data" / "noise_model_dev.json").read_text(encoding="utf-8")))
    segs = build_data(args, [p["ground_truth"] for p in pages], dev_pages, rng)
    pairs = corrupt_pairs(segs, noiser, rng)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if args.tiny:
        model = T5ForConditionalGeneration(T5Config(vocab_size=384, d_model=32, d_ff=64, num_layers=1,
                                                    num_decoder_layers=1, num_heads=2, d_kv=16,
                                                    decoder_start_token_id=0, pad_token_id=0, eos_token_id=1))
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, dtype=torch.float32)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    def dev_cer():
        model.eval()
        scores = []
        for p in dev_pages:
            for e in ("tesseract", "easyocr"):
                hyp, _ = correct(model, tok, p[e], device, args.max_bytes)
                scores.append(cer(p["ground_truth"], hyp))
        model.train()
        return sum(scores) / len(scores)

    raw_dev = sum(cer(p["ground_truth"], p[e]) for p in dev_pages for e in ("tesseract", "easyocr")) / (2 * len(dev_pages))
    log = {"raw_dev_cer": raw_dev, "evals": [], "pairs": len(pairs)}
    print(f"raw OCR CER on the development pages: {raw_dev:.3f}", flush=True)
    best, best_path = float("inf"), out / "byt5_best"
    t0, step, i = time.time(), 0, 0
    # args.batch is the effective batch; on out-of-memory the micro-batch is
    # halved and gradients are accumulated, so each update still sees args.batch
    # pairs (review3d had no accumulation and trained at batch 2; D50).
    micro, acc = args.batch, 0
    log["effective_batch"] = args.batch
    model.train()
    while True:
        batch = pairs[i:i + micro]
        i += micro
        if len(batch) < micro:
            rng.shuffle(pairs)
            i = 0
            continue
        enc = tok([b[0] for b in batch], padding=True, truncation=True, max_length=args.max_bytes,
                  return_tensors="pt").to(device)
        lab = tok([b[1] for b in batch], padding=True, truncation=True, max_length=args.max_bytes,
                  return_tensors="pt")["input_ids"].to(device)
        lab[lab == tok.pad_token_id] = -100
        lr = args.lr * min(1.0, (step + 1) / 200)
        for g in opt.param_groups:
            g["lr"] = lr
        try:
            loss = model(**enc, labels=lab).loss
            (loss * micro / args.batch).backward()
        except torch.cuda.OutOfMemoryError:
            # Untested batch size on a T4: halve the micro-batch and go on rather than lose the run.
            opt.zero_grad()
            del enc, lab
            torch.cuda.empty_cache()
            if micro == 1:
                raise
            micro //= 2
            acc = 0
            log["micro_batch"] = micro
            print(f"out of memory: micro-batch {micro}, {args.batch // micro} accumulation steps", flush=True)
            continue
        acc += micro
        if acc < args.batch:
            continue
        acc = 0
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad()
        step += 1
        done = ((args.train_steps and step >= args.train_steps)
                or (time.time() - t0) / 60 >= args.train_minutes)
        if step % 50 == 0:
            print(f"step {step} loss {loss.item():.4f} {(time.time() - t0) / 60:.0f} min", flush=True)
        if step % args.eval_every == 0 or done:
            d = dev_cer()
            log["evals"].append({"step": step, "dev_cer": d, "minutes": (time.time() - t0) / 60})
            print(f"step {step}: development CER {d:.4f} (raw {raw_dev:.4f})", flush=True)
            if d < best:
                best = d
                model.save_pretrained(best_path)
                log["best_step"] = step
        if done:
            break
    log.update({"steps": step, "best_dev_cer": best, "train_minutes": (time.time() - t0) / 60})
    (out / "m2_train_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")

    model = (T5ForConditionalGeneration if args.tiny else AutoModelForSeq2SeqLM).from_pretrained(best_path).to(device)
    model.eval()
    test = sorted(pages, key=lambda p: p["page_id"])[: args.test_pages or None]
    res_path = out / "m2_byt5.jsonl"
    with res_path.open("w", encoding="utf-8") as f:
        for p in test:
            for e in ("tesseract", "easyocr", "ground_truth"):
                t1 = time.time()
                hyp, trunc = correct(model, tok, p[e], device, args.max_bytes)
                f.write(json.dumps({"page_id": p["page_id"], "engine": e, "model_key": MODEL_KEY,
                                    "approach": "chunked_chars_flat", "decoding": "plain", "prompt": "none",
                                    "device": device, "dtype": str(model.dtype), "raw_output": hyp,
                                    "truncated": trunc, "skipped": False, "error": None,
                                    "elapsed_sec": round(time.time() - t1, 1)}, ensure_ascii=False) + "\n")
                f.flush()
            print(f"tested {p['page_id']}", flush=True)
    print(f"wrote {res_path}", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
