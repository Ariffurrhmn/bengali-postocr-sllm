"""Fully trained, task-specific OCR correctors (revision 2026-11, supervisor point 2):
google/byt5-small or Vacaspati/BanglaByT5, selected on held-out validation books.

Data (all from the frozen v2 split, Datasets/revision_2026-11):
  real       Aligned (OCR -> reference) pieces of every text block of the TRAIN pages, both
             engines, from the fresh region-level OCR (ocr/run_revision_ocr.py). Blocks are
             cut into pieces of <=300 reference characters through a Levenshtein alignment
             (corrector_common.aligned_pairs); pieces whose OCR/reference length ratio is
             outside 1/3..3, or whose OCR CER exceeds --max-real-cer (0.6), are dropped
             (failed or reordered crops). Repeated --real-repeat times.
  synthetic  Clean text corrupted with a character noise model fitted on the TRAIN pages'
             region OCR only (eval/make_noise.fit), scaled to a random CER of 0.02-0.40
             (10% left clean so the model learns to copy). Clean text: Bengali Wikipedia
             segments (wikimedia/wikipedia 20231101.bn, streamed, as the pilot) plus the
             TRAIN pages' reference lines repeated --train-line-repeat times.
  filter     Every training piece (real or synthetic, input or target) that shares any
             25-character string with a VALIDATION or TEST reference is dropped.
Selection    Every --eval-every optimizer steps the model corrects the VALIDATION pages'
             region OCR (both engines, block by block); the checkpoint with the lowest mean
             page CER is kept (best/). Early stopping after --patience evaluations without
             improvement, or at the time budget. Test pages are never read here except
             their references, for the overlap filter.
Resume       last/ (weights + optimizer + step + RNG + data position) is written at every
             evaluation; --resume-dir continues from it.
Logs         curve.jsonl (step, loss, lr, val CER, minutes, peak GPU memory), train_log.json.

Usage (Kaggle GPU):
    python correction/train_corrector.py --model banglabyt5 --ocr-region <shard_b_region.jsonl> \\
        <shard_c_region.jsonl> --test-refs <shard_a_region.jsonl> --split-dir <revision_2026-11 lists> \\
        --out-dir /kaggle/working/banglabyt5 --hours 9
Local smoke test: add --tiny --no-wiki --max-steps 6 --eval-every 3.
"""
import argparse
import json
import pickle
import random
import re
import shutil
import sys
import time
from pathlib import Path

from corrector_common import (MODELS, REPO, aligned_pairs, banned_ngrams, correct_texts, flat, load_jsonl,
                              load_model, make_labels, overlaps, read_ids)

sys.path.insert(0, str(REPO / "eval"))
from make_noise import Noiser, fit, prep  # noqa: E402
from metrics import cer  # noqa: E402

SEED = 403


def segments_from(text, lo=60, hi=300):
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


def bengali_share(text):
    letters = [c for c in text if c.isalpha()]
    return sum(0x0980 <= ord(c) <= 0x09FF for c in letters) / len(letters) if letters else 0.0


def calibrate(noiser, segs, target, rng):
    """Noise factor k so that the mean CER of corrupted segments is about `target`."""
    sample = segs[:300]
    lo, hi = 0.0, 20.0
    for _ in range(18):
        k = (lo + hi) / 2
        r = random.Random(rng.random())
        m = sum(cer(s, noiser.corrupt(s, k, r)) for s in sample) / len(sample)
        lo, hi = (k, hi) if m < target else (lo, k)
    return (lo + hi) / 2


def build_data(args, out, rng, train_recs, banned, log):
    # Real pairs.
    real, dropped_ratio, dropped_overlap = [], 0, 0
    for rec in train_recs:
        for reg in rec["regions"]:
            for eng in ("tesseract", "easyocr"):
                for o, r in aligned_pairs(reg[eng], reg["ground_truth"]):
                    if not r or not (1 / 3 <= (len(o) + 1) / (len(r) + 1) <= 3):
                        dropped_ratio += 1
                        continue
                    # Pieces whose OCR barely matches the reference (reordered or lost text in
                    # the crop) would teach rewriting rather than correction; fixed cap, set
                    # from the training pairs' CER distribution, never from validation/test.
                    if cer(r, o) > args.max_real_cer:
                        dropped_ratio += 1
                        continue
                    if overlaps(r, banned) or overlaps(o, banned):
                        dropped_overlap += 1
                        continue
                    real.append((o, r))
    # Noise model from the TRAIN pages only.
    fit_in = out / "noise_fit_input.jsonl"
    with fit_in.open("w", encoding="utf-8") as f:
        for rec in train_recs:
            f.write(json.dumps({"page_id": rec["page_id"], "ground_truth": rec["ground_truth"],
                                "tesseract": rec["tesseract"], "easyocr": rec["easyocr"]}, ensure_ascii=False) + "\n")
    fit(fit_in, out / "noise_model_train.json")
    fit_in.unlink()
    noiser = Noiser(json.loads((out / "noise_model_train.json").read_text(encoding="utf-8")))
    # Clean text.
    segs, wiki_dropped = [], 0
    if not args.no_wiki:
        from datasets import load_dataset

        ds = load_dataset("wikimedia/wikipedia", "20231101.bn", split="train", streaming=True)
        for art in ds:
            for s in segments_from(art["text"]):
                if bengali_share(s) < 0.9:
                    continue
                if overlaps(s, banned):
                    wiki_dropped += 1
                else:
                    segs.append(s)
            if len(segs) >= args.wiki_segments:
                break
    lines = [l.strip() for rec in train_recs for l in prep(rec["ground_truth"]).split("\n") if len(l.strip()) >= 20]
    lines = [l for l in lines if not overlaps(l, banned)]
    segs += lines * args.train_line_repeat
    k20 = calibrate(noiser, [flat(s) for s in (lines or segs)], 0.20, rng)
    per_cer = k20 / 0.20
    synth = []
    for s in segs:
        level = 0.0 if rng.random() < 0.1 else rng.uniform(0.02, 0.40)
        noisy = noiser.corrupt(s, level * per_cer, rng) if level else s
        if overlaps(noisy, banned):
            continue
        synth.append((flat(noisy), flat(s)))
    # How noisy the real pairs are (pieces where the crop OCR reordered or lost text show up
    # as very high CER); logged so the smoke test shows the distribution.
    rc = sorted(cer(r, o) for o, r in real)
    if rc:
        log["real_pair_cer_quantiles"] = {q: round(rc[min(len(rc) - 1, int(q * len(rc)))], 3)
                                          for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)}
        print("real pair CER quantiles:", log["real_pair_cer_quantiles"], flush=True)
    pairs = real * args.real_repeat + synth
    rng.shuffle(pairs)
    log.update({"real_pairs": len(real), "real_repeat": args.real_repeat, "synthetic_pairs": len(synth),
                "wiki_segments": len(segs) - len(lines) * args.train_line_repeat, "train_lines": len(lines),
                "dropped_real_ratio": dropped_ratio, "dropped_real_overlap": dropped_overlap,
                "dropped_wiki_overlap": wiki_dropped, "noise_factor_per_cer": per_cer, "pairs_total": len(pairs)})
    print(f"real pairs {len(real)} (x{args.real_repeat}), synthetic {len(synth)}, total {len(pairs)}; "
          f"dropped: real ratio {dropped_ratio}, real overlap {dropped_overlap}, wiki overlap {wiki_dropped}",
          flush=True)
    return pairs


def val_cer(model, tok, val_recs, device, max_bytes):
    model.eval()
    texts, keys = [], []
    for rec in val_recs:
        for eng in ("tesseract", "easyocr"):
            for reg in rec["regions"]:
                texts.append(reg[eng])
                keys.append((rec["page_id"], eng))
    outs, trunc = correct_texts(model, tok, texts, device, max_bytes)
    model.train()
    hyp = {}
    for k, o in zip(keys, outs):
        hyp.setdefault(k, []).append(o)
    scores = [cer(rec["ground_truth"], "\n".join(hyp[(rec["page_id"], e)]))
              for rec in val_recs for e in ("tesseract", "easyocr")]
    # Share of blocks that hit the length cap without an end token (a stuck model shows ~1.0).
    sample = next(((t[:80], o[:80]) for t, o in zip(texts, outs) if len(t) > 40), None)
    return sum(scores) / len(scores), sum(trunc) / max(1, len(trunc)), sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--ocr-region", type=Path, nargs="+", required=True,
                    help="region OCR files (run_revision_ocr.py) holding the train and validation pages")
    ap.add_argument("--test-refs", type=Path, nargs="*", default=[],
                    help="OCR files of the test pages; only their references are read (overlap filter)")
    ap.add_argument("--split-dir", type=Path, required=True, help="folder with train.txt, val.txt, test_bl.txt")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--resume-dir", type=Path, default=None, help="an earlier out-dir holding last/")
    ap.add_argument("--hours", type=float, default=9.0)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--max-bytes", type=int, default=1024)
    ap.add_argument("--wiki-segments", type=int, default=60000)
    ap.add_argument("--train-line-repeat", type=int, default=10)
    ap.add_argument("--real-repeat", type=int, default=10)
    ap.add_argument("--max-real-cer", type=float, default=0.6,
                    help="drop real pieces whose OCR has a higher CER than this against the reference")
    ap.add_argument("--no-wiki", action="store_true")
    ap.add_argument("--tiny", action="store_true", help="tiny random model (local smoke test)")
    ap.add_argument("--stop-after-steps", type=int, default=0, help="exit after this many steps this run (resume test)")
    args = ap.parse_args()

    import torch

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    train_ids, val_ids = set(read_ids(args.split_dir / "train.txt")), set(read_ids(args.split_dir / "val.txt"))
    test_ids = set(read_ids(args.split_dir / "test_bl.txt"))
    assert not (train_ids & val_ids) and not (train_ids & test_ids) and not (val_ids & test_ids), "split overlap"
    book = lambda p: re.sub(r"_\d+$", "", p)  # same rule as eval/check_leakage.py
    shared = {book(p) for p in test_ids} & {book(p) for p in train_ids | val_ids}
    assert not shared, f"test books also in train/val: {sorted(shared)}"
    recs = {}
    for p in args.ocr_region:
        for r in load_jsonl(p):
            recs[r["page_id"]] = r
    train_recs = sorted((recs[i] for i in train_ids if i in recs), key=lambda r: r["page_id"])
    val_recs = sorted((recs[i] for i in val_ids if i in recs), key=lambda r: r["page_id"])
    missing = sorted((train_ids | val_ids) - set(recs))
    test_refs = [r["ground_truth"] for p in args.test_refs for r in load_jsonl(p) if r["page_id"] in test_ids]
    test_refs += [g["ground_truth"] for p in args.test_refs for r in load_jsonl(p) if r["page_id"] in test_ids
                  for g in r.get("regions", [])]
    assert all(r["page_id"] not in test_ids for r in train_recs + val_recs)
    banned = banned_ngrams([r["ground_truth"] for r in val_recs] + test_refs)
    log = {"model": MODELS[args.model], "train_pages": len(train_recs), "val_pages": len(val_recs),
           "missing_pages": missing, "test_reference_texts_for_filter": len(test_refs), "args": vars(args)}
    log["args"] = {k: str(v) for k, v in log["args"].items()}
    print(f"{args.model}: {len(train_recs)} train pages, {len(val_recs)} val pages, missing {missing}", flush=True)
    assert train_recs and val_recs, "no train or no validation pages found in the OCR files"

    state_file = (args.resume_dir / "last" / "state.pkl") if args.resume_dir else None
    resuming = bool(state_file and state_file.exists())
    if resuming:
        st = pickle.loads(state_file.read_bytes())
        pairs, log = st["pairs"], st["log"]
        rng.setstate(st["rng"])
        if args.resume_dir.resolve() != out.resolve():
            for sub in ("best", "last"):
                if (args.resume_dir / sub).exists() and not (out / sub).exists():
                    shutil.copytree(args.resume_dir / sub, out / sub)
            for f in ("curve.jsonl", "noise_model_train.json"):
                if (args.resume_dir / f).exists() and not (out / f).exists():
                    shutil.copy(args.resume_dir / f, out / f)
        print(f"resuming at step {st['step']}", flush=True)
    else:
        pairs = build_data(args, out, rng, train_recs, banned, log)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok, model = load_model(args.model, args.tiny, path=(out / "last") if resuming else None)
    model.to(device)
    if not args.tiny and ("model_info" not in log or not resuming):
        # Licence, tokenizer and the exact revision loaded (resolved from the Hub, checked
        # against the pin); a resumed run keeps the record from the first session.
        from corrector_common import model_info

        log["model_info"] = model_info(args.model, None if resuming else model)
        (out / "model_info.json").write_text(json.dumps(log["model_info"], indent=1), encoding="utf-8")
        print("model revision:", log["model_info"].get("resolved_revision"),
              "| loaded:", log["model_info"].get("loaded_commit_hash"), "| licence:", log["model_info"]["license"], flush=True)
    opt =torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    step, i, best, bad, elapsed0 = 0, 0, float("inf"), 0, 0.0
    if resuming:
        opt.load_state_dict(torch.load(out / "last" / "optimizer.pt", map_location=device))
        step, i, best, bad, elapsed0 = st["step"], st["i"], st["best"], st["bad"], st["elapsed"]
        torch.set_rng_state(st["torch_rng"])
    else:
        log["raw_val_cer"] = sum(cer(r["ground_truth"], r[e]) for r in val_recs for e in ("tesseract", "easyocr")) / (2 * len(val_recs))
        print(f"raw OCR CER on the validation pages (region level): {log['raw_val_cer']:.4f}", flush=True)
    curve = (out / "curve.jsonl").open("a", encoding="utf-8")
    t0 = time.time() - elapsed0
    run_start, run_steps = time.time(), 0
    micro, acc, losses = args.batch, 0, []
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model.train()

    def save_last():
        (out / "last").mkdir(exist_ok=True)
        model.save_pretrained(out / "last")
        torch.save(opt.state_dict(), out / "last" / "optimizer.pt")
        (out / "last" / "state.pkl").write_bytes(pickle.dumps({
            "pairs": pairs, "log": log, "rng": rng.getstate(), "torch_rng": torch.get_rng_state(),
            "step": step, "i": i, "best": best, "bad": bad, "elapsed": time.time() - t0}))

    stop_reason = None
    while True:
        batch = pairs[i:i + micro]
        i += micro
        if len(batch) < micro:
            rng.shuffle(pairs)
            i = 0
            continue
        enc = tok([b[0] for b in batch], padding=True, truncation=True, max_length=args.max_bytes,
                  return_tensors="pt").to(device)
        lab = make_labels(tok, [b[1] for b in batch], args.max_bytes).to(device)
        lr = args.lr * min(1.0, (step + 1) / args.warmup)
        for g in opt.param_groups:
            g["lr"] = lr
        try:
            loss = model(**enc, labels=lab).loss
            (loss * micro / args.batch).backward()
        except torch.cuda.OutOfMemoryError:
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
        losses.append(loss.item())
        acc += micro
        if acc < args.batch:
            continue
        acc = 0
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad()
        step += 1
        run_steps += 1
        minutes = (time.time() - t0) / 60
        if args.max_steps and step >= args.max_steps:
            stop_reason = "max_steps"
        elif minutes >= args.hours * 60:
            stop_reason = "time budget"
        if step % 50 == 0:
            rate = run_steps / max(1e-9, (time.time() - run_start) / 3600)
            print(f"step {step} loss {sum(losses[-50:]) / len(losses[-50:]):.4f} {minutes:.0f} min "
                  f"({rate:.0f} steps/h)", flush=True)
        if step % args.eval_every == 0 or stop_reason:
            v, trunc_share, sample = val_cer(model, tok, val_recs, device, args.max_bytes)
            peak = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else 0.0
            print(f"  sample (validation block): OCR {sample[0]!r} -> {sample[1]!r}" if sample else "", flush=True)
            row = {"step": step, "loss": sum(losses) / max(1, len(losses)), "lr": lr, "val_cer": v,
                   "val_truncated_share": round(trunc_share, 3),
                   "minutes": minutes, "peak_gpu_gib": round(peak, 2),
                   "steps_per_hour": run_steps / max(1e-9, (time.time() - run_start) / 3600)}
            losses = []
            curve.write(json.dumps(row) + "\n")
            curve.flush()
            print(f"step {step}: validation CER {v:.4f} (raw {log['raw_val_cer']:.4f}), peak {peak:.1f} GiB",
                  flush=True)
            if v < best:
                best, bad = v, 0
                model.save_pretrained(out / "best")
                tok.save_pretrained(out / "best")
                log["best_step"] = step
            else:
                bad += 1
                if bad >= args.patience and not stop_reason:
                    stop_reason = f"early stopping (patience {args.patience})"
            save_last()
        if args.stop_after_steps and run_steps >= args.stop_after_steps and not stop_reason:
            save_last()
            print(f"stopping this run after {run_steps} steps (resume test)", flush=True)
            return
        if stop_reason:
            break
    log.update({"steps": step, "best_val_cer": best, "train_minutes": (time.time() - t0) / 60,
                "stop_reason": stop_reason, "device": device,
                "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
                "torch": torch.__version__, "cuda": torch.version.cuda})
    import platform
    from importlib.metadata import version as pkg_version

    log["python"] = platform.python_version()
    for pkg in ("transformers", "accelerate", "tokenizers"):
        try:
            log[pkg] = pkg_version(pkg)
        except Exception:
            log[pkg] = "not available"
    log["dtype"] = "float32 (no autocast)"
    (out / "train_log.json").write_text(json.dumps(log, indent=1, default=str), encoding="utf-8")
    print(f"done: {stop_reason}; best validation CER {best:.4f} at step {log.get('best_step')}", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
