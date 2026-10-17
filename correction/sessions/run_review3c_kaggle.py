"""Third third-review GPU session on Kaggle "GPU T4 x2": all eight models on the
region-level OCR text (item C5; D42-D44).

Input: ocr_region40.jsonl (ocr/run_region_ocr.py, Kaggle CPU run of
2026-10-17): per page, the PAGE-XML regions joined in reading order, read by
Tesseract and EasyOCR; its ground_truth field is the matching reference. Every
model runs as in the main runs (plain greedy decoding, the main prompt, chunks
of about 250 tokens, float32 or float16 above 3B). The dictionary baseline is
run locally (dictionary_baseline.py).

Two lanes, one GPU each, balanced by the main runs' times; same scheme as
run_review3_kaggle.py (priority order, deadline, resume from an attached
earlier output).

Usage (Kaggle, from correction/): python sessions/run_review3c_kaggle.py --ocr-path <ocr_region40.jsonl> \\
    --out-dir /kaggle/working/review3c --hours 5
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]  # correction/: the jobs run from here
ENGINES = "tesseract,easyocr"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocr-path", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--hours", type=float, default=5.0)
    ap.add_argument("--resume-glob", default="/kaggle/input/**/review3c/*.jsonl")
    ap.add_argument("--dry-run", action="store_true", help="print the jobs only")
    args = ap.parse_args()
    deadline = time.time() + args.hours * 3600
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    for f in glob.glob(args.resume_glob, recursive=True):
        if not (out / Path(f).name).exists():
            shutil.copy(f, out / Path(f).name)
            print("resuming from", f, flush=True)

    def minutes_left():
        return max(0.0, (deadline - time.time()) / 60)

    def sweep(model):
        return (["python", "run_sweep.py", "--split", "eval", "--models", model, "--engines", ENGINES,
                 "--approaches", "chunked", "--decoding", "plain", "--prompt", "en",
                 "--ocr-path", args.ocr_path, "--out", str(out / f"c5_{model}.jsonl"),
                 "--stop-after-minutes", "{left}"], {})

    # Main-run minutes for 80 OCR pages (gpu_eval40.jsonl): TituLLMs 1B Instruct
    # about 190, Qwen3 and TituLLMs 1B about 47, the others 30 or less.
    lane0 = [sweep(m) for m in ("titullm-1b-instruct", "gemma-2b", "banglat5")]
    lane1 = [sweep(m) for m in ("qwen3-1.7b", "titullm-1b", "llama3.2-1b", "titullm-3b-instruct",
                                "phi4-mini")]

    def run_jobs(jobs, extra_env, log_name):
        log = open(out / log_name, "a", encoding="utf-8")
        for cmd, env in jobs:
            if minutes_left() < 5:
                print(f"[{log_name}] deadline reached; skipping the rest", flush=True)
                return
            cmd = [c.replace("{left}", f"{minutes_left():.0f}") for c in cmd]
            label = " ".join(cmd[1:5])
            if args.dry_run:
                print(f"[{log_name}] {' '.join(cmd)}  env={ {**extra_env, **env} }")
                continue
            t0 = time.time()
            print(f"[{log_name}] start {label}", flush=True)
            log.write(f"\n### {' '.join(cmd)} env={ {**extra_env, **env} }\n")
            log.flush()
            r = subprocess.run(cmd, cwd=HERE, env={**os.environ, **extra_env, **env}, stdout=log, stderr=log)
            print(f"[{log_name}] end {label}: exit {r.returncode}, {(time.time() - t0) / 60:.0f} min, "
                  f"{minutes_left():.0f} min left", flush=True)

    threads = [threading.Thread(target=run_jobs, args=(jobs, {"CUDA_VISIBLE_DEVICES": str(i)}, f"lane{i}.log"))
               for i, jobs in enumerate([lane0, lane1])]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print("all jobs done or out of time", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
