"""Last third-review GPU session on Kaggle "GPU T4 x2" (items M2, M7; C5 extra).

  phase 1 (two lanes, one GPU each):
      lane 0: M2 fine-tuning pilot (finetune_byt5.py): ByT5-small trained on
              synthetic noise for --m2-minutes, then tested on the 40 pages
      lane 1: Qwen3 4B on the region-level OCR text (the size trend of M1 on
              the cleaner input; float16)
  phase 2 (both GPUs, device_map "auto"): M7 Qwen3-32B in 4-bit (NF4,
          bitsandbytes) on the 10 pages of the attribution analysis
          (data/m7_pages.txt), Tesseract, clean and EasyOCR text

Same scheme as run_review3_kaggle.py: priority order, deadline, one output
file per job, resume from an attached earlier output (the fine-tuning job
restarts from scratch if it did not finish).

Usage (Kaggle): python run_review3d_kaggle.py --ocr-path <ocr_eval40.jsonl> \\
    --region-path <ocr_region40.jsonl> --dev-path <ocr_dev.jsonl> \\
    --out-dir /kaggle/working/review3d --hours 6
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

HERE = Path(__file__).resolve().parent
M7_PAGES = ",".join((HERE.parent / "data" / "m7_pages.txt").read_text(encoding="utf-8").split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocr-path", required=True)
    ap.add_argument("--region-path", required=True)
    ap.add_argument("--dev-path", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--hours", type=float, default=6.0)
    ap.add_argument("--m2-minutes", type=float, default=100)
    ap.add_argument("--resume-glob", default="/kaggle/input/**/review3d/*.jsonl")
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

    def sweep(model, engines, name, ocr, pages=None):
        return ["python", "run_sweep.py", "--split", "eval", "--models", model, "--engines", engines,
                "--approaches", "chunked", "--decoding", "plain", "--prompt", "en", "--ocr-path", ocr,
                "--out", str(out / f"{name}.jsonl"), "--stop-after-minutes", "{left}",
                *(["--pages", pages] if pages else [])]

    lane0 = [(["python", "finetune_byt5.py", "--ocr-path", args.ocr_path, "--dev-path", args.dev_path,
               "--out-dir", str(out), "--train-minutes", str(args.m2_minutes)], {})]
    lane1 = [(sweep("qwen3-4b", "tesseract,easyocr", "c5_qwen3-4b", args.region_path), {})]
    phase2 = [(sweep("qwen3-32b", "tesseract,ground_truth,easyocr", "m7_qwen3-32b", args.ocr_path, M7_PAGES),
               {"POSTOCR_QUANT": "4bit", "POSTOCR_DEVICE_MAP": "auto", "CUDA_VISIBLE_DEVICES": "0,1"})]

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
    run_jobs(phase2, {}, "phase2.log")
    print("all phases done or out of time", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
