"""Third-review GPU jobs on a Kaggle "GPU T4 x2" session (items C1, C2, X1, C4).

Kaggle counts session time, not GPUs, so a two-GPU session costs the same as
one. The jobs run in three phases, each in priority order, and no job starts
after the deadline (sweeps also stop between pages):

  phase 1 (both GPUs, device_map "auto"; models above 3B in float32):
      C1  first-token check of TituLLMs 3B Instruct in fp16 (one GPU, as in the
          main run) and fp32; then TituLLMs 3B Instruct and Phi-4-mini in fp32
          on all inputs; X1 attribution for these two models
  phase 2 (two lanes in parallel, one GPU each):
      lane 0: C2 TituLLMs 1B with a completion prompt; BanglaT5 with its
              normaliser; X1 for five models; C4 for Gemma-2B-it
      lane 1: X1 for Qwen3 1.7B; C4 for Llama 3.2 1B, then Qwen3 1.7B
  phase 3 (whatever time is left, two lanes): C4 on EasyOCR text.

C4 = decoding presets guarded, penalty_only, ngram_only (plain already exists),
on Tesseract text and the clean-text control. One output file per job, so the
lanes never write to the same file. Rerunning with an earlier output attached
as a Kaggle input resumes it (files are copied into --out-dir first).

Usage (Kaggle): python run_review3_kaggle.py --ocr-path <ocr_eval40.jsonl> \\
    --out-dir /kaggle/working/review3 --hours 9.5
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
EVAL = HERE.parent / "eval"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocr-path", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--hours", type=float, default=9.5)
    ap.add_argument("--resume-glob", default="/kaggle/input/**/review3/*.jsonl")
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

    def sweep(models, engines, decoding, prompt, name):
        return ["python", "run_sweep.py", "--split", "eval", "--models", models, "--engines", engines,
                "--approaches", "chunked", "--decoding", decoding, "--prompt", prompt,
                "--ocr-path", args.ocr_path, "--out", str(out / f"{name}.jsonl"),
                "--stop-after-minutes", "{left}"]

    def xai(model, name=None):
        return ["python", str(EVAL / "xai_gradcam.py"), "--model", model,
                "--ocr-path", args.ocr_path, "--engines", "tesseract,ground_truth",
                "--n-pages", "10", "--max-targets", "24", "--stop-after-minutes", "{left}",
                "--out", str(out / f"xai_{name or model}.jsonl")]

    def first_token(model, precision):
        return ["python", "first_token_check.py", "--model", model, "--ocr-path", args.ocr_path,
                "--out", str(out / f"first_token_{model}_{precision}.jsonl")]

    both_fp32 = {"POSTOCR_DTYPE": "fp32", "POSTOCR_DEVICE_MAP": "auto", "CUDA_VISIBLE_DEVICES": "0,1"}
    gpu = lambda i: {"CUDA_VISIBLE_DEVICES": str(i)}
    c4 = lambda m, eng, tag: [(sweep(m, eng, d, "en", f"c4_{d}_{tag}"), {}) for d in
                              ("guarded", "penalty_only", "ngram_only")]

    phase1 = [
        (first_token("titullm-3b-instruct", "fp16"), {"POSTOCR_DTYPE": "fp16", "CUDA_VISIBLE_DEVICES": "0"}),
        (first_token("titullm-3b-instruct", "fp32"), both_fp32),
        (sweep("titullm-3b-instruct", "tesseract,easyocr,ground_truth", "plain", "en", "c1_fp32"), both_fp32),
        (sweep("phi4-mini", "tesseract,ground_truth", "plain", "en", "c1_fp32_phi4"), both_fp32),
        (xai("titullm-3b-instruct"), both_fp32),
        (xai("phi4-mini"), both_fp32),
    ]
    lane0 = [
        (sweep("titullm-1b", "tesseract,easyocr,ground_truth", "plain", "completion", "c2_completion"), {}),
        (sweep("banglat5", "tesseract,easyocr,ground_truth", "plain", "normalized", "c2_banglat5_norm"), {}),
        *[(xai(m), {}) for m in ("gemma-2b", "llama3.2-1b", "titullm-1b", "titullm-1b-instruct", "banglat5")],
        *c4("gemma-2b", "tesseract,ground_truth", "gemma"),
    ]
    lane1 = [
        (xai("qwen3-1.7b"), {}),
        *c4("llama3.2-1b", "tesseract,ground_truth", "llama"),
        *c4("qwen3-1.7b", "tesseract,ground_truth", "qwen3"),
    ]
    phase3 = [
        [*c4("gemma-2b", "easyocr", "gemma_easy"), *c4("phi4-mini", "tesseract,ground_truth", "phi4")],
        [*c4("llama3.2-1b", "easyocr", "llama_easy"), *c4("qwen3-1.7b", "easyocr", "qwen3_easy")],
    ]

    def run_jobs(jobs, extra_env, log_name):
        log = open(out / log_name, "a", encoding="utf-8")
        for cmd, env in jobs:
            if minutes_left() < 5:
                print(f"[{log_name}] deadline reached; skipping the rest", flush=True)
                return
            cmd = [c.replace("{left}", f"{minutes_left():.0f}") for c in cmd]
            full_env = {**os.environ, **extra_env, **env}
            cwd = HERE
            label = " ".join(cmd[1:5])
            if args.dry_run:
                print(f"[{log_name}] {' '.join(cmd)}  env={ {**extra_env, **env} }")
                continue
            t0 = time.time()
            print(f"[{log_name}] start {label}", flush=True)
            log.write(f"\n### {' '.join(cmd)} env={ {**extra_env, **env} }\n")
            log.flush()
            r = subprocess.run(cmd, cwd=cwd, env=full_env, stdout=log, stderr=log)
            print(f"[{log_name}] end {label}: exit {r.returncode}, {(time.time() - t0) / 60:.0f} min, "
                  f"{minutes_left():.0f} min left", flush=True)

    def parallel(job_lists):
        threads = [threading.Thread(target=run_jobs, args=(jobs, gpu(i), f"lane{i}.log"))
                   for i, jobs in enumerate(job_lists)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    # Phase 1 gets at most 3.5 h, so the parallel phases always run.
    phase1_deadline = min(deadline, time.time() + 3.5 * 3600)
    saved = deadline
    deadline = phase1_deadline
    run_jobs(phase1, {}, "phase1.log")
    deadline = saved
    parallel([lane0, lane1])
    parallel(phase3)
    print("all phases done or out of time", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
