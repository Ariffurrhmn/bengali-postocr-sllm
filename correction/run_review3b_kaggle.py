"""Second third-review GPU session on Kaggle "GPU T4 x2" (items M3, M1, X1 leftovers).

Same scheme as run_review3_kaggle.py: jobs in priority order, no job starts
after the deadline, sweeps stop between pages, one output file per job, and
a rerun with an earlier output attached as a Kaggle input resumes it.

  setup   noise40.jsonl: the 40 references corrupted at CER 0.02-0.30 with the
          noise model of the development pages (eval/make_noise.py apply,
          deterministic; data/noise_model_dev.json)
  phase 1 (two lanes in parallel, one GPU each):
      lane 0: X1 for Phi-4-mini and TituLLMs 3B Instruct (float16; C1 showed
              float16 = float32); M3 noise sweep for Gemma-2B-it and Llama 3.2 1B;
              M1 Gemma 3 1B and Qwen3 0.6B; X1 full maps for one figure
      lane 1: M3 noise sweep for Qwen3 1.7B and Phi-4-mini; M1 Qwen3 4B and
              Llama 3.2 3B (float16)
  phase 2 (both GPUs, device_map "auto"): M1 Qwen3 8B (float16, 16 GB) and
          Gemma 3 4B (float32; Gemma overflows in float16)

M1 models run on Tesseract text, then the clean text, then EasyOCR text, so a
time cut drops EasyOCR first. All runs: plain greedy decoding, the main prompt.

Usage (Kaggle): python run_review3b_kaggle.py --ocr-path <ocr_eval40.jsonl> \\
    --out-dir /kaggle/working/review3b --hours 11
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
NOISE = "noise_0.02,noise_0.05,noise_0.10,noise_0.20,noise_0.30"
M1_INPUTS = "tesseract,ground_truth,easyocr"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocr-path", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--hours", type=float, default=11.0)
    ap.add_argument("--resume-glob", default="/kaggle/input/**/review3b/*.jsonl")
    ap.add_argument("--dry-run", action="store_true", help="print the jobs only")
    args = ap.parse_args()
    deadline = time.time() + args.hours * 3600
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    for f in glob.glob(args.resume_glob, recursive=True):
        if not (out / Path(f).name).exists():
            shutil.copy(f, out / Path(f).name)
            print("resuming from", f, flush=True)

    noise = out / "noise40.jsonl"
    if not noise.exists() and not args.dry_run:
        subprocess.run([sys.executable, "make_noise.py", "apply", "--model",
                        str(HERE.parent / "data" / "noise_model_dev.json"), "--ocr-path", args.ocr_path,
                        "--out", str(noise)], cwd=EVAL, check=True)

    def minutes_left():
        return max(0.0, (deadline - time.time()) / 60)

    def sweep(models, engines, name, ocr=None):
        return ["python", "run_sweep.py", "--split", "eval", "--models", models, "--engines", engines,
                "--approaches", "chunked", "--decoding", "plain", "--prompt", "en",
                "--ocr-path", str(ocr or args.ocr_path), "--out", str(out / f"{name}.jsonl"),
                "--stop-after-minutes", "{left}"]

    def xai(model, name=None, extra=()):
        return ["python", str(EVAL / "xai_gradcam.py"), "--model", model,
                "--ocr-path", args.ocr_path, "--engines", "tesseract,ground_truth",
                "--n-pages", "10", "--max-targets", "24", "--stop-after-minutes", "{left}",
                "--out", str(out / f"xai_{name or model}.jsonl"), *extra]

    def maps(model):
        # Full maps of every generated token on two Tesseract chunks, for one figure.
        return xai(model, f"maps_{model}", ["--n-pages", "2", "--max-targets", "0", "--save-maps"])

    m3 = lambda m, tag: (sweep(m, NOISE, f"m3_{tag}", noise), {})
    m1 = lambda m: (sweep(m, M1_INPUTS, f"m1_{m}"), {})
    lane0 = [
        (xai("phi4-mini"), {}),
        (xai("titullm-3b-instruct"), {}),
        m3("gemma-2b", "gemma"),
        m3("llama3.2-1b", "llama"),
        m1("gemma3-1b"),
        m1("qwen3-0.6b"),
        (maps("gemma-2b"), {}),
        (maps("titullm-1b-instruct"), {}),
    ]
    lane1 = [
        m3("qwen3-1.7b", "qwen3"),
        m3("phi4-mini", "phi4"),
        m1("qwen3-4b"),
        m1("llama3.2-3b"),
    ]
    both = {"POSTOCR_DEVICE_MAP": "auto", "CUDA_VISIBLE_DEVICES": "0,1"}
    phase2 = [
        (sweep("qwen3-8b", M1_INPUTS, "m1_qwen3-8b"), {**both, "POSTOCR_DTYPE": "fp16"}),
        (sweep("gemma3-4b", M1_INPUTS, "m1_gemma3-4b"), {**both, "POSTOCR_DTYPE": "fp32"}),
    ]

    def run_jobs(jobs, extra_env, log_name):
        log = open(out / log_name, "a", encoding="utf-8")
        for cmd, env in jobs:
            if minutes_left() < 5:
                print(f"[{log_name}] deadline reached; skipping the rest", flush=True)
                return
            cmd = [c.replace("{left}", f"{minutes_left():.0f}") for c in cmd]
            full_env = {**os.environ, **extra_env, **env}
            label = " ".join(cmd[1:5])
            if args.dry_run:
                print(f"[{log_name}] {' '.join(cmd)}  env={ {**extra_env, **env} }")
                continue
            t0 = time.time()
            print(f"[{log_name}] start {label}", flush=True)
            log.write(f"\n### {' '.join(cmd)} env={ {**extra_env, **env} }\n")
            log.flush()
            r = subprocess.run(cmd, cwd=HERE, env=full_env, stdout=log, stderr=log)
            print(f"[{log_name}] end {label}: exit {r.returncode}, {(time.time() - t0) / 60:.0f} min, "
                  f"{minutes_left():.0f} min left", flush=True)

    gpu = lambda i: {"CUDA_VISIBLE_DEVICES": str(i)}
    threads = [threading.Thread(target=run_jobs, args=(jobs, gpu(i), f"lane{i}.log"))
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
