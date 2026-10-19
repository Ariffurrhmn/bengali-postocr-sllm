"""External-review GPU session on Kaggle "GPU T4 x2" (review of 2026-10-19, items 4 and 5).

Item 4: do the repetition constraints damage copying models because they also
act on the prompt? The four general-purpose models rerun the decoding ablation
with the constraints restricted to generated tokens (presets penalty_gen,
ngram_gen, guarded_gen in models.py), on Tesseract text and on the clean text,
40 pages, main prompt and chunking. Plain and whole-sequence results exist.

Item 5: the two instruction-tuned TituLLMs checkpoints with the plain-text
completion prompt instead of their shipped template, on Tesseract text (last,
if time is left).

Two lanes, one GPU each, in priority order; deadline and resume as in
run_review3c_kaggle.py. Earlier whole-sequence runs took 25-50 min per model
and preset for 80 pages.

Usage (Kaggle, from correction/): python sessions/run_review4_kaggle.py --ocr-path <ocr_eval40.jsonl> \\
    --out-dir /kaggle/working/review4 --hours 6
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocr-path", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--hours", type=float, default=6.0)
    ap.add_argument("--resume-glob", default="/kaggle/input/**/review4/*.jsonl")
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

    def sweep(model, engines, decoding, prompt, name):
        return (["python", "run_sweep.py", "--split", "eval", "--models", model, "--engines", engines,
                 "--approaches", "chunked", "--decoding", decoding, "--prompt", prompt,
                 "--ocr-path", args.ocr_path, "--out", str(out / f"{name}.jsonl"),
                 "--stop-after-minutes", "{left}"], {})

    def gen(model, tag, presets):
        return [sweep(model, "tesseract,ground_truth", d, "en", f"r4_{d}_{tag}") for d in presets]

    both = ("guarded_gen",)
    halves = ("penalty_gen", "ngram_gen")
    # Balanced by the earlier runs: Qwen3 1.7B about 50 min per preset, the others about 30.
    lane0 = [*gen("qwen3-1.7b", "qwen3", both), *gen("gemma-2b", "gemma", both),
             *gen("qwen3-1.7b", "qwen3", halves), *gen("gemma-2b", "gemma", halves),
             sweep("titullm-3b-instruct", "tesseract", "plain", "completion", "r4_completion_titullm3b")]
    lane1 = [*gen("phi4-mini", "phi4", both), *gen("llama3.2-1b", "llama", both),
             *gen("phi4-mini", "phi4", halves),
             *gen("llama3.2-1b", "llama", halves),
             sweep("titullm-1b-instruct", "tesseract", "plain", "completion", "r4_completion_titullm1b")]

    def run_jobs(jobs, extra_env, log_name):
        log = open(out / log_name, "a", encoding="utf-8")
        for cmd, env in jobs:
            if minutes_left() < 5:
                print(f"[{log_name}] deadline reached; skipping the rest", flush=True)
                return
            cmd = [c.replace("{left}", f"{minutes_left():.0f}") for c in cmd]
            label = " ".join(cmd[5:6] + cmd[11:12] + cmd[13:14])
            if args.dry_run:
                print(f"[{log_name}] {' '.join(cmd)}")
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
