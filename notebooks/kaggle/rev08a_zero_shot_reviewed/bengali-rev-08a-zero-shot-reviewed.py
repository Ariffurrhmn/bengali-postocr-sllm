"""Rev 08a: the eight zero-shot models on the reviewed BL test units (D81), Kaggle 2x T4. Outputs only.
Session A of two (D81). GPU 0: TituLLMs 1B Instruct on the OCR inputs; GPU 1: Qwen3 1.7B, Phi-4-mini,
BanglaT5, TituLLMs 1B, TituLLMs 3B Instruct on all inputs (OCR first, then clean text). About 3.7 h.

Runs correction/run_zero_shot_blocks.py per model, one lane per GPU: phase 1 the OCR inputs
(tesseract, easyocr), general-purpose models first, then the Bengali checkpoints; phase 2 the
clean-text inputs (ground_truth). Plain greedy decoding, the "en" prompt, block by block. Nothing
is scored here; eval/score_zero_shot_reviewed.py scores everything once, locally, after SymSpell
has run too. If the session runs out of time, start it again with this kernel's own output
attached: it resumes from the finished pages.

Session: Accelerator GPU T4 x2, Internet on, Secrets: HF_TOKEN (required: the script stops at once without it).
Attach: output of bengali-rev-06-ocr-reviewed-units, dataset bengali-revision-pages (test_bl.txt).
Download rev08a_zero_shot.zip afterwards."""
import glob, os, shutil, subprocess, sys, threading, time

HOURS = 11.0  # Kaggle stops a session at 12 h
W = "/kaggle/working"
OUT = f"{W}/rev08a_zero_shot"
os.makedirs(OUT, exist_ok=True)
DEADLINE = time.time() + HOURS * 3600
try:
    from kaggle_secrets import UserSecretsClient

    assert UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    sys.exit("HF_TOKEN is not attached: Add-ons > Secrets > tick HF_TOKEN, then Save & Run All")


def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    with open(f"{OUT}/setup.log", "a") as f:
        f.write(f"\n$ {cmd}\nexit {r.returncode}\n{r.stdout[-3000:]}{r.stderr[-3000:]}")
    print(("ok   " if r.returncode == 0 else "FAIL ") + cmd[:150], flush=True)
    return r.returncode == 0


REPO = "/tmp/repo"
sh(f"git clone -q https://github.com/Ariffurrhmn/bengali-postocr-sllm.git {REPO}")
assert os.path.exists(f"{REPO}/correction/run_zero_shot_blocks.py"), "git clone failed or too old: is Internet on?"
if not sh("pip install -q transformers==5.19.0 accelerate sentencepiece protobuf"):
    sh("pip install -q -U transformers accelerate sentencepiece protobuf")
token = None
try:
    from kaggle_secrets import UserSecretsClient

    token = UserSecretsClient().get_secret("HF_TOKEN")
    from huggingface_hub import login

    login(token=token)
    print("Hugging Face login ok", flush=True)
except Exception as e:
    print("WARNING: no Hugging Face login (Gemma and Llama will fail):", e, flush=True)

commit = subprocess.run(["git", "-C", REPO, "log", "--oneline", "-1"], capture_output=True, text=True).stdout
import torch, transformers

env = (f"repo commit: {commit}python: {sys.version.split()[0]}\ntorch: {torch.__version__}\ncuda: {torch.version.cuda}\n"
       f"transformers: {transformers.__version__}\ngpus: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}\n")
open(f"{OUT}/environment.txt", "w").write(env)
print(env)
assert torch.cuda.device_count() == 2, "Set Accelerator to GPU T4 x2"


def one(pattern, what):
    hits = sorted(glob.glob(pattern, recursive=True))
    assert hits, f"not found: {what} ({pattern}); attach the input listed at the top"
    return hits[0]


REGION = one("/kaggle/input/**/reviewed_region.jsonl", "reviewed_region.jsonl")
SPLIT_DIR = os.path.dirname(one("/kaggle/input/**/test_bl.txt", "test_bl.txt"))
print({"region": REGION, "split_dir": SPLIT_DIR}, flush=True)
# resume: copy finished outputs of an earlier run of this kernel, if attached
for f in glob.glob("/kaggle/input/**/rev08a_zero_shot/zs_*.jsonl", recursive=True):
    if not os.path.exists(f"{OUT}/{os.path.basename(f)}"):
        shutil.copy(f, OUT)
        print("resuming from", f, flush=True)

# Main-run GPU minutes (gpu_eval40.jsonl, 40 pages x Tesseract, EasyOCR, clean text): TituLLMs 1B
# Instruct 281, Qwen3 1.7B 69, TituLLMs 1B 70, Gemma 45, Llama 45, TituLLMs 3B Instruct 44,
# Phi-4-mini 35, BanglaT5 6. Lanes about 326 and 269 min at that rate; block-level calls may cost
# more (one length cap per block), so the session can run out and be resumed.
# General-purpose models first in each lane.
LANES = [
    [("titullm-1b-instruct", "tesseract,easyocr")],
    [("qwen3-1.7b", "tesseract,easyocr"), ("phi4-mini", "tesseract,easyocr"), ("banglat5", "tesseract,easyocr"), ("titullm-1b", "tesseract,easyocr"), ("titullm-3b-instruct", "tesseract,easyocr"), ("qwen3-1.7b", "ground_truth"), ("phi4-mini", "ground_truth"), ("banglat5", "ground_truth"), ("titullm-1b", "ground_truth"), ("titullm-3b-instruct", "ground_truth")],
]


def run_lane(i, jobs):
    log = open(f"{OUT}/lane{i}.log", "a", encoding="utf-8")
    for m, engines in jobs:
        left = (DEADLINE - time.time()) / 60
        if left < 5:
            print(f"[lane{i}] deadline reached; skipping the rest", flush=True)
            return
        cmd = [sys.executable, "run_zero_shot_blocks.py", "--ocr-region", REGION, "--split-dir", SPLIT_DIR,
               "--split", "test_bl", "--models", m, "--engines", engines,
               "--out", f"{OUT}/zs_{m}_a.jsonl", "--stop-after-minutes", f"{left - 3:.0f}"]
        t0 = time.time()
        print(f"[lane{i}] start {m} {engines}", flush=True)
        log.write(f"\n### {' '.join(cmd)}\n")
        log.flush()
        e = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i), PYTHONIOENCODING="utf-8", **({"HF_TOKEN": token} if token else {}))
        r = subprocess.run(cmd, cwd=f"{REPO}/correction", env=e, stdout=log, stderr=subprocess.STDOUT)
        print(f"[lane{i}] end {m} {engines}: exit {r.returncode}, {(time.time() - t0) / 60:.0f} min, "
              f"{(DEADLINE - time.time()) / 60:.0f} min left", flush=True)


threads = [threading.Thread(target=run_lane, args=(i, jobs)) for i, jobs in enumerate(LANES)]
for t in threads:
    t.start()
for t in threads:
    t.join()

# completeness count (no scores): lines per model and engine
import json
from collections import Counter

c = Counter()
for f in glob.glob(f"{OUT}/zs_*.jsonl"):
    for l in open(f, encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            if r.get("error") is None:
                c[(r["model_key"], r["engine"])] += 1
lines = [f"{m} {e}: {n} of 41 pages" for (m, e), n in sorted(c.items())]
open(f"{OUT}/counts.txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
shutil.make_archive(f"{W}/rev08a_zero_shot", "zip", W, "rev08a_zero_shot")
print("wrote", f"{W}/rev08a_zero_shot.zip")
