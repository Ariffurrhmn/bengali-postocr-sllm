"""Rev 07: the two trained correctors on the reviewed BL test units (D76, D77, D78), Kaggle 2x T4.
Runs correction/score_trained_corrector.py once per model (one model per GPU) on split test_bl, using the
OCR of rev 06 (reviewed units in the reviewed order). Main metric: per-block CER, character-weighted
within a page, then the mean over pages with a paired bootstrap (D78); joined-page CER and the
table-of-contents / other split are reported too. Nothing is trained or tuned. Run it once.
Session: Accelerator GPU T4 x2, Internet on.
Attach: output of bengali-rev-03-train-both (weights), output of bengali-rev-06-ocr-reviewed-units,
dataset bengali-revision-pages (val.txt, test_bl.txt). Download rev07_score.zip afterwards."""
import glob, json, os, shutil, subprocess, sys, time

W = "/kaggle/working"
OUT_DIR = f"{W}/rev07_score"
os.makedirs(OUT_DIR, exist_ok=True)


def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    with open(f"{OUT_DIR}/setup.log", "a") as f:
        f.write(f"\n$ {cmd}\nexit {r.returncode}\n{r.stdout[-3000:]}{r.stderr[-3000:]}")
    print(("ok   " if r.returncode == 0 else "FAIL ") + cmd[:150], flush=True)
    return r.returncode == 0


REPO = "/tmp/repo"
sh(f"git clone -q https://github.com/Ariffurrhmn/bengali-postocr-sllm.git {REPO}")
assert os.path.exists(f"{REPO}/correction/score_trained_corrector.py"), "git clone failed: is Internet on?"
src = open(f"{REPO}/correction/score_trained_corrector.py", encoding="utf-8").read()
assert "block_scores" in src, "clone is older than the per-block scoring change"
if not sh("pip install -q transformers==5.19.0 accelerate sentencepiece jiwer rapidfuzz scipy"):
    sh("pip install -q -U transformers accelerate sentencepiece jiwer rapidfuzz scipy")
commit = subprocess.run(["git", "-C", REPO, "log", "--oneline", "-1"], capture_output=True, text=True).stdout
import torch, transformers
env = (f"repo commit: {commit}python: {sys.version.split()[0]}\ntorch: {torch.__version__}\ncuda: {torch.version.cuda}\n"
       f"transformers: {transformers.__version__}\ngpus: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}\n")
open(f"{OUT_DIR}/environment.txt", "w").write(env)
print(env)
assert torch.cuda.device_count() == 2, "Set Accelerator to GPU T4 x2"


def one(pattern, what):
    hits = sorted(glob.glob(pattern, recursive=True))
    assert hits, f"not found: {what} ({pattern}); attach the input listed at the top"
    return hits[0]


WEIGHTS = {m: os.path.dirname(one(f"/kaggle/input/**/train/{m}/best/model.safetensors", f"{m} weights")) for m in ("byt5-small", "banglabyt5")}
REGION = one("/kaggle/input/**/reviewed_region.jsonl", "reviewed_region.jsonl")
SPLIT_DIR = os.path.dirname(one("/kaggle/input/**/test_bl.txt", "test_bl.txt"))
print(json.dumps({"weights": WEIGHTS, "region": REGION, "split_dir": SPLIT_DIR}, indent=1))
print(os.path.basename(REGION), sum(1 for _ in open(REGION, encoding="utf-8")), "pages")

procs = []
for gpu, model in enumerate(("byt5-small", "banglabyt5")):
    out = f"{OUT_DIR}/{model}_test_bl"
    cmd = (f"{sys.executable} score_trained_corrector.py --model {model} --weights {WEIGHTS[model]} "
           f"--ocr-region {REGION} --split-dir {SPLIT_DIR} --split test_bl --out-dir {out} > {OUT_DIR}/{model}_test_bl.log 2>&1")
    env_ = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONIOENCODING="utf-8")
    procs.append((model, subprocess.Popen(cmd, shell=True, cwd=f"{REPO}/correction", env=env_)))
t0 = time.time()
for model, p in procs:
    p.wait()
    print(f"{model} finished with exit {p.returncode} after {(time.time() - t0) / 60:.1f} min")
for model in ("byt5-small", "banglabyt5"):
    f = f"{OUT_DIR}/{model}_test_bl/summary.txt"
    print("=" * 20, model)
    print(open(f, encoding="utf-8").read() if os.path.exists(f) else f"MISSING: see {OUT_DIR}/{model}_test_bl.log")
shutil.make_archive(f"{W}/rev07_score", "zip", W, "rev07_score")
print("wrote", f"{W}/rev07_score.zip")
