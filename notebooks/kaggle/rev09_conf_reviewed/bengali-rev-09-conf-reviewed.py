"""Rev 09: word confidences for the reviewed BL scoring units (D84), Kaggle CPU (no GPU quota).
Same crops and engine builds as rev 06: Tesseract 5.3.4 (conda-forge, tessdata_best ben, psm 6) and
EasyOCR 1.7.2. Saves Tesseract word confidences and EasyOCR box confidences per unit, and whether the text
matches the stored rev 06 text. No language model runs and nothing is scored here.
Output: /kaggle/working/conf/reviewed_conf.jsonl (+ environment, log, run_info).
Session: Accelerator None (CPU), Internet on.
Attach: dataset bengali-revision-pages (images), dataset bengali-bl-reviewed-units (reviewed_units.json),
and the output of bengali-rev-06-ocr-reviewed-units (reviewed_region.jsonl). To resume a stopped run,
also attach this notebook's earlier output."""
import glob, hashlib, json, os, shutil, subprocess, sys, time

W = "/kaggle/working"
os.makedirs(W, exist_ok=True)


def sh(cmd, log=f"{W}/setup.log"):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    with open(log, "a") as f:
        f.write(f"\n$ {cmd}\nexit {r.returncode}\n{r.stdout[-3000:]}{r.stderr[-3000:]}")
    print(("ok   " if r.returncode == 0 else "FAIL ") + cmd[:150], flush=True)
    return r.returncode == 0


REPO = "/tmp/repo"
sh(f"git clone -q https://github.com/Ariffurrhmn/bengali-postocr-sllm.git {REPO}")
assert os.path.exists(f"{REPO}/ocr/run_reviewed_units_conf.py"), "clone is older than the confidence script"
commit = subprocess.run(["git", "-C", REPO, "log", "--oneline", "-1"], capture_output=True, text=True).stdout.strip()
print("repo commit:", commit, flush=True)

units = sorted(glob.glob("/kaggle/input/**/reviewed_units.json", recursive=True))
assert units, "Attach the dataset bengali-bl-reviewed-units"
tifs = glob.glob("/kaggle/input/**/*.tif", recursive=True)
assert tifs, "Attach the dataset bengali-revision-pages"
pages = os.path.dirname(tifs[0])
STORED = sorted(glob.glob("/kaggle/input/**/ocr/reviewed_region.jsonl", recursive=True))
assert STORED, "Attach the output of bengali-rev-06-ocr-reviewed-units"
OUT = f"{W}/conf"
os.makedirs(OUT, exist_ok=True)
for f in glob.glob("/kaggle/input/**/conf/reviewed_conf.jsonl", recursive=True):  # resume
    if not os.path.exists(f"{OUT}/reviewed_conf.jsonl"):
        shutil.copy(f, OUT)
        print("resuming from", f)

sh("pip install -q pytesseract easyocr==1.7.2 jiwer rapidfuzz")
sh("mkdir -p /tmp/mm && curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C /tmp/mm bin/micromamba")
sh("/tmp/mm/bin/micromamba create -y -q -p /tmp/tess -c conda-forge tesseract=5.3.4")
TESS = "/tmp/tess/bin/tesseract"
TESSDATA = f"{REPO}/.tessdata"
os.makedirs(TESSDATA, exist_ok=True)
for lang in ("ben", "eng", "osd"):
    sh(f"curl -sL -o {TESSDATA}/{lang}.traineddata https://github.com/tesseract-ocr/tessdata_best/raw/main/{lang}.traineddata")
ben_sha = hashlib.sha256(open(f"{TESSDATA}/ben.traineddata", "rb").read()).hexdigest()
print("ben.traineddata sha256", ben_sha, "| paper's model:", ben_sha == "1cd0129288d1f74f6661c35e638e487bd146a66bb732ecd9e011e48dcb0df623")
assert os.path.exists(TESS), "conda-forge Tesseract 5.3.4 did not install; see setup.log (no fallback: the version must match)"
print(subprocess.run([TESS, "--version"], capture_output=True, text=True).stdout)

t0 = time.time()
log = f"{OUT}/reviewed_conf.log"
with open(log, "a", encoding="utf-8") as f:
    r = subprocess.run([sys.executable, "run_reviewed_units_conf.py", "--dataset-dir", pages, "--units", units[0],
                        "--stored", STORED[0], "--out", f"{OUT}/reviewed_conf"], cwd=f"{REPO}/ocr", stdout=f, stderr=subprocess.STDOUT,
                       env={**os.environ, "TESSDATA_PREFIX": TESSDATA, "TESSERACT_CMD": TESS, "PYTHONIOENCODING": "utf-8"})
print(f"exit {r.returncode} after {(time.time() - t0) / 60:.1f} min")
print("".join(open(log, encoding="utf-8").readlines()[-8:]))
p = f"{OUT}/reviewed_conf.jsonl"
n = sum(1 for l in open(p, encoding="utf-8") if l.strip()) if os.path.exists(p) else 0
print("pages written:", n, "of", len(json.load(open(units[0], encoding="utf-8"))["pages"]))
json.dump({"repo_commit": commit, "ben_traineddata_sha256": ben_sha,
           "tesseract": subprocess.run([TESS, "--version"], capture_output=True, text=True).stdout.splitlines()[:1],
           "exit": r.returncode}, open(f"{OUT}/run_info.json", "w"), indent=1)
shutil.copy(f"{W}/setup.log", f"{OUT}/setup.log")

# completeness and text-match counts only (no scores)
tm = em = nu = 0
if os.path.exists(p):
    for l in open(p, encoding="utf-8"):
        if l.strip():
            for u in json.loads(l)["units"]:
                nu += 1; tm += u["tesseract_matches"]; em += u["easyocr_matches"]
print(f"units {nu}; text equal to rev 06: tesseract {tm}, easyocr {em}")
shutil.make_archive(f"{W}/rev09_conf", "zip", W, "conf")
print("wrote", f"{W}/rev09_conf.zip")
