"""Run a command and keep a record of it in logs/runs/.

Each run gets its own folder, logs/runs/<run_id>/, with
  command.txt   the exact command
  stdout.txt    everything the command printed
  stderr.txt    everything it printed to stderr
  run.json      run id, label, start and end time, duration, exit code,
                git commit (and whether the tree had uncommitted changes),
                Python and package versions, machine
and one line is added to logs/runs.csv so all runs can be listed in one place.

Usage (from the repository root):
    python tools/run_logged.py --label leakage-check -- python eval/check_leakage.py --out-json logs/leakage/leakage_report.json

Everything after "--" is the command. The exit code of the command is passed on.
"""
import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from importlib import metadata
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PACKAGES = ["numpy", "scipy", "jiwer", "rapidfuzz", "regex", "matplotlib", "symspellpy",
            "torch", "transformers", "pytesseract", "easyocr"]


def git(*a):
    try:
        return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def versions():
    out = {}
    for p in PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="short name, e.g. leakage-check")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if not cmd:
        ap.error("no command given after --")

    start = datetime.now().astimezone()
    run_id = start.strftime("%Y%m%d-%H%M%S") + "_" + a.label
    d = REPO / "logs" / "runs" / run_id
    d.mkdir(parents=True, exist_ok=False)
    (d / "command.txt").write_text(subprocess.list2cmdline(cmd) + "\n", encoding="utf-8")

    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    t0 = time.perf_counter()
    with open(d / "stdout.txt", "wb") as so, open(d / "stderr.txt", "wb") as se:
        rc = subprocess.run(cmd, cwd=REPO, stdout=so, stderr=se, env=env).returncode
    secs = round(time.perf_counter() - t0, 2)
    end = datetime.now().astimezone()

    rec = {
        "run_id": run_id, "label": a.label, "command": subprocess.list2cmdline(cmd),
        "start": start.isoformat(timespec="seconds"), "end": end.isoformat(timespec="seconds"),
        "seconds": secs, "exit_code": rc,
        "git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
        "python": sys.version.split()[0], "packages": versions(),
        "machine": f"{platform.system()} {platform.release()} {platform.machine()}",
    }
    (d / "run.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")

    index = REPO / "logs" / "runs.csv"
    new = not index.exists()
    with open(index, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["run_id", "label", "start", "seconds", "exit_code", "git_commit", "git_dirty", "command"])
        w.writerow([run_id, a.label, rec["start"], secs, rc, rec["git_commit"], rec["git_dirty"], rec["command"]])
    print(f"{run_id}: exit {rc} in {secs}s -> logs/runs/{run_id}/")
    sys.exit(rc)


if __name__ == "__main__":
    main()
