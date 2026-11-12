"""Compare two result files line by line on their output text.

Rows are matched on (page_id, engine, model_key, approach) and the raw_output
fields are compared exactly. Timing fields are ignored because they change on
every run. Used to check that a rerun reproduces a saved result file.

Usage:
    python tools/compare_outputs.py results/dict_eval40.jsonl logs/reruns/dict_eval40.jsonl
"""
import json
import sys
from pathlib import Path


def load(path):
    rows = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            rows[(r["page_id"], r["engine"], r["model_key"], r.get("approach"))] = r["raw_output"]
    return rows


def main():
    saved, rerun = load(sys.argv[1]), load(sys.argv[2])
    same = sum(1 for k in saved if k in rerun and saved[k] == rerun[k])
    differ = [k for k in saved if k in rerun and saved[k] != rerun[k]]
    print(f"saved: {len(saved)} rows, rerun: {len(rerun)} rows")
    print(f"identical output: {same}, different: {len(differ)}, "
          f"only in saved: {len(set(saved) - set(rerun))}, only in rerun: {len(set(rerun) - set(saved))}")
    for k in differ[:5]:
        print("  differs:", k)
    sys.exit(0 if same == len(saved) == len(rerun) else 1)


if __name__ == "__main__":
    main()
