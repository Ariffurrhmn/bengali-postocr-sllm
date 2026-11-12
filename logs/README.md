# Logs

Records of commands run with `tools/run_logged.py`. Started on 13 November 2026. Runs before that date were not recorded this way; their original logs (Kaggle and Colab sessions) are in `results/` and are listed in `context/runs.md`.

- `runs.csv`: one line per run (run id, label, start time, seconds, exit code, git commit, whether the tree had uncommitted changes, command).
- `runs/<run_id>/`: `command.txt`, `stdout.txt`, `stderr.txt` and `run.json` (also Python and package versions and the machine).
- `leakage/leakage_report.json`: output of `eval/check_leakage.py`.
- `reruns/`: outputs of local reruns, compared with the saved result files by `tools/compare_outputs.py`. Not committed, because they contain page text.

A failed run stays in the log. For example, the first `paper-build` run failed because the wrapper runs every command from the repository root; the next one used `paper/main.tex` and worked.

To log a new run:

```
python tools/run_logged.py --label <short-name> -- <command> <args>
```
