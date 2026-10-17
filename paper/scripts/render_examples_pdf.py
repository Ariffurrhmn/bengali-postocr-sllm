"""Renders tables/examples_body.tex to tables/examples_body.pdf with XeLaTeX
(Tectonic), so the Bengali example table also appears when the paper is
compiled with pdfLaTeX, Overleaf's default compiler.

Run after paper/scripts/make_example_table.py, from any directory:
    python paper/scripts/render_examples_pdf.py
"""
import shutil
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]  # the paper/ folder
WIDTH = "516pt"  # IEEEtran conference \textwidth, so the table is included at scale 1

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    shutil.copytree(HERE / "tables", tmp / "tables")
    shutil.copytree(HERE / "fonts", tmp / "fonts")
    (tmp / "examples.tex").write_text(
        "\\documentclass[10pt]{standalone}\n"
        "\\input{tables/preamble}\n"
        "\\begin{document}\n"
        f"\\begin{{minipage}}{{{WIDTH}}}\\centering\n"
        "\\input{tables/examples_body}\n"
        "\\end{minipage}\n\\end{document}\n", encoding="utf-8")
    subprocess.run(["tectonic", "-X", "compile", "examples.tex"], cwd=tmp, check=True)
    shutil.copy(tmp / "examples.pdf", HERE / "tables" / "examples_body.pdf")
print(f"Wrote {HERE / 'tables' / 'examples_body.pdf'}")
