"""Builds overleaf-upload.zip: exactly the files the paper needs, with forward
slashes in the paths. (PowerShell's Compress-Archive writes backslashes, which
Overleaf does not read as folders, so tables/ and figures/ go missing.)

Upload on Overleaf with New Project -> Upload Project. It builds with either
compiler: under pdfLaTeX the Bengali table comes from tables/examples_body.pdf.

Usage (from paper/):  python make_overleaf_zip.py
"""
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILES = ["main.tex", "references.bib"]
FOLDERS = ["tables", "figures", "fonts"]

with zipfile.ZipFile(HERE / "overleaf-upload.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for name in FILES:
        z.write(HERE / name, name)
    for folder in FOLDERS:
        for p in sorted((HERE / folder).iterdir()):
            if p.is_file():
                z.write(p, f"{folder}/{p.name}")
    print(f"Wrote {HERE / 'overleaf-upload.zip'} ({len(z.namelist())} files)")
