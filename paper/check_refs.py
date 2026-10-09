"""Sanity check for the LaTeX source: every \\cite key exists in references.bib,
every \\ref has a \\label, and no bib entry is left uncited.
Run from the paper/ folder:  python check_refs.py
"""
import re
from pathlib import Path

here = Path(__file__).parent
files = [here / "main.tex"] + sorted((here / "tables").glob("*.tex"))
tex = "\n".join(f.read_text(encoding="utf-8") for f in files)
bib = (here / "references.bib").read_text(encoding="utf-8")

cited = set()
for m in re.finditer(r"\\cite\{([^}]*)\}", tex):
    cited.update(k.strip() for k in m.group(1).split(","))
keys = set(re.findall(r"@\w+\{([^,]+),", bib))
labels = set(re.findall(r"\\label\{([^}]*)\}", tex))
refs = set(re.findall(r"\\ref\{([^}]*)\}", tex))

print("cited keys:", len(cited))
print("cited but missing from references.bib:", sorted(cited - keys) or "none")
print("in references.bib but never cited:", sorted(keys - cited) or "none")
print("\\ref without a \\label:", sorted(refs - labels) or "none")
