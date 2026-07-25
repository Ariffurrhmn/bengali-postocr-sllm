"""Audits every page's PAGE-XML for presence of actual transcribed text."""
from pathlib import Path
import re

DATASET_DIR = Path(r"D:\Competition_dataset_ImagesPAGEXML")
UNICODE_RE = re.compile(r"<Unicode>([^<]*)</Unicode>")


def main():
    xmls = sorted(DATASET_DIR.glob("*.xml"))
    empty_pages = []
    nonempty_pages = []
    for xml_path in xmls:
        text = xml_path.read_text(encoding="utf-8")
        matches = UNICODE_RE.findall(text)
        total_chars = sum(len(m.strip()) for m in matches)
        if total_chars == 0:
            empty_pages.append(xml_path.stem)
        else:
            nonempty_pages.append((xml_path.stem, len(matches), total_chars))

    print(f"Total pages: {len(xmls)}")
    print(f"Pages with zero transcribed text: {len(empty_pages)}")
    print(f"Pages with transcribed text: {len(nonempty_pages)}")
    print()
    print("Empty pages:")
    for p in empty_pages:
        print(f"  {p}")


if __name__ == "__main__":
    main()
