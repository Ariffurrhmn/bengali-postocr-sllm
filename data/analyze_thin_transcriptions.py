"""
For each of the 52 usable pages (transcribed text present), compute the
fraction of TextRegion elements that actually have non-empty TextEquiv/Unicode
content. A page where most regions are empty despite having *some* text is a
"thin transcription" — weaker ground-truth signal than a page whose regions
are consistently transcribed, even if its total character count is lower.
"""
import re
from pathlib import Path

DATASET_DIR = Path(r"D:\Competition_dataset_ImagesPAGEXML")

REGION_RE = re.compile(r"<TextRegion\b.*?</TextRegion>", re.DOTALL)
UNICODE_RE = re.compile(r"<Unicode>([^<]*)</Unicode>")


def analyze(xml_path: Path):
    text = xml_path.read_text(encoding="utf-8")
    regions = REGION_RE.findall(text)
    if not regions:
        return None  # no TextRegion elements at all

    total_regions = len(regions)
    nonempty_regions = 0
    total_chars = 0
    for region in regions:
        chars = sum(len(m.strip()) for m in UNICODE_RE.findall(region))
        if chars > 0:
            nonempty_regions += 1
        total_chars += chars

    if total_chars == 0:
        return None  # fully empty page, already excluded elsewhere

    coverage = nonempty_regions / total_regions
    return {
        "total_regions": total_regions,
        "nonempty_regions": nonempty_regions,
        "coverage": coverage,
        "total_chars": total_chars,
    }


def main():
    all_ids = sorted(p.stem for p in DATASET_DIR.glob("*.tif"))

    results = []
    for page_id in all_ids:
        xml_path = DATASET_DIR / f"{page_id}.xml"
        stats = analyze(xml_path)
        if stats is None:
            continue
        results.append((page_id, stats))

    results.sort(key=lambda r: (r[1]["coverage"], r[1]["total_chars"]))

    print(f"Analyzed {len(results)} pages with non-zero transcribed text (of 81 total)")
    print()
    print("Thinnest transcriptions (lowest region-coverage fraction):")
    for page_id, s in results[:15]:
        print(
            f"  {page_id}: {s['nonempty_regions']}/{s['total_regions']} regions "
            f"({s['coverage']:.0%} coverage), {s['total_chars']} chars"
        )


if __name__ == "__main__":
    main()
