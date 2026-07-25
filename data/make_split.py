"""
Generates the frozen dev/eval page split for the British Library Bengali corpus.

Of the 81 downloaded pages, 29 have PAGE-XML with region boxes but zero
transcribed ground-truth text (<Unicode> tags empty or absent) — these cannot
be used to compute CER/WER and are excluded. That leaves 52 usable pages; this
script selects 50 of them, dropping the 2 with the "thinnest" transcriptions —
lowest fraction of TextRegion elements that actually have non-empty text
(a page where most regions are empty despite having *some* text is a weaker
ground-truth signal than a uniformly-transcribed page, even a short one) —
and splits into 10 dev / 40 eval.

Run once against the local dataset directory to (re)produce split_dev.txt and
split_eval.txt. Deterministic via a fixed seed — do not change SEED or the
split sizes after the eval split has been used for any reported result.
"""
import argparse
import random
import re
from pathlib import Path

SEED = 403
DEV_SIZE = 10
POOL_SIZE = 50

REGION_RE = re.compile(r"<TextRegion\b.*?</TextRegion>", re.DOTALL)
UNICODE_RE = re.compile(r"<Unicode>([^<]*)</Unicode>")


def transcribed_char_count(xml_path: Path) -> int:
    text = xml_path.read_text(encoding="utf-8")
    return sum(len(m.strip()) for m in UNICODE_RE.findall(text))


def region_coverage(xml_path: Path) -> float:
    """Fraction of TextRegion elements that have non-empty transcribed text.
    Returns 1.0 for pages with no TextRegion elements (nothing to be thin)."""
    text = xml_path.read_text(encoding="utf-8")
    regions = REGION_RE.findall(text)
    if not regions:
        return 1.0
    nonempty = sum(
        1 for r in regions if sum(len(m.strip()) for m in UNICODE_RE.findall(r)) > 0
    )
    return nonempty / len(regions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        default=r"D:\Competition_dataset_ImagesPAGEXML",
        help="Directory containing paired .tif/.xml page files",
    )
    parser.add_argument(
        "--out-dir",
        default=Path(__file__).parent,
        type=Path,
        help="Where to write split_dev.txt / split_eval.txt",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    tifs = {p.stem for p in dataset_dir.glob("*.tif")}
    xmls = {p.stem for p in dataset_dir.glob("*.xml")}

    # Match case-insensitively: the source dataset has at least one page
    # (279_2_a_15_0005) where the .tif and .xml stems differ only in case
    # (279_2_a_15_0005.tif vs 279_2_A_15_0005.xml). Same page, inconsistent
    # casing upstream — not a missing/orphaned file.
    tifs_lower = {s.lower(): s for s in tifs}
    xmls_lower = {s.lower(): s for s in xmls}

    tif_only = tifs_lower.keys() - xmls_lower.keys()
    xml_only = xmls_lower.keys() - tifs_lower.keys()
    if tif_only or xml_only:
        raise SystemExit(
            f"Unpaired files found — tif without xml: {sorted(tifs_lower[k] for k in tif_only)}, "
            f"xml without tif: {sorted(xmls_lower[k] for k in xml_only)}"
        )

    all_page_ids = sorted(tifs_lower[k] for k in tifs_lower)

    char_counts = {
        page_id: transcribed_char_count(dataset_dir / f"{page_id}.xml")
        for page_id in all_page_ids
    }
    usable = [p for p in all_page_ids if char_counts[p] > 0]
    empty = [p for p in all_page_ids if char_counts[p] == 0]

    if len(usable) < POOL_SIZE:
        raise SystemExit(
            f"Only {len(usable)} pages have transcribed text; need {POOL_SIZE}."
        )

    coverage = {p: region_coverage(dataset_dir / f"{p}.xml") for p in usable}

    # Deterministic pool selection: keep the POOL_SIZE pages with the highest
    # region-coverage (ties broken by more total chars, then page_id),
    # dropping the pages with the "thinnest" transcriptions — most TextRegion
    # elements left empty despite the page having some transcribed text.
    usable_sorted = sorted(
        usable, key=lambda p: (-coverage[p], -char_counts[p], p)
    )
    pool = sorted(usable_sorted[:POOL_SIZE])
    dropped_from_usable = sorted(usable_sorted[POOL_SIZE:])

    rng = random.Random(SEED)
    shuffled = pool[:]
    rng.shuffle(shuffled)

    dev = sorted(shuffled[:DEV_SIZE])
    eval_ = sorted(shuffled[DEV_SIZE:])

    out_dir = Path(args.out_dir)
    (out_dir / "split_dev.txt").write_text("\n".join(dev) + "\n", encoding="utf-8")
    (out_dir / "split_eval.txt").write_text("\n".join(eval_) + "\n", encoding="utf-8")
    (out_dir / "excluded_no_ground_truth.txt").write_text(
        "\n".join(empty) + "\n", encoding="utf-8"
    )

    print(f"Total pages downloaded: {len(all_page_ids)}")
    print(f"Pages with zero transcribed text (excluded): {len(empty)}")
    print(f"Usable pages (transcribed text present): {len(usable)}")
    if dropped_from_usable:
        details = ", ".join(
            f"{p} ({coverage[p]:.0%} region coverage)" for p in dropped_from_usable
        )
        print(
            f"Usable pages dropped to reach pool of {POOL_SIZE} "
            f"(thinnest transcriptions): {details}"
        )
    print(f"Dev: {len(dev)} -> split_dev.txt")
    print(f"Eval: {len(eval_)} -> split_eval.txt")


if __name__ == "__main__":
    main()
