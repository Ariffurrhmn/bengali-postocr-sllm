r"""Turn the finished BL review (bl_review JSON) into scoring units, one list per page.

A unit is what gets OCR'd and scored: a text block that is kept, or a box the author added
for a line that sat outside its block. Units come in the reviewed reading order (new_order).

Rules (D77, D78):
  - blocks flagged exclude or phantom are dropped (the checks below insist they hold no text);
  - a block keeps its ALTO lines except the lines an added box is linked to;
  - an added box holds exactly its linked lines;
  - a redrawn block uses new_box, else its original box;
  - reference text of a unit: words of a line joined by one space, lines joined by "\n";
  - no transcription and no ALTO file is changed.

Checks (the run stops if one fails):
  - every token of new_order appears once, and every block appears;
  - every linked line ID exists in that page's ALTO and is linked once;
  - every ALTO line with text, in a block that is kept, ends up in exactly one unit;
  - dropped blocks hold no text.

Usage:
    python tools/build_bl_reviewed_units.py --review results/bl_review/review_final_2026-11-22.json \
        --alto-dir "<dir with the .xml files>" --out results/bl_review/reviewed_units.json
"""
import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def local(tag):
    return tag.rsplit("}", 1)[-1]


def read_alto(path: Path):
    root = ET.parse(path).getroot()
    page = next(e for e in root.iter() if local(e.tag) == "Page")
    size = [float(page.get("WIDTH")), float(page.get("HEIGHT"))]
    blocks = []
    for blk in (e for e in root.iter() if local(e.tag) == "TextBlock"):
        lines = []
        for ln in (e for e in blk if local(e.tag) == "TextLine"):
            words, boxes = [], []
            for s in (e for e in ln if local(e.tag) == "String"):
                w = s.get("CONTENT") or ""
                if w:
                    words.append(w)
                    try:
                        x, y, ww, hh = (float(s.get(k)) for k in ("HPOS", "VPOS", "WIDTH", "HEIGHT"))
                        boxes.append([x, y, x + ww, y + hh])
                    except (TypeError, ValueError):
                        pass
            lines.append({"id": ln.get("ID"), "text": " ".join(words).strip(), "word_boxes": boxes})
        blocks.append({"id": blk.get("ID"), "lines": lines})
    return size, blocks


def union(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def xyxy(box):
    x, y, w, h = box
    return [float(x), float(y), float(x) + float(w), float(y) + float(h)]


def build_page(name, rp, alto_path):
    size, blocks = read_alto(alto_path)
    assert [int(v) for v in rp["alto_page_size"]] == [int(v) for v in size], f"{name}: page size differs"
    keys = [f"b{i}" for i in range(len(blocks))]
    assert [b["id"] for b in blocks] == rp["original_block_ids"], f"{name}: block IDs differ from the review"
    order = rp["new_order"]
    assert len(order) == len(set(order)), f"{name}: a token appears twice in new_order"
    assert set(k for k in order if k[0] == "b") == set(keys), f"{name}: new_order must hold every block"
    assert set(k for k in order if k[0] == "a") == set(rp["added_boxes"]), f"{name}: added boxes missing from new_order"

    line_home = {ln["id"]: i for i, b in enumerate(blocks) for ln in b["lines"]}
    linked, line_by_id = {}, {ln["id"]: ln for b in blocks for ln in b["lines"]}
    for ak, a in rp["added_boxes"].items():
        assert a["linked_lines"], f"{name}: added box {ak} has no linked line"
        for lid in a["linked_lines"]:
            assert lid in line_by_id, f"{name}: {ak} links unknown line {lid}"
            assert lid not in linked, f"{name}: line {lid} linked twice"
            linked[lid] = ak

    units, used, dropped = [], set(), []
    for tok in order:
        if tok[0] == "a":
            a = rp["added_boxes"][tok]
            lines = [line_by_id[l] for l in a["linked_lines"]]
            box = xyxy(a["box"])
            uid, kind = f"{name}:{tok}", "added"
        else:
            i = int(tok[1:])
            bl, fl = rp["blocks"][tok], rp["blocks"][tok]["flags"]
            lines = [ln for ln in blocks[i]["lines"] if ln["id"] not in linked]
            if fl.get("exclude") or fl.get("phantom"):
                assert not any(ln["text"] for ln in blocks[i]["lines"]), f"{name}: dropped block {tok} holds text"
                dropped.append(tok)
                continue
            box = xyxy(bl["new_box"] or bl["original_box"])
            uid, kind = blocks[i]["id"], "block"
        lines = [ln for ln in lines if ln["text"]]
        if not lines:
            dropped.append(tok)
            continue
        for ln in lines:
            assert ln["id"] not in used, f"{name}: line {ln['id']} used twice"
            used.add(ln["id"])
        wb = [b for ln in lines for b in ln["word_boxes"]]
        units.append({"id": uid, "key": tok, "kind": kind, "ground_truth": "\n".join(ln["text"] for ln in lines),
                      "line_ids": [ln["id"] for ln in lines], "box": [round(v, 1) for v in box],
                      "crop_union": [round(v, 1) for v in union([box] + wb)]})
    # every ALTO line with text, in a kept block, is in exactly one unit
    dropped_blocks = {int(t[1:]) for t in dropped if t[0] == "b"}
    for i, b in enumerate(blocks):
        for ln in b["lines"]:
            if ln["text"] and i not in dropped_blocks:
                assert ln["id"] in used, f"{name}: line {ln['id']} of block b{i} is in no unit"
    return {"page_id": name, "source": "BL", "page_type": rp.get("page_type", ""), "status": rp["status"],
            "alto_page_size": size, "units": units, "dropped": dropped,
            "counts": {"blocks": len(blocks), "units": len(units), "added": len(rp["added_boxes"]),
                       "dropped": len(dropped), "alto_lines_with_text": sum(1 for b in blocks for l in b["lines"] if l["text"])}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--review", type=Path, required=True)
    ap.add_argument("--alto-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    review = json.loads(args.review.read_text(encoding="utf-8"))
    pages, problems = [], []
    for name, rp in sorted(review["pages"].items()):
        if rp["status"] != "done" or rp["drop_page"] == "yes":
            print(f"skip {name}: status {rp['status']}, drop_page {rp['drop_page']}")
            continue
        try:
            pages.append(build_page(name, rp, args.alto_dir / f"{name}.xml"))
        except AssertionError as e:
            problems.append(str(e))
    if problems:
        print("\n".join(problems))
        sys.exit(1)
    args.out.write_text(json.dumps({"source_review": args.review.name, "pages": pages}, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    n = sum(p["counts"]["units"] for p in pages)
    print(f"{len(pages)} pages, {n} units, {sum(p['counts']['added'] for p in pages)} added boxes, "
          f"{sum(p['counts']['dropped'] for p in pages)} dropped blocks; checks passed")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
