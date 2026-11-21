r"""Local review app for the BL test pages (reading order, block flags, box fixes).

Builds one self-contained HTML file (index.html) plus downscaled page images
(img/), and a short README.txt. All page data (block IDs, boxes, line text and
line boxes, block text) is embedded as JSON in the HTML, so the app opens from
file:// with no server. Nothing is scored and no transcription is ever changed;
the ALTO files are only read.

Usage:
    python tools/bl_review_app.py --group2 "<Group 2 path>" --out results/bl_review [--limit 3]

Export / import JSON format (schema_version 1)
----------------------------------------------
{
  "schema_version": 1,
  "kind": "bl_review",
  "reviewer": "<name>",
  "exported_at": "<ISO time>",
  "reading_order_rule": "<rule text shown in the app when exported>",
  "pages": {
    "<page name as in test_bl.txt>": {
      "alto_page_size": [W, H],            # coordinate system of every box below (original ALTO pixels)
      "original_block_ids": ["id", ...],   # block order in the ALTO file
      "original_order": ["b0", "b1", ...], # same order as keys (b<i> = i-th TextBlock, 0-based)
      "new_order": ["b1", "a1", "b0"...],  # reviewed reading order; b<i> original block, a<n> added box.
                                           # Every original block exactly once; added boxes appear too.
      "new_order_block_ids": ["id", "added:a1", ...],
      "status": "not_started" | "in_progress" | "done" | "needs_discussion",
      "page_type": "" | "body" | "toc" | "title" | "mixed",
      "drop_page": "undecided" | "yes" | "no",
      "drop_reason": "text",
      "note": "text",
      "blocks": {                          # per original block, keyed by b<i>
        "b0": {"id": "<ALTO block ID>",
               "flags": {"exclude": bool,    # non-text: drawing, stamp, shelfmark
                         "phantom": bool,    # empty / phantom box
                         "box_wrong": bool,  # too small / large / misplaced
                         "text_not_in_box": bool},  # text does not belong to this box
               "note": "text",
               "original_box": [x, y, w, h],
               "new_box": [x, y, w, h] | null,   # redrawn box, original ALTO pixels
               "linked_lines": ["line ID", ...]}  # ALTO lines the author assigned to this box
      },
      "added_boxes": {                     # boxes for visible text that has no box, keyed a<n>
        "a1": {"box": [x, y, w, h], "note": "what text it covers", "linked_lines": ["line ID", ...]}
      },
      "started_at": "<ISO>", "updated_at": "<ISO>"
    }
  }
}
Boxes are [x, y, w, h] in the ALTO page's own pixel units (HPOS, VPOS, WIDTH, HEIGHT).
Page images are shown stretched to the ALTO page size, which is the same rescaling
tools/reading_order_sheets.py applies when the image size differs from the ALTO size.
Line IDs are the ALTO TextLine IDs (or line<i> when a line has no ID).
Import reads: reviewer, and per page the fields status, page_type, drop_page, drop_reason,
note, new_order, blocks[*].flags/note/new_box/linked_lines, added_boxes, started_at, updated_at.
"""
import argparse
import datetime
import json
import os
import sys
import xml.etree.ElementTree as ET

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reading_order_sheets import (ALTO_DIR, DATA_REL, MAX_SIDE, NS, PAGE_LIST,  # noqa: E402
                                  parse_page)

# The reading-order rule shown in the help panel. Change it here only.
RULE_TEXT = ("Reading order = physical order on the page: top to bottom, and within one printed line "
             "left to right. So a header line is read in the order it is printed, whether the page "
             "number or the title comes first. Columns are read left to right.")

SCHEMA_VERSION = 1


def num(el, name):
    v = el.get(name)
    return float(v) if v not in (None, "") else None


def parse_lines(xml_path):
    """Every TextLine on the page: id, parent block index, text, box (or None)."""
    root = ET.parse(xml_path).getroot()
    lines, seen = [], set()
    for bi, tb in enumerate(root.iter(f"{NS}TextBlock")):
        for tl in tb.iter(f"{NS}TextLine"):
            lid = tl.get("ID") or f"line{len(lines)}"
            while lid in seen:
                lid += "_"
            seen.add(lid)
            vals = [num(tl, k) for k in ("HPOS", "VPOS", "WIDTH", "HEIGHT")]
            lines.append({
                "id": lid, "block": bi,
                "text": " ".join(s.get("CONTENT", "") for s in tl.iter(f"{NS}String")).strip(),
                "box": vals if None not in vals else None,
            })
    return lines


HTML = r"""<!doctype html>
<html lang="bn"><head><meta charset="utf-8">
<title>BL test pages: review</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
*{box-sizing:border-box}
body{margin:0;font:14px "Segoe UI","Nirmala UI",sans-serif;color:#222;background:#f4f4f4;height:100vh;display:flex;flex-direction:column}
.bn{font-family:"Nirmala UI","Noto Sans Bengali","Vrinda",sans-serif}
header{background:#263238;color:#fff;padding:6px 10px;display:flex;gap:8px;align-items:center;flex-wrap:wrap}
header button,header input{font:inherit}
header .sp{flex:1}
button{cursor:pointer;border:1px solid #888;background:#fff;border-radius:3px;padding:2px 8px}
button:hover{background:#e8eef2}
button.on{background:#ffd54f;border-color:#a77}
#warn{background:#ffe0b2;color:#a00;padding:3px 10px;display:none}
#main{flex:1;display:flex;min-height:0}
#plist{width:210px;overflow:auto;background:#fff;border-right:1px solid #bbb;font-size:12px}
#plist div{padding:3px 6px;cursor:pointer;border-bottom:1px solid #eee;display:flex;gap:5px;align-items:center}
#plist div.cur{background:#cfe8ff}
.dot{width:10px;height:10px;border-radius:50%;flex:none;border:1px solid #666}
.s_not_started{background:#fff}.s_in_progress{background:#ffca28}.s_done{background:#43a047}.s_needs_discussion{background:#e53935}
#imgcol{flex:1.3;overflow:auto;padding:8px;min-width:200px}
#wrap{position:relative;line-height:0;background:#ccc}
#wrap img{width:100%;display:block}
#wrap svg{position:absolute;left:0;top:0;width:100%;height:100%}
#wrap svg.drawing{cursor:crosshair}
.bx{fill-opacity:.08;stroke-width:3;vector-effect:non-scaling-stroke;cursor:pointer}
.bx.hot,.bx.sel{fill-opacity:.35;stroke-width:5}
.bx.dim{stroke-dasharray:2 4}
.orig{fill:none;stroke-dasharray:8 5;stroke-width:2;vector-effect:non-scaling-stroke;pointer-events:none}
.lnbox{fill:rgba(255,235,0,.35);stroke:#c90;stroke-width:2;vector-effect:non-scaling-stroke;pointer-events:none}
.lbl{font-weight:bold;fill:#fff;pointer-events:none;font-family:Arial,sans-serif}
#side{flex:1;overflow:auto;padding:8px;min-width:340px;border-left:1px solid #bbb;background:#fafafa}
details.help{background:#fffde7;border:1px solid #d9cc6b;padding:4px 8px;margin-bottom:8px}
.pg{background:#fff;border:1px solid #bbb;padding:6px 8px;margin-bottom:8px;display:grid;grid-template-columns:auto 1fr;gap:4px 8px;align-items:center}
.pg textarea,.pg input[type=text]{width:100%;font:inherit}
.row{background:#fff;border:1px solid #bbb;border-left:6px solid #888;margin-bottom:5px;padding:4px 6px}
.row.hot{background:#fff3c4}.row.sel{outline:2px solid #1976d2}
.row.excl{opacity:.75}
.row.dragover{border-top:4px solid #1976d2}
.rh{display:flex;gap:6px;align-items:center;flex-wrap:wrap}
.handle{cursor:grab;font-size:18px;color:#666;user-select:none}
.badge{background:#eee;border-radius:9px;padding:0 7px;font-size:12px}
.txt{max-height:8.5em;overflow:auto;white-space:pre-wrap;word-break:break-word;font-size:16px;background:#f7f7f7;padding:3px 5px;margin:3px 0;line-height:1.35}
.fl label{margin-right:8px;white-space:nowrap;font-size:12px}
.row input.note{width:100%;font:inherit;margin-top:3px}
.lines{max-height:220px;overflow:auto;border:1px solid #ccc;margin-top:3px;font-size:13px}
.lines label{display:block;padding:1px 4px;cursor:pointer}.lines label:hover{background:#fff3c4}
.lines .lt{font-size:15px}
.mode{background:#ffd54f;padding:3px 8px;margin-bottom:6px;display:none}
.tag{font-size:11px;color:#a00}
</style></head><body>
<header>
 <button id="prev">&larr; Prev</button><b id="ptitle"></b><button id="next">Next &rarr;</button>
 <span id="prog"></span><span class="sp"></span>
 Reviewer: <input id="reviewer" size="12">
 <button id="exp">Export JSON</button><button id="imp">Import JSON</button>
 <input type="file" id="impf" accept=".json,application/json" style="display:none">
 <span id="dirty"></span>
</header>
<div id="warn">Browser storage is not available here: nothing is autosaved. Use Export JSON often.</div>
<div id="main">
 <div id="plist"></div>
 <div id="imgcol">
  <div style="margin-bottom:4px;font-size:12px">Image size <input type="range" id="zoom" min="40" max="250" value="100">
   <label><input type="checkbox" id="lblmode"> number boxes by new order</label>
   &nbsp;Show: <select id="vis"><option value="all">all boxes</option><option value="choose">only the ones I tick</option></select>
   <button id="hideall" title="untick every box">Hide all</button><button id="showall" title="tick every box">Tick all</button>
   &nbsp;Label size <input type="range" id="lsize" min="3" max="40" value="14" style="width:70px">
   &nbsp;<span id="pageinfo"></span></div>
  <div id="wrap"><img id="img" alt=""><svg id="svg" preserveAspectRatio="none"></svg></div>
 </div>
 <div id="side">
  <details class="help" open><summary><b>Help</b></summary>
   <p><b>Reading-order rule:</b> <span id="rule"></span></p>
   <p>Boxes on the image are numbered in the order of the XML file (tick the box above to number by your new order). Hover or click a box or a row to highlight both.
   Drag rows by the &#8942;&#8942; handle, or use the arrows, to set reading order. Dashed outline = original box; solid = your redrawn box.
   <i>Redraw</i> then drag on the image. <i>Add box</i> then drag on the image for visible text that has no box. Esc cancels drawing.
   Text is never edited. Everything autosaves in this browser; use Export JSON to keep a copy.</p>
  </details>
  <div class="mode" id="mode"></div>
  <div class="pg" id="pg"></div>
  <div style="margin-bottom:6px"><button id="addbox">+ Add box for text without a box</button></div>
  <div id="blocks"></div>
 </div>
</div>
<script id="data" type="application/json">__DATA__</script>
<script>
"use strict";
const D = JSON.parse(document.getElementById("data").textContent);
const SCHEMA = D.schema_version, STORE = "bl_review_v1";
const COLORS = ["#dc143c","#005ac8","#008c3c","#e67800","#8c00aa","#0096a0"];
const FLAGS = [["exclude","Exclude (non-text: drawing, stamp, shelfmark)"],["phantom","Phantom / empty box"],
  ["box_wrong","Box wrong (size/place)"],["text_not_in_box","Text not belonging to this box"]];
const STATUS = [["not_started","Not started"],["in_progress","In progress"],["done","Done"],["needs_discussion","Needs discussion"]];
const PTYPES = [["","(not set)"],["body","Body text"],["toc","Table of contents"],["title","Title page"],["mixed","Mixed / other"]];
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const now = () => new Date().toISOString();

let S = {schema_version: SCHEMA, reviewer: "", pages: {}};
let shown = new Set(), cur = 0, hot = null, sel = null, mode = null, hotLine = null, dirty = false, storageOK = true;

function defPage(p) {
  return {status:"not_started", page_type:"", drop_page:"undecided", drop_reason:"", note:"",
    order: p.blocks.map((_, i) => "b" + i), blocks:{}, added:{}, next_added:1, started_at:null, updated_at:null};
}
function ps(i) { const p = D.pages[i]; if (!S.pages[p.name]) S.pages[p.name] = defPage(p); return S.pages[p.name]; }
function bs(st, k) { return st.blocks[k] || (st.blocks[k] = {flags:{}, note:"", new_box:null, linked_lines:[]}); }
function ent(k) { return k[0] === "a" ? "a" : "b"; }

function normalize(p, st) { // repair: every block exactly once, drop unknown keys
  const valid = new Set(p.blocks.map((_, i) => "b" + i)); Object.keys(st.added || {}).forEach(k => valid.add(k));
  const seen = new Set(), o = [];
  (st.order || []).forEach(k => { if (valid.has(k) && !seen.has(k)) { seen.add(k); o.push(k); } });
  valid.forEach(k => { if (!seen.has(k)) o.push(k); });
  st.order = o; st.blocks = st.blocks || {}; st.added = st.added || {};
  let n = st.next_added || 1; Object.keys(st.added).forEach(k => { n = Math.max(n, +k.slice(1) + 1); }); st.next_added = n;
  return st;
}

function save() {
  try { localStorage.setItem(STORE, JSON.stringify(S)); } catch (e) { storageOK = false; $("warn").style.display = "block"; }
  dirty = true; $("dirty").textContent = "changes not exported yet";
}
function load() {
  try { const r = localStorage.getItem(STORE); if (r) { const o = JSON.parse(r); if (o && o.pages) S = o; } }
  catch (e) { storageOK = false; $("warn").style.display = "block"; }
  D.pages.forEach((p, i) => { ps(i); normalize(p, S.pages[p.name]); });
  $("reviewer").value = S.reviewer || "";
}
function touch(i) {
  const st = ps(i), t = now();
  if (st.status === "not_started") st.status = "in_progress";
  if (!st.started_at) st.started_at = t; st.updated_at = t; save(); renderList(); renderPageFields(true);
}

// ---------- geometry helpers
function pageBox(p, st, k) { // box currently in force for a key
  if (k[0] === "a") return st.added[k].box;
  const b = st.blocks[k]; return (b && b.new_box) || p.blocks[+k.slice(1)].box;
}
function xmlNo(k, st) { return k[0] === "b" ? (+k.slice(1) + 1) : "+" + k.slice(1); }
function colorOf(k) { return k[0] === "a" ? "#00796b" : COLORS[(+k.slice(1) + 1) % COLORS.length]; }
function labelOf(k, st) { return $("lblmode").checked ? (st.order.indexOf(k) + 1) : xmlNo(k, st); }
function keyText(p, st, k) { return k[0] === "a" ? "" : p.blocks[+k.slice(1)].text; }

// ---------- rendering
function renderList() {
  $("plist").innerHTML = D.pages.map((p, i) => {
    const st = S.pages[p.name];
    return `<div data-i="${i}" class="${i === cur ? "cur" : ""}"><span class="dot s_${st.status}"></span>${i + 1}. ${esc(p.name)}${st.drop_page === "yes" ? ' <span class="tag">drop</span>' : ""}</div>`;
  }).join("");
  const c = {}; D.pages.forEach(p => { const s = S.pages[p.name].status; c[s] = (c[s] || 0) + 1; });
  $("prog").textContent = STATUS.map(([k, l]) => `${l}: ${c[k] || 0}`).join(" | ");
}
function renderPageFields(keepFocus) {
  const st = ps(cur);
  if (keepFocus) { const s = $("pg_status"); if (s) s.value = st.status; return; }
  $("pg").innerHTML = `
   <label>Status</label><select id="pg_status">${STATUS.map(([k, l]) => `<option value="${k}">${l}</option>`).join("")}</select>
   <label>Page type</label><select id="pg_type">${PTYPES.map(([k, l]) => `<option value="${k}">${l}</option>`).join("")}</select>
   <label>Drop page from test set?</label><span>${["undecided","yes","no"].map(v => `<label><input type="radio" name="drop" value="${v}"> ${v}</label>`).join(" ")}</span>
   <label>Reason</label><input type="text" id="pg_reason" class="bn">
   <label>Page note</label><textarea id="pg_note" rows="3" class="bn" placeholder="Thoughts to discuss"></textarea>`;
  $("pg_status").value = st.status; $("pg_type").value = st.page_type; $("pg_reason").value = st.drop_reason; $("pg_note").value = st.note;
  document.querySelectorAll("input[name=drop]").forEach(r => { r.checked = r.value === st.drop_page; });
}
function renderBlocks() {
  const p = D.pages[cur], st = ps(cur);
  $("blocks").innerHTML = st.order.map((k, pos) => {
    const b = k[0] === "b" ? p.blocks[+k.slice(1)] : null, bsx = k[0] === "b" ? (st.blocks[k] || {flags:{}}) : null;
    const fl = bsx ? bsx.flags || {} : {}, a = k[0] === "a" ? st.added[k] : null;
    const txt = b ? (b.text ? esc(b.text) : "<i>(empty)</i>") : "";
    const lines = (bsx ? bsx.linked_lines : a.linked_lines) || [];
    return `<div class="row ${fl.exclude ? "excl" : ""}" data-k="${k}" draggable="false" style="border-left-color:${colorOf(k)}">
     <div class="rh"><span class="handle" title="drag to reorder">&#8942;&#8942;</span><b>${pos + 1}.</b>
      <span class="badge" style="background:${colorOf(k)};color:#fff">${k[0] === "a" ? "added " + k.slice(1) : "XML " + (+k.slice(1) + 1)}</span>
      <code>${b ? esc(b.id) : ""}</code><label class="fl" style="margin:0"><input type="checkbox" data-show="1" ${shown.has(k) ? "checked" : ""}> show</label><span class="sp" style="flex:1"></span>
      <button data-act="up">&uarr;</button><button data-act="down">&darr;</button></div>
     ${b ? `<div class="txt bn">${txt}</div>` : ""}
     ${b ? `<div class="fl">${FLAGS.map(([f, l]) => `<label><input type="checkbox" data-flag="${f}" ${fl[f] ? "checked" : ""}> ${l}</label>`).join("")}</div>` : ""}
     <input class="note bn" data-field="note" placeholder="${a ? "Which text does this box cover?" : "Note"}" value="${esc((bsx ? bsx.note : a.note) || "")}">
     <div style="margin-top:3px"><button data-act="redraw">${a ? "Redraw this box" : "Redraw box"}</button>
      ${b && bsx.new_box ? '<button data-act="resetbox">Back to original box</button>' : ""}
      <button data-act="lines">Link ALTO lines (${lines.length})</button>
      ${a ? '<button data-act="del">Delete added box</button>' : ""}</div>
     <div class="lines" style="display:none"></div></div>`;
  }).join("");
  renderOverlay();
}
function renderOverlay() {
  const p = D.pages[cur], st = ps(cur), svg = $("svg");
  const fs = p.ph / 1000 * $("lsize").value, parts = [], choose = $("vis").value === "choose";
  st.order.forEach(k => {
    if (choose && !shown.has(k) && k !== hot && !(mode && mode.key === k)) return;
    const col = colorOf(k), b = k[0] === "b" ? st.blocks[k] : null, fl = (b && b.flags) || {};
    const dim = fl.exclude || fl.phantom ? " dim" : "";
    const cls = (hot === k ? " hot" : "") + (sel === k ? " sel" : "");
    if (b && b.new_box) {
      const o = p.blocks[+k.slice(1)].box;
      parts.push(`<rect class="orig" x="${o[0]}" y="${o[1]}" width="${o[2]}" height="${o[3]}" stroke="${col}"/>`);
    }
    const bx = pageBox(p, st, k);
    parts.push(`<rect class="bx${dim}${cls}" data-k="${k}" x="${bx[0]}" y="${bx[1]}" width="${bx[2]}" height="${bx[3]}" stroke="${col}" fill="${col}"/>`);
    const lab = String(labelOf(k, st)), w = fs * (0.75 * lab.length + 0.7);
    parts.push(`<rect x="${bx[0]}" y="${bx[1]}" width="${w}" height="${fs * 1.25}" fill="${col}" pointer-events="none"/>` +
      `<text class="lbl" x="${bx[0] + fs * 0.25}" y="${bx[1] + fs * 0.95}" font-size="${fs}">${lab}</text>`);
  });
  if (hotLine) { const l = p.lines.find(x => x.id === hotLine); if (l && l.box) parts.push(`<rect class="lnbox" x="${l.box[0]}" y="${l.box[1]}" width="${l.box[2]}" height="${l.box[3]}"/>`); }
  parts.push(`<rect id="drawlayer" x="0" y="0" width="${p.pw}" height="${p.ph}" fill="transparent" style="display:${mode ? "block" : "none"}"/>`);
  svg.innerHTML = parts.join("");
  svg.classList.toggle("drawing", !!mode);
  document.querySelectorAll(".row").forEach(r => { r.classList.toggle("hot", r.dataset.k === hot); r.classList.toggle("sel", r.dataset.k === sel); });
}
function setHot(k) { hot = k; renderOverlay(); }
function gotoPage(i) {
  if (i < 0 || i >= D.pages.length) return; cur = i; shown = new Set(); hot = sel = mode = hotLine = null; showMode();
  const p = D.pages[i];
  $("svg").setAttribute("viewBox", `0 0 ${p.pw} ${p.ph}`);
  $("img").src = "img/" + p.name + ".jpg";
  $("ptitle").textContent = `${i + 1}/${D.pages.length}  ${p.name}`;
  $("pageinfo").textContent = `${p.blocks.length} blocks, ALTO ${p.pw}x${p.ph}` + (p.rescaled ? `, image ${p.iw}x${p.ih} (boxes rescaled)` : "");
  try { localStorage.setItem(STORE + "_cur", i); } catch (e) {}
  renderList(); renderPageFields(); renderBlocks();
  $("imgcol").scrollTop = 0; $("side").scrollTop = 0;
}
function showMode() {
  const m = $("mode");
  m.style.display = mode ? "block" : "none";
  if (mode) m.textContent = (mode.type === "add" ? "Drag on the image to draw the new box" : "Drag on the image to redraw the box of " + mode.key) + "  (Esc to cancel)";
  document.querySelectorAll("[data-act=redraw],#addbox").forEach(b => b.classList.remove("on"));
}

// ---------- reorder
function move(k, d) {
  const st = ps(cur), i = st.order.indexOf(k), j = i + d; if (j < 0 || j >= st.order.length) return;
  st.order.splice(i, 1); st.order.splice(j, 0, k); touch(cur); renderBlocks();
  const r = document.querySelector(`.row[data-k="${k}"]`); if (r) r.scrollIntoView({block: "nearest"});
}
let dragK = null;
$("blocks").addEventListener("mousedown", e => { // only the handle starts a drag
  const r = e.target.closest(".row"); if (!r) return;
  r.draggable = !!e.target.closest(".handle");
});
$("blocks").addEventListener("dragstart", e => { const r = e.target.closest(".row"); if (!r || !r.draggable) { e.preventDefault(); return; } dragK = r.dataset.k; e.dataTransfer.setData("text/plain", dragK); e.dataTransfer.effectAllowed = "move"; });
$("blocks").addEventListener("dragover", e => {
  if (!dragK) return; e.preventDefault();
  document.querySelectorAll(".dragover").forEach(x => x.classList.remove("dragover"));
  const r = e.target.closest(".row"); if (r) r.classList.add("dragover");
});
$("blocks").addEventListener("drop", e => {
  if (!dragK) return; e.preventDefault();
  const r = e.target.closest(".row"); const st = ps(cur);
  if (r && r.dataset.k !== dragK) {
    st.order.splice(st.order.indexOf(dragK), 1);
    st.order.splice(st.order.indexOf(r.dataset.k), 0, dragK); // drop = insert before the target row
    touch(cur);
  }
  dragK = null; renderBlocks();
});
$("blocks").addEventListener("dragend", () => { dragK = null; document.querySelectorAll(".dragover").forEach(x => x.classList.remove("dragover")); });

// ---------- row interactions
$("blocks").addEventListener("mouseover", e => { const r = e.target.closest(".row"); if (r && hot !== r.dataset.k) setHot(r.dataset.k); });
$("blocks").addEventListener("mouseleave", () => setHot(null));
$("blocks").addEventListener("click", e => {
  const r = e.target.closest(".row"); if (!r) return; const k = r.dataset.k, st = ps(cur), p = D.pages[cur];
  const act = e.target.dataset.act;
  if (!act) {
    if (e.target.dataset.show) { if (e.target.checked) shown.add(k); else shown.delete(k); sel = k; renderOverlay(); return; }
    sel = k; renderOverlay(); return;
  }
  if (act === "up") move(k, -1); else if (act === "down") move(k, 1);
  else if (act === "redraw") { mode = {type: "redraw", key: k}; showMode(); e.target.classList.add("on"); renderOverlay(); }
  else if (act === "resetbox") { bs(st, k).new_box = null; touch(cur); renderBlocks(); }
  else if (act === "del") {
    if (!confirm("Delete this added box?")) return;
    delete st.added[k]; st.order = st.order.filter(x => x !== k); touch(cur); renderBlocks();
  } else if (act === "lines") {
    const box = r.querySelector(".lines");
    if (box.style.display === "block") { box.style.display = "none"; hotLine = null; renderOverlay(); return; }
    const own = k[0] === "a" ? st.added[k].linked_lines : bs(st, k).linked_lines;
    box.innerHTML = p.lines.map(l => `<label data-lid="${esc(l.id)}"><input type="checkbox" ${own.includes(l.id) ? "checked" : ""} value="${esc(l.id)}"> <code>${esc(l.id)}</code> <span class="badge">XML ${l.block + 1}</span> <span class="lt bn">${esc(l.text) || "<i>(empty)</i>"}</span></label>`).join("") || "<i>No lines</i>";
    box.style.display = "block";
  }
});
$("blocks").addEventListener("change", e => {
  const r = e.target.closest(".row"); if (!r) return; const k = r.dataset.k, st = ps(cur);
  if (e.target.dataset.flag) {
    bs(st, k).flags[e.target.dataset.flag] = e.target.checked; r.classList.toggle("excl", !!bs(st, k).flags.exclude); touch(cur); renderOverlay();
  } else if (e.target.closest(".lines")) {
    const arr = k[0] === "a" ? st.added[k].linked_lines : bs(st, k).linked_lines, v = e.target.value;
    const i = arr.indexOf(v); if (e.target.checked && i < 0) arr.push(v); if (!e.target.checked && i >= 0) arr.splice(i, 1);
    r.querySelector("[data-act=lines]").textContent = `Link ALTO lines (${arr.length})`; touch(cur);
  }
});
$("blocks").addEventListener("input", e => {
  const r = e.target.closest(".row"); if (!r || e.target.dataset.field !== "note") return; const k = r.dataset.k, st = ps(cur);
  if (k[0] === "a") st.added[k].note = e.target.value; else bs(st, k).note = e.target.value; touch(cur);
});
$("blocks").addEventListener("mouseover", e => {
  const l = e.target.closest("[data-lid]"); const id = l ? l.dataset.lid : null;
  if (id !== hotLine) { hotLine = id; renderOverlay(); }
});
$("svg").addEventListener("mouseover", e => { if (!mode && e.target.dataset.k) setHot(e.target.dataset.k); });
$("svg").addEventListener("mouseleave", () => { if (!mode) setHot(null); });
$("svg").addEventListener("click", e => {
  if (mode) return; const k = e.target.dataset.k; if (!k) return; sel = k;
  if ($("vis").value === "choose") { shown.add(k); const c = document.querySelector(`.row[data-k="${k}"] [data-show]`); if (c) c.checked = true; }
  renderOverlay();
  const r = document.querySelector(`.row[data-k="${k}"]`); if (r) r.scrollIntoView({block: "center", behavior: "smooth"});
});
$("addbox").addEventListener("click", e => { mode = {type: "add"}; showMode(); e.target.classList.add("on"); renderOverlay(); });

// ---------- drawing by dragging on the image
(function () {
  const svg = $("svg"); let start = null, ghost = null;
  function pt(e) {
    const r = svg.getBoundingClientRect(), p = D.pages[cur];
    return [Math.max(0, Math.min(p.pw, (e.clientX - r.left) / r.width * p.pw)), Math.max(0, Math.min(p.ph, (e.clientY - r.top) / r.height * p.ph))];
  }
  svg.addEventListener("pointerdown", e => {
    if (!mode) return; e.preventDefault(); svg.setPointerCapture(e.pointerId); start = pt(e);
    ghost = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    ghost.setAttribute("fill", "rgba(25,118,210,.2)"); ghost.setAttribute("stroke", "#1976d2"); ghost.setAttribute("stroke-width", 2);
    ghost.setAttribute("vector-effect", "non-scaling-stroke"); svg.appendChild(ghost);
  });
  svg.addEventListener("pointermove", e => {
    if (!start) return; const q = pt(e);
    ghost.setAttribute("x", Math.min(start[0], q[0])); ghost.setAttribute("y", Math.min(start[1], q[1]));
    ghost.setAttribute("width", Math.abs(q[0] - start[0])); ghost.setAttribute("height", Math.abs(q[1] - start[1]));
  });
  svg.addEventListener("pointerup", e => {
    if (!start) return; const q = pt(e), s = start; start = null;
    const box = [Math.min(s[0], q[0]), Math.min(s[1], q[1]), Math.abs(q[0] - s[0]), Math.abs(q[1] - s[1])].map(v => Math.round(v));
    if (box[2] < 5 || box[3] < 5) { renderOverlay(); return; } // too small: treat as a click, stay in draw mode
    const st = ps(cur); let k = mode.key;
    if (mode.type === "add") { k = "a" + st.next_added++; st.added[k] = {box, note: "", linked_lines: []}; st.order.push(k); }
    else if (k[0] === "a") st.added[k].box = box; else bs(st, k).new_box = box;
    mode = null; showMode(); touch(cur); sel = k; renderBlocks();
    const r = document.querySelector(`.row[data-k="${k}"]`); if (r) r.scrollIntoView({block: "center"});
  });
})();
document.addEventListener("keydown", e => { if (e.key === "Escape" && mode) { mode = null; showMode(); renderOverlay(); } });

// ---------- page-level fields, navigation
$("pg").addEventListener("change", e => {
  const st = ps(cur);
  if (e.target.id === "pg_status") { st.status = e.target.value; st.updated_at = now(); if (!st.started_at) st.started_at = st.updated_at; save(); renderList(); return; }
  if (e.target.id === "pg_type") st.page_type = e.target.value;
  if (e.target.name === "drop") st.drop_page = e.target.value;
  touch(cur);
});
$("pg").addEventListener("input", e => {
  const st = ps(cur);
  if (e.target.id === "pg_reason") st.drop_reason = e.target.value; else if (e.target.id === "pg_note") st.note = e.target.value; else return;
  touch(cur);
});
$("plist").addEventListener("click", e => { const d = e.target.closest("[data-i]"); if (d) gotoPage(+d.dataset.i); });
$("prev").onclick = () => gotoPage(cur - 1); $("next").onclick = () => gotoPage(cur + 1);
$("zoom").oninput = e => { $("wrap").style.width = e.target.value + "%"; };
$("lblmode").onchange = renderOverlay; $("lsize").oninput = renderOverlay;
function syncShow() { document.querySelectorAll(".row").forEach(r => { r.querySelector("[data-show]").checked = shown.has(r.dataset.k); }); renderOverlay(); }
$("vis").onchange = () => { if ($("vis").value === "choose" && !shown.size && sel) shown.add(sel); syncShow(); };
$("hideall").onclick = () => { $("vis").value = "choose"; shown.clear(); syncShow(); };
$("showall").onclick = () => { $("vis").value = "choose"; ps(cur).order.forEach(k => shown.add(k)); syncShow(); };
$("reviewer").oninput = e => { S.reviewer = e.target.value; save(); };

// ---------- export / import
function buildExport() {
  const out = {schema_version: SCHEMA, kind: "bl_review", reviewer: S.reviewer || "", exported_at: now(),
    reading_order_rule: D.rule, pages: {}};
  D.pages.forEach(p => {
    const st = S.pages[p.name], blocks = {}, added = {};
    p.blocks.forEach((b, i) => {
      const k = "b" + i, s = st.blocks[k] || {};
      const f = s.flags || {};
      blocks[k] = {id: b.id, flags: {exclude: !!f.exclude, phantom: !!f.phantom, box_wrong: !!f.box_wrong, text_not_in_box: !!f.text_not_in_box},
        note: s.note || "", original_box: b.box, new_box: s.new_box || null, linked_lines: s.linked_lines || []};
    });
    Object.keys(st.added).forEach(k => { added[k] = st.added[k]; });
    out.pages[p.name] = {alto_page_size: [p.pw, p.ph], original_block_ids: p.blocks.map(b => b.id),
      original_order: p.blocks.map((_, i) => "b" + i), new_order: st.order,
      new_order_block_ids: st.order.map(k => k[0] === "b" ? p.blocks[+k.slice(1)].id : "added:" + k),
      status: st.status, page_type: st.page_type, drop_page: st.drop_page, drop_reason: st.drop_reason, note: st.note,
      blocks, added_boxes: added, started_at: st.started_at, updated_at: st.updated_at};
  });
  return out;
}
$("exp").onclick = () => {
  const blob = new Blob([JSON.stringify(buildExport(), null, 1)], {type: "application/json"});
  const a = document.createElement("a"), d = new Date().toISOString().slice(0, 10);
  a.href = URL.createObjectURL(blob); a.download = `review_${d}.json`; document.body.appendChild(a); a.click(); a.remove();
  dirty = false; $("dirty").textContent = "exported " + new Date().toLocaleTimeString();
};
$("imp").onclick = () => $("impf").click();
$("impf").onchange = e => {
  const f = e.target.files[0]; if (!f) return; const rd = new FileReader();
  rd.onload = () => {
    try {
      const o = JSON.parse(rd.result);
      if (o.kind !== "bl_review" || !o.pages) throw new Error("not a bl_review file");
      if (o.schema_version !== SCHEMA) throw new Error("schema_version " + o.schema_version + " not supported (app: " + SCHEMA + ")");
      if (!confirm("Replace the review state in this browser with the file's content?")) return;
      const ns = {schema_version: SCHEMA, reviewer: o.reviewer || "", pages: {}};
      let missing = 0;
      D.pages.forEach(p => {
        const x = o.pages[p.name]; const st = defPage(p);
        if (x) {
          Object.assign(st, {status: x.status || st.status, page_type: x.page_type || "", drop_page: x.drop_page || "undecided",
            drop_reason: x.drop_reason || "", note: x.note || "", order: x.new_order || st.order,
            added: x.added_boxes || {}, started_at: x.started_at || null, updated_at: x.updated_at || null});
          Object.keys(x.blocks || {}).forEach(k => { const b = x.blocks[k]; st.blocks[k] = {flags: b.flags || {}, note: b.note || "", new_box: b.new_box || null, linked_lines: b.linked_lines || []}; });
          Object.values(st.added).forEach(a => { a.linked_lines = a.linked_lines || []; a.note = a.note || ""; });
        } else missing++;
        ns.pages[p.name] = normalize(p, st);
      });
      S = ns; $("reviewer").value = S.reviewer; save(); gotoPage(cur);
      alert("Imported." + (missing ? " " + missing + " pages were not in the file (left as not started)." : ""));
    } catch (err) { alert("Import failed: " + err.message); }
  };
  rd.readAsText(f); e.target.value = "";
};
window.addEventListener("beforeunload", e => { if (dirty && !storageOK) { e.preventDefault(); e.returnValue = ""; } });

$("rule").textContent = D.rule;
load();
let start = 0; try { start = +localStorage.getItem(STORE + "_cur") || 0; } catch (e) {}
dirty = false; gotoPage(Math.min(start, D.pages.length - 1));
</script></body></html>
"""

README = """BL test pages review app
========================
Generated by tools/bl_review_app.py on {today}. Everything stays on this computer.

Open
  Double-click index.html (any recent Chrome, Edge or Firefox). No server needed.
  Keep the img/ folder next to index.html.

Use
  Pick a page in the left list (the dot shows its status), or use Prev / Next.
  Boxes on the image are numbered in XML order. Drag rows in the right-hand list
  (grab the dotted handle) or use the arrows to set the reading order.
  Tick flags, write notes, redraw a box (drag on the image), add a box for text
  that has none, link ALTO lines to a box. Set the page status, page type and the
  "drop page" answer on the right. Transcription text is never edited.
  Rule in force for reading order: {rule}

Saving
  The app autosaves in this browser (localStorage) after every change. That copy
  lives only in this browser profile on this machine, so also export now and then.
  Export JSON  -> downloads review_<date>.json. Save it as
                  results/bl_review/review_<date>.json (e.g. review_2026-11-22.json).
  Import JSON  -> loads such a file (replaces the state in this browser); use it to
                  resume after clearing the browser data or to move to another machine.
  The file format is described at the top of tools/bl_review_app.py.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group2", required=True, help="path of the 'Group 2' folder")
    ap.add_argument("--out", default=os.path.join("results", "bl_review"))
    ap.add_argument("--limit", type=int, default=0, help="only the first N pages (testing)")
    args = ap.parse_args()

    data = os.path.join(args.group2, DATA_REL)
    with open(os.path.join(data, PAGE_LIST), encoding="utf-8") as f:
        names = [ln.strip() for ln in f if ln.strip()]
    if args.limit:
        names = names[: args.limit]
    img_dir = os.path.join(args.out, "img")
    os.makedirs(img_dir, exist_ok=True)

    pages, rescaled = [], []
    for name in names:
        xml_path = os.path.join(data, ALTO_DIR, name + ".xml")
        pw, ph, blocks = parse_page(xml_path)
        lines = parse_lines(xml_path)
        img = Image.open(os.path.join(data, ALTO_DIR, name + ".tif"))
        iw, ih = img.size
        resc = abs(iw / pw - 1) > 0.02 or abs(ih / ph - 1) > 0.02
        if resc:
            rescaled.append(name)
        scale = min(1.0, MAX_SIDE / max(iw, ih))
        small = img.convert("RGB").resize((max(1, round(iw * scale)), max(1, round(ih * scale))))
        small.save(os.path.join(img_dir, name + ".jpg"), quality=88)
        pages.append({
            "name": name, "pw": pw, "ph": ph, "iw": iw, "ih": ih, "rescaled": resc,
            "blocks": [{"id": b["id"], "box": [b["x"], b["y"], b["w"], b["h"]], "text": b["text"]} for b in blocks],
            "lines": lines,
        })

    payload = json.dumps({"schema_version": SCHEMA_VERSION, "rule": RULE_TEXT, "pages": pages},
                         ensure_ascii=False).replace("</", "<\\/")
    with open(os.path.join(args.out, "index.html"), "w", encoding="utf-8") as f:
        f.write(HTML.replace("__DATA__", payload))
    with open(os.path.join(args.out, "README.txt"), "w", encoding="utf-8") as f:
        f.write(README.format(today=datetime.date.today().isoformat(), rule=RULE_TEXT))
    print(f"{len(pages)} pages written to {args.out}")
    print(f"{len(rescaled)} pages with rescaled boxes: " + ", ".join(rescaled))
    print("pages with no blocks: " + ", ".join(p['name'] for p in pages if not p['blocks']))


if __name__ == "__main__":
    main()
