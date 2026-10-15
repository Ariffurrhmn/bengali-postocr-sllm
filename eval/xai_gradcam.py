"""Token-level Grad-CAM for the correction models (review item X1, third
review: the reviewer asked for an explainability analysis "like Grad-CAM").

Grad-CAM (Selvaraju et al.) weights a CNN layer's feature maps by the gradient
of the target score, averaged over spatial positions, and keeps the positive
part. A transformer has no spatial feature map, but its hidden states form a
map over token positions, so the same recipe applies there:

  literal   alpha = mean over positions of d(score)/dH;  cam[p] = ReLU(sum_d alpha_d H[p, d])
  hires     cam[p] = ReLU(sum_d dscore/dH[p, d] * H[p, d])   (element-wise variant, as in HiResCAM)

The score is the logit of each token the model actually generated (greedy,
plain decoding, the same prompt and chunks as the main runs). For every
generated token the map is split into three parts of the sequence the model
saw: the OCR text it was asked to correct, the rest of the prompt
(instruction and chat template), and its own earlier output. For generated
tokens that copy an OCR token (token-level alignment of output and OCR text),
we also record whether the map peaks on that source token (+-1 position):
a copying model should rely on the aligned input, a regenerating one on the
instruction or its own output.

Decoder-only models: maps over the whole sequence. BanglaT5 (encoder-decoder):
maps over the encoder states, which hold only the OCR text, so only the
source-peak measure applies.

Usage (from eval/; CPU for small models, GPU for the rest):
    python xai_gradcam.py --model qwen3-0.6b --engines tesseract,ground_truth \\
        --n-pages 4 --out ../results/xai_qwen3-0.6b.jsonl
"""
import argparse
import difflib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "correction"))
from chunking import chunk_text  # noqa: E402
from models import (CHAT_TEMPLATE_KWARGS, MAX_NEW_TOKENS_CAP, MAX_NEW_TOKENS_FLOOR,  # noqa: E402
                    MAX_NEW_TOKENS_MULTIPLIER, SEQ2SEQ_MODELS, build_prompt, load_model)

sys.stdout.reconfigure(encoding="utf-8")


def max_new_tokens(tokenizer, text):
    n = len(tokenizer.encode(text, add_special_tokens=False))
    return min(MAX_NEW_TOKENS_CAP, max(MAX_NEW_TOKENS_FLOOR, int(n * MAX_NEW_TOKENS_MULTIPLIER)))


def decoder_prompt(model_key, tokenizer, ocr_text):
    """Prompt ids exactly as in correction/models.py, plus the token span of
    the OCR text inside it (from character offsets of the rendered template)."""
    messages = [{"role": "user", "content": build_prompt("en", ocr_text)}]
    kw = CHAT_TEMPLATE_KWARGS.get(model_key, {})
    ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_dict=True,
                                        return_tensors="pt", **kw)["input_ids"][0].tolist()
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **kw)
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    match = enc["input_ids"] == ids
    if not match:
        enc = tokenizer(text, add_special_tokens=True, return_offsets_mapping=True)
        match = enc["input_ids"] == ids
    start = text.rfind(ocr_text)
    end = start + len(ocr_text)
    ocr_pos = [i for i, (a, b) in enumerate(enc["offset_mapping"]) if b > start and a < end and b > a]
    if not match:  # fall back to the re-tokenised text, and say so
        ids = enc["input_ids"]
    return ids, ocr_pos, match


def cams(score, layers, n_pos):
    grads = torch.autograd.grad(score, layers, retain_graph=True)
    out = []
    for g, h in zip(grads, layers):
        g, h = g[0, :n_pos].float(), h[0, :n_pos].float()
        literal = torch.relu((h * g.mean(0)).sum(-1))
        hires = torch.relu((h * g).sum(-1))
        out.append({"literal": literal.detach().cpu().numpy(), "hires": hires.detach().cpu().numpy()})
    return out


def norm(m):
    s = m.sum()
    return m / s if s > 0 else m


def copy_alignment(src_ids, out_ids):
    """Generated index -> OCR token index, for output tokens inside an
    identical run of token ids (difflib on the two id sequences)."""
    sm = difflib.SequenceMatcher(a=src_ids, b=out_ids, autojunk=False)
    m = {}
    for blk in sm.get_matching_blocks():
        for k in range(blk.size):
            m[blk.b + k] = blk.a + k
    return m


def analyse_chunk(model_key, tokenizer, model, text, max_targets, layer_frac, save_maps=False):
    dev = model.device
    seq2seq = model_key in SEQ2SEQ_MODELS
    emb_layer = model.get_input_embeddings() if not seq2seq else model.get_encoder().get_input_embeddings()
    mnt = max_new_tokens(tokenizer, text)
    if seq2seq:
        enc_ids = tokenizer(text, return_tensors="pt", truncation=True)["input_ids"].to(dev)
        with torch.no_grad():
            gen = model.generate(input_ids=enc_ids, max_new_tokens=mnt, do_sample=False)[0]
        dec_in = gen.unsqueeze(0)  # starts with the decoder start token
        out_ids = gen[1:].tolist()
        embeds = emb_layer(enc_ids).detach().requires_grad_(True)
        res = model(inputs_embeds=embeds, decoder_input_ids=dec_in, output_hidden_states=True)
        hs = res.encoder_hidden_states
        src_ids = enc_ids[0].tolist()
        ocr_pos = list(range(len(src_ids)))
        n_prompt, template_match = len(src_ids), True
    else:
        ids, ocr_pos, template_match = decoder_prompt(model_key, tokenizer, text)
        p_ids = torch.tensor([ids], device=dev)
        with torch.no_grad():
            gen = model.generate(input_ids=p_ids, attention_mask=torch.ones_like(p_ids),
                                 max_new_tokens=mnt, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)[0]
        full = gen.unsqueeze(0)
        print(f"  generated {len(gen) - len(ids)} tokens", file=sys.stderr, flush=True)
        n_prompt = len(ids)
        out_ids = gen[n_prompt:].tolist()
        embeds = emb_layer(full).detach().requires_grad_(True)
        # Base model only: the output head is applied to the one position each
        # score needs, instead of materialising logits for the whole sequence
        # (which ran a 16 GB machine out of memory in the backward pass).
        res = model.base_model(inputs_embeds=embeds, output_hidden_states=True)
        hs = res.hidden_states
        head = model.get_output_embeddings()
        src_ids = [ids[p] for p in ocr_pos]
    special = set(tokenizer.all_special_ids)
    n_layers = len(hs) - 1
    layer_idx = sorted({0, *[max(1, min(n_layers, round(f * n_layers))) for f in layer_frac]})
    layers = [hs[i] for i in layer_idx]
    align = copy_alignment(src_ids, out_ids)
    cand = [j for j, t in enumerate(out_ids) if t not in special]
    step = max(1, len(cand) // max_targets) if max_targets else 1
    targets = cand[::step][:max_targets] if max_targets else cand
    print(f"  scoring {len(targets)} targets", file=sys.stderr, flush=True)
    records = []
    ocr_set = set(ocr_pos)
    seq = src_ids if seq2seq else ids + out_ids
    # Positions left out of every map: the first token and special tokens
    # (attention sinks with very large activations, which otherwise take most
    # of a gradient-times-activation map) and, for decoder-only models, the
    # position whose hidden state produces the scored logit (late layers put
    # most of their mass there, which says nothing about the context).
    sink = {0} | {p for p, t in enumerate(seq) if t in special}
    for j in targets:
        if seq2seq:
            score = res.logits[0, j, out_ids[j]]
            n_pos = len(src_ids)
        else:
            h_last = res.last_hidden_state[0, n_prompt + j - 1]
            score = head(h_last.to(head.weight.device))[out_ids[j]]
            n_pos = n_prompt + j
        maps = cams(score, layers, n_pos)
        rec = {"j": j, "token": tokenizer.decode([out_ids[j]]), "copy_of": align.get(j)}
        current = set() if seq2seq else {n_pos - 1}
        for li, m in zip(layer_idx, maps):
            for kind, v in m.items():
                v = norm(v)
                key = f"L{li}_{kind}"
                sink_share = float(sum(v[p] for p in sink if p < len(v)))
                cur_share = float(sum(v[p] for p in current))
                v = v.copy()
                for p in (sink | current):
                    if p < len(v):
                        v[p] = 0.0
                v = norm(v)
                if save_maps:
                    rec.setdefault("maps", {})[key] = [round(float(x), 5) for x in v]
                if seq2seq:
                    rec[key] = {"ocr": float(v.sum())}
                else:
                    rec[key] = {"ocr": float(v[ocr_pos].sum()),
                                "prefix": float(v[n_prompt:].sum()),
                                "prompt_other": float(sum(x for p, x in enumerate(v[:n_prompt]) if p not in ocr_set))}
                rec[key]["excluded_sink"] = sink_share
                rec[key]["excluded_current"] = cur_share
                if j in align and v.sum() > 0:
                    src = ocr_pos[align[j]]
                    ocr_vals = np.array([v[p] for p in ocr_pos])
                    peak = ocr_pos[int(ocr_vals.argmax())]
                    win = [p for p in (src - 1, src, src + 1) if 0 <= p < len(v)]
                    rec[key]["src_peak"] = bool(abs(peak - src) <= 1)
                    rec[key]["src_mass"] = float(sum(v[p] for p in win))
        records.append(rec)
    prompt_tokens = [tokenizer.decode([t]) for t in (src_ids if seq2seq else ids)] if save_maps else None
    return {"prompt_tokens": prompt_tokens, "ocr_pos": ocr_pos, "n_prompt": n_prompt, "n_ocr_tokens": len(ocr_pos), "n_out": len(out_ids),
            "template_ids_match": bool(template_match), "layers": layer_idx, "n_layers": n_layers,
            "copy_share": len(align) / len(out_ids) if out_ids else 0.0,
            "output": tokenizer.decode(out_ids, skip_special_tokens=True), "targets": records}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ocr-path", type=Path, default=REPO_ROOT / "results" / "ocr_eval40.jsonl")
    ap.add_argument("--engines", default="tesseract,easyocr,ground_truth")
    ap.add_argument("--n-pages", type=int, default=10, help="pages sampled with seed 403")
    ap.add_argument("--chunks-per-page", type=int, default=1)
    ap.add_argument("--max-targets", type=int, default=24, help="generated tokens scored per chunk (0 = all)")
    ap.add_argument("--layer-frac", default="0.5,0.75", help="layers besides the embeddings, as fractions of depth")
    ap.add_argument("--save-maps", action="store_true", help="store the full maps (large)")
    ap.add_argument("--stop-after-minutes", type=float, default=None,
                    help="start no new chunk after this many minutes")
    ap.add_argument("--threads", type=int, default=0, help="CPU threads (0 = keep the default)")
    ap.add_argument("--fp32", action="store_true", help="cast weights to float32 (CPU loads bfloat16)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    pages = [json.loads(l) for l in args.ocr_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    pages.sort(key=lambda r: r["page_id"])
    pages = random.Random(403).sample(pages, min(args.n_pages, len(pages)))
    if args.threads:
        torch.set_num_threads(args.threads)
    tokenizer, model = load_model(args.model)
    model.requires_grad_(False)
    if args.fp32:
        model.float()
    fracs = [float(x) for x in args.layer_frac.split(",")]
    started = time.time()
    done = set()
    if args.out.exists():
        for l in args.out.read_text(encoding="utf-8").splitlines():
            r = json.loads(l)
            done.add((r["page_id"], r["engine"], r["chunk"]))
    with args.out.open("a", encoding="utf-8") as f:
        for page in pages:
            for e in args.engines.split(","):
                for c, chunk in enumerate(chunk_text(page[e], tokenizer)[:args.chunks_per_page]):
                    if (page["page_id"], e, c) in done:
                        continue
                    if args.stop_after_minutes is not None and time.time() - started > args.stop_after_minutes * 60:
                        print("time budget reached; stopping", flush=True)
                        return
                    t0 = time.time()
                    r = analyse_chunk(args.model, tokenizer, model, chunk, args.max_targets, fracs, args.save_maps)
                    r.update({"model": args.model, "page_id": page["page_id"], "engine": e, "chunk": c,
                              "sec": round(time.time() - t0, 1)})
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    f.flush()
                    print(f"{args.model} {page['page_id']} {e} chunk {c}: {len(r['targets'])} targets, "
                          f"copy share {r['copy_share']:.2f}, {r['sec']} s", flush=True)


if __name__ == "__main__":
    main()
