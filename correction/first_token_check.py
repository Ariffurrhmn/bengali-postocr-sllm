"""First-token precision check (third review, item C1).

TituLLMs 3B Instruct ran in float16 and returned 43 empty outputs of 120,
each ending at the first generated token. This records, for every chunk the
main run gave the model, what the first decoding step looks like: whether
any hidden state or logit is inf/NaN, the largest absolute activation per
layer (float16 overflows above 65504), the end-of-sequence probability, and
the greedy first token. Run it once per precision (POSTOCR_DTYPE=fp16 /
fp32, see models.load_model) and compare the two files.

Usage (Kaggle, from correction/):
    POSTOCR_DTYPE=fp16 python first_token_check.py --model titullm-3b-instruct \\
        --ocr-path <ocr_eval40.jsonl> --out first_token_fp16.jsonl
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch

from chunking import chunk_text
from models import CHAT_TEMPLATE_KWARGS, build_prompt, load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ocr-path", type=Path, required=True)
    ap.add_argument("--engines", default="tesseract,easyocr,ground_truth")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-pages", type=int, default=0, help="0 = all pages")
    args = ap.parse_args()

    tok, model = load_model(args.model)
    eos = set(model.generation_config.eos_token_id if isinstance(model.generation_config.eos_token_id, list)
              else [model.generation_config.eos_token_id]) | {tok.eos_token_id}
    eos.discard(None)
    label = os.environ.get("POSTOCR_DTYPE", str(model.dtype))
    pages = [json.loads(l) for l in args.ocr_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.max_pages:
        pages = pages[:args.max_pages]
    with args.out.open("w", encoding="utf-8") as f:
        for page in pages:
            for e in args.engines.split(","):
                for c, chunk in enumerate(chunk_text(page[e], tok)):
                    msgs = [{"role": "user", "content": build_prompt("en", chunk)}]
                    inp = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                                  return_dict=True, **CHAT_TEMPLATE_KWARGS.get(args.model, {}))
                    inp = inp.to(model.device)
                    with torch.no_grad():
                        out = model(**inp, output_hidden_states=True)
                    logits = out.logits[0, -1].float()
                    finite = bool(torch.isfinite(logits).all())
                    probs = torch.softmax(logits, -1) if finite else None
                    top = torch.topk(logits, 5)
                    rec = {"model": args.model, "precision": label, "page_id": page["page_id"], "engine": e,
                           "chunk": c, "prompt_tokens": int(inp["input_ids"].shape[-1]),
                           "logits_finite": finite,
                           "hidden_finite": [bool(torch.isfinite(h).all()) for h in out.hidden_states],
                           "hidden_max_abs": [float(h.abs().max()) for h in out.hidden_states],
                           "first_token": int(top.indices[0]),
                           "first_is_eos": int(top.indices[0]) in eos,
                           "eos_prob": float(sum(probs[i] for i in eos)) if finite else None,
                           "top5": [[int(i), tok.decode([int(i)])] for i in top.indices]}
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
            print(page["page_id"], "done", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
