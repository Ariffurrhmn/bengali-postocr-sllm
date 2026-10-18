"""Checks that Transformers' repetition penalty and n-gram ban act on prompt
tokens (draft-04 correction C2). Uses the installed Transformers version and a
cached decoder-only model; one forward step, so it runs on a CPU.

1. repetition_penalty=1.3: the processed score of every token id that occurs in
   the prompt (and only there) is the raw logit divided by 1.3 if positive,
   multiplied by 1.3 if negative; all other token ids are unchanged.
2. no_repeat_ngram_size=4: a prompt that ends with the first three tokens of a
   4-gram it already contains gets the fourth token banned (score -inf).

Usage:
    HF_HUB_OFFLINE=1 python eval/check_prompt_constraints.py --model Qwen/Qwen3-0.6B \
        --out-json results/prompt_constraints_<version>.json
"""
import argparse
import json
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

# The first reference line of Table XXI, as an OCR-correction prompt.
TEXT = "শাখায় ঝুলিয়া থাকে, রাত্রি কালে আহার নিমিত্ত বহির্গত"
PROMPT = "Correct the OCR errors in this Bengali text:\n" + TEXT + "\nCorrected text:\n"


def first_step(model, ids, **kw):
    out = model.generate(ids, max_new_tokens=1, do_sample=False, output_scores=True, output_logits=True,
                         return_dict_in_generate=True, pad_token_id=model.config.eos_token_id, **kw)
    return out.logits[0][0].float(), out.scores[0][0].float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()

    # 1. Repetition penalty.
    ids = tok(PROMPT, return_tensors="pt").input_ids
    raw, proc = first_step(model, ids, repetition_penalty=1.3)
    in_prompt = torch.zeros_like(raw, dtype=torch.bool)
    in_prompt[ids[0].unique()] = True
    expected = torch.where(raw > 0, raw / 1.3, raw * 1.3)
    penalised = torch.isclose(proc, expected, rtol=1e-5, atol=1e-5) & ~torch.isclose(proc, raw, rtol=0, atol=0)
    unchanged = torch.isclose(proc, raw, rtol=0, atol=0)
    rep = {"prompt_tokens": int(ids.shape[1]), "distinct_prompt_ids": int(in_prompt.sum()),
           "prompt_ids_penalised": int((penalised & in_prompt).sum()),
           "other_ids_unchanged": int((unchanged & ~in_prompt).sum()), "other_ids": int((~in_prompt).sum())}

    # 2. n-gram ban: the line's tokens, then its first three tokens again.
    line = tok(TEXT, add_special_tokens=False).input_ids
    ids2 = torch.tensor([line + line[:3]])
    raw2, proc2 = first_step(model, ids2, no_repeat_ngram_size=4)
    ban = {"banned_id": line[3], "raw_score": float(raw2[line[3]]), "processed_score": float(proc2[line[3]]),
           "ids_banned": int(torch.isinf(proc2).sum())}

    result = {"transformers": transformers.__version__, "torch": torch.__version__, "model": args.model,
              "repetition_penalty": rep, "no_repeat_ngram": ban,
              "penalty_acts_on_prompt": rep["prompt_ids_penalised"] == rep["distinct_prompt_ids"]
                                        and rep["other_ids_unchanged"] == rep["other_ids"],
              "ngram_ban_acts_on_prompt": ban["processed_score"] == float("-inf")
                                          and ban["raw_score"] != float("-inf")}
    print(json.dumps(result, indent=1))
    if args.out_json:
        args.out_json.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
