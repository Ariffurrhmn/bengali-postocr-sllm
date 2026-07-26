"""Isolated timing probe: loads one model and generates a small, fixed
number of tokens with live per-step progress printed, so we get hard
per-token timing data quickly instead of guessing from a long, silent run.

Also runs a second generation call WITHOUT repetition_penalty/no_repeat_ngram_size
to check whether those settings are the source of any slowdown, and a third
call using multiple CPU threads explicitly, in case thread config is the issue.
"""
import io
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from models import MODEL_IDS

REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
    model_key = sys.argv[1] if len(sys.argv) > 1 else "llama3.2-1b"

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"torch.get_num_threads() = {torch.get_num_threads()}")
    print(f"torch version: {torch.__version__}")

    model_id = MODEL_IDS[model_key]
    print(f"Loading {model_key} ({model_id})...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16)
    model.eval()
    print(f"Loaded in {time.time() - t0:.1f}s")

    prompt = "Correct this OCR text: hello world example test."
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )
    input_len = inputs["input_ids"].shape[-1]
    print(f"Input length: {input_len} tokens")

    for label, gen_kwargs in [
        ("baseline (no repetition controls)", dict(max_new_tokens=20, do_sample=False)),
        (
            "with repetition_penalty + no_repeat_ngram_size",
            dict(
                max_new_tokens=20,
                do_sample=False,
                repetition_penalty=1.3,
                no_repeat_ngram_size=4,
            ),
        ),
    ]:
        print(f"\n--- {label} ---")
        t0 = time.time()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                **gen_kwargs,
            )
        elapsed = time.time() - t0
        n_generated = output_ids.shape[-1] - input_len
        print(f"Generated {n_generated} tokens in {elapsed:.1f}s "
              f"({elapsed / max(n_generated, 1):.2f} s/token)")
        text = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)
        print(f"Output: {text!r}")


if __name__ == "__main__":
    main()
