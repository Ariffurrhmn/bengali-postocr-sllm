"""Zero-shot correction wrappers for the 5 models named in the methodology.

Two loading/inference paths, matching how each model actually works:
  - Causal LM + chat template (Phi-3 Mini, Llama 3.2 1B, Gemma 2B, TituLLMs 1B):
    instruction-tuned, prompted via tokenizer.apply_chat_template.
  - Seq2seq (BanglaT5): not instruction-tuned, no chat template — prompted
    with a direct text-to-text framing instead.

Greedy decoding (do_sample=False) is used throughout: this is a correction
task, not creative generation, and determinism matters for reproducibility.
"""
from dataclasses import dataclass

MODEL_IDS = {
    "phi3-mini": "microsoft/Phi-3-mini-4k-instruct",
    "llama3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "gemma-2b": "google/gemma-2b-it",
    "titullm-1b": "hishab/titulm-llama-3.2-1b-v1.1",
    "banglat5": "csebuetnlp/banglat5",
}

CAUSAL_LM_MODELS = {"phi3-mini", "llama3.2-1b", "gemma-2b", "titullm-1b"}
SEQ2SEQ_MODELS = {"banglat5"}

CORRECTION_INSTRUCTION = (
    "The following text was produced by OCR on a historical Bengali document "
    "and may contain recognition errors. Correct the OCR errors and output "
    "only the corrected Bengali text, with no explanation, preamble, or "
    "additional commentary.\n\nOCR text:\n{ocr_text}"
)


MIN_OCR_CHARS_FOR_CORRECTION = 10  # below this, OCR output is treated as too
                                    # degenerate to correct meaningfully — the
                                    # model has almost no real signal and tends
                                    # to confabulate rather than correct


@dataclass
class CorrectionResult:
    model_key: str
    raw_output: str
    truncated: bool
    skipped: bool = False  # True if OCR input was too short/empty to attempt
                            # correction — raw_output equals the input verbatim


def load_model(model_key: str, hf_token: str | None = None):
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
    )

    model_id = MODEL_IDS[model_key]
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)

    # bfloat16 (not float32): halves memory footprint, which matters on
    # free-tier Colab's ~12-13GB RAM (Phi-3 Mini at fp32 was getting killed
    # partway through weight loading). bfloat16 is well-supported for CPU
    # inference in modern torch/transformers and standard practice for
    # small-LM CPU inference — unlike fp16, it doesn't have CPU numerical
    # stability issues. Applied to all 5 models for consistency.
    dtype = torch.bfloat16

    if model_key in SEQ2SEQ_MODELS:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id, token=hf_token, dtype=dtype
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, token=hf_token, dtype=dtype
        )
    model.eval()
    return tokenizer, model


MAX_NEW_TOKENS_CAP = 512  # hard ceiling regardless of input length
MAX_NEW_TOKENS_MULTIPLIER = 1.5  # corrected output should be roughly the
                                  # same length as the OCR input, not an
                                  # unrelated fixed size — capping relative
                                  # to input length bounds worst-case runtime
                                  # per page (whole-page prompts with a flat
                                  # 512-token cap were taking 20+ min/page on
                                  # free Colab CPU for some models, likely
                                  # generating close to the full cap before
                                  # naturally stopping)
MAX_NEW_TOKENS_FLOOR = 64  # minimum, so very short OCR inputs still get
                            # enough room for a real corrected line


def correct_text(
    model_key: str, tokenizer, model, ocr_text: str, max_new_tokens: int | None = None
) -> CorrectionResult:
    import torch

    if len(ocr_text.strip()) < MIN_OCR_CHARS_FOR_CORRECTION:
        return CorrectionResult(
            model_key=model_key, raw_output=ocr_text, truncated=False, skipped=True
        )

    if model_key in SEQ2SEQ_MODELS:
        # BanglaT5 is not instruction-tuned: no chat template, direct
        # text-to-text framing instead of a "please fix this" instruction.
        inputs = tokenizer(ocr_text, return_tensors="pt", truncation=True)
    else:
        prompt = CORRECTION_INSTRUCTION.format(ocr_text=ocr_text)
        messages = [{"role": "user", "content": prompt}]
        # apply_chat_template(..., return_dict=True) returns a BatchEncoding
        # (dict-like, with attention_mask) rather than a bare tensor — needed
        # so **inputs below passes attention_mask to generate() too, not
        # just input_ids.
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

    input_len = inputs["input_ids"].shape[-1]

    if max_new_tokens is None:
        ocr_token_len = len(tokenizer.encode(ocr_text, add_special_tokens=False))
        max_new_tokens = min(
            MAX_NEW_TOKENS_CAP,
            max(MAX_NEW_TOKENS_FLOOR, int(ocr_token_len * MAX_NEW_TOKENS_MULTIPLIER)),
        )

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy decoding: deterministic, appropriate
                               # for a correction task (not creative generation)
            repetition_penalty=1.3,  # greedy decoding on noisy/short OCR
            no_repeat_ngram_size=4,  # input is prone to degenerate repetition
                                      # loops (seen in practice on this task,
                                      # e.g. a phrase repeated until max_new_tokens
                                      # is hit) — both are standard, deterministic
                                      # guards, not sampling/randomness
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    if model_key in SEQ2SEQ_MODELS:
        generated_ids = output_ids[0]
    else:
        generated_ids = output_ids[0][input_len:]  # strip the echoed prompt

    truncated = generated_ids.shape[-1] >= max_new_tokens
    raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return CorrectionResult(model_key=model_key, raw_output=raw_output, truncated=truncated)
