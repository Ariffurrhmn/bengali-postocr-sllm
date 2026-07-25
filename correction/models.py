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


@dataclass
class CorrectionResult:
    model_key: str
    raw_output: str
    truncated: bool


def load_model(model_key: str, hf_token: str | None = None):
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
    )

    model_id = MODEL_IDS[model_key]
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)

    if model_key in SEQ2SEQ_MODELS:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id, token=hf_token, torch_dtype=torch.float32
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, token=hf_token, torch_dtype=torch.float32
        )
    model.eval()
    return tokenizer, model


def correct_text(
    model_key: str, tokenizer, model, ocr_text: str, max_new_tokens: int = 512
) -> CorrectionResult:
    import torch

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

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy decoding: deterministic, appropriate
                               # for a correction task (not creative generation)
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    if model_key in SEQ2SEQ_MODELS:
        generated_ids = output_ids[0]
    else:
        generated_ids = output_ids[0][input_len:]  # strip the echoed prompt

    truncated = generated_ids.shape[-1] >= max_new_tokens
    raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return CorrectionResult(model_key=model_key, raw_output=raw_output, truncated=truncated)
