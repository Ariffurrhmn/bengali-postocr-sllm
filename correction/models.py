"""Zero-shot correction wrappers for the 5 models named in the methodology.

Two loading/inference paths, matching how each model actually works:
  - Causal LM + chat template (Phi-3 Mini, Llama 3.2 1B, Gemma 2B, TituLLMs 1B,
    Qwen3 1.7B):
    instruction-tuned, prompted via tokenizer.apply_chat_template.
  - Seq2seq (BanglaT5): not instruction-tuned, no chat template — prompted
    with a direct text-to-text framing instead.

Greedy decoding (do_sample=False) is used throughout: this is a correction
task, not creative generation, and determinism matters for reproducibility.
"""
import json
import os
from dataclasses import dataclass
from pathlib import Path


def _configure_cpu_threads():
    """PyTorch doesn't always auto-detect available cores correctly in
    containerized environments (observed: torch.get_num_threads()==1 on a
    Colab CPU runtime with more cores available, causing ~1 s/token
    generation and multi-minute-per-page runtimes). Explicitly set thread
    count to the actual CPU count Python sees."""
    import torch

    cpu_count = os.cpu_count() or 1
    if torch.get_num_threads() < cpu_count:
        torch.set_num_threads(cpu_count)


_configure_cpu_threads()

MODEL_IDS = {
    "phi3-mini": "microsoft/Phi-3-mini-4k-instruct",
    "llama3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "gemma-2b": "google/gemma-2b-it",
    "titullm-1b": "hishab/titulm-llama-3.2-1b-v1.1",
    "banglat5": "csebuetnlp/banglat5",
    # Added in revision (review item E-a): a 2025 small LM, ungated, fits
    # free Colab RAM in bf16.
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
    # Added in revision. Phi-4-mini (review item M5) supersedes Phi-3 Mini.
    # The TituLLMs 1B checkpoint above (v1.1) is a base model, not
    # instruction-tuned; the v2.0 Instruct checkpoints are (review item E-b).
    "phi4-mini": "microsoft/Phi-4-mini-instruct",
    "titullm-1b-instruct": "hishab/titulm-llama-3.2-1b-v2.0-Instruct-v1.0",
    "titullm-3b-instruct": "hishab/titulm-llama-3.2-3b-v2.0-Instruct-v1.0",
    # Third review (M1 size series; also the local pilot of the X1 attribution).
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    # Third review M1: Qwen3 size series, Gemma 3 and the larger Llama 3.2.
    "qwen3-4b": "Qwen/Qwen3-4B",
    "qwen3-8b": "Qwen/Qwen3-8B",
    "gemma3-1b": "google/gemma-3-1b-it",
    "gemma3-4b": "google/gemma-3-4b-it",
    "llama3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
    # Third review M7: one large model, 4-bit (POSTOCR_QUANT=4bit) across two T4s.
    "qwen3-32b": "Qwen/Qwen3-32B",
}

CAUSAL_LM_MODELS = {"phi3-mini", "llama3.2-1b", "gemma-2b", "titullm-1b", "qwen3-1.7b",
                    "phi4-mini", "titullm-1b-instruct", "titullm-3b-instruct",
                    "qwen3-0.6b", "qwen3-4b", "qwen3-8b", "gemma3-1b", "gemma3-4b",
                    "llama3.2-3b", "qwen3-32b"}

# Models above ~3B do not fit the T4's 15GB in float32, so on GPU they run
# in float16 (see load_model). Gemma overflows in float16, so Gemma 3 4B runs
# in float32 split over two GPUs instead (POSTOCR_DTYPE/POSTOCR_DEVICE_MAP).
# Qwen3 8B (16 GB in float16) also needs both GPUs.
GPU_FLOAT16_MODELS = {"phi3-mini", "phi4-mini", "titullm-3b-instruct", "qwen3-4b", "qwen3-8b",
                      "llama3.2-3b"}
SEQ2SEQ_MODELS = {"banglat5"}

# Qwen3 is a hybrid reasoning model; without this its chat template lets it
# open a <think> block before answering. Passed only to models that need it.
CHAT_TEMPLATE_KWARGS = {m: {"enable_thinking": False}
                        for m in ("qwen3-1.7b", "qwen3-0.6b", "qwen3-4b", "qwen3-8b", "qwen3-32b")}

CORRECTION_INSTRUCTION = (
    "The following text was produced by OCR on a historical Bengali document "
    "and may contain recognition errors. Correct the OCR errors and output "
    "only the corrected Bengali text, with no explanation, preamble, or "
    "additional commentary.\n\nOCR text:\n{ocr_text}"
)

# Prompt variants for the revision (review item M4). "en" is the original
# prompt; the others change one thing each.
PROMPTS = {
    "en": CORRECTION_INSTRUCTION,
    # The reviewer's wording: preserve historical spelling, leave it if unsure.
    "en_preserve": (
        "The following text was produced by OCR on a historical Bengali document "
        "and may contain recognition errors. Correct the OCR errors and output "
        "only the corrected Bengali text, with no explanation, preamble, or "
        "additional commentary. Preserve the historical spelling of the original "
        "text; if you are not sure whether something is an error, leave it "
        "unchanged.\n\nOCR text:\n{ocr_text}"
    ),
    # The original instruction in Bengali.
    "bn": (
        "নিচের লেখাটি একটি ঐতিহাসিক বাংলা নথি থেকে OCR দিয়ে তৈরি করা হয়েছে এবং "
        "এতে শনাক্তকরণের ভুল থাকতে পারে। OCR-এর ভুলগুলো সংশোধন করুন এবং শুধু "
        "সংশোধিত বাংলা লেখাটি দিন; কোনো ব্যাখ্যা, ভূমিকা বা বাড়তি মন্তব্য দেবেন না।"
        "\n\nOCR লেখা:\n{ocr_text}"
    ),
    # The original instruction plus three worked examples from development
    # pages (data/make_fewshot.py writes data/fewshot_examples.json).
    "fewshot": None,
    # Third review C2. "completion": a plain-text prompt that ends where the
    # corrected text should begin, for the base checkpoint (TituLLMs 1B v1.1),
    # given without any chat template. "normalized": for BanglaT5 only, the
    # OCR text passed through the csebuetnlp normaliser its model card asks for.
    "completion": (
        "The following text was produced by OCR on a historical Bengali document "
        "and may contain recognition errors. Below it is the same text with the "
        "OCR errors corrected.\n\nOCR text:\n{ocr_text}\n\nCorrected text:\n"
    ),
    "normalized": None,
}
FEWSHOT_PATH = Path(__file__).resolve().parent.parent / "data" / "fewshot_examples.json"


def build_prompt(prompt_key: str, ocr_text: str) -> str:
    if prompt_key != "fewshot":
        return PROMPTS[prompt_key].format(ocr_text=ocr_text)
    examples = json.loads(FEWSHOT_PATH.read_text(encoding="utf-8"))
    head, tail = CORRECTION_INSTRUCTION.split("\n\nOCR text:\n")
    shots = "".join(f"\n\nOCR text:\n{ex['ocr']}\nCorrected text:\n{ex['corrected']}"
                    for ex in examples)
    return (head + "\n\nExamples:" + shots
            + "\n\nNow correct the following text.\n\nOCR text:\n" + tail.format(ocr_text=ocr_text))


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

    # CPU: bfloat16 (not float32) halves memory, which matters on free-tier
    # Colab's ~12-13GB RAM (Phi-3 Mini at fp32 was getting killed partway
    # through weight loading); unlike fp16 it has no CPU stability issues.
    # GPU (revision runs, Colab T4): float32. The T4 has no native bfloat16,
    # and Gemma is known to overflow in float16; every model fits in the
    # T4's 15GB at float32. Weights load straight onto the GPU so the
    # float32 copy never has to fit in system RAM.
    # Models over ~3B use float16 on GPU, since float32 would not fit.
    if torch.cuda.is_available():
        dtype = torch.float16 if model_key in GPU_FLOAT16_MODELS else torch.float32
        device_map = "cuda"
    else:
        dtype, device_map = torch.bfloat16, None
    # Overrides for the third review (C1: rerun the fp16 models in fp32, split
    # over two T4s with device_map "auto"). Set through the environment so
    # every script that calls load_model honours them.
    if os.environ.get("POSTOCR_DTYPE"):
        dtype = {"fp32": torch.float32, "fp16": torch.float16,
                 "bf16": torch.bfloat16}[os.environ["POSTOCR_DTYPE"]]
    if os.environ.get("POSTOCR_DEVICE_MAP"):
        device_map = os.environ["POSTOCR_DEVICE_MAP"]

    # Third review M7: 4-bit NF4 weights (bitsandbytes), float16 compute, for
    # the large model split over two T4s. Set POSTOCR_QUANT=4bit.
    extra = {}
    if os.environ.get("POSTOCR_QUANT") == "4bit":
        from transformers import BitsAndBytesConfig

        extra["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
        dtype = torch.float16

    if model_key in SEQ2SEQ_MODELS:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id, token=hf_token, dtype=dtype, device_map=device_map, **extra
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, token=hf_token, dtype=dtype, device_map=device_map, **extra
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


def _make_progress_heartbeat(every_n: int = 10):
    """Returns a StoppingCriteria that never actually stops generation, but
    prints a heartbeat every `every_n` tokens — so a long single-page
    generation call shows visible progress instead of looking identical to
    a hang until it finishes (this looked like a stuck/frozen run in
    practice on Colab, when it was really just slow)."""
    import sys
    import time

    from transformers import StoppingCriteria

    class _ProgressHeartbeat(StoppingCriteria):
        def __init__(self):
            self.count = 0
            self.t0 = time.time()

        def __call__(self, input_ids, scores, **kwargs):
            self.count += 1
            if self.count % every_n == 0:
                elapsed = time.time() - self.t0
                print(
                    f"    ...{self.count} tokens generated ({elapsed:.0f}s, "
                    f"{elapsed / self.count:.2f}s/token)",
                    file=sys.stderr,
                )
            return False  # never stop early — progress reporter, not a real stopping condition

    return _ProgressHeartbeat()


# "guarded" is what the paper's runs used. In HF generate(), both guards also
# see the prompt tokens of decoder-only models, so they penalise copying the
# OCR text. "plain" drops them (repetition_penalty 1.0, no n-gram ban) and
# keeps only the max_new_tokens length guard: the decoding ablation.
DECODING_PRESETS = {
    "guarded": dict(repetition_penalty=1.3, no_repeat_ngram_size=4),
    "plain": dict(),
    # Third review C4: the two halves of "guarded" separately (2x2 ablation).
    "penalty_only": dict(repetition_penalty=1.3),
    "ngram_only": dict(no_repeat_ngram_size=4),
}


def correct_text(
    model_key: str,
    tokenizer,
    model,
    ocr_text: str,
    max_new_tokens: int | None = None,
    decoding: str = "guarded",
    prompt: str = "en",
) -> CorrectionResult:
    import torch

    if len(ocr_text.strip()) < MIN_OCR_CHARS_FOR_CORRECTION:
        return CorrectionResult(
            model_key=model_key, raw_output=ocr_text, truncated=False, skipped=True
        )

    if model_key in SEQ2SEQ_MODELS:
        # BanglaT5 is not instruction-tuned: no chat template, direct
        # text-to-text framing instead of a "please fix this" instruction.
        if prompt == "normalized":
            from normalizer import normalize as bn_normalize  # pip install git+https://github.com/csebuetnlp/normalizer
            ocr_text = bn_normalize(ocr_text)
        inputs = tokenizer(ocr_text, return_tensors="pt", truncation=True)
    elif prompt == "completion":
        # Plain text to continue, no chat template (base checkpoints).
        inputs = tokenizer(build_prompt(prompt, ocr_text), return_tensors="pt")
    else:
        messages = [{"role": "user", "content": build_prompt(prompt, ocr_text)}]
        # apply_chat_template(..., return_dict=True) returns a BatchEncoding
        # (dict-like, with attention_mask) rather than a bare tensor — needed
        # so **inputs below passes attention_mask to generate() too, not
        # just input_ids.
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            **CHAT_TEMPLATE_KWARGS.get(model_key, {}),
        )

    inputs = inputs.to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    if max_new_tokens is None:
        ocr_token_len = len(tokenizer.encode(ocr_text, add_special_tokens=False))
        max_new_tokens = min(
            MAX_NEW_TOKENS_CAP,
            max(MAX_NEW_TOKENS_FLOOR, int(ocr_token_len * MAX_NEW_TOKENS_MULTIPLIER)),
        )

    from transformers import StoppingCriteriaList

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy decoding: deterministic, appropriate
                               # for a correction task (not creative generation)
            # "guarded": greedy decoding on noisy/short OCR input is prone to
            # degenerate repetition loops (seen in practice on this task, e.g.
            # a phrase repeated until max_new_tokens is hit)
            **DECODING_PRESETS[decoding],
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            stopping_criteria=StoppingCriteriaList([_make_progress_heartbeat(every_n=10)]),
        )

    if model_key in SEQ2SEQ_MODELS:
        generated_ids = output_ids[0]
    else:
        generated_ids = output_ids[0][input_len:]  # strip the echoed prompt

    truncated = generated_ids.shape[-1] >= max_new_tokens
    raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return CorrectionResult(model_key=model_key, raw_output=raw_output, truncated=truncated)
