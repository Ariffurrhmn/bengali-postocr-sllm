"""Shared pieces of the trained correctors (revision 2026-11, supervisor point 2):
model choice, chunking, batched greedy correction, aligned real OCR->reference pairs,
and the 25-character overlap filter."""
import json
import sys
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "eval"))

# Pinned Hugging Face revisions (resolved 2026-11-20 from the model pages; both still the
# head of main on 2026-11-21). licence, card and tokenizer were read from the model cards
# on 2026-11-21 and are copied into every run's environment record by model_info().
MODELS = {
    "byt5-small": {"id": "google/byt5-small", "revision": "68377bdc18a2ffec8a0533fef03b1c513a4dd49d",
                   "model_key": "byt5-small-ft", "license": "apache-2.0",
                   "card_url": "https://huggingface.co/google/byt5-small",
                   "tokenizer": "ByT5Tokenizer: raw UTF-8 bytes, 256 byte ids + 3 specials, no merges"},
    "banglabyt5": {"id": "Vacaspati/BanglaByT5", "revision": "945fac4100f5d3797498fe694af787bd9761efa0",
                   "model_key": "banglabyt5-ft", "license": "apache-2.0",
                   "card_url": "https://huggingface.co/Vacaspati/BanglaByT5",
                   "tokenizer": "byte-level BPE (tokenizer.json: 256 byte symbols + 23 merges, vocab 384), "
                                "config says ByT5Tokenizer; same T5 architecture as byt5-small, "
                                "continued pretraining on Bangla (Findings of EMNLP 2025, doi 10.18653/v1/2025.findings-emnlp.297)"},
}
NGRAM = 25
CHUNK_CHARS = 300


def load_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").split("\n") if l.strip()]


def read_ids(path):
    return [l.strip() for l in Path(path).read_text(encoding="utf-8").split("\n") if l.strip()]


def flat(text: str) -> str:
    """NFC, whitespace runs (newlines included) collapsed to one space."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def chunks_of(text: str, size: int = CHUNK_CHARS) -> list[str]:
    """Whole lines grouped into chunks of at most `size` characters; a longer line is
    split at spaces (or hard-cut if it has none). Each chunk is flattened to one line,
    as in training."""
    pieces = []
    for line in text.split("\n"):
        line = " ".join(line.split())
        while len(line) > size:
            cut = line.rfind(" ", 0, size)
            cut = cut if cut > size // 3 else size
            pieces.append(line[:cut].strip())
            line = line[cut:].strip()
        if line:
            pieces.append(line)
    out, cur = [], ""
    for p in pieces:
        if cur and len(cur) + 1 + len(p) > size:
            out.append(cur)
            cur = ""
        cur = (cur + " " + p).strip()
    if cur:
        out.append(cur)
    return out


def banned_ngrams(texts) -> set:
    banned = set()
    for t in texts:
        t = flat(t)
        banned.update(t[i:i + NGRAM] for i in range(max(1, len(t) - NGRAM + 1)))
    return banned


def overlaps(text: str, banned: set) -> bool:
    t = flat(text)
    return any(t[i:i + NGRAM] in banned for i in range(max(1, len(t) - NGRAM + 1)))


def aligned_pairs(ocr: str, ref: str, size: int = CHUNK_CHARS) -> list[tuple[str, str]]:
    """Cut a (OCR, reference) pair of one text block into aligned pieces of at most
    `size` reference characters, at reference spaces, mapping each cut to the OCR
    through a Levenshtein alignment (rapidfuzz opcodes)."""
    from rapidfuzz.distance import Levenshtein

    o, r = flat(ocr), flat(ref)
    if not r:
        return []
    if len(r) <= size:
        return [(o, r)]
    ops = Levenshtein.opcodes(r, o)

    def map_pos(i):
        for op in ops:
            if op.src_start <= i < op.src_end or (op.src_start == op.src_end == i):
                if op.tag == "equal":
                    return op.dest_start + (i - op.src_start)
                return op.dest_start
        return len(o)

    cuts, start = [0], 0
    while len(r) - start > size:
        c = r.rfind(" ", start + 1, start + size)
        c = c if c > start + size // 3 else start + size
        cuts.append(c)
        start = c
    cuts.append(len(r))
    out = []
    for a, b in zip(cuts, cuts[1:]):
        oa, ob = map_pos(a), (map_pos(b) if b < len(r) else len(o))
        out.append((o[oa:ob].strip(), r[a:b].strip()))
    return out


def load_model(name: str, tiny: bool = False, path=None):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5Config, T5ForConditionalGeneration

    spec = MODELS[name]
    if name == "banglabyt5":
        # BanglaByT5 ships its own byte-level BPE tokenizer (tokenizer.json: 256 byte
        # symbols, 23 merges, <s>=0 <pad>=1 </s>=2), declared as T5Tokenizer, which
        # Transformers 5 cannot rebuild from that file, so it is loaded as the fast
        # tokenizer it is, with the file's fixed 512 padding/truncation switched off.
        # End-of-sequence convention, checked on the released weights (2026-11-20, CPU):
        # given a sentence without any end token and decoder start 0, the model copies
        # it and then emits id 1 (the tokenizer's "<pad>") - the model config's eos 1;
        # with the tokenizer's "</s>" (id 2) appended to the input it emits a stray byte
        # first. So: inputs carry no end token, targets end with id 1, the model's own
        # config (pad 0, eos 1, decoder start 0) is kept, and label padding is -100.
        from transformers import PreTrainedTokenizerFast

        tok = PreTrainedTokenizerFast.from_pretrained(spec["id"], revision=spec["revision"])
        tok._tokenizer.no_padding()
        tok._tokenizer.no_truncation()
        tok.target_eos_id = 1
    else:
        tok = AutoTokenizer.from_pretrained(spec["id"], revision=spec["revision"])
        tok.target_eos_id = None  # ByT5Tokenizer already appends </s> (id 1)
    if path is not None:
        cls = T5ForConditionalGeneration if tiny else AutoModelForSeq2SeqLM
        model = cls.from_pretrained(path, dtype=torch.float32)
    elif tiny:
        model = T5ForConditionalGeneration(T5Config(
            vocab_size=384, d_model=32, d_ff=64, num_layers=1, num_decoder_layers=1, num_heads=2, d_kv=16,
            decoder_start_token_id=0, pad_token_id=0, eos_token_id=1))
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(spec["id"], revision=spec["revision"], dtype=torch.float32)
    return tok, model


def model_info(name: str, model=None) -> dict:
    """Licence, card, tokenizer and the exact Hugging Face revision of a model, for the run's
    environment record and train_log.json. The revision is resolved from the Hub at call time
    (a mismatch with the pin stops the run) and, when `model` was just loaded from the Hub,
    read back from the loaded config. Hub failures are recorded, not raised."""
    spec = MODELS[name]
    info = {k: spec[k] for k in ("id", "revision", "license", "card_url", "tokenizer")}
    info["pinned_revision"] = info.pop("revision")
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        mi = api.model_info(spec["id"], revision=spec["revision"])
        info["resolved_revision"] = mi.sha
        info["hub_license"] = (mi.card_data.to_dict().get("license") if mi.card_data else None)
        info["main_head_now"] = api.model_info(spec["id"]).sha
        assert mi.sha == spec["revision"], f"{spec['id']}: Hub resolves the pin to {mi.sha}"
    except AssertionError:
        raise
    except Exception as e:  # offline, rate limit, ...
        info["hub_lookup_error"] = f"{type(e).__name__}: {e}"
    if model is not None:
        info["loaded_commit_hash"] = getattr(model.config, "_commit_hash", None)
        if info["loaded_commit_hash"] and info["loaded_commit_hash"] != spec["revision"]:
            raise RuntimeError(f"{spec['id']}: loaded {info['loaded_commit_hash']}, pinned {spec['revision']}")
    return info


def make_labels(tok, texts, max_bytes):
    """Target ids with the model's end token, padded with -100 (never with a real id:
    BanglaByT5's tokenizer pad id 1 is the model's end token)."""
    import torch

    seqs = tok(list(texts), truncation=True, max_length=max_bytes - 1)["input_ids"]
    if tok.target_eos_id is not None:
        seqs = [s + [tok.target_eos_id] for s in seqs]
    n = max(len(s) for s in seqs)
    return torch.tensor([s + [-100] * (n - len(s)) for s in seqs])


def correct_texts(model, tok, texts, device, max_bytes=1024, batch=16):
    """Greedy correction of many texts: every text is split by chunks_of, all chunks are
    corrected in length-sorted batches, and each text's chunks are re-joined with "\\n".
    Returns (outputs, truncated flags)."""
    import torch

    items = []
    for ti, t in enumerate(texts):
        for ci, c in enumerate(chunks_of(t)):
            items.append((ti, ci, c))
    items.sort(key=lambda x: len(x[2].encode("utf-8")))
    res, trunc = {}, [False] * len(texts)
    for s in range(0, len(items), batch):
        part = items[s:s + batch]
        enc = tok([p[2] for p in part], return_tensors="pt", padding=True, truncation=True,
                  max_length=max_bytes).to(device)
        n_in = int(enc["attention_mask"].sum(1).max())
        cap = min(max_bytes, int(n_in * 1.3) + 16)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=cap, do_sample=False)
        eos = model.generation_config.eos_token_id
        for p, g in zip(part, gen):
            res[(p[0], p[1])] = tok.decode(g, skip_special_tokens=True)
            if not (g[1:] == eos).any():  # no end token: stopped at the length cap
                trunc[p[0]] = True
    outs = []
    for ti, t in enumerate(texts):
        n = len(chunks_of(t))
        outs.append("\n".join(res[(ti, ci)] for ci in range(n)) if n else "")
    return outs, trunc
