"""CER/WER scoring with Unicode NFC normalization, following the methodology's
Kanerva-et-al.-style protocol: normalize both reference and hypothesis to NFC
before scoring (Bengali conjuncts/matras can be represented with different
codepoint sequences that look identical but compare unequal without this)."""
import unicodedata

import jiwer


def normalize(text: str) -> str:
    """NFC-normalize and collapse incidental whitespace differences that
    aren't meaningful transcription differences (OCR/GT line-break variance)."""
    text = unicodedata.normalize("NFC", text)
    return " ".join(text.split())


def cer(reference: str, hypothesis: str) -> float:
    ref = normalize(reference)
    hyp = normalize(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return jiwer.cer(ref, hyp)


def wer(reference: str, hypothesis: str) -> float:
    ref = normalize(reference)
    hyp = normalize(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return jiwer.wer(ref, hyp)


def score_pair(reference: str, hypothesis: str) -> dict:
    return {"cer": cer(reference, hypothesis), "wer": wer(reference, hypothesis)}


def block_scores(references, hypotheses) -> dict:
    """Per-block scoring of one page (D78): the sum of the block edit distances divided by the
    sum of the block reference characters (words for WER). Blocks whose reference is empty
    after normalisation are skipped. A block with several lines is one reference string."""
    refs, hyps = [], []
    for r, h in zip(references, hypotheses):
        r, h = normalize(r), normalize(h)
        if r:
            refs.append(r)
            hyps.append(h)
    if not refs:
        return {"cer": 0.0, "wer": 0.0, "blocks": 0}
    return {"cer": float(jiwer.cer(refs, hyps)), "wer": float(jiwer.wer(refs, hyps)), "blocks": len(refs)}
