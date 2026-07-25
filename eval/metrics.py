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
