"""Hand-written test cases for eval/metrics.py — run before trusting CER/WER
on real OCR data. Covers: identical text, known edits, NFC normalization of
Bengali conjuncts, and empty-reference edge cases."""
from metrics import cer, normalize, wer


def test_identical_text_zero_error():
    text = "উত্তর মীমাংসা ।"
    assert cer(text, text) == 0.0
    assert wer(text, text) == 0.0


def test_known_single_char_substitution():
    # "।" (Bengali danda) replaced with "|" (pipe) — 1 char edit in a
    # 16-character reference (including spaces) — this mirrors a real
    # Tesseract failure mode seen in the dry run (danda -> pipe).
    ref = "উত্তর মীমাংসা ।"
    hyp = "উত্তর মীমাংসা |"
    result = cer(ref, hyp)
    assert result > 0.0, "substitution should register as nonzero CER"
    assert result < 0.15, f"single-char sub on a short string should be a small CER, got {result}"


def test_nfc_normalization_matches_decomposed_form():
    # Same visible Bengali text, but one string uses the precomposed vowel
    # sign U+09CB (BENGALI VOWEL SIGN O) and the other uses its canonical
    # decomposition U+09C7 U+09BE (vowel sign E + AA length mark). Both
    # render identically but compare unequal without NFC normalization —
    # this is a real, narrow case (most Bengali marks, e.g. U+09C0, have NO
    # decomposition and are unaffected; verified by scanning the Unicode
    # Bengali block — only vowel signs O/AU and 3 nukta letters decompose).
    # Built from explicit codepoints, not literal source text, so the test
    # doesn't depend on how the editor/toolchain normalizes the source file.
    ka = chr(0x0995)
    vowel_o_precomposed = chr(0x09CB)
    vowel_e = chr(0x09C7)
    length_mark_aa = chr(0x09BE)
    composed = ka + vowel_o_precomposed
    decomposed = ka + vowel_e + length_mark_aa

    assert composed != decomposed, "test setup: forms should differ pre-normalization"
    assert normalize(composed) == normalize(decomposed), (
        "NFC normalization should make composed/decomposed forms compare equal"
    )
    assert cer(composed, decomposed) == 0.0, (
        "CER should be zero between NFC- and NFD-encoded versions of the same text"
    )


def test_empty_reference_nonempty_hypothesis_is_full_error():
    assert cer("", "কিছু লেখা") == 1.0
    assert wer("", "কিছু লেখা") == 1.0


def test_empty_reference_and_hypothesis_is_zero_error():
    assert cer("", "") == 0.0
    assert wer("", "") == 0.0


def test_completely_wrong_hypothesis_has_high_cer():
    ref = "প্রথমাধ্যায়ের প্রথম পাদ ।"
    hyp = "xyz"
    assert cer(ref, hyp) > 0.8


def test_whitespace_only_differences_are_ignored():
    ref = "উত্তর  মীমাংসা ।"  # double space
    hyp = "উত্তর মীমাংসা ।"  # single space
    assert cer(ref, hyp) == 0.0, "incidental whitespace differences should not count as errors"


if __name__ == "__main__":
    import sys

    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            print(f"FAIL {t.__name__}: {e}")
            failures += 1
    print()
    if failures:
        print(f"{failures}/{len(tests)} tests failed")
        sys.exit(1)
    print(f"All {len(tests)} tests passed")
