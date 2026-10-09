# Open items

## Post-submission feedback (Sep 2026)

Reviewed 24 Aug 2026; verdict "not acceptable at current stage". All 50 items are transcribed in `reviewer-feedback/feedback.md`. Related work:
- `reviewer-feedback/published-references.md`: published versions of the circled preprints (6 of 8 published).
- `reviewer-feedback/candidate-references.md` and `paper-reviews.md`: 14 candidate papers read in full, with cite/redo verdicts.
- `results/extended_eval.txt`: new analyses answering the stats/validation/XAI comments (D27).
- **Still to do (user):** ask the reviewer what "XAI", "validation methods" and "not right claim" mean; fix the references; rewrite the paper.

**Draft errors found:** Section IV-A misreads bbOCR (bbOCR never tested EasyOCR, so our ordering agrees with it rather than contradicting it); Beshirov is vol. 26 no. 1 art. 4, not no. 4; Levchenko is missing pages 75–85 and the location.

## Unfinished work (from the paper's own limitations)

- **Alignment-based overgeneration filter** (D23). Specified, not built. Kanerva's method: character-level local alignment (biopython `PairwiseAligner`) to strip leading and trailing text that doesn't align with the input. Would show how much of the damage is preamble or commentary rather than real rewriting. Needs no new model runs: it can be applied to the existing `correction_eval.jsonl`.
- **Remaining 25 eval pages.** Only 15 of the 40 were run. It would cost roughly 24 h × 25/15 ≈ 40 h of Colab CPU for the chunked approach.
- **Phi-3 Mini** (D20). Never completed.
- **Whole-page approach on eval.** The code supports it; it only ran on dev. Dev suggested whole-page may do less damage for TituLLMs (D19).
- **Prompt variation.** One fixed prompt only (D10).
- **Future directions named in the paper's conclusion:** constrained or edit-distance-bounded decoding; line- or word-level correction; supervised fine-tuning (the approach that worked for Bulgarian in Beshirov et al.); simple guards on output length and Bengali-character share.

## Known inconsistencies and loose ends

- **Proposal vs paper hypothesis.** The proposal states ≥10% CER reduction; methodology dropped it (D03). Make sure the final paper doesn't reintroduce it.
- **Zenodo DOI.** The proposal promised to archive splits and code on Zenodo with a DOI. The paper only cites GitHub.
- **Table III layout.** In the submitted PDF, the rightmost column ("Tr", truncation %) is cut off at the page edge.
- **Paper source missing.** The paper was written in Overleaf. The export zip (`bengali-postocr-overleaf.zip`, once in the repo root) and the PDF in Downloads are both gone. The only local copy found is `Draft Paper/222_CSE403_Draft_Paper.pdf` in the Drive folder. The Overleaf project itself should still be online.
- **README references `Draft Paper/Methodology_PostOCR.docx`.** That file is no longer in the Drive "Draft Paper" folder. There is also an old mojibake character (�) in "British Library's" in that docx.
- **Dev baseline numbers differ between runs.** Tesseract dev CER was 0.371 in the first local scoring and ~0.350 in the Colab sweep scoring. Probably a different Tesseract build (local 5.4.0 against Colab's apt package); not verified. Dev numbers are never reported, so this is low priority.
- **Result files aren't versioned** (D26). The eval outputs exist only in local `results/` and on Drive. Consider committing the small summary files (`bootstrap_eval.json`, `figures/evaluation_matrix.csv`), or archiving the full outputs with the Zenodo deposit.
