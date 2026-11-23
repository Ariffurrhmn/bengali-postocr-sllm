# Claim-to-evidence check, 23 November 2026

This file lists the main claims of the abstract, introduction and conclusion, and where each one comes from. Every held-out number was checked line by line against the result file named here. The 40-page numbers were checked number by number in October; this round regenerated all their tables from the raw results (identical, byte for byte), compared the three data figures against their tables, and confirmed that every number in the prose appears in a table or result file.

| # | Claim (as in the paper) | Where | Evidence | Source file | Checked | Caveat in paper |
|---|---|---|---|---|---|---|
| 1 | No zero-shot method lowered per-block CER on the 41 held-out pages | Abstract, Results, RQ1, Conclusion | Table VIII (test_zeroshot) | `results/rev08_score/summary.txt` | yes | block-level input, not whole pages |
| 2 | CER rose significantly for 17 of 18 method-and-engine pairs | Abstract, Results, RQ1 | Table VIII | same | yes (Gemma-2B-it on Tesseract is the exception, Holm 0.093) | |
| 3 | 29 of 36 tests significant, 27 increases | claims table, D82 | Table VIII | same | yes | |
| 4 | Gemma-2B-it and Qwen3 1.7B lowered WER on Tesseract text | Abstract, Results, RQ1, Conclusion | Table VIII | same | yes: 0.742 to 0.692 (Holm 0.012) and 0.722 (Holm 0.023) | Qwen3 interval includes zero; now said in the text |
| 5 | Both trained correctors raised CER on Tesseract text (+0.105, +0.096), not significant on EasyOCR (+0.057, +0.019) | Abstract, Results, RQ4 | Table IX (test_trained) | `results/rev07_holm.json`, `rev07_score/*/summary.txt` | yes | |
| 6 | Both lowered EasyOCR WER (−0.016, −0.065) | Abstract, Results, RQ4, Conclusion | Table IX | same | yes (Holm 0.040 and <0.001) | ByT5-small interval includes zero |
| 7 | Validation CER: best 0.330 and 0.324 against raw 0.296 | Results | text only | `results/bengali-rev-03-train-both/train/*/train_log.json`, D76 | yes (0.3299, 0.3237, 0.2956) | |
| 8 | 1,185 real pairs, 69,822 synthetic, 81,672 in all; 60,062 Wikipedia segments, 976 lines | Methods | text only | `train_log.json` | yes | |
| 9 | Guarded variant within 0.004 of raw OCR | Results | Table IX | rev07 summaries | yes (+0.002, +0.004, −0.001, −0.001) | post-hoc, secondary |
| 10 | Gating gain of the 40 pages did not recur; no gated method lowered CER significantly | Abstract, Results, RQ3, Conclusion | Table X (test_gating) | `results/rev09_gated/summary.txt` | yes (Qwen3 Tesseract 70: +0.001 [−0.002, +0.005]) | added after zero-shot scores were known |
| 11 | 40 of 54 gating tests significant, more pages worse in each | Results | Table X | same | yes | one cell has a negative mean (now in the text) |
| 12 | Gated output always below ungated output | Results | Table X | same | yes (all 54 cells) | |
| 13 | 1,250 units = 1,213 kept blocks + 37 added; 9 of 1,222 blocks not scored | Methods | text only | `results/bl_review/reviewed_units.json`, review file | yes (was wrong: said six) | |
| 14 | 41 test pages, 36 groups, 45,080 characters; 62/37 train, 14/9 validation | Methods | Table (test_split) | `data/split_v2/split_manifest.csv` | yes (regenerated) | |
| 15 | No ungated model significantly lowers CER on the 40 pages; 23 of 36 tests significant, 22 increases | Results, RQ1 | Table XI (eval_matrix), Fig. 3 | `summaries/bootstrap_gpu40.json`, `extended_gpu40.json` | yes (recounted; figure matches table) | exploratory since the held-out test |
| 16 | Qwen3 1.7B gating at 70 on the 40 pages: 0.341 to 0.337, about 1% | Abstract, RQ1 | Table (gated) | `summaries/gated_gpu40.json` | yes | exploratory |
| 17 | Repetition constraints made every general model worse (33 of 33) | Abstract, RQ3 | Table (decoding), Fig. (decoding) | `summaries/decoding_ablation.json`, `review4_runs.json` | yes (figure matches table; per-constraint numbers re-derived) | |
| 18 | Synthetic noise: every model worse at every level down to 0.020 | Results, Conclusion | Table (noise), Fig. (noise) | `summaries/noise_runs40.json` | yes (figure matches table) | |
| 19 | Every method changed error-free text | Abstract, RQ2, Conclusion | Tables VIII, IX notes; Table (clean) | rev08, rev07 summaries; `summaries/main_general40.json` | yes | held-out counts of zero-shot and trained models not directly comparable (now said) |

## Errors found and fixed in this round

1. The paper said six boxes were removed from the test pages. Nine of the 1,222 blocks are not scored: seven flagged as non-text or empty, and two with no transcribed text. Fixed in the paper and noted under D77.
2. The paper said units under 10 characters pass through unchanged for every method. Only the zero-shot language models skip them; the dictionary and the trained correctors process every block. Fixed in the method and limitations text.
3. The paper said the trained correctors changed more error-free text than Gemma-2B-it, Phi-4-mini and Qwen3 1.7B. The counts are not comparable (different block coverage, different averaging), and the per-block clean outputs of the trained models were not saved, so no fair comparison can be made now. The claim is removed and the table notes say why.
4. RQ4 said the trained correctors' CER was higher than that of "the copying zero-shot models". On EasyOCR text BanglaByT5 (0.2907) and Phi-4-mini (0.2906) are level. Now it names Gemma-2B-it and Qwen3 1.7B, which holds on both engines.
5. Two sentences said gating never brought a model below the raw OCR. Gemma-2B-it at Tesseract 70 is 0.347 against 0.350 (not significant). Now "not significantly below".
6. The gating table note pasted the raw summary lines. Rewritten from the same numbers.
7. The abstract's "17 of the 18 zero-shot tests" now says method-and-engine pairs, so it cannot be read as all tests.

## Figures

- Fig. 3 (change in CER, 40 pages): all 18 points and intervals match Table XI.
- Noise figure: all 20 points match the noise table.
- Decoding figure: all points match the decoding table.
- Method figure: shows only the 40-page pipeline. The caption now says so and lists how the held-out run differs. Redrawing the figure itself is an option, not done.
- The dataset, per-page, scatter and attribution figures were not changed and were checked in October.

## Left for the author

- **Title.** It is still a question. The skill and `paper/WRITING-GUIDE.md` both say a title should state the finding; it is your and Bijon Sir's decision. Candidates:
  - *Small Language Models Do Not Correct Historical Bengali OCR: A Book-Disjoint Test of Eight Zero-Shot Models and Two Trained Correctors*
  - *Leave the OCR Alone: Neither Zero-Shot nor Trained Small Models Lower Character Errors in Historical Bengali Print*
- **Read the draft.** Ask any human reader to mark where they got lost, not typos.
- The whole-page zero-shot run on the test pages (Kaggle 04, 20 November) is still unscored and unused.
- The Bengali examples table needs you to judge which changes are OCR errors and which are valid old spellings.
- Upload the new single-file version to Overleaf before editing there.
