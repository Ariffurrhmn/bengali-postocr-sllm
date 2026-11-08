# Google AI review: focused manuscript revision

The author's pasted single-file manuscript contains 15 prose revisions beyond the repository version. All were merged into `paper/main.tex` and preserved while applying the reporting corrections below. No experimental outputs or summary statistics were changed, and no new model runs were performed.

## Addressed

- Abstract specifies code-point CER, Qwen3 4B/8B, and the 0.004 absolute / approximately 1% relative gating gain. Conclusion specifies the same model sizes for the edit-level evidence.
- Fusion CER and WER are both descriptive; significance superscripts removed by the table generator. Historical post-hoc family selection remains disclosed in the narrative.
- German task-specific fine-tuning is separated from zero-shot work. Its published PDF reports no average CER reduction and a best 22% WER reduction (ACL Anthology 2025.resourceful-1.26, section 5.1).
- Added published BanglaByT5 and IndicBART context, explicitly unevaluated here.
- Corrected dataset context: the British Library's 155-image collection includes the 81 REID2019 images and 74 Transkribus images. It is not 155 additional held-out pages. No unsupported 2021 publication date was added.
- Checked split identifiers using the existing `book_of` rule: 10 development books; one identifier, `14048_D_11`, overlaps evaluation. Page split and implications for the supervised pilot are now explicit.
- Clarified whitespace word tokens and punctuation, multiset recall, integer generation cap, contiguous training-text exclusion, insertion ownership, word/character alignment reconciliation, confidence gating blocks and neighbours, fusion ranks and tie handling, Grad-CAM ReLU/normalisation/layer rounding/token alignment/chance approximation.
- Model objective and instruction-tuning confounds distinguished from the effect of Bengali pretraining; pilot undertraining and unigram-noise confounds stated.
- 32B subset remains descriptive. Removed the blanket mathematical impossibility claim: Holm is a step-down procedure, so multiplying a minimum p-value by the entire family is not a universal lower bound for all ranked hypotheses. No reported result or family was changed.
- CPU/GPU check explicitly cannot isolate device from precision effects.
- Added missing wMER to the method diagram; clarified filter segmentation caveat and limitations of lexicon absence as a proxy for historical spelling.
- Bibliography: corrected BanglaT5 pages to 726–735, verified against ACL Anthology 2023.findings-eacl.54. The AI review's proposed 583–596 is incorrect. Added REID2019 pages 1527–1532, verified against PRImA's publication record.

## Still requires detailed saved outputs

`results/` in this checkout contains only `.gitkeep`. `summaries/review4_runs.json` is also absent. The earlier project log D64 records completion-prompt Tesseract CER 1.642 → 1.336 (Holm p=0.014) for TituLLMs 1B Instruct and 0.958 → 0.960 (not significant) for 3B Instruct. These findings are now reported in Appendix precision/prompt-format prose, where readers were previously referred to an incomplete table. No confidence intervals were invented. Adding full rows to generated `checks.tex` requires restoring `review4_runs.json` and extending its generator; the existing caption's delta range describes only the checks actually tabulated.

## Deferred experiments and editorial scope

Full training schedules/multiple seeds, 32B on 40 pages, additional baselines, held-out gating/fusion/crop validation, complete CPU timings, controlled precision/device reruns, containerised OCR regeneration, and orthography stratification remain future work. Venue-specific shortening waits for venue selection. No test family was reduced to obtain significance.

## Primary sources checked

- https://aclanthology.org/2025.findings-emnlp.297/
- https://aclanthology.org/2022.findings-acl.145/
- https://aclanthology.org/2025.resourceful-1.26.pdf
- https://aclanthology.org/2023.findings-eacl.54/
- https://bl.iro.bl.uk/entities/digitalcollection/dbce3eb4-d8d3-4e84-a26f-cd5636d16651
- https://schema.primaresearch.org/publications/ICDAR2019_Clausner_REID2019

## Verification

Final Tectonic build succeeds (31 pages), with no overfull boxes, missing-character warnings, or unresolved references. All 62 citation keys and all cross-references resolve. Rendered all pages as contact sheets and inspected the abstract, attribution equations, fusion table and prompt-check appendix at readable resolution; no clipping or overlaps found. Regenerated `main-single.tex` and `overleaf-final.zip`. Dictionary/fusion data rows retain their original numerical values; both fusion significance superscripts are removed. The summary-only generator and figure generator pass Python compilation.
