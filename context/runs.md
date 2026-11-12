# Runs: what was executed, where its output and log are

Every model and OCR run behind the paper, with its output file in `results/` (local, gitignored; a copy is in the Google Drive folder) and the log that survives. Written on 2026-11-13 from the result files, the logs and `decisions.md`. Where a log does not exist, this file says so. Nothing here was reconstructed.

"Used" means the paper reports numbers from it.

## OCR
| Run | Date, where | Output | Log | Used |
|---|---|---|---|---|
| Whole-page OCR, 10 development pages | 2026-07-25, local | `ocr_dev.jsonl` | none | dev only (noise model, few-shot examples, fine-tuning lines) |
| Whole-page OCR, 15-page first version | 2026-08-02, Colab CPU | `ocr_eval.jsonl` | none | CPU timing and early decoding comparison |
| Whole-page OCR, 40 evaluation pages (main) | 2026-10-11, Colab CPU | `ocr_eval40.jsonl` | none; versions in ENVIRONMENT.md | yes, input of every main run |
| OCR with word confidences | 2026-10-12, Colab CPU | `ocr_conf40.jsonl` | none | yes (gating, fusion) |
| Region-level OCR | 2026-10-16/17, Kaggle CPU, Tesseract part redone on Colab | `region/ocr_region40*.jsonl` | `region/run.log`, `region/setup.log` | yes |

## Correction with language models
| Run | Date, where | Output | Log | Used |
|---|---|---|---|---|
| First version: 4 models, guarded decoding, 15 pages | 2026-07-27/28, Colab CPU | `correction_eval.jsonl` (120) | none (one disconnect at 84/120, resumed) | decoding comparison only |
| Plain-decoding reruns, 15 pages | 2026-10-09/10, Colab CPU | `ablation_plain_decoding.jsonl` (150) | none | CPU timing, decoding comparison |
| Main run: 8 models x 40 pages x Tesseract, EasyOCR, clean text | 2026-10-11, Kaggle T4 | `gpu_eval40.jsonl` (960) | `gpu_sweep40.log` | yes |
| Prompt and chunking variants | 2026-10-12, Kaggle T4 | `gpu_variants40.jsonl` (1,280) | `gpu_variants40.log` | yes |
| review3: float32 checks, completion prompt, BanglaT5 normaliser, 2x2 decoding ablation, Grad-CAM | 2026-10-16, Kaggle 2x T4 | `review3/` | `review3/setup.log`, `phase1.log`, `lane0.log`, `lane1.log`, `environment.txt` | yes |
| review3b: six more models (size series), synthetic noise sweep, more Grad-CAM | 2026-10-17, Kaggle 2x T4 | `review3b/` | `review3b/*.log`, `environment.txt` | yes |
| review3c: all models on region-level text | 2026-10-17, Kaggle 2x T4 | `review3c/` | `review3c/*.log`, `environment.txt` | yes |
| review3d: Qwen3 32B 4-bit on 10 pages, Qwen3 4B on region text, first fine-tuning run | 2026-10-18, Kaggle 2x T4 | `review3d/` | `review3d/*.log`, `environment.txt` | yes, except the fine-tuning run (see below) |
| review3e: fine-tuning rerun (ByT5-small) | 2026-10-18, Kaggle T4 | `review3e/m2_byt5.jsonl` | `review3e/lane0.log` (loss and dev CER per step), `setup.log`, `environment.txt` | yes |
| review4: constraints on generated tokens only, completion prompt for the Instruct checkpoints | 2026-10-19, Kaggle 2x T4 | `review4/` | `review4/*.log`, `environment.txt` | yes |
| Prompt-constraint check (Qwen3 0.6B, Transformers 5.19.0) | 2026-10-19, local CPU | `prompt_constraints_5.19.0.json` | none | yes (text only) |
| Grad-CAM pilot (Qwen3 0.6B) | 2026-10-15, local CPU | `xai_pilot_qwen3-0.6b.jsonl` | none | no, pipeline test |

The Kaggle `environment.txt` files record the repository commit, Python, PyTorch, CUDA, Transformers and the GPUs of each session.

## Without a language model
| Run | Date, where | Output | Log | Used |
|---|---|---|---|---|
| SymSpell, subtitle lexicon | 2026-10-11, local | `dict_eval40.jsonl` | none then; rerun 2026-11-13 with log, identical | yes |
| SymSpell variants (Wikipedia lexicon, edit distance 1) | 2026-10-18, local | `m4_dict_*.jsonl` | `m4_lexicon_build.log`; rerun 2026-11-13 with log, identical | yes |
| Engine fusion | 2026-10-18, local | `m6_fusion40.jsonl` | none then; rerun 2026-11-13 with log, identical | yes |

## Analyses, tables and figures
Rerun on 2026-11-13 with `paper/scripts/reproduce_tables.sh` under `tools/run_logged.py` (`logs/runs/*reproduce-tables`). All 20 summary files came out byte-identical to the ones in `summaries/`, and every generated table in `paper/tables/` came out unchanged.

## Failed or repeated runs
- Phi-3 Mini was dropped before the main run because it was too slow on CPU (D20). It was never evaluated.
- The first review3 session died after one minute because `HF_TOKEN` was not attached, and was started again.
- The first fine-tuning run (review3d) was invalid: the test chunks had line breaks the model never saw in training. It was fixed and rerun as review3e (D50). The paper uses review3e. The review3d numbers are kept in `summaries/` only for the record.
- Tesseract 5.3.4 no longer gives the text of the 11 October run (D44), so the main OCR file is kept as data and not regenerated.
- The August CPU sweep disconnected at 84/120 and resumed from its output file.

## What is missing
- Console logs of the Colab OCR runs and of the July/August first-version runs. They were never saved.
- The fine-tuned ByT5-small checkpoint stayed in the Kaggle session output and was not downloaded.
- The notebooks in `notebooks/` were committed without their cell outputs. The executed versions may still be on Kaggle and Colab.
