# Engineering notes

Things that went wrong or will go wrong again. Each entry has the symptom, the cause, and the fix or rule going forward.

## Bugs fixed

1. **Chat-template input missing attention_mask** (`0ccc144`). `apply_chat_template(..., return_tensors="pt")` returns a bare tensor. The code re-wrapped it as `{"input_ids": ...}`, which crashed and silently dropped `attention_mask`. Fix: pass `return_dict=True`.
2. **Nested clone in Colab** (`cf8644b`). Re-running the clone cell created `bengali-postocr-sllm/bengali-postocr-sllm/`, so relative `%cd` chains wrote results one level off. Fix: clone only if missing (otherwise `git pull`), and use absolute paths from `REPO_DIR` everywhere.
3. **Degenerate repetition loop** (`0fad60c`). Greedy decoding repeated one phrase until the cap. Fix: `repetition_penalty=1.3`, `no_repeat_ngram_size=4`.
4. **Phi-3 killed for lack of memory while loading** (`47f9111`). fp32 weights didn't fit in free Colab RAM. Fix: bf16 for all models.
5. **Flat 512-token cap** (`cc978bf`). Made pages slow. Fix: cap scales with input length. This was a partial cause only; see 6.
6. **PyTorch on 1 thread** (`e79cbec`). The real cause of the ~1 s/token slowness on Colab. Fix: `torch.set_num_threads(os.cpu_count())`. The thread count is printed at sweep start.
7. **Case-mismatched filenames** (`5934a74`). `279_2_a_15_0005.tif` sits beside `279_2_A_15_0005.xml`. Windows hid the mismatch; Linux Colab crashed. Fix: `resolve_file()` in `ocr/run_ocr.py` tries the exact path, then a case-insensitive scan.

## False alarm worth remembering

The sweep only printed a line *after* each page finished, so a 7-minute page looked like a hang. Llama was killed at 18–30 min "stuck" when 6 pages had in fact completed. Fix: a per-10-token heartbeat plus a "starting page …" line (`4c181d3`). Before assuming a hang, check the output JSONL.

## Environment gotchas

- **Windows hides case bugs.** `Path.exists()` is case-insensitive on Windows. Test filename logic with a deliberately case-sensitive check.
- **Bengali console output crashes on Windows.** Re-wrap `sys.stdout` as UTF-8 (`sys.stdout.reconfigure(encoding="utf-8")`), once per process.
- **Bengali literals in source files can get silently normalised** by editors or tooling before Python sees them. For exact-codepoint test data, build strings with `chr(0x…)`.
- **NFC in Bengali:** only vowel signs O and AU and the three nukta letters decompose. ী (U+09C0) does not, so don't use it as an NFC test case.
- **Tesseract Bengali langpack:** not bundled. It is kept in the gitignored `.tessdata/` with `TESSDATA_PREFIX`; Colab downloads it in the notebook.
- **Local disk:** C: had under 1 GB free, so models can't be downloaded locally. Test model code with `gpt2` if necessary.
- **Gated models:** Llama 3.2 and Gemma need an accepted HF licence and a read token. On Colab the token comes from the `HF_TOKEN` secret. Never paste tokens into notebooks or chat.
- **`huggingface-cli` is deprecated**; use `hf auth login`.
- **transformers 5.x:** use `dtype=`, not `torch_dtype=`.

## Colab workflow

- The Colab VM is ephemeral. The notebook re-clones or pulls, reinstalls dependencies, downloads tessdata, mounts Drive and unzips `My Drive/Dataset/REID2019.zip` every session.
- Results go to `My Drive/bengali-postocr-results/` so they survive a runtime reset.
- **Colab's "Save a copy in GitHub" commits directly to `master`** (with cell outputs). Always `git pull` locally before pushing, and `git pull` inside Colab after pushing a fix.
- Free tier: ~90-min idle disconnect and session caps. Plan for resume, not for an unattended overnight run.

## Revision (October 2026)

8. **Repetition guards act on the prompt.** In Transformers `generate`, `repetition_penalty` and `no_repeat_ngram_size` see the whole sequence, which for decoder-only models includes the prompt. For a copy-heavy task like correction they penalise copying the input. Rule: never use them for correction; if a loop guard is needed, rely on the length cap.
9. **Hard-coded page counts.** `extended_analysis.py` split difficulty bands as pages [:5], [5:10], [10:], correct only for 15 pages. Fixed to thirds of the page count. Rule: derive counts from the data; grep for literal 15/16 when the sample size changes.
10. **OCR is not reproducible across months.** Colab's apt Tesseract changed between August and October 2026; re-OCR of the same 15 pages changed the Tesseract text on all of them (EasyOCR almost unchanged). Rule: record engine versions in every OCR file; never mix OCR runs from different dates.
11. **Model size vs T4 memory.** Models above ~3B do not fit the T4's 15 GB in float32; they run in float16 (`GPU_FLOAT16_MODELS`). Gemma is known to overflow in float16, so it stays in float32.

12. **U+2028/U+2029 inside JSONL** (2026-10-16). Qwen3 under guarded decoding writes Unicode line/paragraph separators; `str.splitlines()` breaks the JSON line there. Rule: read JSONL with `.split("
")`.
13. **Kaggle secrets are per notebook.** A newly imported notebook does not see `HF_TOKEN` until it is ticked under Add-ons → Secrets; the first review3 run died after one minute for this reason.
14. **fp32 across two T4s is slow.** Phase 1 (two 3-4B models in fp32 with device_map auto) used its whole 3.5 h cap. Budget about 2x the single-GPU fp16 time.

## Environment gotchas (added)

- **Local disk C: is nearly full.** Hugging Face downloads default to C:; set `HF_HUB_CACHE=G:/hf-cache`. Google Drive streams files by caching them on C:, so reading a large Drive file can fail with "No space left on device".
- **Local RAM (16 GB) cannot load Gemma 2B;** run models on Colab/Kaggle.
- **Kaggle background runs:** use Save Version → Save & Run All; the notebook stops starting new pages after 11 h so the output is saved. Kaggle's Download button may only show the file in a browser tab; copying that text into a file worked and was validated (line count, JSON parse, no U+FFFD, size).
- **Bash heredocs collapse double backslashes** in this Windows setup, breaking Python or LaTeX edit scripts. Write such scripts to a file first.
- **BanglaT5 tokenizer** needs protobuf for the fast conversion in Transformers 5; `eval/model_profile.py` reads its SentencePiece model directly.
- **Local Tesseract** (scoop, 5.5.3) sets its own `TESSDATA_PREFIX` without Bengali; `run_ocr.py` only uses `setdefault`, so pass `TESSDATA_PREFIX=G:/bengali-postocr-sllm/.tessdata TESSERACT_CMD=tesseract` explicitly. 5.5.3 is not the engine of the main run (5.3.4): use it for plumbing tests only.
- **Tesseract on small region crops** (2026-10-16): the whole-page default page-segmentation mode 3 returns nothing for many short heading crops; `run_region_ocr.py` takes `--psm` ; a local 8-page test chose --psm 6 with plain bounding-box crops (see D41).
15. **Tesseract output drifted within one version** (2026-10-17, D44). Colab apt and Kaggle conda-forge Tesseract 5.3.4 give identical text today, but not the text of the 11 October main run (CER 0.22 between them), which Colab still reproduced on 12 October. Same model file, same image decoding, same dpi. Rule: never compare OCR from different days; always re-OCR whole pages in the same run as any new OCR setting (run_region_ocr.py --verify does this).
