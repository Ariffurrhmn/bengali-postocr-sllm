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
- **Bengali console output crashes on Windows.** Re-wrap `sys.stdout` as UTF-8 (see SETUP.md).
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
