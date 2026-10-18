# I am checking the novelty claims and related work of a paper before submission.

**Question:** I am checking the novelty claims and related work of a paper before submission. The paper: zero-shot post-OCR correction of historical Bengali printed books (1713-1914) with small open language models (0.25-8B parameters, plus one 32B model), Tesseract and EasyOCR output, character error rate (CER) as the main metric. Findings: no model lowers CER; general models copy their input, Bengali-pretrained models regenerate it; every method damages error-free text; synthetic noise down to CER 0.02 still gets worse; larger models fix more words (WER falls) but not CER.

Please answer three parts:

Part 1 (most important). Has any published paper reported that repetition penalty (e.g. Keskar et al. CTRL-style penalty, Hugging Face RepetitionPenaltyLogitsProcessor) or no-repeat n-gram constraints applied during decoding also penalise tokens from the prompt/input, and that this harms tasks that require copying the input (post-OCR correction, grammatical error correction, text normalisation, paraphrase, summarisation with extraction, machine translation of copyable spans, code editing)? Also look for papers that report repetition penalty or n-gram blocking degrading copy-heavy or edit-style generation tasks in general. List each relevant paper with what exactly it shows, and say clearly whether the specific point (penalty acting on prompt tokens in decoder-only models misleads evaluation of correction) appears to be already published.

Part 2. Find work since 2023 on language-model or neural post-OCR correction for Bengali/Bangla or other Indic scripts (Hindi, Sanskrit, Tamil, etc.), especially historical documents, and any work that evaluates whether LLM correction damages already-correct text (over-correction) in OCR or grammatical error correction. Highlight anything evaluating zero-shot small models.

Part 3. Find studies of how post-OCR correction benefit depends on input error rate (break-even CER at which correction stops helping), and studies of fine-tuning byte-level or sequence-to-sequence models (ByT5, mT5, BART) on synthetic OCR noise and how well that transfers to real OCR errors.

For every paper give authors, year, venue, and the specific finding with a quote. Be explicit when evidence is absent.

# The **prompt-penalty novelty**, **Indic post-OCR** evidence, and **error-rate transfer** literature are only partially covered

The evidence splits into three facets: whether **decoding penalties** already explain copy-task failures, what **recent Indic and historical post-OCR** work shows about LMs and over-correction, and how **input error rate and synthetic-noise training** affect post-OCR benefit.

## Part 1

The specific claim that a repetition penalty or no-repeat n-gram constraint in **decoder-only generation penalizes prompt/input tokens** and therefore confounds evaluation on correction tasks does **not appear to be explicitly published in the 20 supplied papers**. The decoding papers here study repetition in open-ended generation, summarization, and translation, and some show quality loss from over-penalizing repetition, but none explicitly state that prompt tokens are penalized during decoding and that this harms copy-heavy correction evaluation [[1.1]](https://consensus.app/papers/on-decoding-strategies-for-neural-text-generators-wiher-meister/44ba459c14fd510d9c10980e26d748cc/) [[2.1]](https://consensus.app/papers/rap-a-metric-for-balancing-repetition-and-performance-in-huang-nguyen/fc3ce0744b105a2884024be6ba4d72b0/) [[3.1]](https://consensus.app/papers/penalty-decoding-well-suppress-the-selfreinforcement-zhu-hao/bc39636d496957c683f80216d6a5b7d5/).

- **Jin Xu, Xiaojiang Liu, Jianhao Yan, Deng Cai, Huayang Li, Jian Li (2022, arXiv)** analyze repetition as a self-reinforcement effect: “the more times a sentence is repeated in the context, the higher the probability of continuing to generate that sentence” and propose DITTO to penalize sentence-level repetitions during training, but they study open-ended generation and summarization, not copy-heavy editing tasks [[4.1]](https://consensus.app/papers/learning-to-break-the-loop-analyzing-and-mitigating-xu-liu/e6317d1711625ed3b2eac61cf1a66ea7/) [[4.2]](https://consensus.app/papers/learning-to-break-the-loop-analyzing-and-mitigating-xu-liu/e6317d1711625ed3b2eac61cf1a66ea7/).
- **Gian Wiher, Clara Meister, Ryan Cotterell (2022, TACL)** show that decoding behavior is **task-specific**: “the properties elicited by various decoding strategies do not always transfer across natural language generation tasks,” which supports the general idea that anti-repetition settings can misfire outside open-ended generation, but they do not discuss prompt-token penalties or correction tasks [[1.2]](https://consensus.app/papers/on-decoding-strategies-for-neural-text-generators-wiher-meister/44ba459c14fd510d9c10980e26d748cc/) [[1.3]](https://consensus.app/papers/on-decoding-strategies-for-neural-text-generators-wiher-meister/44ba459c14fd510d9c10980e26d748cc/).
- **Wenhong Zhu, Hong-ping Hao, Rui Wang (2023, arXiv)** explicitly adopt Keskar-style repetition penalty because models “tend to repeat the previous context,” and note that “tuning the repetition penalty hyperparameter can be challenging,” with excessive penalty causing “overly short sentences,” but again in open-ended generation rather than copy-preserving editing [[3.2]](https://consensus.app/papers/penalty-decoding-well-suppress-the-selfreinforcement-zhu-hao/bc39636d496957c683f80216d6a5b7d5/) [[3.3]](https://consensus.app/papers/penalty-decoding-well-suppress-the-selfreinforcement-zhu-hao/bc39636d496957c683f80216d6a5b7d5/) [[3.4]](https://consensus.app/papers/penalty-decoding-well-suppress-the-selfreinforcement-zhu-hao/bc39636d496957c683f80216d6a5b7d5/).

## Part 1 Evidence Table

| Paper | Authors, Year, Venue | What It Shows | Quote | Relevance To Your Specific Novelty Claim |
|---|---|---|---|---|
| Learning to Break the Loop | Xu et al., 2022, arXiv | Repetition is reinforced by prior context; mitigation can improve summarization/open-ended generation | “the more times a sentence is repeated in the context, the higher the probability of continuing to generate that sentence” [[4.1]](https://consensus.app/papers/learning-to-break-the-loop-analyzing-and-mitigating-xu-liu/e6317d1711625ed3b2eac61cf1a66ea7/) | **Indirect only** |
| On Decoding Strategies for Neural Text Generators | Wiher et al., 2022, TACL | Decoding strategy effects differ strongly by task | “the properties elicited by various decoding strategies do not always transfer across natural language generation tasks” [[1.2]](https://consensus.app/papers/on-decoding-strategies-for-neural-text-generators-wiher-meister/44ba459c14fd510d9c10980e26d748cc/) | **Indirect only** |
| Penalty Decoding | Zhu et al., 2023, arXiv | Repetition penalty helps open-ended generation, but hyperparameter choice is difficult and can shorten outputs | “determining the optimal repetition penalty value is challenging” [[3.5]](https://consensus.app/papers/penalty-decoding-well-suppress-the-selfreinforcement-zhu-hao/bc39636d496957c683f80216d6a5b7d5/) | **Indirect only** |
| RAP | Huang et al., 2025, preprint | Repetition penalty reduces repetition but can hurt task performance if overused | “excessive use of RPP can compromise the overall quality” [[2.1]](https://consensus.app/papers/rap-a-metric-for-balancing-repetition-and-performance-in-huang-nguyen/fc3ce0744b105a2884024be6ba4d72b0/) | **Indirect only** |
| Mitigating the Language Mismatch and Repetition Issues in LLM-based Machine Translation via Model Editing | Wang et al., 2024, arXiv | Repetition harms MT quality; naive fixes have “significant negative side-effect on the general translation quality” | “drastically affect the translation quality” [[5.1]](https://consensus.app/papers/mitigating-the-language-mismatch-and-repetition-issues-in-wang-li/33e6d15f345a5b719543330a9080501c/) and “significant negative side-effect on the general translation quality” [[5.2]](https://consensus.app/papers/mitigating-the-language-mismatch-and-repetition-issues-in-wang-li/33e6d15f345a5b719543330a9080501c/) | **Indirect only** |

No supplied paper reports the narrower mechanism you care about: that **decoder-only repetition penalty/no-repeat blocking counts tokens already present in the prompt**, thereby discouraging legitimate copying of correct spans in OCR correction, GEC, normalization, code editing, or extractive summarization. On the current evidence set, that looks like a **plausible novelty claim**, especially if your paper demonstrates a concrete evaluation artifact from that decoding choice.

## Part 2

Since 2023, the supplied literature shows substantial activity in **historical post-OCR correction** and some clear work on **Indic scripts**, but **nothing here is a Bengali zero-shot small-open-LLM study**. The closest Indic evidence is supervised Hindi/Gujarati/Marathi correction with larger-context PLMs and earlier Sanskrit benchmarks; the closest zero-shot LLM work is historical multilingual or English post-OCR, not Bengali [[6.1]](https://consensus.app/papers/a-framework-and-dataset-for-contextual-postocr-correction-bhandari-harit/2d881d6b070a57b2ab7a1dbfebf7dc05/) [[7.1]](https://consensus.app/papers/postcorrection-of-historical-text-transcripts-with-large-boros-ehrmann/f03628243373535a8c00303966736b39/) [[8.1]](https://consensus.app/papers/postocr-correction-with-openais-gpt-models-on-challenging-zhang-haverals/f3f35e8c306a515a9ddffe5c0eb3d6f1/).

- **Abhishek Bhandari and Gaurav Harit (2026, ACM TALLIP)** report supervised context-aware correction for Indic scripts, reducing CER from **10.20% to 6.73% for Hindi**, **6.10% to 1.39% for Gujarati**, and **8.19% to 3.29% for Marathi**, with largest gains on “character confusion and conjunct/ligature errors” [[6.1]](https://consensus.app/papers/a-framework-and-dataset-for-contextual-postocr-correction-bhandari-harit/2d881d6b070a57b2ab7a1dbfebf7dc05/) [[6.2]](https://consensus.app/papers/a-framework-and-dataset-for-contextual-postocr-correction-bhandari-harit/2d881d6b070a57b2ab7a1dbfebf7dc05/).
- **Ayush Maheshwari, Amrith Krishna, Ganesh Ramakrishnan (2022, arXiv)** provide the strongest Sanskrit benchmark in the set: all pretrained LM baselines beat OCR, and **ByT5+SLP1** reaches **2.98% CER** versus OCR **3.89%**, while CopyNet worsens both metrics, showing that copy-aware architectures do not automatically help [[9.1]](https://consensus.app/papers/a-benchmark-and-dataset-for-postocr-text-correction-in-maheshwari-krishna/10d7fc2c6e175cbc89bcf5172df4d79b/) [[9.2]](https://consensus.app/papers/a-benchmark-and-dataset-for-postocr-text-correction-in-maheshwari-krishna/10d7fc2c6e175cbc89bcf5172df4d79b/).
- **Elizabeth Soper, Stanley Fujimoto, Yen-Yun Yu (2021, W-NUT / listed here as 2021 report)** fine-tune **BART** for OCR newspaper text and report a **29.4% improvement in character accuracy**, an important earlier seq2seq baseline for historical OCR correction [[10.1]](https://consensus.app/papers/bart-for-postcorrection-of-ocr-newspaper-text-soper-fujimoto/b918f9ffc5a857ec848135c064444511/) [[10.2]](https://consensus.app/papers/bart-for-postcorrection-of-ocr-newspaper-text-soper-fujimoto/b918f9ffc5a857ec848135c064444511/).

## Recent Post-OCR Findings

| Paper | Authors, Year, Venue | Setting | Specific Finding With Quote |
|---|---|---|---|
| Leveraging LLMs for Post-OCR Correction of Historical Newspapers | Thomas, Gaizauskas, Lu, 2024, LT4HALA @ LREC-COLING | Historical newspapers, instruction-tuned Llama 2 vs BART | Llama 2 “outperform[s] BART, achieving a **54.51% reduction** in the character error rate against BART’s **23.30%**” [[11.1]](https://consensus.app/papers/leveraging-llms-for-postocr-correction-of-historical-thomas-gaizauskas/a310dd45303d5a0ab835d02ec2655960/) |
| Post-Correction of Historical Text Transcripts with Large Language Models | Boros, Ehrmann, Romanello, Najem-Meyer, Kaplan, 2024, exploratory study | 14 foundation models, zero/few-shot, multiple languages and periods | “LLMs are **anything but efficient** at this task” [[7.2]](https://consensus.app/papers/postcorrection-of-historical-text-transcripts-with-large-boros-ehrmann/f03628243373535a8c00303966736b39/) |
| Post-OCR Correction with OpenAI's GPT Models on Challenging English Prosody Texts | Zhang, Haverals, Naydan, Kernighan, 2024, DocEng | Historical English, zero-shot GPT prompting | Models “struggle with OCR texts that are either **extremely erroneous or already good**” and “tend to **overcorrect** works with already good OCR quality” [[8.2]](https://consensus.app/papers/postocr-correction-with-openais-gpt-models-on-challenging-zhang-haverals/f3f35e8c306a515a9ddffe5c0eb3d6f1/) [[8.3]](https://consensus.app/papers/postocr-correction-with-openais-gpt-models-on-challenging-zhang-haverals/f3f35e8c306a515a9ddffe5c0eb3d6f1/) |
| PreP-OCR | Guan et al., 2025, arXiv | Historical English/French/Spanish, ByT5 post-correction | After post-correction, **69.12%** of segments improve, **24.26%** are unchanged, and **6.62%** worsen [[12.1]](https://consensus.app/papers/prepocr-a-complete-pipeline-for-document-image-guan-lin/30b55b1a701d5824a485d0bb1d4bcacc/) [[12.2]](https://consensus.app/papers/prepocr-a-complete-pipeline-for-document-image-guan-lin/30b55b1a701d5824a485d0bb1d4bcacc/) |
| A proposal for post-OCR spelling correction using Language Models | Santos, Dantas, Sousa, 2024, LatinX in AI @ NeurIPS | OCR/HTR correction with Bart, ByT5, Llama 1/2 | “the smaller LMs outperformed the LLMs” and “ByT5 with byte-level tokenization improved CER and WER” [[13.1]](https://consensus.app/papers/a-proposal-for-postocr-spelling-correction-using-language-santos-dantas/aed1a1689173510d85ec89ff98d108ef/) [[13.2]](https://consensus.app/papers/a-proposal-for-postocr-spelling-correction-using-language-santos-dantas/aed1a1689173510d85ec89ff98d108ef/) |

On **over-correction**, the strongest direct OCR evidence is that GPT models “tend to overcorrect works with already good OCR quality” and struggle when OCR is “already good,” while PreP-OCR quantifies a non-trivial harmed subset and shows GPT-4o can get worse after post-correction because it produces “contextually plausible but factually incorrect hallucinations” [[8.2]](https://consensus.app/papers/postocr-correction-with-openais-gpt-models-on-challenging-zhang-haverals/f3f35e8c306a515a9ddffe5c0eb3d6f1/) [[8.3]](https://consensus.app/papers/postocr-correction-with-openais-gpt-models-on-challenging-zhang-haverals/f3f35e8c306a515a9ddffe5c0eb3d6f1/) [[12.2]](https://consensus.app/papers/prepocr-a-complete-pipeline-for-document-image-guan-lin/30b55b1a701d5824a485d0bb1d4bcacc/) [[12.3]](https://consensus.app/papers/prepocr-a-complete-pipeline-for-document-image-guan-lin/30b55b1a701d5824a485d0bb1d4bcacc/) [[12.4]](https://consensus.app/papers/prepocr-a-complete-pipeline-for-document-image-guan-lin/30b55b1a701d5824a485d0bb1d4bcacc/). Within the supplied set, there is **no Bengali/Bangla paper**, **no zero-shot small-open-model Indic post-OCR paper**, and **no grammatical error correction paper** directly evaluating damage to already-correct text.

## Part 3

The supplied papers support your **error-rate dependence** claim in a general sense, but they do **not** provide a clean published **break-even CER threshold** as specific as “correction stops helping below CER 0.02.” They do show that correction quality depends strongly on initial OCR quality, that synthetic training noise level matters, and that under-corrupted synthetic data often transfers better than over-corrupted data [[9.3]](https://consensus.app/papers/a-benchmark-and-dataset-for-postocr-text-correction-in-maheshwari-krishna/10d7fc2c6e175cbc89bcf5172df4d79b/) [[14.1]](https://consensus.app/papers/effective-synthetic-data-and-testtime-adaptation-for-ocr-guan-xu/9cf85d4b00fe571ba0b1153f4bc298ce/) [[15.1]](https://consensus.app/papers/scrambled-text-finetuning-language-models-for-ocr-error-bourne/29799409937c5b4d969b416b2275f724/).

- **James Zhang et al. (2024, DocEng)** report that GPT models “struggle with OCR texts that are either extremely erroneous or already good,” which is direct evidence of a **non-monotonic benefit by input quality**, but without a numeric break-even threshold [[8.2]](https://consensus.app/papers/postocr-correction-with-openais-gpt-models-on-challenging-zhang-haverals/f3f35e8c306a515a9ddffe5c0eb3d6f1/).
- **Shu-Hao Guan et al. (2024, study on synthetic data and test-time adaptation)** explicitly “define Character Error Rate (CER) thresholds for ‘effective’ and ‘ineffective’ synthetic data,” and show weakly supervised multi-noise synthetic data with **ByT5** can reduce CER by **62.95%**, rising to **68.67%** with test-time adaptation [[14.1]](https://consensus.app/papers/effective-synthetic-data-and-testtime-adaptation-for-ocr-guan-xu/9cf85d4b00fe571ba0b1153f4bc298ce/) [[14.2]](https://consensus.app/papers/effective-synthetic-data-and-testtime-adaptation-for-ocr-guan-xu/9cf85d4b00fe571ba0b1153f4bc298ce/) [[14.3]](https://consensus.app/papers/effective-synthetic-data-and-testtime-adaptation-for-ocr-guan-xu/9cf85d4b00fe571ba0b1153f4bc298ce/).
- **Jonathan Bourne (2025, IJDAR)** finds that with synthetic corruption, “**training on under-corrupted data is better than over-corrupted data**,” and that models trained on synthetic data reduce CER by **55%** and outperform models trained on real data, which is strong transfer evidence but not a Bengali or zero-shot result [[15.2]](https://consensus.app/papers/scrambled-text-finetuning-language-models-for-ocr-error-bourne/29799409937c5b4d969b416b2275f724/) [[15.1]](https://consensus.app/papers/scrambled-text-finetuning-language-models-for-ocr-error-bourne/29799409937c5b4d969b416b2275f724/) [[15.3]](https://consensus.app/papers/scrambled-text-finetuning-language-models-for-ocr-error-bourne/29799409937c5b4d969b416b2275f724/).

The broader fine-tuning literature in the supplied set consistently favors **byte-level or seq2seq post-correction** over naive prompting for noisy text. BART fine-tuning improved historical OCR newspaper text by **29.4% character accuracy** [[10.2]](https://consensus.app/papers/bart-for-postcorrection-of-ocr-newspaper-text-soper-fujimoto/b918f9ffc5a857ec848135c064444511/). Sanskrit results favored **ByT5**, reaching **2.98% CER** from **3.89% OCR**, though byte-level sequence length could also cause truncation and worse settings if not tuned [[9.1]](https://consensus.app/papers/a-benchmark-and-dataset-for-postocr-text-correction-in-maheshwari-krishna/10d7fc2c6e175cbc89bcf5172df4d79b/) [[9.4]](https://consensus.app/papers/a-benchmark-and-dataset-for-postocr-text-correction-in-maheshwari-krishna/10d7fc2c6e175cbc89bcf5172df4d79b/) [[9.5]](https://consensus.app/papers/a-benchmark-and-dataset-for-postocr-text-correction-in-maheshwari-krishna/10d7fc2c6e175cbc89bcf5172df4d79b/). Monolingual byte-level models in 2025 also “consistently reduce error rates,” including **12.78% CER** improvement on BRESSAY, reinforcing transferability of byte-level correction in noisy settings [[16.1]](https://consensus.app/papers/a-proposal-of-postocr-spelling-correction-using-araújo-bezerra/9681ec863dce5ed4afc13072afc1c710/) [[16.2]](https://consensus.app/papers/a-proposal-of-postocr-spelling-correction-using-araújo-bezerra/9681ec863dce5ed4afc13072afc1c710/).

Your paper’s strongest likely novelty is therefore **not** “LLMs can fail at post-OCR correction” or “over-correction exists,” both of which are already published, but the more specific combination of **historical Bengali**, **zero-shot small open models**, **failure even at very low synthetic CER**, and especially the still-unpublished-looking claim that **decoder-time repetition controls can penalize prompt tokens and thus artifactually worsen copy-heavy correction evaluation**.
 
_These search results were found and analyzed using Consensus, an AI-powered search engine for research. Try it at https://consensus.app. © 2026 Consensus NLP, Inc. Personal, non-commercial use only; redistribution requires copyright holders’ consent._
 
## References
 
`1.` Wiher G, Meister C, Cotterell R. On Decoding Strategies for Neural Text Generators. *Transactions of the Association for Computational Linguistics.* 2022;10:997-1012. doi:10.1162/tacl_a_00502
 
> [1.1] This work—in contrast—provides a comprehensive analysis of the interaction between language generation tasks and decoding strategies. (Abstract)
 
> [1.2] Yet the properties elicited by various decoding strategies do not always transfer across natural language generation tasks. (Abstract)
 
> [1.3] For example, the nature of the diversity–quality trade-off in language generation is very task-specific; the length bias often attributed to beam search is not constant across tasks. (Abstract)
 
`2.` Huang D, Nguyen T, Liausvia F, Wang Z. RAP: A Metric for Balancing Repetition and Performance in Open-Source Large Language Models. 2025. doi:10.18653/v1/2025.naacl-long.69
 
> [2.1] The repetition penalty parameter (RPP) aims to mitigate this issue by preventing repeated content generation, but excessive use of RPP can compromise the overall quality. (Abstract)
 
`3.` Zhu W, Hao H, Wang R. Penalty Decoding: Well Suppress the Self-Reinforcement Effect in Open-Ended Text Generation. *ArXiv.* 2023;abs/2310.14971. doi:10.48550/arxiv.2310.14971
 
> [3.1] To tackle this problem, we have introduced the repetition penalty proposed by Keskar et al. (2019). (Body)
 
> [3.2] For instance, when presented with a prefix such as "Barack Obama was born in Honolulu, Hawaii. He was born in," the model (e.g., GPT2-XL) tends to repeat the previous context. (Body)
 
> [3.3] However, tuning the repetition penalty hyperparameter can be challenging. (Body)
 
> [3.4] In addition, we introduce a length penalty to address overly short sentences caused by excessive penalties. (Abstract)
 
> [3.5] However, determining the optimal repetition penalty value is challenging. (Abstract)
 
`4.` Xu J, Liu X, Yan J, Cai D, Li H, Li J. Learning to Break the Loop: Analyzing and Mitigating Repetitions for Neural Text Generation. *ArXiv.* 2022;abs/2206.02369. doi:10.48550/arxiv.2206.02369
 
> [4.1] 2) The sentence-level repetitions have a \textit{self-reinforcement effect}: the more times a sentence is repeated in the context, the higher the probability of continuing to generate that sentence; (Abstract)
 
> [4.2] Extensive experiments on open-ended text generation (Wikitext-103) and text summarization (CNN/DailyMail) demonstrate the generality and effectiveness of our method. (Abstract)
 
`5.` Wang W, Li Z, Lian D,  C, Song L, Wei Y. Mitigating the Language Mismatch and Repetition Issues in LLM-based Machine Translation via Model Editing. *ArXiv.* 2024;abs/2410.07054. doi:10.48550/arxiv.2410.07054
 
> [5.1] In the work, we focus on utilizing LLMs to perform machine translation, where we observe that two patterns of errors frequently occur and drastically affect the translation quality: language mismatch and repetition. (Abstract)
 
> [5.2] The work sets out to explore the potential for mitigating these two issues by leveraging model editing methods, e.g., by locating Feed-Forward Network (FFN) neurons or something that are responsible for the errors and deactivating them in the inference time.We find that directly applying such methods either limited effect on the targeted errors or has significant negative side-effect on the general translation quality, indicating that the located components may also be crucial for ensuring machine translation with LLMs on the rails.To this end, we propose to refine the located components by fetching the intersection of the locating results under different language settings, filtering out the aforementioned information that is irrelevant to targeted errors. (Abstract)
 
`6.` Bhandari A, Harit G. A Framework and Dataset for Contextual Post-OCR Correction. *ACM Transactions on Asian and Low-Resource Language Information Processing.* 2026;25:1 - 20. doi:10.1145/3815575
 
> [6.1] Experiments demonstrate substantial reductions in the character error rates (10.20% to 6.73% for Hindi, 6.10% to 1.39% for Gujarati, and 8.19% to 3.29% for Marathi) as well as the word error rates (28.53% to 14.57% for Hindi, 20.85% to 4.16% for Gujarati, and 24.89% to 8.28% for Marathi). (Abstract)
 
> [6.2] Error-type analysis indicates the largest improvements for character confusion and conjunct/ligature errors. (Abstract)
 
`7.` Boros E, Ehrmann M, Romanello M, Najem-Meyer S, Kaplan F. Post-Correction of Historical Text Transcripts with Large Language Models: An Exploratory Study. 2024. doi:10.18653/v1/2024.latechclfl-1.14
 
> [7.1] To this end, we evaluate fourteen foundation language models from four model series against various post-correction benchmarks comprising different languages, time periods, document types, and different transcription quality and origins. (Body)
 
> [7.2] Our evaluation shows that LLMs are anything but efficient at this task. (Abstract)
 
`8.` Zhang J, Haverals W, Naydan M, Kernighan B. Post-OCR Correction with OpenAI's GPT Models on Challenging English Prosody Texts. *Proceedings of the ACM Symposium on Document Engineering 2024.* 2024. doi:10.1145/3685650.3685669
 
> [8.1] In this work, we assess three recent LLMs released by OpenAI (GPT 3.5-turbo, 4, and 4-turbo) on their ability to perform post-OCR correction on the textual OCR outputs provided by the PPA. (Body)
 
> [8.2] We also observe that the models struggle with OCR texts that are either extremely erroneous or already good. (Body)
 
> [8.3] The models tend to overcorrect works with already good OCR quality but perform well overall, with the best model setup improving the Character Error Rate (CER) by a mean of 18.92%. (Abstract)
 
`9.` Maheshwari A, Krishna A, Ramakrishnan G. A Benchmark and Dataset for Post-OCR text correction in Sanskrit. *ArXiv.* 2022;abs/2211.07980. doi:10.48550/arxiv.2211.07980
 
> [9.1] , ByT5 (Xue et al., 2022), in conjunction with phonetic encoding (SLP1), among these benchmarks (all described in § 3) reports a character and word error rates of 2.98% and 23.19% respectively, as against that of 3.89% and 30.23% from the original OCR. (Body)
 
> [9.2] In our experiments, CopyNet's predictions worsen as per both our metrics, resulting in a CER and WER of 13.25% and 50.38% respectively. (Body)
 
> [9.3] Further, as the performance of the post-OCR text correction system is highly dependent on the predictions of the OCR, we also release the test dataset used for testing our current OCR. (Body)
 
> [9.4] However, with ByT5, we find that our model produces truncated outputs mostly due to an increase in sequence length in ByT5 due to the byte-level vocabulary used in it. (Body)
 
> [9.5] The output from ByT5-Dev has a CER of 6.17% and a WER of 27.72%, higher than that of ByT5SLP1, when a sequence length of 1024 was used. (Body)
 
`10.` Soper E, Fujimoto S, Yu Y. BART for Post-Correction of OCR Newspaper Text. 2021. doi:10.18653/v1/2021.wnut-1.31
 
> [10.1] We cast error correction as a translation task, and fine-tune BART, a transformerbased sequence-to-sequence language model pretrained to denoise corrupted text. (Body)
 
> [10.2] We are the first to use sentence-level transformer models for OCR post-correction, and our best model achieves a 29.4% improvement in character accuracy over the original noisy OCR text. (Body)
 
`11.` Thomas A, Gaizauskas RJ, Lu H. Leveraging LLMs for Post-OCR Correction of Historical Newspapers. *Proceedings of the Third Workshop on Language Technologies for Historical and Ancient Languages (LT4HALA) @ LREC-COLING-2024.* 2024. doi:10.63317/5aezhy5iacm9
 
> [11.1] We achieve a significant enhancement in OCR quality with Llama 2 outperforming BART, achieving a 54.51% reduction in the character error rate against BART’s 23.30%. (Abstract)
 
`12.` Guan S, Lin M, Xu C, et al. PreP-OCR: A Complete Pipeline for Document Image Restoration and Enhanced OCR Accuracy. *ArXiv.* 2025;abs/2505.20429. doi:10.18653/v1/2025.acl-long.749
 
> [12.1] When post-OCR correction is applied, Tesseract's CER is further reduced from 1.99% to 1.30%. (Body)
 
> [12.2] Overall, 69.12% of text segments experience a CER decrease, 24.26% remain unchanged, and 6.62% increase. (Body)
 
> [12.3] However, GPT-4o's CER generally increases at this stage. (Body)
 
> [12.4] This outcome stems from GPT-4o's tendency to produce contextually plausible but factually incorrect hallucinations (Yang et al., 2024), which often evade detection by the correction model due to the absence of clear spelling or grammatical errors. (Body)
 
`13.` Santos S, Dantas B, De Sousa AF, Zanchettin C. A proposal for post-OCR spelling correction using Language Models. *LatinX in AI at Neural Information Processing Systems Conference 2024.* 2024. doi:10.52591/202412104
 
> [13.1] The results indicate that the smaller LMs outperformed the LLMs in terms of error rate reduction (CER and WER). (Abstract)
 
> [13.2] ByT5 with byte-level tokenization improved CER and WER, proving performance for texts with high noise. (Abstract)
 
`14.` Guan S, Xu C, Lin M, Greene D. Effective Synthetic Data and Test-Time Adaptation for OCR Correction. 2024. doi:10.18653/v1/2024.emnlp-main.862
 
> [14.1] We define Character Error Rate (CER) thresholds for “effective” and “ineffective” synthetic data, allowing us to create more useful multi-noise level synthetic datasets. (Abstract)
 
> [14.2] Our experiments show that using weak supervision and effective synthetic data with multiple noise levels can reduce the Character Error Rate of OCR texts by 62.95% when used in conjunction with the ByT5 model. (Body)
 
> [14.3] Notably, the ByT5 model achieves a CER reduction of 68.67% without relying on manually annotated data (Abstract)
 
`15.` Bourne J. Scrambled text: fine-tuning language models for OCR error correction using synthetic data. *International Journal on Document Analysis and Recognition (IJDAR).* 2025;28:741 - 755. doi:10.1007/s10032-025-00522-0
 
> [15.1] Key findings include training on under-corrupted data is better than over-corrupted data; non-uniform character level corruption is better than uniform corruption; more tokens-per-observation outperforms more observations for a fixed token budget. (Body)
 
> [15.2] Models trained on synthetic data reduce the character error rate by 55% and word error rate by 32% over the base LM and outperform models trained on real data. (Body)
 
> [15.3] Comparing the models showed how valuable synthetic data is, with the synthetic model outperforming all the real datasets. (Discussion)
 
`16.` De Araújo SS, Bezerra BL, De Sousa AF. A Proposal of Post-OCR Spelling Correction Using Monolingual Byte-level Language Models. *Proceedings of the 2025 ACM Symposium on Document Engineering.* 2025. doi:10.1145/3704268.3748673
 
> [16.1] The results demonstrate that Monobyte consistently reduces error rates, reinforcing the advantages of byte-level tokenization for spelling correction in noisy contexts, while raising important considerations about computational efficiency for practical applications. (Body)
 
> [16.2] In RIMES, reductions are 13.48% (CER) and 33.34% (WER), while in BRESSAY, Monobyte improves CER by 12.78% and WER by 40.62%. (Abstract)
