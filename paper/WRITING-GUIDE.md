# Writing guide for this paper

Lessons taken from a video on writing research papers (transcript shared by the author on 2026-10-15), turned into concrete actions for `main.tex`. The video quotes studies on citations without naming them, so the numbers below are the video's claims, not checked by us. Use them as working rules, and never cite them in the paper.

## 1. Narrative: background, problem, solution
A paper reads like a short film: tell readers what they are about to hear, show them the problem, then show how you dealt with it. The video reports that papers whose abstracts follow this kind of narrative are cited more often.

**For this paper:**
- **Background:** Bengali archives hold huge amounts of printed text that OCR reads badly (CER around 0.3 here). Small language models are cheap and are already used to "clean" OCR text.
- **Problem (make it painful, not "nobody studied this"):** cleaning OCR with a language model can quietly damage text that was already correct, and nobody checks. BnMMLU cleaned OCR output with GPT-3.5 and never measured the result against a reference. An archive that runs a small model over its scans may make its text worse without knowing it. Evaluations can mislead too: a common decoding default makes a copying model look destructive.
- **Solution / what we did:** measure it properly. 40 pages, two engines, eight models, a clean-text control and a decoding ablation. Answer: keep the raw OCR, or let a model change only the words the engine is unsure of; never apply repetition penalties to the prompt.

**Action:** rewrite the gap paragraph at the end of Sec I (it currently ends with "We found no controlled study ...", which is the boring "nobody has studied this" framing) so that the gap is the risk of silent damage and misleading evaluations. Keep the related-work facts; change the framing.

## 2. Write backwards
Start from the conclusions and work back: conclusions → results → methods → discussion → introduction → abstract → title. The introduction comes near the end because only then do you know what you are introducing.

**For this paper:** the GPU runs of the third review (C1, C2, C4, X1, then C5, M1-M3, M7) may change the results. Order of work once they are in:
1. Fix the conclusions: the 3-4 sentences a reader should remember.
2. Results (Sec IV, V) and their tables and figures.
3. Methods (Sec III): only what the results need.
4. Discussion (Sec VI), tied to the tables.
5. Introduction, including the gap paragraph (Sec 1 above).
6. Abstract (about 200 words, narrative order).
7. Title (Sec 3 below).

## 3. Title: state the finding
- Titles that state the finding are cited more. Question titles get downloaded but cited less, and are hard to find again in a reference manager.
- Short helps; a colon often helps.
- A little humour can help memory, but only if it does not hide the finding.

**Our title is a question** ("Does Zero-Shot Small-Model Post-Correction Help Bengali OCR? ..."). Change it to a statement once the final results are in (C1/C2 might change the Bengali-checkpoint story). Candidates:
- *Small Language Models Do Not Correct Historical Bengali OCR, and a Decoding Default Makes Them Look Worse*
- *Zero-Shot Small Language Models Do Not Correct Historical Bengali OCR: Eight Models, Two Engines and a Decoding Pitfall*
- *Leave the OCR Alone: Small Language Models Do Not Correct Historical Bengali Print* (light humour, still states the finding)

Check the title with Bijon Sir: the current one was a response to an earlier review comment ("title versus evidence").

## 4. Plain words, few acronyms
- More jargon in the title and abstract goes with fewer citations. Most acronyms in the literature appear fewer than 10 times, so readers will not know ours.
- Test: read the abstract aloud to speech-to-text (Word dictation). If it stumbles, the wording is too technical.

**For this paper:**
- The abstract currently uses only CER, OCR and the engine names (checked 2026-10-15). Keep it that way: no cMER, wMER, TOST, HiResCAM, Holm or "chunk" in the abstract or title.
- In the body, define each acronym once and use as few as possible. wMER, cMER and TOST are needed in tables; in prose prefer "bounded word error rate" and "equivalence test" where the sentence allows.
- Explain the Grad-CAM adaptation (X1) in one plain sentence before any formula: "which part of its input each output word relied on".

## 5. No hype words
Words like novel, robust, unprecedented and innovative have multiplied in abstracts and no longer persuade; they make readers suspicious.

**For this paper (checked 2026-10-15):** no "novel", "unprecedented" or "innovative". "Robust" appears only in the technical sense (robustness checks; "not robust" for the Qwen3 gating gain, the reviewer's wording). Keep it so. A negative result is credible exactly because it is stated plainly.

## 6. Preprint
The video reports about 36% more citations and more online attention for papers with a preprint (e.g. arXiv). The science still matters most: title features explain only a small part of citation differences.

**For this paper:** consider an arXiv preprint once the paper is final. Check first that the course and the target venue allow it; this is the author's and Bijon Sir's decision. It would also answer R2 (a citable public version), together with a Zenodo deposit of code and outputs.

## 7. Getting the writing done
- Name the block when stuck (the draft feels bad, no idea yet, no motivation, chasing a perfect sentence, "tomorrow", needing quiet, other chores) and do the smallest possible next step: two minutes, one paragraph.
- Allow a bad first draft; sentences get fixed after review.
- Fixed, protected writing slots each week work better than bingeing before a deadline. Someone who asks "what did you write this week?" helps.

## Checklist before each new draft goes to Bijon Sir
- [ ] Abstract follows background → problem → what we did → what we found → what to do, in about 200 words, with no acronyms beyond CER and OCR.
- [ ] The gap in Sec I is framed as a problem (silent damage, misleading evaluations), not as "not studied".
- [ ] The title states the finding.
- [ ] No hype words (`grep -i -w "novel\|unprecedented\|innovative\|groundbreaking" paper/main.tex` finds nothing).
- [ ] The conclusion gives the practical advice in one or two sentences.
- [ ] Read the abstract aloud once.
