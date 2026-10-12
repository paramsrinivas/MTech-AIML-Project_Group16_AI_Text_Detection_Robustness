# Assignment 16 — Submission Guide

## Project

**AI-Text Detection Robustness**

This project evaluates a TF-IDF + LinearSVC classifier on the HC3 dataset under original AI text, a phrase-removal variant, and generated semantic paraphrases. A separate larger experiment tests deterministic discourse-marker substitutions.

## Deliverables

- **Final report:** `reports/Assignment16_Final_Report.docx`
- **Presentation:** `presentation/Group16_AI_Text_Detection_Robustness.pptx`
- **Code:** `src/`
- **Metrics, predictions, and figures:** `results/`
- **Dataset notes:** `data/README.md`
- **Setup and reproduction instructions:** `README.md`

The notebook is an inspection/illustration companion. Use the scripts in `src/` to reproduce the reported experiments.

## Main paired pilot

The evaluation uses 300 held-out question pairs; each condition contains 300 human controls and 300 AI passages.

| Metric | Original AI | Phrase-removal variant | Semantic paraphrase |
|---|---:|---:|---:|
| Accuracy | 98.83% | 98.83% | 89.67% |
| AI recall | 98.00% | 98.00% | 79.67% |
| AI F1 score | 98.82% | 98.82% | 88.52% |
| False-positive rate | 0.33% | 0.33% | 0.33% |

Recall drops by 18.33 percentage points for generated paraphrases (294/300 original AI passages detected versus 239/300 paraphrases detected); false negatives increase from 6 to 61.

**Interpretation limit:** the phrase-removal function changed 0 of the 300 sampled passages, so identical metrics are not evidence that the detector withstands effective surface edits. The semantic paraphraser sometimes shortens passages, omits information, or changes facts; the result cannot isolate semantic rewriting from length/compression and content changes.

## Supplementary edit experiment

The separate `src/run_robustness.py` experiment uses whitespace normalization and at most one deterministic discourse-marker substitution. It is not the same edit procedure as the paired pilot.

| Evaluation group | Passages | Accuracy / AI recall | False-positive rate |
|---|---:|---:|---:|
| Original AI-only group | 4,959 | 98.37% | Not applicable |
| Edited AI-only group | 4,959 | 98.02% | Not applicable |
| Human-only group | 10,198 | 99.26% accuracy | 0.74% |

The supplementary groups contain only one class each, so they are not balanced human-versus-AI evaluations. In paired AI predictions, 17 of 4,959 predictions changed and the AI detection rate fell by 0.34 percentage points.

## Reproduction

See `README.md` for environment setup and the full run order. The semantic-paraphrase stage downloads `humarin/chatgpt_paraphraser_on_T5_base` on first use. Preserve validated outputs before rerunning stages that overwrite files.

## Limitations

The results cover one dataset, one principal classifier, one grouped split, and specific editing/generation procedures. The paraphrases are not verified as meaning-preserving, and the source distribution is dominated by Reddit ELI5 examples. Do not generalize these results to all detectors or treat a detector prediction as proof of authorship.
