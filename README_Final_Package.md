# Assignment 16 — Submission Guide

## Project

**AI-Text Detection Robustness**

The project evaluates a TF-IDF + LinearSVC detector on original HC3 AI passages, two distinct surface-edit procedures, and generated semantic paraphrases. The experiments are specific to this dataset, model, and transformation pipeline.

## Deliverables

- **Final report:** `reports/Assignment16_Final_Report.docx`
- **Presentation:** `presentation/Group16_AI_Text_Detection_Robustness.pptx`
- **Code:** `src/`
- **Metrics, predictions, and figures:** `results/`
- **Dataset and data notes:** `data/`
- **Exploratory notebook:** `notebooks/experiment.ipynb`

The notebook is an exploratory companion, not the canonical end-to-end runner. Use the scripts and the reproduction sequence in `README.md` to regenerate experiment outputs.

## Main paired result

The main paired evaluation used 300 held-out question pairs, with 300 human passages and 300 AI passages per condition.

| Metric | Original AI | Phrase-removal surface edit | Semantic paraphrase |
|---|---:|---:|---:|
| Accuracy | 98.83% | 98.83% | 89.67% |
| AI recall | 98.00% | 98.00% | 79.67% |
| AI F1 score | 98.82% | 98.82% | 88.52% |
| False-positive rate | 0.33% | 0.33% | 0.33% |

AI recall decreased by 18.33 percentage points for the paraphrase condition, and false negatives increased from 6 to 61. The phrase-removal edit changed 0 of the 300 AI passages, so identical metrics in that condition are not evidence of successful robustness testing. The semantic paraphrases may be shorter and may alter meaning, so the recall difference cannot be attributed to semantic rewriting alone.

## Supplementary connective-substitution result

The separate `src/run_robustness.py` experiment applies whitespace/punctuation normalization and one connective substitution where possible.

| Evaluation group | Passages | Accuracy | AI recall | False-positive rate |
|---|---:|---:|---:|---:|
| Original AI | 4,959 | 98.37% | 98.37% | Not applicable |
| Connective-edited AI | 4,959 | 98.02% | 98.02% | Not applicable |
| Human | 10,198 | 99.26% | Not applicable | 0.74% |

These are class-specific groups, not balanced evaluations. Read this table separately from the main paired evaluation.

## Reproduction

Follow the setup and run order in `README.md`. After the existing paraphrase-generation and paired-evaluation steps, run:

```bash
python src/analyze_paraphrase_length.py
```

This creates a truncation-control comparison and paired uncertainty outputs. Its results must be generated in the target environment before any new metrics are quoted. Keep the original paraphrase file and existing result outputs when comparing runs.

## Limitations

The study covers one dataset and one classical classifier. The HC3 source distribution is imbalanced, and the paraphraser can compress text, omit information, or change facts. The results describe the observed pipeline rather than all detectors or verified meaning-preserving paraphrases. Detector predictions are not proof of authorship.
