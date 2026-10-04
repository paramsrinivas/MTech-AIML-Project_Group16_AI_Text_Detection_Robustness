# Assignment 16 — Final Package

This folder documents the final deliverables for the AI Text Detection Robustness project.

## Deliverables

- `reports/Assignment16_Final_Report.docx` — written report.
- `presentation/Group16_AI_Text_Detection_Robustness.pptx` — 10-slide presentation.
- `results/` — evaluation metrics, predictions, summaries, and figures.
- `src/` — scripts used for data preparation, experiments, and evaluation.
- `data/` — dataset files and data documentation.

## Main findings

The evaluation covered 300 held-out question pairs, with 300 human and 300 AI passages per condition.

| Metric | Original AI | Surface-edited AI | Semantic paraphrase |
|---|---:|---:|---:|
| Accuracy | 98.83% | 98.83% | 89.67% |
| AI recall | 98.00% | 98.00% | 79.67% |
| AI F1 score | 98.82% | 98.82% | 88.52% |
| False-positive rate | 0.33% | 0.33% | 0.33% |

AI recall fell by 18.33 percentage points after semantic paraphrasing, while false negatives increased from 6 to 61.

## Interpretation and limitation

Some generated paraphrases changed facts or omitted information. The results therefore reflect this paraphrasing process and should not be interpreted as a test of meaning-preserving paraphrases alone.

The experiment evaluates one classifier on one dataset. Its findings should not be generalized to every AI detector or writing style.
