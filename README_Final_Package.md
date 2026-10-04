# Assignment 16 — Submission Guide

## Project

**AI-Text Detection Robustness**

This project studies whether editing and paraphrasing affect a classifier's ability to distinguish human-written text from AI-generated text. The experiments compare original AI text, surface-edited text, and semantically paraphrased text.

## Where to find everything

- **Final report:** `reports/Assignment16_Final_Report.docx`
- **Presentation:** `presentation/Group16_AI_Text_Detection_Robustness.pptx`
- **Experiment and evaluation code:** `src/`
- **Evaluation metrics and predictions:** `results/`
- **Figures:** `results/figures/`
- **Dataset:** `data/`
- **Baseline notebook:** `notebooks/experiment.ipynb`

Start with the final report for the full discussion, then use the results files to inspect the metrics and predictions.

## Main result

The final evaluation used 300 held-out question pairs, with 300 human passages and 300 AI passages in each condition.

| Metric | Original AI | Surface-edited AI | Semantic paraphrase |
|---|---:|---:|---:|
| Accuracy | 98.83% | 98.83% | 89.67% |
| AI recall | 98.00% | 98.00% | 79.67% |
| AI F1 score | 98.82% | 98.82% | 88.52% |
| False-positive rate | 0.33% | 0.33% | 0.33% |

The largest change appeared after semantic paraphrasing. AI recall dropped by 18.33 percentage points, and false negatives increased from 6 to 61. Surface editing produced almost no change in the reported evaluation.

## How to reproduce the experiments

The experiment and evaluation scripts are in `src/`. Check `requirements.txt` for the Python dependencies and `README.md` for the project setup instructions before running them.

The notebook demonstrates dataset checks and a baseline model using sample data. It is not the source of the final reported metrics.

## Limitations

Some generated paraphrases changed facts or left out information. The results therefore reflect the paraphrasing process used in this project; they do not isolate meaning-preserving paraphrasing alone.

The evaluation covers one classifier and one dataset. The findings should not be treated as evidence that all AI detectors behave in the same way, or that detector predictions establish who wrote a passage.