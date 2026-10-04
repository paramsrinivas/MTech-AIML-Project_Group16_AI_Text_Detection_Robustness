# AI Text Detection Robustness

**Course:** AI26501 Modern AI Assignments (SP-2026)  
**Assignment:** 16 — AI Text Detection Robustness

## Project overview

This project tests whether a machine-learning classifier can distinguish human-written passages from AI-generated passages after the wording of the AI text has been changed.

We used the HC3 dataset and evaluated a TF-IDF text representation with a LinearSVC classifier. The evaluation compared original AI text, surface-edited text, and semantically paraphrased text against human-written passages.

The main question was whether changes in wording would affect the classifier's ability to recognize AI-generated text.

## Method

The classifier uses word-level TF-IDF features with a LinearSVC model. The experiment evaluates the same task under three text conditions:

- **Original:** AI-generated passages in their original form.
- **Surface-edited:** AI passages with surface-level changes.
- **Semantic paraphrase:** AI passages rewritten by a paraphrasing process.

The model was trained on the original training data. The held-out evaluation used 300 question pairs, with 300 human passages and 300 AI passages per condition.

The evaluation reports accuracy, AI precision, AI recall, F1 score, false-positive rate, and confusion matrices.

## Results

| Metric | Original AI | Surface-edited | Semantic paraphrase |
|---|---:|---:|---:|
| Accuracy | 98.83% | 98.83% | 89.67% |
| AI precision | 99.66% | 99.66% | 99.58% |
| AI recall | 98.00% | 98.00% | 79.67% |
| AI F1 score | 98.82% | 98.82% | 88.52% |
| False-positive rate | 0.33% | 0.33% | 0.33% |

AI recall decreased by 18.33 percentage points after semantic paraphrasing. The number of AI passages classified as human increased from 6 to 61.

These results suggest that semantic rewriting affected this classifier more than the surface edits used in this experiment. They do not establish that all AI detectors behave the same way.

## Repository structure

- `data/` — HC3 dataset files and data notes.
- `notebooks/` — experiment notebook.
- `src/` — data preparation, model, evaluation, and robustness scripts.
- `results/` — predictions, metrics, summaries, and figures.
- `reports/` — final written report.
- `presentation/` — final presentation slides.
- `requirements.txt` — Python dependencies.

## Environment setup

The project was developed on macOS using Python. To create a fresh environment and install the listed dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The saved results can be inspected directly in `results/`. To review the experiment implementation, start with `src/prepare_hc3.py`, `src/model.py`, `src/run_experiment.py`, `src/run_robustness.py`, and `src/evaluate_final.py`.

Check each script before rerunning it, since the required inputs and execution options differ by stage.

## Supplementary robustness experiment

The project also includes a separate test of lightweight surface edits in
`results/robustness_condition_metrics.csv`. In this experiment, the detector
was trained on original training passages and evaluated on held-out human
passages, original AI passages, and edited copies of the AI passages.

| Evaluation group | Passages | Accuracy | AI recall | False-positive rate |
|---|---:|---:|---:|---:|
| Original AI | 4,959 | 98.37% | 98.37% | Not applicable |
| Surface-edited AI | 4,959 | 98.35% | 98.35% | Not applicable |
| Human | 10,198 | 99.26% | Not applicable | 0.74% |

These are class-specific groups, not three balanced evaluations containing
both human and AI passages. Consequently, metrics such as false-positive
rate for AI-only groups and AI recall for the human-only group are not
meaningful in this table. These results should not be compared directly
with the main paired evaluation above.

## Limitations

The evaluation covers one dataset, one classical classifier, and the text produced by the paraphrasing process used in this project. Manual inspection found some paraphrases that changed facts or omitted information. Therefore, the results measure performance on these generated outputs, not only on meaning-preserving paraphrases.

AI-text detectors can produce both false positives and false negatives. Their predictions should not be treated as proof of authorship.

## Main deliverables

- `reports/Assignment16_Final_Report.docx`
- `presentation/Group16_AI_Text_Detection_Robustness.pptx`
- Evaluation tables and figures under `results/`
