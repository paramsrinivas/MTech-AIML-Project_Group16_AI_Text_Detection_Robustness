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

The project was developed with Python 3.12 on macOS. From the repository root, create and activate a virtual environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Reproduce the experiments

Run the commands from the repository root, in the order shown below.

**1. Prepare the dataset**

```bash
python src/prepare_hc3.py
```

This creates `data/hc3_original.csv`. Dataset preparation requires internet access to retrieve the HC3 dataset if it is not already cached.

**2. Run the baseline experiment**

```bash
python src/run_experiment.py
```

This trains and evaluates the baseline classifiers and saves their metrics, predictions, split manifest, and figure under `results/`.

**3. Run the surface-edit robustness experiment**

```bash
python src/run_robustness.py
```

This evaluates held-out human passages, original AI passages, and surface-edited AI passages.

**4. Generate semantic paraphrases**

```bash
python src/generate_paraphrases.py
```

This uses `humarin/chatgpt_paraphraser_on_T5_base` and writes `results/semantic_paraphrases.csv`. The first run requires internet access to download the model and tokenizer. Generation can take substantial time and may require considerable memory.

**5. Evaluate semantic paraphrases**

```bash
python src/evaluate_final.py
```

This reconstructs the grouped train/test split, trains the detector on original training text, and evaluates the saved paired sample. It writes the final metrics, predictions, summary, and figures under `results/`.

**Reproduction notes**

- Run commands from the repository root so relative paths resolve correctly.
- The experiments use a fixed random seed of 42 and group questions by `topic_id` to prevent question overlap between training and test data.
- The semantic paraphrase evaluation uses a 300-question paired sample. Its results should not be compared directly with the larger supplementary surface-edit evaluation.
- The paraphrase-generation step is separate from evaluation. If the saved paraphrases already exist, do not rerun generation unnecessarily.
- Re-running stages may overwrite generated files. Preserve a copy of validated outputs before rerunning experiments.

To inspect the saved results without rerunning the experiments, open the CSV files and figures under `results/`.
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
