# AI Text Detection Robustness

**Course:** AI26501 Modern AI Assignments (SP-2026)  
**Assignment:** 16 — AI Text Detection Robustness

## Project overview

This project evaluates a TF-IDF + LinearSVC classifier trained to distinguish human-written passages from AI-generated passages in the English HC3 dataset. It examines original AI text, a phrase-removal surface-edit condition, a separate connective-substitution condition, and generated semantic paraphrases.

The central question is how this detector behaves under these specific text transformations. The results do not establish how all AI detectors perform.

## Main paired evaluation

The main evaluation uses 300 held-out question pairs. Each condition contains 300 human passages and 300 AI passages, and the same human controls are reused across conditions. The detector is trained on original training text only. The topic-grouped split uses random seed 42 and has no question overlap between training and test data.

| Metric | Original AI | Phrase-removal surface edit | Semantic paraphrase |
|---|---:|---:|---:|
| Accuracy | 98.83% | 98.83% | 89.67% |
| AI precision | 99.66% | 99.66% | 99.58% |
| AI recall | 98.00% | 98.00% | 79.67% |
| AI F1 score | 98.82% | 98.82% | 88.52% |
| False-positive rate | 0.33% | 0.33% | 0.33% |

AI recall was 18.33 percentage points lower for the generated paraphrases than for the original AI passages; false negatives increased from 6 to 61. This is an observed difference for this sample and generation pipeline, not proof that meaning-preserving paraphrasing alone caused the difference. Paraphrases were often shorter, and some changed facts or omitted information. A length-matched truncation control is available in `src/analyze_paraphrase_length.py`; run it before interpreting the semantic condition causally.

**Important surface-edit qualification:** in the main paired evaluation, the phrase-removal transformation changed 0 of the 300 AI passages. Its identical metrics therefore do not demonstrate robustness to effective surface changes. The supplementary experiment below uses a different, deterministic connective-substitution edit.

## Supplementary connective-substitution experiment

The separate `src/run_robustness.py` experiment applies whitespace/punctuation normalization and, where possible, substitutes one discourse connective (for example, “However” with “Nevertheless”). It evaluates 4,959 held-out original AI passages, their edited counterparts, and 10,198 human passages as separate class-specific groups.

| Evaluation group | Passages | Accuracy | AI recall | False-positive rate |
|---|---:|---:|---:|---:|
| Original AI | 4,959 | 98.37% | 98.37% | Not applicable |
| Connective-edited AI | 4,959 | 98.02% | 98.02% | Not applicable |
| Human | 10,198 | 99.26% | Not applicable | 0.74% |

These are class-specific groups, not balanced human-versus-AI evaluations. False-positive rate is not meaningful for AI-only groups, and AI recall is not meaningful for the human-only group. The paired prediction file records how often the model's prediction changed between original and edited versions.

## Repository structure

- `data/` — prepared HC3 dataset and data notes.
- `notebooks/` — exploratory notebook.
- `src/` — preparation, training, evaluation, and robustness scripts.
- `results/` — saved predictions, metrics, summaries, and figures.
- `reports/` — final written report.
- `presentation/` — final presentation.
- `requirements.txt` — Python dependencies.

## Environment and reproduction

Developed with Python 3.12 on macOS. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run stages from the repository root so relative paths resolve correctly.

1. Prepare the dataset:
   ```bash
   python src/prepare_hc3.py
   ```
   This creates `data/hc3_original.csv`; internet access may be needed to download HC3.
2. Run baseline models:
   ```bash
   python src/run_experiment.py
   ```
3. Run the supplementary connective-edit experiment:
   ```bash
   python src/run_robustness.py
   ```
4. Generate semantic paraphrases (model download and substantial runtime required on first run):
   ```bash
   python src/generate_paraphrases.py
   ```
5. Evaluate the paired paraphrase sample:
   ```bash
   python src/evaluate_final.py
   ```
6. Evaluate the length-matched truncation control and paired uncertainty:
   ```bash
   python src/analyze_paraphrase_length.py
   ```

The generation step uses `humarin/chatgpt_paraphraser_on_T5_base`. The analysis script uses the same grouped split and detector settings as the main paired evaluation. Do not regenerate paraphrases unless needed; generation settings and library/model revisions can affect outputs. Inspect the generated text for meaning changes before interpreting it as a semantic robustness test.

## Limitations

- The study uses one dataset and one classical detector; results may not generalize to other datasets, domains, or detectors.
- HC3 source representation is imbalanced, so aggregate metrics are strongly influenced by the dominant sources. Small source-specific groups are unstable.
- Paraphrases are generated model outputs, not verified meaning-preserving rewrites. Length/compression and semantic changes are potential confounders.
- A single grouped holdout is used; confidence intervals and paired tests quantify uncertainty within this sample but do not replace external validation.
- Detector outputs are not proof of authorship.

## Main deliverables

- `reports/Assignment16_Final_Report.docx`
- `presentation/Group16_AI_Text_Detection_Robustness.pptx`
- Metrics, predictions, and figures under `results/`
