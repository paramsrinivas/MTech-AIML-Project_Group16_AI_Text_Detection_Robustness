# Assignment 16 — AI-Generated Text Detection Under Light Editing

**Course:** AI26501 Modern AI Assignments (SP-2026)  
**Project:** Human vs AI-generated text detection and robustness to editing  
**Execution environment:** macOS, CPU-first Python workflow

## Research question
How does light editing or paraphrasing affect a text classifier's ability to distinguish AI-generated passages from human-written passages?

## Hypotheses
- **H1:** A TF-IDF classifier can distinguish original AI passages from human-written passages above a majority-class baseline on held-out topics.
- **H2:** Detection performance on edited AI passages is lower than on original AI passages.
- **H3:** A model that looks strong under a random row split may perform worse when entire topics are held out.

These are hypotheses to test, not assumed results.

## Experimental conditions
1. `human`: human-written passages.
2. `ai_original`: AI-generated passages before editing.
3. `ai_edited`: lightly edited/paraphrased versions of the AI passages.

For the primary binary classification task, `human` is label `0`; both AI conditions are label `1`. Report original-AI and edited-AI results separately. Do not treat edited AI text as human text.

## Data integrity rules
- The supplied `data/sample_data.csv` is **demo-only** and must not be used to report project results.
- Real human passages must be written by consenting human contributors or obtained from a clearly documented permitted source. Preserve provenance and any relevant license/consent.
- Generate AI passages with a documented model/provider, date, and prompt.
- Edit only AI passages for the edited condition; retain parent IDs to link versions.
- Match topics and keep passage lengths reasonably comparable.
- Split by `topic_id` or source/document group, not by individual rows. Related variants must never cross train/test boundaries.
- Keep a data collection log and do not include personal or confidential information.

## Repository layout
```text
MTech-AIML-Project_Group16_AI_Text_Detection_Robustness/
├── README.md
├── requirements.txt
├── data/
│   ├── README.md
│   └── sample_data.csv
├── notebooks/
│   └── experiment.ipynb
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   └── evaluation.py
├── results/
│   ├── results.csv
│   ├── comparison_table.csv
│   └── figures/
└── presentation/
    └── Group16_AI_Text_Detection_Robustness.pptx  # add after real experiments
```

## macOS setup
```bash
cd MTech-AIML-Project_Group16_AI_Text_Detection_Robustness
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m jupyter notebook
```

Open `notebooks/experiment.ipynb`. To run the core experiment as a script:
```bash
python -m src.model --data data/your_dataset.csv --out results
```

## Required dataset schema
One row per passage/version:
- `passage_id`: unique row ID
- `topic_id`: topic grouping used for group-wise split
- `condition`: `human`, `ai_original`, or `ai_edited`
- `label`: `0` for human, `1` for AI-origin
- `text`: passage text
- `parent_id`: original passage ID for edited versions; blank for original rows
- `provenance`: how the passage was obtained
- `generator`: AI model/provider for AI rows; blank for human rows
- `edit_method`: editing protocol for edited AI rows; blank otherwise

## Analysis protocol
- Primary split: group-wise 80/20 split by `topic_id`, with a fixed random seed. If the number of topics is too small for a stable holdout, collect more topics rather than silently switching to row-wise splitting.
- Models: TF-IDF word n-grams + Logistic Regression; TF-IDF word n-grams + Linear SVM.
- Fit all vectorizers and models on training data only.
- Evaluate on (a) human vs original AI and (b) human vs edited AI, using the same held-out topics.
- Metrics: accuracy, precision, recall, F1, false-positive rate, confusion matrix, and performance drop from original to edited AI.
- Add confidence intervals or repeated group splits if the dataset size permits.
- Save exact split IDs, predictions, model settings, package versions, and random seeds.

## Important limitations
AI-text detectors are not reliable proof of authorship. Results may depend on model family, topic, language, passage length, editing method, and source distribution. Do not claim universal detection ability from a small dataset. Avoid using detector scores for high-stakes judgments about individuals.

## Deliverables
Fill `results/results.csv`, `results/comparison_table.csv`, and `results/figures/` only after running the experiment on validated research data. Prepare the 8–10 slide presentation from actual results; do not invent metrics.
