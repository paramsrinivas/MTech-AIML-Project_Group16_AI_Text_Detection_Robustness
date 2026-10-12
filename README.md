# AI Text Detection Robustness

**Course:** AI26501 Modern AI Assignments (SP-2026)  
**Assignment:** 16 — AI Text Detection Robustness

## Research question

How does a TF-IDF + LinearSVC detector trained on original HC3 passages behave when held-out AI passages are altered? This repository contains a baseline evaluation, two different lightweight-edit procedures, and a semantic-paraphrase pilot.

## Dataset and split

The project uses the English HC3 corpus from `Hello-SimpleAI/HC3`. The preparation script writes `data/hc3_original.csv`. Experiments retain questions with both human and original AI passages and use `GroupShuffleSplit` with seed 42 and a 20% test split, grouping by `topic_id`. The reported split has 60,955 training passages, 15,157 test passages, and zero overlapping question IDs.

Labels are 0 for human text and 1 for AI-origin text. The main detector uses word-level TF-IDF unigrams/bigrams and LinearSVC. Results describe this dataset, split, model, and edit pipeline; they do not establish universal detector performance.

## Main paired pilot

The paired pilot uses 300 held-out question pairs. Each condition contains 300 human controls and 300 AI passages. The same human controls are paired with original AI text, a phrase-removal variant, or model-generated semantic paraphrases.

| Metric | Original AI | Phrase-removal variant | Semantic paraphrase |
|---|---:|---:|---:|
| Accuracy | 98.83% | 98.83% | 89.67% |
| AI precision | 99.66% | 99.66% | 99.58% |
| AI recall | 98.00% | 98.00% | 79.67% |
| AI F1 score | 98.82% | 98.82% | 88.52% |
| False-positive rate | 0.33% | 0.33% | 0.33% |

AI recall is 18.33 percentage points lower for generated paraphrases than for original AI text (294/300 versus 239/300 detected). False negatives increase from 6 to 61.

**Important qualification:** the phrase-removal function used in the paired pilot removes a matching opening phrase from a small list (for example, “Certainly,” or “Sure,”). In the saved sample it changed 0 of 300 passages, so identical metrics are expected and do not demonstrate robustness to effective edits. The semantic paraphraser may shorten passages, omit information, or alter facts. These results therefore do not isolate the effect of meaning-preserving rewriting from length/compression or content changes.

## Supplementary edit experiment

`src/run_robustness.py` is a separate, larger held-out experiment. Its edit procedure normalizes whitespace and makes at most one deterministic discourse-marker substitution (for example, “However,” to “Nevertheless,”). It is not the same transformation as the phrase-removal function in the paired pilot.

| Evaluation group | Passages | Accuracy / AI recall | False-positive rate |
|---|---:|---:|---:|
| Original AI-only group | 4,959 | 98.37% | Not applicable |
| Edited AI-only group | 4,959 | 98.02% | Not applicable |
| Human-only group | 10,198 | 99.26% accuracy | 0.74% |

These are class-specific groups, not balanced human-versus-AI evaluations. FPR is undefined for AI-only groups; AI recall is not informative for the human-only group. In the paired AI predictions, the detection rate changed from 98.37% to 98.02%, a 0.34 percentage-point decrease, and 17/4,959 predictions changed. This is a small observed effect for this specific deterministic edit rule.

## Reproduction

Use Python 3.12 or a compatible Python environment. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python src/prepare_hc3.py
python src/run_experiment.py
python src/run_robustness.py
python src/generate_paraphrases.py
python src/evaluate_final.py
```

The preparation and model-generation steps need internet access. The paraphrase model is `humarin/chatgpt_paraphraser_on_T5_base`; first-time downloads and CPU/MPS inference may take time. If validated paraphrases already exist, preserve them and skip generation unless intentionally regenerating the sample.

The notebook is a results-inspection/illustration companion, not the canonical end-to-end experiment runner. The scripts in `src/` are the reproducibility entry points. Outputs are written under `results/`.

## Limitations

- One dataset, one principal detector family, and one grouped split; estimates may vary on other datasets, domains, and models.
- The 300-pair pilot is a subset selected from held-out AI passages no longer than 220 tokenizer tokens. It is not a random sample of every HC3 passage.
- Source composition is imbalanced and dominated by `reddit_eli5`; small source-level groups are not stable evidence for source-specific claims.
- Generated paraphrases are not manually verified as meaning-preserving. Some are substantially shorter and may omit or alter content.
- The supplementary edit test and paired pilot use different edit procedures and different evaluation sets. Do not merge their metrics into one direct comparison.
- No claim is made that AI detectors can reliably establish authorship for an individual passage.

## Deliverables

- `reports/Assignment16_Final_Report.docx`
- `presentation/Group16_AI_Text_Detection_Robustness.pptx`
- Metrics, predictions, summaries, and figures under `results/`
