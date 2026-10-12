# Data Description

## Dataset

The experiments use the English portion of **HC3 (Human ChatGPT Comparison Corpus)** from Hugging Face dataset `Hello-SimpleAI/HC3`, using the `default` configuration and `refs/convert/parquet` revision in the preparation script.

## Files

- `hc3_original.csv` — prepared human and original AI passages.
- `sample_data.csv` — sample data for inspection or testing.
- `../results/semantic_paraphrases.csv` — paired pilot sample and generated paraphrases.

## Preparation

`src/prepare_hc3.py` loads HC3, extracts question/answer and source information, normalizes whitespace, filters passages shorter than 40 characters, assigns label 0 to human text and label 1 to AI text, and creates stable identifiers/hashes. The output is `data/hc3_original.csv`.

## Editing conditions are not interchangeable

1. **Original AI:** original ChatGPT answers.
2. **Paired-pilot phrase-removal variant:** `src/generate_paraphrases.py` normalizes whitespace and removes one matching opening phrase from a short list (such as “Certainly,” or “Sure,”). In the saved 300-passage sample, this function changed 0 passages. Its identical metrics therefore do not establish robustness to effective edits.
3. **Supplementary deterministic edit:** `src/run_robustness.py` normalizes whitespace and applies at most one discourse-marker substitution (such as “However,” to “Nevertheless,”) where a listed phrase matches. This is a distinct transformation.
4. **Semantic paraphrase:** `src/generate_paraphrases.py` uses `humarin/chatgpt_paraphraser_on_T5_base`. These are generated outputs, not verified meaning-preserving rewrites.

## Split and sample

The main scripts use a question/topic-grouped `GroupShuffleSplit` with seed 42 and test size 0.20. The paired pilot selects up to 300 held-out AI passages that meet the model tokenizer limit of 220 tokens and joins each with a human passage from the same question. This eligibility filter limits how broadly the pilot sample can be generalized.

## Extended paired evaluation

`src/run_extended_evaluation.py` evaluates five paired conditions using the same 300 held-out question topics: original AI, the legacy phrase-removal variant, a deterministic discourse-marker light edit, T5 semantic paraphrase, and a truncation/compression control. It writes extended metrics, exact paired McNemar tests, per-passage predictions, word-length analysis, and a JSON summary under `results/`. The truncation control cuts the original AI passage to the paraphrase's whitespace-word count when the original is longer; for one pair (topic 11934), the original has 31 words and the paraphrase has 33, so the original is left unchanged. This is disclosed in the summary and means one length mismatch remains.

## Interpretation

Generated paraphrases can omit details, alter facts, and be shorter than their source. Detector performance on them reflects both the paraphrasing process and resulting text characteristics; it does not isolate meaning-preserving semantic rewriting alone. The supplementary deterministic-edit experiment is a different sample and should be reported separately.

## Reproducibility

The dataset revision, model name, seed, split size, and sample size are defined in the scripts. The preparation and generation stages require internet access on first use. Model output can vary with library versions and runtime settings.
