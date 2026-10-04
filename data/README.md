# Data Description

## Overview

This project studies how well an AI text detector performs when AI-generated passages are modified. The experiments use the English portion of the **HC3 (Human ChatGPT Comparison Corpus)** dataset, which contains questions, human-written answers, and ChatGPT-generated answers.

The dataset is loaded from Hugging Face using `Hello-SimpleAI/HC3`, with the `default` configuration and the `refs/convert/parquet` revision.

## Files

- `hc3_original.csv` — the prepared dataset containing the original human and AI passages.
- `sample_data.csv` — a sample dataset used for inspection or testing.
- `../results/semantic_paraphrases.csv` — generated semantic paraphrases used in the robustness evaluation.

The semantic paraphrases are stored under `results/`, rather than in the `data/` directory.

## Data Preparation

The script `src/prepare_hc3.py` prepares the original HC3 data for the experiments.

During preparation, the script:

- Loads the dataset from Hugging Face.
- Extracts the question, human answers, ChatGPT answers, and source information.
- Cleans whitespace while preserving the passage text otherwise.
- Uses a minimum text length of 40 characters.
- Assigns label `0` to human-written text and label `1` to original AI-generated text.
- Uses SHA-256 hashes to support consistent identification of text passages.

The prepared data is saved to `data/hc3_original.csv`. Edited and paraphrased passages are generated separately during later experiment stages.

## Text Variations

The project evaluates three forms of AI-generated text:

1. **Original AI text:** The ChatGPT answers from the prepared HC3 dataset.
2. **Surface-edited AI text:** The script `src/run_robustness.py` applies simple changes, including whitespace normalization and removal of selected introductory phrases. It does not perform semantic paraphrasing.
3. **Semantic paraphrases:** The script `src/generate_paraphrases.py` uses the Hugging Face Transformers library and the `humarin/chatgpt_paraphraser_on_T5_base` model to generate alternative versions of selected AI passages.

The paraphrasing script uses random seed `42`, a test split of `20%`, a sample size of `300`, and a maximum input length of `220` tokens.

## Evaluation Notes

The semantic paraphrases are generated model outputs. They may change the original meaning or introduce factual errors, so they should not be treated as verified, meaning-preserving rewrites.

The project compares detector performance across the original, surface-edited, and semantically paraphrased conditions. The evaluation scripts and result files contain the corresponding metrics.

## Reproducibility

The main preparation and paraphrasing scripts are located in `src/`. The Hugging Face dataset and model are specified in those scripts. The fixed random seed helps make sampling reproducible, although exact generated paraphrases can depend on the software environment and generation settings.

