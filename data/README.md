# Data Description

## Overview

The project uses the English HC3 (Human ChatGPT Comparison Corpus) dataset, which contains questions, human-written answers, ChatGPT-generated answers, and source information. The dataset is loaded from Hugging Face using `Hello-SimpleAI/HC3`, configuration `default`, revision `refs/convert/parquet`.

## Files

- `hc3_original.csv` — prepared original human and AI passages.
- `sample_data.csv` — small sample used for inspection or testing.
- `../results/semantic_paraphrases.csv` — generated paraphrases and paired human/original AI text used in the main robustness evaluation.

## Data preparation

`src/prepare_hc3.py` loads HC3, extracts the question, human answer, ChatGPT answer, and source, performs whitespace cleaning, filters passages shorter than 40 characters, labels human passages `0` and original AI passages `1`, and creates stable identifiers using SHA-256 hashes.

The prepared dataset is saved to `data/hc3_original.csv`. Edited and paraphrased text is generated in separate experiment stages.

## Text variations

1. **Original AI text:** ChatGPT-generated answers from HC3.
2. **Phrase-removal surface edit (main paired experiment):** `src/generate_paraphrases.py` normalizes whitespace and punctuation and removes one matching introductory phrase from a short list, such as “Certainly,” or “Sure,”. In the saved 300-passage sample this transformation changed 0 passages.
3. **Connective-substitution edit (supplementary experiment):** `src/run_robustness.py` normalizes whitespace/punctuation and, where possible, substitutes one discourse connective, such as “However” with “Nevertheless.” This is a separate transformation from the phrase-removal edit above.
4. **Semantic paraphrases:** `src/generate_paraphrases.py` uses the Hugging Face Transformers model `humarin/chatgpt_paraphraser_on_T5_base` to generate alternate text. The configured seed is 42, test size is 20%, sample size is 300, and maximum source length is 220 tokenizer tokens.

## Evaluation notes

The main detector is trained only on original training passages. Question IDs are grouped during train/test splitting to prevent passages from the same question appearing in both sets. The semantic evaluation uses a 300-question paired sample, limited to AI passages no longer than 220 tokens. This eligibility rule restricts the population represented by the results.

Generated paraphrases may compress the original text, omit content, or change facts. They are not verified meaning-preserving rewrites. The length-matched truncation control in `src/analyze_paraphrase_length.py` is intended to help assess how much detection loss may be associated with reduced input length; truncation is a diagnostic control and is not a meaning-preserving edit.

## Reproducibility

The fixed seed helps make sampling reproducible, but exact generated outputs may depend on package versions, model/tokenizer revisions, generation settings, and hardware. Record those details when reproducing the paraphrase stage. Inspect examples for meaning preservation before drawing causal conclusions.
