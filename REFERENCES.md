# References and implementation notes

1. **HC3 dataset:** Guo, B. et al. (2023). *How Close is ChatGPT to Human Experts? Comparison Corpus, Evaluation, and Detection*. Hugging Face dataset repository: https://huggingface.co/datasets/Hello-SimpleAI/HC3
2. **Semantic paraphraser:** `humarin/chatgpt_paraphraser_on_T5_base`, Hugging Face model repository: https://huggingface.co/humarin/chatgpt_paraphraser_on_T5_base
3. **TF-IDF and LinearSVC:** scikit-learn documentation: https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction and https://scikit-learn.org/stable/modules/svm.html#classification
4. **Grouped evaluation:** scikit-learn `GroupShuffleSplit` documentation: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html

## Reproducibility cautions

- The dataset is loaded from a converted Parquet revision in the preparation script; record the resolved dataset revision and package versions when rerunning.
- The paraphraser's model repository name is recorded in generated rows, but exact output reproducibility can depend on model revision, Transformers/PyTorch versions, and generation settings.
- The 300-pair semantic-paraphrase evaluation is a selected pilot constrained by input token length, not a representative sample of all HC3 text.
- The paired-pilot phrase-removal transformation and supplementary discourse-marker substitution are distinct procedures. Do not call both simply “surface editing” without naming the method.
- The generated paraphrases have not been validated as meaning-preserving. Changes in detector recall may reflect length, omissions, altered meaning, or lexical distribution.
