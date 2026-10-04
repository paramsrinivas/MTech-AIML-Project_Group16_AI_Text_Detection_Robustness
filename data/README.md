# Data collection and schema

## Do not use the demo file as research evidence
`sample_data.csv` exists only to verify that the pipeline loads and validates data. It is not a valid human-vs-AI benchmark and must not be used to report accuracy or other results.

## Recommended manageable dataset
Start with 12–20 topics and aim for at least:
- 1–2 human passages per topic;
- 1–2 original AI passages per topic;
- 1 edited version for each original AI passage.

A larger set of topics is preferable to many near-duplicate passages from only a few topics. Treat these numbers as a starting target, not a guarantee of statistical power.

## Human passage sourcing
Use short passages written by consenting participants specifically for this study, or text from a source whose license and provenance permit the intended use. Record the source, date, and consent/license basis. Do not label text as human-written unless its provenance supports that label.

## AI passage protocol
Use the same documented prompt template across topics, for example:
> Write one factual explanatory passage of 120–180 words about [TOPIC] for a general university audience. Use a neutral tone, avoid headings and bullet points, and do not mention that you are an AI.

Record model/provider and date. If comparing different AI systems, record the generator and report this as a separate analysis.

## Editing protocol
Create one lightly edited/paraphrased version per AI passage. Define editing rules before editing, for example: preserve facts and meaning, change some sentence structures and wording, do not add new evidence, and target moderate rather than complete rewriting. Record whether editing was human-assisted or model-assisted. Do not mix editing methods without recording them.

## Required columns
`passage_id, topic_id, condition, label, text, parent_id, provenance, generator, edit_method`

- `condition`: `human`, `ai_original`, or `ai_edited`
- `label`: `0` human; `1` AI-origin (including edited AI)
- `parent_id`: original AI passage ID for edited versions
- Keep all versions from a topic in the same train/test partition.
