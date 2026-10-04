# Data Collection and Schema

## About the data

We use `sample_data.csv` only to check whether the code can load and validate the data. It is a sample file, so we do not use it for model training or for reporting the final results.

## How we plan to collect the data

We aim to cover around 12–20 topics in the dataset. For each topic, we will try to collect 1–2 human-written passages and 1–2 original AI-generated passages. We will also prepare an edited version of each AI passage.

These numbers are only an initial target. We would rather have passages covering different topics than collect many passages that say almost the same thing.

## Human-written text

For the human-written part, we will use text from participants who agree to take part in the study, or from sources that allow us to use their text. We will keep a record of the source, collection date, and any relevant consent or licence details.

We will label a passage as human-written only when we have a reasonable basis for confirming where it came from.

## AI-generated text

To keep the collection process consistent, we will use the same prompt format for each topic. For example:

> Write a factual explanatory passage of 120–180 words about [TOPIC] for a general university audience. Use a neutral tone, avoid headings and bullet points, and do not mention that you are an AI.

We will record which model or provider generated each passage and when it was created. If we use more than one AI system, we will keep track of which system produced each passage so the results can be compared properly.

## Edited passages

We will make one lightly edited or paraphrased version of each original AI passage. The aim is to change some wording and sentence structure without changing the original meaning or adding new facts.

We will also record how each passage was edited, such as whether the changes were made manually or with AI assistance. If we use different editing methods, we will document them rather than treating them all as the same process.

## CSV format

Each row in the dataset will represent one passage. The CSV file will contain these columns:

`passage_id, topic_id, condition, label, text, parent_id, provenance, generator, edit_method`

The columns are used as follows:

- `passage_id`: ID assigned to each passage.
- `topic_id`: Topic covered by the passage.
- `condition`: Whether the passage is `human`, `ai_original`, or `ai_edited`.
- `label`: `0` for human-written text and `1` for AI-origin text, including edited AI passages.
- `text`: Actual passage content.
- `parent_id`: ID of the original AI passage linked to an edited version.
- `provenance`: Where the passage came from and its source details.
- `generator`: AI model or provider used, where applicable.
- `edit_method`: How the passage was edited, where applicable.

When we split the data for training and testing, passages from the same topic will stay in the same split. This helps prevent closely related passages from appearing in both sets and giving us an overly optimistic evaluation.