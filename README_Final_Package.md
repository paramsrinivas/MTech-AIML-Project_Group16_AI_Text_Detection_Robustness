# Assignment 16 — AI Text Detection Robustness

## Deliverables
- `Group16_AI_Text_Detection_Robustness.pptx` — 10-slide presentation
- `Assignment16_Final_Report.docx` — written report (to be generated in the next step)
- `figures/` — charts used in the presentation/report

## Main results
Controlled paired evaluation: 300 held-out question pairs, 600 passages per condition.
- Original AI: accuracy 98.83%, AI recall 98.00%, AI F1 98.82%, FPR 0.33%
- Surface-edited AI: same reported aggregate metrics as original
- Semantic paraphrase: accuracy 89.67%, AI recall 79.67%, AI F1 88.52%, FPR 0.33%

AI recall fell by 18.33 percentage points after semantic paraphrasing; false negatives rose from 6 to 61.

## Caveat
Manual inspection found paraphrases that changed facts or omitted information. The paraphrase results measure robustness to this paraphraser's outputs, not exclusively meaning-preserving paraphrases.

## Before submission
1. Add team member names.
2. Confirm README and requirements reflect final scripts/dependencies.
3. Retain source code, result CSV/JSON, figures, and data preparation instructions.
4. Verify slides render and rehearse for approximately 20 minutes.
