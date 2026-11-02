# Human agreement check

20 test turns, 2,227 words (10 chair and 10 reporter turns of 40-200 words, one per conference and role, seed 0). Annotators: A, B. Model labels are the saved test predictions; no model was retrained. 95% CIs: percentile bootstrap over turns (1,000 resamples).

F1 and kappa are symmetric, so the order of a pair does not matter. Within-turn macro-F1 leaves out the last word of each turn (its sentence end is given by the turn boundary). The models were trained on transcripts punctuated in the same house style, so they are expected to agree with the transcript more than a person does; the annotator pair is the human-to-human reference.

Marks used on these words:

| Source | COMMA | PERIOD | QUESTION |
|---|---|---|---|
| transcript | 228 | 126 | 15 |
| A | 120 | 131 | 17 |
| B | 209 | 116 | 15 |
| deberta_s0 | 215 | 134 | 14 |
| deberta_s1 | 230 | 125 | 13 |
| deberta_s2 | 235 | 124 | 14 |
| crf | 157 | 115 | 9 |

Agreement:

| Pair | Macro-F1 [95% CI] | Within-turn | COMMA | PERIOD | QUESTION | Mark detection | Kappa [95% CI] |
|---|---|---|---|---|---|---|---|
| transcript vs A | 0.787 [0.731, 0.836] | 0.777 | 0.598 | 0.887 | 0.875 | 0.826 | 0.752 [0.706, 0.808] |
| transcript vs B | 0.865 [0.825, 0.898] | 0.849 | 0.819 | 0.843 | 0.933 | 0.931 | 0.866 [0.836, 0.894] |
| A vs B | 0.795 [0.734, 0.849] | 0.784 | 0.644 | 0.866 | 0.875 | 0.849 | 0.776 [0.729, 0.833] |
| transcript vs deberta_s0 | 0.802 [0.743, 0.848] | 0.776 | 0.808 | 0.838 | 0.759 | 0.934 | 0.858 [0.826, 0.883] |
| A vs deberta_s0 | 0.762 [0.696, 0.819] | 0.765 | 0.579 | 0.868 | 0.839 | 0.811 | 0.736 [0.681, 0.796] |
| B vs deberta_s0 | 0.845 [0.785, 0.899] | 0.833 | 0.835 | 0.872 | 0.828 | 0.933 | 0.876 [0.836, 0.912] |
| transcript vs deberta_s1 | 0.789 [0.726, 0.854] | 0.755 | 0.825 | 0.829 | 0.714 | 0.936 | 0.862 [0.836, 0.886] |
| A vs deberta_s1 | 0.732 [0.662, 0.804] | 0.718 | 0.594 | 0.867 | 0.733 | 0.808 | 0.732 [0.670, 0.800] |
| B vs deberta_s1 | 0.831 [0.751, 0.906] | 0.811 | 0.843 | 0.863 | 0.786 | 0.932 | 0.875 [0.835, 0.912] |
| transcript vs deberta_s2 | 0.826 [0.782, 0.870] | 0.787 | 0.825 | 0.824 | 0.828 | 0.938 | 0.864 [0.837, 0.888] |
| A vs deberta_s2 | 0.771 [0.723, 0.823] | 0.754 | 0.603 | 0.871 | 0.839 | 0.808 | 0.738 [0.685, 0.797] |
| B vs deberta_s2 | 0.859 [0.811, 0.907] | 0.835 | 0.838 | 0.842 | 0.897 | 0.931 | 0.871 [0.836, 0.905] |
| transcript vs crf | 0.576 [0.480, 0.662] | 0.578 | 0.582 | 0.647 | 0.500 | 0.788 | 0.659 [0.615, 0.694] |
| A vs crf | 0.519 [0.416, 0.603] | 0.542 | 0.361 | 0.659 | 0.538 | 0.670 | 0.546 [0.491, 0.604] |
| B vs crf | 0.558 [0.479, 0.627] | 0.544 | 0.607 | 0.649 | 0.417 | 0.799 | 0.673 [0.634, 0.708] |

## Model minus annotator, both scored against the transcript

Positive: the model's punctuation is closer to the transcript than the person's. Paired bootstrap over the same turns.

| Model | Annotator | Difference in macro-F1 [95% CI] |
|---|---|---|
| deberta_s0 | A | +0.015 [-0.040, +0.065] |
| deberta_s0 | B | -0.063 [-0.111, -0.026] |
| deberta_s1 | A | +0.003 [-0.063, +0.076] |
| deberta_s1 | B | -0.076 [-0.137, -0.011] |
| deberta_s2 | A | +0.039 [-0.014, +0.092] |
| deberta_s2 | B | -0.040 [-0.083, +0.002] |
| crf | A | -0.210 [-0.305, -0.123] |
| crf | B | -0.289 [-0.386, -0.192] |
