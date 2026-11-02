# Saturation check: test

178,893 words, 1,823 speaker turns, 22 press conferences. 84.0% of words are O, so predicting no punctuation at all already gives that accuracy. Macro-F1 is over COMMA, PERIOD and QUESTION. Turn exact match is the share of turns with every label right. 303 turns are a single word (279 of them from the moderator, who mostly names the next reporter); they are easy to get right and raise exact match, so the last column leaves them out. 95% CIs: cluster bootstrap over press conferences. Computed from the saved predictions; no model was retrained.

| System | Accuracy [95% CI] | Macro-F1 [95% CI] | Turn exact match [95% CI] | Exact match, turns of 2+ words |
|---|---|---|---|---|
| all O (no model) | 0.840 [0.833, 0.847] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| heuristic | 0.873 [0.871, 0.876] | 0.257 [0.242, 0.272] | 0.231 [0.212, 0.249] | 0.080 [0.061, 0.099] |
| hmm | 0.856 [0.850, 0.861] | 0.256 [0.241, 0.270] | 0.225 [0.205, 0.244] | 0.073 [0.055, 0.092] |
| crf | 0.921 [0.919, 0.924] | 0.562 [0.543, 0.579] | 0.237 [0.220, 0.254] | 0.087 [0.070, 0.103] |
| bilstm_softmax_s0 | 0.932 [0.930, 0.935] | 0.728 [0.712, 0.745] | 0.275 [0.255, 0.294] | 0.132 [0.112, 0.152] |
| bilstm_softmax_s1 | 0.935 [0.932, 0.937] | 0.742 [0.726, 0.756] | 0.279 [0.261, 0.297] | 0.137 [0.120, 0.153] |
| bilstm_softmax_s2 | 0.934 [0.931, 0.937] | 0.735 [0.716, 0.753] | 0.275 [0.257, 0.293] | 0.137 [0.120, 0.153] |
| bilstm_crf_s0 | 0.932 [0.930, 0.935] | 0.732 [0.712, 0.748] | 0.272 [0.253, 0.289] | 0.128 [0.112, 0.145] |
| bilstm_crf_s1 | 0.934 [0.931, 0.937] | 0.739 [0.720, 0.756] | 0.279 [0.259, 0.296] | 0.136 [0.118, 0.153] |
| bilstm_crf_s2 | 0.934 [0.931, 0.937] | 0.737 [0.717, 0.755] | 0.283 [0.266, 0.300] | 0.141 [0.124, 0.157] |
| bert_s0 | 0.960 [0.958, 0.962] | 0.839 [0.829, 0.850] | 0.325 [0.304, 0.346] | 0.191 [0.169, 0.212] |
| bert_s1 | 0.961 [0.959, 0.962] | 0.840 [0.829, 0.851] | 0.318 [0.295, 0.340] | 0.184 [0.162, 0.204] |
| bert_s2 | 0.960 [0.958, 0.962] | 0.837 [0.826, 0.848] | 0.323 [0.304, 0.339] | 0.189 [0.170, 0.206] |
| deberta_s0 | 0.969 [0.968, 0.971] | 0.868 [0.857, 0.878] | 0.340 [0.319, 0.361] | 0.211 [0.186, 0.233] |
| deberta_s1 | 0.970 [0.968, 0.971] | 0.873 [0.863, 0.882] | 0.351 [0.330, 0.371] | 0.224 [0.203, 0.244] |
| deberta_s2 | 0.969 [0.968, 0.971] | 0.871 [0.861, 0.880] | 0.351 [0.330, 0.371] | 0.224 [0.201, 0.245] |
