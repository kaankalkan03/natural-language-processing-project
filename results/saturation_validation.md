# Saturation check: validation

66,089 words, 650 speaker turns, 8 press conferences. 83.9% of words are O, so predicting no punctuation at all already gives that accuracy. Macro-F1 is over COMMA, PERIOD and QUESTION. Turn exact match is the share of turns with every label right. 98 turns are a single word (96 of them from the moderator, who mostly names the next reporter); they are easy to get right and raise exact match, so the last column leaves them out. 95% CIs: cluster bootstrap over press conferences. Computed from the saved predictions; no model was retrained.

With only 8 conferences the intervals are too narrow (see the README); read them as indicative.

| System | Accuracy [95% CI] | Macro-F1 [95% CI] | Turn exact match [95% CI] | Exact match, turns of 2+ words |
|---|---|---|---|---|
| all O (no model) | 0.839 [0.831, 0.847] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| heuristic | 0.876 [0.871, 0.880] | 0.291 [0.270, 0.311] | 0.237 [0.222, 0.252] | 0.101 [0.082, 0.127] |
| hmm | 0.857 [0.849, 0.864] | 0.263 [0.245, 0.280] | 0.226 [0.210, 0.239] | 0.089 [0.073, 0.108] |
| crf | 0.929 [0.927, 0.931] | 0.608 [0.599, 0.616] | 0.235 [0.221, 0.247] | 0.100 [0.082, 0.123] |
| bilstm_softmax_s0 | 0.939 [0.937, 0.941] | 0.758 [0.745, 0.771] | 0.283 [0.273, 0.295] | 0.156 [0.136, 0.175] |
| bilstm_softmax_s1 | 0.941 [0.939, 0.943] | 0.767 [0.751, 0.786] | 0.291 [0.276, 0.306] | 0.165 [0.149, 0.181] |
| bilstm_softmax_s2 | 0.939 [0.937, 0.942] | 0.755 [0.745, 0.764] | 0.288 [0.273, 0.301] | 0.161 [0.135, 0.191] |
| bilstm_crf_s0 | 0.939 [0.937, 0.941] | 0.755 [0.741, 0.768] | 0.288 [0.277, 0.299] | 0.161 [0.140, 0.183] |
| bilstm_crf_s1 | 0.941 [0.938, 0.943] | 0.764 [0.748, 0.781] | 0.286 [0.264, 0.308] | 0.159 [0.127, 0.194] |
| bilstm_crf_s2 | 0.940 [0.938, 0.943] | 0.761 [0.750, 0.774] | 0.285 [0.272, 0.297] | 0.158 [0.134, 0.186] |
| bert_s0 | 0.963 [0.961, 0.964] | 0.848 [0.839, 0.858] | 0.345 [0.317, 0.375] | 0.228 [0.200, 0.259] |
| bert_s1 | 0.964 [0.962, 0.965] | 0.851 [0.838, 0.865] | 0.342 [0.314, 0.372] | 0.225 [0.195, 0.254] |
| bert_s2 | 0.963 [0.961, 0.965] | 0.850 [0.840, 0.862] | 0.340 [0.325, 0.359] | 0.223 [0.204, 0.243] |
| deberta_s0 | 0.971 [0.970, 0.973] | 0.881 [0.870, 0.893] | 0.365 [0.344, 0.387] | 0.252 [0.219, 0.288] |
| deberta_s1 | 0.971 [0.969, 0.972] | 0.880 [0.871, 0.889] | 0.362 [0.349, 0.376] | 0.248 [0.230, 0.268] |
| deberta_s2 | 0.970 [0.968, 0.972] | 0.875 [0.862, 0.888] | 0.355 [0.341, 0.370] | 0.241 [0.216, 0.268] |
