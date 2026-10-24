# Berkay Hakan Sadi

- Student ID: 2309011065
- GitHub username: `hakanstarlord`
- Record date: 23 October 2026 (drafted from the project's session notes; to be
  checked and completed by the member)

## Documented work

- 22 October 2026: chose the task and data after reviewing the instructor's
  project rubric: punctuation restoration as token classification on FOMC
  press conference transcripts (a ready-made NER dataset was considered first
  and dropped).
- 22–23 October 2026: directed the development sessions with the AI assistant
  and made the project decisions: classical baselines (heuristic, HMM with
  Viterbi, CRF) with bootstrap confidence intervals; a BiLSTM-CRF written from
  scratch and trained on CPU; BERT and DeBERTa-v3 on Google Colab; the no-CRF
  ablation, error analysis, figures and README.
- 23 October 2026: ran the BERT and DeBERTa-v3 notebook
  (`notebooks/transformers_colab.ipynb`) on Google Colab with an A100 GPU,
  decided to extend the learning-rate grid after the first search picked the
  largest rate, ran the extension and brought back both result archives.
- 23 October 2026: fixed all model choices and asked for the single
  evaluation on the test split.

## AI assistance

Most code, analysis and documentation were written by an AI assistant
(Claude, in Claude Code sessions) at this member's request and under this
member's direction. This record does not claim that the member wrote the code
by hand.
