# Berkay Hakan Sadi

- Student ID: 2309011065
- GitHub username: `hakanstarlord`
- Record date: 23 October 2026, updated 2 November 2026 (drafted from the
  project's session notes)

## Responsibility area

Project lead: choice of task and data, model and evaluation decisions, the
transformer runs on Colab, the public repository, the answers to the
instructor's dataset questions (saturation check and human agreement check)
and the split of the remaining work in the team. The development work was done
in sessions with an AI assistant that the member directed.

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
- 24 October 2026: uploaded the project to the team's public repository (the
  first project commit on top of Kaan's initial commit) and entered Group 8's
  dataset and task in the instructor's "NLP Datasets" sheet.
- 26 October 2026: had the README terms aligned with the Week 6
  token-classification slides and pushed the change (`README.md`).
- 27 October 2026: added the raw data source URL to the instructor's sheet
  and, to answer the instructor's questions on the dataset (only four labels,
  saturation, source of truth), decided to keep the four labels and to add a
  saturation check and a human agreement check (`fomcpunct/saturation.py`,
  `fomcpunct/agreement.py`, `annotation/`).
- 29 October 2026: split the remaining work in the team (two annotators, a
  fresh-setup test, a literature search, the final deck) and had a draft deck
  with speaker notes prepared, on which the final deck builds.
- 30–31 October 2026: checked Aamer's reproduction report and added the data
  rebuild note to the README; checked Moath's sources against the papers and
  added the Related work section (`README.md`).
- 1 November 2026: sent the annotation sheets to the two annotators, scored
  the human agreement check (`results/agreement/`), checked Kaan's deck (chart
  labels, slide 5 values and speaker note) and prepared study notes and the
  speaker script for the presentation.
- 2 November 2026: committed and pushed the human agreement check, the
  saturation report and the README updates.
