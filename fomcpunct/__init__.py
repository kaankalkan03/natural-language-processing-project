"""Punctuation restoration for FOMC press conference transcripts."""

LABELS = ("O", "COMMA", "PERIOD", "QUESTION")
PUNCT_LABELS = LABELS[1:]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
