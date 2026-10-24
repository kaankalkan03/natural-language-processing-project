"""Split cleaned text into words and derive one punctuation label per word.

A word's label is the punctuation that follows it in the transcript:

    COMMA     , : and dashes (pauses inside a sentence)
    PERIOD    . ! ;  (sentence ends)
    QUESTION  ?
    O         nothing

The model input is the lower-cased word with surrounding punctuation removed.
Periods that belong to abbreviations (U.S., a.m., Mr.) stay inside the word.
"""

from __future__ import annotations

import re
from collections import Counter

TRAILING = set(",.?!;:—")
ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "st", "jr", "sr", "vs", "ex"}  # "ex." = excluding
DOTTED = re.compile(r"(?:[A-Za-z]{1,2}\.)+[A-Za-z]{1,2}")  # U.S, a.m, e.g, Ph.D
# After an abbreviation such as "U.S." the period may also end the sentence.
# We only call it a sentence end when the next word is a common sentence opener.
SENTENCE_OPENERS = {"i", "we", "the", "and", "so", "but", "it", "that", "this", "there",
                    "they", "you", "in", "as", "our", "if", "what", "how", "now", "well",
                    "let", "my", "those", "these", "he", "she", "thank", "thanks", "yes", "no"}


def mark_to_label(marks: str) -> str:
    if "?" in marks:
        return "QUESTION"
    if any(c in marks for c in ".!;"):
        return "PERIOD"
    if any(c in marks for c in ",:—"):
        return "COMMA"
    return "O"


def is_abbreviation(word: str) -> bool:
    return bool(DOTTED.fullmatch(word)) or word.lower() in ABBREVIATIONS


def split_words(text: str, log: Counter | None = None) -> list[tuple[str, str]]:
    """Return (word, marks) pairs; `word` keeps its case, `marks` is the raw trailing punctuation."""
    log = log if log is not None else Counter()
    pairs: list[list] = []  # [word, marks, is_abbreviation]
    for token in text.split():
        lead = ""
        while token and token[0] in TRAILING:
            lead += token[0]
            token = token[1:]
        if lead:  # marks written before a word (", and") or a token of marks only ("—")
            kind = "leading_marks" if token else "standalone_marks"
            if pairs:
                pairs[-1][1] += lead
                log[f"{kind}_{'moved' if token else 'merged'}"] += 1
            else:
                log[f"{kind}_dropped"] += 1
            if not token:
                continue
        word, marks = token, ""
        while word and word[-1] in TRAILING:
            marks = word[-1] + marks
            word = word[:-1]
        abbreviation = False
        if marks.startswith(".") and is_abbreviation(word):
            word, marks, abbreviation = word + ".", marks[1:], True
            log["abbreviation_periods_kept"] += 1
        if not word:
            if pairs:
                pairs[-1][1] += marks
                log["standalone_marks_merged"] += 1
            else:
                log["standalone_marks_dropped"] += 1
            continue
        pairs.append([word, marks, abbreviation])

    for i, (word, marks, abbreviation) in enumerate(pairs):
        if abbreviation and not marks and i + 1 < len(pairs):
            following = pairs[i + 1][0]
            if following[0].isupper() and following.lower() in SENTENCE_OPENERS:
                pairs[i][1] = "."
                log["abbreviation_sentence_ends"] += 1
    return [(word, marks) for word, marks, _ in pairs]


def to_example(text: str, log: Counter | None = None) -> tuple[list[str], list[str], list[str]]:
    """Return lower-cased input words, labels and raw marks for one turn."""
    pairs = split_words(text, log)
    words = [word.lower() for word, _ in pairs]
    marks = [marks for _, marks in pairs]
    labels = [mark_to_label(m) for m in marks]
    return words, labels, marks
