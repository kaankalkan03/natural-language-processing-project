"""Turn a transcript PDF into cleaned speaker turns.

Steps (each one is counted in the cleaning log):
1. Extract page text with pdfplumber.
2. Drop the running header, the "Page X of N" footer and correction
   footnotes ("1 Chair Powell intended to say ...") with their markers.
3. Drop title lines before the first speaker label and split the text into
   speaker turns ("CHAIR POWELL.", "MICHELLE SMITH.", reporter names).
4. Repair line breaks, drop stage notes such as [Laughter], unwrap editorial
   insertions such as [percent], and normalize quotes, dashes and fractions.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pdfplumber

MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
HEADER = re.compile(rf"^(?:{MONTHS}) \d{{1,2}}, \d{{4}} .*Press ?Conference")
CHAIR_IN_HEADER = re.compile(r"Chair(?:man)? (\w+)’s")
FOOTER = re.compile(r"^(?:Page )?\d+ of \d+$")
FOOTNOTE = re.compile(r"^(\d{1,2}) (?:The )?Chair(?:man)?\b")
# Two to five upper-case words, or the 2011-era generic label QUESTION.
# Words may start with Mc/Mac (VICTORIA McGRANE) and use accented capitals (HONORÉ).
# Labels end with a period; two transcripts (2018-09-26, 2024-03-20) use a colon
# once, and one Chair label (2021-11-03) has no mark at all (SPEAKER_NO_MARK).
_LABEL_WORD = r"(?:Ma?c)?[A-ZÀ-ÖØ-Þ][A-ZÀ-ÖØ-Þ'’\-]+"
SPEAKER = re.compile(rf"^({_LABEL_WORD}(?: {_LABEL_WORD}){{1,4}}|QUESTION)[.:](?: |$)")
SPEAKER_NO_MARK = re.compile(r"^(CHAIR(?:MAN)? [A-Z][A-Z'’\-]+) (?=[A-Z])")
# A footnote marker follows punctuation that ends a word or a four-digit year,
# possibly with a closing quote ("economy.1", "in 2021.1", 'say.”1');
# decimals such as "2.5 percent" never match.
_MARKER = (r"(?:(?<=[A-Za-z)][.,?!;:])|(?<=[A-Za-z)][.,?!;:]”)|(?<=[A-Za-z)]”)"
           r"|(?<=\b\d{{4}}[.,?!;:])|(?<=\b\d{{4}}[.,?!;:]”))(?:{n})(?=\s|$)")
MODERATOR_LABELS = {"MICHELLE SMITH", "MICHELLE SMTH", "MICHELE SMITH", "JOE PAVEL"}
STAGE_NOTE_WORDS = ("laugh", "inaudible", "indaudible", "no response", "cough",
                    "aside", "crosstalk", "speaker unknown", "pause", "audio", "unintelligible",
                    "applause", "gesturing", "silence", "throat")
FRACTIONS = {"½": " 1/2", "¼": " 1/4", "¾": " 3/4", "⅛": " 1/8", "⅜": " 3/8",
             "⅝": " 5/8", "⅞": " 7/8", "⅓": " 1/3", "⅔": " 2/3"}


def read_pages(pdf_path: Path) -> list[list[str]]:
    with pdfplumber.open(pdf_path) as pdf:
        return [(page.extract_text() or "").splitlines() for page in pdf.pages]


def strip_page(lines: list[str], log: Counter, footnotes: list[str]) -> list[str]:
    """Drop header, footer and correction footnotes; append footnote numbers to `footnotes`."""
    lines = [line.rstrip() for line in lines if line.strip()]
    if lines and HEADER.match(lines[0]):
        lines = lines[1:]
        log["headers_removed"] += 1
    if lines and FOOTER.match(lines[-1].strip()):
        lines = lines[:-1]
        log["footers_removed"] += 1
    # Correction footnotes sit at the bottom of the page.
    for i in range(max(0, len(lines) - 8), len(lines)):
        if FOOTNOTE.match(lines[i]):
            numbers = [m.group(1) for m in map(FOOTNOTE.match, lines[i:]) if m]
            footnotes.extend(numbers)
            log["footnotes_removed"] += len(numbers)
            log["footnote_lines_removed"] += len(lines) - i
            lines = lines[:i]
            break
    return lines


def remove_footnote_markers(texts: list[str], footnotes: list[str], log: Counter) -> list[str]:
    """Remove the first marker of each removed footnote, in reading order."""
    texts = list(texts)
    for number in footnotes:
        pattern = re.compile(_MARKER.format(n=number))
        for i, text in enumerate(texts):
            texts[i], found = pattern.subn("", text, count=1)
            if found:
                log["footnote_markers_removed"] += 1
                break
        else:
            log["footnote_markers_not_found"] += 1
    return texts


def join_lines(lines: list[str]) -> str:
    text = ""
    for line in lines:
        line = line.strip()
        if not text:
            text = line
        elif re.search(r"[A-Za-z]-$", text) or text.endswith(("—", "–")):
            text += line  # compound word or dash split across lines
        else:
            text += " " + line
    return text


def role_of(label: str) -> str:
    if label.startswith("CHAI"):  # also catches typos such as CHAIMRAN
        return "chair"
    if label in MODERATOR_LABELS:
        return "moderator"
    return "reporter"


def _bracket(match: re.Match, log: Counter) -> str:
    inner = match.group(1)
    if any(word in inner.lower() for word in STAGE_NOTE_WORDS) or re.fullmatch(r"[\d:.]+", inner):
        # stage notes and audio time stamps such as [0:24:15.5]
        log["stage_notes_dropped"] += 1
        return " "
    log["editorial_insertions_unwrapped"] += 1
    return inner


def normalize(text: str, log: Counter) -> str:
    text = re.sub(r"\[([^\]]*)\]", lambda m: _bracket(m, log), text)
    # Keep legal references like 13(3); drop other parentheses but keep their words.
    text = re.sub(r"(\w)\((\w{1,3})\)", r"\1<\2>", text)
    text = text.replace("(", "").replace(")", "")
    text = re.sub(r"(\w)<(\w{1,3})>", r"\1(\2)", text)
    for symbol, replacement in FRACTIONS.items():
        text = text.replace(symbol, replacement)
    # Quotes: an opening quote typed inside a word is an apostrophe (we‘d);
    # a closing single quote after punctuation goes unless a letter follows (U.S.’s).
    text = re.sub(r"(?<=[A-Za-z])‘(?=[A-Za-z])", "’", text)
    text = re.sub(r"[“”\"‘®]", "", text)
    text = re.sub(r"(?<=[.,?!;:—–])’(?![A-Za-z])", "", text)
    text = text.replace("’", "'")
    # Ellipses mark omissions in quoted text or trailing off; they get no label.
    text = text.replace("…", " ")
    text = re.sub(r"(?<!\S)\.(?: \.){2}(?!\S)", " ", text)  # spaced ". . ."
    text = re.sub(r"(?<=[A-Za-z]),(?=[A-Za-z])", ", ", text)  # missing space after comma
    # An en dash between two word characters joins a compound or a range
    # (Dodd–Frank, U.S.–China, 2013–2014); spaced en dashes are pauses.
    text = re.sub(r"(?<=[\w.])–(?=\w)", "-", text)
    text = text.replace("–", "—").replace("--", "—")
    text = text.replace("—", "— ")  # a dash belongs to the word before it
    return re.sub(r"\s+", " ", text).strip()


def clean_pdf(pdf_path: Path) -> dict:
    """Return {"chair": surname, "turns": [{speaker, role, text}], "log": Counter}."""
    log: Counter = Counter()
    pages = read_pages(pdf_path)
    log["pages"] = len(pages)
    header = next((line for page in pages for line in page if HEADER.match(line)), "")
    chair_match = CHAIR_IN_HEADER.search(header)
    chair = chair_match.group(1) if chair_match else "UNKNOWN"

    footnotes: list[str] = []
    body = [line for page in pages for line in strip_page(page, log, footnotes)]
    turns: list[dict] = []
    for line in body:
        match = SPEAKER.match(line) or SPEAKER_NO_MARK.match(line)
        if match:
            label = match.group(1)
            turns.append({"speaker": label, "role": role_of(label), "lines": [line[match.end():]]})
        elif turns:
            turns[-1]["lines"].append(line)
        else:
            log["title_lines_dropped"] += 1

    texts = remove_footnote_markers([join_lines(t["lines"]) for t in turns], footnotes, log)
    cleaned = []
    for turn, text in zip(turns, texts):
        text = normalize(text, log)
        if not text:
            log["empty_turns_dropped"] += 1
            continue
        speaker = chair.upper() if turn["role"] == "chair" else turn["speaker"]
        cleaned.append({"speaker": speaker, "role": turn["role"], "text": text})
        log[f"turns_{turn['role']}"] += 1
    return {"chair": chair, "turns": cleaned, "log": log}
