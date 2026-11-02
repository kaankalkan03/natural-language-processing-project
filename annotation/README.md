# Human agreement check: instructions

Our labels come from one source: the punctuation in the Federal Reserve's
published transcripts. There is no second, "correct" version to check them
against. To estimate how noisy that single reference is, two team members
punctuate the same 20 test turns on their own, without seeing the original.
We then compare the two people with each other, with the transcript and with
the models.

- `annotator_A.txt`, `annotator_B.txt`: the two sheets (same text; each
  annotator takes one).
- `sample.json`: which turns were drawn (the key to the sheets).
- Scoring code: `fomcpunct/agreement.py`. Results go to `results/agreement/`.

## For the annotators

Time: about 30–45 minutes for 20 turns (2,227 words).

1. **Work alone.** Do not talk about the sheet with the other annotator until
   both of you are done.
2. **Do not look anything up.** Do not search for the transcripts or open
   `sample.json`, `data/` or `results/` before you finish. Do not use AI tools,
   grammar checkers or autocorrect. Plain Notepad or a code editor is best.
3. **Only type punctuation between the words.** Use `,` `.` `?`. If you would
   naturally use them, `;` `:` `!` and a dash are also fine; write the dash as
   ` -- ` (with spaces).
4. **Do not add, delete, reorder or re-spell words.** Leave words such as
   `u.s.`, `2.5`, `long-run`, `that's` and `$3,000` exactly as they are.
5. **Capital letters and line breaks do not matter.** You may capitalize if it
   helps you read, but it is ignored.
6. **Sentence end after an abbreviation:** type a second period, for example
   `the u.s.. we think`.
7. **Do not edit** the lines that start with `===` or `#`, and do not add
   separator lines (such as `---`) between blocks.
8. Punctuate each block the way you think a published transcript of that
   speech should look. Each block is one speaker's turn at an FOMC press
   conference (the Chair answering, or a reporter asking); blocks are not in
   order and are not connected.
9. Save as plain text (UTF-8, the Notepad default), keep the file name, and
   send the file back.

Example (not one of the 20 turns). The sheet would show:

```text
=== Txx ===
thank you mr. chairman so you left rates unchanged today do you expect another cut
```

You would type:

```text
=== Txx ===
thank you, mr. chairman. so, you left rates unchanged today. do you expect another cut?
```

How the marks become labels (the same mapping as the dataset): `,` `:` and
dashes → COMMA; `.` `!` `;` → PERIOD; `?` → QUESTION (wins if a word has two
marks); an ellipsis (`...`, `. . .` or `…`), quotes and brackets → no label.

## For the team: scoring

```bash
python -m fomcpunct.agreement score annotation/annotator_A.txt annotation/annotator_B.txt
```

The command first checks every sheet. If a word was changed, added or
dropped, it reports the first such problem in each turn and stops; fix the
sheet and run it again until it passes. Warnings (scoring still runs) flag a
turn with no punctuation at all (probably skipped), a line with only
punctuation, and an abbreviation such as `u.s.` followed by a capital letter
but no second period (the annotator may have meant a sentence end; ask). A
sheet that is not UTF-8 is reported, not scored.

The report `results/agreement/agreement.md` lists how many marks each source
used on these words, then compares transcript vs A, transcript vs B, A vs B,
and each saved model (DeBERTa-v3 seeds 0–2 and the CRF by default) vs the
transcript, A and B:

- macro-F1 over COMMA, PERIOD and QUESTION with a 95% bootstrap CI over
  turns, within-turn macro-F1 (last word of each turn left out), per-class F1
  and mark-detection F1;
- Cohen's kappa over all four labels with a CI;
- the paired difference "model vs transcript minus annotator vs transcript".

F1 and kappa are symmetric, so it does not matter which side of a pair is
called the reference.

## Sample

`python -m fomcpunct.agreement sample` (defaults) drew 10 chair and 10 reporter
turns of 40–200 words from the test split, each from a different press
conference (20 of the 22), with seed 0. The sheets were written before any
annotation. The command refuses to overwrite them unless `--force` is given;
do not regenerate them once they have been sent out. `sample.json` stores a
hash of each turn's words, so scoring stops if the data files change.

## Reading the results with care

- 20 turns is a small sample: the CIs are wide, and QUESTION rests on few
  transcript labels.
- The annotators are students, not professional transcribers, and may not be
  native speakers of English. Disagreement with the transcript therefore mixes
  reference noise, optional punctuation and annotator error. The check shows
  how much room the task leaves for different valid punctuation; it does not
  show which version is correct.
- The models were trained on transcripts punctuated in the same house style,
  so they are expected to agree with the transcript more than a person does.
  The A vs B pair is the human-to-human reference.
- The models never saw these turns' labels, but the test split as a whole was
  scored before this check was designed. The check changes no model and no
  test number.
