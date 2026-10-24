"""Error analysis of saved predictions (validation by default).

Usage:
    python -m fomcpunct.error_analysis --models heuristic hmm crf bilstm_crf_s0 \
        [--focus bilstm_crf_s0] [--reference crf] [--split validation] [--tag name]

Reads saved predictions only (it trains nothing) and writes
results/error_analysis/<split>[_<tag>].json (all numbers) and
results/error_analysis/<split>[_<tag>].md (tables and examples). Use validation while
models are still being chosen; analyse the test split only after the single
final test run.

Sections:
1. Where errors go: gold -> predicted counts for each model.
2. Turn length: macro-F1 by turn-length bucket (all words and within-turn),
   and prepared speech (the Chair's opening statement) against spontaneous
   answers and reporter questions.
3. Commas: by the next word (how often the transcript has a comma before
   common words, and each model's recall there) and by the transcript's
   mark (written comma, dash, colon) or a repeated word.
4. Questions: recall by how the question sentence starts.
5. Agreement: errors shared by all models; errors the focus model fixes or
   adds against a reference model.
6. Examples: a fixed-seed sample of the focus model's most common error types,
   shown as text with gold and predicted punctuation.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np

from . import LABELS, PUNCT_LABELS
from .evaluate import MIN_SLICE_SUPPORT, SEEDED, load_predictions
from .metrics import confusion, scores
from .run_baselines import load_split

RESULTS = Path("results")
LENGTH_BUCKETS = ((1, 10), (11, 30), (31, 100), (101, 300), (301, None))
MARK = {"O": "", "COMMA": ",", "PERIOD": ".", "QUESTION": "?"}
WH = {"what", "what's", "why", "how", "when", "where", "which", "who", "whom", "whose"}
AUX = {"do", "does", "did", "is", "are", "was", "were", "am", "can", "could", "will", "would", "should",
       "shall", "may", "might", "must", "have", "has", "had", "isn't", "aren't", "wasn't", "weren't",
       "don't", "doesn't", "didn't", "can't", "couldn't", "won't", "wouldn't", "shouldn't", "haven't",
       "hasn't", "hadn't"}
# Words a question often opens with before its real start ("and can you...", "so how...").
LEAD_IN = {"and", "so", "but", "well", "okay", "ok", "now", "also", "then", "yes", "yeah", "right", "just",
           "um", "uh", "oh"}
QUESTION_START_WINDOW = 6
FIXED_CLASSES = ("COMMA", "PERIOD")
N_NEXT_WORDS = 12
N_EXAMPLES = 4
CONTEXT = 8


def error_pairs(gold: list[list[str]], pred: list[list[str]], within_turn: bool) -> dict:
    cm = confusion([g[:-1] for g in gold] if within_turn else gold,
                   [p[:-1] for p in pred] if within_turn else pred)
    errors = int(cm.sum() - np.trace(cm))
    pairs = {f"{LABELS[g]}->{LABELS[p]}": int(cm[g, p]) for g in range(4) for p in range(4) if g != p}
    return {"words": int(cm.sum()), "errors": errors, "error_rate": errors / cm.sum(),
            "pairs": dict(sorted(pairs.items(), key=lambda kv: -kv[1]))}


def subset_scores(examples: list[dict], predictions: list[list[str]], idx: list[int]) -> dict:
    """Within-turn scores on a subset of turns, averaged over a FIXED class set.

    Macro-F1 here is over COMMA and PERIOD in every subset, so subsets are
    comparable (QUESTION is too rare in some subsets and absent from opening
    statements; its F1 is given separately when it has enough gold labels).
    A class with fewer than MIN_SLICE_SUPPORT gold labels makes the macro-F1 None.
    """
    gold = [examples[i]["labels"][:-1] for i in idx]
    pred = [predictions[i][:-1] for i in idx]
    s = scores(confusion(gold, pred))
    pc = s["per_class"]
    enough = all(pc[c]["support"] >= MIN_SLICE_SUPPORT for c in FIXED_CLASSES)
    return {"turns": len(idx), "words": sum(len(examples[i]["words"]) for i in idx),
            "macro_f1_comma_period": float(np.mean([pc[c]["f1"] for c in FIXED_CLASSES])) if enough else None,
            "question_f1": pc["QUESTION"]["f1"] if pc["QUESTION"]["support"] >= MIN_SLICE_SUPPORT else None,
            "per_class": {c: pc[c] for c in PUNCT_LABELS}}


def turn_length(examples: list[dict], predictions: list[list[str]]) -> list[dict]:
    rows = []
    for low, high in LENGTH_BUCKETS:
        idx = [i for i, ex in enumerate(examples) if len(ex["words"]) >= low and (high is None or len(ex["words"]) <= high)]
        rows.append({"bucket": f"{low}-{high}" if high else f"{low}+", **subset_scores(examples, predictions, idx)})
    return rows


def speech_kinds(examples: list[dict], predictions: list[list[str]]) -> list[dict]:
    """Prepared speech (the Chair's opening statement, turn 0) against spontaneous speech."""
    kinds = {"Chair opening statement (prepared, turn 0)": lambda ex: ex["role"] == "chair" and ex["id"].endswith("-t000"),
             "Chair answers": lambda ex: ex["role"] == "chair" and not ex["id"].endswith("-t000"),
             "Reporter questions": lambda ex: ex["role"] == "reporter"}
    return [{"kind": kind, **subset_scores(examples, predictions, [i for i, ex in enumerate(examples) if test(ex)])}
            for kind, test in kinds.items()]


def chair_kinds(examples: list[dict], predictions: list[list[str]]) -> list[dict]:
    """The Chair's own speech, one row per Chair (e.g. a Chair never seen in training)."""
    chairs = sorted({ex["chair"] for ex in examples if ex["role"] == "chair"})
    return [{"kind": f"Chair {name}", **subset_scores(
                examples, predictions, [i for i, ex in enumerate(examples) if ex["role"] == "chair" and ex["chair"] == name])}
            for name in chairs]


def commas_by_next_word(examples: list[dict], predictions: dict[str, list[list[str]]]) -> list[dict]:
    """For the words that most often follow a gold comma: comma rate in the gold data and each model's recall."""
    after_comma, seen = Counter(), Counter()
    for ex in examples:
        for i in range(len(ex["words"]) - 1):
            seen[ex["words"][i + 1]] += 1
            if ex["labels"][i] == "COMMA":
                after_comma[ex["words"][i + 1]] += 1
    rows = []
    for word, n in after_comma.most_common(N_NEXT_WORDS):
        row = {"next_word": word, "gold_commas": n, "occurrences": seen[word], "comma_rate": n / seen[word],
               "recall": {}, "predicted_commas": {}}
        for model, preds in predictions.items():
            hits = predicted = 0
            for ex, pred in zip(examples, preds):
                for i in range(len(ex["words"]) - 1):
                    if ex["words"][i + 1] != word:
                        continue
                    predicted += pred[i] == "COMMA"
                    hits += ex["labels"][i] == "COMMA" and pred[i] == "COMMA"
            row["recall"][model] = hits / n
            row["predicted_commas"][model] = predicted
        rows.append(row)
    return rows


def comma_kinds(examples: list[dict], predictions: dict[str, list[list[str]]]) -> list[dict]:
    """Comma recall by what the transcript wrote, and for repeated words ("the, the").

    The groups do not overlap: a comma before a repeated word counts only in the
    last group (almost all of those are written ","), so the mark groups compare
    commas in ordinary positions.
    """
    def repeat(ex, i):
        return i + 1 < len(ex["words"]) and ex["words"][i] == ex["words"][i + 1]

    kinds = {"written ','": lambda ex, i: "," in ex["marks"][i] and not repeat(ex, i),
             "dash": lambda ex, i: "—" in ex["marks"][i] and "," not in ex["marks"][i] and not repeat(ex, i),
             "colon": lambda ex, i: (":" in ex["marks"][i] and "," not in ex["marks"][i]
                                     and "—" not in ex["marks"][i] and not repeat(ex, i)),
             "repeated word (any mark)": repeat}
    rows = []
    for kind, test in kinds.items():
        positions = [(t, i) for t, ex in enumerate(examples) for i, label in enumerate(ex["labels"])
                     if label == "COMMA" and test(ex, i)]
        rows.append({"kind": kind, "gold_commas": len(positions), "recall": {
            m: (sum(p[t][i] == "COMMA" for t, i in positions) / len(positions)) if positions else None
            for m, p in predictions.items()}})
    return rows


def sentence_start(labels: list[str], end: int) -> int:
    """Index of the first word of the gold sentence that ends at `end`."""
    i = end - 1
    while i >= 0 and labels[i] not in ("PERIOD", "QUESTION"):
        i -= 1
    return i + 1


def question_kind(words: list[str]) -> str:
    """Group a question by its start, after skipping lead-in words such as "and", "so", "well"."""
    k = 0
    while k < len(words) - 1 and words[k] in LEAD_IN:
        k += 1
    if words[k] in WH:
        return "wh-word first"
    if words[k] in AUX:
        return "auxiliary verb first"
    if any(w in WH or w in AUX for w in words[k + 1:k + QUESTION_START_WINDOW]):
        return f"wh-word or auxiliary within the first {QUESTION_START_WINDOW} words"
    return f"no wh-word or auxiliary in the first {QUESTION_START_WINDOW} words"


def questions(examples: list[dict], predictions: dict[str, list[list[str]]], focus: str) -> dict:
    groups: dict[str, list[tuple[int, int]]] = {}
    first_words, missed_kinds = Counter(), Counter()
    for t, ex in enumerate(examples):
        for i, label in enumerate(ex["labels"]):
            if label != "QUESTION":
                continue
            start = sentence_start(ex["labels"], i)
            sentence = ex["words"][start:i + 1]
            kind = question_kind(sentence)
            groups.setdefault(kind, []).append((t, i))
            if predictions[focus][t][i] != "QUESTION":
                first_words[sentence[0]] += 1
                missed_kinds[kind] += 1
    out = {"by_start": {}, "turn_final_vs_within": {}, "focus_missed_first_words": first_words.most_common(10),
           "focus_missed_by_start": dict(missed_kinds.most_common())}
    for kind, positions in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        out["by_start"][kind] = {"gold": len(positions), "recall": {
            m: sum(p[t][i] == "QUESTION" for t, i in positions) / len(positions) for m, p in predictions.items()}}
    for where in ("turn-final", "within turn"):
        positions = [(t, i) for t, ex in enumerate(examples) for i, label in enumerate(ex["labels"])
                     if label == "QUESTION" and (i == len(ex["labels"]) - 1) == (where == "turn-final")]
        out["turn_final_vs_within"][where] = {"gold": len(positions), "recall": {
            m: (sum(p[t][i] == "QUESTION" for t, i in positions) / len(positions)) if positions else None
            for m, p in predictions.items()}}
    return out


def agreement(examples: list[dict], predictions: dict[str, list[list[str]]], focus: str, reference: str) -> dict:
    by_gold = {label: Counter() for label in LABELS}
    for t, ex in enumerate(examples):
        for i, gold in enumerate(ex["labels"]):
            wrong = [predictions[m][t][i] != gold for m in predictions]
            f, r = predictions[focus][t][i] != gold, predictions[reference][t][i] != gold
            c = by_gold[gold]
            c["words"] += 1
            c["all_models_wrong"] += all(wrong)
            c["no_model_wrong"] += not any(wrong)
            c["fixed_by_focus"] += r and not f
            c["fixed_by_focus_turn_final"] += r and not f and i == len(ex["labels"]) - 1
            c["added_by_focus"] += f and not r
            c["both_wrong"] += f and r
    return {"models": list(predictions), "focus": focus, "reference": reference,
            "by_gold_label": {label: dict(c) for label, c in by_gold.items()}}


def render(words: list[str], labels: list[str], lo: int, hi: int, target: int) -> str:
    parts = []
    for i in range(lo, hi):
        token = words[i] + MARK[labels[i]]
        parts.append(f"[{token}]" if i == target else token)
    return ("... " if lo > 0 else "") + " ".join(parts) + (" ..." if hi < len(words) else "")


def examples_of_errors(examples: list[dict], preds: list[list[str]], pairs: list[str], seed: int = 0) -> dict:
    found: dict[str, list[tuple[int, int]]] = {pair: [] for pair in pairs}
    for t, ex in enumerate(examples):
        for i, (gold, guess) in enumerate(zip(ex["labels"], preds[t])):
            key = f"{gold}->{guess}"
            if key in found:
                found[key].append((t, i))
    rng = random.Random(seed)
    out = {}
    for pair, positions in found.items():
        chosen = sorted(rng.sample(positions, min(N_EXAMPLES, len(positions))))
        out[pair] = []
        for t, i in chosen:
            ex = examples[t]
            lo, hi = max(0, i - CONTEXT), min(len(ex["words"]), i + CONTEXT + 1)
            out[pair].append({"id": ex["id"], "role": ex["role"], "position": i, "source_mark": ex["marks"][i],
                              "gold": render(ex["words"], ex["labels"], lo, hi, i),
                              "predicted": render(ex["words"], preds[t], lo, hi, i)})
    return out


def pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def f3(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def report_lines(r: dict) -> list[str]:
    models, focus, reference = r["models"], r["focus"], r["reference"]
    lines = [f"# Error analysis ({r['split']})", "",
             f"{r['documents']} press conferences, {r['turns']} turns, {r['words']:,} words. Generated by "
             "`python -m fomcpunct.error_analysis`; numbers in the JSON file next to this one.",
             f"Focus model: `{focus}`; reference model: `{reference}`. Labels mark the punctuation after a word.", ""]
    if r["split"] == "validation":
        lines += ["Validation is the model-selection set, so tuned models look slightly better here than they "
                  "will on test.", ""]
    if sum(bool(SEEDED.match(m)) for m in models) > 1:
        lines += ["Columns for different seeds of one model show how much a number moves with the training seed; "
                  "do not read differences of that size between models.", ""]

    lines += ["## 1. Where the errors go", "",
              "Share of each model's errors by gold -> predicted label (all words; within-turn in the JSON).", "",
              "| Gold -> predicted | " + " | ".join(models) + " |", "|---|" + "---|" * len(models)]
    top = list(r["errors"][focus]["all_words"]["pairs"])[:8]
    for pair in top:
        cells = []
        for m in models:
            e = r["errors"][m]["all_words"]
            cells.append(f"{e['pairs'][pair]:,} ({pct(e['pairs'][pair] / e['errors'] if e['errors'] else None)})")
        lines.append(f"| {pair} | " + " | ".join(cells) + " |")
    lines.append("| all errors (error rate) | " + " | ".join(
        f"{r['errors'][m]['all_words']['errors']:,} ({pct(r['errors'][m]['all_words']['error_rate'])})"
        for m in models) + " |")

    fixed = " and ".join(FIXED_CLASSES)
    note = (f"Within-turn words only. Cells: macro-F1 over {fixed} (the same classes in every row, so rows are "
            f"comparable) / QUESTION F1 (n/a below {MIN_SLICE_SUPPORT} gold questions). n/a in the first part: "
            f"{fixed} has fewer than {MIN_SLICE_SUPPORT} gold labels.")

    def cell(row: dict) -> str:
        return f"{f3(row['macro_f1_comma_period'])} / {f3(row['question_f1'])}"

    lines += ["", "## 2. Turn length and kind of speech", "", "By turn length in words. " + note, "",
              "| Turn length | Turns | Words | " + " | ".join(models) + " |", "|---|---|---|" + "---|" * len(models)]
    for k, row in enumerate(r["turn_length"][focus]):
        lines.append(f"| {row['bucket']} | {row['turns']} | {row['words']:,} | "
                     + " | ".join(cell(r["turn_length"][m][k]) for m in models) + " |")
    lines += ["", "Prepared against spontaneous speech (the opening statement is read from a script and has no "
              "questions). Same cells as above.", "", "| Speech | Turns | Words | " + " | ".join(models) + " |",
              "|---|---|---|" + "---|" * len(models)]
    for k, row in enumerate(r["speech_kinds"][focus]):
        lines.append(f"| {row['kind']} | {row['turns']} | {row['words']:,} | "
                     + " | ".join(cell(r["speech_kinds"][m][k]) for m in models) + " |")
    if len(r["chair_kinds"][focus]) > 1:
        lines += ["", "The Chair's own speech by Chair (all of that Chair's turns, opening statements included). "
                  "Same cells as above.", "", "| Speech | Turns | Words | " + " | ".join(models) + " |",
                  "|---|---|---|" + "---|" * len(models)]
        for k, row in enumerate(r["chair_kinds"][focus]):
            lines.append(f"| {row['kind']} | {row['turns']} | {row['words']:,} | "
                         + " | ".join(cell(r["chair_kinds"][m][k]) for m in models) + " |")
    lines += ["", "Comma recall / PERIOD recall within turns, same groups:", "",
              "| Speech | " + " | ".join(models) + " |", "|---|" + "---|" * len(models)]
    for k, row in enumerate(r["speech_kinds"][focus]):
        lines.append(f"| {row['kind']} | " + " | ".join(
            f"{f3(r['speech_kinds'][m][k]['per_class']['COMMA']['recall'])} / "
            f"{f3(r['speech_kinds'][m][k]['per_class']['PERIOD']['recall'])}" for m in models) + " |")

    lines += ["", "## 3. Commas by the next word", "",
              "For the words that most often follow a gold comma: how many gold commas come before them, how often "
              "the gold text has a comma before that word at all, and each model's comma recall there.", "",
              "| Next word | Gold commas | Comma rate before it | " + " | ".join(f"recall {m}" for m in models) + " |",
              "|---|---|---|" + "---|" * len(models)]
    for row in r["commas_by_next_word"]:
        lines.append(f"| {row['next_word']} | {row['gold_commas']} | {pct(row['comma_rate'])} | "
                     + " | ".join(pct(row["recall"][m]) for m in models) + " |")
    lines += ["", "Comma recall by what the transcript wrote. The last row is every gold comma followed by the "
              "same word (\"the, the\"; almost all written \",\"); the other rows leave those out.", "",
              "| Comma kind | Gold commas | " + " | ".join(models) + " |", "|---|---|" + "---|" * len(models)]
    for row in r["comma_kinds"]:
        lines.append(f"| {row['kind']} | {row['gold_commas']:,} | "
                     + " | ".join(pct(row["recall"][m]) for m in models) + " |")

    q = r["questions"]
    lines += ["", "## 4. Questions", "", "Question recall by how the gold question sentence starts, after skipping "
              "lead-in words (" + ", ".join(sorted(LEAD_IN)) + ").", "",
              "| Start of the question | Gold | " + " | ".join(models) + " |", "|---|---|" + "---|" * len(models)]
    for kind, row in q["by_start"].items():
        lines.append(f"| {kind} | {row['gold']} | " + " | ".join(pct(row["recall"][m]) for m in models) + " |")
    for where, row in q["turn_final_vs_within"].items():
        lines.append(f"| {where} | {row['gold']} | " + " | ".join(pct(row["recall"][m]) for m in models) + " |")
    lines += ["", f"Questions that `{focus}` misses: by start, "
              + ", ".join(f"{k} {n}" for k, n in q["focus_missed_by_start"].items()) + "; by first word, "
              + ", ".join(f"{w} ({n})" for w, n in q["focus_missed_first_words"]) + "."]

    a = r["agreement"]
    lines += ["", "## 5. Agreement between models", "",
              f"Per gold label: words every model in this report gets wrong, words no model gets wrong, and "
              f"`{focus}` against `{reference}` (\"fixed\": the reference is wrong and the focus right; in brackets, "
              "how many of those are the last word of a turn).", "",
              "| Gold label | Words | All models wrong | No model wrong | Fixed by focus | Added by focus | Both wrong |",
              "|---|---|---|---|---|---|---|"]
    for label, c in a["by_gold_label"].items():
        lines.append(f"| {label} | {c['words']:,} | {c['all_models_wrong']:,} | {c['no_model_wrong']:,} | "
                     f"{c['fixed_by_focus']:,} ({c['fixed_by_focus_turn_final']:,}) | {c['added_by_focus']:,} | "
                     f"{c['both_wrong']:,} |")

    lines += ["", f"## 6. Examples of `{focus}` errors", "",
              f"Up to {N_EXAMPLES} examples per error type, sampled with a fixed seed. The word in brackets carries "
              "the error; its label is the mark right after it. `source mark` is the punctuation in the "
              "transcript (a dash becomes COMMA).", ""]
    for pair, rows in r["examples"].items():
        lines += [f"### {pair}", ""]
        if not rows:
            lines += ["None.", ""]
            continue
        for row in rows:
            lines += [f"- `{row['id']}` ({row['role']}, source mark `{row['source_mark'] or 'none'}`)", "",
                      "  ```text", f"  gold: {row['gold']}", f"  pred: {row['predicted']}", "  ```", ""]
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--focus", help="model whose errors are shown as examples (default: last of --models)")
    parser.add_argument("--reference", help="model the focus is compared with (default: first of --models)")
    parser.add_argument("--tag", default="", help="suffix for the output files: <split>_<tag>.md/.json")
    args = parser.parse_args()
    focus, reference = args.focus or args.models[-1], args.reference or args.models[0]
    for name in (focus, reference):
        if name not in args.models:
            raise SystemExit(f"{name} must also be listed in --models")

    examples = load_split(args.split)
    predictions = {m: load_predictions(m, args.split, examples) for m in args.models}
    gold = [ex["labels"] for ex in examples]
    report = {
        "split": args.split, "documents": len({ex["doc"] for ex in examples}), "turns": len(examples),
        "words": sum(len(ex["words"]) for ex in examples), "models": args.models, "focus": focus,
        "reference": reference,
        "errors": {m: {"all_words": error_pairs(gold, p, False), "within_turn": error_pairs(gold, p, True)}
                   for m, p in predictions.items()},
        "turn_length": {m: turn_length(examples, p) for m, p in predictions.items()},
        "speech_kinds": {m: speech_kinds(examples, p) for m, p in predictions.items()},
        "chair_kinds": {m: chair_kinds(examples, p) for m, p in predictions.items()},
        "commas_by_next_word": commas_by_next_word(examples, predictions),
        "comma_kinds": comma_kinds(examples, predictions),
        "questions": questions(examples, predictions, focus),
        "agreement": agreement(examples, predictions, focus, reference),
    }
    top_pairs = list(report["errors"][focus]["all_words"]["pairs"])[:6]
    if "QUESTION->PERIOD" not in top_pairs:
        top_pairs.append("QUESTION->PERIOD")
    report["examples"] = examples_of_errors(examples, predictions[focus], top_pairs)

    out_dir = RESULTS / "error_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.split}_{args.tag}" if args.tag else args.split
    (out_dir / f"{stem}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out_dir / f"{stem}.md").write_text("\n".join(report_lines(report)) + "\n", encoding="utf-8")
    print(f"wrote {out_dir / f'{stem}.md'} and .json ({len(args.models)} models, focus {focus})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
