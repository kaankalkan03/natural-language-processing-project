"""Human agreement check: how noisy is the transcript punctuation we train and score on?

The labels come from one source: the punctuation in the published transcripts.
Nobody has a second, "correct" version. This module estimates how much that
single reference can be trusted. Two people re-punctuate the same sample of test
turns without seeing the original. We then compare them with each other, with
the transcript and with the frozen model predictions on the same words.

Usage:
    python -m fomcpunct.agreement sample     # pick turns, write blank sheets
    python -m fomcpunct.agreement score annotation/annotator_A.txt annotation/annotator_B.txt

`sample` writes annotation/sample.json (which turns; the key to the sheets) and
one blank sheet per annotator. A sheet shows each turn as lower-cased words
without punctuation, under a header line such as "=== T01 ===". The annotator
types punctuation between the words. `score` reads the sheets back, checks that
no word was changed, turns the typed marks into labels with the same mapping as
the dataset (tokens.mark_to_label) and writes results/agreement/agreement.{md,json}.

The sample and the scoring use only saved predictions; no model is retrained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import textwrap
from pathlib import Path

import numpy as np

from . import LABEL2ID, LABELS, PUNCT_LABELS
from .metrics import bootstrap_ci, class_f1, confusion, macro_f1, paired_bootstrap, scores
from .run_baselines import load_split
from .tokens import SENTENCE_OPENERS, mark_to_label

ANNOTATION = Path("annotation")
RESULTS = Path("results")
HEADER = re.compile(r"^[ \t]*===[ \t]*T(\d+)[ \t]*===[ \t]*$", re.MULTILINE)
# Characters an annotator may type between words. Only , . ? ! ; : and dashes
# change the label; quotes and brackets are allowed and ignored.
GAP_PUNCTUATION = set(",.?!;:-–—…\"'()[]")
# Curly quotes, non-breaking spaces, and the Turkish dotted capital I and dotless i
# (Shift+i on a Turkish keyboard types "İ", which would not match "in" or "it").
QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', " ": " ",
                        "İ": "I", "ı": "i"})
# Ellipses get no label, as in the dataset cleaning: "...", spaced ". . ." and "…".
# A fourth period ("....") is an ellipsis plus a sentence end.
ELLIPSIS = re.compile(r"\.(?:\s*\.){2}|…")
DEFAULT_MODELS = ("deberta_s0", "deberta_s1", "deberta_s2", "crf")
SHEET_WIDTH = 88

SHEET_PREAMBLE = """\
# Punctuation sheet: annotator {name}
# Read annotation/README.md first. In short:
# - Work alone. Do not look up the transcripts, do not use AI or grammar tools,
#   and do not compare with the other annotator until both sheets are done.
# - Only type punctuation between the words: , . ? (and, if you would use them,
#   ; : ! or a dash written as " -- "). Do not add, delete or change words.
# - Capital letters and line breaks do not matter.
# - To end a sentence after an abbreviation such as u.s. type a second period: u.s..
# - Do not edit the lines that start with "===" or "#", and do not add separator lines.
# Each block below is one speaker's turn in an FOMC press conference.
"""


class SheetError(ValueError):
    """An annotated sheet does not match the sampled words."""


# ---------------------------------------------------------------- sampling


def sample_turns(examples: list[dict], per_role: int = 10, min_words: int = 40,
                 max_words: int = 200, seed: int = 0) -> list[dict]:
    """Pick `per_role` chair and `per_role` reporter turns, each from a different conference.

    Conferences are shuffled once; chair turns take conferences from the front of
    the order and reporter turns from the back, so the two roles rarely share a
    conference. Moderator turns are mostly one-liners and are left out. Returns
    the chosen turns in a shuffled order.
    """
    rng = np.random.default_rng(seed)
    docs = sorted({ex["doc"] for ex in examples})
    order = [docs[i] for i in rng.permutation(len(docs))]
    candidates: dict[tuple[str, str], list[dict]] = {}
    for ex in examples:
        if ex["role"] in ("chair", "reporter") and min_words <= len(ex["words"]) <= max_words:
            candidates.setdefault((ex["role"], ex["doc"]), []).append(ex)
    chosen: list[dict] = []
    for role, doc_order in (("chair", order), ("reporter", order[::-1])):
        docs_with_turns = [doc for doc in doc_order if (role, doc) in candidates]
        if len(docs_with_turns) < per_role:
            raise ValueError(f"only {len(docs_with_turns)} conferences have {role} turns "
                             f"of {min_words}-{max_words} words; asked for {per_role}")
        for doc in docs_with_turns[:per_role]:
            pool = candidates[(role, doc)]
            chosen.append(pool[int(rng.integers(len(pool)))])
    return [chosen[i] for i in rng.permutation(len(chosen))]


def render_sheet(name: str, turns: list[dict]) -> str:
    blocks = [SHEET_PREAMBLE.format(name=name)]
    for i, ex in enumerate(turns, start=1):
        body = textwrap.fill(" ".join(ex["words"]), width=SHEET_WIDTH,
                             break_long_words=False, break_on_hyphens=False)
        blocks.append(f"=== T{i:02d} ===\n{body}\n")
    return "\n".join(blocks)


# ----------------------------------------------------------------- parsing


def read_sheet(text: str) -> dict[int, str]:
    """Split a sheet into {turn number: annotated text}; lines starting with # are dropped."""
    text = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    headers = list(HEADER.finditer(text))
    out: dict[int, str] = {}
    for h, nxt in zip(headers, headers[1:] + [None]):
        number = int(h.group(1))
        if number in out:
            raise SheetError(f"turn T{number:02d} appears twice")
        out[number] = text[h.end():nxt.start() if nxt else len(text)]
    return out


def gap_label(gap: str) -> str:
    """Label for the characters typed between two words (or after the last word)."""
    marks = ELLIPSIS.sub(" ", gap)
    marks = "".join("—" if c in "-–—" else c for c in marks if not c.isspace())
    return mark_to_label(marks)


def _context(text: str, at: int) -> str:
    return " ".join(text[max(0, at - 30):at + 30].split())


def parse_turn(annotated: str, words: list[str]) -> tuple[list[str], list[str]]:
    """Align the annotated text with the given words; return (labels, warnings).

    Each expected word must appear in order, spelled as given (case is ignored).
    Between two words only whitespace and punctuation may appear. An unspaced
    hyphen between two words would join them, so it is rejected. The function
    stops at the first problem in the turn.
    """
    original = annotated.translate(QUOTES)
    # Lower-case one character at a time so positions in `text` and `original` agree.
    text = "".join(c.lower() if len(c.lower()) == 1 else c for c in original)
    pos = 0
    gaps: list[str] = []
    starts: list[int] = []
    for k, word in enumerate(words):
        start = pos
        while True:
            end = pos + len(word)
            if text.startswith(word, pos) and (end == len(text) or not text[end].isalnum()):
                break
            if pos < len(text) and (text[pos].isspace() or text[pos] in GAP_PUNCTUATION):
                pos += 1
                continue
            where = f"word {k + 1} of {len(words)}, expected {word!r}"
            found = f"found {_context(text, pos)!r}" if pos < len(text) else "the turn ended"
            raise SheetError(f"{where}; {found}")
        gap = text[start:pos]
        starts.append(pos)
        if k > 0:
            # As in the dataset, an em dash is a pause even without spaces ("at—at"), but an
            # unspaced hyphen or en dash joins two words into one.
            joined = re.sub(r"-{2,}", "—", gap)  # "--" is a dash, as in clean.py
            if any(c in "-–" for c in joined) and not any(c.isspace() for c in gap):
                raise SheetError(f"words {k} and {k + 1} ({words[k - 1]!r}, {word!r}) are joined "
                                 f"by {gap!r}; write a pause dash with spaces: ' -- '")
            gaps.append(gap)
        pos = end
    rest = text[pos:]
    if any(not (c.isspace() or c in GAP_PUNCTUATION) for c in rest):
        raise SheetError(f"text after the last word {words[-1]!r}: {_context(rest, 0)!r}")
    gaps.append(rest)
    labels = [gap_label(g) for g in gaps]
    warnings = []
    for k in range(len(words) - 1):
        # The dataset marks "U.S. We" as a sentence end (a capitalized common sentence
        # opener after an abbreviation, tokens.split_words); lower-casing hides that here.
        if (words[k].endswith(".") and labels[k] == "O" and words[k + 1] in SENTENCE_OPENERS
                and original[starts[k + 1]].isupper()):
            warnings.append(f"word {k + 1} {words[k]!r} is followed by a capital letter; if the "
                            f"sentence ends there, type a second period ({words[k]}.)")
    return labels, warnings


def parse_sheet(text: str, turns: list[dict]) -> tuple[list[list[str]], list[str]]:
    """Labels for every sampled turn, plus warnings. Raises SheetError listing all problems."""
    blocks = read_sheet(text)
    errors, warnings, labels = [], [], []
    expected = set(range(1, len(turns) + 1))
    if set(blocks) != expected:
        missing, extra = sorted(expected - set(blocks)), sorted(set(blocks) - expected)
        errors.append(f"turn headers do not match the sample (missing {missing}, unexpected {extra})")
    for i, ex in enumerate(turns, start=1):
        if i not in blocks:
            continue
        try:
            turn_labels, turn_warnings = parse_turn(blocks[i], ex["words"])
        except SheetError as err:
            errors.append(f"T{i:02d}: {err}")
            continue
        warnings += [f"T{i:02d}: {w}" for w in turn_warnings]
        for line in blocks[i].splitlines():
            if line.strip() and not any(c.isalnum() for c in line):
                warnings.append(f"T{i:02d}: a line holds only {line.strip()!r}; it counts as the "
                                "mark after the word before it")
        if all(label == "O" for label in turn_labels):
            warnings.append(f"T{i:02d} has no punctuation at all; was it skipped?")
        labels.append(turn_labels)
    if errors:
        raise SheetError("\n".join(errors))
    return labels, warnings


# ----------------------------------------------------------------- scoring


def cohen_kappa(cm: np.ndarray) -> np.ndarray:
    """Cohen's kappa over the four labels; works on one 4x4 matrix or a stack (Bx4x4)."""
    cm = np.asarray(cm, dtype=float)
    n = cm.sum(axis=(-2, -1))
    observed = np.trace(cm, axis1=-2, axis2=-1) / n
    chance = (cm.sum(axis=-1) * cm.sum(axis=-2)).sum(axis=-1) / n ** 2
    return (observed - chance) / (1 - chance)


def turn_confusions(reference: list[list[str]], other: list[list[str]],
                    within_turn: bool = False) -> np.ndarray:
    """One 4x4 matrix per turn (the bootstrap resamples turns); within_turn drops the last word."""
    cut = slice(None, -1) if within_turn else slice(None)
    return np.stack([confusion([r[cut]], [o[cut]]) for r, o in zip(reference, other, strict=True)])


def compare(reference: list[list[str]], other: list[list[str]], resamples: int, seed: int) -> dict:
    cms = turn_confusions(reference, other)
    s = scores(cms.sum(axis=0))
    return {
        "words": int(cms.sum()),
        "macro_f1": s["macro_f1"],
        "macro_f1_ci": bootstrap_ci(cms, n_resamples=resamples, seed=seed),
        "within_turn_macro_f1": float(macro_f1(turn_confusions(reference, other, True).sum(axis=0))),
        "f1": {label: s["per_class"][label]["f1"] for label in PUNCT_LABELS},
        "mark_detection_f1": s["mark_detection_f1"],
        "kappa": float(cohen_kappa(cms.sum(axis=0))),
        "kappa_ci": bootstrap_ci(cms, metric=cohen_kappa, n_resamples=resamples, seed=seed),
        "confusion": s["confusion"],
    }


def words_hash(words: list[str]) -> str:
    return hashlib.sha256("\n".join(words).encode("utf-8")).hexdigest()[:16]


def load_model_labels(model: str, split: str, turns: list[dict]) -> list[list[str]]:
    path = RESULTS / "predictions" / f"{model}.{split}.jsonl"
    with path.open(encoding="utf-8") as handle:
        by_id = {row["id"]: row["labels"] for row in map(json.loads, handle)}
    missing = [ex["id"] for ex in turns if ex["id"] not in by_id]
    if missing:
        raise ValueError(f"{path} has no prediction for {missing[:3]}")
    return [by_id[ex["id"]] for ex in turns]


def score(sheets: dict[str, list[list[str]]], turns: list[dict], models: dict[str, list[list[str]]],
          resamples: int = 1000, seed: int = 0) -> dict:
    """All pairwise comparisons. Each pair is (reference, other); F1 and kappa are symmetric."""
    transcript = [ex["labels"] for ex in turns]
    pairs: list[tuple[str, str, list, list]] = []
    names = list(sheets)
    for name in names:
        pairs.append(("transcript", name, transcript, sheets[name]))
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pairs.append((a, b, sheets[a], sheets[b]))
    for model, labels in models.items():
        pairs.append(("transcript", model, transcript, labels))
        for name in names:
            pairs.append((name, model, sheets[name], labels))
    marks_used = {name: {label: sum(seq.count(label) for seq in labels) for label in PUNCT_LABELS}
                  for name, labels in [("transcript", transcript), *sheets.items(), *models.items()]}
    rows =[{"reference": r, "other": o, **compare(rl, ol, resamples, seed)} for r, o, rl, ol in pairs]
    # Does a model agree with the transcript more or less than a person does?
    differences = []
    for model, labels in models.items():
        model_cms = turn_confusions(transcript, labels)
        for name in names:
            diff = paired_bootstrap(model_cms, turn_confusions(transcript, sheets[name]),
                                    n_resamples=resamples, seed=seed)
            differences.append({"model": model, "annotator": name, **diff})
    return {"turns": len(turns), "words": sum(len(ex["words"]) for ex in turns),
            "resamples": resamples, "marks_used": marks_used, "rows": rows,
            "model_minus_annotator_vs_transcript": differences}


def ci_text(ci: dict) -> str:
    return f"{ci['point']:.3f} [{ci['low']:.3f}, {ci['high']:.3f}]"


def write_report(report: dict, sample: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "agreement.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Human agreement check",
        "",
        f"{report['turns']} test turns, {report['words']:,} words "
        f"({sample['criteria']['per_role']} chair and {sample['criteria']['per_role']} reporter turns "
        f"of {sample['criteria']['min_words']}-{sample['criteria']['max_words']} words, one per "
        f"conference and role, seed {sample['criteria']['seed']}). Annotators: "
        f"{', '.join(report['annotators'])}. Model labels are the saved test predictions; no model "
        f"was retrained. 95% CIs: percentile bootstrap over turns ({report['resamples']:,} resamples).",
        "",
        "F1 and kappa are symmetric, so the order of a pair does not matter. Within-turn "
        "macro-F1 leaves out the last word of each turn (its sentence end is given by the turn "
        "boundary). The models were trained on transcripts punctuated in the same house style, so "
        "they are expected to agree with the transcript more than a person does; the annotator "
        "pair is the human-to-human reference.",
        "",
        "Marks used on these words:",
        "",
        "| Source | COMMA | PERIOD | QUESTION |",
        "|---|---|---|---|",
        *[f"| {name} | {c['COMMA']} | {c['PERIOD']} | {c['QUESTION']} |"
          for name, c in report["marks_used"].items()],
        "",
        "Agreement:",
        "",
        "| Pair | Macro-F1 [95% CI] | Within-turn | COMMA | PERIOD | QUESTION | Mark detection | Kappa [95% CI] |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in report["rows"]:
        f1 = row["f1"]
        lines.append(
            f"| {row['reference']} vs {row['other']} | {ci_text(row['macro_f1_ci'])} | "
            f"{row['within_turn_macro_f1']:.3f} | {f1['COMMA']:.3f} | {f1['PERIOD']:.3f} | "
            f"{f1['QUESTION']:.3f} | {row['mark_detection_f1']:.3f} | {ci_text(row['kappa_ci'])} |")
    lines += [
        "",
        "## Model minus annotator, both scored against the transcript",
        "",
        "Positive: the model's punctuation is closer to the transcript than the person's. Paired "
        "bootstrap over the same turns.",
        "",
        "| Model | Annotator | Difference in macro-F1 [95% CI] |",
        "|---|---|---|",
    ]
    for d in report["model_minus_annotator_vs_transcript"]:
        lines.append(f"| {d['model']} | {d['annotator']} | "
                     f"{d['difference']:+.3f} [{d['low']:+.3f}, {d['high']:+.3f}] |")
    lines.append("")
    (out_dir / "agreement.md").write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------- CLI


def cmd_sample(args) -> int:
    examples = load_split(args.split)
    turns = sample_turns(examples, args.per_role, args.min_words, args.max_words, args.seed)
    out = Path(args.out)
    sheets = [out / f"annotator_{name}.txt" for name in args.annotators]
    existing = [p for p in [out / "sample.json", *sheets] if p.exists()]
    if existing and not args.force:
        print(f"refusing to overwrite {', '.join(map(str, existing))} (use --force)", file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)
    sample = {
        "split": args.split,
        "criteria": {"per_role": args.per_role, "min_words": args.min_words,
                     "max_words": args.max_words, "seed": args.seed},
        "turns": [{"key": f"T{i:02d}", "id": ex["id"], "doc": ex["doc"], "role": ex["role"],
                   "words": len(ex["words"]), "words_sha256": words_hash(ex["words"])}
                  for i, ex in enumerate(turns, start=1)],
    }
    (out / "sample.json").write_text(json.dumps(sample, indent=2) + "\n", encoding="utf-8")
    for name, path in zip(args.annotators, sheets):
        path.write_text(render_sheet(name, turns), encoding="utf-8", newline="\n")
    words = sum(t["words"] for t in sample["turns"])
    print(f"{len(turns)} turns, {words} words from {len({t['doc'] for t in turns})} conferences "
          f"-> {out / 'sample.json'} and {len(sheets)} sheets")
    return 0


def load_sample(path: Path) -> tuple[dict, list[dict]]:
    sample = json.loads(path.read_text(encoding="utf-8"))
    by_id = {ex["id"]: ex for ex in load_split(sample["split"])}
    turns = [by_id[t["id"]] for t in sample["turns"]]
    if any(words_hash(ex["words"]) != t["words_sha256"] for ex, t in zip(turns, sample["turns"])):
        raise ValueError("the data files changed since the sample was drawn")
    return sample, turns


def cmd_score(args) -> int:
    sample, turns = load_sample(Path(args.sample))
    sheets, failed = {}, False
    for path in map(Path, args.sheets):
        name = path.stem.removeprefix("annotator_")
        if name in sheets:
            print(f"two sheets are named {name!r}", file=sys.stderr)
            return 1
        try:
            labels, warnings = parse_sheet(path.read_text(encoding="utf-8-sig"), turns)
        except UnicodeDecodeError:
            print(f"{path}: not UTF-8 text; save it again as UTF-8 (Notepad: Save as, "
                  "Encoding: UTF-8)", file=sys.stderr)
            failed = True
            continue
        except SheetError as err:
            print(f"{path}:\n{textwrap.indent(str(err), '  ')}", file=sys.stderr)
            failed = True
            continue
        for w in warnings:
            print(f"{path}: warning: {w}", file=sys.stderr)
        sheets[name] = labels
    if failed:
        return 1
    if len(sheets) < 2:
        print("need at least two annotated sheets", file=sys.stderr)
        return 1
    models = {m: load_model_labels(m, sample["split"], turns) for m in args.models}
    report = score(sheets, turns, models, args.resamples, args.seed)
    report["annotators"] = list(sheets)
    report["models"] = list(models)
    write_report(report, sample, Path(args.out))
    print(f"wrote {Path(args.out) / 'agreement.md'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("sample", help="pick turns and write blank sheets")
    p.add_argument("--split", default="test")
    p.add_argument("--per-role", type=int, default=10)
    p.add_argument("--min-words", type=int, default=40)
    p.add_argument("--max-words", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--annotators", nargs="+", default=["A", "B"])
    p.add_argument("--out", default=str(ANNOTATION))
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_sample)
    s = sub.add_parser("score", help="score annotated sheets")
    s.add_argument("sheets", nargs="+")
    s.add_argument("--sample", default=str(ANNOTATION / "sample.json"))
    s.add_argument("--models", nargs="*", default=list(DEFAULT_MODELS))
    s.add_argument("--resamples", type=int, default=1000)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", default=str(RESULTS / "agreement"))
    s.set_defaults(func=cmd_score)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
