"""Is the benchmark saturated? Accuracy, macro-F1 and turn exact match side by side.

The course's Project 2 guideline calls a task trivial if even the baseline
scores 90% accuracy or about 85% exact match (EM). Here about 84% of words carry
no mark (O), so a system that never predicts punctuation already reaches 84%
accuracy: accuracy mostly measures the share of O. This report gives, for every
saved model and for that all-O system:

- accuracy over all words;
- macro-F1 over COMMA, PERIOD and QUESTION (the main metric);
- turn exact match: the share of speaker turns in which every word has the
  right label.

Each comes with a 95% cluster-bootstrap CI over press conferences.

Usage:
    python -m fomcpunct.saturation --split test

Writes results/saturation_<split>.{md,json}. It reads saved predictions only;
no model is retrained.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from .evaluate import FEW_CLUSTERS, RESULTS, doc_confusions, load_predictions
from .metrics import bootstrap_ci, macro_f1
from .run_baselines import load_split

ALL_O = "all O (no model)"
ORDER = ("heuristic", "hmm", "crf", "bilstm_softmax", "bilstm_crf", "bert", "deberta")
SEEDED = re.compile(r"^(.*)_s(\d+)$")


def accuracy(cm: np.ndarray) -> np.ndarray:
    """Share of words with the right label; one 4x4 matrix or a stack (Bx4x4)."""
    return np.trace(cm, axis1=-2, axis2=-1) / cm.sum(axis=(-2, -1))


def exact_share(counts: np.ndarray) -> np.ndarray:
    """counts[..., 0] = turns with every label right, counts[..., 1] = all turns."""
    return counts[..., 0] / counts[..., 1]


def doc_exact_counts(examples: list[dict], predictions: list[list[str]], docs: list[str],
                     min_words: int = 1) -> np.ndarray:
    """Per conference: [turns with every label right, turns], over turns of at least min_words."""
    index = {doc: i for i, doc in enumerate(docs)}
    counts = np.zeros((len(docs), 2), dtype=np.int64)
    for ex, pred in zip(examples, predictions, strict=True):
        if len(ex["labels"]) >= min_words:
            counts[index[ex["doc"]]] += (ex["labels"] == pred, 1)
    return counts


def saved_models(split: str) -> list[str]:
    names = [p.name[:-len(f".{split}.jsonl")] for p in (RESULTS / "predictions").glob(f"*.{split}.jsonl")]

    def key(name: str):
        match = SEEDED.match(name)
        base, seed = (match.group(1), int(match.group(2))) if match else (name, -1)
        return (ORDER.index(base) if base in ORDER else len(ORDER), base, seed)

    return sorted(names, key=key)


def report(split: str, models: list[str], resamples: int = 1000, seed: int = 0) -> dict:
    examples = load_split(split)
    docs = sorted({ex["doc"] for ex in examples})
    systems = {ALL_O: [["O"] * len(ex["words"]) for ex in examples]}
    systems.update({m: load_predictions(m, split, examples) for m in models})
    rows = {}
    for name, predictions in systems.items():
        cms = doc_confusions(examples, predictions, docs)
        counts = doc_exact_counts(examples, predictions, docs)
        longer = doc_exact_counts(examples, predictions, docs, min_words=2)
        rows[name] = {
            "accuracy": bootstrap_ci(cms, metric=accuracy, n_resamples=resamples, seed=seed),
            "macro_f1": bootstrap_ci(cms, metric=macro_f1, n_resamples=resamples, seed=seed),
            "turn_exact_match": bootstrap_ci(counts, metric=exact_share, n_resamples=resamples, seed=seed),
            "turn_exact_match_2plus_words": bootstrap_ci(longer, metric=exact_share,
                                                         n_resamples=resamples, seed=seed),
            "exact_turns": int(counts[:, 0].sum()),
        }
    o_share = sum(label == "O" for ex in examples for label in ex["labels"]) / sum(
        len(ex["labels"]) for ex in examples)
    one_word = [ex for ex in examples if len(ex["words"]) == 1]
    return {"split": split, "documents": len(docs), "turns": len(examples),
            "words": sum(len(ex["words"]) for ex in examples), "o_share": o_share,
            "one_word_turns": len(one_word),
            "one_word_turns_moderator": sum(ex["role"] == "moderator" for ex in one_word),
            "models": rows}


def ci_text(ci: dict) -> str:
    return f"{ci['point']:.3f} [{ci['low']:.3f}, {ci['high']:.3f}]"


def write(result: dict, out_dir: Path = RESULTS) -> Path:
    split = result["split"]
    (out_dir / f"saturation_{split}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [
        f"# Saturation check: {split}",
        "",
        f"{result['words']:,} words, {result['turns']:,} speaker turns, {result['documents']} press "
        f"conferences. {result['o_share']:.1%} of words are O, so predicting no punctuation at all "
        "already gives that accuracy. Macro-F1 is over COMMA, PERIOD and QUESTION. Turn exact match "
        "is the share of turns with every label right. "
        f"{result['one_word_turns']:,} turns are a single word "
        f"({result['one_word_turns_moderator']:,} of them from the moderator, who mostly names the "
        "next reporter); they are easy to get right and raise exact match, so the last column "
        "leaves them out. 95% CIs: cluster bootstrap over press conferences. Computed from the "
        "saved predictions; no model was retrained.",
    ]
    if result["documents"] < FEW_CLUSTERS:
        lines += ["", f"With only {result['documents']} conferences the intervals are too narrow "
                      "(see the README); read them as indicative."]
    lines += [
        "",
        "| System | Accuracy [95% CI] | Macro-F1 [95% CI] | Turn exact match [95% CI] | "
        "Exact match, turns of 2+ words |",
        "|---|---|---|---|---|",
    ]
    for name, row in result["models"].items():
        lines.append(f"| {name} | {ci_text(row['accuracy'])} | {ci_text(row['macro_f1'])} | "
                     f"{ci_text(row['turn_exact_match'])} | "
                     f"{ci_text(row['turn_exact_match_2plus_words'])} |")
    lines.append("")
    path = out_dir / f"saturation_{split}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--split", default="test")
    parser.add_argument("--models", nargs="*", help="default: every model with saved predictions")
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    models = args.models if args.models else saved_models(args.split)
    print(f"wrote {write(report(args.split, models, args.resamples, args.seed))}")


if __name__ == "__main__":
    main()
