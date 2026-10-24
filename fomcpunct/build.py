"""Build the cleaned and model-ready datasets from the raw PDFs.

Usage:
    python -m fomcpunct.build [--raw data/raw] [--clean data/clean] [--out data/processed]

Outputs:
    data/clean/turns.jsonl         cleaned speaker turns with punctuation
    data/clean/cleaning_log.csv    what was removed or changed, per transcript
    data/processed/{train,validation,test}.jsonl
                                   one speaker turn per line: lower-cased words,
                                   labels and the raw marks behind the labels
    data/processed/stats.json      split sizes and label counts

Split by press conference date, so no transcript is in two splits:
train <= 2022, validation = 2023, test >= 2024.
Models that need short inputs (transformers) cut turns into windows themselves;
every model is scored on the same words.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from . import LABELS
from .clean import clean_pdf
from .tokens import to_example

SPLITS = ("train", "validation", "test")


def split_of(date: str) -> str:
    year = int(date[:4])
    if year <= 2022:
        return "train"
    if year == 2023:
        return "validation"
    return "test"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--clean", type=Path, default=Path("data/clean"))
    parser.add_argument("--out", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    args.clean.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)

    with (args.raw / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))

    turn_rows, log_rows = [], []
    examples: dict[str, list[dict]] = {name: [] for name in SPLITS}
    token_log: Counter = Counter()
    for entry in manifest:
        date = entry["date"]
        doc = clean_pdf(args.raw / entry["file"])
        split = split_of(date)
        doc_tokens: Counter = Counter()
        for t, turn in enumerate(doc["turns"]):
            turn_rows.append({"doc": date, "chair": doc["chair"], "turn": t, **turn})
            words, labels, marks = to_example(turn["text"], doc_tokens)
            if not words:
                doc_tokens["turns_without_words"] += 1
                continue
            examples[split].append({
                "id": f"{date}-t{t:03d}",
                "doc": date,
                "chair": doc["chair"],
                "role": turn["role"],
                "words": words,
                "labels": labels,
                "marks": marks,
            })
            doc_tokens["words"] += len(words)
        token_log.update(doc_tokens)
        log_rows.append({"doc": date, "split": split, "chair": doc["chair"], **doc["log"], **doc_tokens})
        print(f"{date} {split:<10} {doc['chair']:<9} turns={len(doc['turns']):>3} words={doc_tokens['words']:>6}")

    write_jsonl(args.clean / "turns.jsonl", turn_rows)
    fields = sorted({key for row in log_rows for key in row} - {"doc", "split", "chair"})
    with (args.clean / "cleaning_log.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["doc", "split", "chair", *fields],
                                restval=0, lineterminator="\n")
        writer.writeheader()
        writer.writerows(log_rows)

    train_vocab = {w for row in examples["train"] for w in row["words"]}
    stats = {"labels": list(LABELS), "splits": {}}
    for name in SPLITS:
        rows = examples[name]
        write_jsonl(args.out / f"{name}.jsonl", rows)
        label_counts = Counter(label for row in rows for label in row["labels"])
        n_words = sum(label_counts.values())
        words = [w for row in rows for w in row["words"]]
        stats["splits"][name] = {
            "documents": len({row["doc"] for row in rows}),
            "turns": len(rows),
            "words": n_words,
            "label_counts": {label: label_counts[label] for label in LABELS},
            "label_share": {label: round(label_counts[label] / n_words, 4) for label in LABELS},
            "oov_rate_vs_train": round(sum(w not in train_vocab for w in words) / len(words), 4),
            "turns_by_role": dict(Counter(row["role"] for row in rows)),
            "chair_turns": dict(Counter(row["chair"] for row in rows if row["role"] == "chair")),
        }
    stats["tokenization"] = {k: v for k, v in sorted(token_log.items()) if k != "words"}
    (args.out / "stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats["splits"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
