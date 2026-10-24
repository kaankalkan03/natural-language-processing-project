"""Score saved predictions with bootstrap confidence intervals.

Usage:
    python -m fomcpunct.evaluate --split validation --models heuristic hmm crf

Writes results/<split>_metrics.json (all numbers) and results/<split>_summary.md
(tables). For every model:

- macro-F1 over COMMA/PERIOD/QUESTION with a 95% cluster-bootstrap CI over
  press conferences, and per-class F1 with CIs;
- the same scores on within-turn words only. The input is one speaker turn,
  so the last word of a turn almost always ends a sentence; scoring without
  those words shows how well a model finds sentence ends it is not given;
- slices (speaker role, commas written as "," versus dashes);
- paired-bootstrap differences against every earlier model in --models.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

import numpy as np

from . import LABEL2ID, PUNCT_LABELS
from .metrics import bootstrap_ci, class_f1, confusion, paired_bootstrap, scores
from .run_baselines import load_split

RESULTS = Path("results")
FEW_CLUSTERS = 15
MIN_SLICE_SUPPORT = 30
SEEDED = re.compile(r"^(.*)_s(\d+)$")


def load_predictions(model: str, split: str, examples: list[dict]) -> list[list[str]]:
    path = RESULTS / "predictions" / f"{model}.{split}.jsonl"
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    by_id = {row["id"]: row["labels"] for row in rows}
    if len(rows) != len(by_id):
        raise ValueError(f"{path} repeats some ids")
    if len(rows) != len(examples) or any(ex["id"] not in by_id for ex in examples):
        raise ValueError(f"{path} does not cover exactly the {split} turns")
    return [by_id[ex["id"]] for ex in examples]


def doc_confusions(examples: list[dict], predictions: list[list[str]], docs: list[str],
                   within_turn: bool = False) -> np.ndarray:
    """One 4x4 confusion matrix per press conference; within_turn drops each turn's last word."""
    index = {doc: i for i, doc in enumerate(docs)}
    cut = slice(None, -1) if within_turn else slice(None)
    cms = np.zeros((len(docs), 4, 4), dtype=np.int64)
    for ex, pred in zip(examples, predictions, strict=True):
        cms[index[ex["doc"]]] += confusion([ex["labels"][cut]], [pred[cut]])
    return cms


def with_cis(cms: np.ndarray, resamples: int, seed: int) -> dict:
    result = scores(cms.sum(axis=0))
    result["macro_f1_ci"] = bootstrap_ci(cms, n_resamples=resamples, seed=seed)
    result["class_f1_ci"] = {
        label: bootstrap_ci(cms, metric=lambda cm, i=LABEL2ID[label]: class_f1(cm)[..., i],
                            n_resamples=resamples, seed=seed)
        for label in PUNCT_LABELS
    }
    return result


def slice_scores(cm: np.ndarray) -> dict:
    """Macro-F1 over classes with at least MIN_SLICE_SUPPORT gold labels in the slice.

    Classes with a handful of gold labels would make the slice score swing on a
    few words (e.g. 5 moderator commas), so they are left out of the average but
    kept in the per-class numbers, with gold and predicted counts.
    """
    s = scores(cm)
    used = [c for c in PUNCT_LABELS if s["per_class"][c]["support"] >= MIN_SLICE_SUPPORT]
    return {
        "macro_f1": float(np.mean([s["per_class"][c]["f1"] for c in used])) if used else None,
        "classes_used": used,
        "f1": {c: s["per_class"][c]["f1"] for c in PUNCT_LABELS},
        "support": {c: s["per_class"][c]["support"] for c in PUNCT_LABELS},
        "predicted": {c: int(cm[:, LABEL2ID[c]].sum()) for c in PUNCT_LABELS},
        "words": int(cm.sum()),
    }


def slices(examples: list[dict], predictions: list[list[str]]) -> dict:
    out = {}
    for role in ("chair", "reporter", "moderator"):
        pairs = [(ex["labels"], p) for ex, p in zip(examples, predictions) if ex["role"] == role]
        if pairs:
            out[f"role={role}"] = slice_scores(confusion(*zip(*pairs)))
    out["turn_final_words"] = slice_scores(
        confusion([ex["labels"][-1:] for ex in examples], [p[-1:] for p in predictions]))
    for source, name in ((",", "comma"), ("—", "dash")):
        hits = total = 0
        for ex, pred in zip(examples, predictions):
            for mark, label, guess in zip(ex["marks"], ex["labels"], pred):
                if label == "COMMA" and source in mark:
                    total += 1
                    hits += guess == "COMMA"
        out[f"comma_recall_from_{name}"] = {"recall": hits / total if total else None, "gold": total}
    return out


def seed_groups(report: dict) -> dict:
    """Mean, SD and range over models trained with different seeds (names ending in _s<seed>)."""
    groups: dict[str, list[str]] = {}
    for model in report["models"]:
        match = SEEDED.match(model)
        if match:
            groups.setdefault(match.group(1), []).append(model)
    out = {}
    for base, models in groups.items():
        if len(models) < 2:
            continue
        out[base] = {"models": models}
        for view in ("all_words", "within_turn"):
            values = [report["models"][m][view]["macro_f1"] for m in models]
            out[base][view] = {"mean": statistics.mean(values), "sd": statistics.stdev(values),
                               "min": min(values), "max": max(values)}
    return out


def fmt(ci: dict) -> str:
    return f"{ci['point']:.3f} [{ci['low']:.3f}, {ci['high']:.3f}]"


def summary_lines(report: dict) -> list[str]:
    split, n_docs = report["split"], report["documents"]
    lines = [f"# {split.capitalize()} results", "",
             f"{n_docs} press conferences, {report['words']:,} words. 95% CIs: bootstrap over press "
             f"conferences ({report['bootstrap']['resamples']} resamples, seed {report['bootstrap']['seed']}).",
             "Within-turn scores leave out the last word of every turn (the turn boundary is given "
             "input, and that word almost always ends a sentence)."]
    if n_docs < FEW_CLUSTERS:
        lines.append(f"With only {n_docs} conferences, percentile intervals are too narrow "
                     "(the code audit estimated about 85% real coverage); read them as indicative.")
    if split == "validation":
        lines.append("Validation is the model-selection set: scores of tuned models are optimistic.")
    lines += ["", "| Model | Macro-F1 [95% CI] | Within-turn macro-F1 [95% CI] | COMMA F1 | PERIOD F1 | "
              "QUESTION F1 | ms / 1,000 words |", "|---|---|---|---|---|---|---|"]
    for model, r in report["models"].items():
        latency = r.get("latency", {})
        ms = latency.get("ms_per_1000_words", "")
        if ms != "" and not str(latency.get("device", "cpu")).startswith("cpu"):
            ms = f"{ms} ({latency['device']}, batched)"  # GPU: not comparable with the CPU rows
        lines.append(f"| {model} | {fmt(r['all_words']['macro_f1_ci'])} | {fmt(r['within_turn']['macro_f1_ci'])} | "
                     + " | ".join(fmt(r["all_words"]["class_f1_ci"][c]) for c in PUNCT_LABELS) + f" | {ms} |")
    if report.get("seed_groups"):
        lines += ["", "Seed variation (each seed is also a row above, with its own CI):", "",
                  "| Model | Seeds | Macro-F1 mean ± SD [min, max] | Within-turn mean ± SD [min, max] |",
                  "|---|---|---|---|"]
        for base, g in report["seed_groups"].items():
            cells = [f"{g[v]['mean']:.3f} ± {g[v]['sd']:.3f} [{g[v]['min']:.3f}, {g[v]['max']:.3f}]"
                     for v in ("all_words", "within_turn")]
            lines.append(f"| {base} | {len(g['models'])} | " + " | ".join(cells) + " |")
    lines += ["", "Within-turn per-class F1:", "", "| Model | COMMA | PERIOD | QUESTION |", "|---|---|---|---|"]
    for model, r in report["models"].items():
        lines.append(f"| {model} | " + " | ".join(fmt(r["within_turn"]["class_f1_ci"][c]) for c in PUNCT_LABELS) + " |")
    if report["comparisons"]:
        lines += ["", "| Comparison | Macro-F1 difference [95% CI] | Share of resamples not better | "
                  "Within-turn difference [95% CI] |", "|---|---|---|---|"]
        for c in report["comparisons"]:
            a, w = c["all_words"], c["within_turn"]
            lines.append(f"| {c['model']} vs {c['versus']} | {a['difference']:+.3f} [{a['low']:+.3f}, {a['high']:+.3f}] "
                         f"| {a['share_not_better']:.3f} | {w['difference']:+.3f} [{w['low']:+.3f}, {w['high']:+.3f}] |")
    lines += ["", f"Slices (macro-F1 over classes with at least {MIN_SLICE_SUPPORT} gold labels in the slice; "
              "per-class F1, gold and predicted counts in the JSON):", "",
              "| Model | Chair | Reporter | Moderator | Turn-final words | Comma recall (,) | Comma recall (dash) |",
              "|---|---|---|---|---|---|---|"]
    show = lambda v: "n/a" if v is None else f"{v:.3f}"  # noqa: E731
    for model, r in report["models"].items():
        s = r["slices"]
        cells = [show(s[k]["macro_f1"]) if k in s else "n/a"
                 for k in ("role=chair", "role=reporter", "role=moderator", "turn_final_words")]
        cells += [show(s["comma_recall_from_comma"]["recall"]), show(s["comma_recall_from_dash"]["recall"])]
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    examples = load_split(args.split)
    docs = sorted({ex["doc"] for ex in examples})
    report = {"split": args.split, "documents": len(docs), "words": sum(len(ex["words"]) for ex in examples),
              "bootstrap": {"unit": "press conference", "resamples": args.resamples, "seed": args.seed},
              "models": {}, "comparisons": []}
    cms = {}
    for model in args.models:
        predictions = load_predictions(model, args.split, examples)
        cms[model] = {view: doc_confusions(examples, predictions, docs, within_turn=view == "within_turn")
                      for view in ("all_words", "within_turn")}
        result = {view: with_cis(cms[model][view], args.resamples, args.seed) for view in cms[model]}
        result["slices"] = slices(examples, predictions)
        latency = RESULTS / "latency" / f"{model}.{args.split}.json"
        if latency.exists():
            result["latency"] = json.loads(latency.read_text(encoding="utf-8"))
        report["models"][model] = result
    for i, a in enumerate(args.models):
        for b in args.models[:i]:
            report["comparisons"].append({"model": a, "versus": b, **{
                view: paired_bootstrap(cms[a][view], cms[b][view], n_resamples=args.resamples, seed=args.seed)
                for view in ("all_words", "within_turn")}})

    report["seed_groups"] = seed_groups(report)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{args.split}_metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = summary_lines(report)
    (RESULTS / f"{args.split}_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
