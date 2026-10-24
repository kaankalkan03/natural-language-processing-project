"""Tune the classical baselines on validation, then write predictions.

Usage:
    python -m fomcpunct.run_baselines --models heuristic hmm crf --tune [--jobs 6]
    python -m fomcpunct.run_baselines --models heuristic hmm crf --split test

--tune fits every grid setting on train and keeps the one with the best
validation macro-F1 (results/tuning/<model>.json, results/params/<model>.json).
CRF trials run in parallel processes, each with its own model file in a
folder of its own; the best model is kept as models/crf.crfsuite next to
models/crf.json (parameters, model and train-data SHA-256, feature-code
fingerprint, crfsuite version, training log). Later runs reuse it only if all
of these still match; otherwise the CRF is retrained. HMM and heuristic are
refit from the saved parameters (deterministic, a few seconds).

Predictions go to results/predictions/<model>.<split>.jsonl and CPU latency to
results/latency/<model>.<split>.json (median of 5 timed runs after a warm-up;
time this in a separate run without --tune, with nothing else running).
The test split is only predicted when asked for explicitly, after all model
choices are fixed.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import platform
import shutil
import statistics
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import version
from pathlib import Path

from .crf import CRFTagger, feature_code_hash
from .heuristic import HeuristicTagger
from .hmm import HMMTagger
from .metrics import confusion, macro_f1

GRIDS = {
    "heuristic": {},
    "hmm": {"k_emit": [0.001, 0.01, 0.1, 1.0], "min_count": [1, 2, 3]},
    "crf": {"c1": [0.1, 0.3, 0.5, 1.0, 2.0], "c2": [0.001, 0.01, 0.1, 0.3, 1.0], "max_iterations": [1000]},
}
DATA = Path("data/processed")
RESULTS, MODELS = Path("results"), Path("models")
CRF_MODEL, CRF_CARD = MODELS / "crf.crfsuite", MODELS / "crf.json"
TIMED_RUNS = 5


def load_split(name: str, data_dir: Path = DATA) -> list[dict]:
    with (data_dir / f"{name}.jsonl").open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validation_score(tagger, validation: list[dict]) -> float:
    predictions = tagger.predict([ex["words"] for ex in validation])
    return round(float(macro_f1(confusion([ex["labels"] for ex in validation], predictions))), 5)


def grid(model: str) -> list[dict]:
    keys = list(GRIDS[model])
    return [dict(zip(keys, values)) for values in itertools.product(*(GRIDS[model][k] for k in keys))]


def crf_card(params: dict, training_log: dict) -> dict:
    """What a saved CRF depends on; a later run reuses the model only if all of it matches."""
    return {"params": params, "model_sha256": sha256(CRF_MODEL), "train_sha256": sha256(DATA / "train.jsonl"),
            "feature_code": feature_code_hash(), "python_crfsuite": version("python-crfsuite"),
            **training_log}


def crf_trial(index: int, params: dict, trial_dir: str) -> dict:
    """One CRF grid point in its own process and model file."""
    tagger = CRFTagger(Path(trial_dir) / f"trial_{index:02d}.crfsuite", **params).fit(load_split("train"))
    return {"params": params, "validation_macro_f1": validation_score(tagger, load_split("validation")),
            "train_sha256": sha256(DATA / "train.jsonl"), "model_file": str(tagger.model_path),
            **tagger.training_log}


def tune(model: str, train: list[dict], validation: list[dict], jobs: int) -> dict:
    trial_dir = None
    if model == "crf":
        MODELS.mkdir(parents=True, exist_ok=True)
        trial_dir = Path(tempfile.mkdtemp(dir=MODELS, prefix="crf_trials_"))  # one folder per run
        train_hash = sha256(DATA / "train.jsonl")
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = [pool.submit(crf_trial, i, params, str(trial_dir)) for i, params in enumerate(grid(model))]
            trials = [future.result() for future in futures]
        if any(t.pop("train_sha256") != train_hash for t in trials):
            raise RuntimeError("train.jsonl changed during CRF tuning")
    else:
        trials = []
        for params in grid(model) or [{}]:
            trials.append({"params": params,
                           "validation_macro_f1": validation_score(build(model, params).fit(train), validation)})
    for trial in trials:
        print(f"  {model} {trial['params']} -> validation macro-F1 {trial['validation_macro_f1']:.4f}"
              + (f" ({trial['iterations']} iterations, converged={trial['converged']})" if model == "crf" else ""))
    best = max(trials, key=lambda t: t["validation_macro_f1"])
    write_json(RESULTS / "tuning" / f"{model}.json", {
        "selection": "validation macro-F1",
        "trials": [{k: v for k, v in t.items() if k != "model_file"} for t in trials]})
    if model == "crf":  # keep the best trial's model instead of retraining it
        shutil.copyfile(best["model_file"], CRF_MODEL)
        write_json(CRF_CARD, crf_card(best["params"], {k: best[k] for k in (
            "iterations", "loss", "active_features", "converged", "termination")}))
        shutil.rmtree(trial_dir, ignore_errors=True)
    write_json(RESULTS / "params" / f"{model}.json", best["params"])
    return best["params"]


def build(model: str, params: dict):
    if model == "heuristic":
        return HeuristicTagger()
    if model == "hmm":
        return HMMTagger(**params)
    raise ValueError(model)


def ready_tagger(model: str, params: dict, train: list[dict]):
    if model != "crf":
        return build(model, params).fit(train)  # deterministic refit
    card = json.loads(CRF_CARD.read_text(encoding="utf-8")) if CRF_CARD.exists() else {}
    if not CRF_MODEL.exists():
        reason = "no saved model"
    else:
        expected = {"params": params, "model_sha256": sha256(CRF_MODEL), "train_sha256": sha256(DATA / "train.jsonl"),
                    "feature_code": feature_code_hash(), "python_crfsuite": version("python-crfsuite")}
        reason = ", ".join(key for key, value in expected.items() if card.get(key) != value)
    if not reason:
        return CRFTagger(CRF_MODEL, **params).load()
    print(f"  retraining the CRF (saved model does not match: {reason})")
    tagger = CRFTagger(CRF_MODEL, **params).fit(train)
    write_json(CRF_CARD, crf_card(params, tagger.training_log))
    return tagger


def timed_predict(tagger, examples: list[dict]) -> tuple[list[list[str]], dict]:
    """Predict once as a warm-up, then time TIMED_RUNS full passes (single process, CPU)."""
    sequences = [ex["words"] for ex in examples]
    n_words = sum(len(s) for s in sequences)
    predictions = tagger.predict(sequences)
    wall = []
    for _ in range(TIMED_RUNS):
        start = time.perf_counter()
        again = tagger.predict(sequences)
        wall.append(time.perf_counter() - start)
        if again != predictions:
            raise RuntimeError("predictions changed between runs")
    per_k = [s / n_words * 1e6 for s in wall]  # milliseconds per 1,000 words
    # Wall-clock only: Windows process CPU time advances in 15.6 ms steps, which is
    # longer than one heuristic pass, so it cannot be measured this way.
    return predictions, {
        "words": n_words, "timed_runs": TIMED_RUNS,
        "ms_per_1000_words": round(statistics.median(per_k), 2),
        "ms_per_1000_words_min": round(min(per_k), 2),
        "ms_per_1000_words_max": round(max(per_k), 2),
        "device": "cpu (one process, one thread)", "processor": platform.processor(),
        "cpu_count": os.cpu_count(), "python": platform.python_version(),
        "note": ("wall-clock median after a warm-up, including feature extraction. The CPU mixes "
                 "performance and efficiency cores, so repeated runs can differ by up to about 4x; "
                 "compare orders of magnitude only."),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", default=["heuristic", "hmm", "crf"], choices=list(GRIDS))
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--jobs", type=int, default=4, help="parallel CRF trials")
    args = parser.parse_args()

    train = load_split("train")
    validation = load_split("validation") if args.tune else None
    target = load_split(args.split)
    for model in args.models:
        if args.tune:
            params = tune(model, train, validation, args.jobs)
        else:
            params = json.loads((RESULTS / "params" / f"{model}.json").read_text(encoding="utf-8"))
        tagger = ready_tagger(model, params, train)
        predictions, latency = timed_predict(tagger, target)
        out = RESULTS / "predictions" / f"{model}.{args.split}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8", newline="\n") as handle:
            for ex, labels in zip(target, predictions, strict=True):
                handle.write(json.dumps({"id": ex["id"], "labels": labels}) + "\n")
        write_json(RESULTS / "latency" / f"{model}.{args.split}.json", latency)
        print(f"{model}: params={params} wrote {out} ({latency['ms_per_1000_words']} ms / 1000 words)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
