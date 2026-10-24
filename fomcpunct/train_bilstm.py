"""Tune, train and run the BiLSTM-CRF on CPU.

Usage:
    python -m fomcpunct.train_bilstm --tune [--jobs 8 --threads 2]
    python -m fomcpunct.train_bilstm --seeds 0 1 2 [--jobs 3 --threads 2]
    python -m fomcpunct.train_bilstm --seeds 0 1 2 --split test      # final step only

--tune trains every GRID setting with seed 0 (early stopping on validation
macro-F1, at most FIXED["max_epochs"] epochs) and keeps the best one
(results/tuning/bilstm_crf.json, results/params/bilstm_crf.json); its model
becomes the seed-0 model. --seeds trains the chosen setting with each seed.

Every model is saved as models/bilstm_crf_s<seed>.pt with a card (.json). The
card holds a fingerprint taken when training starts and checked again when it
ends: grid and fixed settings, seed, thread count, SHA-256 of train and
validation data, a hash of the code that shapes the model (bilstm_crf.py,
metrics.py, __init__.py) and the torch version. A saved model is reused only if
its fingerprint and file hash still match. --seeds refuses to run if the tuning
record was made under a different fingerprint (rerun --tune), and
--split test never retrains: it stops if a saved model does not match.

Each seed's predictions are written as model "bilstm_crf_s<seed>"
(results/predictions/, results/latency/; timing uses one thread, like the
other baselines). Learning curves go to results/training/.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

from .bilstm_crf import BiLSTMCRFTagger, history_json
from .run_baselines import DATA, MODELS, RESULTS, load_split, sha256, timed_predict, write_json

GRID = {"hidden": [128, 256], "layers": [1, 2], "dropout": [0.3, 0.5]}
FIXED = {"embed_dim": 100, "lr": 2e-3, "batch_size": 16, "max_epochs": 60, "patience": 5,
         "min_count": 2, "word_dropout": 0.05, "clip": 5.0}
NAME = "bilstm_crf"
MODEL_CODE = ("bilstm_crf.py", "metrics.py", "__init__.py")


def model_path(seed: int) -> Path:
    return MODELS / f"{NAME}_s{seed}.pt"


def environment(fixed: dict, threads: int) -> dict:
    """Everything outside the grid point and seed that shapes a trained model."""
    code = hashlib.sha256(b"".join((Path(__file__).parent / name).read_bytes() for name in MODEL_CODE))
    return {"fixed": fixed, "threads": threads, "train_sha256": sha256(DATA / "train.jsonl"),
            "validation_sha256": sha256(DATA / "validation.jsonl"), "code": code.hexdigest()[:16],
            "torch": torch.__version__}


def card_matches(params: dict, seed: int, fixed: dict, threads: int) -> str:
    """Empty string if the saved model can be reused, else the fields that differ."""
    path = model_path(seed)
    card_path = path.with_suffix(".json")
    if not path.exists() or not card_path.exists():
        return "no saved model"
    card = json.loads(card_path.read_text(encoding="utf-8"))
    expected = {"params": params, "seed": seed, **environment(fixed, threads), "model_sha256": sha256(path)}
    return ", ".join(k for k, v in expected.items() if card.get(k) != v)


def train_one(params: dict, fixed: dict, seed: int, threads: int, out_path: str, label: str) -> dict:
    """Train one model in its own process; the fingerprint is taken before and checked after training."""
    before = environment(fixed, threads)
    tagger = BiLSTMCRFTagger(out_path, seed=seed, threads=threads, **fixed, **params)
    tagger.fit(load_split("train"), load_split("validation"), log=lambda line: print(f"[{label}]{line}", flush=True))
    if environment(fixed, threads) != before:
        raise RuntimeError(f"{label}: code or data changed while training; the model is not trustworthy")
    return {"params": params, "seed": seed, "model_file": out_path, "fingerprint": before,
            **tagger.summary(), "history": tagger.history}


def save_model(result: dict) -> None:
    """Copy a trained model to its seed path and write its card and learning curve."""
    if environment(result["fingerprint"]["fixed"], result["fingerprint"]["threads"]) != result["fingerprint"]:
        raise RuntimeError("code or data changed after training; not saving a card for this model")
    path = model_path(result["seed"])
    path.parent.mkdir(parents=True, exist_ok=True)
    if Path(result["model_file"]).resolve() != path.resolve():
        shutil.copyfile(result["model_file"], path)
    write_json(path.with_suffix(".json"), {
        "params": result["params"], "seed": result["seed"], **result["fingerprint"], "model_sha256": sha256(path),
        **{k: result[k] for k in ("best_epoch", "best_validation_macro_f1", "epochs_run", "stopped_early")}})
    tagger = BiLSTMCRFTagger(path, threads=result["fingerprint"]["threads"]).load()
    out = RESULTS / "training" / f"{NAME}_s{result['seed']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(history_json(tagger), encoding="utf-8")


def tune(jobs: int, threads: int, fixed: dict = FIXED, grid: dict = GRID) -> dict:
    keys = list(grid)
    settings = [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]
    MODELS.mkdir(parents=True, exist_ok=True)
    trial_dir = Path(tempfile.mkdtemp(dir=MODELS, prefix="bilstm_trials_"))
    trials = []
    try:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = {pool.submit(train_one, params, fixed, 0, threads, str(trial_dir / f"trial_{i:02d}.pt"),
                                   f"trial {i}"): params for i, params in enumerate(settings)}
            for future in as_completed(futures):
                try:
                    t = future.result()
                except Exception as error:  # keep the other trials
                    trials.append({"params": futures[future], "error": repr(error)})
                    print(f"  {futures[future]} FAILED: {error!r}", flush=True)
                    continue
                trials.append(t)
                print(f"  {t['params']} -> validation macro-F1 {t['best_validation_macro_f1']:.4f} (best epoch "
                      f"{t['best_epoch']} of {t['epochs_run']}, stopped early: {t['stopped_early']})", flush=True)
        done = [t for t in trials if "error" not in t]
        if not done:
            raise RuntimeError("every tuning trial failed")
        trials.sort(key=lambda t: settings.index(t["params"]))
        best = max(done, key=lambda t: t["best_validation_macro_f1"])
        write_json(RESULTS / "tuning" / f"{NAME}.json", {
            "selection": "best validation macro-F1 (early-stopping epoch), seed 0", "grid": grid,
            "fingerprint": environment(fixed, threads),
            "trials": [{k: v for k, v in t.items() if k not in ("model_file", "fingerprint")} for t in trials]})
        save_model(best)
        write_json(RESULTS / "params" / f"{NAME}.json", best["params"])
        return best["params"]
    finally:
        shutil.rmtree(trial_dir, ignore_errors=True)


def check_tuning(fixed: dict, threads: int) -> dict:
    """The chosen params, if the tuning record was made under the current fingerprint."""
    record = json.loads((RESULTS / "tuning" / f"{NAME}.json").read_text(encoding="utf-8"))
    current = environment(fixed, threads)
    changed = [k for k, v in current.items() if record["fingerprint"].get(k) != v]
    if changed:
        raise SystemExit(f"tuning was done under a different {', '.join(changed)}; rerun --tune")
    return json.loads((RESULTS / "params" / f"{NAME}.json").read_text(encoding="utf-8"))


def train_seeds(params: dict, seeds: list[int], jobs: int, threads: int, fixed: dict = FIXED) -> None:
    to_train = []
    for seed in seeds:
        reason = card_matches(params, seed, fixed, threads)
        if reason:
            print(f"  seed {seed}: training ({reason})", flush=True)
            to_train.append(seed)
        else:
            print(f"  seed {seed}: reusing the saved model", flush=True)
    if not to_train:
        return
    with ProcessPoolExecutor(max_workers=min(jobs, len(to_train))) as pool:
        futures = [pool.submit(train_one, params, fixed, seed, threads, str(model_path(seed)), f"seed {seed}")
                   for seed in to_train]
        for future in as_completed(futures):
            result = future.result()
            save_model(result)
            print(f"  seed {result['seed']}: best epoch {result['best_epoch']} of {result['epochs_run']}, "
                  f"validation macro-F1 {result['best_validation_macro_f1']:.4f}", flush=True)


def predict_seeds(seeds: list[int], split: str) -> None:
    target = load_split(split)
    for seed in seeds:
        tagger = BiLSTMCRFTagger(model_path(seed), threads=1).load()
        predictions, latency = timed_predict(tagger, target)
        name = f"{NAME}_s{seed}"
        out = RESULTS / "predictions" / f"{name}.{split}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8", newline="\n") as handle:
            for ex, labels in zip(target, predictions, strict=True):
                handle.write(json.dumps({"id": ex["id"], "labels": labels}) + "\n")
        write_json(RESULTS / "latency" / f"{name}.{split}.json", latency)
        print(f"{name}: wrote {out} ({latency['ms_per_1000_words']} ms / 1000 words)", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--seeds", nargs="*", type=int, default=[])
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    parser.add_argument("--jobs", type=int, default=4, help="parallel training processes")
    parser.add_argument("--threads", type=int, default=2, help="torch threads per training process")
    args = parser.parse_args()
    seeds = args.seeds or [0]

    if args.split == "test":
        if args.tune:
            raise SystemExit("--tune and --split test cannot be combined")
        params = check_tuning(FIXED, args.threads)
        stale = {seed: card_matches(params, seed, FIXED, args.threads) for seed in seeds}
        stale = {seed: reason for seed, reason in stale.items() if reason}
        if stale:
            raise SystemExit(f"saved models do not match (not retraining for the test split): {stale}")
    else:
        params = tune(args.jobs, args.threads) if args.tune else check_tuning(FIXED, args.threads)
        train_seeds(params, seeds, args.jobs, args.threads)
    predict_seeds(seeds, args.split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
