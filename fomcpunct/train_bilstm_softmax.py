"""Ablation run: the chosen BiLSTM-CRF setting without the CRF layer (CPU).

Usage:
    python -m fomcpunct.train_bilstm_softmax --seeds 0 1 2 [--jobs 3 --threads 2]
    python -m fomcpunct.train_bilstm_softmax --seeds 0 1 2 --split test   # final step only

There is no separate tuning: the model reuses the hidden size, layers and
dropout chosen for the BiLSTM-CRF on validation (results/params/bilstm_crf.json)
and the same fixed settings (train_bilstm.FIXED), so the only difference is the
output layer. That setting was tuned with the CRF, so it may not be the best
one for the softmax model; the comparison answers "what does removing the CRF
from this model cost?", not "which is the best possible BiLSTM?". The run
refuses to start if the BiLSTM-CRF tuning record was made under a different
fingerprint.

Models are saved as models/bilstm_softmax_s<seed>.pt with a card (.json)
holding the same kind of fingerprint as the BiLSTM-CRF cards (here the code
hash also covers bilstm_softmax.py). Predictions and latency are written as
model "bilstm_softmax_s<seed>"; learning curves go to results/training/.
--split test never retrains.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

from .bilstm_crf import history_json
from .bilstm_softmax import BiLSTMSoftmaxTagger
from .run_baselines import DATA, MODELS, RESULTS, load_split, sha256, timed_predict, write_json
from .train_bilstm import FIXED, check_tuning

NAME = "bilstm_softmax"
MODEL_CODE = ("bilstm_softmax.py", "bilstm_crf.py", "metrics.py", "__init__.py")
PARAMS_SOURCE = "results/params/bilstm_crf.json (BiLSTM-CRF tuning, seed 0)"


def model_path(seed: int) -> Path:
    return MODELS / f"{NAME}_s{seed}.pt"


def environment(fixed: dict, threads: int) -> dict:
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
    before = environment(fixed, threads)
    tagger = BiLSTMSoftmaxTagger(out_path, seed=seed, threads=threads, **fixed, **params)
    tagger.fit(load_split("train"), load_split("validation"), log=lambda line: print(f"[{label}]{line}", flush=True))
    if environment(fixed, threads) != before:
        raise RuntimeError(f"{label}: code or data changed while training; the model is not trustworthy")
    return {"params": params, "seed": seed, "model_file": out_path, "fingerprint": before, **tagger.summary()}


def save_model(result: dict) -> None:
    if environment(result["fingerprint"]["fixed"], result["fingerprint"]["threads"]) != result["fingerprint"]:
        raise RuntimeError("code or data changed after training; not saving a card for this model")
    path = model_path(result["seed"])
    if Path(result["model_file"]).resolve() != path.resolve():
        shutil.copyfile(result["model_file"], path)
    write_json(path.with_suffix(".json"), {
        "params": result["params"], "params_source": PARAMS_SOURCE, "seed": result["seed"],
        **result["fingerprint"], "model_sha256": sha256(path),
        **{k: result[k] for k in ("best_epoch", "best_validation_macro_f1", "epochs_run", "stopped_early")}})
    tagger = BiLSTMSoftmaxTagger(path, threads=result["fingerprint"]["threads"]).load()
    out = RESULTS / "training" / f"{NAME}_s{result['seed']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(history_json(tagger), encoding="utf-8")


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
    MODELS.mkdir(parents=True, exist_ok=True)
    failed = {}
    with ProcessPoolExecutor(max_workers=min(jobs, len(to_train))) as pool:
        futures = {pool.submit(train_one, params, fixed, seed, threads, str(model_path(seed)), f"seed {seed}"): seed
                   for seed in to_train}
        for future in as_completed(futures):
            try:  # one failed seed must not lose the cards of the others
                result = future.result()
                save_model(result)
            except Exception as error:
                failed[futures[future]] = repr(error)
                print(f"  seed {futures[future]} FAILED: {error!r}", flush=True)
                continue
            print(f"  seed {result['seed']}: best epoch {result['best_epoch']} of {result['epochs_run']}, "
                  f"validation macro-F1 {result['best_validation_macro_f1']:.4f}", flush=True)
    if failed:
        raise RuntimeError(f"seeds failed: {failed}")


def predict_seeds(seeds: list[int], split: str) -> None:
    target = load_split(split)
    for seed in seeds:
        tagger = BiLSTMSoftmaxTagger(model_path(seed), threads=1).load()
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
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    parser.add_argument("--jobs", type=int, default=3, help="parallel training processes")
    parser.add_argument("--threads", type=int, default=2, help="torch threads per training process")
    args = parser.parse_args()

    params = check_tuning(FIXED, args.threads)  # the BiLSTM-CRF setting, if its record is current
    if args.split == "test":
        stale = {seed: card_matches(params, seed, FIXED, args.threads) for seed in args.seeds}
        stale = {seed: reason for seed, reason in stale.items() if reason}
        if stale:
            raise SystemExit(f"saved models do not match (not retraining for the test split): {stale}")
    else:
        train_seeds(params, args.seeds, args.jobs, args.threads)
    predict_seeds(args.seeds, args.split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
