"""Fine-tune a pretrained transformer encoder for punctuation restoration (GPU, e.g. Colab).

Models: BERT (`bert-base-uncased`, a vanilla method from the slides) and
DeBERTa-v3 (`microsoft/deberta-v3-base`, a newer model), pinned to fixed Hub
revisions. Each gets a linear token-classification head over the four labels.
Weights are loaded in float32 (the DeBERTa-v3 checkpoint is stored in float16,
and transformers 5 would otherwise load it as float16, which breaks
mixed-precision training).

Words and subwords. The data are lists of words; the tokenizer splits a word
into one or more subwords. A word's label is put on its first subword; the
other subwords get -100 (ignored by the loss). At prediction time a word takes
the label predicted for its first subword.

Long turns. Turns can be longer than the model input, so each turn is cut into
windows of at most `max_len` subwords (special tokens included) that overlap
by `overlap` words. Every word is owned by exactly one window: the one where it
has the most context on its weaker side (the turn start or end counts as
unlimited context). Only owned words are trained on and predicted from a
window, so the model is never asked to label a word right where a window cuts
the turn, and training and prediction use the same rule. (The exception is a
single word longer than a whole window, which gets a window of its own; no such
word occurs in this data.)

Training: AdamW with linear warm-up and decay, gradient clipping, mixed
precision on GPU, a fixed number of epochs. After every epoch the model labels
the validation split; when validation macro-F1 improves, the validation and
test predictions of that epoch are written to the run folder (model selection
only; test predictions are not scored here). No weights are stored, so a run
needs only a few MB of Drive space. A run is finished when its card.json is
written (last); a run that was cut off starts again from the beginning, and a
finished run is skipped when the same run is requested again. All models are
scored on test once, at the end of the project, with fomcpunct.evaluate.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from . import LABEL2ID, LABELS
from .metrics import confusion, macro_f1

# key -> (Hub model, pinned revision)
MODELS = {"bert": ("bert-base-uncased", "86b5e0934494bd15c9632b12f734a8a67f723594"),
          "deberta": ("microsoft/deberta-v3-base", "8ccc9b6f36199bec6961081d44eb72fb3f7353f3")}
IGNORE = -100
SPLITS = ("train", "validation", "test")
CODE_FILES = ("transformer.py", "metrics.py", "__init__.py")
# Hash of the code as imported, so a run is tied to the code that actually ran.
CODE_HASH = hashlib.sha256(b"".join((Path(__file__).parent / name).read_bytes() for name in CODE_FILES)).hexdigest()[:16]


@dataclass
class Config:
    max_len: int = 256        # subwords per window, special tokens included
    overlap: int = 64         # words shared by neighbouring windows
    batch_size: int = 16
    eval_batch_size: int = 64
    epochs: int = 4
    warmup: float = 0.1       # share of steps with linear warm-up
    weight_decay: float = 0.01
    clip: float = 1.0
    fp16: bool = True         # mixed precision, used only on a CUDA device


# ---- windows ------------------------------------------------------------------
def subword_counts(tokenizer, words: list[str]) -> list[int]:
    """Number of subwords of each word, tokenized the same way as in a split-into-words input."""
    if not words:
        return []
    ids = tokenizer(words, add_special_tokens=False)["input_ids"]
    return [len(x) for x in ids]


def make_windows(counts: list[int], budget: int, overlap: int) -> list[tuple[int, int]]:
    """Greedy windows [start, end) of words whose subwords fit in `budget`.

    Neighbours share `overlap` words, but never more than half a window, so a
    window always moves forward by at least half its length (otherwise windows
    holding fewer words than `overlap` would advance one word at a time).
    """
    n, windows, start = len(counts), [], 0
    while start < n:
        end, total = start, 0
        while end < n and total + counts[end] <= budget:
            total += counts[end]
            end += 1
        end = max(end, start + 1)  # a single over-long word still gets its own window
        windows.append((start, end))
        if end >= n:
            break
        start = max(start + 1, end - min(overlap, (end - start) // 2))
    return windows


def owners(n: int, windows: list[tuple[int, int]]) -> list[int]:
    """Index of the window that owns each word: the most context on the weaker side wins (ties: earlier)."""
    best, best_context = [-1] * n, [-1.0] * n
    for k, (start, end) in enumerate(windows):
        for i in range(start, end):
            left = math.inf if start == 0 else i - start
            right = math.inf if end == n else end - 1 - i
            context = min(left, right)
            if context > best_context[i]:
                best[i], best_context[i] = k, context
    return best


def encode_examples(examples: list[dict], tokenizer, cfg: Config, with_labels: bool) -> list[dict]:
    """Windows ready for the model; each owned word gets a target (subword position, word index)."""
    items = []
    for t, ex in enumerate(examples):
        words = ex["words"]
        windows = make_windows(subword_counts(tokenizer, words), cfg.max_len - 2, cfg.overlap)
        own = owners(len(words), windows)
        for k, (start, end) in enumerate(windows):
            enc = tokenizer(words[start:end], is_split_into_words=True, truncation=True, max_length=cfg.max_len)
            first: dict[int, int] = {}
            for position, w in enumerate(enc.word_ids()):
                if w is not None and w not in first:
                    first[w] = position
            if len(first) != end - start:  # truncation dropped words: the counts were wrong
                raise RuntimeError(f"turn {ex.get('id', t)}: window {start}-{end} lost words to truncation")
            labels = [IGNORE] * len(enc["input_ids"])
            targets = []
            for w, position in first.items():
                i = start + w
                if own[i] == k:
                    targets.append((position, i))
                    if with_labels:
                        labels[position] = LABEL2ID[ex["labels"][i]]
            items.append({"turn": t, "input_ids": enc["input_ids"], "labels": labels, "targets": targets})
    return items


def coverage_problems(items: list[dict], examples: list[dict]) -> int:
    """Number of words that are not predicted exactly once (should be 0)."""
    seen = [[0] * len(ex["words"]) for ex in examples]
    for item in items:
        for _, i in item["targets"]:
            seen[item["turn"]][i] += 1
    return sum(1 for turn in seen for count in turn if count != 1)


def collate(items: list[dict], pad_id: int) -> dict:
    width = max(len(x["input_ids"]) for x in items)
    ids = torch.full((len(items), width), pad_id, dtype=torch.long)
    labels = torch.full((len(items), width), IGNORE, dtype=torch.long)
    attention = torch.zeros((len(items), width), dtype=torch.long)
    for b, x in enumerate(items):
        n = len(x["input_ids"])
        ids[b, :n] = torch.tensor(x["input_ids"])
        labels[b, :n] = torch.tensor(x["labels"])
        attention[b, :n] = 1
    return {"input_ids": ids, "attention_mask": attention, "labels": labels}


# ---- prediction ---------------------------------------------------------------
@torch.no_grad()
def predict(model, tokenizer, examples: list[dict], cfg: Config, device, items: list[dict] | None = None):
    """Labels for every word of every turn (items: pre-encoded windows, to skip re-tokenizing)."""
    model.eval()
    if items is None:
        items = encode_examples(examples, tokenizer, cfg, with_labels=False)
    if coverage_problems(items, examples):
        raise RuntimeError("some words are not covered exactly once by the windows")
    out = [["O"] * len(ex["words"]) for ex in examples]
    order = sorted(range(len(items)), key=lambda j: len(items[j]["input_ids"]))
    use_amp = cfg.fp16 and device.type == "cuda"
    for start in range(0, len(order), cfg.eval_batch_size):
        batch = [items[j] for j in order[start:start + cfg.eval_batch_size]]
        inputs = collate(batch, tokenizer.pad_token_id)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(input_ids=inputs["input_ids"].to(device),
                           attention_mask=inputs["attention_mask"].to(device)).logits
        best = logits.argmax(dim=-1).cpu()
        for b, item in enumerate(batch):
            for position, i in item["targets"]:
                out[item["turn"]][i] = LABELS[int(best[b, position])]
    return out


# ---- bookkeeping --------------------------------------------------------------
def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_split(data_dir: Path, name: str) -> list[dict]:
    with (Path(data_dir) / f"{name}.jsonl").open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def identity(model_key: str, lr: float, seed: int, cfg: Config, data_dir: Path,
             model: tuple[str, str | None] | None = None) -> dict:
    """What makes two runs the same run; a finished run is skipped only under the same identity."""
    name, revision = model or MODELS[model_key]
    return {"model": name, "revision": revision, "lr": lr, "seed": seed, "config": asdict(cfg),
            "data": {s: sha256(Path(data_dir) / f"{s}.jsonl") for s in SPLITS}, "code": CODE_HASH}


def environment(device) -> dict:
    import transformers
    return {"torch": torch.__version__, "transformers": transformers.__version__,
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "cuda": torch.version.cuda}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_predictions(path: Path, examples: list[dict], predictions: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for ex, labels in zip(examples, predictions, strict=True):
            handle.write(json.dumps({"id": ex["id"], "labels": labels}) + "\n")
    os.replace(tmp, path)


def run_dir_for(workdir: Path, model_key: str, lr: float, seed: int) -> Path:
    return Path(workdir) / "runs" / f"{model_key}_lr{lr:g}_s{seed}"


# ---- one training run ---------------------------------------------------------
def train_run(model_key: str, lr: float, seed: int, data_dir: Path, workdir: Path, cfg: Config = Config(),
              model: tuple[str, str | None] | None = None, log=print) -> dict:
    """Train one run (or skip it if it already finished); returns its card.

    `model` overrides the (name, revision) pair from MODELS; the tests use tiny models.
    """
    from transformers import AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup

    run_dir = run_dir_for(workdir, model_key, lr, seed)
    card_path = run_dir / "card.json"
    say = lambda msg: log(f"[{run_dir.name}] {msg}")  # noqa: E731
    expected = identity(model_key, lr, seed, cfg, data_dir, model)
    if card_path.exists():
        card = json.loads(card_path.read_text(encoding="utf-8"))
        if card["identity"] == expected:
            say("already finished; skipping")
            return card
        raise RuntimeError(f"{run_dir} holds a finished run with a different identity; rename it first")
    if run_dir.exists():  # leftovers of a run that was cut off
        say("unfinished run found; starting it again")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)
    name, revision = expected["model"], expected["revision"]
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
    net = AutoModelForTokenClassification.from_pretrained(
        name, revision=revision, dtype=torch.float32, num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)), label2id=LABEL2ID, ignore_mismatched_sizes=True).to(device)
    dtypes = sorted({str(p.dtype) for p in net.parameters()})
    if dtypes != ["torch.float32"]:
        raise RuntimeError(f"model parameters must be float32, got {dtypes}")

    splits = {s: load_split(data_dir, s) for s in SPLITS}
    train_items = encode_examples(splits["train"], tokenizer, cfg, with_labels=True)
    eval_items = {s: encode_examples(splits[s], tokenizer, cfg, with_labels=False) for s in ("validation", "test")}
    for s, ex in (("train", splits["train"]), *((s, splits[s]) for s in eval_items)):
        if coverage_problems(train_items if s == "train" else eval_items[s], ex):
            raise RuntimeError(f"some {s} words are not covered exactly once")
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(train_items, batch_size=cfg.batch_size, shuffle=True, generator=generator,
                                         collate_fn=lambda b: collate(b, tokenizer.pad_token_id))
    no_decay = ("bias", "LayerNorm.weight", "layernorm", "layer_norm")
    groups = [{"params": [p for n, p in net.named_parameters() if not any(k in n for k in no_decay)],
               "weight_decay": cfg.weight_decay},
              {"params": [p for n, p in net.named_parameters() if any(k in n for k in no_decay)],
               "weight_decay": 0.0}]
    optimizer = torch.optim.AdamW(groups, lr=lr)
    total_steps = cfg.epochs * len(loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(cfg.warmup * total_steps), total_steps)
    use_amp = cfg.fp16 and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    gold_validation = [ex["labels"] for ex in splits["validation"]]
    history, best_score, best_epoch = [], -1.0, 0
    for epoch in range(1, cfg.epochs + 1):
        net.train()
        started, total_loss, steps = time.time(), 0.0, 0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = net(**batch).loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"loss is {loss.item()} at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss, steps = total_loss + loss.item(), steps + 1
        validation = predict(net, tokenizer, splits["validation"], cfg, device, eval_items["validation"])
        score = float(macro_f1(confusion(gold_validation, validation)))
        history.append({"epoch": epoch, "train_loss": round(total_loss / steps, 5),
                        "validation_macro_f1": round(score, 5), "minutes": round((time.time() - started) / 60, 2)})
        if score > best_score:  # keep this epoch's predictions instead of its weights
            best_score, best_epoch = score, epoch
            write_predictions(run_dir / "predictions.validation.jsonl", splits["validation"], validation)
            write_predictions(run_dir / "predictions.test.jsonl", splits["test"],
                              predict(net, tokenizer, splits["test"], cfg, device, eval_items["test"]))
        write_json(run_dir / "progress.json", {"history": history, "best_epoch": best_epoch})
        say(f"epoch {epoch}: train loss {total_loss / steps:.4f}, validation macro-F1 {score:.4f}"
            + (" (best so far)" if best_epoch == epoch else ""))

    card = {"identity": expected, "environment": environment(device), "best_epoch": best_epoch,
            "best_validation_macro_f1": round(best_score, 5), "history": history,
            "parameters": sum(p.numel() for p in net.parameters()), "parameter_dtypes": dtypes,
            "train_windows": len(train_items),
            "latency": latency(net, tokenizer, splits["validation"], cfg, device)}
    write_json(card_path, card)  # written last: a card means the run is complete
    say(f"done: best epoch {best_epoch}, validation macro-F1 {best_score:.4f}")
    return card


def latency(net, tokenizer, examples: list[dict], cfg: Config, device, runs: int = 3) -> dict:
    """Wall-clock time to label `examples` (tokenization included), median of `runs` after a warm-up.

    Measured with the final epoch's weights; the time does not depend on the weights.
    """
    predict(net, tokenizer, examples, cfg, device)
    times = []
    for _ in range(runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        predict(net, tokenizer, examples, cfg, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    n_words = sum(len(ex["words"]) for ex in examples)
    per_k = [t / n_words * 1e6 for t in times]
    amp = cfg.fp16 and device.type == "cuda"
    return {"words": n_words, "timed_runs": runs, "ms_per_1000_words": round(statistics.median(per_k), 2),
            "ms_per_1000_words_min": round(min(per_k), 2), "ms_per_1000_words_max": round(max(per_k), 2),
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "batch_size": cfg.eval_batch_size,
            "note": f"{device.type}, batches of {cfg.eval_batch_size} windows, {'fp16' if amp else 'fp32'}"}


# ---- selection and export -----------------------------------------------------
def select_lr(model_key: str, lrs: list[float], workdir: Path) -> float:
    """Best learning rate by the seed-0 runs' validation macro-F1; writes the tuning record."""
    trials = []
    for lr in lrs:
        card_path = run_dir_for(workdir, model_key, lr, 0) / "card.json"
        if not card_path.exists():
            raise RuntimeError(f"seed-0 run for lr={lr:g} has not finished")
        card = json.loads(card_path.read_text(encoding="utf-8"))
        trials.append({"lr": lr, "best_epoch": card["best_epoch"],
                       "best_validation_macro_f1": card["best_validation_macro_f1"], "history": card["history"]})
    best = max(trials, key=lambda t: t["best_validation_macro_f1"])
    write_json(Path(workdir) / "results" / "tuning" / f"{model_key}.json",
               {"selection": "best validation macro-F1 (best epoch), seed 0", "trials": trials})
    write_json(Path(workdir) / "results" / "params" / f"{model_key}.json", {"lr": best["lr"]})
    return best["lr"]


def export(model_key: str, lr: float, seeds: list[int], workdir: Path) -> None:
    """Copy the chosen runs' outputs into results/ under the names <model_key>_s<seed>."""
    results = Path(workdir) / "results"
    for seed in seeds:
        run_dir = run_dir_for(workdir, model_key, lr, seed)
        card = json.loads((run_dir / "card.json").read_text(encoding="utf-8"))
        name = f"{model_key}_s{seed}"
        for split in ("validation", "test"):
            target = results / "predictions" / f"{name}.{split}.jsonl"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(run_dir / f"predictions.{split}.jsonl", target)
        write_json(results / "latency" / f"{name}.validation.json", card["latency"])
        write_json(results / "latency" / f"{name}.test.json", {**card["latency"], "measured_on": "validation"})
        write_json(results / "training" / f"{name}.json", card)
