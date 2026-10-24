"""BiLSTM-CRF tagger written from scratch in PyTorch (no CRF library).

Architecture (Lample et al., 2016, without character features):

    word ids -> embedding (trained from scratch) -> dropout
             -> bidirectional LSTM -> dropout -> linear layer: one score per label
             -> linear-chain CRF (label transitions, start and end scores)

The CRF is trained with the negative log-likelihood of the gold label path:

    loss = log Z(x) - score(x, y),  score = start[y1] + sum_t emit[t, y_t]
                                            + sum_t trans[y_{t-1}, y_t] + end[y_n]

where log Z is computed with the forward algorithm (log-sum-exp over all
paths) and decoding uses the Viterbi algorithm. Both run on padded batches with
a mask. Digits are mapped to 0 ("2.5" -> "0.0") and words seen fewer than
`min_count` times in training become <unk>; during training each word is
replaced by <unk> with probability `word_dropout`, so <unk> gets a useful
embedding. Training uses Adam, gradient clipping and early stopping on the
validation macro-F1 (model selection only; the test split is not used).

Loss weighting (a deliberate choice): batches hold turns of similar length,
and each batch's summed NLL is divided by its number of words, so every batch
counts about the same. Short turns (often reporter questions) therefore weigh
more per word than long Chair answers. In a small check during the code audit
this raised validation macro-F1 by about 0.04, mostly through QUESTION.
"""

from __future__ import annotations

import copy
import json
import re
from collections import Counter
from pathlib import Path

import torch
from torch import nn

from . import LABEL2ID, LABELS
from .metrics import confusion, macro_f1

PAD, UNK = "<pad>", "<unk>"


def normalize(word: str) -> str:
    return re.sub(r"\d", "0", word)


class CRF(nn.Module):
    """Linear-chain CRF over K labels with start, transition and end scores."""

    def __init__(self, n_labels: int):
        super().__init__()
        self.start = nn.Parameter(torch.empty(n_labels).uniform_(-0.1, 0.1))
        self.end = nn.Parameter(torch.empty(n_labels).uniform_(-0.1, 0.1))
        self.trans = nn.Parameter(torch.empty(n_labels, n_labels).uniform_(-0.1, 0.1))  # [previous, current]

    def log_partition(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """log Z for each sequence. emissions (B, T, K); mask (B, T) bool with mask[:, 0] True."""
        # unbind once: indexing emissions[:, t] inside the loop would make the
        # backward pass allocate a full-size gradient per step (quadratic in T)
        steps, live = emissions.unbind(1), mask.unsqueeze(2).unbind(1)
        alpha = self.start + steps[0]
        for t in range(1, len(steps)):
            step = torch.logsumexp(alpha.unsqueeze(2) + self.trans, dim=1) + steps[t]
            alpha = torch.where(live[t], step, alpha)
        return torch.logsumexp(alpha + self.end, dim=1)

    def path_score(self, emissions: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Score of the given label path for each sequence (vectorized). labels (B, T) long."""
        weights = mask.to(emissions.dtype)
        emitted = emissions.gather(2, labels.unsqueeze(2)).squeeze(2)          # (B, T)
        moves = self.trans[labels[:, :-1], labels[:, 1:]]                       # (B, T-1)
        last = labels.gather(1, (mask.sum(dim=1) - 1).unsqueeze(1)).squeeze(1)
        return (self.start[labels[:, 0]] + (emitted * weights).sum(dim=1)
                + (moves * weights[:, 1:]).sum(dim=1) + self.end[last])

    def nll(self, emissions: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Summed negative log-likelihood of the gold paths."""
        return (self.log_partition(emissions, mask) - self.path_score(emissions, labels, mask)).sum()

    @torch.no_grad()
    def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> list[list[int]]:
        """Viterbi: best label path for each sequence (padded steps keep the score and point to themselves)."""
        n_batch, _, n_labels = emissions.shape
        identity = torch.arange(n_labels).expand(n_batch, n_labels)
        steps, lives = emissions.unbind(1), mask.unsqueeze(2).unbind(1)
        score = self.start + steps[0]
        pointers = []
        for t in range(1, len(steps)):
            best, previous = (score.unsqueeze(2) + self.trans).max(dim=1)
            score = torch.where(lives[t], best + steps[t], score)
            pointers.append(torch.where(lives[t], previous, identity))
        label = (score + self.end).argmax(dim=1)
        path = [label]
        for back in reversed(pointers):
            label = back.gather(1, label.unsqueeze(1)).squeeze(1)
            path.append(label)
        path = torch.stack(path[::-1], dim=1)
        lengths = mask.sum(dim=1).tolist()
        return [path[b, :n].tolist() for b, n in enumerate(lengths)]


def reverse_within_length(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Reverse each sequence of a padded batch (B, T, D) inside its own length; padding stays at the end."""
    steps = torch.arange(x.size(1)).unsqueeze(0).expand(x.size(0), -1)
    flipped = lengths.unsqueeze(1) - 1 - steps
    index = torch.where(flipped >= 0, flipped, steps)
    return x.gather(1, index.unsqueeze(2).expand_as(x))


class BiLSTM(nn.Module):
    """Bidirectional LSTM on padded batches without packing.

    The forward LSTM reads the padded batch as is (padding comes after the
    words, so it cannot affect them). The backward LSTM reads each sequence
    reversed inside its own length, so it also starts at the real last word.
    This gives the same outputs as a packed nn.LSTM(bidirectional=True) at all
    real positions (checked in the tests); packed sequences are avoided because
    their backward pass is very slow on CPU for long turns (27.7 s against
    0.17 s for a 16 x 2,184 batch here).
    """

    def __init__(self, input_dim: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.forward_lstms = nn.ModuleList(
            nn.LSTM(input_dim if i == 0 else 2 * hidden, hidden, batch_first=True) for i in range(layers))
        self.backward_lstms = nn.ModuleList(
            nn.LSTM(input_dim if i == 0 else 2 * hidden, hidden, batch_first=True) for i in range(layers))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        for i, (fwd, bwd) in enumerate(zip(self.forward_lstms, self.backward_lstms)):
            if i > 0:
                x = self.drop(x)  # between layers, as in nn.LSTM(dropout=...)
            ahead, _ = fwd(x)
            behind, _ = bwd(reverse_within_length(x, lengths))
            x = torch.cat([ahead, reverse_within_length(behind, lengths)], dim=2)
        return x


class BiLSTMCRF(nn.Module):
    def __init__(self, vocab_size: int, n_labels: int, embed_dim: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.drop = nn.Dropout(dropout)
        self.lstm = BiLSTM(embed_dim, hidden, layers, dropout)
        self.out = nn.Linear(2 * hidden, n_labels)
        self.crf = CRF(n_labels)

    def emissions(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        h = self.lstm(self.drop(self.embed(ids)), lengths)
        return self.out(self.drop(h))


class BiLSTMCRFTagger:
    name = "bilstm_crf"

    def __init__(self, model_path: str | Path, embed_dim: int = 100, hidden: int = 128, layers: int = 1,
                 dropout: float = 0.5, lr: float = 2e-3, batch_size: int = 16, max_epochs: int = 60,
                 patience: int = 5, min_count: int = 2, word_dropout: float = 0.05, clip: float = 5.0,
                 seed: int = 0, threads: int = 4):
        self.model_path = Path(model_path)
        self.config = {"embed_dim": embed_dim, "hidden": hidden, "layers": layers, "dropout": dropout,
                       "lr": lr, "batch_size": batch_size, "max_epochs": max_epochs, "patience": patience,
                       "min_count": min_count, "word_dropout": word_dropout, "clip": clip}
        self.seed, self.threads = seed, threads
        self.history: list[dict] = []

    # ---- data -------------------------------------------------------------
    def _ids(self, words: list[str]) -> list[int]:
        unk = self.vocab[UNK]
        return [self.vocab.get(normalize(w), unk) for w in words]

    def _batch(self, sequences: list[list[int]], labels: list[list[int]] | None = None):
        lengths = torch.tensor([len(s) for s in sequences])
        width = int(lengths.max())
        ids = torch.zeros(len(sequences), width, dtype=torch.long)
        tags = torch.zeros(len(sequences), width, dtype=torch.long)
        for i, s in enumerate(sequences):
            ids[i, :len(s)] = torch.tensor(s)
            if labels is not None:
                tags[i, :len(s)] = torch.tensor(labels[i])
        mask = torch.arange(width).unsqueeze(0) < lengths.unsqueeze(1)
        return ids, tags, mask, lengths

    # ---- training ---------------------------------------------------------
    def fit(self, train: list[dict], validation: list[dict], log=print) -> "BiLSTMCRFTagger":
        torch.manual_seed(self.seed)
        torch.set_num_threads(self.threads)
        cfg = self.config
        counts = Counter(normalize(w) for ex in train for w in ex["words"])
        kept = sorted(w for w, c in counts.items() if c >= cfg["min_count"])
        self.vocab = {w: i for i, w in enumerate([PAD, UNK, *kept])}
        self.model = BiLSTMCRF(len(self.vocab), len(LABELS), cfg["embed_dim"], cfg["hidden"],
                               cfg["layers"], cfg["dropout"])
        optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg["lr"])
        generator = torch.Generator().manual_seed(self.seed)

        data = [(self._ids(ex["words"]), [LABEL2ID[y] for y in ex["labels"]]) for ex in train]
        data.sort(key=lambda pair: len(pair[0]))  # batches of similar length waste less padding
        batches = [data[i:i + cfg["batch_size"]] for i in range(0, len(data), cfg["batch_size"])]
        gold = [ex["labels"] for ex in validation]
        best_score, best_state, best_epoch, waited = -1.0, None, 0, 0
        for epoch in range(1, cfg["max_epochs"] + 1):
            self.model.train()
            total_loss, total_words = 0.0, 0
            for b in torch.randperm(len(batches), generator=generator).tolist():
                seqs, labels = zip(*batches[b])
                ids, tags, mask, lengths = self._batch(list(seqs), list(labels))
                if cfg["word_dropout"] > 0:
                    drop = (torch.rand(ids.shape, generator=generator) < cfg["word_dropout"]) & mask
                    ids = ids.masked_fill(drop, self.vocab[UNK])
                loss = self.model.crf.nll(self.model.emissions(ids, lengths), tags, mask)
                n_words = int(mask.sum())
                optimizer.zero_grad()
                (loss / n_words).backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg["clip"])
                optimizer.step()
                total_loss, total_words = total_loss + loss.item(), total_words + n_words
            score = float(macro_f1(confusion(gold, self.predict([ex["words"] for ex in validation]))))
            self.history.append({"epoch": epoch, "train_nll_per_word": round(total_loss / total_words, 5),
                                 "validation_macro_f1": round(score, 5)})
            log(f"    epoch {epoch:2d}: train NLL/word {total_loss / total_words:.4f}, validation macro-F1 {score:.4f}")
            if score > best_score:
                best_score, best_state, best_epoch, waited = score, copy.deepcopy(self.model.state_dict()), epoch, 0
            else:
                waited += 1
                if waited >= cfg["patience"]:
                    break
        self.model.load_state_dict(best_state)
        self.best_epoch, self.best_validation_macro_f1 = best_epoch, best_score
        # True if validation stopped improving; False if max_epochs cut training short
        self.stopped_early = waited >= cfg["patience"]
        self.save()
        return self

    # ---- inference --------------------------------------------------------
    @torch.no_grad()
    def predict(self, sequences: list[list[str]], batch_size: int = 64) -> list[list[str]]:
        self.model.eval()
        order = sorted(range(len(sequences)), key=lambda i: len(sequences[i]))
        out: list[list[str] | None] = [None] * len(sequences)
        for start in range(0, len(order), batch_size):
            chunk = order[start:start + batch_size]
            nonempty = [i for i in chunk if sequences[i]]
            for i in chunk:
                if not sequences[i]:
                    out[i] = []
            if not nonempty:
                continue
            ids, _, mask, lengths = self._batch([self._ids(sequences[i]) for i in nonempty])
            paths = self.model.crf.decode(self.model.emissions(ids, lengths), mask)
            for i, path in zip(nonempty, paths):
                out[i] = [LABELS[k] for k in path]
        return out

    # ---- persistence ------------------------------------------------------
    def save(self) -> None:
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": self.model.state_dict(), "vocab": self.vocab, "config": self.config,
                    "seed": self.seed, "best_epoch": self.best_epoch, "stopped_early": self.stopped_early,
                    "best_validation_macro_f1": self.best_validation_macro_f1, "history": self.history},
                   self.model_path)

    def load(self) -> "BiLSTMCRFTagger":
        torch.set_num_threads(self.threads)
        saved = torch.load(self.model_path, weights_only=False)
        self.config, self.vocab, self.seed = saved["config"], saved["vocab"], saved["seed"]
        self.history, self.best_epoch = saved["history"], saved["best_epoch"]
        self.best_validation_macro_f1 = saved["best_validation_macro_f1"]
        self.stopped_early = saved["stopped_early"]
        cfg = self.config
        self.model = BiLSTMCRF(len(self.vocab), len(LABELS), cfg["embed_dim"], cfg["hidden"],
                               cfg["layers"], cfg["dropout"])
        self.model.load_state_dict(saved["state"])
        return self

    def summary(self) -> dict:
        return {"config": self.config, "seed": self.seed, "vocab_size": len(self.vocab),
                "parameters": sum(p.numel() for p in self.model.parameters()),
                "best_epoch": self.best_epoch, "best_validation_macro_f1": round(self.best_validation_macro_f1, 5),
                "epochs_run": len(self.history), "stopped_early": self.stopped_early}


def history_json(tagger: BiLSTMCRFTagger) -> str:
    return json.dumps({"summary": tagger.summary(), "history": tagger.history}, indent=2) + "\n"
