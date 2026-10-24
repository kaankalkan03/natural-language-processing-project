"""First-order hidden Markov model with Viterbi decoding, written from scratch.

Hidden states are the labels (O, COMMA, PERIOD, QUESTION); each state emits
the word it is attached to. A turn is scored as

    P(y, w) = P(y_1 | START) * prod_t P(y_t | y_{t-1}) * prod_t P(w_t | y_t) * P(END | y_n)

All probabilities are maximum-likelihood counts with add-k smoothing. Training
words seen fewer than `min_count` times are replaced by <unk> (or <num> for
words with digits). The <unk>/<num> emissions are always estimated from rare
training words (seen fewer than max(min_count, 2) times), so words that never
occur in training get data-based emissions even when min_count = 1. Without
this, <unk> would carry only smoothing mass, which is largest in the rarest
states and turned most unseen words into PERIOD (found in the code audit).
With min_count = 1 a singleton counts both for itself and for <unk>, so the
<unk> row is a held-out-style estimate and the emission totals per state are
slightly larger than the number of training words.
Decoding uses the Viterbi algorithm in log space.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from . import LABEL2ID, LABELS

UNK, NUM = "<unk>", "<num>"


def unknown_symbol(word: str) -> str:
    return NUM if any(c.isdigit() for c in word) else UNK


def viterbi(log_start: np.ndarray, log_trans: np.ndarray, log_emit: np.ndarray,
            log_end: np.ndarray) -> list[int]:
    """Best state path. log_emit has shape (T, S): log P(w_t | state) for each position."""
    steps, n_states = log_emit.shape
    if steps == 0:
        return []
    score = log_start + log_emit[0]
    back = np.zeros((steps, n_states), dtype=np.int64)
    for t in range(1, steps):
        candidates = score[:, None] + log_trans  # (previous state, current state)
        back[t] = candidates.argmax(axis=0)
        score = candidates.max(axis=0) + log_emit[t]
    score = score + log_end
    path = [int(score.argmax())]
    for t in range(steps - 1, 0, -1):
        path.append(int(back[t, path[-1]]))
    return path[::-1]


class HMMTagger:
    name = "hmm"

    def __init__(self, k_emit: float = 0.1, k_trans: float = 1.0, min_count: int = 2):
        self.k_emit, self.k_trans, self.min_count = k_emit, k_trans, min_count

    def params(self) -> dict:
        return {"k_emit": self.k_emit, "k_trans": self.k_trans, "min_count": self.min_count}

    def _symbol(self, word: str) -> int:
        index = self.vocab.get(word)
        return index if index is not None else self.vocab[unknown_symbol(word)]

    def fit(self, examples: list[dict]) -> "HMMTagger":
        counts = Counter(w for ex in examples for w in ex["words"])
        kept = sorted(w for w, c in counts.items() if c >= self.min_count)
        self.vocab = {w: i for i, w in enumerate([UNK, NUM, *kept])}
        n_states, n_words = len(LABELS), len(self.vocab)
        rare_below = max(self.min_count, 2)

        start = np.zeros(n_states)
        trans = np.zeros((n_states, n_states))
        end = np.zeros(n_states)
        emit = np.zeros((n_states, n_words))
        for ex in examples:
            states = [LABEL2ID[label] for label in ex["labels"]]
            start[states[0]] += 1
            end[states[-1]] += 1
            for prev, cur in zip(states, states[1:]):
                trans[prev, cur] += 1
            for state, word in zip(states, ex["words"]):
                if counts[word] >= self.min_count:
                    emit[state, self.vocab[word]] += 1
                if counts[word] < rare_below:
                    emit[state, self.vocab[unknown_symbol(word)]] += 1

        k = self.k_trans
        self.log_start = np.log((start + k) / (start.sum() + k * n_states))
        # Each state either moves to another state or ends the turn.
        out = np.concatenate([trans, end[:, None]], axis=1) + k
        out /= out.sum(axis=1, keepdims=True)
        self.log_trans, self.log_end = np.log(out[:, :-1]), np.log(out[:, -1])
        emit += self.k_emit
        self.log_emit = np.log(emit / emit.sum(axis=1, keepdims=True))
        return self

    def predict_one(self, words: list[str]) -> list[str]:
        if not words:
            return []
        symbols = [self._symbol(w) for w in words]
        path = viterbi(self.log_start, self.log_trans, self.log_emit[:, symbols].T, self.log_end)
        return [LABELS[s] for s in path]

    def predict(self, sequences: list[list[str]]) -> list[list[str]]:
        return [self.predict_one(words) for words in sequences]
