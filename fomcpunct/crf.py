"""Linear-chain CRF with hand-written window features (python-crfsuite, L-BFGS).

Unlike the HMM, a CRF can condition each label on any input features, so it
can look at the next words; the punctuation after a word depends strongly on
what follows ("... percent. So ...", "... which ..."). Features of word i, all
taken from the lower-cased input words only (never from the gold marks):

- the word itself, its last three letters and its shape (number, apostrophe,
  hyphen, dot, alphabetic);
- the words at offsets -2, -1, +1, +2, +3 (with <s>/</s> padding) and the
  bigrams (w-1, w0), (w0, w+1), (w+1, w+2);
- repeat flags for spoken repairs: the next word equals this word ("at at"),
  or this word ends a bigram that is repeated right after it (the first "the"
  in "of the of the");
- first/last word of the turn (the turn boundary is given input).
"""

from __future__ import annotations

import hashlib
import inspect
import shutil
import tempfile
from pathlib import Path

import pycrfsuite

WINDOW = (-2, -1, 1, 2, 3)


def shape(word: str) -> str:
    if any(c.isdigit() for c in word):
        return "num"
    if "'" in word:
        return "apos"
    if "-" in word:
        return "hyph"
    if "." in word:
        return "dot"
    return "alpha" if word.isalpha() else "other"


def features(words: list[str], i: int) -> list[str]:
    n, w = len(words), words[i]
    get = lambda j: words[j] if 0 <= j < n else ("<s>" if j < 0 else "</s>")  # noqa: E731
    feats = ["bias", f"w0={w}", f"suf3={w[-3:]}", f"shape={shape(w)}"]
    feats += [f"w{off:+d}={get(i + off)}" for off in WINDOW]
    feats += [f"w-1|w0={get(i - 1)}|{w}", f"w0|w+1={w}|{get(i + 1)}", f"w+1|w+2={get(i + 1)}|{get(i + 2)}"]
    if get(i + 1) == w:
        feats.append("repeat_next")
    if i >= 1 and get(i - 1) == get(i + 1) and w == get(i + 2):
        feats.append("repeat_bigram")
    if i == n - 1:
        feats.append("last_in_turn")
    if i == 0:
        feats.append("first_in_turn")
    # python-crfsuite takes plain strings as attribute names (weight 1.0); no escaping needed.
    return feats


def sequence_features(words: list[str]) -> list[list[str]]:
    return [features(words, i) for i in range(len(words))]


def feature_code_hash() -> str:
    """Fingerprint of the feature and training code, stored with saved models."""
    source = (inspect.getsource(shape) + inspect.getsource(features) + repr(WINDOW)
              + inspect.getsource(CRFTagger.fit))  # trainer options live in fit()
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]


def termination(log_text: str) -> tuple[bool, str]:
    """crfsuite's L-BFGS end message and whether it means convergence.

    Converged: "terminated with the stopping criteria" (delta test) or
    "resulted in convergence" (gradient-norm test). Not converged: "terminated
    with the maximum number of iterations" or an error code.
    """
    line = next((x.strip() for x in log_text.splitlines()
                 if x.startswith("L-BFGS") and ("terminated" in x or "resulted" in x)), "")
    return ("stopping criteria" in line or "resulted in convergence" in line), line


class CRFTagger:
    name = "crf"

    def __init__(self, model_path: str | Path, c1: float = 0.1, c2: float = 0.01,
                 max_iterations: int = 1000):
        self.c1, self.c2, self.max_iterations = c1, c2, max_iterations
        self.model_path = Path(model_path)

    def params(self) -> dict:
        return {"c1": self.c1, "c2": self.c2, "max_iterations": self.max_iterations}

    def fit(self, examples: list[dict]) -> "CRFTagger":
        trainer = pycrfsuite.Trainer(verbose=False)
        for ex in examples:
            trainer.append(sequence_features(ex["words"]), ex["labels"])
        trainer.set_params({"c1": self.c1, "c2": self.c2, "max_iterations": self.max_iterations,
                            "feature.possible_transitions": True})
        # crfsuite cannot write or open paths with non-ASCII characters (this repo
        # sits under "Masaüstü"), so train into a temporary folder and copy.
        with tempfile.TemporaryDirectory() as tmp:
            tmp_model = Path(tmp) / "model.crfsuite"
            trainer.train(str(tmp_model))
            if not tmp_model.exists():
                raise RuntimeError("crfsuite did not write a model file")
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(tmp_model, self.model_path)
        last = trainer.logparser.last_iteration
        converged, message = termination("".join(trainer.logparser.log))
        self.training_log = {
            "iterations": last["num"],
            "loss": last.get("loss"),
            "active_features": last.get("active_features"),
            "converged": converged,
            "termination": message,
        }
        return self.load()

    def load(self) -> "CRFTagger":
        self._model_bytes = self.model_path.read_bytes()  # the tagger keeps a pointer to it
        self.tagger = pycrfsuite.Tagger()
        self.tagger.open_inmemory(self._model_bytes)
        return self

    def predict(self, sequences: list[list[str]]) -> list[list[str]]:
        return [list(self.tagger.tag(sequence_features(words))) for words in sequences]
