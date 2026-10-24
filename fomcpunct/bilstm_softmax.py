"""Ablation: the BiLSTM-CRF without its CRF layer.

Everything is the same as in bilstm_crf.py (vocabulary, embedding, BiLSTM,
dropout, linear layer, Adam, batching, word dropout, gradient clipping, early
stopping on validation macro-F1, and the loss divided by each batch's word
count) except the output layer: each word's label is predicted on its own with
a softmax, trained with word-level cross-entropy and decoded with argmax. There
are no label-transition scores, so the model cannot learn, for example, that
two periods rarely follow each other.

bilstm_crf.py is deliberately left unchanged: the saved BiLSTM-CRF models carry
a hash of that file, and editing it would make them stale. The two methods
copied from BiLSTMCRFTagger (fit and load) differ only in the model they build.
"""

from __future__ import annotations

import copy
from collections import Counter

import torch
from torch import nn
from torch.nn import functional as F

from . import LABEL2ID, LABELS
from .bilstm_crf import PAD, UNK, BiLSTM, BiLSTMCRFTagger, normalize
from .metrics import confusion, macro_f1


class SoftmaxHead(nn.Module):
    """Takes the CRF's place: same nll/decode interface, but labels are chosen word by word."""

    def nll(self, emissions: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Summed cross-entropy over real (unpadded) words."""
        return F.cross_entropy(emissions[mask], labels[mask], reduction="sum")

    @torch.no_grad()
    def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> list[list[int]]:
        best = emissions.argmax(dim=2)
        return [best[b, :n].tolist() for b, n in enumerate(mask.sum(dim=1).tolist())]


class BiLSTMSoftmax(nn.Module):
    def __init__(self, vocab_size: int, n_labels: int, embed_dim: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.drop = nn.Dropout(dropout)
        self.lstm = BiLSTM(embed_dim, hidden, layers, dropout)
        self.out = nn.Linear(2 * hidden, n_labels)
        self.crf = SoftmaxHead()  # named crf so BiLSTMCRFTagger.predict works unchanged; no parameters

    def emissions(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        h = self.lstm(self.drop(self.embed(ids)), lengths)
        return self.out(self.drop(h))


class BiLSTMSoftmaxTagger(BiLSTMCRFTagger):
    name = "bilstm_softmax"

    def _new_model(self) -> BiLSTMSoftmax:
        cfg = self.config
        return BiLSTMSoftmax(len(self.vocab), len(LABELS), cfg["embed_dim"], cfg["hidden"],
                             cfg["layers"], cfg["dropout"])

    def fit(self, train: list[dict], validation: list[dict], log=print) -> "BiLSTMSoftmaxTagger":
        # Same steps as BiLSTMCRFTagger.fit; only the model class differs.
        torch.manual_seed(self.seed)
        torch.set_num_threads(self.threads)
        cfg = self.config
        counts = Counter(normalize(w) for ex in train for w in ex["words"])
        kept = sorted(w for w, c in counts.items() if c >= cfg["min_count"])
        self.vocab = {w: i for i, w in enumerate([PAD, UNK, *kept])}
        self.model = self._new_model()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg["lr"])
        generator = torch.Generator().manual_seed(self.seed)

        data = [(self._ids(ex["words"]), [LABEL2ID[y] for y in ex["labels"]]) for ex in train]
        data.sort(key=lambda pair: len(pair[0]))
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
            # same key as the BiLSTM-CRF history; here the value is word-level cross-entropy
            self.history.append({"epoch": epoch, "train_nll_per_word": round(total_loss / total_words, 5),
                                 "validation_macro_f1": round(score, 5)})
            log(f"    epoch {epoch:2d}: train CE/word {total_loss / total_words:.4f}, validation macro-F1 {score:.4f}")
            if score > best_score:
                best_score, best_state, best_epoch, waited = score, copy.deepcopy(self.model.state_dict()), epoch, 0
            else:
                waited += 1
                if waited >= cfg["patience"]:
                    break
        self.model.load_state_dict(best_state)
        self.best_epoch, self.best_validation_macro_f1 = best_epoch, best_score
        self.stopped_early = waited >= cfg["patience"]
        self.save()
        return self

    def load(self) -> "BiLSTMSoftmaxTagger":
        torch.set_num_threads(self.threads)
        saved = torch.load(self.model_path, weights_only=False)
        self.config, self.vocab, self.seed = saved["config"], saved["vocab"], saved["seed"]
        self.history, self.best_epoch = saved["history"], saved["best_epoch"]
        self.best_validation_macro_f1 = saved["best_validation_macro_f1"]
        self.stopped_early = saved["stopped_early"]
        self.model = self._new_model()
        self.model.load_state_dict(saved["state"])  # strict: a BiLSTM-CRF file (with crf.* weights) fails here
        return self
