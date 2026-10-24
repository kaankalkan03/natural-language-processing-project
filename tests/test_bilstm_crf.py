"""Tests for the from-scratch CRF layer and the BiLSTM-CRF tagger."""

import contextlib
import io
import itertools
import json
import os
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from fomcpunct import train_bilstm
from fomcpunct.bilstm_crf import CRF, BiLSTM, BiLSTMCRFTagger, normalize, reverse_within_length
from fomcpunct.metrics import confusion, macro_f1

TOY = ([{"words": ["we", "think", "so"], "labels": ["O", "O", "PERIOD"]}] * 12
       + [{"words": ["is", "that", "right"], "labels": ["O", "O", "QUESTION"]}] * 12
       + [{"words": ["well", "we", "know"], "labels": ["COMMA", "O", "PERIOD"]}] * 12)


def brute_force(crf: CRF, emissions: torch.Tensor, length: int):
    """All label paths of one sequence: (log Z, best path, score function)."""
    K = emissions.size(-1)

    def score(path):
        s = crf.start[path[0]] + emissions[0, path[0]] + crf.end[path[-1]]
        for t in range(1, length):
            s = s + crf.trans[path[t - 1], path[t]] + emissions[t, path[t]]
        return s

    paths = list(itertools.product(range(K), repeat=length))
    scores = torch.stack([score(p) for p in paths])
    return torch.logsumexp(scores, dim=0), list(paths[int(scores.argmax())]), score


class CRFTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.crf = CRF(4)
        with torch.no_grad():  # larger scores than the default init make the test sharper
            for p in self.crf.parameters():
                p.normal_(0, 1)
        self.lengths = [4, 1, 3, 2]
        self.emissions = torch.randn(len(self.lengths), 4, 4)
        self.mask = torch.arange(4).unsqueeze(0) < torch.tensor(self.lengths).unsqueeze(1)
        self.labels = torch.randint(0, 4, (len(self.lengths), 4))

    @torch.no_grad()
    def test_log_partition_matches_brute_force_with_padding(self):
        log_z = self.crf.log_partition(self.emissions, self.mask)
        for b, n in enumerate(self.lengths):
            expected, _, _ = brute_force(self.crf, self.emissions[b], n)
            self.assertAlmostEqual(float(log_z[b]), float(expected), places=4)

    @torch.no_grad()
    def test_path_score_matches_brute_force(self):
        scores = self.crf.path_score(self.emissions, self.labels, self.mask)
        for b, n in enumerate(self.lengths):
            _, _, score = brute_force(self.crf, self.emissions[b], n)
            self.assertAlmostEqual(float(scores[b]), float(score(self.labels[b, :n].tolist())), places=4)

    @torch.no_grad()
    def test_viterbi_matches_brute_force(self):
        paths = self.crf.decode(self.emissions, self.mask)
        for b, n in enumerate(self.lengths):
            _, best, _ = brute_force(self.crf, self.emissions[b], n)
            self.assertEqual(paths[b], best)

    def test_nll_is_positive_and_differentiable(self):
        loss = self.crf.nll(self.emissions.requires_grad_(), self.labels, self.mask)
        self.assertGreater(loss.item(), 0.0)
        loss.backward()
        self.assertTrue(all(p.grad is not None for p in self.crf.parameters()))

    @torch.no_grad()
    def test_batched_viterbi_on_many_padded_sequences(self):
        # padded steps need identity back-pointers; a wrong pointer shows up only in shorter sequences
        torch.manual_seed(0)
        crf = CRF(4)
        for p in crf.parameters():
            p.normal_(0, 1)
        lengths = [1, 6] + torch.randint(1, 7, (30,)).tolist()
        emissions = torch.randn(len(lengths), 6, 4)
        mask = torch.arange(6).unsqueeze(0) < torch.tensor(lengths).unsqueeze(1)
        paths = crf.decode(emissions, mask)
        for b, n in enumerate(lengths):
            self.assertEqual(paths[b], brute_force(crf, emissions[b], n)[1], b)

    def test_gradients_match_finite_differences(self):
        torch.manual_seed(2)
        crf = CRF(3).double()
        emissions = torch.randn(3, 4, 3, dtype=torch.float64, requires_grad=True)
        mask = torch.arange(4).unsqueeze(0) < torch.tensor([4, 1, 3]).unsqueeze(1)
        labels = torch.randint(0, 3, (3, 4))

        class NLL(nn.Module):
            def __init__(self):
                super().__init__()
                self.crf = crf

            def forward(self, em):
                return self.crf.nll(em, labels, mask)

        def loss(em, start, trans, end):  # parameters as inputs, so gradcheck checks their gradients too
            return torch.func.functional_call(NLL(), {"crf.start": start, "crf.trans": trans, "crf.end": end}, (em,))

        params = [p.detach().clone().requires_grad_() for p in (crf.start, crf.trans, crf.end)]
        self.assertTrue(torch.autograd.gradcheck(loss, (emissions, *params)))

    def test_padding_values_do_not_matter(self):
        noisy = self.emissions.clone()
        noisy[~self.mask] = 1000.0
        self.assertTrue(torch.allclose(self.crf.log_partition(noisy, self.mask),
                                       self.crf.log_partition(self.emissions, self.mask)))
        self.assertEqual(self.crf.decode(noisy, self.mask), self.crf.decode(self.emissions, self.mask))


class BiLSTMTests(unittest.TestCase):
    @torch.no_grad()
    def test_matches_packed_bidirectional_lstm(self):
        torch.manual_seed(1)
        for layers in (1, 2):
            ours = BiLSTM(5, 3, layers, dropout=0.0).eval()
            ref = nn.LSTM(5, 3, num_layers=layers, bidirectional=True, batch_first=True).eval()
            for i in range(layers):  # copy the reference weights into our two LSTMs
                for name in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
                    getattr(ours.forward_lstms[i], f"{name}_l0").copy_(getattr(ref, f"{name}_l{i}"))
                    getattr(ours.backward_lstms[i], f"{name}_l0").copy_(getattr(ref, f"{name}_l{i}_reverse"))
            lengths = torch.tensor([6, 2, 4])
            x = torch.randn(3, 6, 5)
            packed, _ = ref(pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False))
            expected, _ = pad_packed_sequence(packed, batch_first=True, total_length=6)
            got = ours(x, lengths)
            for b, n in enumerate(lengths.tolist()):
                self.assertTrue(torch.allclose(got[b, :n], expected[b, :n], atol=1e-6), (layers, b))

    def test_reverse_within_length(self):
        x = torch.arange(8.0).view(2, 4, 1)
        out = reverse_within_length(x, torch.tensor([4, 2]))
        self.assertEqual(out.squeeze(2).tolist(), [[3, 2, 1, 0], [5, 4, 6, 7]])


class TaggerTests(unittest.TestCase):
    def fit(self, tmp, seed=0):
        return BiLSTMCRFTagger(Path(tmp) / f"m{seed}.pt", embed_dim=16, hidden=16, max_epochs=40,
                               patience=40, min_count=1, word_dropout=0.0, lr=0.02, seed=seed,
                               threads=1).fit(TOY, TOY, log=lambda *_: None)

    def test_learns_toy_patterns_and_round_trips(self):
        with tempfile.TemporaryDirectory() as tmp:
            tagger = self.fit(tmp)
            words = [["we", "think", "so"], ["is", "that", "right"], ["well", "we", "know"], []]
            expected = [["O", "O", "PERIOD"], ["O", "O", "QUESTION"], ["COMMA", "O", "PERIOD"], []]
            self.assertEqual(tagger.predict(words), expected)
            loaded = BiLSTMCRFTagger(tagger.model_path, threads=1).load()
            self.assertEqual(loaded.predict(words), expected)
            self.assertEqual(loaded.best_epoch, tagger.best_epoch)

    def test_same_seed_same_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = self.fit(tmp + "/a"), self.fit(tmp + "/b")
            for key, value in a.model.state_dict().items():
                self.assertTrue(torch.equal(value, b.model.state_dict()[key]), key)
            self.assertEqual(a.history, b.history)

    def test_vocabulary_digits_and_unknown_words(self):
        self.assertEqual(normalize("2.5"), "0.0")
        train = TOY + [{"words": ["about", "116,000", "jobs"], "labels": ["O", "O", "PERIOD"]}]
        validation = [{"words": ["only", "in", "validation"], "labels": ["O", "O", "PERIOD"]}]
        with tempfile.TemporaryDirectory() as tmp:
            tagger = BiLSTMCRFTagger(Path(tmp) / "m.pt", embed_dim=8, hidden=8, max_epochs=2, min_count=1,
                                     threads=1).fit(train, validation, log=lambda *_: None)
            self.assertNotIn("validation", tagger.vocab)          # vocabulary from train only
            self.assertEqual(tagger._ids(["999,999"]), [tagger.vocab["000,000"]])
            self.assertEqual(tagger._ids(["zzz"]), [tagger.vocab["<unk>"]])
            self.assertEqual(len(tagger.predict([["zzz", "qqq", "116,000"]])[0]), 3)

    def test_predict_keeps_input_order_with_mixed_lengths(self):
        with tempfile.TemporaryDirectory() as tmp:
            tagger = self.fit(tmp)
            words = [["we", "think", "so", "is", "that", "right"], [], ["well"], ["is", "that", "right"], [],
                     ["we", "know"]] * 3
            together = tagger.predict(words, batch_size=4)
            self.assertEqual(together, [tagger.predict([w])[0] for w in words])
            self.assertEqual(together, tagger.predict(words, batch_size=4))  # eval mode: no dropout noise

    def test_best_epoch_is_restored(self):
        with tempfile.TemporaryDirectory() as tmp:
            tagger = BiLSTMCRFTagger(Path(tmp) / "m.pt", embed_dim=16, hidden=16, max_epochs=8, patience=2,
                                     min_count=1, word_dropout=0.0, lr=0.02, threads=1)
            tagger.fit(TOY, TOY, log=lambda *_: None)
            scores = [h["validation_macro_f1"] for h in tagger.history]
            self.assertEqual(tagger.best_epoch, scores.index(max(scores)) + 1)
            gold = [ex["labels"] for ex in TOY]
            restored = float(macro_f1(confusion(gold, tagger.predict([ex["words"] for ex in TOY]))))
            self.assertAlmostEqual(restored, max(scores), places=4)
            self.assertEqual(tagger.stopped_early, len(scores) - tagger.best_epoch >= 2)


class TrainPipelineTests(unittest.TestCase):
    """Tiny end-to-end run of train_bilstm: tune, reuse, refuse stale models and stale tuning."""

    FIXED = {"embed_dim": 8, "lr": 0.02, "batch_size": 8, "max_epochs": 2, "patience": 2,
             "min_count": 1, "word_dropout": 0.0, "clip": 5.0}
    GRID = {"hidden": [8], "layers": [1], "dropout": [0.0, 0.1]}

    def test_tune_reuse_and_stale_checks(self):
        rows = [dict(ex, id=f"d1-t{i:03d}", doc="d1", role="chair", marks=[""] * 3) for i, ex in enumerate(TOY)]
        old_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.chdir(tmp)
                Path("data/processed").mkdir(parents=True)
                for split in ("train", "validation"):
                    Path(f"data/processed/{split}.jsonl").write_text(
                        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
                with contextlib.redirect_stdout(io.StringIO()):
                    params = train_bilstm.tune(jobs=1, threads=1, fixed=self.FIXED, grid=self.GRID)
                self.assertIn(params, [{"hidden": 8, "layers": 1, "dropout": d} for d in (0.0, 0.1)])
                self.assertEqual(train_bilstm.card_matches(params, 0, self.FIXED, 1), "")
                self.assertEqual(train_bilstm.check_tuning(self.FIXED, 1), params)
                self.assertEqual(sorted(p.name for p in Path("models").iterdir()),
                                 ["bilstm_crf_s0.json", "bilstm_crf_s0.pt"])  # trial folder removed
                with contextlib.redirect_stdout(io.StringIO()) as out:
                    train_bilstm.train_seeds(params, [0], jobs=1, threads=1, fixed=self.FIXED)
                self.assertIn("reusing", out.getvalue())
                changed = dict(self.FIXED, lr=0.01)
                self.assertIn("fixed", train_bilstm.card_matches(params, 0, changed, 1))
                self.assertIn("threads", train_bilstm.card_matches(params, 0, self.FIXED, 2))
                with self.assertRaises(SystemExit):
                    train_bilstm.check_tuning(changed, 1)
            finally:
                os.chdir(old_cwd)


if __name__ == "__main__":
    unittest.main()
