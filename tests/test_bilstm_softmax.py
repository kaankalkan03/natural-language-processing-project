"""Tests for the ablation: the BiLSTM tagger without the CRF layer."""

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch.nn import functional as F

from fomcpunct import bilstm_crf, train_bilstm, train_bilstm_softmax
from fomcpunct.bilstm_crf import BiLSTMCRFTagger
from fomcpunct.bilstm_softmax import BiLSTMSoftmax, BiLSTMSoftmaxTagger, SoftmaxHead

TOY = ([{"words": ["we", "think", "so"], "labels": ["O", "O", "PERIOD"]}] * 12
       + [{"words": ["is", "that", "right"], "labels": ["O", "O", "QUESTION"]}] * 12
       + [{"words": ["well", "we", "know"], "labels": ["COMMA", "O", "PERIOD"]}] * 12)


class SoftmaxHeadTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.lengths = [4, 1, 3]
        self.emissions = torch.randn(3, 4, 4)
        self.mask = torch.arange(4).unsqueeze(0) < torch.tensor(self.lengths).unsqueeze(1)
        self.labels = torch.randint(0, 4, (3, 4))

    def test_nll_is_summed_cross_entropy_over_real_words(self):
        expected = sum(F.cross_entropy(self.emissions[b, :n], self.labels[b, :n], reduction="sum")
                       for b, n in enumerate(self.lengths))
        got = SoftmaxHead().nll(self.emissions, self.labels, self.mask)
        self.assertAlmostEqual(float(got), float(expected), places=5)

    def test_padding_values_do_not_matter(self):
        noisy = self.emissions.clone()
        noisy[~self.mask] = 100.0
        labels = self.labels.clone()
        labels[~self.mask] = 3
        head = SoftmaxHead()
        self.assertAlmostEqual(float(head.nll(noisy, labels, self.mask)),
                               float(head.nll(self.emissions, self.labels, self.mask)), places=5)
        self.assertEqual(head.decode(noisy, self.mask), head.decode(self.emissions, self.mask))

    def test_decode_is_argmax_cut_to_length(self):
        paths = SoftmaxHead().decode(self.emissions, self.mask)
        self.assertEqual([len(p) for p in paths], self.lengths)
        for b, n in enumerate(self.lengths):
            self.assertEqual(paths[b], self.emissions[b, :n].argmax(dim=1).tolist())


class SoftmaxTaggerTests(unittest.TestCase):
    def fit(self, path, seed=0):
        return BiLSTMSoftmaxTagger(path, embed_dim=16, hidden=16, max_epochs=40, patience=40, min_count=1,
                                   word_dropout=0.0, lr=0.02, seed=seed, threads=1).fit(TOY, TOY, log=lambda *_: None)

    def test_learns_toy_patterns_round_trips_and_has_no_crf_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            tagger = self.fit(Path(tmp) / "m.pt")
            words = [["we", "think", "so"], ["is", "that", "right"], ["well", "we", "know"], []]
            expected = [["O", "O", "PERIOD"], ["O", "O", "QUESTION"], ["COMMA", "O", "PERIOD"], []]
            self.assertEqual(tagger.predict(words), expected)
            self.assertFalse([k for k in tagger.model.state_dict() if k.startswith("crf.")])
            loaded = BiLSTMSoftmaxTagger(tagger.model_path, threads=1).load()
            self.assertEqual(loaded.predict(words), expected)
            self.assertEqual(loaded.best_epoch, tagger.best_epoch)

    def test_same_seed_same_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = self.fit(Path(tmp) / "a.pt"), self.fit(Path(tmp) / "b.pt")
            for key, value in a.model.state_dict().items():
                self.assertTrue(torch.equal(value, b.model.state_dict()[key]), key)
            self.assertEqual(a.history, b.history)

    def test_fit_is_the_crf_taggers_fit_with_another_model(self):
        """The copied fit/load must stay identical to BiLSTMCRFTagger's (only the model class may differ)."""
        args = dict(embed_dim=16, hidden=16, max_epochs=6, patience=6, min_count=1, word_dropout=0.1, lr=0.02,
                    seed=3, threads=1)
        with tempfile.TemporaryDirectory() as tmp:
            ours = BiLSTMSoftmaxTagger(Path(tmp) / "a.pt", **args).fit(TOY, TOY, log=lambda *_: None)
            with mock.patch.object(bilstm_crf, "BiLSTMCRF", BiLSTMSoftmax):
                theirs = BiLSTMCRFTagger(Path(tmp) / "b.pt", **args).fit(TOY, TOY, log=lambda *_: None)
                theirs_loaded = BiLSTMCRFTagger(theirs.model_path, threads=1).load()
            self.assertEqual(ours.vocab, theirs.vocab)
            self.assertEqual(ours.history, theirs.history)
            self.assertEqual((ours.best_epoch, ours.stopped_early), (theirs.best_epoch, theirs.stopped_early))
            for key, value in ours.model.state_dict().items():
                self.assertTrue(torch.equal(value, theirs.model.state_dict()[key]), key)
            ours_loaded = BiLSTMSoftmaxTagger(ours.model_path, threads=1).load()
            for key, value in ours_loaded.model.state_dict().items():
                self.assertTrue(torch.equal(value, theirs_loaded.model.state_dict()[key]), key)

    def test_crf_and_softmax_files_cannot_be_swapped(self):
        with tempfile.TemporaryDirectory() as tmp:
            crf = BiLSTMCRFTagger(Path(tmp) / "crf.pt", embed_dim=8, hidden=8, max_epochs=1, min_count=1,
                                  threads=1).fit(TOY, TOY, log=lambda *_: None)
            with self.assertRaises(RuntimeError):
                BiLSTMSoftmaxTagger(crf.model_path, threads=1).load()
            soft = BiLSTMSoftmaxTagger(Path(tmp) / "soft.pt", embed_dim=8, hidden=8, max_epochs=1, min_count=1,
                                       threads=1).fit(TOY, TOY, log=lambda *_: None)
            with self.assertRaises(RuntimeError):
                BiLSTMCRFTagger(soft.model_path, threads=1).load()


class SoftmaxPipelineTests(unittest.TestCase):
    """Tiny end-to-end run: BiLSTM-CRF tuning record, then the ablation seeds, reuse and stale checks."""

    FIXED = {"embed_dim": 8, "lr": 0.02, "batch_size": 8, "max_epochs": 2, "patience": 2,
             "min_count": 1, "word_dropout": 0.0, "clip": 5.0}
    GRID = {"hidden": [8], "layers": [1], "dropout": [0.0]}

    def test_train_reuse_predict_and_stale_checks(self):
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
                    train_bilstm.tune(jobs=1, threads=1, fixed=self.FIXED, grid=self.GRID)
                params = train_bilstm.check_tuning(self.FIXED, 1)
                with contextlib.redirect_stdout(io.StringIO()):
                    train_bilstm_softmax.train_seeds(params, [0, 1], jobs=1, threads=1, fixed=self.FIXED)
                    train_bilstm_softmax.predict_seeds([0, 1], "validation")
                for seed in (0, 1):
                    self.assertEqual(train_bilstm_softmax.card_matches(params, seed, self.FIXED, 1), "")
                    card = json.loads(Path(f"models/bilstm_softmax_s{seed}.json").read_text(encoding="utf-8"))
                    self.assertIn("bilstm_crf.json", card["params_source"])
                    preds = Path(f"results/predictions/bilstm_softmax_s{seed}.validation.jsonl").read_text(
                        encoding="utf-8").splitlines()
                    self.assertEqual(len(preds), len(rows))
                    self.assertTrue(Path(f"results/training/bilstm_softmax_s{seed}.json").exists())
                # the BiLSTM-CRF seed-0 model from tuning is untouched
                self.assertEqual(train_bilstm.card_matches(params, 0, self.FIXED, 1), "")
                with contextlib.redirect_stdout(io.StringIO()) as out:
                    train_bilstm_softmax.train_seeds(params, [0], jobs=1, threads=1, fixed=self.FIXED)
                self.assertIn("reusing", out.getvalue())
                self.assertIn("fixed", train_bilstm_softmax.card_matches(params, 0, dict(self.FIXED, lr=0.01), 1))
                self.assertIn("params", train_bilstm_softmax.card_matches(dict(params, hidden=16), 0, self.FIXED, 1))
                self.assertEqual(train_bilstm_softmax.card_matches(params, 5, self.FIXED, 1), "no saved model")
                # the code hash is part of the card: other model code -> stale
                with mock.patch.object(train_bilstm_softmax, "MODEL_CODE", ("bilstm_softmax.py", "metrics.py")):
                    self.assertIn("code", train_bilstm_softmax.card_matches(params, 0, self.FIXED, 1))
                # --split test never trains: a seed without a saved model stops the run
                argv = ["train_bilstm_softmax", "--seeds", "0", "7", "--split", "test", "--threads", "1"]
                with mock.patch.object(train_bilstm_softmax, "FIXED", self.FIXED), mock.patch("sys.argv", argv):
                    with self.assertRaises(SystemExit) as stop:
                        train_bilstm_softmax.main()
                self.assertIn("not retraining", str(stop.exception))
                self.assertFalse(Path("models/bilstm_softmax_s7.pt").exists())
            finally:
                os.chdir(old_cwd)


if __name__ == "__main__":
    unittest.main()
