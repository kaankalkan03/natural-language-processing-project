"""Tests for metrics, the bootstrap, Viterbi decoding and the baselines."""

import contextlib
import io
import itertools
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fomcpunct import evaluate, run_baselines
from fomcpunct.crf import CRFTagger, features, termination
from fomcpunct.heuristic import HeuristicTagger
from fomcpunct.hmm import HMMTagger, viterbi
from fomcpunct.metrics import bootstrap_ci, confusion, macro_f1, paired_bootstrap, scores

GOLD = [["O", "COMMA", "PERIOD", "O", "QUESTION"]]
PRED = [["O", "PERIOD", "PERIOD", "COMMA", "QUESTION"]]


class MetricTests(unittest.TestCase):
    def test_hand_computed_scores(self):
        s = scores(confusion(GOLD, PRED))
        self.assertAlmostEqual(s["per_class"]["COMMA"]["f1"], 0.0)
        self.assertAlmostEqual(s["per_class"]["PERIOD"]["f1"], 2 / 3)
        self.assertAlmostEqual(s["per_class"]["QUESTION"]["f1"], 1.0)
        self.assertAlmostEqual(s["macro_f1"], (0 + 2 / 3 + 1) / 3)
        self.assertAlmostEqual(s["micro_f1"], 4 / 7)  # TP 2, predicted 4, gold 3
        self.assertAlmostEqual(s["mark_detection_f1"], 6 / 7)
        self.assertAlmostEqual(s["accuracy"], 3 / 5)

    def test_length_mismatch_is_an_error(self):
        with self.assertRaises(ValueError):
            confusion([["O", "O"]], [["O"]])

    def test_bootstrap_of_identical_documents_has_no_spread(self):
        cm = confusion(GOLD, PRED)
        ci = bootstrap_ci(np.stack([cm, cm, cm]), n_resamples=200)
        self.assertAlmostEqual(ci["low"], ci["point"])
        self.assertAlmostEqual(ci["high"], ci["point"])

    def test_bootstrap_interval_contains_point_and_stays_in_range(self):
        good = confusion([["PERIOD", "COMMA", "QUESTION"]], [["PERIOD", "COMMA", "QUESTION"]])
        bad = confusion([["PERIOD", "COMMA", "QUESTION"]], [["O", "O", "O"]])
        ci = bootstrap_ci(np.stack([good, bad, good, bad]), n_resamples=500)
        self.assertLessEqual(ci["low"], ci["point"])
        self.assertGreaterEqual(ci["high"], ci["point"])
        self.assertGreaterEqual(ci["low"], 0.0)
        self.assertLessEqual(ci["high"], 1.0)

    def test_paired_bootstrap_against_itself(self):
        cms = np.stack([confusion(GOLD, PRED)] * 3)
        result = paired_bootstrap(cms, cms, n_resamples=100)
        self.assertEqual(result["difference"], 0.0)
        self.assertEqual(result["share_not_better"], 1.0)

    def test_vectorized_macro_f1_matches_single(self):
        cm = confusion(GOLD, PRED)
        self.assertAlmostEqual(float(macro_f1(np.stack([cm, cm]))[1]), float(macro_f1(cm)))


class ViterbiTests(unittest.TestCase):
    def test_matches_brute_force(self):
        rng = np.random.default_rng(0)
        for steps in (1, 2, 3, 5):
            log_start, log_end = np.log(rng.dirichlet(np.ones(4))), np.log(rng.uniform(0.1, 1, 4))
            log_trans = np.log(rng.dirichlet(np.ones(4), size=4))
            log_emit = np.log(rng.uniform(0.01, 1, size=(steps, 4)))

            def score(path):
                s = log_start[path[0]] + log_emit[0, path[0]] + log_end[path[-1]]
                return s + sum(log_trans[a, b] + log_emit[t + 1, b]
                               for t, (a, b) in enumerate(zip(path, path[1:])))

            best = max(itertools.product(range(4), repeat=steps), key=score)
            self.assertEqual(viterbi(log_start, log_trans, log_emit, log_end), list(best))


class BaselineTests(unittest.TestCase):
    TRAIN = ([{"words": ["we", "think", "so"], "labels": ["O", "O", "PERIOD"]}] * 20
             + [{"words": ["we", "rarely", "think", "so"], "labels": ["O", "O", "O", "PERIOD"]},
                {"words": ["we", "seldom", "think", "so"], "labels": ["O", "O", "O", "PERIOD"]}])

    def test_hmm_learns_a_simple_pattern(self):
        hmm = HMMTagger(k_emit=0.01, min_count=1).fit(self.TRAIN)
        self.assertEqual(hmm.predict([["we", "think", "so"]]), [["O", "O", "PERIOD"]])
        self.assertEqual(hmm.predict([[]]), [[]])

    def test_hmm_unseen_word_is_scored_like_rare_training_words(self):
        # Rare training words ("rarely", "seldom") are O, so an unseen word in the
        # same place must not become a sentence end (audit finding for min_count=1).
        for min_count in (1, 2):
            hmm = HMMTagger(k_emit=0.01, min_count=min_count).fit(self.TRAIN)
            self.assertEqual(hmm.predict_one(["we", "zzz", "think", "so"]), ["O", "O", "O", "PERIOD"])

    def test_heuristic_rules(self):
        words = "is that right so we think at at the level you know".split()
        labels = HeuristicTagger().predict_one(words)
        self.assertEqual(labels[2], "QUESTION")   # "is ..." sentence, ends before "so"
        self.assertEqual(labels[6], "COMMA")      # repeated word: "at, at"
        self.assertEqual(labels[-1], "PERIOD")    # end of turn
        self.assertEqual(labels[-2], "O")         # "you" before "know"

    def test_heuristic_discourse_comma_needs_the_bigram(self):
        tagger = HeuristicTagger()
        self.assertEqual(tagger.predict_one("you know it is fine".split())[1], "COMMA")
        self.assertEqual(tagger.predict_one("we know it is fine".split())[1], "O")


class CRFTests(unittest.TestCase):
    def test_window_and_repeat_features(self):
        words = "we are at at the level of the of the economy".split()
        self.assertIn("w+3=at", features(words, 0))
        self.assertIn("w+3=of", features(words, 3))
        self.assertIn("w-1|w0=we|are", features(words, 1))
        self.assertIn("repeat_next", features(words, 2))       # first "at" of "at at"
        self.assertNotIn("repeat_next", features(words, 3))
        self.assertIn("repeat_bigram", features(words, 7))      # first "the" of "of the of the"
        self.assertIn("last_in_turn", features(words, len(words) - 1))
        self.assertIn("w-2=<s>", features(words, 0))
        self.assertIn("w+1=</s>", features(words, len(words) - 1))

    def test_fit_predict_and_training_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            tagger = CRFTagger(Path(tmp) / "m.crfsuite", c1=0.0, c2=0.01, max_iterations=50)
            tagger.fit(BaselineTests.TRAIN)
            self.assertEqual(tagger.predict([["we", "think", "so"]]), [["O", "O", "PERIOD"]])
            self.assertLessEqual(tagger.training_log["iterations"], 50)
            capped = CRFTagger(Path(tmp) / "c.crfsuite", c1=0.0, c2=0.01, max_iterations=2).fit(BaselineTests.TRAIN)
            self.assertFalse(capped.training_log["converged"])
            done = CRFTagger(Path(tmp) / "d.crfsuite", c1=0.1, c2=0.01, max_iterations=5000).fit(BaselineTests.TRAIN)
            self.assertTrue(done.training_log["converged"])

    def test_termination_messages(self):
        self.assertEqual(termination("x\nL-BFGS terminated with the stopping criteria\n"),
                         (True, "L-BFGS terminated with the stopping criteria"))
        self.assertTrue(termination("L-BFGS resulted in convergence\n")[0])
        self.assertFalse(termination("L-BFGS terminated with the maximum number of iterations\n")[0])
        self.assertFalse(termination("L-BFGS terminated with error code (-1001)\n")[0])

    def test_model_path_with_non_ascii_characters(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "Masaüstü" / "m.crfsuite"
            tagger = CRFTagger(path, c1=0.0, c2=0.01, max_iterations=30).fit(BaselineTests.TRAIN)
            self.assertTrue(path.exists())
            again = CRFTagger(path).load()
            self.assertEqual(again.predict([["we", "think", "so"]]), tagger.predict([["we", "think", "so"]]))


class RunBaselinesTests(unittest.TestCase):
    """Tiny end-to-end run: tune the CRF, reuse the saved model, retrain when train data change."""

    def test_tune_reuse_and_retrain(self):
        rows = [dict(ex, id=f"d1-t{i:03d}", doc="d1", role="chair", marks=[""] * len(ex["words"]))
                for i, ex in enumerate(BaselineTests.TRAIN)]
        old_cwd, old_grid = os.getcwd(), run_baselines.GRIDS["crf"]
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.chdir(tmp)
                Path("data/processed").mkdir(parents=True)
                for split in ("train", "validation"):
                    Path(f"data/processed/{split}.jsonl").write_text(
                        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
                run_baselines.GRIDS["crf"] = {"c1": [0.0, 0.1], "c2": [0.01], "max_iterations": [30]}
                validation = run_baselines.load_split("validation")
                with contextlib.redirect_stdout(io.StringIO()):
                    params = run_baselines.tune("crf", rows, validation, jobs=2)
                card = json.loads(Path("models/crf.json").read_text(encoding="utf-8"))
                self.assertEqual(card["params"], params)
                self.assertEqual(card["model_sha256"], run_baselines.sha256(Path("models/crf.crfsuite")))
                self.assertEqual(json.loads(Path("results/params/crf.json").read_text()), params)
                self.assertEqual(len(json.loads(Path("results/tuning/crf.json").read_text())["trials"]), 2)
                self.assertEqual(sorted(p.name for p in Path("models").iterdir()), ["crf.crfsuite", "crf.json"])

                before = Path("models/crf.crfsuite").stat().st_mtime_ns
                run_baselines.ready_tagger("crf", params, rows)  # reuse: file untouched
                self.assertEqual(Path("models/crf.crfsuite").stat().st_mtime_ns, before)

                Path("data/processed/train.jsonl").write_text(
                    "".join(json.dumps(r) + "\n" for r in rows[:-1]), encoding="utf-8")
                with contextlib.redirect_stdout(io.StringIO()) as out:
                    run_baselines.ready_tagger("crf", params, rows[:-1])
                self.assertIn("train_sha256", out.getvalue())
            finally:
                run_baselines.GRIDS["crf"] = old_grid
                os.chdir(old_cwd)


class EvaluateTests(unittest.TestCase):
    EXAMPLES = [{"id": "d1-t000", "doc": "d1", "role": "chair", "words": ["a", "b", "c"],
                 "labels": ["COMMA", "O", "PERIOD"], "marks": [",", "", "."]},
                {"id": "d2-t000", "doc": "d2", "role": "moderator", "words": ["x", "y"],
                 "labels": ["O", "PERIOD"], "marks": ["", "."]}]

    def test_within_turn_drops_the_last_word(self):
        preds = [["COMMA", "O", "PERIOD"], ["O", "PERIOD"]]
        cms = evaluate.doc_confusions(self.EXAMPLES, preds, ["d1", "d2"], within_turn=True)
        self.assertEqual(int(cms.sum()), 3)
        self.assertEqual(int(cms[0].sum()), 2)

    def test_slice_macro_uses_classes_with_enough_support(self):
        gold = [["O", "PERIOD"] * 30 + ["COMMA"] * 5]
        pred = [["O", "PERIOD"] * 30 + ["O"] * 5]
        s = evaluate.slice_scores(confusion(gold, pred))
        self.assertEqual(s["classes_used"], ["PERIOD"])  # 5 commas are too few to average in
        self.assertAlmostEqual(s["macro_f1"], 1.0)
        self.assertEqual(s["support"]["COMMA"], 5)
        self.assertEqual(s["predicted"]["COMMA"], 0)
        self.assertIsNone(evaluate.slice_scores(confusion([["O", "PERIOD"]], [["O", "PERIOD"]]))["macro_f1"])

    def test_seed_groups(self):
        report = {"models": {name: {"all_words": {"macro_f1": f}, "within_turn": {"macro_f1": f - 0.1}}
                             for name, f in (("crf", 0.6), ("bilstm_crf_s0", 0.6), ("bilstm_crf_s1", 0.7),
                                             ("bilstm_crf_s2", 0.8))}}
        groups = evaluate.seed_groups(report)
        self.assertEqual(list(groups), ["bilstm_crf"])
        self.assertAlmostEqual(groups["bilstm_crf"]["all_words"]["mean"], 0.7)
        self.assertAlmostEqual(groups["bilstm_crf"]["all_words"]["sd"], 0.1)
        self.assertAlmostEqual(groups["bilstm_crf"]["within_turn"]["min"], 0.5)

    def test_duplicate_prediction_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            old, evaluate.RESULTS = evaluate.RESULTS, Path(tmp)
            try:
                (Path(tmp) / "predictions").mkdir()
                rows = [{"id": "d1-t000", "labels": ["O", "O", "O"]}] * 2 + [{"id": "d2-t000", "labels": ["O", "O"]}]
                (Path(tmp) / "predictions" / "m.validation.jsonl").write_text(
                    "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
                with self.assertRaises(ValueError):
                    evaluate.load_predictions("m", "validation", self.EXAMPLES)
            finally:
                evaluate.RESULTS = old


if __name__ == "__main__":
    unittest.main()
