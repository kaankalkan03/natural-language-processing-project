"""Tests for the transformer windows, word/subword alignment and the training run.

Tests that need a Hugging Face tokenizer or model use only files already in
the local cache and are skipped otherwise.
"""

import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fomcpunct import LABELS
from fomcpunct.transformer import Config, coverage_problems, encode_examples, make_windows, owners

try:
    from transformers import AutoTokenizer
except ImportError:  # transformers is needed only for the Colab runs
    AutoTokenizer = None

DATA = Path(__file__).resolve().parents[1] / "data" / "processed"
TINY_BERT = "hf-internal-testing/tiny-random-BertForTokenClassification"
TINY_DEBERTA = "hf-internal-testing/tiny-random-DebertaV2ForTokenClassification"


def cached_tokenizer(name):
    if AutoTokenizer is None:
        raise unittest.SkipTest("transformers is not installed")
    try:
        return AutoTokenizer.from_pretrained(name, local_files_only=True)
    except Exception as error:  # not in the local cache
        raise unittest.SkipTest(f"{name} not cached: {error}")


class WindowTests(unittest.TestCase):
    def test_windows_overlap_and_cover_the_turn(self):
        self.assertEqual(make_windows([1] * 20, budget=10, overlap=4), [(0, 10), (6, 16), (12, 20)])
        self.assertEqual(make_windows([1] * 5, budget=10, overlap=4), [(0, 5)])
        self.assertEqual(make_windows([], budget=10, overlap=4), [])

    def test_small_windows_still_move_forward_by_half(self):
        # 4 words fit in a window but the overlap is 64 words: the step must not shrink to 1 word
        windows = make_windows([5] * 100, budget=20, overlap=64)
        self.assertTrue(all(b[0] - a[0] >= 2 for a, b in zip(windows, windows[1:])))
        self.assertLessEqual(len(windows), 50)

    def test_over_long_word_gets_its_own_window(self):
        self.assertEqual(make_windows([3, 50, 2], budget=10, overlap=4), [(0, 1), (1, 2), (2, 3)])

    def test_owner_is_the_window_with_most_context(self):
        windows = make_windows([1] * 20, budget=10, overlap=4)
        own = owners(20, windows)
        self.assertEqual(own[0], 0)      # turn start: unlimited left context
        self.assertEqual(own[8], 1)      # 1 word right of it in window 0, 2 on its weaker side in window 1
        self.assertEqual(own[19], 2)     # turn end: unlimited right context

    def test_every_word_owned_once_by_a_window_that_contains_it(self):
        rng = random.Random(0)
        for _ in range(200):
            counts = [rng.randint(1, 4) for _ in range(rng.randint(1, 120))]
            windows = make_windows(counts, budget=rng.randint(4, 40), overlap=rng.randint(0, 12))
            own = owners(len(counts), windows)
            for i, k in enumerate(own):
                start, end = windows[k]
                self.assertTrue(start <= i < end)


class AlignmentTests(unittest.TestCase):
    def test_real_turns_are_covered_once_with_first_subword_labels(self):
        tokenizer = cached_tokenizer("bert-base-uncased")
        with (DATA / "validation.jsonl").open(encoding="utf-8") as handle:
            examples = [json.loads(line) for line in handle]
        examples = sorted(examples, key=lambda ex: len(ex["words"]))[-5:] + examples[:20]  # longest turns too
        cfg = Config(max_len=64, overlap=16)
        items = encode_examples(examples, tokenizer, cfg, with_labels=True)
        self.assertEqual(coverage_problems(items, examples), 0)
        self.assertGreater(len(items), len(examples))  # long turns were cut into several windows
        for item in items:
            self.assertLessEqual(len(item["input_ids"]), 64)
            for position, i in item["targets"]:
                ex = examples[item["turn"]]
                self.assertEqual(LABELS[item["labels"][position]], ex["labels"][i])
            labelled = [p for p, y in enumerate(item["labels"]) if y != -100]
            self.assertEqual(sorted(labelled), sorted(p for p, _ in item["targets"]))


class TrainRunTests(unittest.TestCase):
    def make_data(self, tmp: str) -> Path:
        data_dir = Path(tmp) / "data"
        data_dir.mkdir()
        for split, n in (("train", 40), ("validation", 10), ("test", 10)):
            with (DATA / f"{split}.jsonl").open(encoding="utf-8") as handle:
                rows = [json.loads(line) for _, line in zip(range(n), handle)]
            (data_dir / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        return data_dir

    def test_tiny_run_writes_outputs_restarts_and_skips(self):
        cached_tokenizer(TINY_BERT)
        import fomcpunct.transformer as tr

        with tempfile.TemporaryDirectory() as tmp:
            data_dir, workdir = self.make_data(tmp), Path(tmp) / "work"
            cfg = Config(max_len=48, overlap=8, batch_size=8, eval_batch_size=8, epochs=2)
            tiny = (TINY_BERT, None)

            # the first attempt dies at the very end, before the card is written (a Colab disconnect)
            with mock.patch.object(tr, "latency", side_effect=RuntimeError("disconnected")):
                with self.assertRaises(RuntimeError):
                    tr.train_run("bert", 5e-4, 0, data_dir, workdir, cfg, model=tiny, log=lambda *_: None)
            run_dir = tr.run_dir_for(workdir, "bert", 5e-4, 0)
            self.assertFalse((run_dir / "card.json").exists())
            self.assertEqual(list(run_dir.glob("*.pt")), [])  # no weights are ever stored

            messages = []
            card = tr.train_run("bert", 5e-4, 0, data_dir, workdir, cfg, model=tiny, log=messages.append)
            self.assertTrue(any("starting it again" in m for m in messages))
            self.assertEqual(len(card["history"]), 2)
            self.assertEqual(card["parameter_dtypes"], ["torch.float32"])
            self.assertEqual(card["identity"]["code"], tr.CODE_HASH)
            for split, n in (("validation", 10), ("test", 10)):
                rows = [json.loads(x) for x in (run_dir / f"predictions.{split}.jsonl").read_text().splitlines()]
                self.assertEqual(len(rows), n)
                self.assertTrue(all(set(r["labels"]) <= set(LABELS) for r in rows))

            messages = []
            tr.train_run("bert", 5e-4, 0, data_dir, workdir, cfg, model=tiny, log=messages.append)
            self.assertTrue(any("skipping" in m for m in messages))

            self.assertEqual(tr.select_lr("bert", [5e-4], workdir), 5e-4)
            tr.export("bert", 5e-4, [0], workdir)
            self.assertTrue((workdir / "results" / "predictions" / "bert_s0.test.jsonl").exists())

    def test_predictions_come_from_the_best_epoch(self):
        # no weights are kept, so the files must hold exactly the best epoch's predictions
        cached_tokenizer(TINY_BERT)
        import fomcpunct.transformer as tr

        calls = []
        real_predict = tr.predict

        def recording_predict(net, tokenizer, examples, *args, **kwargs):
            out = real_predict(net, tokenizer, examples, *args, **kwargs)
            calls.append((examples[0]["id"], out))
            return out

        with tempfile.TemporaryDirectory() as tmp:
            data_dir, workdir = self.make_data(tmp), Path(tmp) / "work"
            cfg = Config(max_len=48, overlap=8, batch_size=8, eval_batch_size=8, epochs=3)
            validation_id = json.loads((data_dir / "validation.jsonl").read_text().splitlines()[0])["id"]
            test_id = json.loads((data_dir / "test.jsonl").read_text().splitlines()[0])["id"]
            with mock.patch.object(tr, "predict", side_effect=recording_predict), \
                    mock.patch.object(tr, "macro_f1", side_effect=[0.1, 0.3, 0.2]):
                card = tr.train_run("bert", 5e-4, 0, data_dir, workdir, cfg, model=(TINY_BERT, None),
                                    log=lambda *_: None)
            self.assertEqual(card["best_epoch"], 2)
            self.assertEqual([h["validation_macro_f1"] for h in card["history"]], [0.1, 0.3, 0.2])
            validation_calls = [out for first, out in calls if first == validation_id]
            test_calls = [out for first, out in calls if first == test_id]
            run_dir = tr.run_dir_for(workdir, "bert", 5e-4, 0)
            read = lambda split: [json.loads(x)["labels"] for x in  # noqa: E731
                                  (run_dir / f"predictions.{split}.jsonl").read_text().splitlines()]
            self.assertEqual(read("validation"), validation_calls[1])  # epoch 2 of 3
            self.assertEqual(read("test"), test_calls[1])              # written at epochs 1 and 2 only
            self.assertEqual(len(test_calls), 2)

            with self.assertRaises(RuntimeError):  # same run folder, different identity
                tr.train_run("bert", 5e-4, 0, data_dir, workdir, Config(max_len=48, overlap=8, epochs=1),
                             model=(TINY_BERT, None), log=lambda *_: None)

    def test_fp16_checkpoint_is_trained_in_fp32(self):
        # microsoft/deberta-v3-base is stored in float16 with no dtype in its config; transformers 5
        # would load it as float16, and mixed-precision training then fails at the first step.
        cached_tokenizer(TINY_DEBERTA)
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        import fomcpunct.transformer as tr

        with tempfile.TemporaryDirectory() as tmp:
            fp16_dir = Path(tmp) / "fp16_model"
            net = AutoModelForTokenClassification.from_pretrained(TINY_DEBERTA, local_files_only=True).half()
            net.save_pretrained(fp16_dir)
            AutoTokenizer.from_pretrained(TINY_DEBERTA, local_files_only=True).save_pretrained(fp16_dir)
            config = json.loads((fp16_dir / "config.json").read_text())
            for key in ("dtype", "torch_dtype"):
                config.pop(key, None)
            (fp16_dir / "config.json").write_text(json.dumps(config))
            reloaded = AutoModelForTokenClassification.from_pretrained(fp16_dir)
            self.assertEqual(next(reloaded.parameters()).dtype, torch.float16)  # the trap is real

            cfg = Config(max_len=48, overlap=8, batch_size=8, eval_batch_size=8, epochs=1)
            card = tr.train_run("deberta", 5e-4, 0, self.make_data(tmp), Path(tmp) / "work", cfg,
                                model=(str(fp16_dir), None), log=lambda *_: None)
            self.assertEqual(card["parameter_dtypes"], ["torch.float32"])


class ImportTests(unittest.TestCase):
    def test_import_checks_identity_and_paths(self):
        import zipfile
        from dataclasses import asdict

        from fomcpunct import colab
        from fomcpunct.transformer import MODELS

        local = colab.local_fingerprint()

        def card(key, seed, **changes):
            model, revision = MODELS[key]
            identity = {"data": local["data"], "code": local["code"], "model": model, "revision": revision,
                        "seed": seed, "config": asdict(Config())}
            identity.update(changes)
            return json.dumps({"identity": identity, "environment": {"device": "Tesla T4"}})

        with tempfile.TemporaryDirectory() as tmp:
            def archive(files: dict) -> Path:
                path = Path(tmp) / f"r{len(list(Path(tmp).iterdir()))}.zip"
                with zipfile.ZipFile(path, "w") as z:
                    for name, content in files.items():
                        z.writestr(name, content)
                return path

            fake_repo = Path(tmp) / "repo"
            with mock.patch.object(colab, "REPO", fake_repo), \
                    mock.patch.object(colab, "local_fingerprint", return_value=local):
                good = archive({"results/training/bert_s0.json": card("bert", 0),
                                "results/predictions/bert_s0.validation.jsonl": "{}\n"})
                colab.import_results(good, force=False)
                self.assertTrue((fake_repo / "results" / "predictions" / "bert_s0.validation.jsonl").exists())

                refused = [
                    {"results/training/bert_s0.json": card("bert", 0, data={"train": "x"})},      # other data
                    {"results/training/bert_s0.json": card("deberta", 0)},                        # other model
                    {"results/training/bert_s0.json": card("bert", 7)},                           # other seed
                    {"results/training/bert_s0.json": card("bert", 0, revision="main")},         # other revision
                    {"results/training/bert_s0.json": "[]"},                                      # not a card
                    {"results/training/bert_s0.json": card("bert", 0), "results/../escape.txt": "x"},
                    {"results/training/bert_s0.json": card("bert", 0),
                     "results/predictions/bert_s9.test.jsonl": "{}\n"},                           # no card
                ]
                for files in refused:
                    with self.assertRaises(SystemExit, msg=str(list(files))):
                        colab.import_results(archive(files), force=False)


if __name__ == "__main__":
    unittest.main()
