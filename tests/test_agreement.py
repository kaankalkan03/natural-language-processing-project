"""Tests for the human agreement sheets and the saturation report."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fomcpunct import agreement, saturation
from fomcpunct.agreement import SheetError, cohen_kappa, parse_sheet, parse_turn, render_sheet
from fomcpunct.metrics import confusion

DATA = Path("data/processed")


def punctuated(words: list[str], marks: list[str]) -> str:
    """Write a turn back as text, each word followed by its raw transcript marks."""
    return " ".join(word + mark for word, mark in zip(words, marks))


class ParseTurnTests(unittest.TestCase):
    def labels(self, text, words):
        return parse_turn(text, words)[0]

    def test_marks_between_words(self):
        words = "good afternoon we remain focused is that right".split()
        text = "Good afternoon. We remain focused, is that right?"
        self.assertEqual(self.labels(text, words),
                         ["O", "PERIOD", "O", "O", "COMMA", "O", "O", "QUESTION"])

    def test_dataset_mapping(self):
        words = ["a", "b", "c", "d", "e", "f", "g"]
        text = "a: b; c! d -- e — f?— g"
        self.assertEqual(self.labels(text, words),
                         ["COMMA", "PERIOD", "PERIOD", "COMMA", "COMMA", "QUESTION", "O"])

    def test_unspaced_em_dash_is_a_pause(self):
        self.assertEqual(self.labels("at—at a rate", ["at", "at", "a", "rate"]),
                         ["COMMA", "O", "O", "O"])

    def test_unspaced_hyphen_joins_words(self):
        with self.assertRaisesRegex(SheetError, "joined"):
            parse_turn("long-run goals", ["long", "run", "goals"])
        with self.assertRaisesRegex(SheetError, "joined"):
            parse_turn("long–run goals", ["long", "run", "goals"])

    def test_words_with_inner_punctuation(self):
        words = ["the", "u.s.", "rate", "is", "2.5", "and/or", "$3,000", "401(k)", "that's", "long-run"]
        text = "The U.S. rate is 2.5, and/or $3,000; 401(k) that’s long-run."
        self.assertEqual(self.labels(text, words),
                         ["O", "O", "O", "O", "COMMA", "O", "PERIOD", "O", "O", "PERIOD"])

    def test_sentence_end_after_abbreviation(self):
        self.assertEqual(self.labels("in the u.s.. we", ["in", "the", "u.s.", "we"]),
                         ["O", "O", "PERIOD", "O"])

    def test_ellipsis_quotes_and_line_breaks(self):
        words = ["well", "we", "said", "patient", "here"]
        text = 'well... we\nsaid "patient," (here)'
        self.assertEqual(self.labels(text, words), ["O", "O", "O", "COMMA", "O"])

    def test_ellipsis_variants(self):
        words = ["a", "b", "c", "d", "e"]
        # "...", ". . ." and the ellipsis character get no label; a fourth period ends the sentence
        self.assertEqual(self.labels("a... b . . . c… d.... e", words),
                         ["O", "O", "O", "PERIOD", "O"])
        self.assertEqual(self.labels("a…. b", ["a", "b"]), ["PERIOD", "O"])

    def test_unspaced_double_hyphen_is_a_dash(self):
        self.assertEqual(self.labels("at--at a rate", ["at", "at", "a", "rate"]),
                         ["COMMA", "O", "O", "O"])

    def test_turkish_keyboard_capitals(self):
        words = ["inflation", "is", "high", "i", "think"]
        self.assertEqual(self.labels("İnflation İs high. İ think", words),
                         ["O", "O", "PERIOD", "O", "O"])
        self.assertEqual(self.labels("ınflation is high", words[:3]), ["O", "O", "O"])

    def test_abbreviation_before_a_capital_warns(self):
        labels, warnings = parse_turn("in the U.S. We think", ["in", "the", "u.s.", "we", "think"])
        self.assertEqual(labels, ["O"] * 5)
        self.assertEqual(len(warnings), 1)
        self.assertIn("u.s..", warnings[0])
        self.assertEqual(parse_turn("in the U.S.. We think", ["in", "the", "u.s.", "we", "think"])[1], [])
        self.assertEqual(parse_turn("in the U.S. economy", ["in", "the", "u.s.", "economy"])[1], [])
        # a title before a name is not a sentence end
        self.assertEqual(parse_turn("thank you, Mr. Chairman.", ["thank", "you", "mr.", "chairman"])[1], [])

    def test_marks_before_the_first_word_are_ignored(self):
        self.assertEqual(self.labels(", yes.", ["yes"]), ["PERIOD"])

    def test_changed_added_and_missing_words(self):
        words = ["inflation", "has", "eased"]
        for text in ("inflation has eaesd.", "inflation has really eased.", "inflation eased.",
                     "inflation haseased.", "inflation has eased. thanks"):
            with self.subTest(text=text), self.assertRaises(SheetError):
                parse_turn(text, words)

    def test_prefix_of_a_longer_word_does_not_match(self):
        with self.assertRaises(SheetError):
            parse_turn("about it", ["a", "it"])


class SheetTests(unittest.TestCase):
    TURNS = [{"id": "x-t1", "doc": "x", "role": "chair", "words": "we kept rates unchanged".split(),
              "labels": ["O", "O", "O", "PERIOD"]},
             {"id": "y-t2", "doc": "y", "role": "reporter", "words": "will you cut rates".split(),
              "labels": ["O", "O", "O", "QUESTION"]}]

    def test_blank_sheet_round_trip_warns(self):
        labels, warnings = parse_sheet(render_sheet("A", self.TURNS), self.TURNS)
        self.assertEqual(labels, [["O"] * 4, ["O"] * 4])
        self.assertEqual(len(warnings), 2)

    def test_filled_sheet(self):
        sheet = render_sheet("A", self.TURNS).replace("unchanged", "unchanged.").replace(
            "cut rates", "cut rates?")
        labels, warnings = parse_sheet(sheet, self.TURNS)
        self.assertEqual(labels, [t["labels"] for t in self.TURNS])
        self.assertEqual(warnings, [])

    def test_all_errors_are_listed(self):
        sheet = render_sheet("A", self.TURNS).replace("kept", "kep").replace("will", "would")
        with self.assertRaises(SheetError) as caught:
            parse_sheet(sheet, self.TURNS)
        self.assertIn("T01", str(caught.exception))
        self.assertIn("T02", str(caught.exception))

    def test_missing_header(self):
        sheet = render_sheet("A", self.TURNS).replace("=== T02 ===", "")
        with self.assertRaisesRegex(SheetError, "missing"):
            parse_sheet(sheet, self.TURNS)

    def test_indented_header_and_punctuation_only_line(self):
        sheet = render_sheet("A", self.TURNS).replace("=== T01 ===", "  === T01 ===").replace(
            "unchanged", "unchanged\n---")
        labels, warnings = parse_sheet(sheet, self.TURNS)
        self.assertEqual(labels[0], ["O", "O", "O", "COMMA"])
        self.assertTrue(any("only '---'" in w for w in warnings))

    def test_bom_and_windows_line_endings(self):
        sheet = "﻿" + render_sheet("A", self.TURNS).replace("\n", "\r\n")
        labels, _ = parse_sheet(sheet.encode("utf-8").decode("utf-8-sig"), self.TURNS)
        self.assertEqual(len(labels), 2)


class SampleTests(unittest.TestCase):
    def examples(self):
        out = []
        for d in range(6):
            for t, role in enumerate(["chair", "reporter", "moderator", "chair", "reporter"]):
                n = 50 if t != 2 else 3
                out.append({"id": f"d{d}-t{t}", "doc": f"d{d}", "role": role, "words": ["w"] * n,
                            "labels": ["O"] * n})
        return out

    def test_sample_constraints_and_determinism(self):
        a = agreement.sample_turns(self.examples(), per_role=3, seed=1)
        b = agreement.sample_turns(self.examples(), per_role=3, seed=1)
        self.assertEqual([x["id"] for x in a], [x["id"] for x in b])
        self.assertEqual(len(a), 6)
        for role in ("chair", "reporter"):
            docs = [x["doc"] for x in a if x["role"] == role]
            self.assertEqual(len(docs), 3)
            self.assertEqual(len(set(docs)), 3)
        self.assertEqual(len({x["doc"] for x in a}), 6)  # front and back of the order do not meet

    def test_too_few_conferences(self):
        with self.assertRaises(ValueError):
            agreement.sample_turns(self.examples(), per_role=7)


class ScoreTests(unittest.TestCase):
    def test_kappa_hand_computed(self):
        cm = np.zeros((4, 4))
        cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1] = 40, 10, 5, 45
        # observed 0.85; chance (50*45 + 50*55) / 100^2 = 0.5
        self.assertAlmostEqual(float(cohen_kappa(cm)), 0.7)
        self.assertEqual(cohen_kappa(np.stack([cm, cm])).shape, (2,))

    def test_identical_annotations_score_one(self):
        gold = [["O", "COMMA", "PERIOD", "QUESTION"], ["O", "PERIOD"]]
        turns = [{"words": ["w"] * len(g), "labels": g} for g in gold]
        result = agreement.score({"A": gold, "B": gold}, turns, {}, resamples=50)
        for row in result["rows"]:
            self.assertAlmostEqual(row["macro_f1"], 1.0)
            self.assertAlmostEqual(row["kappa"], 1.0)

    def test_pairs_and_symmetry(self):
        gold = [["O", "COMMA", "PERIOD", "O", "QUESTION"]]
        a = [["O", "PERIOD", "PERIOD", "COMMA", "QUESTION"]]
        b = [["COMMA", "COMMA", "O", "O", "QUESTION"]]
        turns = [{"words": ["w"] * 5, "labels": gold[0]}]
        result = agreement.score({"A": a, "B": b}, turns, {"m": a}, resamples=50)
        pairs = [(r["reference"], r["other"]) for r in result["rows"]]
        self.assertEqual(pairs, [("transcript", "A"), ("transcript", "B"), ("A", "B"),
                                 ("transcript", "m"), ("A", "m"), ("B", "m")])
        ab = next(r for r in result["rows"] if (r["reference"], r["other"]) == ("A", "B"))
        swapped = agreement.compare(b, a, 50, 0)
        self.assertAlmostEqual(ab["macro_f1"], swapped["macro_f1"])
        self.assertAlmostEqual(ab["kappa"], swapped["kappa"])
        self.assertAlmostEqual(result["rows"][3]["macro_f1"], result["rows"][0]["macro_f1"])
        diff = result["model_minus_annotator_vs_transcript"]
        self.assertAlmostEqual(next(d for d in diff if d["annotator"] == "A")["difference"], 0.0)


class SaturationTests(unittest.TestCase):
    def test_exact_match_and_accuracy(self):
        examples = [{"doc": "a", "labels": ["O", "PERIOD"]}, {"doc": "a", "labels": ["COMMA", "O"]},
                    {"doc": "b", "labels": ["O", "QUESTION"]}]
        pred = [["O", "PERIOD"], ["O", "O"], ["O", "QUESTION"]]
        counts = saturation.doc_exact_counts(examples, pred, ["a", "b"])
        self.assertEqual(counts.tolist(), [[1, 2], [1, 1]])
        self.assertAlmostEqual(float(saturation.exact_share(counts.sum(axis=0))), 2 / 3)
        cm = confusion([e["labels"] for e in examples], pred)
        self.assertAlmostEqual(float(saturation.accuracy(cm)), 5 / 6)

    def test_exact_match_without_one_word_turns(self):
        examples = [{"doc": "a", "labels": ["PERIOD"]}, {"doc": "a", "labels": ["COMMA", "O"]}]
        pred = [["PERIOD"], ["O", "O"]]
        counts = saturation.doc_exact_counts(examples, pred, ["a"], min_words=2)
        self.assertEqual(counts.tolist(), [[0, 1]])


@unittest.skipUnless((DATA / "test.jsonl").exists(), "processed data not available")
class DataRoundTripTests(unittest.TestCase):
    def test_transcript_marks_give_the_dataset_labels(self):
        """Typing each turn with its own transcript marks must give back the labels."""
        for split in ("train", "validation", "test"):
            with (DATA / f"{split}.jsonl").open(encoding="utf-8") as handle:
                examples = [json.loads(line) for line in handle]
            for ex in examples:
                labels, _ = parse_turn(punctuated(ex["words"], ex["marks"]), ex["words"])
                self.assertEqual(labels, ex["labels"], ex["id"])

    def test_changed_data_is_detected(self):
        with (DATA / "test.jsonl").open(encoding="utf-8") as handle:
            ex = json.loads(handle.readline())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.json"
            entry = {"key": "T01", "id": ex["id"], "words": len(ex["words"]),
                     "words_sha256": agreement.words_hash(ex["words"])}
            path.write_text(json.dumps({"split": "test", "turns": [entry]}), encoding="utf-8")
            self.assertEqual(agreement.load_sample(path)[1][0]["id"], ex["id"])
            entry["words_sha256"] = "0" * 16
            path.write_text(json.dumps({"split": "test", "turns": [entry]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed"):
                agreement.load_sample(path)


if __name__ == "__main__":
    unittest.main()
