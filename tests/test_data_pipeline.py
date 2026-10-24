"""Unit tests for cleaning and labelling rules. Run: python -m unittest -v"""

import unittest
from collections import Counter

from fomcpunct.build import split_of
from fomcpunct.clean import (FOOTNOTE, SPEAKER, SPEAKER_NO_MARK, normalize, remove_footnote_markers,
                             role_of)
from fomcpunct.tokens import mark_to_label, to_example


class LabelTests(unittest.TestCase):
    def test_mark_mapping(self):
        self.assertEqual(mark_to_label(""), "O")
        self.assertEqual(mark_to_label(","), "COMMA")
        self.assertEqual(mark_to_label(":"), "COMMA")
        self.assertEqual(mark_to_label("—"), "COMMA")
        self.assertEqual(mark_to_label("."), "PERIOD")
        self.assertEqual(mark_to_label(";"), "PERIOD")
        self.assertEqual(mark_to_label("?—"), "QUESTION")

    def test_words_are_lowercased_and_stripped(self):
        words, labels, _ = to_example("Good afternoon. How are you?")
        self.assertEqual(words, ["good", "afternoon", "how", "are", "you"])
        self.assertEqual(labels, ["O", "PERIOD", "O", "O", "QUESTION"])

    def test_abbreviation_period_stays_in_word(self):
        words, labels, _ = to_example("The U.S. economy is strong.")
        self.assertEqual(words[1], "u.s.")
        self.assertEqual(labels[1], "O")
        self.assertEqual(labels[-1], "PERIOD")

    def test_abbreviation_can_end_a_sentence(self):
        words, labels, _ = to_example("It is good for the U.S. We think so.")
        self.assertEqual(words[5], "u.s.")
        self.assertEqual(labels[5], "PERIOD")

    def test_numbers_keep_inner_punctuation(self):
        words, labels, _ = to_example("Gains averaged 116,000 and 2.2 percent.")
        self.assertIn("116,000", words)
        self.assertIn("2.2", words)
        self.assertEqual(labels[words.index("116,000")], "O")

    def test_more_abbreviations(self):
        words, labels, _ = to_example("Core services ex. housing rose. He has a Ph.D. in economics.")
        self.assertIn("ex.", words)
        self.assertIn("ph.d.", words)
        self.assertEqual(labels[words.index("ex.")], "O")
        self.assertEqual(labels[words.index("ph.d.")], "O")

    def test_dash_becomes_comma_on_previous_word(self):
        text = normalize("of a recession—sorry, of a downturn", Counter())
        words, labels, _ = to_example(text)
        self.assertEqual(words[:4], ["of", "a", "recession", "sorry"])
        self.assertEqual(labels[2:4], ["COMMA", "COMMA"])


class CleaningTests(unittest.TestCase):
    def test_stage_notes_dropped_insertions_unwrapped(self):
        log = Counter()
        self.assertEqual(normalize("Thank you. [Laughter] The rate[s] rose.", log),
                         "Thank you. The rates rose.")
        self.assertEqual(log["stage_notes_dropped"], 1)
        self.assertEqual(log["editorial_insertions_unwrapped"], 1)

    def test_more_stage_notes_and_time_stamps(self):
        log = Counter()
        text = normalize("Well [gesturing with hands apart] it [0:24:15.5] is [Silence] fine.", log)
        self.assertEqual(text, "Well it is fine.")
        self.assertEqual(log["stage_notes_dropped"], 3)

    def test_en_dash_compounds_and_spaced_dashes(self):
        self.assertEqual(normalize("Dodd–Frank and 2013–2014", Counter()), "Dodd-Frank and 2013-2014")
        words, labels, _ = to_example(normalize("the outlook – and risks", Counter()))
        self.assertEqual(words, ["the", "outlook", "and", "risks"])
        self.assertEqual(labels[1], "COMMA")

    def test_spaced_ellipsis_gets_no_label(self):
        words, labels, _ = to_example(normalize("said SVB . . . has big losses.", Counter()))
        self.assertEqual(words, ["said", "svb", "has", "big", "losses"])
        self.assertEqual(labels[1], "O")
        words, labels, _ = to_example(normalize("“the risks . . .” he said", Counter()))
        self.assertEqual(labels[words.index("risks")], "O")

    def test_apostrophes_and_abbreviation_dashes(self):
        self.assertEqual(normalize("the U.S.’s role and we‘d say ‘yes.’ Then", Counter()),
                         "the U.S.'s role and we'd say yes. Then")
        words, labels, _ = to_example(normalize("U.S.–China trade", Counter()))
        self.assertEqual(words, ["u.s.-china", "trade"])
        self.assertEqual(labels[0], "O")

    def test_laugh_notes(self):
        self.assertEqual(normalize("so [laughs] there", Counter()), "so there")

    def test_footnote_markers(self):
        log = Counter()
        texts = remove_footnote_markers(["grew in 2021.1 However, 2.5 percent", "economy.2 And",
                                         "meant to say.”3 And", "was transitory”4 then", "in 2021.”5 So"],
                                        ["1", "2", "3", "4", "5"], log)
        self.assertEqual(texts, ["grew in 2021. However, 2.5 percent", "economy. And", "meant to say.” And",
                                 "was transitory” then", "in 2021.” So"])
        self.assertEqual(log["footnote_markers_removed"], 5)

    def test_mark_counters_count_each_case_once(self):
        log = Counter()
        to_example("We grew — and , then stopped.", log)
        self.assertEqual(log["standalone_marks_merged"], 2)
        self.assertEqual(log["leading_marks_moved"], 0)
        log = Counter()
        to_example("We grew ,and then stopped.", log)
        self.assertEqual(log["leading_marks_moved"], 1)
        self.assertEqual(log["standalone_marks_merged"], 0)
        self.assertTrue(FOOTNOTE.match("1 The Chairman meant to say: we could"))
        self.assertTrue(FOOTNOTE.match("2 Chair Powell intended to say"))

    def test_parentheses_and_quotes(self):
        self.assertEqual(normalize("under section 13(3) (the “facility”)", Counter()),
                         "under section 13(3) the facility")

    def test_speaker_labels(self):
        self.assertTrue(SPEAKER.match("CHAIR POWELL. Good afternoon."))
        self.assertTrue(SPEAKER.match("CHAIRMAN POWELL: Good afternoon."))
        self.assertTrue(SPEAKER.match("QUESTION. Mr. Chairman,"))
        self.assertTrue(SPEAKER.match("VICTORIA McGRANE. Thank you."))
        self.assertTrue(SPEAKER.match("HUGUES HONORÉ. Merci."))
        self.assertIsNone(SPEAKER.match("NPR. About financial conditions"))
        self.assertIsNone(SPEAKER.match("U.S. GDP. grew"))
        match = SPEAKER_NO_MARK.match("CHAIR POWELL You know, I think")
        self.assertEqual(match.group(1), "CHAIR POWELL")
        self.assertEqual("CHAIR POWELL You know, I think"[match.end():], "You know, I think")
        self.assertIsNone(SPEAKER_NO_MARK.match("CARES Act funds were"))
        self.assertIsNone(SPEAKER_NO_MARK.match("CHAIR POWELL said that"))
        self.assertEqual(role_of("CHAIMRAN POWELL"), "chair")
        self.assertEqual(role_of("MICHELLE SMITH"), "moderator")
        self.assertEqual(role_of("JOE PAVEL"), "moderator")
        self.assertEqual(role_of("NICK TIMIRAOS"), "reporter")


class SplitTests(unittest.TestCase):
    def test_split_by_year(self):
        self.assertEqual(split_of("20221214"), "train")
        self.assertEqual(split_of("20230201"), "validation")
        self.assertEqual(split_of("20240131"), "test")



if __name__ == "__main__":
    unittest.main()
