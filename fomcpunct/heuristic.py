"""Rule-based baseline (no learning).

Word lists were chosen from training-split statistics only. Rules for each
word, in the order the code checks them (the first rule that applies wins):

1. Last word of the turn: PERIOD.
2. Next word is the same word (spoken repair, "at at"): COMMA.
   Train: 91.7% of 2,797 repeats are followed by a COMMA.
3. Next word is "so" or "thank": PERIOD.
   Train: the word before "so" has PERIOD 62.7% of 4,145 times; before "thank" 49.1% of 401.
4. COMMA after "you know", "of course", "i mean", "again" or "hi", or before
   "which" or "but". Train: "you know" 97.8%, "of course" 90.8%, "i mean" 93.6%,
   "again" 75.8%, "hi" 75.3%; COMMA before "which" 70.5%, before "but" 49.2%
   (any mark before "but" about 93%).
5. A sentence (from the turn start or after a PERIOD from rules 1-3) whose first
   word is a question opener ends with QUESTION instead of PERIOD.
   Train: these 17 openers start a question 90.6% of 1,251 times
   (is 98%, do 95%, how 95%, what 62%).
"""

from __future__ import annotations

COMMA_AFTER = {"again", "hi"}
COMMA_AFTER_BIGRAMS = {("you", "know"), ("of", "course"), ("i", "mean")}
COMMA_BEFORE = {"which", "but"}
PERIOD_BEFORE = {"so", "thank"}
QUESTION_OPENERS = {"is", "are", "do", "does", "did", "how", "can", "could", "would",
                    "why", "what", "has", "have", "will", "should", "was", "were"}


class HeuristicTagger:
    name = "heuristic"

    def fit(self, examples: list[dict]) -> "HeuristicTagger":
        return self  # rules are fixed

    def predict_one(self, words: list[str]) -> list[str]:
        n = len(words)
        labels = ["O"] * n
        for i, word in enumerate(words):
            previous = words[i - 1] if i > 0 else None
            following = words[i + 1] if i + 1 < n else None
            if following is None:
                labels[i] = "PERIOD"
            elif following == word:
                labels[i] = "COMMA"
            elif following in PERIOD_BEFORE:
                labels[i] = "PERIOD"
            elif (word in COMMA_AFTER or (previous, word) in COMMA_AFTER_BIGRAMS
                  or following in COMMA_BEFORE):
                labels[i] = "COMMA"
        start = 0
        for i, label in enumerate(labels):
            if label == "PERIOD":
                if words[start] in QUESTION_OPENERS:
                    labels[i] = "QUESTION"
                start = i + 1
        return labels

    def predict(self, sequences: list[list[str]]) -> list[list[str]]:
        return [self.predict_one(words) for words in sequences]
