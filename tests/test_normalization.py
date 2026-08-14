"""Tests for F5-TTS official text normalization.

The expected values are taken from the reference artifacts of the original runs
(``eval_*_f5exact.json``), so these cases pin the normalizer to the behaviour
that produced the published WER numbers.
"""

from bon_tts.normalization import normalize_en


class TestPunctuationDeletion:
    def test_deletes_rather_than_spaces(self):
        # A trailing hyphen must vanish without leaving a token boundary.
        assert normalize_en("he proceeded-") == "he proceeded"

    def test_stacked_punctuation(self):
        assert normalize_en('So there is to me"! added Sandford.') == "so there is to me added sandford"

    def test_mixed_quotes_and_commas(self):
        raw = 'I will make no unjust use of what I know," he replied with firmness. "I believe you, my Lord".'
        expected = "i will make no unjust use of what i know he replied with firmness i believe you my lord"
        assert normalize_en(raw) == expected

    def test_apostrophes_are_removed(self):
        # F5-TTS strips apostrophes on both reference and hypothesis, so the
        # asymmetry cancels; "dont" vs "don't" never costs a word error.
        assert normalize_en("Don't stop.") == "dont stop"


class TestCaseAndWhitespace:
    def test_lowercases(self):
        assert normalize_en("Bill Harmon") == "bill harmon"

    def test_collapses_whitespace(self):
        assert normalize_en("  too   many    spaces  ") == "too many spaces"

    def test_newlines_and_tabs(self):
        assert normalize_en("line one\n\tline two") == "line one line two"


class TestEdgeCases:
    def test_empty_string(self):
        assert normalize_en("") == ""

    def test_punctuation_only(self):
        assert normalize_en("...!?") == ""

    def test_idempotent(self):
        # Reference artifacts store already-normalized text; re-normalizing it
        # must be a no-op or validate_reproduction would double-transform.
        once = normalize_en('Hello, world! It"s fine.')
        assert normalize_en(once) == once

    def test_digits_survive(self):
        assert normalize_en("Room 42.") == "room 42"
