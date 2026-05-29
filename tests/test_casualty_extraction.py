"""Tests for casualty extraction from free-text headlines/descriptions.

Previously the extractor only matched digit forms ("12 killed"). News and
alert text frequently uses English number words instead — especially for
small counts ("five killed", "twelve dead", "twenty-three civilians
killed"). This module pins down both forms going through one extraction
pipeline.

Run:
    pytest tests/test_casualty_extraction.py -v
"""

from __future__ import annotations

import pytest

from src.services.signal import (
    _parse_number_phrase,
    extract_casualties_from_text,
)


class TestParseNumberPhrase:
    """Unit tests for the digits-or-words parser itself."""

    @pytest.mark.parametrize("raw,expected", [
        ("5", 5),
        ("12", 12),
        ("100", 100),
        ("0", 0),
    ])
    def test_digit_forms(self, raw: str, expected: int):
        assert _parse_number_phrase(raw) == expected

    @pytest.mark.parametrize("raw,expected", [
        ("one", 1),
        ("five", 5),
        ("nine", 9),
        ("ten", 10),
        ("twelve", 12),
        ("fourteen", 14),
        ("nineteen", 19),
        ("twenty", 20),
        ("fifty", 50),
        ("ninety", 90),
    ])
    def test_single_word_numbers(self, raw: str, expected: int):
        assert _parse_number_phrase(raw) == expected

    @pytest.mark.parametrize("raw,expected", [
        ("twenty-one", 21),
        ("twenty one", 21),
        ("thirty-five", 35),
        ("forty three", 43),
        ("ninety-nine", 99),
    ])
    def test_compound_tens_and_ones(self, raw: str, expected: int):
        assert _parse_number_phrase(raw) == expected

    @pytest.mark.parametrize("raw,expected", [
        ("one hundred", 100),
        ("two hundred", 200),
        ("nine hundred", 900),
    ])
    def test_hundreds(self, raw: str, expected: int):
        assert _parse_number_phrase(raw) == expected

    @pytest.mark.parametrize("raw,expected", [
        ("one hundred and fifty", 150),
        ("two hundred fifty", 250),
        ("three hundred and twenty-five", 325),
        ("five hundred and one", 501),
    ])
    def test_hundreds_with_remainder(self, raw: str, expected: int):
        assert _parse_number_phrase(raw) == expected

    @pytest.mark.parametrize("raw,expected", [
        ("one thousand", 1000),
        ("two thousand", 2000),
        ("one thousand five hundred", 1500),
        ("two thousand and fifty", 2050),
    ])
    def test_thousands(self, raw: str, expected: int):
        assert _parse_number_phrase(raw) == expected

    def test_unknown_token_returns_none(self):
        """A token we don't recognise must NOT silently produce a number."""
        assert _parse_number_phrase("twentyish") is None
        assert _parse_number_phrase("forty-bazillion") is None
        assert _parse_number_phrase("") is None
        assert _parse_number_phrase("   ") is None

    def test_case_insensitive(self):
        assert _parse_number_phrase("Fourteen") == 14
        assert _parse_number_phrase("TWENTY-ONE") == 21


class TestExtractCasualtiesFromText:
    """Black-box extraction over realistic title/description strings.
    Exercises the full regex + parser chain."""

    # ── Digit-form regression coverage ────────────────────────────────────

    @pytest.mark.parametrize("text,expected", [
        ("12 killed in airstrike", 12),
        ("At least 5 dead after blast", 5),
        ("3 fatalities reported overnight", 3),
        ("Death toll of 14 from flash floods", 14),
        ("Strike on market leaves 6 dead", 6),
        ("23 civilians killed in clashes", 23),
    ])
    def test_digit_forms_still_work(self, text: str, expected: int):
        assert extract_casualties_from_text(text) == expected

    # ── Word-form: this is the new behaviour ──────────────────────────────

    @pytest.mark.parametrize("text,expected", [
        ("Five killed in roadside blast", 5),
        ("Twelve dead after dam collapse", 12),
        ("Fourteen killed in overnight airstrike", 14),
        ("Twenty civilians killed in market attack", 20),
        ("Death toll of fourteen reported", 14),
        ("Leaving twelve dead in border clash", 12),
        ("Eight people were killed in the explosion", 8),
    ])
    def test_word_forms_resolve(self, text: str, expected: int):
        assert extract_casualties_from_text(text) == expected

    @pytest.mark.parametrize("text,expected", [
        ("Twenty-one civilians killed in raid", 21),
        ("Thirty-five killed in convoy ambush", 35),
        ("Forty-three dead after building collapse", 43),
        ("Death toll rose to ninety-nine", 99),
    ])
    def test_hyphenated_compounds(self, text: str, expected: int):
        assert extract_casualties_from_text(text) == expected

    @pytest.mark.parametrize("text,expected", [
        ("One hundred killed in earthquake", 100),
        ("Two hundred dead after flooding", 200),
        ("Death toll of one hundred and fifty reported", 150),
    ])
    def test_hundreds(self, text: str, expected: int):
        assert extract_casualties_from_text(text) == expected

    # ── Mixed and prioritisation ──────────────────────────────────────────

    def test_max_across_multiple_mentions(self):
        """When a story mentions running tallies, return the largest count
        — that's typically the most up-to-date estimate."""
        text = "Initial reports of five killed; death toll rose to fourteen."
        assert extract_casualties_from_text(text) == 14

    def test_word_and_digit_in_same_text(self):
        """Mixed forms — pick the max regardless of which form gave it."""
        text = "Twelve dead in overnight blast; 23 civilians killed in follow-up."
        assert extract_casualties_from_text(text) == 23

    def test_uses_title_and_description(self):
        """Multiple inputs are searched; max wins across them."""
        assert extract_casualties_from_text(
            "Five killed in market blast",
            "Local sources later confirmed twelve fatalities.",
        ) == 12

    # ── Boundaries / negative cases ───────────────────────────────────────

    def test_returns_none_when_no_pattern_matches(self):
        assert extract_casualties_from_text("Floods displace residents in Khartoum") is None
        assert extract_casualties_from_text("Markets disrupted by security operation") is None

    def test_injured_displaced_not_counted_as_casualties(self):
        """Population-affected phrasing (injured/displaced) must NOT bleed
        into the casualty count."""
        assert extract_casualties_from_text("Twelve injured in blast") is None
        assert extract_casualties_from_text("Twenty residents displaced") is None

    def test_handles_none_and_empty_inputs(self):
        assert extract_casualties_from_text(None) is None
        assert extract_casualties_from_text("") is None
        assert extract_casualties_from_text(None, "", None) is None

    def test_word_only_no_killed_does_not_extract(self):
        """The number word alone, without one of the casualty verbs nearby,
        must not produce a hit — otherwise stray prose ('twelve months
        ago') would pollute the count."""
        assert extract_casualties_from_text(
            "Twelve months ago the region was peaceful",
        ) is None
