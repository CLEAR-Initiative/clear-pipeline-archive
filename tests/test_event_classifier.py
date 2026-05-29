"""Tests for the local EventClassifier and its taxonomy.

The classifier loads a sentence-transformers model (~80MB) and computes
embeddings for every taxonomy row on construction — too heavy to load
unconditionally in unit tests. We split the coverage:

  1. **Taxonomy regression tests** — pure-Python, parse the JSON, assert
     properties of the keyword list. These run on every CI and catch the
     specific failure mode this file was created for (overly generic
     keywords like "local" leaking into a category and producing false
     positives on routine Dataminr attribution suffixes).

  2. **End-to-end classifier tests** — actually load the model and run
     `predict()` on the offending Dataminr headline. Skipped gracefully
     when sentence-transformers / numpy / rapidfuzz aren't installed, so
     they're free in environments where those deps are absent.

Background: a Dataminr signal whose headline ends with the attribution
"… Local News Outlet Photo via Mangish" was classified as "severe local
storm". Root cause: `"local"` was a key_word for severe-local-storm, and
the lexical scorer matched `\\blocal\\b` on the word "Local" in the
attribution suffix — producing a non-zero lexical score for the storm
category while the actual conflict-related categories got zero lexical
hits ("Army" doesn't match `\\barmed\\b`, no battle/clash/fighting/etc.
in the headline). The fix was to drop the standalone `"local"` keyword;
the phrase `"severe local storm"` stays in key_phrases so genuine storm
signals still resolve.

Run:
    pytest tests/test_event_classifier.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_TAXONOMY_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "services" / "event_categories.json"
)


# The Dataminr signal that triggered the original bug. Reproduced verbatim
# from production data so the test pins the exact failure mode.
DATAMINR_HEADLINE = (
    "Sudanese Army reportedly targets strongholds of Rapid Support Forces "
    "in Nyala, Sudan: Local News Outlet Photo via Mangish."
)


# ─── Tier 1: taxonomy regression (cheap, no model load) ───────────────────


def _load_taxonomy() -> list[dict]:
    return json.loads(_TAXONOMY_PATH.read_text(encoding="utf-8"))


class TestTaxonomyRegressions:
    """Pure-Python checks on the taxonomy JSON. No model load required."""

    def test_severe_storm_does_not_have_local_keyword(self):
        """`local` was the standalone keyword that matched 'Local News Outlet'
        in Dataminr attribution suffixes and tipped storm over actual conflict
        categories. The phrase 'severe local storm' must still survive in
        key_phrases — that's the legitimate matcher for genuine storm
        signals."""
        taxonomy = _load_taxonomy()
        storm = next(
            (row for row in taxonomy if row.get("id") == "st"),
            None,
        )
        assert storm is not None, "severe-local-storm category (id='st') missing"
        keywords_lower = {str(k).lower() for k in storm.get("key_words", [])}
        assert "local" not in keywords_lower, (
            "Standalone 'local' keyword reintroduced into the severe storm "
            "category — this matches 'Local News Outlet' in Dataminr "
            "attribution suffixes and produces false-positive storm "
            "classifications on military headlines."
        )
        # The longer phrase must still be present so genuine storm signals
        # still match via key_phrases.
        phrases_lower = {str(p).lower() for p in storm.get("key_phrases", [])}
        assert "severe local storm" in phrases_lower

    def test_no_attribution_suffix_words_are_standalone_keywords(self):
        """Dataminr headlines routinely end with attribution like:
        '… Local News Outlet Photo via Mangish', '… Source via Twitter',
        '… Reuters via Reuters'. These attribution words are too generic
        to be standalone keywords in any disaster category — they'd
        produce false positives the moment any Dataminr signal mentions
        its source. Locks in the broader class of bug, not just 'local'."""
        attribution_words = {
            "local", "news", "outlet", "source", "via",
            "photo", "video", "report",
        }
        taxonomy = _load_taxonomy()
        offenders: list[tuple[str, str]] = []
        for row in taxonomy:
            cat_id = str(row.get("id", "?"))
            for kw in row.get("key_words", []):
                if str(kw).lower() in attribution_words:
                    offenders.append((cat_id, str(kw)))
        assert not offenders, (
            "Generic Dataminr attribution words used as standalone keywords; "
            "these will false-positive on signal source suffixes:\n  "
            + "\n  ".join(f"category {cat!r} → keyword {kw!r}" for cat, kw in offenders)
        )


# ─── Tier 2: end-to-end against the real classifier ──────────────────────


@pytest.fixture(scope="module")
def classifier():
    """Load the real classifier once per test module. Heavy (~80MB model +
    embedding compute on every taxonomy row), so the fixture lifetime is
    module-scoped to amortise the cost across tests in this file.

    Gracefully skips when sentence-transformers / numpy / rapidfuzz aren't
    installed — useful for fast-path test runs that don't pull ML deps."""
    try:
        from src.services.event_classifier import EventClassifier
        return EventClassifier()
    except ImportError as exc:
        pytest.skip(f"EventClassifier dependencies not installed: {exc}")


class TestDataminrHeadlineClassification:
    """End-to-end: the offending Dataminr signal classifies as conflict,
    not severe storm."""

    def test_does_not_classify_as_severe_storm(self, classifier):
        """The whole point of the fix. Even if the top label isn't 'armed
        clash' exactly (semantic ambiguity is acceptable for a one-line
        title with no description), it must not be a natural-hazard storm."""
        result = classifier.predict(DATAMINR_HEADLINE, top_k=3)
        top = result["top_k"][0]
        assert top["type_level_3"] != "severe local storm", (
            f"Headline classified as severe storm — pre-fix bug regression.\n"
            f"  top result: {top}\n"
            f"  full top-3: {result['top_k']}"
        )
        # Stronger: the top label shouldn't even be in natural-hazard land.
        assert top["type_level_1"] != "natural hazard", (
            f"Headline classified as a natural hazard ({top['type_level_3']}), "
            f"but the text is about military conflict.\n  top result: {top}"
        )

    def test_classifies_as_conflict_level_1(self, classifier):
        """Positive assertion: military conflict headlines should land in
        the 'conflict' top-level category."""
        result = classifier.predict(DATAMINR_HEADLINE, top_k=3)
        top = result["top_k"][0]
        assert top["type_level_1"] == "conflict", (
            f"Expected level_1='conflict' for a military headline; "
            f"got {top['type_level_1']!r} ({top['type_level_3']!r}).\n"
            f"  full top-3: {result['top_k']}"
        )

    def test_local_news_outlet_suffix_alone_is_not_storm(self, classifier):
        """The minimal reproducer: just the attribution suffix on its own
        must not classify as severe storm. If this ever passes pre-fix and
        fails post-fix, the regression has reintroduced the 'local'
        keyword somewhere."""
        result = classifier.predict("Local News Outlet via Reuters", top_k=3)
        top = result["top_k"][0]
        assert top["type_level_3"] != "severe local storm"
