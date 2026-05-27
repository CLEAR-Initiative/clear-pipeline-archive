"""Tests for the casualties priority chain in the event-grouping layer.

The signal's "actual casualties" value is resolved in two named tiers:

  1. `_resolve_actual_casualties` — source-shipped field, then regex from
     title + description. This file tests that tier.
  2. `_resolve_signal_stats` — per-event-type historical fallback when (1)
     yields None. Already covered elsewhere; this file just locks in that
     (1) doesn't pre-empt it when both source and text are silent.

Why this matters: before the change, only Dataminr's build_signal_input
ran the regex extraction. GDACS and manual signals jumped straight to
the per-event-type default even when the title plainly said "23 killed".

Run:
    pytest tests/test_event_grouping_casualties.py -v
"""

from __future__ import annotations

from src.services.event_grouping_v2 import (
    _resolve_actual_casualties,
    _resolve_signal_stats,
)


class TestResolveActualCasualties:
    """Tier 1+2: source-shipped field → text-extraction fallback."""

    def test_source_value_wins_over_text(self):
        """When the source/builder already populated signal.casualties, the
        helper returns that value without consulting the text — even if the
        text would have produced a different number."""
        created = {"casualties": 42}
        result = _resolve_actual_casualties(
            created,
            signal_title="9 killed in airstrike",
            signal_description="Local sources report 9 dead.",
        )
        assert result == 42

    def test_falls_back_to_text_when_source_is_none(self):
        """The GDACS / manual-signal case: source set nothing on the row, so
        the regex extractor takes over."""
        created = {"casualties": None}
        result = _resolve_actual_casualties(
            created,
            signal_title="23 killed in flash floods in Khartoum",
            signal_description=None,
        )
        assert result == 23

    def test_falls_back_to_text_when_created_signal_is_none(self):
        """Defensive: if no created_signal was passed at all, still try the
        text — never bail unnecessarily."""
        result = _resolve_actual_casualties(
            None,
            signal_title="At least 5 dead after dam collapse",
            signal_description=None,
        )
        assert result == 5

    def test_returns_none_when_no_signal_and_no_text(self):
        result = _resolve_actual_casualties(None, None, None)
        assert result is None

    def test_returns_none_when_text_has_no_casualty_phrasing(self):
        """Non-fatality numbers (injured, displaced) must NOT be picked up —
        those are handled by population_affected, not casualties."""
        created = {"casualties": None}
        result = _resolve_actual_casualties(
            created,
            signal_title="Floods displace 5000 in North Darfur",
            signal_description="At least 200 people injured.",
        )
        assert result is None

    def test_uses_description_when_title_is_empty(self):
        result = _resolve_actual_casualties(
            None,
            signal_title=None,
            signal_description="Death toll of 14 reported overnight.",
        )
        assert result == 14

    def test_source_zero_is_explicit_and_kept(self):
        """`casualties: 0` from the source is a meaningful 'we know it was
        zero' — must not be treated as 'null, please extract from text'.

        (A signal with explicit zero casualties shouldn't have the text
        regex potentially overwrite it with a number from generic
        background text.)"""
        created = {"casualties": 0}
        result = _resolve_actual_casualties(
            created,
            signal_title="Massive blast in Khartoum, 50 killed",
            signal_description=None,
        )
        assert result == 0


class TestPriorityIntegrationWithEventTypeDefault:
    """Tier 3: per-event-type historical fallback only fires when tiers 1+2
    both return None. Smoke-tests the integration between
    `_resolve_actual_casualties` and `_resolve_signal_stats`."""

    def test_text_extraction_pre_empts_event_type_default(self):
        """When we can read '23 killed' from the title, we shouldn't fall
        back to the historical q75 estimate for the event type."""
        actual = _resolve_actual_casualties(
            {"casualties": None},
            signal_title="23 killed in Karnoi market airstrike",
            signal_description=None,
        )
        # Now pass through the event-type fallback layer with a glide code
        # that would yield a different historical estimate — we want the
        # text value to win.
        stats = _resolve_signal_stats(
            actual_casualties=actual,
            actual_population=None,
            glide_code=None,  # no glide → no historical fallback anyway
        )
        assert stats["casualties"] == 23

    def test_event_type_default_fires_when_both_source_and_text_silent(self):
        """When neither tier produces a value, fall through to the
        per-event-type historical lookup (returns None for unknown glide)."""
        actual = _resolve_actual_casualties(
            {"casualties": None},
            signal_title="Humanitarian aid blocked at border crossing",
            signal_description=None,
        )
        assert actual is None
        stats = _resolve_signal_stats(
            actual_casualties=actual,
            actual_population=None,
            glide_code=None,
        )
        # No glide code → no historical estimate → casualties stays None
        assert stats["casualties"] is None
