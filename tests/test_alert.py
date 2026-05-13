"""Tests for `src.services.alert` — focused on the staleness gate added to
suppress immediate alert emails for backdated / replayed signals.

The gate matters because Dataminr (and to a lesser extent ACLED/GDACS) can
deliver an alert days after the underlying incident; without this check the
analyst gets a same-day email about a week-old event. See the
`alert_max_signal_age_hours` setting in `src.config`.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest

from src.services import alert as alert_service


# ─── helpers ───────────────────────────────────────────────────────────────


def _iso(ts: datetime) -> str:
    """Standard ISO 8601 with +00:00 offset."""
    return ts.isoformat()


def _iso_z(ts: datetime) -> str:
    """Match Dataminr's wire format (trailing Z instead of +00:00)."""
    return ts.astimezone(UTC).isoformat().replace("+00:00", "Z")


# ─── _is_stale_signal ──────────────────────────────────────────────────────


class TestIsStaleSignal:
    """The pure helper that decides whether a publishedAt is past the
    configured staleness threshold. Behaviour the gate relies on:
      - missing / unparseable timestamps fall through (we'd rather fire late
        than swallow an alert silently)
      - threshold = 0 disables the gate entirely
    """

    def test_disabled_when_threshold_is_zero(self, monkeypatch):
        monkeypatch.setattr(alert_service.settings, "alert_max_signal_age_hours", 0)
        ancient = _iso(datetime.now(UTC) - timedelta(days=365))
        assert alert_service._is_stale_signal(ancient) is False

    def test_returns_false_when_published_at_is_none(self, monkeypatch):
        monkeypatch.setattr(alert_service.settings, "alert_max_signal_age_hours", 48)
        assert alert_service._is_stale_signal(None) is False

    def test_returns_false_when_published_at_is_empty_string(self, monkeypatch):
        monkeypatch.setattr(alert_service.settings, "alert_max_signal_age_hours", 48)
        assert alert_service._is_stale_signal("") is False

    def test_returns_false_when_published_at_is_unparseable(self, monkeypatch):
        monkeypatch.setattr(alert_service.settings, "alert_max_signal_age_hours", 48)
        assert alert_service._is_stale_signal("not a timestamp") is False

    def test_fresh_signal_within_threshold(self, monkeypatch):
        monkeypatch.setattr(alert_service.settings, "alert_max_signal_age_hours", 48)
        recent = _iso(datetime.now(UTC) - timedelta(hours=12))
        assert alert_service._is_stale_signal(recent) is False

    def test_signal_just_inside_threshold_is_not_stale(self, monkeypatch):
        monkeypatch.setattr(alert_service.settings, "alert_max_signal_age_hours", 48)
        boundary = _iso(datetime.now(UTC) - timedelta(hours=47, minutes=59))
        assert alert_service._is_stale_signal(boundary) is False

    def test_signal_beyond_threshold_is_stale(self, monkeypatch):
        monkeypatch.setattr(alert_service.settings, "alert_max_signal_age_hours", 48)
        old = _iso(datetime.now(UTC) - timedelta(hours=72))
        assert alert_service._is_stale_signal(old) is True

    def test_handles_dataminr_z_suffix(self, monkeypatch):
        """Dataminr serialises timestamps with trailing Z. fromisoformat on
        Python <3.11 chokes on it, hence the explicit `.replace("Z", "+00:00")`
        in the helper. Lock in the behaviour with a regression test."""
        monkeypatch.setattr(alert_service.settings, "alert_max_signal_age_hours", 48)
        week_old = _iso_z(datetime.now(UTC) - timedelta(days=7))
        assert alert_service._is_stale_signal(week_old) is True

    def test_future_timestamps_are_not_stale(self, monkeypatch):
        """A future publishedAt yields a negative age — must not be classed
        as stale. Should never happen in practice but defending against
        clock-skew edge cases is cheap."""
        monkeypatch.setattr(alert_service.settings, "alert_max_signal_age_hours", 48)
        tomorrow = _iso(datetime.now(UTC) + timedelta(hours=1))
        assert alert_service._is_stale_signal(tomorrow) is False


# ─── maybe_escalate ────────────────────────────────────────────────────────


class TestMaybeEscalate:
    """End-to-end gate tests. We mock the two downstream escalators so the
    test stays hermetic — no Claude / GraphQL calls. The contract under test:
      - stale → short-circuit (no escalator invoked, returns None)
      - fresh + v2 → escalate_to_alert
      - fresh + v1 → assess_and_escalate
      - signal_published_at omitted → backwards-compat (no gate)
    """

    def test_stale_signal_skips_escalation(self, monkeypatch, event):
        monkeypatch.setattr(alert_service.settings, "alert_max_signal_age_hours", 48)
        monkeypatch.setattr(alert_service.settings, "grouping_algo", "v2")
        stale = _iso(datetime.now(UTC) - timedelta(days=7))

        with (
            patch.object(alert_service, "escalate_to_alert") as mock_escalate,
            patch.object(alert_service, "assess_and_escalate") as mock_assess,
        ):
            result = alert_service.maybe_escalate(
                event=event,
                signal_summaries=["test"],
                max_severity=5,
                signal_published_at=stale,
            )

        assert result is None
        mock_escalate.assert_not_called()
        mock_assess.assert_not_called()

    def test_v2_fresh_signal_calls_escalate_to_alert(self, monkeypatch, event):
        monkeypatch.setattr(alert_service.settings, "alert_max_signal_age_hours", 48)
        monkeypatch.setattr(alert_service.settings, "grouping_algo", "v2")
        recent = _iso(datetime.now(UTC) - timedelta(hours=1))
        sentinel_alert = {"id": "alert_v2", "status": "published"}

        with patch.object(
            alert_service, "escalate_to_alert", return_value=sentinel_alert
        ) as mock_escalate:
            result = alert_service.maybe_escalate(
                event=event,
                signal_summaries=["recent"],
                max_severity=5,
                signal_published_at=recent,
            )

        mock_escalate.assert_called_once_with(event)
        assert result is sentinel_alert

    def test_v1_fresh_signal_calls_assess_and_escalate(self, monkeypatch, event):
        monkeypatch.setattr(alert_service.settings, "alert_max_signal_age_hours", 48)
        monkeypatch.setattr(alert_service.settings, "grouping_algo", "v1")
        recent = _iso(datetime.now(UTC) - timedelta(hours=1))
        sentinel_alert = {"id": "alert_v1", "status": "draft"}

        with patch.object(
            alert_service, "assess_and_escalate", return_value=sentinel_alert
        ) as mock_assess:
            result = alert_service.maybe_escalate(
                event=event,
                signal_summaries=["summary"],
                max_severity=4,
                signal_published_at=recent,
            )

        mock_assess.assert_called_once_with(
            event=event,
            signal_summaries=["summary"],
            max_severity=4,
        )
        assert result is sentinel_alert

    def test_omitted_published_at_does_not_gate(self, monkeypatch, event):
        """`signal_published_at` defaults to None so older callers (or sources
        without a timestamp) still escalate."""
        monkeypatch.setattr(alert_service.settings, "alert_max_signal_age_hours", 48)
        monkeypatch.setattr(alert_service.settings, "grouping_algo", "v2")
        sentinel_alert = {"id": "alert_no_ts"}

        with patch.object(
            alert_service, "escalate_to_alert", return_value=sentinel_alert
        ) as mock_escalate:
            result = alert_service.maybe_escalate(
                event=event,
                signal_summaries=[],
                max_severity=5,
            )

        mock_escalate.assert_called_once()
        assert result is sentinel_alert

    def test_threshold_zero_disables_gate_even_for_old_signals(self, monkeypatch, event):
        """Operator escape hatch: setting alert_max_signal_age_hours=0 turns
        off the gate. Useful for backfill jobs replaying historical signals."""
        monkeypatch.setattr(alert_service.settings, "alert_max_signal_age_hours", 0)
        monkeypatch.setattr(alert_service.settings, "grouping_algo", "v2")
        very_old = _iso(datetime.now(UTC) - timedelta(days=365))

        with patch.object(alert_service, "escalate_to_alert") as mock_escalate:
            alert_service.maybe_escalate(
                event=event,
                signal_summaries=["historical"],
                max_severity=5,
                signal_published_at=very_old,
            )

        mock_escalate.assert_called_once_with(event)
