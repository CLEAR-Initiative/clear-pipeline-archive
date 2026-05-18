"""Tests for the auto-escalation gates added to `process_manual_signal`.

The task fans out an alert (via `escalate_event` → `escalateEvent` mutation,
which fires emails) **only when all three guards pass**:

  1. `source_type` is one of TRUSTED_SOURCE_NAMES
     (field_officer / partner / government)
  2. classification severity is >= 4
  3. the signal's publishedAt is not stale (within
     `alert_max_signal_age_hours`)

Pre-gate behaviour: trusted-source signals auto-escalated regardless of
severity or age — every record-keeping entry from a field officer triggered
the email fan-out. The two new gates bring the manual path in line with
the Dataminr/GDACS/ACLED paths.

These tests mock every external dependency (Claude, GraphQL, the local
classifier) so the suite is hermetic — no network, no DB.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from src.models.clear import SignalClassification
from src.tasks.process import process_manual_signal


# ─── helpers ───────────────────────────────────────────────────────────────


def _iso(ts: datetime) -> str:
    return ts.isoformat()


def _classification(severity: int) -> SignalClassification:
    return SignalClassification(
        disaster_types=["fl"],
        relevance=0.8,
        severity=severity,
        summary="test summary",
    )


# Stable kwargs the gate logic doesn't depend on. Tests override the
# severity/source/publishedAt fields they care about.
_BASE_KWARGS = {
    "signal_id": "sig_test",
    "title": "Test signal",
    "description": "Test description",
    "user_id": "user_test",
}


# ─── fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def fresh_published_at() -> str:
    """Within the 48h window — must not trip the staleness gate."""
    return _iso(datetime.now(UTC) - timedelta(minutes=5))


@pytest.fixture
def stale_published_at() -> str:
    """Past the 48h window — must trip the staleness gate."""
    return _iso(datetime.now(UTC) - timedelta(days=7))


@pytest.fixture
def mocks(monkeypatch):
    """Patch every external collaborator the task uses. Each test then asserts
    on the `escalate` mock to verify whether the gate let the call through.
    Defaults:
      - classify_locally returns a severity-4 classification (just at the
        threshold; tests override per-case).
      - dispatch_group_signal returns a stable event id.
      - escalate_event is the assert target — never called unless the gates
        pass.
    """
    classify_mock = MagicMock(return_value=_classification(4))
    update_severity_mock = MagicMock()
    dispatch_mock = MagicMock(return_value={"id": "evt_test"})
    escalate_mock = MagicMock(return_value={"id": "esc_test"})

    monkeypatch.setattr("src.tasks.process.classify_locally", classify_mock)
    monkeypatch.setattr("src.tasks.process.update_signal_severity", update_severity_mock)
    monkeypatch.setattr("src.tasks.process.dispatch_group_signal", dispatch_mock)
    monkeypatch.setattr("src.tasks.process.escalate_event", escalate_mock)

    # Force the v2 classification path so classify_locally is consulted and
    # the test doesn't accidentally land in the Claude-calling branch.
    monkeypatch.setattr("src.tasks.process.settings.grouping_algo", "v2")
    monkeypatch.setattr("src.tasks.process.settings.alert_max_signal_age_hours", 48)

    return {
        "classify": classify_mock,
        "dispatch": dispatch_mock,
        "escalate": escalate_mock,
        "update_severity": update_severity_mock,
    }


def _invoke(**overrides):
    """Run the task synchronously with default + override kwargs."""
    kwargs = {**_BASE_KWARGS, **overrides}
    process_manual_signal.apply(kwargs=kwargs).get(disable_sync_subtasks=False)


# ─── gate behaviour ────────────────────────────────────────────────────────


class TestAutoEscalateGates:
    """End-to-end gate assertions on the Stage 3 auto-escalate block.

    Pattern: customise the classifier severity / kwargs, invoke the task,
    then check whether `escalate_event` was called.
    """

    def test_trusted_high_sev_fresh_escalates(self, mocks, fresh_published_at):
        mocks["classify"].return_value = _classification(5)
        _invoke(source_type="field_officer", severity=5, signal_published_at=fresh_published_at)
        mocks["escalate"].assert_called_once_with("evt_test", "user_test")

    def test_non_trusted_source_skips(self, mocks, fresh_published_at):
        # Only field_officer / partner / government count as trusted. A
        # generic source name must NOT escalate even at severity 5.
        mocks["classify"].return_value = _classification(5)
        _invoke(source_type="user_report", severity=5, signal_published_at=fresh_published_at)
        mocks["escalate"].assert_not_called()

    def test_low_severity_skips(self, mocks, fresh_published_at):
        # severity = 3 < 4 → severity gate fires.
        mocks["classify"].return_value = _classification(3)
        _invoke(source_type="field_officer", severity=3, signal_published_at=fresh_published_at)
        mocks["escalate"].assert_not_called()

    def test_boundary_severity_4_escalates(self, mocks, fresh_published_at):
        # Exactly at the threshold — must escalate (mirrors the auto-poll paths).
        mocks["classify"].return_value = _classification(4)
        _invoke(source_type="partner", severity=4, signal_published_at=fresh_published_at)
        mocks["escalate"].assert_called_once_with("evt_test", "user_test")

    def test_stale_published_at_skips(self, mocks, stale_published_at):
        # >48h old — staleness gate fires even at severity 5.
        mocks["classify"].return_value = _classification(5)
        _invoke(source_type="government", severity=5, signal_published_at=stale_published_at)
        mocks["escalate"].assert_not_called()

    def test_missing_published_at_does_not_block(self, mocks):
        # publishedAt = None → is_stale_signal returns False → gate passes.
        # Older callers / backfills without the kwarg shouldn't lose their
        # ability to alert.
        mocks["classify"].return_value = _classification(5)
        _invoke(source_type="field_officer", severity=5, signal_published_at=None)
        mocks["escalate"].assert_called_once_with("evt_test", "user_test")

    def test_threshold_zero_disables_staleness_gate(self, mocks, monkeypatch, stale_published_at):
        # Operator escape hatch: setting the threshold to 0 turns off the
        # staleness gate entirely.
        monkeypatch.setattr("src.tasks.process.settings.alert_max_signal_age_hours", 0)
        mocks["classify"].return_value = _classification(5)
        _invoke(source_type="field_officer", severity=5, signal_published_at=stale_published_at)
        mocks["escalate"].assert_called_once_with("evt_test", "user_test")

    def test_event_grouping_failure_blocks_escalation(self, mocks, fresh_published_at):
        # No event → nothing to escalate. The task returns early before the
        # gate block; escalate_event must not be touched.
        mocks["dispatch"].return_value = None
        mocks["classify"].return_value = _classification(5)
        _invoke(source_type="field_officer", severity=5, signal_published_at=fresh_published_at)
        mocks["escalate"].assert_not_called()
