"""Shared pytest fixtures for clear-pipeline.

Keeps the per-test boilerplate down by giving every test a clean copy of the
critical settings flags via monkeypatch and a tiny event-shaped dict to pass
into escalation helpers.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def event() -> dict:
    """Minimal event shape — escalation helpers only read `event["id"]`."""
    return {"id": "evt_test_123"}
