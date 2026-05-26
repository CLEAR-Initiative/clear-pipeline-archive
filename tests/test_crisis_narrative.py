"""Tests for the crisis narrative generation path.

What this pins down:

  1. `CrisisNarrative` accepts the new `{title, description, tldr}` shape
     and rejects the old `{title, summary}` shape — so a stale Claude
     prompt response can't sneak through silently.
  2. `_generate_narrative` returns the summary as a JSON-serialised string
     of `{description, tldr}` — the crises.summary column stays a string;
     UI consumers JSON.parse it on read.
  3. The prompt template asks for the structured shape so Claude actually
     produces it.

Run:
    pytest tests/test_crisis_narrative.py -v
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from src.models.clear import CrisisNarrative
from src.prompts.crisis import USER_PROMPT_TEMPLATE, build_crisis_prompt
from src.tasks.crisis import _generate_narrative


class TestCrisisNarrativeModel:
    """Pydantic-level shape checks. Cheap and load-bearing — any Claude
    response that's missing the new fields will fail validation here."""

    def test_accepts_new_shape(self):
        result = CrisisNarrative.model_validate({
            "title": "Floods in North Darfur",
            "description": "Heavy seasonal rains have flooded several towns in "
                           "North Darfur, displacing residents and damaging crops.",
            "tldr": [
                "Seasonal floods hit North Darfur towns.",
                "Roughly 5,000 residents displaced, croplands destroyed.",
                "Risk of waterborne disease and food shortages rising.",
            ],
        })
        assert result.title == "Floods in North Darfur"
        assert result.description.startswith("Heavy seasonal rains")
        assert len(result.tldr) == 3

    def test_rejects_old_summary_shape(self):
        """The old `{title, summary}` shape must not validate — otherwise a
        prompt that's been left on an old worker would silently store the
        wrong format."""
        with pytest.raises(Exception):
            CrisisNarrative.model_validate({
                "title": "Floods in North Darfur",
                "summary": "Some narrative text.",
            })

    def test_rejects_missing_description(self):
        with pytest.raises(Exception):
            CrisisNarrative.model_validate({
                "title": "Floods",
                "tldr": ["a", "b", "c"],
            })

    def test_rejects_missing_tldr(self):
        with pytest.raises(Exception):
            CrisisNarrative.model_validate({
                "title": "Floods",
                "description": "A narrative paragraph.",
            })


class TestPromptShape:
    """The prompt must instruct Claude to emit the new schema. If the
    template ever drifts back to the flat `summary` shape, the pipeline
    will produce CrisisNarrative validation errors at runtime — better to
    catch it here in tests."""

    def test_template_requests_description_and_tldr(self):
        assert "description" in USER_PROMPT_TEMPLATE
        assert "tldr" in USER_PROMPT_TEMPLATE
        # The old field name must NOT be the structural response key any more.
        # (We still mention "summary" in prose for context but never as the JSON key.)
        assert '"summary"' not in USER_PROMPT_TEMPLATE

    def test_build_crisis_prompt_includes_events_and_locations(self):
        events = [
            {"title": "Flood in El Fasher", "description": "", "types": ["flood"],
             "severity": 4, "populationAffected": 2000},
        ]
        prompt = build_crisis_prompt(events, ["North Darfur", "Khartoum"])
        assert "Flood in El Fasher" in prompt
        assert "North Darfur" in prompt
        assert "Khartoum" in prompt
        # Prompt embeds the JSON template structurally
        assert '"tldr"' in prompt
        assert '"description"' in prompt


class TestGenerateNarrative:
    """End-to-end through `_generate_narrative` with Claude mocked. Pins
    that the summary lands as a JSON-serialised string of {description, tldr}."""

    _EVENT_FIXTURE = [
        {
            "title": "Heavy rain causes flooding",
            "description": "Northern districts inundated.",
            "types": ["flood"],
            "severity": 4,
            "populationAffected": 12000,
            "generalLocation": {"name": "North Darfur"},
        },
    ]

    _CLAUDE_RESPONSE = {
        "title": "Floods in North Darfur",
        "description": "Heavy seasonal rains have flooded several towns in "
                       "North Darfur, displacing residents and damaging crops.",
        "tldr": [
            "Seasonal floods hit North Darfur towns.",
            "Roughly 12,000 residents displaced, croplands destroyed.",
            "Risk of waterborne disease and food shortages rising.",
        ],
    }

    def test_returns_title_and_serialised_summary(self):
        with patch("src.tasks.crisis.call_claude", return_value=self._CLAUDE_RESPONSE):
            result = _generate_narrative(self._EVENT_FIXTURE)

        assert result is not None
        title, summary = result
        assert title == "Floods in North Darfur"
        # `summary` is a JSON-serialised string of {description, tldr}.
        parsed = json.loads(summary)
        assert set(parsed.keys()) == {"description", "tldr"}
        assert parsed["description"] == self._CLAUDE_RESPONSE["description"]
        assert parsed["tldr"] == self._CLAUDE_RESPONSE["tldr"]

    def test_summary_is_valid_json(self):
        """The serialised summary must always round-trip through json.loads —
        otherwise frontends/consumers JSON.parse will throw."""
        with patch("src.tasks.crisis.call_claude", return_value=self._CLAUDE_RESPONSE):
            _, summary = _generate_narrative(self._EVENT_FIXTURE)  # type: ignore[misc]
        # No exception means it's valid JSON.
        json.loads(summary)

    def test_no_unicode_escaping_of_non_ascii(self):
        """Sudan place names and Arabic transliterations should pass through
        as-is, not get \\u-escaped (we use ensure_ascii=False)."""
        response = dict(self._CLAUDE_RESPONSE)
        response["description"] = "Floods reported in Al-Geneina and El Fasher."
        with patch("src.tasks.crisis.call_claude", return_value=response):
            _, summary = _generate_narrative(self._EVENT_FIXTURE)  # type: ignore[misc]
        assert "Al-Geneina" in summary
        assert "\\u" not in summary

    def test_returns_none_on_validation_failure(self):
        """If Claude returns the old `{title, summary}` shape, the model
        rejects it and `_generate_narrative` returns None (best-effort —
        the crisis just won't get a narrative this round)."""
        old_shape = {"title": "Floods", "summary": "Plain text."}
        with patch("src.tasks.crisis.call_claude", return_value=old_shape):
            result = _generate_narrative(self._EVENT_FIXTURE)
        assert result is None

    def test_returns_none_for_empty_event_list(self):
        """Sanity: don't call Claude for an empty event list."""
        with patch("src.tasks.crisis.call_claude") as mock_claude:
            result = _generate_narrative([])
        assert result is None
        mock_claude.assert_not_called()
