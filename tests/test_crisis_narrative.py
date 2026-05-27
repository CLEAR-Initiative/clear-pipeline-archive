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

from src.models.clear import (
    CrisisNarrative,
    CrisisNeedsClarification,
    CrisisScenarios,
)
from src.prompts.crisis import (
    CLARIFICATION_USER_PROMPT_TEMPLATE,
    SCENARIOS_USER_PROMPT_TEMPLATE,
    USER_PROMPT_TEMPLATE,
    build_clarification_prompt,
    build_crisis_prompt,
    build_scenarios_prompt,
)
from src.tasks.crisis import (
    _generate_narrative,
    _generate_needs_clarification,
    _generate_scenarios,
)


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


# ─── Scenarios ─────────────────────────────────────────────────────────────


class TestCrisisScenariosModel:
    """Pydantic-level shape checks for the forward-scenarios payload that
    lands on crises.scenarios (JSONB)."""

    _VALID = {
        "most_likely": "Conflict intensifies in North Darfur over the next 4-6 weeks.",
        "best_case": "Ceasefire agreement reduces violence and reopens humanitarian access.",
        "worst_case": "Escalation cuts off Al Fasher entirely; food and medical supplies collapse.",
        "description": "Security dynamics are deteriorating, rainy season is approaching, "
                       "humanitarian access remains restricted; funding shortfalls limit response capacity.",
    }

    def test_accepts_full_payload(self):
        s = CrisisScenarios.model_validate(self._VALID)
        assert s.most_likely.startswith("Conflict")
        assert s.description.startswith("Security")

    @pytest.mark.parametrize("missing_field", [
        "most_likely", "best_case", "worst_case", "description",
    ])
    def test_rejects_missing_field(self, missing_field: str):
        bad = {k: v for k, v in self._VALID.items() if k != missing_field}
        with pytest.raises(Exception):
            CrisisScenarios.model_validate(bad)


class TestScenariosPromptShape:
    """Lock in the JSON schema the scenarios prompt asks Claude for."""

    def test_template_requests_all_four_fields(self):
        for field in ("most_likely", "best_case", "worst_case", "description"):
            assert f'"{field}"' in SCENARIOS_USER_PROMPT_TEMPLATE, (
                f"Scenarios prompt template missing JSON field {field!r}"
            )

    def test_template_covers_scenario_variables(self):
        """The prompt must list the scenario variables (per the spec) so
        Claude knows what to consider when writing `description`."""
        keywords = [
            "Political", "Economic", "Environmental",
            "public health", "Policy", "access",
            "Response capacity", "Population movements",
        ]
        for kw in keywords:
            assert kw.lower() in SCENARIOS_USER_PROMPT_TEMPLATE.lower(), (
                f"Scenarios prompt missing variable {kw!r}"
            )

    def test_build_scenarios_prompt_embeds_events_and_locations(self):
        events = [
            {"title": "Flood in El Fasher", "description": "Northern districts flooded.",
             "types": ["flood"], "severity": 4, "populationAffected": 5000},
        ]
        prompt = build_scenarios_prompt(events, ["North Darfur", "Khartoum"])
        assert "Flood in El Fasher" in prompt
        assert "North Darfur" in prompt
        assert "Khartoum" in prompt
        # Same structured JSON template fields appear in the rendered prompt.
        for field in ("most_likely", "best_case", "worst_case", "description"):
            assert f'"{field}"' in prompt


class TestGenerateScenarios:
    """End-to-end through `_generate_scenarios` with Claude mocked. Pins
    that the returned dict has the JSONB-ready shape callers can pass
    straight to the GraphQL mutation."""

    _EVENT_FIXTURE = [
        {
            "title": "Sustained airstrikes in Al Fasher",
            "description": "Civilians targeted across multiple districts.",
            "types": ["conflict"],
            "severity": 5,
            "populationAffected": 50000,
            "generalLocation": {"name": "North Darfur"},
        },
    ]

    _CLAUDE_RESPONSE = {
        "most_likely": "Conflict intensifies; humanitarian access deteriorates over 4-6 weeks.",
        "best_case": "Local truce holds; convoys restart limited deliveries to Al Fasher.",
        "worst_case": "Full siege of Al Fasher; mass starvation and disease outbreaks.",
        "description": "Security trending worse, response capacity strained, "
                       "rainy season approaching, funding gaps widening.",
    }

    def test_returns_dict_with_all_four_fields(self):
        with patch("src.tasks.crisis.call_claude", return_value=self._CLAUDE_RESPONSE):
            result = _generate_scenarios(self._EVENT_FIXTURE)

        assert result is not None
        assert set(result.keys()) == {
            "most_likely", "best_case", "worst_case", "description",
        }
        assert result["most_likely"].startswith("Conflict")
        assert result["worst_case"].startswith("Full siege")

    def test_result_is_jsonable(self):
        """The dict must be directly serialisable for the GraphQL JSON
        scalar — no Pydantic objects leaking through."""
        import json as _json
        with patch("src.tasks.crisis.call_claude", return_value=self._CLAUDE_RESPONSE):
            result = _generate_scenarios(self._EVENT_FIXTURE)
        # No exception means it's plain dict + strings.
        _json.dumps(result)

    def test_returns_none_on_validation_failure(self):
        """If Claude returns a malformed shape (missing field), we return
        None — scenarios are best-effort enrichment, the crisis still gets
        title/summary/populationInArea from the rest of the task."""
        bad = {
            "most_likely": "...",
            "best_case": "...",
            # missing worst_case + description
        }
        with patch("src.tasks.crisis.call_claude", return_value=bad):
            result = _generate_scenarios(self._EVENT_FIXTURE)
        assert result is None

    def test_returns_none_for_empty_event_list(self):
        """Sanity: don't call Claude for an empty event list."""
        with patch("src.tasks.crisis.call_claude") as mock_claude:
            result = _generate_scenarios([])
        assert result is None
        mock_claude.assert_not_called()

    def test_uses_scenarios_prompt_version(self):
        """Make sure the call is tagged with the scenarios-specific prompt
        version so insights telemetry can distinguish the two stages."""
        with patch("src.tasks.crisis.call_claude", return_value=self._CLAUDE_RESPONSE) as mock_claude:
            _generate_scenarios(self._EVENT_FIXTURE)
        _, kwargs = mock_claude.call_args
        assert kwargs.get("stage") == "crisis-scenarios"
        assert kwargs.get("prompt_version") == "crisis-scenarios-v1"


# ─── Needs clarification (NRC SAF) ─────────────────────────────────────────


class TestCrisisNeedsClarificationModel:
    """Pydantic-level shape checks for the clarification payload that lands
    inside `crises.needs.clarification`."""

    def test_accepts_clarification_string(self):
        text = (
            "- Severity — Severe overall, driven by food security and protection.\n"
            "- Drivers — Displacement has severed access to markets.\n"
            "- Response gaps — Education cluster absent in 3W.\n"
            "- Priority action — RNA-first to verify school closures."
        )
        result = CrisisNeedsClarification.model_validate({"clarification": text})
        assert result.clarification.startswith("- Severity")

    def test_rejects_missing_field(self):
        with pytest.raises(Exception):
            CrisisNeedsClarification.model_validate({"other": "foo"})


class TestClarificationPromptShape:
    """Pin the SAF prompt template so any drift away from the spec gets
    caught here rather than in production output."""

    def test_template_includes_saf_severity_scale(self):
        for level in (
            "Minimal", "Stressed", "Severe", "Extreme", "Catastrophic",
        ):
            assert level in CLARIFICATION_USER_PROMPT_TEMPLATE, (
                f"SAF severity level {level!r} missing from clarification prompt"
            )

    def test_template_includes_four_bullet_labels(self):
        for label in (
            "Severity", "Drivers", "Response gaps", "Priority action",
        ):
            assert label in CLARIFICATION_USER_PROMPT_TEMPLATE

    def test_template_requests_dash_prefixed_bullets(self):
        """The spec explicitly says 'each starting with a dash (-)'."""
        assert "starting with a dash" in CLARIFICATION_USER_PROMPT_TEMPLATE

    def test_template_flags_data_age_consideration(self):
        """The SAF prompt must remind the model that the MSNA data is
        8 months old so confidence is downgraded when warranted."""
        assert "8 months" in CLARIFICATION_USER_PROMPT_TEMPLATE

    def test_build_clarification_prompt_embeds_events_and_locality_data(self):
        events = [
            {
                "title": "Flood in El Fasher", "description": "Northern flooded.",
                "types": ["flood"], "severity": 4, "populationAffected": 5000,
                "generalLocation": {
                    "name": "North Darfur",
                    "metadata": [
                        {"type": "MSNA", "data": {"food_insecure_pct": 42}},
                        {"type": "OCHA-3W", "data": {"clusters": ["WASH", "Health"]}},
                    ],
                },
            },
        ]
        prompt = build_clarification_prompt(events, ["North Darfur"])
        assert "Flood in El Fasher" in prompt
        assert "North Darfur" in prompt
        # Locality metadata flows into the prompt verbatim so the LLM can
        # cite actual indicator values.
        assert "MSNA" in prompt
        assert "OCHA-3W" in prompt
        assert "food_insecure_pct" in prompt

    def test_build_clarification_prompt_flags_missing_metadata(self):
        """When no location metadata is available, the prompt block must
        say so explicitly so the model downgrades confidence instead of
        hallucinating indicator values."""
        events = [
            {
                "title": "Event without locality data", "description": "",
                "types": ["conflict"], "severity": 3,
                "generalLocation": {"name": "Unknown District"},
            },
        ]
        prompt = build_clarification_prompt(events, ["Unknown District"])
        assert "no MSNA" in prompt or "locality metadata available" in prompt


class TestGenerateNeedsClarification:
    """End-to-end through `_generate_needs_clarification` with Claude
    mocked. Pins the return value (plain string) and the telemetry tags."""

    _EVENT_FIXTURE = [
        {
            "title": "Sustained airstrikes in Al Fasher",
            "description": "Civilians targeted across multiple districts.",
            "types": ["conflict"],
            "severity": 5,
            "populationAffected": 50000,
            "generalLocation": {
                "name": "North Darfur",
                "metadata": [
                    {"type": "MSNA", "data": {"food_insecure_pct": 58}},
                ],
            },
        },
    ]

    _CLAUDE_RESPONSE = {
        "clarification": (
            "- Severity — Severe overall, driven by food security and protection; "
            "confidence Medium given 8-month MSNA age.\n"
            "- Drivers — Displacement has severed access to markets, compounding "
            "pre-existing food insecurity.\n"
            "- Response gaps — Education cluster absent in 3W; NRC has core "
            "competency.\n"
            "- Priority action — Assessment-first (RNA) to verify school closure "
            "type and current water quality."
        ),
    }

    def test_returns_clarification_string(self):
        with patch("src.tasks.crisis.call_claude", return_value=self._CLAUDE_RESPONSE):
            result = _generate_needs_clarification(self._EVENT_FIXTURE)

        assert isinstance(result, str)
        assert result.startswith("- Severity")
        assert result.count("\n- ") == 3  # four bullets total

    def test_returns_none_on_validation_failure(self):
        """Claude returning a malformed shape (missing `clarification`) →
        None. Best-effort enrichment, no exception bubbles up."""
        with patch("src.tasks.crisis.call_claude", return_value={"other": "x"}):
            result = _generate_needs_clarification(self._EVENT_FIXTURE)
        assert result is None

    def test_returns_none_for_empty_event_list(self):
        with patch("src.tasks.crisis.call_claude") as mock_claude:
            result = _generate_needs_clarification([])
        assert result is None
        mock_claude.assert_not_called()

    def test_uses_clarification_prompt_version(self):
        """Telemetry tag must distinguish this stage from narrative/scenarios."""
        with patch("src.tasks.crisis.call_claude", return_value=self._CLAUDE_RESPONSE) as mock_claude:
            _generate_needs_clarification(self._EVENT_FIXTURE)
        _, kwargs = mock_claude.call_args
        assert kwargs.get("stage") == "crisis-clarification"
        assert kwargs.get("prompt_version") == "crisis-clarification-v1"
