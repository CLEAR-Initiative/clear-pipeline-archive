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

import anthropic
import httpx
import pytest

from src.models.clear import (
    NEEDS_SECTORS,
    CrisisNarrative,
    CrisisNeedsAnalysis,
    CrisisScenarios,
)
from src.prompts.crisis import (
    NEEDS_ANALYSIS_SYSTEM_PROMPT,
    NEEDS_ANALYSIS_USER_PROMPT_TEMPLATE,
    SCENARIOS_USER_PROMPT_TEMPLATE,
    USER_PROMPT_TEMPLATE,
    build_crisis_prompt,
    build_needs_analysis_prompt,
    build_scenarios_prompt,
)
from src.clients.claude import ClaudeRateLimited
from src.tasks.crisis import (
    _generate_narrative,
    _generate_needs_analysis,
    _generate_scenarios,
)


def _fake_overloaded_error() -> anthropic.APIStatusError:
    """Construct a real 529-shaped APIStatusError the way the SDK raises
    one (the production 529 surfaces as `anthropic._exceptions.OverloadedError`,
    a private subclass of the public `APIStatusError` — we catch the base
    class in production, so constructing the base class here is enough)."""
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    resp = httpx.Response(status_code=529, request=req)
    return anthropic.APIStatusError(
        "Overloaded", response=resp, body={"type": "error"},
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

    def test_re_raises_transient_anthropic_errors(self):
        """Anthropic 5xx / 529 Overloaded / rate-limit errors must NOT be
        swallowed as None — they bubble to `enrich_crisis` so Celery can
        retry the whole task instead of producing a crisis with no
        narrative. This is the bug the 529 incident exposed."""
        with patch("src.tasks.crisis.call_claude", side_effect=_fake_overloaded_error()):
            with pytest.raises(anthropic.APIStatusError):
                _generate_narrative(self._EVENT_FIXTURE)

    def test_re_raises_claude_rate_limited(self):
        """`ClaudeRateLimited` (wrapped 429) must also bubble so Celery
        applies the retry_after backoff, not swallowed."""
        rl = ClaudeRateLimited("Rate limited", retry_after=30.0)
        with patch("src.tasks.crisis.call_claude", side_effect=rl):
            with pytest.raises(ClaudeRateLimited):
                _generate_narrative(self._EVENT_FIXTURE)

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

    def test_re_raises_transient_anthropic_errors(self):
        """Anthropic API errors bubble so Celery retries — see the
        same test on _generate_narrative for full rationale."""
        with patch("src.tasks.crisis.call_claude", side_effect=_fake_overloaded_error()):
            with pytest.raises(anthropic.APIStatusError):
                _generate_scenarios(self._EVENT_FIXTURE)

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


# ─── Needs analysis (NRC SAF) ──────────────────────────────────────────────


class TestCrisisNeedsAnalysisModel:
    """Pydantic-level shape checks for the analysis payload that lands
    under `crises.needs.{generalSummary, sector}`. The `sector` field is
    keyed by canonical NRC sector names (NEEDS_SECTORS); unknown keys are
    rejected so the LLM can't introduce hallucinated sector names."""

    _FULL_SECTOR_PAYLOAD = {
        "generalSummary": [
            "Severe — food security and protection drive; confidence Medium (8-month MSNA).",
            "Displacement has severed market access, compounding pre-existing food insecurity.",
            "Education cluster absent in 3W; NRC has core competency.",
            "Priority action: assessment-first (RNA) pending school-closure verification.",
        ],
        "sector": {
            "Shelter": {
                "description": "Severe — 40% of HHs report inadequate shelter.",
                "severity": "Severe",
                "responseGap": False,
                "nrcRelevant": True,
            },
            "WASH": {
                "description": "Stressed — water trucking covers most needs.",
                "severity": "Stressed",
                "responseGap": False,
                "nrcRelevant": True,
            },
            "Protection": {
                "description": "Extreme — civilian targeting reported.",
                "severity": "Extreme",
                "responseGap": True,
                "nrcRelevant": False,
            },
            "Health": {
                "description": "Severe — clinics damaged, supply chain disrupted.",
                "severity": "Severe",
                "responseGap": True,
                "nrcRelevant": False,
            },
            "Food Security": {
                "description": "Severe — 58% food insecure per MSNA.",
                "severity": "Severe",
                "responseGap": False,
                "nrcRelevant": False,
            },
            "Education": {
                "description": "Minimal — schools out of session; not driving the crisis.",
                "severity": "Minimal",
                "responseGap": True,
                "nrcRelevant": True,
            },
        },
    }

    def test_accepts_all_six_sectors(self):
        result = CrisisNeedsAnalysis.model_validate(self._FULL_SECTOR_PAYLOAD)
        assert len(result.generalSummary) == 4
        # Brevity invariant for the fixture: every bullet ≤ 25 words.
        # Drift in either direction (looser fixture, tighter cap) should
        # show up here before it shows up in production.
        for bullet in result.generalSummary:
            assert len(bullet.split()) <= 25, (
                f"Fixture bullet exceeds 25-word brevity cap: {bullet!r}"
            )
        assert set(result.sector.keys()) == set(NEEDS_SECTORS)
        assert result.sector["Food Security"].description.startswith("Severe")

    def test_accepts_partial_sector_subset(self):
        """LLM may legitimately produce only the sectors that matter for a
        crisis — partial subsets of NEEDS_SECTORS are accepted."""
        payload = {
            "generalSummary": ["bullet"],
            "sector": {
                "Food Security": {
                    "description": "Severe.", "severity": "Severe",
                    "responseGap": False, "nrcRelevant": False,
                },
                "Protection": {
                    "description": "Extreme.", "severity": "Extreme",
                    "responseGap": True, "nrcRelevant": False,
                },
            },
        }
        result = CrisisNeedsAnalysis.model_validate(payload)
        assert set(result.sector.keys()) == {"Food Security", "Protection"}

    def test_rejects_unknown_sector_key(self):
        """A sector name outside NEEDS_SECTORS is a hallucination — reject it."""
        payload = {
            "generalSummary": ["bullet"],
            "sector": {
                "Shelter": {
                    "description": "...", "severity": "Severe",
                    "responseGap": False, "nrcRelevant": True,
                },
                "Logistics": {
                    "description": "Not a canonical NRC sector",
                    "severity": "Minimal", "responseGap": False, "nrcRelevant": False,
                },
            },
        }
        with pytest.raises(Exception):
            CrisisNeedsAnalysis.model_validate(payload)

    def test_sector_entry_accepts_extra_fields(self):
        """Beyond the four required fields, sector entries accept additional
        content (indicator percentages, cluster actors, recommendation) so
        the schema can grow without a Pydantic change."""
        payload = {
            "generalSummary": ["bullet"],
            "sector": {
                "Food Security": {
                    "description": "Severe — 58% food insecure.",
                    "severity": "Severe",
                    "responseGap": False,
                    "nrcRelevant": True,
                    "actors": ["WFP", "FAO"],
                    "recommendation": "Stabilisation response",
                },
            },
        }
        result = CrisisNeedsAnalysis.model_validate(payload)
        # Extras flow through via model_dump for downstream JSON storage.
        dumped = result.model_dump()
        assert dumped["sector"]["Food Security"]["actors"] == ["WFP", "FAO"]
        assert dumped["sector"]["Food Security"]["recommendation"] == "Stabilisation response"

    @pytest.mark.parametrize(
        "missing_field",
        ["description", "severity", "responseGap", "nrcRelevant"],
    )
    def test_rejects_sector_entry_missing_required_field(self, missing_field: str):
        full = {
            "description": "...",
            "severity": "Severe",
            "responseGap": False,
            "nrcRelevant": True,
        }
        partial = {k: v for k, v in full.items() if k != missing_field}
        payload = {"generalSummary": ["bullet"], "sector": {"Shelter": partial}}
        with pytest.raises(Exception):
            CrisisNeedsAnalysis.model_validate(payload)

    @pytest.mark.parametrize(
        "level",
        ["Minimal", "Stressed", "Severe", "Extreme", "Catastrophic"],
    )
    def test_accepts_all_saf_severity_levels(self, level: str):
        payload = {
            "generalSummary": ["bullet"],
            "sector": {
                "Shelter": {
                    "description": "...",
                    "severity": level,
                    "responseGap": False,
                    "nrcRelevant": True,
                },
            },
        }
        result = CrisisNeedsAnalysis.model_validate(payload)
        assert result.sector["Shelter"].severity == level

    def test_rejects_invalid_severity_level(self):
        """Casing/spelling matters — the Literal type is strict so the
        prompt's exact strings are the only accepted values."""
        payload = {
            "generalSummary": ["bullet"],
            "sector": {
                "Shelter": {
                    "description": "...",
                    "severity": "severe",  # lower-case — rejected
                    "responseGap": False,
                    "nrcRelevant": True,
                },
            },
        }
        with pytest.raises(Exception):
            CrisisNeedsAnalysis.model_validate(payload)

        payload["sector"]["Shelter"]["severity"] = "Critical"  # not a SAF level
        with pytest.raises(Exception):
            CrisisNeedsAnalysis.model_validate(payload)

    @pytest.mark.parametrize("missing", ["generalSummary", "sector"])
    def test_rejects_missing_top_level_field(self, missing: str):
        bad = {k: v for k, v in self._FULL_SECTOR_PAYLOAD.items() if k != missing}
        with pytest.raises(Exception):
            CrisisNeedsAnalysis.model_validate(bad)

    def test_general_summary_is_list_of_strings(self):
        """generalSummary must be a list, not a single string. The prompt
        asks for 4 bullets and the new shape is array-shaped."""
        bad = {**self._FULL_SECTOR_PAYLOAD, "generalSummary": "single string"}
        with pytest.raises(Exception):
            CrisisNeedsAnalysis.model_validate(bad)

    def test_general_summary_rejects_empty_list(self):
        bad = {**self._FULL_SECTOR_PAYLOAD, "generalSummary": []}
        with pytest.raises(Exception):
            CrisisNeedsAnalysis.model_validate(bad)

    def test_general_summary_rejects_blank_entries(self):
        """Non-empty list of NON-EMPTY strings — whitespace-only entries fail."""
        bad = {
            **self._FULL_SECTOR_PAYLOAD,
            "generalSummary": ["valid", "   "],
        }
        with pytest.raises(Exception):
            CrisisNeedsAnalysis.model_validate(bad)

    def test_general_summary_rejects_non_string_entries(self):
        bad = {**self._FULL_SECTOR_PAYLOAD, "generalSummary": ["valid", 42]}
        with pytest.raises(Exception):
            CrisisNeedsAnalysis.model_validate(bad)

    @pytest.mark.parametrize("count", [1, 3, 4, 5, 6])
    def test_general_summary_tolerates_bullet_count_drift(self, count: int):
        """The prompt asks for exactly 4, but Claude occasionally drifts.
        We accept anything ≥1 so a 3-or-5 response still produces useful
        output rather than nothing."""
        payload = {
            **self._FULL_SECTOR_PAYLOAD,
            "generalSummary": [f"bullet {i + 1}" for i in range(count)],
        }
        result = CrisisNeedsAnalysis.model_validate(payload)
        assert len(result.generalSummary) == count


class TestNeedsAnalysisPromptShape:
    """Pin the SAF prompt template so any drift away from the spec gets
    caught here rather than in production output."""

    def test_template_includes_saf_severity_scale(self):
        for level in (
            "Minimal", "Stressed", "Severe", "Extreme", "Catastrophic",
        ):
            assert level in NEEDS_ANALYSIS_USER_PROMPT_TEMPLATE, (
                f"SAF severity level {level!r} missing from needs analysis prompt"
            )

    def test_template_asks_for_general_summary_and_sector(self):
        """Both top-level output keys must appear in the JSON schema block."""
        assert '"generalSummary"' in NEEDS_ANALYSIS_USER_PROMPT_TEMPLATE
        assert '"sector"' in NEEDS_ANALYSIS_USER_PROMPT_TEMPLATE

    def test_template_asks_for_exactly_four_bullets(self):
        """The prompt must specify the bullet count explicitly so Claude
        produces a 4-element array on `generalSummary`. If this drifts,
        consumers expecting 4 bullets will see ragged counts."""
        assert "EXACTLY 4 bullet points" in NEEDS_ANALYSIS_USER_PROMPT_TEMPLATE

    def test_template_caps_bullet_length(self):
        """The prompt must enforce per-bullet brevity. Without an explicit
        cap, Claude routinely produces 3-4 line bullets that don't render
        well on responder UIs. The cap is concrete (≤25 words) rather than
        the vague 'single sentence' phrasing that drifted in production."""
        # The exact-cap token + the rationale phrase ('brevity matters')
        # together pin the constraint. Either alone is fragile to small
        # prompt edits.
        assert "≤25 words" in NEEDS_ANALYSIS_USER_PROMPT_TEMPLATE
        assert "Brevity matters" in NEEDS_ANALYSIS_USER_PROMPT_TEMPLATE

    def test_template_shows_general_summary_as_json_array(self):
        """The JSON schema example must render generalSummary as an array,
        not a single string. Catches a silent regression where the example
        block falls back to the single-paragraph form."""
        # The array opening `[` must follow the `"generalSummary":` key.
        # Find the index and check the next non-whitespace token is `[`.
        idx = NEEDS_ANALYSIS_USER_PROMPT_TEMPLATE.find('"generalSummary":')
        assert idx != -1
        tail = NEEDS_ANALYSIS_USER_PROMPT_TEMPLATE[idx + len('"generalSummary":'):].lstrip()
        assert tail.startswith("["), (
            "JSON schema example shows generalSummary as a non-array; "
            "Claude will return a string and the Pydantic list[str] validator "
            "will reject every response."
        )
        assert '"description"' in NEEDS_ANALYSIS_USER_PROMPT_TEMPLATE

    def test_template_lists_all_six_sectors(self):
        """The prompt must list every canonical sector verbatim so the LLM
        produces matching JSON keys (and the validator accepts the result)."""
        for sector in NEEDS_SECTORS:
            assert sector in NEEDS_ANALYSIS_USER_PROMPT_TEMPLATE, (
                f"Sector {sector!r} missing from needs analysis prompt"
            )

    def test_template_uses_sectors_as_json_keys(self):
        """The JSON schema block in the prompt should use the exact sector
        strings as object keys (with the quoted form) so the LLM gets the
        casing/spelling right."""
        for sector in NEEDS_SECTORS:
            assert f'"{sector}"' in NEEDS_ANALYSIS_USER_PROMPT_TEMPLATE

    def test_template_requests_all_per_sector_fields(self):
        """Every required field on SectorAnalysis must be named in the
        prompt — if the prompt drifts away from any of them the LLM
        responses will start failing Pydantic validation."""
        for field in ("description", "severity", "responseGap", "nrcRelevant"):
            assert f'"{field}"' in NEEDS_ANALYSIS_USER_PROMPT_TEMPLATE, (
                f"Per-sector field {field!r} missing from the prompt JSON schema"
            )

    def test_template_covers_saf_dimensions(self):
        """The two SAF dimensions the analyst is asked to apply must both
        be referenced in the prompt."""
        assert "Dimension 6" in NEEDS_ANALYSIS_SYSTEM_PROMPT or "Dimension 6" in NEEDS_ANALYSIS_USER_PROMPT_TEMPLATE
        assert "Dimension 7" in NEEDS_ANALYSIS_SYSTEM_PROMPT or "Dimension 7" in NEEDS_ANALYSIS_USER_PROMPT_TEMPLATE

    def test_template_flags_data_age_consideration(self):
        """The SAF prompt must remind the model that the MSNA data is
        8 months old so confidence is downgraded when warranted."""
        assert "8 months" in (
            NEEDS_ANALYSIS_USER_PROMPT_TEMPLATE + NEEDS_ANALYSIS_SYSTEM_PROMPT
        )

    def test_build_needs_analysis_prompt_embeds_events_and_locality_data(self):
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
        prompt = build_needs_analysis_prompt(events, ["North Darfur"])
        assert "Flood in El Fasher" in prompt
        assert "North Darfur" in prompt
        # Locality metadata flows into the prompt verbatim so the LLM can
        # cite actual indicator values.
        assert "MSNA" in prompt
        assert "OCHA-3W" in prompt
        assert "food_insecure_pct" in prompt

    def test_build_needs_analysis_prompt_flags_missing_metadata(self):
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
        prompt = build_needs_analysis_prompt(events, ["Unknown District"])
        assert "no MSNA" in prompt or "locality metadata available" in prompt


class TestGenerateNeedsAnalysis:
    """End-to-end through `_generate_needs_analysis` with Claude mocked.
    Pins the return dict shape and the telemetry tags."""

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
        "generalSummary": [
            "Severe — food security and protection drive; confidence Medium (8-month MSNA).",
            "Displacement has severed market access, compounding pre-existing food insecurity.",
            "Education cluster absent in 3W; NRC has core competency.",
            "Priority action: assessment-first (RNA).",
        ],
        "sector": {
            "Shelter": {
                "description": "Severe — 40% inadequate shelter.",
                "severity": "Severe", "responseGap": False, "nrcRelevant": True,
            },
            "WASH": {
                "description": "Stressed — water trucking covers most needs.",
                "severity": "Stressed", "responseGap": False, "nrcRelevant": True,
            },
            "Protection": {
                "description": "Extreme — civilian targeting reported.",
                "severity": "Extreme", "responseGap": True, "nrcRelevant": False,
            },
            "Health": {
                "description": "Severe — clinics damaged.",
                "severity": "Severe", "responseGap": True, "nrcRelevant": False,
            },
            "Food Security": {
                "description": "Severe — 58% food insecure per MSNA.",
                "severity": "Severe", "responseGap": False, "nrcRelevant": False,
            },
            "Education": {
                "description": "Stressed — schools functioning at reduced capacity.",
                "severity": "Stressed", "responseGap": True, "nrcRelevant": True,
            },
        },
    }

    def test_returns_dict_with_general_summary_and_sector(self):
        with patch("src.tasks.crisis.call_claude", return_value=self._CLAUDE_RESPONSE):
            result = _generate_needs_analysis(self._EVENT_FIXTURE)

        assert result is not None
        assert set(result.keys()) == {"generalSummary", "sector"}
        assert isinstance(result["generalSummary"], list)
        assert len(result["generalSummary"]) == 4
        assert result["generalSummary"][0].startswith("Severe")
        # Per-sector breakdown contains all six canonical sectors and every
        # required SectorAnalysis field on each entry.
        assert set(result["sector"].keys()) == set(NEEDS_SECTORS)
        for sector_name, entry in result["sector"].items():
            assert {"description", "severity", "responseGap", "nrcRelevant"} <= set(entry.keys()), (
                f"Sector {sector_name!r} missing required fields: got {sorted(entry.keys())}"
            )
        assert result["sector"]["Food Security"]["severity"] == "Severe"
        assert result["sector"]["Protection"]["responseGap"] is True

    def test_result_is_jsonable(self):
        """The dict must be directly serialisable for the GraphQL JSON
        scalar — no Pydantic objects leaking through."""
        import json as _json
        with patch("src.tasks.crisis.call_claude", return_value=self._CLAUDE_RESPONSE):
            result = _generate_needs_analysis(self._EVENT_FIXTURE)
        _json.dumps(result)

    def test_returns_none_on_validation_failure(self):
        """Claude returning a malformed shape (missing one of the top-level
        keys) → None. Best-effort enrichment, no exception bubbles up."""
        with patch(
            "src.tasks.crisis.call_claude",
            return_value={"generalSummary": ["bullet"]},  # missing `sector`
        ):
            result = _generate_needs_analysis(self._EVENT_FIXTURE)
        assert result is None

    def test_re_raises_transient_anthropic_errors(self):
        """Anthropic API errors bubble so Celery retries — see the
        same test on _generate_narrative for full rationale."""
        with patch("src.tasks.crisis.call_claude", side_effect=_fake_overloaded_error()):
            with pytest.raises(anthropic.APIStatusError):
                _generate_needs_analysis(self._EVENT_FIXTURE)

    def test_returns_none_for_empty_event_list(self):
        with patch("src.tasks.crisis.call_claude") as mock_claude:
            result = _generate_needs_analysis([])
        assert result is None
        mock_claude.assert_not_called()

    def test_uses_needs_analysis_prompt_version(self):
        """Telemetry tag must distinguish this stage from narrative/scenarios."""
        with patch("src.tasks.crisis.call_claude", return_value=self._CLAUDE_RESPONSE) as mock_claude:
            _generate_needs_analysis(self._EVENT_FIXTURE)
        _, kwargs = mock_claude.call_args
        assert kwargs.get("stage") == "crisis-needs-analysis"
        assert kwargs.get("prompt_version") == "crisis-needs-analysis-v1"
