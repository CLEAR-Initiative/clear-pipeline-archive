"""End-to-end tests for geoparser wiring in build_signal_input.

These tests originally surfaced two bugs:

  Bug 1 (feeder, build_signal_input):
      When `subHeadline` was null the resulting description was None, so the
      geoparser only saw the headline. The liveBrief and intelAgents content
      — where the landmark is actually mentioned — was never fed in.
      Fixed by passing `extra_body_text` (built from liveBrief +
      intelAgents) into `enrich_with_geoparser`.

  Bug 2 (geoparser regex):
      The preposition pattern `(in|at|near|around|outside)` missed "on",
      which the liveBrief text uses ("drone strike ON Nyala Airport").
      Fixed by adding "on" to `_PREP_PATTERN` and adding day-of-week + month
      names to GENERIC_STOPWORDS so "on Monday" / "on May 19" don't fire.

The tests below verify both fixes hold end-to-end against the actual
Dataminr payload that triggered the investigation.

Run:
    pytest tests/test_signal_geoparser.py -v
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.models.dataminr import DataminrSignal
from src.services import geoparser as gp
from src.services.signal import build_signal_input


# A real-world Dataminr payload supplied for diagnosis.
SAMPLE_PAYLOAD: dict = {
    "alertId": "62466149480725616088-1779254502000-1",
    "headline": "Explosions reported during overnight drone strike in Nyala, Sudan: Local Source via Facebook.",
    "alertType": {"name": "Urgent"},
    "subHeadline": None,
    "liveBrief": [
        {
            "summary": (
                "The Sudanese Armed Forces have reportedly launched a drone "
                "strike on Nyala Airport in Sudan, causing three large "
                "explosions in the city. A cargo plane was targeted during "
                "the attack, resulting in power outages in different parts "
                "and a large plume of smoke. Fires have reportedly broken "
                "out in Nyala."
            )
        }
    ],
    "intelAgents": [
        {
            "summary": [
                {
                    "title": "Event Location",
                    "content": [
                        "The incident occurred at Nyala Airport and its "
                        "surroundings in the capital of South Darfur state, Sudan."
                    ],
                },
                {
                    "title": "Property Damage",
                    "content": [
                        "Sudanese Armed Forces airstrikes caused complete "
                        "destruction of Rapid Support Forces weapons depots, "
                        "ammunition, drone equipment, and several combat "
                        "vehicles at Nyala Airport and surrounding areas."
                    ],
                },
            ]
        }
    ],
    "alertTimestamp": "2026-05-20T05:36:47.194Z",
    "estimatedEventLocation": {
        "name": "Nyala, Sudan",
        "coordinates": [12.0518011, 24.8804853],
        "probabilityRadius": 9.212916104291082,
    },
}


def _nominatim_airport_hit() -> dict:
    """Fake LocationIQ/Nominatim hit for 'Nyala Airport'."""
    return {
        "lat": "12.0537",
        "lon": "24.9543",
        "importance": 0.55,
        "class": "aeroway",
        "type": "aerodrome",
        "display_name": "Nyala Airport, South Darfur, Sudan",
        "address": {"country_code": "sd"},
    }


class TestBug1FeederFix:
    """build_signal_input must feed liveBrief + intelAgents content to the
    geoparser via `extra_body_text` — not just the headline + (possibly
    null) subHeadline-based description."""

    def test_geoparser_is_called(self):
        """Sanity check: the geoparser fires for a Dataminr signal."""
        signal = DataminrSignal.model_validate(SAMPLE_PAYLOAD)

        with patch("src.services.signal.geoparse_signal", return_value=None) as mock_gp, \
             patch("src.services.signal.find_or_create_landmark_l4") as mock_promo:
            build_signal_input(signal, source_id="src_test")

        assert mock_gp.called, "geoparse_signal was never called"
        mock_promo.assert_not_called()

    def test_geoparser_receives_livebrief_and_intel_agents_text(self):
        """The combined body text passed to the geoparser must contain the
        liveBrief summary and the intelAgents 'Event Location' content."""
        signal = DataminrSignal.model_validate(SAMPLE_PAYLOAD)

        captured: dict = {}

        def capture(title, description=None, **kwargs):
            captured["title"] = title
            captured["description"] = description
            return None

        with patch("src.services.signal.geoparse_signal", side_effect=capture), \
             patch("src.services.signal.find_or_create_landmark_l4"):
            build_signal_input(signal, source_id="src_test")

        assert captured["title"] == SAMPLE_PAYLOAD["headline"]
        body = captured["description"] or ""
        # The fix: liveBrief and intelAgents content reach the geoparser.
        assert "Nyala Airport" in body
        assert SAMPLE_PAYLOAD["liveBrief"][0]["summary"][:40] in body
        assert (
            SAMPLE_PAYLOAD["intelAgents"][0]["summary"][0]["content"][0][:40]
            in body
        )

    def test_stored_description_unchanged_by_extra_body_text(self):
        """The extra body text fed to the geoparser must NOT leak into the
        signal row's stored description — that field stays subHeadline-only
        (None for this payload)."""
        signal = DataminrSignal.model_validate(SAMPLE_PAYLOAD)

        with patch("src.services.signal.geoparse_signal", return_value=None), \
             patch("src.services.signal.find_or_create_landmark_l4"):
            input_data = build_signal_input(signal, source_id="src_test")

        # subHeadline is null → stored description stays None. The richer
        # text only goes to the geoparser, never to the DB column.
        assert input_data["description"] is None

    def test_end_to_end_resolves_to_nyala_airport_landmark(self):
        """End-to-end with the geoparser un-mocked; only Nominatim mocked.

        Post-fix behavior: the geoparser sees the headline + liveBrief +
        intelAgents text, extracts 'Nyala Airport' (via the 'at Nyala
        Airport' phrasing in intelAgents — now also via 'on Nyala Airport'
        in liveBrief thanks to Bug 2 fix), classifies as landmark, and
        resolves through Nominatim."""
        signal = DataminrSignal.model_validate(SAMPLE_PAYLOAD)

        # Mock both Nominatim AND the L4 promotion so the test stays
        # hermetic. Return a "no-op" promo result so the helper just
        # records geoparsedData and doesn't set locationId.
        with patch.object(gp.nominatim, "search", return_value=[_nominatim_airport_hit()]), \
             patch("src.services.signal.find_or_create_landmark_l4",
                   return_value={"locationId": None, "reused": False,
                                 "pointType": None, "abortedReason": None}):
            input_data = build_signal_input(signal, source_id="src_test")

        assert "geoparsedData" in input_data
        gpd = input_data["geoparsedData"]
        assert gpd["candidate"] == "Nyala Airport"
        assert gpd["kind"] == "landmark"
        assert gpd["lat"] == pytest.approx(12.0537)
        assert gpd["lng"] == pytest.approx(24.9543)
        assert gpd["country_code"] == "sd"


class TestBug2RegexFix:
    """The preposition regex must include 'on' so that
    'strike on Nyala Airport' (the liveBrief phrasing) extracts the
    landmark. Days/months are stopwords so 'on Monday' / 'on May 19' don't
    produce candidates."""

    LIVE_BRIEF_TEXT = SAMPLE_PAYLOAD["liveBrief"][0]["summary"]
    INTEL_AGENTS_TEXT = SAMPLE_PAYLOAD["intelAgents"][0]["summary"][0]["content"][0]

    def test_intel_agents_at_preposition_matches(self):
        """'at Nyala Airport' — `at` has always been in the preposition list."""
        candidates = gp._extract_from_text(self.INTEL_AGENTS_TEXT, "body")
        names = {c.name for c in candidates}
        assert "Nyala Airport" in names
        airport = next(c for c in candidates if c.name == "Nyala Airport")
        assert gp._is_landmark(airport.name)

    def test_live_brief_on_preposition_matches(self):
        """'strike on Nyala Airport' — `on` is now in the preposition list.
        This was Bug 2 before the fix."""
        candidates = gp._extract_from_text(self.LIVE_BRIEF_TEXT, "body")
        names = {c.name for c in candidates}
        assert "Nyala Airport" in names, (
            f"Expected 'Nyala Airport' to be extracted from liveBrief now "
            f"that 'on' is a recognised preposition. Got: {sorted(names)}"
        )
        airport = next(c for c in candidates if c.name == "Nyala Airport")
        assert gp._is_landmark(airport.name)

    def test_on_monday_is_not_extracted(self):
        """Day-of-week stopword guard: 'on Monday' must NOT yield 'Monday'
        as a place candidate even though the regex now includes 'on'."""
        candidates = gp._extract_from_text(
            "The strike happened on Monday morning", "body",
        )
        names = {c.name for c in candidates}
        assert "Monday" not in names

    def test_on_may_19_is_not_extracted(self):
        """Month-name stopword guard: 'on May 19' must NOT yield 'May' as
        a place candidate. (Dataminr text routinely contains date phrases
        like 'On May 19, 2026'.)"""
        candidates = gp._extract_from_text(
            "The event began on May 19 at 22:00", "body",
        )
        names = {c.name for c in candidates}
        assert "May" not in names
