"""Tests for the signal geoparser.

The geoparser has four independently-testable layers:

  1. Extraction      — does the regex pull the right candidates out?
  2. Classification  — is a candidate correctly tagged landmark vs admin?
  3. Disqualification — does "transferred to X" correctly demote X?
  4. Ranking + resolution — does the right candidate end up as the result?

Each layer is exercised separately. The full `geoparse_signal` integration
is tested with `src.clients.nominatim.search` mocked, so the suite is
hermetic — no network, no Redis, no GraphQL.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.services import geoparser as gp


# ─── Extraction ───────────────────────────────────────────────────────────


class TestExtraction:
    """Regex-based candidate extraction. Stopwords are filtered in this
    layer so the ranking layer never sees `Sudanese` / `RSF` etc."""

    def test_after_in_preposition(self):
        candidates = gp._extract_from_text("Armed clash in Al Fasher", "title")
        names = [c.name for c in candidates]
        assert "Al Fasher" in names

    def test_after_near_preposition(self):
        candidates = gp._extract_from_text("Fire hotspots detected near Markib", "title")
        assert any(c.name == "Markib" for c in candidates)

    def test_comma_separated_takes_leading_part(self):
        candidates = gp._extract_from_text(
            "Drone strikes reported in Al Fasher, North Darfur, Sudan", "title",
        )
        # Both prepositional ("in Al Fasher") and comma-lead patterns
        # should hit; the resulting candidate set contains "Al Fasher".
        names = {c.name for c in candidates}
        assert "Al Fasher" in names

    def test_stopwords_filtered(self):
        # "Sudanese" matches the capitalisation pattern but is in
        # ORG_STOPWORDS — must not appear as a candidate.
        candidates = gp._extract_from_text(
            "Sudanese Armed Forces engage RSF in Al Fasher", "title",
        )
        names = {c.name.lower() for c in candidates}
        assert "sudanese" not in names
        assert "rsf" not in names
        assert "armed forces" not in names

    def test_landmark_extracted(self):
        candidates = gp._extract_from_text("Strike at Nyala Airport", "title")
        assert any(c.name == "Nyala Airport" for c in candidates)

    def test_empty_text_returns_no_candidates(self):
        assert gp._extract_from_text("", "title") == []
        assert gp._extract_from_text("   ", "title") == []

    def test_no_recognisable_places(self):
        # Whole sentence has no capitalised proper nouns matching our
        # patterns — return empty rather than guessing.
        candidates = gp._extract_from_text("the situation worsened today", "title")
        assert candidates == []

    def test_dedupes_same_name_in_same_field(self):
        # "Khartoum" appearing twice in one field collapses to one candidate,
        # at the earliest position.
        candidates = gp._extract_from_text(
            "Reports in Khartoum say more strikes near Khartoum", "title",
        )
        khartoum = [c for c in candidates if c.name == "Khartoum"]
        assert len(khartoum) == 1


# ─── Classification ───────────────────────────────────────────────────────


class TestLandmarkDetection:
    """Suffix-based landmark vs admin classification."""

    @pytest.mark.parametrize("name", [
        "Nyala Airport",
        "Al Fasher Hospital",
        "Kosti Bridge",
        "Wad Madani University",
        "Port Sudan Port",
        "El Geneina IDP Camp",
        "Khartoum International Airport",
    ])
    def test_landmark_names_detected(self, name):
        assert gp._is_landmark(name) is True

    @pytest.mark.parametrize("name", [
        "Al Fasher",
        "Khartoum",
        "North Darfur",
        "Wad Madani",
    ])
    def test_admin_names_not_flagged(self, name):
        assert gp._is_landmark(name) is False

    def test_case_insensitive(self):
        assert gp._is_landmark("nyala airport") is True
        assert gp._is_landmark("NYALA AIRPORT") is True

    def test_partial_word_not_a_match(self):
        # "Khairport" must not match the "airport" suffix (no leading whitespace).
        assert gp._is_landmark("Khairport") is False


# ─── Disqualification ─────────────────────────────────────────────────────


class TestDisqualification:
    """Phrase-based filtering of incidental mentions."""

    def test_transferred_to_demotes(self):
        # The conservative extraction patterns deliberately don't pull a
        # candidate out of "transferred to Khartoum" — there's no positive
        # locator preposition. Test the disqualifier helper in isolation
        # against a manually-built candidate at the right position, which
        # mirrors how the helper will be used when other extraction passes
        # (or future NER) do hand it a candidate from this kind of phrase.
        text = "Clashes erupted; wounded transferred to Khartoum"
        c = gp.Candidate(name="Khartoum", field="body", position=text.index("Khartoum"))
        assert gp._is_disqualified(c, text) is True

    def test_fled_toward_demotes(self):
        text = "Civilians fled toward Mellit"
        candidates = gp._extract_from_text(text, "body")
        # The "near"/"in"/"at" patterns won't match here, but the candidate
        # may not be extracted at all by these conservative patterns.
        # Build a candidate manually to test the disqualifier in isolation.
        manual = gp.Candidate(name="Mellit", field="body", position=text.index("Mellit"))
        assert gp._is_disqualified(manual, text) is True

    def test_plain_in_does_not_disqualify(self):
        # "in" is a positive locator, not a disqualifier.
        text = "Armed clash in Al Fasher"
        candidate = next(c for c in gp._extract_from_text(text, "title") if c.name == "Al Fasher")
        assert gp._is_disqualified(candidate, text) is False

    def test_position_zero_never_disqualified(self):
        # A candidate at the very start of the text has nothing preceding it.
        text = "Al Fasher hit by strike"
        c = gp.Candidate(name="Al Fasher", field="title", position=0)
        assert gp._is_disqualified(c, text) is False


# ─── Ranking ──────────────────────────────────────────────────────────────


class TestRanking:
    """Score composition: kind > field > position; disqualified drops to floor."""

    def test_landmark_beats_admin_in_same_field(self):
        admin = gp.Candidate(name="Nyala", field="title", position=20, kind="admin")
        landmark = gp.Candidate(name="Nyala Airport", field="title", position=20, kind="landmark")
        ranked = gp._rank([admin, landmark])
        assert ranked[0] is landmark

    def test_title_beats_body_at_same_position(self):
        title_hit = gp.Candidate(name="Al Fasher", field="title", position=10, kind="admin")
        body_hit  = gp.Candidate(name="Al Fasher", field="body",  position=10, kind="admin")
        ranked = gp._rank([title_hit, body_hit])
        assert ranked[0] is title_hit

    def test_earlier_position_beats_later_in_same_field(self):
        early = gp.Candidate(name="Al Fasher", field="title", position=5,  kind="admin")
        late  = gp.Candidate(name="Khartoum",  field="title", position=50, kind="admin")
        ranked = gp._rank([early, late])
        assert ranked[0] is early

    def test_disqualified_falls_to_bottom_even_if_landmark(self):
        # A disqualified landmark in the title still ranks below a clean admin
        # candidate elsewhere — the 0.1 penalty dominates.
        disqualified_landmark = gp.Candidate(
            name="Khartoum Hospital", field="title", position=20,
            kind="landmark", disqualified=True,
        )
        clean_admin = gp.Candidate(
            name="Al Fasher", field="title", position=10, kind="admin",
        )
        ranked = gp._rank([disqualified_landmark, clean_admin])
        assert ranked[0] is clean_admin


# ─── Nominatim result selection ───────────────────────────────────────────


class TestResultSelection:
    """The filter that picks one Nominatim hit from a ranked list."""

    def _result(self, *, importance: float, country: str = "sd", **kwargs) -> dict:
        return {
            "importance": importance,
            "address": {"country_code": country},
            "lat": "12.0",
            "lon": "33.0",
            **kwargs,
        }

    def test_picks_highest_importance_in_expected_country(self):
        results = [
            self._result(importance=0.6, country="sd"),
            self._result(importance=0.8, country="sd"),
            self._result(importance=0.95, country="us"),  # filtered out by country
        ]
        best = gp._pick_best_nominatim_result(results, {"sd"})
        assert best is results[1]

    def test_low_importance_results_still_accepted(self):
        """LocationIQ scores for Sudan-region places routinely fall below 0.3
        (e.g., 'El Obeid Teaching Hospital' at 0.0001). We deliberately do
        not impose an importance floor — Nominatim already orders by
        relevance, and dropping low-importance results means losing valid
        matches in low-density OSM regions."""
        weak = self._result(importance=0.0001)
        assert gp._pick_best_nominatim_result([weak], {"sd"}) is weak

    def test_rejects_linear_feature_classes(self):
        """Highways/railways/waterways are linear, not point — promoting one
        to an A4 attributes signals to an arbitrary point on a road."""
        road = self._result(importance=0.6, **{"class": "highway"})
        place = self._result(importance=0.1, **{"class": "place"})
        # Even though the highway has higher importance, the place wins
        # because the highway class is rejected outright.
        assert gp._pick_best_nominatim_result([road, place], {"sd"}) is place

    def test_returns_none_when_only_rejected_classes(self):
        rail = self._result(importance=0.7, **{"class": "railway"})
        water = self._result(importance=0.5, **{"class": "waterway"})
        assert gp._pick_best_nominatim_result([rail, water], {"sd"}) is None

    def test_empty_input_returns_none(self):
        assert gp._pick_best_nominatim_result([], {"sd"}) is None

    def test_unparseable_importance_skipped(self):
        bad = self._result(importance="oops")  # type: ignore[arg-type]
        good = self._result(importance=0.5)
        assert gp._pick_best_nominatim_result([bad, good], {"sd"}) is good


# ─── End-to-end ────────────────────────────────────────────────────────────


class TestGeoparseSignal:
    """Full `geoparse_signal` flow with the Nominatim client mocked."""

    def _nominatim_hit(self, *, lat="13.6", lon="25.3", importance=0.65,
                       cls="place", typ="city", country="sd",
                       display_name="Al Fasher, North Darfur, Sudan") -> dict:
        return {
            "lat": lat, "lon": lon, "importance": importance,
            "class": cls, "type": typ,
            "display_name": display_name,
            "address": {"country_code": country},
        }

    def test_resolves_admin_match(self):
        with patch.object(gp.nominatim, "search", return_value=[self._nominatim_hit()]):
            result = gp.geoparse_signal("Armed clash in Al Fasher")
        assert result is not None
        assert result.candidate == "Al Fasher"
        assert result.kind == "admin"
        assert result.field == "title"
        assert result.lat == pytest.approx(13.6)
        assert result.lng == pytest.approx(25.3)
        assert result.country_code == "sd"

    def test_resolves_landmark_over_admin(self):
        # Title contains both "Nyala" (admin) and "Nyala Airport" (landmark) —
        # the landmark should win.
        airport = self._nominatim_hit(
            lat="12.06", lon="24.97", importance=0.55,
            cls="aeroway", typ="aerodrome",
            display_name="Nyala Airport, South Darfur, Sudan",
        )
        with patch.object(gp.nominatim, "search", return_value=[airport]) as mock_search:
            result = gp.geoparse_signal("Drone strike at Nyala Airport reported in Nyala")
        assert result is not None
        assert result.candidate == "Nyala Airport"
        assert result.kind == "landmark"
        assert result.osm_class == "aeroway"
        # Only one call to nominatim — we never tried other candidates after the
        # top one resolved.
        assert mock_search.call_count == 1

    def test_returns_none_when_nominatim_empty(self):
        with patch.object(gp.nominatim, "search", return_value=None):
            assert gp.geoparse_signal("Armed clash in Al Fasher") is None

    def test_low_importance_hits_now_resolve(self):
        """A low-importance hit for a real Sudan place (e.g. a teaching
        hospital at importance ~0.0001) should still resolve. This locks in
        the removal of the old 0.3 importance floor — that floor was silently
        dropping the bulk of valid matches in low-density OSM regions."""
        weak = self._nominatim_hit(importance=0.0001, cls="amenity")
        with patch.object(gp.nominatim, "search", return_value=[weak]):
            result = gp.geoparse_signal("Armed clash in Al Fasher")
        assert result is not None
        assert result.importance == pytest.approx(0.0001)

    def test_returns_none_when_only_rejected_class_results(self):
        """A signal that only resolves to a highway/railway/waterway should
        produce no geoparser hit — those classes aren't point locations."""
        road = self._nominatim_hit(importance=0.6, cls="highway", typ="primary")
        with patch.object(gp.nominatim, "search", return_value=[road]):
            assert gp.geoparse_signal("Armed clash in Al Fasher") is None

    def test_returns_none_when_no_candidates_extracted(self):
        # No capitalised proper nouns matching our patterns.
        with patch.object(gp.nominatim, "search") as mock_search:
            assert gp.geoparse_signal("the situation worsened") is None
        # We must not have called Nominatim at all — extraction failed
        # before resolution.
        mock_search.assert_not_called()

    def test_does_not_use_disqualified_top_candidate(self):
        # When the only candidate is disqualified, the function bails
        # rather than calling Nominatim with bad data. (Note: testing the
        # bail by ensuring nominatim isn't called.)
        text = "Wounded transferred to Khartoum"
        with patch.object(gp.nominatim, "search") as mock_search:
            result = gp.geoparse_signal(text)
        # Either we found no candidate (our patterns won't match this text
        # because "in/at/near/around/outside" aren't present and there's no
        # comma list either) — or we found one and it was disqualified.
        # Both paths result in None and no Nominatim call.
        assert result is None
        mock_search.assert_not_called()

    def test_passes_country_codes_to_nominatim(self):
        with patch.object(gp.nominatim, "search", return_value=[self._nominatim_hit()]) as mock_search:
            gp.geoparse_signal("Armed clash in Al Fasher")
        # By default, Sudan-focused.
        _, kwargs = mock_search.call_args
        assert kwargs.get("country_codes") == "sd"

    def test_accepts_custom_country_codes(self):
        hit = self._nominatim_hit(country="ng")
        with patch.object(gp.nominatim, "search", return_value=[hit]) as mock_search:
            result = gp.geoparse_signal(
                "Armed clash in Maiduguri", expected_country_codes={"ng"},
            )
        _, kwargs = mock_search.call_args
        assert kwargs.get("country_codes") == "ng"
        assert result is not None
        assert result.country_code == "ng"
