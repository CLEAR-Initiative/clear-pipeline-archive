"""Tests for ground-intel classification (`src.services.ground_intel` +
`src.tasks.ground`) — the WhatsApp Signal Pipeline's intelligence stage.

All tests are hermetic: messages come from a fully SYNTHETIC fixture
(tests/fixtures/ground_messages.json — invented text modelled on the shapes
described in the PRD, no real chat content), Claude responses are canned
dicts standing in for recorded model output, and GraphQL is mocked. No
network calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "ground_messages.json"

SOURCE_ID = "gsrc_test_1"

# What a well-behaved model run says about the fixture. Stands in for a
# recorded ground_classify response.
CANNED_CLASSIFY = {
    "classifications": [
        {"id": "gm_01", "classification": "field_report"},
        {"id": "gm_02", "classification": "field_report"},
        {"id": "gm_03", "classification": "field_report"},
        {"id": "gm_04", "classification": "field_report"},
        {"id": "gm_05", "classification": "news_digest"},
        {"id": "gm_06", "classification": "operational"},
        {"id": "gm_07", "classification": "chatter"},
        {"id": "gm_08", "classification": "field_report"},
        {"id": "gm_09", "classification": "field_report"},
        {"id": "gm_10", "classification": "field_report"},
    ]
}


# What a well-behaved model run proposes for the fixture's field reports.
# Stands in for a recorded ground_thread response. The four Zalingei-style
# messages (initial report + location correction + follow-up strikes +
# media) are ONE incident; the checkpoint rumour and the compound
# report + retraction are separate incidents.
CANNED_THREADS = {
    "threads": [
        {
            "title": "Drone strikes near Zalingei",
            "lifecycle_state": "corrected",
            "message_ids": ["gm_01", "gm_02", "gm_03", "gm_04"],
        },
        {
            "title": "Reported checkpoint on the west road",
            "lifecycle_state": "reported",
            "message_ids": ["gm_08"],
        },
        {
            "title": "Reported strikes on the Galaxy compound",
            "lifecycle_state": "retracted",
            "message_ids": ["gm_09", "gm_10"],
        },
    ]
}


def canned_claude(_system: str, _user: str, *, stage: str | None = None, **_kwargs):
    """Dispatch canned responses per stage, like a recorded transcript."""
    if stage == "ground_classify":
        return CANNED_CLASSIFY
    if stage == "ground_thread":
        return CANNED_THREADS
    raise AssertionError(f"unexpected Claude stage {stage!r}")


def load_messages() -> list[dict]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def classified_messages() -> list[dict]:
    """The fixture as it looks after the classification pass wrote back."""
    by_id = {row["id"]: row["classification"] for row in CANNED_CLASSIFY["classifications"]}
    return [{**m, "classification": by_id[m["id"]]} for m in load_messages()]


# ─── classification service ────────────────────────────────────────────────


class TestClassifyMessages:
    def test_labels_every_message_with_valid_classes(self):
        from src.services import ground_intel

        with patch.object(ground_intel, "call_claude", return_value=CANNED_CLASSIFY):
            rows = ground_intel.classify_messages(load_messages())

        assert len(rows) == 10
        by_id = {r["messageId"]: r["classification"] for r in rows}
        assert by_id["gm_01"] == "field_report"
        assert by_id["gm_05"] == "news_digest"
        assert by_id["gm_06"] == "operational"
        assert by_id["gm_07"] == "chatter"

    def test_every_row_carries_the_upsert_input_shape(self):
        from src.services import ground_intel

        with patch.object(ground_intel, "call_claude", return_value=CANNED_CLASSIFY):
            rows = ground_intel.classify_messages(load_messages())

        for row in rows:
            assert set(row) == {"messageId", "classification", "uncertaintyMarker"}

    def test_invalid_label_leaves_message_unclassified(self):
        """A label outside the four-class vocabulary must NOT be written back
        — the message stays unclassified and is retried on a later run."""
        from src.services import ground_intel

        bad = {
            "classifications": [
                {"id": "gm_01", "classification": "spam"},
                {"id": "gm_02", "classification": "field_report"},
            ]
        }
        with patch.object(ground_intel, "call_claude", return_value=bad):
            rows = ground_intel.classify_messages(load_messages()[:2])

        assert [r["messageId"] for r in rows] == ["gm_02"]

    def test_missing_label_leaves_message_unclassified(self):
        from src.services import ground_intel

        partial = {"classifications": [{"id": "gm_01", "classification": "field_report"}]}
        with patch.object(ground_intel, "call_claude", return_value=partial):
            rows = ground_intel.classify_messages(load_messages()[:2])

        assert [r["messageId"] for r in rows] == ["gm_01"]

    def test_large_batches_are_chunked(self):
        from src.services import ground_intel

        many = [
            {"id": f"gm_{n:03d}", "text": "strike reported", "sentAt": "", "senderRef": "m"}
            for n in range(120)
        ]

        def canned(_system, _user, **_kwargs):
            # Echo back a valid label for every id present in the prompt
            return {
                "classifications": [
                    {"id": m["id"], "classification": "field_report"}
                    for m in many
                    if f"[{m['id']}]" in _user
                ]
            }

        with patch.object(ground_intel, "call_claude", side_effect=canned) as mock_call:
            rows = ground_intel.classify_messages(many)

        assert len(rows) == 120
        assert mock_call.call_count == 3  # 120 / CLASSIFY_CHUNK_SIZE(50)


# ─── uncertainty markers (deterministic) ───────────────────────────────────


class TestUncertaintyMarkers:
    @pytest.mark.parametrize(
        ("text", "marker"),
        [
            ("Several casualties feared. Unconfirmed so far.", "unconfirmed"),
            ("This is not yet confirmed by anyone.", "unconfirmed"),
            ("Hearing a rumour of a new checkpoint.", "rumour"),
            ("Rumors of movement to the south.", "rumour"),
            ("An unverified report of shelling.", "unverified"),
            ("The convoy was allegedly stopped en route.", "alleged"),
            ("Clashes reported near the market this morning.", None),
            ("", None),
            (None, None),
        ],
    )
    def test_detects_contributor_uncertainty_tags(self, text, marker):
        from src.services.ground_intel import detect_uncertainty_marker

        assert detect_uncertainty_marker(text) == marker

    def test_most_cautious_marker_wins_when_several_appear(self):
        """"Rumour only for now, no confirmation" carries both a rumour tag
        and an unconfirmed tag — the weaker-credibility one is preserved."""
        from src.services.ground_intel import detect_uncertainty_marker

        assert detect_uncertainty_marker("Rumour only for now, no confirmation.") == "rumour"

    def test_markers_are_preserved_in_the_classification_write_back(self):
        """PRD requirement: a source message carrying "unconfirmed" or
        "rumour" yields a derived signal that preserves that marker."""
        from src.services import ground_intel

        with patch.object(ground_intel, "call_claude", return_value=CANNED_CLASSIFY):
            rows = ground_intel.classify_messages(load_messages())

        marker_by_id = {r["messageId"]: r["uncertaintyMarker"] for r in rows}
        assert marker_by_id["gm_01"] == "unconfirmed"  # "Unconfirmed so far."
        assert marker_by_id["gm_08"] == "rumour"  # "Rumour only for now..."
        assert marker_by_id["gm_05"] is None  # news digest, no marker
        assert marker_by_id["gm_06"] is None  # operational, no marker


# ─── lifecycle states ──────────────────────────────────────────────────────


class TestLifecycleStates:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("this turned out to be misreporting", True),
            ("the earlier report was misreported", True),
            ("we retract yesterday's report", True),
            ("false alarm, all clear", True),
            ("the strike did not happen", True),
            ("Correction: it was not at the grain market", False),
            ("two more strikes heard from the same direction", False),
        ],
    )
    def test_retraction_detection(self, text, expected):
        from src.services.ground_intel import is_retraction

        assert is_retraction(text) is expected

    def test_single_message_thread_is_always_reported(self):
        from src.services.ground_intel import derive_lifecycle_state

        msg = {"text": "Strike reported near the bridge."}
        assert derive_lifecycle_state("updated", [msg]) == "reported"
        assert derive_lifecycle_state("confirmed", [msg]) == "reported"

    def test_valid_model_state_passes_through_for_multi_message_threads(self):
        from src.services.ground_intel import derive_lifecycle_state

        msgs = [{"text": "report"}, {"text": "more detail"}]
        for state in ("updated", "confirmed", "corrected"):
            assert derive_lifecycle_state(state, msgs) == state

    def test_invalid_model_state_falls_back_to_updated(self):
        from src.services.ground_intel import derive_lifecycle_state

        msgs = [{"text": "report"}, {"text": "more detail"}]
        assert derive_lifecycle_state("escalated", msgs) == "updated"
        assert derive_lifecycle_state(None, msgs) == "updated"

    def test_retraction_message_overrides_model_state(self):
        from src.services.ground_intel import derive_lifecycle_state

        msgs = [
            {"text": "Strikes reported on the compound."},
            {"text": "This turned out to be misreporting. No strikes took place."},
        ]
        # Whatever the model proposed, an explicit withdrawal wins.
        for state in ("updated", "confirmed", "corrected", None):
            assert derive_lifecycle_state(state, msgs) == "retracted"

    def test_retraction_fixture_case_sets_thread_state_retracted(self):
        """Acceptance case (the 13 Apr misreporting pattern): the compound
        report + its retraction thread ends up `retracted` even when the
        model proposed a milder state."""
        from src.services import ground_intel

        proposal = {
            "threads": [
                {
                    "title": "Reported strikes on the Galaxy compound",
                    "lifecycle_state": "corrected",  # model was too mild
                    "message_ids": ["gm_09", "gm_10"],
                }
            ]
        }
        with patch.object(ground_intel, "call_claude", return_value=proposal):
            threads = ground_intel.build_threads(SOURCE_ID, classified_messages())

        assert threads[0]["messageIds"] == ["gm_09", "gm_10"]
        assert threads[0]["lifecycleState"] == "retracted"

    def test_correction_chain_stays_corrected_not_retracted(self):
        """A location correction is a corrected thread — the incident stands.
        Only an explicit withdrawal flips to retracted."""
        from src.services import ground_intel

        with patch.object(ground_intel, "call_claude", side_effect=canned_claude):
            threads = ground_intel.build_threads(SOURCE_ID, classified_messages())

        zalingei = next(t for t in threads if "gm_01" in t["messageIds"])
        assert zalingei["lifecycleState"] == "corrected"


# ─── threading service ─────────────────────────────────────────────────────


class TestBuildThreads:
    def test_zalingei_style_incident_clusters_to_one_thread(self):
        """The reference case: initial drone-strike report, a location
        correction, related follow-up strikes, and a media message all
        cluster into a SINGLE incident thread."""
        from src.services import ground_intel

        with patch.object(ground_intel, "call_claude", side_effect=canned_claude):
            threads = ground_intel.build_threads(SOURCE_ID, classified_messages())

        zalingei = [t for t in threads if "gm_01" in t["messageIds"]]
        assert len(zalingei) == 1
        assert zalingei[0]["messageIds"] == ["gm_01", "gm_02", "gm_03", "gm_04"]
        # ...and no other thread claims any of the chain
        for t in threads:
            if t is not zalingei[0]:
                assert not set(t["messageIds"]) & {"gm_01", "gm_02", "gm_03", "gm_04"}

    def test_thread_inputs_carry_the_upsert_shape(self):
        from src.services import ground_intel

        with patch.object(ground_intel, "call_claude", side_effect=canned_claude):
            threads = ground_intel.build_threads(SOURCE_ID, classified_messages())

        assert threads  # non-empty
        for t in threads:
            assert set(t) == {"groundSourceId", "title", "lifecycleState", "messageIds"}
            assert t["groundSourceId"] == SOURCE_ID
            assert t["title"]
            assert t["messageIds"]

    def test_only_unthreaded_field_reports_are_candidates(self):
        """news_digest / operational / chatter messages and already-threaded
        field reports must never reach the threading prompt."""
        from src.services import ground_intel

        messages = classified_messages()
        for m in messages:
            if m["id"] == "gm_01":
                m["threadId"] = "gth_existing"

        seen_prompts = []

        def spy(_system, _user, **kwargs):
            seen_prompts.append(_user)
            return {"threads": []}

        with patch.object(ground_intel, "call_claude", side_effect=spy):
            ground_intel.build_threads(SOURCE_ID, messages)

        prompt = seen_prompts[0]
        for excluded in ("gm_01", "gm_05", "gm_06", "gm_07"):
            assert f"[{excluded}]" not in prompt
        for included in ("gm_02", "gm_03", "gm_04", "gm_08", "gm_09", "gm_10"):
            assert f"[{included}]" in prompt

    def test_unknown_ids_are_dropped_and_double_claims_keep_first(self):
        from src.services import ground_intel

        proposal = {
            "threads": [
                {
                    "title": "Thread A",
                    "lifecycle_state": "updated",
                    "message_ids": ["gm_01", "gm_02", "gm_999"],
                },
                {
                    "title": "Thread B",
                    "lifecycle_state": "updated",
                    "message_ids": ["gm_02", "gm_03"],
                },
            ]
        }
        with patch.object(ground_intel, "call_claude", return_value=proposal):
            threads = ground_intel.build_threads(SOURCE_ID, classified_messages())

        assert threads[0]["messageIds"] == ["gm_01", "gm_02"]
        assert threads[1]["messageIds"] == ["gm_03"]

    def test_no_candidates_means_no_model_call(self):
        from src.services import ground_intel

        chatter_only = [
            {"id": "gm_c", "text": "hi", "classification": "chatter", "threadId": None}
        ]
        with patch.object(ground_intel, "call_claude") as mock_call:
            threads = ground_intel.build_threads(SOURCE_ID, chatter_only)

        assert threads == []
        mock_call.assert_not_called()

    def test_empty_proposal_titles_get_a_fallback(self):
        from src.services import ground_intel

        proposal = {
            "threads": [
                {"title": "  ", "lifecycle_state": "updated", "message_ids": ["gm_01", "gm_02"]}
            ]
        }
        with patch.object(ground_intel, "call_claude", return_value=proposal):
            threads = ground_intel.build_threads(SOURCE_ID, classified_messages())

        assert threads[0]["title"].startswith("Reports of a drone strike")


# ─── the Celery task (mocked GraphQL transport) ────────────────────────────


class TestClassifyGroundMessagesTask:
    def _run(self, messages, claude=canned_claude):
        from src.services import ground_intel
        from src.tasks import ground as task_module

        with (
            patch.object(
                task_module,
                "ground_messages_for_classification",
                return_value=messages,
            ) as mock_query,
            patch.object(
                task_module,
                "upsert_ground_message_classifications",
                side_effect=lambda inputs: [
                    {"id": i["messageId"], "classification": i["classification"]}
                    for i in inputs
                ],
            ) as mock_upsert,
            patch.object(
                task_module,
                "upsert_ground_threads",
                side_effect=lambda inputs: [
                    {"id": f"gth_{n}"} for n in range(len(inputs))
                ],
            ) as mock_threads,
            patch.object(ground_intel, "call_claude", side_effect=claude),
        ):
            result = task_module.classify_ground_messages.apply(args=[SOURCE_ID]).get()
        return result, mock_query, mock_upsert, mock_threads

    def test_tracer_classifies_fixture_and_writes_back(self):
        """Acceptance tracer: fixture messages in → classifications written
        back through the GraphQL surface, one row per message."""
        result, mock_query, mock_upsert, _threads = self._run(load_messages())

        assert result == {
            "ground_source_id": SOURCE_ID,
            "messages_fetched": 10,
            "messages_classified": 10,
            "threads_upserted": 3,
        }
        mock_query.assert_called_once_with(SOURCE_ID, 500)

        mock_upsert.assert_called_once()
        written = mock_upsert.call_args.args[0]
        assert {w["messageId"] for w in written} == {m["id"] for m in load_messages()}
        valid = {"field_report", "news_digest", "operational", "chatter"}
        assert all(w["classification"] in valid for w in written)

    def test_task_writes_threads_through_graphql(self):
        """End-to-end within the task: freshly classified field reports are
        threaded in the same run and written via upsertGroundThreads —
        the Zalingei chain arriving as one thread."""
        _result, _query, _upsert, mock_threads = self._run(load_messages())

        mock_threads.assert_called_once()
        inputs = mock_threads.call_args.args[0]
        assert len(inputs) == 3
        zalingei = [t for t in inputs if "gm_01" in t["messageIds"]]
        assert len(zalingei) == 1
        assert zalingei[0]["messageIds"] == ["gm_01", "gm_02", "gm_03", "gm_04"]
        assert all(t["groundSourceId"] == SOURCE_ID for t in inputs)

    def test_already_classified_messages_are_not_reclassified(self):
        messages = load_messages()
        for m in messages[:8]:
            m["classification"] = "chatter"

        def claude(_system, _user, *, stage=None, **_kwargs):
            if stage == "ground_classify":
                return {
                    "classifications": [
                        {"id": "gm_09", "classification": "field_report"},
                        {"id": "gm_10", "classification": "field_report"},
                    ]
                }
            return {"threads": []}

        result, _query, mock_upsert, _threads = self._run(messages, claude=claude)

        assert result["messages_classified"] == 2
        written = mock_upsert.call_args.args[0]
        assert {w["messageId"] for w in written} == {"gm_09", "gm_10"}

    def test_no_messages_means_no_model_call_and_no_write(self):
        from src.services import ground_intel
        from src.tasks import ground as task_module

        with (
            patch.object(
                task_module, "ground_messages_for_classification", return_value=[]
            ),
            patch.object(
                task_module, "upsert_ground_message_classifications"
            ) as mock_upsert,
            patch.object(task_module, "upsert_ground_threads") as mock_threads,
            patch.object(ground_intel, "call_claude") as mock_call,
        ):
            result = task_module.classify_ground_messages.apply(args=[SOURCE_ID]).get()

        assert result["messages_fetched"] == 0
        mock_call.assert_not_called()
        mock_upsert.assert_not_called()
        mock_threads.assert_not_called()

    def test_graphql_client_error_is_not_retried(self):
        from src.clients.graphql import GraphQLClientError
        from src.tasks import ground as task_module

        with patch.object(
            task_module,
            "ground_messages_for_classification",
            side_effect=GraphQLClientError("bad request"),
        ):
            with pytest.raises(GraphQLClientError):
                task_module.classify_ground_messages.apply(
                    args=[SOURCE_ID], throw=True
                ).get()
