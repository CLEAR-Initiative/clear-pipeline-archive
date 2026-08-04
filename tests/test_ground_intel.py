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


# ─── cross-run threading (append to existing threads) ──────────────────────


def second_run_state():
    """The fixture as run 2 sees it: the Galaxy compound report (gm_09) was
    threaded in run 1; its retraction (gm_10) arrived later and is still
    un-threaded."""
    messages = classified_messages()
    for m in messages:
        if m["id"] == "gm_09":
            m["threadId"] = "gth_r1"
    existing = [
        {
            "id": "gth_r1",
            "title": "Reported strikes on the Galaxy compound",
            "lifecycleState": "reported",
            "reviewState": "pending",
            "messageIds": ["gm_09"],
        }
    ]
    return messages, existing


class TestCrossRunThreading:
    def test_existing_threads_are_offered_in_the_prompt(self):
        from src.services import ground_intel

        messages, existing = second_run_state()
        seen_prompts = []

        def spy(_system, _user, **_kwargs):
            seen_prompts.append(_user)
            return {"threads": []}

        with patch.object(ground_intel, "call_claude", side_effect=spy):
            ground_intel.build_threads(SOURCE_ID, messages, existing)

        prompt = seen_prompts[0]
        assert "Existing incident threads" in prompt
        assert "[gth_r1]" in prompt
        # The member's text is shown (it is still in the fetch window)...
        assert "Strikes reported this evening on the Galaxy compound" in prompt
        # ...but the already-threaded message is NOT a candidate.
        assert "[gm_09]" not in prompt

    def test_retraction_in_a_later_run_retracts_the_original_thread(self):
        """The reviewer's cross-run gap: a retraction processed a run after
        its incident must APPEND to the original thread (flipping it to
        retracted), not become an orphan single-message thread."""
        from src.services import ground_intel

        messages, existing = second_run_state()
        proposal = {
            "threads": [
                {
                    "title": "Reported strikes on the Galaxy compound",
                    "lifecycle_state": "corrected",  # model was too mild
                    "message_ids": ["gm_10"],
                    "thread_id": "gth_r1",
                }
            ]
        }
        with patch.object(ground_intel, "call_claude", return_value=proposal):
            threads = ground_intel.build_threads(SOURCE_ID, messages, existing)

        assert len(threads) == 1  # no orphan thread
        t = threads[0]
        assert set(t) == {
            "groundSourceId", "threadId", "title", "lifecycleState", "messageIds",
        }
        assert t["threadId"] == "gth_r1"
        assert t["messageIds"] == ["gm_10"]  # only the NEW message is attached
        assert t["lifecycleState"] == "retracted"  # deterministic override

    def test_append_title_falls_back_to_the_existing_threads_title(self):
        from src.services import ground_intel

        messages, existing = second_run_state()
        proposal = {
            "threads": [
                {
                    "title": "  ",
                    "lifecycle_state": "updated",
                    "message_ids": ["gm_10"],
                    "thread_id": "gth_r1",
                }
            ]
        }
        with patch.object(ground_intel, "call_claude", return_value=proposal):
            threads = ground_intel.build_threads(SOURCE_ID, messages, existing)

        assert threads[0]["title"] == "Reported strikes on the Galaxy compound"

    def test_batch_boundary_append_counts_unseen_members_for_lifecycle(self):
        """An incident straddling two fetch windows: the earlier members are
        no longer in view, but they still count for thread size — a benign
        follow-up append must not degrade to 'reported'."""
        from src.services import ground_intel

        messages = [
            {
                "id": "gm_20",
                "text": "Two more strikes heard near the water point this afternoon.",
                "sentAt": "2026-04-12T14:00:00+00:00",
                "senderRef": "member_07",
                "hasMedia": False,
                "classification": "field_report",
                "threadId": None,
            }
        ]
        existing = [
            {
                "id": "gth_b",
                "title": "Drone strikes near Zalingei",
                "lifecycleState": "corrected",
                "reviewState": "pending",
                "messageIds": ["gm_x1", "gm_x2"],  # outside the window
            }
        ]
        proposal = {
            "threads": [
                {
                    "title": "Drone strikes near Zalingei",
                    "lifecycle_state": "updated",
                    "message_ids": ["gm_20"],
                    "thread_id": "gth_b",
                }
            ]
        }
        with patch.object(ground_intel, "call_claude", return_value=proposal):
            threads = ground_intel.build_threads(SOURCE_ID, messages, existing)

        assert threads[0]["threadId"] == "gth_b"
        assert threads[0]["lifecycleState"] == "updated"  # 3 members total

    def test_unknown_thread_id_falls_back_to_a_new_thread(self):
        from src.services import ground_intel

        messages, existing = second_run_state()
        proposal = {
            "threads": [
                {
                    "title": "Reported checkpoint on the west road",
                    "lifecycle_state": "reported",
                    "message_ids": ["gm_08"],
                    "thread_id": "gth_nope",
                }
            ]
        }
        with patch.object(ground_intel, "call_claude", return_value=proposal):
            threads = ground_intel.build_threads(SOURCE_ID, messages, existing)

        assert len(threads) == 1
        assert "threadId" not in threads[0]
        assert threads[0]["lifecycleState"] == "reported"

    def test_promoted_threads_are_not_append_targets(self):
        """The server refuses appends to promoted threads — they are neither
        offered in the prompt nor honoured if the model names one anyway."""
        from src.services import ground_intel

        messages, existing = second_run_state()
        existing[0]["reviewState"] = "approved_public"
        seen_prompts = []

        def spy(_system, _user, **_kwargs):
            seen_prompts.append(_user)
            return {
                "threads": [
                    {
                        "title": "Galaxy compound report withdrawn",
                        "lifecycle_state": "retracted",
                        "message_ids": ["gm_10"],
                        "thread_id": "gth_r1",
                    }
                ]
            }

        with patch.object(ground_intel, "call_claude", side_effect=spy):
            threads = ground_intel.build_threads(SOURCE_ID, messages, existing)

        assert "[gth_r1]" not in seen_prompts[0]
        assert len(threads) == 1
        assert "threadId" not in threads[0]  # falls back to a new thread


# ─── the Celery task (mocked GraphQL transport) ────────────────────────────


class TestClassifyGroundMessagesTask:
    def _run(self, messages, claude=canned_claude, existing_threads=None):
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
                "ground_threads_for_source",
                return_value=existing_threads or [],
            ),
            patch.object(
                task_module,
                "upsert_ground_message_classifications",
                # Contract v2: the server returns a scalar upserted-row count.
                side_effect=lambda inputs: len(inputs),
            ) as mock_upsert,
            patch.object(
                task_module,
                "upsert_ground_threads",
                # Contract v2: the server returns a scalar list of thread ids.
                side_effect=lambda inputs: [f"gth_{n}" for n in range(len(inputs))],
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

    def test_two_run_retraction_appends_to_run1_thread_without_orphan(self):
        """Two-run scenario: the incident threaded in run 1; its retraction
        arrives in run 2. The retraction must append to the run-1 thread
        (flipping it retracted) — never create an orphan thread."""
        messages = [
            {
                "id": "gm_a1",
                "text": "Strikes reported this evening on the Galaxy compound south of town.",
                "sentAt": "2026-04-12T19:40:00+00:00",
                "senderRef": "member_06",
                "hasMedia": False,
                "classification": "field_report",
                "threadId": "gth_run1",  # threaded by run 1
            },
            {
                "id": "gm_a2",
                "text": "About yesterday's report: this turned out to be misreporting. No strikes took place.",
                "sentAt": "2026-04-13T08:15:00+00:00",
                "senderRef": "member_06",
                "hasMedia": False,
                "classification": None,  # arrived after run 1
                "threadId": None,
            },
        ]
        existing = [
            {
                "id": "gth_run1",
                "title": "Reported strikes on the Galaxy compound",
                "lifecycleState": "reported",
                "reviewState": "pending",
                "messageIds": ["gm_a1"],
            }
        ]

        def claude(_system, _user, *, stage=None, **_kwargs):
            if stage == "ground_classify":
                return {
                    "classifications": [
                        {"id": "gm_a2", "classification": "field_report"}
                    ]
                }
            return {
                "threads": [
                    {
                        "title": "Reported strikes on the Galaxy compound",
                        "lifecycle_state": "corrected",  # model too mild
                        "message_ids": ["gm_a2"],
                        "thread_id": "gth_run1",
                    }
                ]
            }

        result, _query, _upsert, mock_threads = self._run(
            messages, claude=claude, existing_threads=existing
        )

        assert result["threads_upserted"] == 1
        inputs = mock_threads.call_args.args[0]
        assert len(inputs) == 1  # append only — no orphan thread
        assert inputs[0]["threadId"] == "gth_run1"
        assert inputs[0]["messageIds"] == ["gm_a2"]
        assert inputs[0]["lifecycleState"] == "retracted"

    def test_batch_boundary_tail_appends_to_run1_thread(self):
        """Batch-boundary scenario: an incident straddles two fetch windows.
        Run 2 sees only the tail messages; they append to the run-1 thread
        instead of forming a duplicate incident."""
        messages = [
            {
                "id": "gm_b3",
                "text": "Two more strikes heard from the same direction near Zalingei in the last hour.",
                "sentAt": "2026-04-12T09:02:00+00:00",
                "senderRef": "member_07",
                "hasMedia": False,
                "classification": None,
                "threadId": None,
            },
            {
                "id": "gm_b4",
                "text": "Smoke visible over the north road now",
                "sentAt": "2026-04-12T09:05:00+00:00",
                "senderRef": "member_07",
                "hasMedia": True,
                "classification": None,
                "threadId": None,
            },
        ]
        existing = [
            {
                "id": "gth_zal",
                "title": "Drone strikes near Zalingei",
                "lifecycleState": "corrected",
                "reviewState": "pending",
                # run-1 members — no longer inside the fetch window
                "messageIds": ["gm_b1", "gm_b2"],
            }
        ]

        def claude(_system, _user, *, stage=None, **_kwargs):
            if stage == "ground_classify":
                return {
                    "classifications": [
                        {"id": "gm_b3", "classification": "field_report"},
                        {"id": "gm_b4", "classification": "field_report"},
                    ]
                }
            return {
                "threads": [
                    {
                        "title": "Drone strikes near Zalingei",
                        "lifecycle_state": "updated",
                        "message_ids": ["gm_b3", "gm_b4"],
                        "thread_id": "gth_zal",
                    }
                ]
            }

        result, _query, _upsert, mock_threads = self._run(
            messages, claude=claude, existing_threads=existing
        )

        assert result["threads_upserted"] == 1
        inputs = mock_threads.call_args.args[0]
        assert len(inputs) == 1
        assert inputs[0]["threadId"] == "gth_zal"
        assert inputs[0]["messageIds"] == ["gm_b3", "gm_b4"]
        # 2 unseen run-1 members + 2 new → multi-message; model state stands.
        assert inputs[0]["lifecycleState"] == "updated"

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
                task_module, "ground_threads_for_source"
            ) as mock_existing,
            patch.object(
                task_module, "upsert_ground_message_classifications"
            ) as mock_upsert,
            patch.object(task_module, "upsert_ground_threads") as mock_threads,
            patch.object(ground_intel, "call_claude") as mock_call,
        ):
            result = task_module.classify_ground_messages.apply(args=[SOURCE_ID]).get()

        assert result["messages_fetched"] == 0
        mock_call.assert_not_called()
        mock_existing.assert_not_called()  # nothing to thread → no thread fetch
        mock_upsert.assert_not_called()
        mock_threads.assert_not_called()

    def test_task_is_registered_under_the_contract_name(self):
        """clear-api enqueues by the bare name "classify_ground_messages"
        (see docs/GROUND_INTEL.md). If the task ever loses that exact
        registration, enqueued messages are silently discarded by the
        worker — this is the tripwire."""
        from src.celery_app import app as celery_app

        assert "classify_ground_messages" in celery_app.tasks

    def test_graphql_contract_v2_document_shapes(self):
        """Tripwire for the canonical clear-api contract (v2): exact input
        type literals, and SCALAR returns on both mutations — a selection
        set on a scalar field is a GraphQL validation error server-side."""
        from src.clients import graphql

        assert (
            "[GroundMessageClassificationInput!]!"
            in graphql.UPSERT_GROUND_MESSAGE_CLASSIFICATIONS
        )
        # Scalar Int! return — the field call must NOT open a selection set.
        assert (
            "upsertGroundMessageClassifications(inputs: $inputs)\n}"
            in graphql.UPSERT_GROUND_MESSAGE_CLASSIFICATIONS
        )

        assert "[GroundThreadUpsertInput!]!" in graphql.UPSERT_GROUND_THREADS
        assert "UpsertGroundThreadInput" not in graphql.UPSERT_GROUND_THREADS
        # Scalar [String]! return — no selection set here either.
        assert (
            "upsertGroundThreads(inputs: $inputs)\n}" in graphql.UPSERT_GROUND_THREADS
        )

        for field in ("id", "title", "lifecycleState", "reviewState", "messageIds"):
            assert field in graphql.GROUND_THREADS_FOR_SOURCE

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
