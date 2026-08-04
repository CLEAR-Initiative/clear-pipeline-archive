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


def load_messages() -> list[dict]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


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


# ─── the Celery task (mocked GraphQL transport) ────────────────────────────


class TestClassifyGroundMessagesTask:
    def _run(self, messages, classify_response=CANNED_CLASSIFY):
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
            patch.object(ground_intel, "call_claude", return_value=classify_response),
        ):
            result = task_module.classify_ground_messages.apply(args=[SOURCE_ID]).get()
        return result, mock_query, mock_upsert

    def test_tracer_classifies_fixture_and_writes_back(self):
        """Acceptance tracer: fixture messages in → classifications written
        back through the GraphQL surface, one row per message."""
        result, mock_query, mock_upsert = self._run(load_messages())

        assert result == {
            "ground_source_id": SOURCE_ID,
            "messages_fetched": 10,
            "messages_classified": 10,
        }
        mock_query.assert_called_once_with(SOURCE_ID, 500)

        mock_upsert.assert_called_once()
        written = mock_upsert.call_args.args[0]
        assert {w["messageId"] for w in written} == {m["id"] for m in load_messages()}
        valid = {"field_report", "news_digest", "operational", "chatter"}
        assert all(w["classification"] in valid for w in written)

    def test_already_classified_messages_are_not_reclassified(self):
        messages = load_messages()
        for m in messages[:8]:
            m["classification"] = "chatter"

        canned = {
            "classifications": [
                {"id": "gm_09", "classification": "field_report"},
                {"id": "gm_10", "classification": "field_report"},
            ]
        }
        result, _query, mock_upsert = self._run(messages, classify_response=canned)

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
            patch.object(ground_intel, "call_claude") as mock_call,
        ):
            result = task_module.classify_ground_messages.apply(args=[SOURCE_ID]).get()

        assert result["messages_fetched"] == 0
        mock_call.assert_not_called()
        mock_upsert.assert_not_called()

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
