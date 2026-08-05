"""Tests for the darfur24.com signal source (`src.clients.darfur24` +
`src.tasks.poll_darfur24`).

All tests are hermetic: the feed is a recorded fixture
(tests/fixtures/darfur24_feed.xml, captured from the live English + Arabic
feeds on 2026-08-04), Redis is a tiny in-memory fake, and GraphQL is mocked.
No network calls.

The two acceptance criteria under test:
  - new darfur24 articles become signals with title/description/url/
    publishedAt and `externalId: darfur24:{slug}`
  - re-polling creates no duplicates (Redis seen-set layer)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from src.clients import darfur24

FIXTURE = Path(__file__).parent / "fixtures" / "darfur24_feed.xml"

EN_SLUG = "rsf-fighters-burn-fuel-tanker-after-dispute-in-west-kordofan"


class FakeRedis:
    """Just enough of the redis-py surface for the darfur24 client."""

    def __init__(self):
        self.store: dict[str, str] = {}

    def get(self, key):
        return self.store.get(key)

    def set(self, key, value):
        self.store[key] = value

    def setex(self, key, ttl, value):
        self.store[key] = value

    def exists(self, key):
        return 1 if key in self.store else 0

    def delete(self, key):
        self.store.pop(key, None)


@pytest.fixture
def fake_redis(monkeypatch) -> FakeRedis:
    r = FakeRedis()
    monkeypatch.setattr(darfur24, "_redis", r)
    return r


@pytest.fixture
def recorded_feed(monkeypatch) -> str:
    """Serve the recorded fixture instead of hitting darfur24.com."""
    body = FIXTURE.read_text(encoding="utf-8")
    monkeypatch.setattr(darfur24, "_fetch_feed", lambda url: body)
    return body


# ─── feed parsing ──────────────────────────────────────────────────────────


class TestFeedParsing:
    def test_parses_valid_items_and_skips_broken_ones(self, fake_redis, recorded_feed):
        """Fixture has 4 items: 2 English, 1 Arabic, 1 with no <link> —
        the linkless one cannot yield a slug/externalId and must be dropped."""
        articles = darfur24.fetch_darfur24_articles()
        assert len(articles) == 3

    def test_article_carries_signal_fields(self, fake_redis, recorded_feed):
        article = darfur24.fetch_darfur24_articles()[0]

        assert article["darfur24_id"] == EN_SLUG
        assert article["title"] == (
            "RSF fighters burn fuel tanker after dispute in West Kordofan"
        )
        assert article["url"] == (
            f"https://darfur24.com/en/2026/08/04/{EN_SLUG}/"
        )
        # pubDate "Tue, 04 Aug 2026 10:29:27 +0000" → ISO-8601 UTC
        assert article["published_at"] == "2026-08-04T10:29:27+00:00"

    def test_description_is_plain_text_without_boilerplate(self, fake_redis, recorded_feed):
        description = darfur24.fetch_darfur24_articles()[0]["description"]

        assert description.startswith("Kordofan, August 04 (Darfur24)")
        assert "<p>" not in description
        assert "appeared first on" not in description  # WordPress footer stripped
        assert len(description) <= 500

    def test_arabic_slug_is_percent_decoded(self, fake_redis, recorded_feed):
        """Arabic-edition permalinks arrive percent-encoded; the slug (and so
        the externalId) must be the decoded, stable form."""
        arabic = darfur24.fetch_darfur24_articles()[2]

        assert "%" not in arabic["darfur24_id"]
        assert arabic["darfur24_id"].startswith("خلافات")
        assert arabic["title"] == "خلافات داخلية بقوات الدعم السريع في ود بندة"

    def test_raw_data_is_json_serializable(self, fake_redis, recorded_feed):
        import json

        for article in darfur24.fetch_darfur24_articles():
            json.dumps(article["raw"])  # must not raise

    def test_missing_pub_date_yields_none(self):
        import xml.etree.ElementTree as ET

        item = ET.fromstring(
            "<item><title>t</title>"
            "<link>https://darfur24.com/en/2026/08/04/some-slug/</link></item>"
        )
        parsed = darfur24._parse_item(item)
        assert parsed is not None
        assert parsed["published_at"] is None

    def test_fetch_failure_returns_empty_list(self, fake_redis, monkeypatch):
        monkeypatch.setattr(darfur24, "_fetch_feed", lambda url: None)
        assert darfur24.fetch_darfur24_articles() == []

    def test_malformed_xml_returns_empty_list(self, fake_redis, monkeypatch):
        monkeypatch.setattr(darfur24, "_fetch_feed", lambda url: "<rss><channel>")
        assert darfur24.fetch_darfur24_articles() == []


# ─── dedupe ────────────────────────────────────────────────────────────────


class TestDedupe:
    def test_second_poll_returns_no_articles(self, fake_redis, recorded_feed):
        """The Redis seen-set must swallow a full feed replay — once the
        articles have been marked seen (which the poll task does after
        signal creation)."""
        first = darfur24.fetch_darfur24_articles()
        for article in first:
            darfur24.mark_seen(article["darfur24_id"])
        second = darfur24.fetch_darfur24_articles()

        assert len(first) == 3
        assert second == []

    def test_fetch_does_not_mark_seen(self, fake_redis, recorded_feed):
        """Regression (expo-383): fetching must NOT touch the seen-set —
        marking happens only after successful signal creation. A fetch whose
        downstream creation fails must leave every article re-fetchable."""
        first = darfur24.fetch_darfur24_articles()

        assert len(first) == 3
        assert not any(k.startswith("darfur24:seen:") for k in fake_redis.store)

        # Nothing was marked → a replay yields the same articles again.
        second = darfur24.fetch_darfur24_articles()
        assert len(second) == 3

    def test_single_fetch_dedupes_within_batch(self, fake_redis, monkeypatch):
        """Two configured feeds serving the same items must not yield the
        same slug twice in one fetch (there is no Redis write in between)."""
        body = FIXTURE.read_text(encoding="utf-8")
        monkeypatch.setattr(darfur24, "_fetch_feed", lambda url: body)
        monkeypatch.setattr(
            darfur24.settings, "darfur24_feed_urls",
            "https://darfur24.com/en/feed/,https://darfur24.com/feed/",
        )

        articles = darfur24.fetch_darfur24_articles()
        slugs = [a["darfur24_id"] for a in articles]
        assert len(slugs) == len(set(slugs)) == 3

    def test_seen_keys_are_slug_scoped(self, fake_redis, recorded_feed):
        darfur24.mark_seen(EN_SLUG)
        assert f"darfur24:seen:{EN_SLUG}" in fake_redis.store

    def test_last_synced_set_only_when_new_articles(self, fake_redis, recorded_feed):
        darfur24.fetch_darfur24_articles()
        assert darfur24.get_last_synced() is not None

        for article in darfur24.fetch_darfur24_articles():
            darfur24.mark_seen(article["darfur24_id"])

        stamp = fake_redis.store["darfur24:last_synced"]
        darfur24.fetch_darfur24_articles()  # all deduped → no update
        assert fake_redis.store["darfur24:last_synced"] == stamp


# ─── signal input (poll task) ──────────────────────────────────────────────


class TestBuildSignalInput:
    def _article(self, **overrides) -> dict:
        article = {
            "darfur24_id": EN_SLUG,
            "title": "RSF fighters burn fuel tanker after dispute in West Kordofan",
            "description": "Kordofan, August 04 (Darfur24) …",
            "url": f"https://darfur24.com/en/2026/08/04/{EN_SLUG}/",
            "published_at": "2026-08-04T10:29:27+00:00",
            "raw": {"title": "…"},
        }
        article.update(overrides)
        return article

    def test_external_id_is_darfur24_slug(self):
        from src.tasks.poll_darfur24 import _build_signal_input

        input_data = _build_signal_input(self._article(), "src_123")
        assert input_data["externalId"] == f"darfur24:{EN_SLUG}"
        assert input_data["sourceId"] == "src_123"

    def test_signal_carries_title_description_url_published_at(self):
        from src.tasks.poll_darfur24 import _build_signal_input

        article = self._article()
        input_data = _build_signal_input(article, "src_123")

        assert input_data["title"] == article["title"]
        assert input_data["description"] == article["description"]
        assert input_data["url"] == article["url"]
        assert input_data["publishedAt"] == article["published_at"]
        assert input_data["rawData"] == article["raw"]

    def test_no_invented_severity_or_coordinates(self):
        """News articles carry no structured severity/casualty/coordinate
        data — the input must not fabricate any (unlike ACLED/GDACS)."""
        from src.tasks.poll_darfur24 import _build_signal_input

        input_data = _build_signal_input(self._article(), "src_123")
        for forbidden in ("severity", "casualties", "lat", "lng"):
            assert forbidden not in input_data

    def test_missing_published_at_falls_back_to_now(self):
        from src.tasks.poll_darfur24 import _build_signal_input

        input_data = _build_signal_input(self._article(published_at=None), "src_123")
        assert input_data["publishedAt"]  # non-empty ISO timestamp


# ─── poll task end-to-end (mocked GraphQL) ─────────────────────────────────


class TestPollTask:
    def test_repolling_creates_no_duplicate_signals(self, fake_redis, recorded_feed):
        """Acceptance criterion: run the poll twice against the same feed —
        the second round must create zero signals."""
        from src.tasks import poll_darfur24 as task_module

        with (
            patch.object(
                task_module,
                "get_data_sources",
                return_value=[{"id": "src_d24", "name": "darfur24"}],
            ),
            patch.object(
                task_module, "create_signal", return_value={"id": "sig_1"}
            ) as mock_create,
        ):
            first = task_module.poll_darfur24.apply().get()
            second = task_module.poll_darfur24.apply().get()

        assert first == {"articles_found": 3, "signals_created": 3, "failed": 0}
        assert second == {"articles_found": 0, "signals_created": 0}
        assert mock_create.call_count == 3

        external_ids = {
            call.args[0]["externalId"] for call in mock_create.call_args_list
        }
        assert f"darfur24:{EN_SLUG}" in external_ids
        assert len(external_ids) == 3  # all distinct

    def test_missing_data_source_row_raises(self, fake_redis, recorded_feed):
        """The darfur24 dataSources row must exist in the CLEAR API — fail
        loudly (and non-retryably surface the config error) when absent."""
        from src.tasks import poll_darfur24 as task_module

        # reset the module-level cache
        task_module._darfur24_source_id = None

        with patch.object(task_module, "get_data_sources", return_value=[]):
            with pytest.raises(Exception) as excinfo:
                task_module.poll_darfur24.apply(throw=True).get()

        assert "darfur24" in str(excinfo.value)

    def test_missing_source_marks_nothing_then_next_run_ingests_all(
        self, fake_redis, recorded_feed
    ):
        """Regression (expo-383, observed on dev 4-5 Aug 2026): polls that ran
        before the darfur24 data_sources row existed marked every article
        seen without creating any signal — `already_seen=10,
        signals_created=0` forever after. A failed run must mark NOTHING so
        the next successful run ingests every article exactly once."""
        from src.tasks import poll_darfur24 as task_module

        # Round 1: the data source row does not exist yet → the poll fails …
        with patch.object(task_module, "get_data_sources", return_value=[]):
            with pytest.raises(Exception):
                task_module.poll_darfur24.apply(throw=True).get()

        # … and the seen-set must be untouched.
        assert not any(k.startswith("darfur24:seen:") for k in fake_redis.store)

        # Round 2: source row now exists → all articles ingest exactly once.
        task_module._darfur24_source_id = None
        with (
            patch.object(
                task_module,
                "get_data_sources",
                return_value=[{"id": "src_d24", "name": "darfur24"}],
            ),
            patch.object(
                task_module, "create_signal", return_value={"id": "sig_1"}
            ) as mock_create,
        ):
            result = task_module.poll_darfur24.apply().get()
            replay = task_module.poll_darfur24.apply().get()

        assert result == {"articles_found": 3, "signals_created": 3, "failed": 0}
        assert replay == {"articles_found": 0, "signals_created": 0}
        assert mock_create.call_count == 3
        external_ids = [c.args[0]["externalId"] for c in mock_create.call_args_list]
        assert len(external_ids) == len(set(external_ids)) == 3

    def test_one_bad_article_does_not_sink_the_batch(self, fake_redis, recorded_feed):
        from src.tasks import poll_darfur24 as task_module

        calls = {"n": 0}

        def flaky_create(input_data):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("boom")
            return {"id": f"sig_{calls['n']}"}

        with (
            patch.object(
                task_module,
                "get_data_sources",
                return_value=[{"id": "src_d24", "name": "darfur24"}],
            ),
            patch.object(task_module, "create_signal", side_effect=flaky_create),
        ):
            result = task_module.poll_darfur24.apply().get()
            retry = task_module.poll_darfur24.apply().get()

        assert result == {"articles_found": 3, "signals_created": 2, "failed": 1}
        # expo-383: the failed article was NOT marked seen, so the next poll
        # retries exactly that one (and only that one).
        assert retry == {"articles_found": 1, "signals_created": 1, "failed": 0}


@pytest.fixture(autouse=True)
def _reset_source_id_cache():
    """The poll task caches the source id at module level; isolate tests."""
    yield
    import src.tasks.poll_darfur24 as task_module

    task_module._darfur24_source_id = None
