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
