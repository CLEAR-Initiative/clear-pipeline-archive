"""
Translation quality A/B eval.

Pulls the description of the last N events from clear-api, translates each
into Arabic with every requested LLM (Claude Sonnet, Claude Haiku, GPT-4o,
GPT-4o-mini, Gemini Pro, Gemini Flash by default), and writes one Notion
row per (event, model) so reviewers can grade them with the Good / Bad /
Okayish counters.

Usage:
    # First time (extras: openai, google-genai, notion-client):
    uv pip install -e ".[eval]"

    # Then:
    uv run python scripts/eval_translations.py
    uv run python scripts/eval_translations.py --limit 10
    uv run python scripts/eval_translations.py --models claude-haiku,gpt-4o-mini
    uv run python scripts/eval_translations.py --dry-run    # no API calls, no Notion writes

Notion setup:
    Create an internal integration at notion.so/profile/integrations.
    The integration MUST have read + update content capabilities (it's
    the default for internal integrations — just leave the capability
    boxes checked).
    Copy the secret token into NOTION_TOKEN in .env.
    Share the target *database itself* (not just the parent page) with
    the integration: open the database → ⋯ menu above the table →
    Connections → enable the integration.

    Default DB id (from the issue): 3815043895cf80d4aede000c05ea6334
    (a view id from a ?v= URL — the script auto-resolves it to the
    underlying database).

    Columns are created on first run if they don't exist:
      English  — title (the row's title column is auto-renamed to this)
      Arabic   — rich text
      Model    — rich text (kept as-is if you set it to Select)
      Good     — number
      Bad      — number
      Okayish  — number
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Load .env early so `os.environ` sees everything before SDK imports
# (they can read keys at import time on some versions).
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.clients import graphql  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─── Models registry ──────────────────────────────────────────────────────────
# Each entry is (script-key, vendor, model-id-for-API, human-label-for-Notion).
# Add a row here to make a model available. Vendor decides which SDK is used.

@dataclass(frozen=True)
class ModelSpec:
    key: str          # short name passed via --models
    vendor: str       # "claude" | "openai" | "gemini"
    model_id: str     # actual model id for the vendor SDK
    label: str        # human-readable name written to the Notion `Model` column


MODELS: tuple[ModelSpec, ...] = (
    ModelSpec("claude-sonnet", "claude", "claude-sonnet-4-6", "Claude Sonnet 4.6"),
    ModelSpec("claude-haiku",  "claude", "claude-haiku-4-5-20251001", "Claude Haiku 4.5"),
    ModelSpec("gpt-4o",        "openai", "gpt-4o", "OpenAI GPT-4o"),
    ModelSpec("gpt-4o-mini",   "openai", "gpt-4o-mini", "OpenAI GPT-4o mini"),
    ModelSpec("gemini-pro",    "gemini", "gemini-2.5-pro", "Gemini 2.5 Pro"),
    ModelSpec("gemini-flash",  "gemini", "gemini-2.5-flash", "Gemini 2.5 Flash"),
)


# Same prompt for every model so grades reflect model quality, not
# prompt engineering. Kept terse on purpose — humanitarian text often
# carries proper nouns, acronyms and place names that we want
# preserved, not paraphrased.
SYSTEM_PROMPT = (
    "You are a professional translator for humanitarian crisis content. "
    "Translate the input from English into Modern Standard Arabic. "
    "Preserve technical terms, proper nouns, place names, dates, numbers, "
    "acronyms, and NRC sector names. Output ONLY the translation — no "
    "commentary, no quotes, no romanization."
)


# ─── Per-vendor translation calls ─────────────────────────────────────────────
# Each function is lazily-imported so a missing optional SDK only fails
# the models that depend on it. That lets `--models claude-haiku` work
# without openai / google-genai installed.


def _translate_claude(model_id: str, english: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    resp = client.messages.create(
        model=model_id,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": english}],
    )
    return resp.content[0].text.strip()


def _translate_openai(model_id: str, english: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": english},
        ],
        # OpenAI accepts None to skip the cap; 2048 is generous for one
        # paragraph of Arabic.
        max_completion_tokens=2048,
    )
    return (resp.choices[0].message.content or "").strip()


def _translate_gemini(model_id: str, english: str) -> str:
    from google import genai

    # Google's SDK reads GOOGLE_API_KEY by default; accept GEMINI_API_KEY
    # as an alias for parity with the rest of the env file.
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY or GEMINI_API_KEY must be set")
    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(
        model=model_id,
        contents=english,
        config={
            "system_instruction": SYSTEM_PROMPT,
            "max_output_tokens": 2048,
        },
    )
    return (resp.text or "").strip()


_VENDOR_FNS = {
    "claude": _translate_claude,
    "openai": _translate_openai,
    "gemini": _translate_gemini,
}

# Maps each vendor to the import name + the pip package the user needs to
# install (they differ for google-genai → `from google import genai`).
# Driving the pre-flight check off this dict means adding a new vendor
# only touches one place.
_VENDOR_IMPORTS: dict[str, tuple[str, str]] = {
    "claude": ("anthropic", "anthropic"),
    "openai": ("openai", "openai"),
    "gemini": ("google.genai", "google-genai"),
}


def check_vendor_dependencies(selected: list[ModelSpec]) -> list[str]:
    """Verify every SDK the chosen models need is importable. Returns
    the list of vendor *pip names* whose imports failed — the caller
    prints a single actionable install command instead of letting the
    run die mid-batch with a noisy stack trace per row."""
    import importlib

    needed_vendors = {m.vendor for m in selected}
    missing: list[str] = []
    for vendor in needed_vendors:
        spec = _VENDOR_IMPORTS.get(vendor)
        if spec is None:
            continue
        module_name, pip_name = spec
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing.append(pip_name)
    return missing


def translate(model: ModelSpec, english: str) -> str | None:
    """Translate a single string with one model. Returns None on
    failure so a missing API key for one vendor doesn't abort the run."""
    fn = _VENDOR_FNS.get(model.vendor)
    if fn is None:
        logger.error("[%s] unknown vendor %r", model.key, model.vendor)
        return None
    try:
        out = fn(model.model_id, english)
    except KeyError as exc:
        # Missing env var — be specific.
        logger.warning("[%s] skipped: missing env %s", model.key, exc.args[0])
        return None
    except Exception as exc:
        logger.error("[%s] translation failed: %s", model.key, exc)
        return None
    return out or None


# ─── Notion writes ────────────────────────────────────────────────────────────


# Notion's rich_text accepts max 2000 chars per block. Long descriptions
# get split so we don't lose content.
NOTION_RICH_TEXT_LIMIT = 2000


def _chunk_text(s: str, limit: int = NOTION_RICH_TEXT_LIMIT) -> list[dict]:
    """Build the rich_text array Notion expects, splitting at the limit
    so long descriptions don't 400 the request."""
    if not s:
        return []
    return [
        {"type": "text", "text": {"content": s[i : i + limit]}}
        for i in range(0, len(s), limit)
    ]


def _build_properties(
    db_schema: dict,
    english: str,
    arabic: str,
    model_label: str,
) -> dict:
    """Match each of (English, Arabic, Model, Good, Bad, Okayish) to the
    actual property type on the target DB. The user's DB might have
    `Model` as a Select or as a Rich Text — we read the schema once and
    serialise accordingly."""

    def _prop(name: str) -> dict:
        prop = db_schema.get(name)
        if prop is None:
            raise RuntimeError(
                f"Notion DB is missing the {name!r} column. Existing columns: "
                f"{sorted(db_schema.keys())}"
            )
        return prop

    out: dict = {}

    eng_prop = _prop("English")
    if eng_prop["type"] == "title":
        out["English"] = {"title": _chunk_text(english)}
    else:
        out["English"] = {"rich_text": _chunk_text(english)}

    arabic_prop = _prop("Arabic")
    out["Arabic"] = {arabic_prop["type"]: _chunk_text(arabic)} if arabic_prop["type"] in (
        "rich_text", "title",
    ) else {"rich_text": _chunk_text(arabic)}

    model_prop = _prop("Model")
    if model_prop["type"] == "select":
        out["Model"] = {"select": {"name": model_label}}
    elif model_prop["type"] == "multi_select":
        out["Model"] = {"multi_select": [{"name": model_label}]}
    else:
        out["Model"] = {"rich_text": _chunk_text(model_label)}

    # Counters start at 0; reviewers bump them by hand. Reject any
    # non-number type loudly so a misnamed column doesn't silently break.
    for counter in ("Good", "Bad", "Okayish"):
        prop = _prop(counter)
        if prop["type"] != "number":
            raise RuntimeError(
                f"Notion column {counter!r} must be type 'number' "
                f"(got {prop['type']!r})."
            )
        out[counter] = {"number": 0}

    return out


def _connect_notion():
    from notion_client import Client

    token = os.environ.get("NOTION_TOKEN")
    if not token:
        raise RuntimeError("NOTION_TOKEN must be set in .env")
    return Client(auth=token)


def _normalise_notion_id(raw: str) -> str:
    """Accept a Notion URL, a 32-char hex id, or a dashed UUID.
    Notion URLs come in a few flavours:
      .../d/<id>            top-level database
      .../p/<page_id>?v=…   page with an inline database (the ?v= is a view id)
      .../<title>-<id>      "pretty" URL with a slug

    Anything that isn't 32 hex chars after stripping dashes / queries
    isn't a Notion id at all — flag that loudly.
    """
    s = raw.strip()
    # URL → last path segment, query string dropped.
    if "://" in s or s.startswith("notion.") or s.startswith("app.notion"):
        s = s.split("?", 1)[0].rstrip("/").split("/")[-1]
        # Pretty URLs put a slug before the id: "Crisis-eval-3815…".
        if "-" in s and len(s.replace("-", "")) > 32:
            s = s.split("-")[-1]
    s = s.replace("-", "")
    if len(s) != 32 or any(c not in "0123456789abcdefABCDEF" for c in s):
        raise ValueError(
            f"Doesn't look like a Notion id: {raw!r}. "
            "Expected a 32-char hex id, dashed UUID, or a notion.so URL."
        )
    return s.lower()


def _resolve_data_source_id(notion, given_id: str) -> str:
    """Resolve whatever the operator pasted (data-source id, database
    id, page id, view id, full URL) to a *data_source_id* — the new
    object the 2025-09 Notion API uses for schema reads and writes.

    Why this is more involved than `databases.retrieve(id)`:
      - In the new API, a "database" is a thin container; its schema
        lives on a child *data source*. `databases.retrieve` returns
        `properties: {}`. `databases.update(..., properties=...)`
        silently no-ops. Everything schema-shaped happens against the
        data_source endpoint.
      - URLs still expose database / page / view ids. The operator
        can't always tell which is which from a URL.

    Strategy (each step short-circuits on a data_source with non-empty
    properties; falls through otherwise):
      1. Try the id as a data_source directly.
      2. Try the id as a database → use its first listed data_source.
      3. Recursively walk block children for child_database blocks and
         repeat step 2 on each.
    """
    from notion_client.errors import APIResponseError

    norm = _normalise_notion_id(given_id)
    diagnostics: list[str] = []
    # (data_source_id, column_count, source) — we collect all candidates
    # Notion lets us read, then pick the most-populated. Even a
    # 0-column DS is kept so the self-heal step can patch it.
    candidates: list[tuple[str, int, str]] = []

    def _try_data_source(ds_id: str, source: str) -> None:
        try:
            ds = notion.data_sources.retrieve(data_source_id=ds_id)
        except APIResponseError as exc:
            diagnostics.append(
                f"{source}: data_sources.retrieve({ds_id}) "
                f"failed code={getattr(exc, 'code', '?')}"
            )
            return
        props = ds.get("properties") or {}
        candidates.append((ds_id, len(props), source))
        if not props:
            diagnostics.append(
                f"{source}: data_sources.retrieve({ds_id}) returned 0 columns "
                "— the integration may see the data source envelope but isn't "
                "connected to the source DB."
            )

    def _try_database(db_id: str, source: str) -> None:
        """A database now exposes its data_sources as an array; pick the
        first (most DBs have exactly one)."""
        try:
            db = notion.databases.retrieve(database_id=db_id)
        except APIResponseError as exc:
            diagnostics.append(
                f"{source}: databases.retrieve({db_id}) "
                f"failed code={getattr(exc, 'code', '?')}"
            )
            return
        data_sources = db.get("data_sources") or []
        if data_sources:
            ds_id = data_sources[0]["id"]
            _try_data_source(ds_id, f"data_source inside DB {db_id} ({source})")
            return
        # Old-API fallback: properties used to live on the database
        # object itself. Honour it if Notion still returns them.
        props = db.get("properties") or {}
        if props:
            candidates.append((db_id, len(props), f"{source} (legacy DB API)"))
        else:
            diagnostics.append(
                f"{source}: databases.retrieve({db_id}) returned no "
                "data_sources and no properties."
            )

    # 1) Try as direct data_source id.
    _try_data_source(norm, "direct data_source id")

    # 2) Try as a database id.
    _try_database(norm, "direct DB id")

    # 3) Walk the block tree for inline databases — DBs are often a
    # child_database block on a parent page.
    seen_inline_dbs: list[str] = []
    block_count = {"value": 0}

    def _walk(block_id: str, depth: int) -> None:
        if depth > 4 or block_count["value"] > 200:
            return
        try:
            page = notion.blocks.children.list(block_id=block_id)
        except APIResponseError as exc:
            diagnostics.append(
                f"blocks.children.list({block_id}) failed at depth={depth} "
                f"code={getattr(exc, 'code', '?')}"
            )
            return
        for block in page.get("results", []):
            block_count["value"] += 1
            btype = block.get("type")
            if btype == "child_database":
                seen_inline_dbs.append(block["id"])
            elif block.get("has_children"):
                _walk(block["id"], depth + 1)

    _walk(norm, depth=0)

    if seen_inline_dbs:
        logger.info(
            "Walked %d block(s) under %s; found %d inline database(s) — probing each…",
            block_count["value"], norm, len(seen_inline_dbs),
        )
        for cand in seen_inline_dbs:
            _try_database(cand, f"inline DB under {norm}")
    else:
        diagnostics.append(
            f"block walk: scanned {block_count['value']} block(s) under {norm}, "
            "no child_database found"
        )

    if not candidates:
        raise RuntimeError(
            "Could not resolve the id to a data source.\n"
            "Tried:\n  - " + "\n  - ".join(diagnostics) + "\n"
            "Most common causes:\n"
            "  1. The integration isn't connected to the database. In Notion, "
            "open the database → ⋯ above the table → Connections → enable "
            "your integration.\n"
            "  2. The id you pasted is from a workspace your token can't see. "
            "Make sure NOTION_TOKEN comes from an integration in the SAME "
            "workspace as the database."
        )

    candidates.sort(key=lambda c: c[1], reverse=True)
    best_id, best_count, best_source = candidates[0]
    extras = "" if best_count > 0 else " — schema empty; will create columns"
    logger.info(
        "Resolved data_source %s via %s: %d column(s)%s",
        best_id, best_source, best_count, extras,
    )
    return best_id


def _fetch_data_source_schema(notion, data_source_id: str) -> dict:
    """Return the data source's `properties` map (possibly empty — the
    caller's self-heal step adds anything missing). Replaces the old
    `databases.retrieve` path: in the 2025-09 API, properties live on
    the data source, not on the database container."""
    ds = notion.data_sources.retrieve(data_source_id=data_source_id)
    return ds.get("properties") or {}


# Required schema for the eval. Title goes on the row's title column
# (every Notion DB has exactly one); everything else is a plain
# rich_text or number column.
_REQUIRED_SCHEMA: dict[str, str] = {
    "English": "title",
    "Arabic":  "rich_text",
    "Model":   "rich_text",
    "Good":    "number",
    "Bad":     "number",
    "Okayish": "number",
}


def _ensure_required_columns(notion, data_source_id: str, schema: dict) -> dict:
    """Self-heal the data source's schema so every column the script
    needs exists. Returns the schema after any patches.

    Behaviour:
      - A column with the right name already present is left alone,
        regardless of its current type — `_build_properties` adapts to
        whatever type the operator picked (title vs rich_text, select
        vs rich_text on Model, etc.).
      - For 'English': if no column called 'English' exists but the data
        source has a title column under a different name (the default
        'Name' or 'Page'), rename it to 'English'. Notion data sources
        can only have one title column so we can't add a second.
      - For everything else: create with the canonical type
        (rich_text / number) via data_sources.update.
    """
    # Locate the existing title column — every Notion DB has exactly one.
    title_name = next(
        (name for name, prop in schema.items() if prop.get("type") == "title"),
        None,
    )

    rename_ops: dict = {}
    add_ops: dict = {}

    for name, ideal_type in _REQUIRED_SCHEMA.items():
        if name in schema:
            continue  # column exists under the expected name — leave alone

        if ideal_type == "title":
            # Rename the existing title column rather than adding one —
            # Notion rejects a second title property with a 400.
            if title_name and title_name != name:
                rename_ops[title_name] = {"name": name}
                title_name = name  # track for the rest of the pass
            else:
                # No title column at all (shouldn't happen for a real DB)
                # — fall back to rich_text so the row still gets data.
                add_ops[name] = {"rich_text": {}}
        elif ideal_type == "rich_text":
            add_ops[name] = {"rich_text": {}}
        elif ideal_type == "number":
            add_ops[name] = {"number": {}}

    if not rename_ops and not add_ops:
        return schema

    if rename_ops:
        logger.info(
            "Renaming Notion column(s): %s",
            {old: new["name"] for old, new in rename_ops.items()},
        )
    if add_ops:
        logger.info("Creating missing Notion column(s): %s", sorted(add_ops.keys()))

    try:
        notion.data_sources.update(
            data_source_id=data_source_id,
            properties={**rename_ops, **add_ops},
        )
    except Exception as exc:
        # The integration needs "Update content" capability on the data
        # source to patch the schema. Surface the precondition rather
        # than bombing with a generic stack trace mid-batch.
        raise RuntimeError(
            f"Couldn't patch Notion data source schema: {exc}. "
            "The integration needs UPDATE access on the database — open it "
            "in Notion → ⋯ → Connections → verify the integration has "
            "write permission, not just read."
        )

    # Re-fetch so callers see the canonical post-update view.
    ds = notion.data_sources.retrieve(data_source_id=data_source_id)
    new_schema = ds.get("properties") or {}

    # Verify the schema patch actually applied. Notion sometimes returns
    # 200 OK on schema writes without applying anything (typical when
    # the integration only has read access on this data source). Bail
    # HERE instead of burning 120 LLM calls writing into a void.
    still_missing = [
        n for n in _REQUIRED_SCHEMA
        if n not in new_schema and _REQUIRED_SCHEMA[n] != "title"
    ]
    title_still_unrenamed = (
        "English" not in new_schema
        and any(p.get("type") == "title" for p in new_schema.values())
    )
    if still_missing or title_still_unrenamed:
        # Use the integration's own access list to give the operator a
        # concrete pointer. If our id doesn't appear there, the
        # connection is missing; if it does but is empty, the source
        # database is somewhere else (e.g. a linked DB pointing
        # elsewhere).
        accessible = _list_accessible_data_sources(notion)
        accessible_summary = "\n".join(
            f"    - {entry['id']}  cols={entry['column_count']}  title={entry['title']!r}"
            for entry in accessible
        ) or "    (none)"

        raise RuntimeError(
            "Notion accepted data_sources.update with HTTP 200 but the schema "
            "didn't change.\n"
            f"  - Columns we asked for that still aren't there: "
            f"{still_missing or '(only the title rename was lost)'}\n"
            f"  - Schema we ended up with: {sorted(new_schema.keys()) or '[]'}\n"
            f"  - data_source_id we wrote to: {data_source_id}\n\n"
            "Data sources your integration token CAN actually see + write to "
            "(from notion.search):\n"
            f"{accessible_summary}\n\n"
            "What this usually means:\n"
            "  - Your target data source isn't in the list above → the "
            "integration isn't connected to it. In Notion, open the database "
            "itself → ⋯ above the table → Connections → enable your "
            "integration.\n"
            "  - Your target data source IS in the list but has 0 columns → "
            "it's a linked/synced view, not the source database. Find the "
            "source DB (the one without the chain-link icon in the sidebar) "
            "and use ITS id."
        )

    return new_schema


def _list_accessible_data_sources(notion) -> list[dict]:
    """Use Notion's search endpoint to enumerate the data sources the
    current integration token has access to. Used as a diagnostic when
    a schema write silently no-ops — the operator can see at a glance
    whether the id they're targeting is in the integration's reach.

    The 2025-09 API rejects `filter.value="database"` here; the only
    valid object types for the search filter are "page" and
    "data_source".
    """
    try:
        result = notion.search(
            filter={"property": "object", "value": "data_source"},
            page_size=20,
        )
    except Exception as exc:
        return [{"id": "?", "column_count": -1, "title": f"search failed: {exc}"}]

    out: list[dict] = []
    for ds in result.get("results", []):
        # Data source titles come back as the `name` field (a rich-text
        # array) — keep `title` as the diagnostic key so the rendering
        # code stays stable.
        name_blocks = ds.get("name") or ds.get("title") or []
        title = "".join(b.get("plain_text", "") for b in name_blocks) or "(untitled)"
        out.append({
            "id": ds.get("id", "?"),
            "column_count": len(ds.get("properties") or {}),
            "title": title,
        })
    return out


# ─── Event fetch ──────────────────────────────────────────────────────────────

EVENTS_FOR_EVAL = """
query EventsForEval {
  events {
    id
    title
    description
    firstSignalCreatedAt
  }
}
"""


def _fetch_recent_events(limit: int) -> list[dict]:
    """Pull events from clear-api, sort by firstSignalCreatedAt desc,
    keep the most recent N with a non-empty description. Skipping
    empty-description rows is the right call — they'd produce a Notion
    row with nothing for a human to grade."""
    data = graphql._execute(EVENTS_FOR_EVAL)  # type: ignore[attr-defined]
    events = data.get("events") or []
    events = [e for e in events if (e.get("description") or "").strip()]
    events.sort(key=lambda e: e.get("firstSignalCreatedAt") or "", reverse=True)
    return events[:limit]


# ─── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Translate last N event descriptions into Arabic with each "
                    "supported LLM, write results to a Notion DB.",
    )
    parser.add_argument("--limit", type=int, default=20,
                        help="Number of recent events to translate. Default: 20.")
    parser.add_argument(
        "--models",
        default=",".join(m.key for m in MODELS),
        help=f"Comma-separated model keys to run. "
             f"Available: {','.join(m.key for m in MODELS)}.",
    )
    parser.add_argument(
        "--db-id",
        default=os.environ.get(
            "NOTION_TRANSLATION_DB_ID",
            "3815043895cf80d4aede000c05ea6334",
        ),
        help="Notion database id to write rows into. Defaults to the env "
             "var NOTION_TRANSLATION_DB_ID, then the hardcoded eval DB.",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Translate to stdout only; skip Notion writes.")
    args = parser.parse_args()

    wanted_keys = {k.strip() for k in args.models.split(",") if k.strip()}
    selected = [m for m in MODELS if m.key in wanted_keys]
    if not selected:
        logger.error(
            "No matching models. Asked for %s; available: %s",
            sorted(wanted_keys), [m.key for m in MODELS],
        )
        sys.exit(2)

    # Fail fast on missing SDKs — running 20 events × 6 models only to
    # discover at row 3 that openai isn't installed is the worst version
    # of this error. Surface it once, with the exact install command.
    missing = check_vendor_dependencies(selected)
    if missing:
        pip_args = " ".join(missing)
        logger.error(
            "Selected models need SDKs that aren't installed: %s\n"
            "Install everything in one go with:\n"
            "    uv pip install -e \".[eval]\"\n"
            "Or only the missing ones:\n"
            "    uv pip install %s",
            ", ".join(missing), pip_args,
        )
        sys.exit(2)

    logger.info(
        "Eval starting: limit=%d models=%s db=%s dry_run=%s",
        args.limit, [m.key for m in selected], args.db_id, args.dry_run,
    )

    events = _fetch_recent_events(args.limit)
    if not events:
        logger.error("No events with descriptions found in clear-api.")
        sys.exit(2)
    logger.info("Pulled %d event description(s) from clear-api.", len(events))

    notion = None
    db_schema: dict = {}
    resolved_data_source_id = args.db_id
    if not args.dry_run:
        notion = _connect_notion()
        # The id you paste might be a data source id, a database id, a
        # page id, a view id, or a full URL — auto-resolve to the
        # data_source_id that backs the schema. (Notion's 2025-09 API
        # moved properties off the database object and onto its data
        # sources; everything below talks to the data source.)
        resolved_data_source_id = _resolve_data_source_id(notion, args.db_id)
        db_schema = _fetch_data_source_schema(notion, resolved_data_source_id)
        logger.info(
            "Notion data source %s loaded: %d column(s) — %s",
            resolved_data_source_id, len(db_schema), sorted(db_schema.keys()),
        )
        # Self-heal: add any required columns the data source is missing
        # and rename the default title column to "English" so the row
        # title carries the English text. No-op when everything's
        # already in place.
        db_schema = _ensure_required_columns(notion, resolved_data_source_id, db_schema)

    pairs = [(event, model) for event in events for model in selected]
    logger.info(
        "Total translations to produce: %d events × %d models = %d",
        len(events), len(selected), len(pairs),
    )

    written = 0
    failed = 0
    # If Notion writes start failing in a row, abort early — translation
    # calls are expensive and we don't want to spend 119 of them just to
    # discover the first one couldn't write.
    consecutive_notion_failures = 0
    NOTION_FAILURE_ABORT_THRESHOLD = 5
    for i, (event, model) in enumerate(pairs, start=1):
        english = (event.get("description") or "").strip()
        if not english:
            continue

        logger.info(
            "[%d/%d] event=%s model=%s len=%d",
            i, len(pairs), event["id"], model.key, len(english),
        )
        arabic = translate(model, english)
        if not arabic:
            failed += 1
            continue

        if args.dry_run:
            preview = arabic.replace("\n", " ")[:120]
            logger.info("    → %s%s", preview, "…" if len(arabic) > 120 else "")
            continue

        try:
            props = _build_properties(db_schema, english, arabic, model.label)
            notion.pages.create(  # type: ignore[union-attr]
                # 2025-09 API: rows belong to a data source, not the
                # parent database. `database_id` parents still work for
                # legacy DBs but get rejected for new ones.
                parent={"data_source_id": resolved_data_source_id},
                properties=props,
            )
            written += 1
            consecutive_notion_failures = 0
        except Exception as exc:
            failed += 1
            consecutive_notion_failures += 1
            logger.error("    Notion write failed: %s", exc)
            if consecutive_notion_failures >= NOTION_FAILURE_ABORT_THRESHOLD:
                logger.error(
                    "Aborting: %d consecutive Notion writes have failed. "
                    "Most often this is a permission or DB-id mismatch — "
                    "fix that before burning more LLM credits. "
                    "(written=%d so far; remaining=%d skipped)",
                    consecutive_notion_failures, written, len(pairs) - i,
                )
                break

        # Notion's rate limit is ~3 requests/second. A small sleep keeps
        # the script polite without slowing the 100-page run materially.
        time.sleep(0.4)

    logger.info(
        "Done. Translations attempted=%d written=%d failed=%d",
        len(pairs), written, failed,
    )


if __name__ == "__main__":
    main()
