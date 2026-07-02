"""Anthropic Claude API wrapper for structured ML operations.

Rate-limit / transient-error handling:
  - The Anthropic SDK auto-retries 408/409/429/5xx with exponential backoff.
    We bump max_retries so the SDK absorbs most bursts without propagating.
  - If the SDK still exhausts retries, `call_claude` catches `RateLimitError`
    and re-raises it as `ClaudeRateLimited`, which carries a suggested
    `retry_after` seconds. Callers (Celery tasks) should use this value when
    scheduling their own retry so they don't re-hit the same ceiling.
"""

import json
import logging
import random
import re
import time

import anthropic

from src.clients.insights import record_call
from src.config import settings

logger = logging.getLogger(__name__)

_client: anthropic.Anthropic | None = None


class ClaudeRateLimited(Exception):
    """Raised when Claude returns 429 even after SDK-level retries.
    `retry_after` is the recommended delay (seconds) before retrying."""

    def __init__(self, message: str, retry_after: float) -> None:
        super().__init__(message)
        self.retry_after = retry_after


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        # Keep the SDK default of max_retries=2. Previously we bumped this
        # to 5 to absorb transient idle-connection drops during long
        # `.create()` calls, but switching call_claude to `.stream()`
        # (below) eliminates that failure mode — the connection stays
        # warm via periodic events. Extra retries now just widen the
        # window in which a duplicate delivery could pile on before the
        # dedup lock catches it.
        #
        # timeout=600s (10 min) matches Anthropic's own guidance for
        # long-running requests. Streaming makes this a per-idle-period
        # ceiling in practice, so it very rarely fires.
        _client = anthropic.Anthropic(
            api_key=settings.anthropic_api_key,
            timeout=600.0,
        )
    return _client


def _retry_after_from_error(err: Exception) -> float:
    """Extract a retry-after delay from an Anthropic error, with fallback.

    Anthropic 429 responses include a `retry-after` header (seconds). We add
    a small jitter so parallel workers don't all wake up simultaneously.
    """
    default = 30.0
    retry_after = default

    resp = getattr(err, "response", None)
    if resp is not None:
        headers = getattr(resp, "headers", None) or {}
        raw = headers.get("retry-after") or headers.get("Retry-After")
        if raw:
            try:
                retry_after = float(raw)
            except (TypeError, ValueError):
                pass

    # Jitter: ±25% to avoid thundering-herd
    jitter = retry_after * 0.25
    return retry_after + random.uniform(-jitter, jitter)


def _extract_json(text: str) -> str | None:
    """Extract a JSON object from a response that may contain prose or markdown.

    Strategies (in order):
      1. Already pure JSON
      2. JSON inside ```json ... ``` or ``` ... ``` markdown fences
      3. First balanced { ... } object found in the text
    """
    text = text.strip()

    # 1. Pure JSON
    if text.startswith("{") and text.endswith("}"):
        return text

    # 2. Markdown code fence
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL | re.IGNORECASE)
    if fence_match:
        inner = fence_match.group(1).strip()
        if inner.startswith("{"):
            return inner

    # 3. First balanced { ... } object in the text
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return None


def model_for_stage(stage: str | None) -> str:
    """Resolve the model to use for a given stage. Stage-specific override
    in settings wins; falls back to settings.claude_model."""
    if not stage:
        return settings.claude_model
    override = getattr(settings, f"claude_model_{stage}", "") or ""
    return override or settings.claude_model


def call_claude(
    system_prompt: str,
    user_prompt: str,
    *,
    stage: str | None = None,
    prompt_version: str | None = None,
    signal_id: str | None = None,
    event_id: str | None = None,
    model: str | None = None,
    max_tokens: int = 1024,
) -> dict:
    """
    Call Claude with a system + user prompt and parse JSON response.

    The system prompt should instruct Claude to respond with valid JSON.

    Model selection priority:
      1. explicit `model=` parameter
      2. settings.claude_model_<stage> if set
      3. settings.claude_model (global default)

    Telemetry: when stage and prompt_version are passed, every call (success,
    parse failure, or API failure including rate-limit) is reported to the
    insights dashboard via record_call(). Telemetry is fire-and-forget —
    never raises.

    Raises:
      ClaudeRateLimited: 429 after SDK retries. Callers should reschedule
        the Celery task with `countdown=err.retry_after`.
      anthropic.APIStatusError: any other non-retryable API error.
      json.JSONDecodeError: model output was unparseable even after
        JSON-extraction fallbacks.
    """
    client = _get_client()
    chosen_model = model or model_for_stage(stage)
    started = time.monotonic()
    raw_text = ""
    parsed: dict | None = None
    parse_error: str | None = None
    api_error: BaseException | None = None
    rate_limit_after: float | None = None
    usage_dict: dict[str, int | None] = {}

    try:
        # Use streaming per Anthropic's guidance for long-running requests:
        #   > Consider using the streaming Messages API [...] for long
        #   > running requests, especially those over 10 minutes. Avoid
        #   > setting a large max_tokens value without using the streaming
        #   > Messages API [...] Some networks may drop idle connections
        #   > after a variable period of time, which can cause the request
        #   > to fail or timeout without receiving a response from
        #   > Anthropic.
        # For our translate stage (max_tokens=16384, 2 locales × 4 fields)
        # the 60s idle-drop was reliably firing pre-fix and driving the
        # SDK's max_retries=5 loop that showed up as a "Retrying request
        # to /v1/messages in Xs" cascade on both workers processing the
        # same crisis. `.stream()` keeps the connection warm via periodic
        # events; `.get_final_message()` reassembles the full Message
        # object at the end so the rest of this function (usage,
        # stop_reason, content[0].text) sees exactly the same shape it
        # got from `.create()`.
        with client.messages.stream(
            model=chosen_model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        ) as stream:
            response = stream.get_final_message()

        # Message.content is a list of ContentBlock variants — TextBlock
        # (`.text`), ToolUseBlock (no `.text`), ThinkingBlock (`.thinking`,
        # no `.text`), etc. Today's prompts always return a single
        # TextBlock, but this call site is shared across every stage
        # (classify, group, translate, needs, scenarios) and any future
        # config change (extended thinking, tool use, multi-block
        # responses) would silently break a `response.content[0].text`
        # lookup:
        #   - `content == []`     → IndexError
        #   - `content[0]` is ThinkingBlock → AttributeError on `.text`
        #   - multiple TextBlocks → only the first is read, tail is dropped
        # `getattr(block, "text", "")` skips non-text blocks; the join
        # concatenates any interleaved text (thinking blocks contribute
        # empty strings). Safe on both today's shape and any of the
        # above evolutions.
        raw_text = "".join(
            getattr(block, "text", "") for block in (response.content or [])
        ).strip()
        # Translate Anthropic SDK usage names → insights API names
        u = response.usage
        usage_dict = {
            "input_tokens": getattr(u, "input_tokens", None),
            "output_tokens": getattr(u, "output_tokens", None),
            "cache_read_tokens": getattr(u, "cache_read_input_tokens", None),
            "cache_create_tokens": getattr(u, "cache_creation_input_tokens", None),
        }

        # `stop_reason == "max_tokens"` means Claude stopped mid-output
        # because the budget ran out. The text is then truncated mid-
        # token, the JSON is unclosed, and `_extract_json` can't recover
        # anything useful. Detect it explicitly so the failure message
        # points at the right knob to turn (the caller's `max_tokens`)
        # instead of pretending the model misbehaved.
        stop_reason = getattr(response, "stop_reason", None)
        truncated = stop_reason == "max_tokens"

        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError:
            if truncated:
                parse_error = (
                    f"Response truncated by max_tokens={max_tokens} "
                    f"(stop_reason=max_tokens). Raise max_tokens or shrink the prompt."
                )
            else:
                extracted = _extract_json(raw_text)
                if extracted:
                    try:
                        parsed = json.loads(extracted)
                    except json.JSONDecodeError as e:
                        parse_error = f"JSONDecodeError after extraction: {e}"
                else:
                    parse_error = "Could not extract valid JSON from response"
        if truncated and parsed is not None:
            # We somehow got valid JSON despite a max_tokens stop — extremely
            # unlikely but logging it so we'd notice if the heuristic ever
            # misfires for a future model that emits trailing whitespace
            # before terminating.
            logger.warning(
                "[CLAUDE] stop_reason=max_tokens but response parsed cleanly; "
                "treating as success. Consider raising max_tokens (current=%d).",
                max_tokens,
            )
    except anthropic.RateLimitError as err:
        rate_limit_after = _retry_after_from_error(err)
        logger.warning(
            "[CLAUDE] Rate-limited (after SDK retries). Will suggest retry_after=%.1fs. Error: %s",
            rate_limit_after, err,
        )
        api_error = err
        parse_error = f"RateLimitError: {err}"
    except anthropic.APIStatusError as err:
        logger.error("[CLAUDE] API error: status=%s message=%s", err.status_code, err)
        api_error = err
        parse_error = f"{type(err).__name__}: {err}"
    except Exception as exc:
        api_error = exc
        parse_error = f"{type(exc).__name__}: {exc}"

    latency_ms = int((time.monotonic() - started) * 1000)

    if stage and prompt_version:
        try:
            record_call(
                stage=stage,
                prompt_version=prompt_version,
                model=chosen_model,
                signal_id=signal_id,
                event_id=event_id,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                raw_response=raw_text,
                parsed_response=parsed,
                parse_error=parse_error,
                usage=usage_dict,
                latency_ms=latency_ms,
            )
        except Exception as telemetry_exc:
            logger.warning("[insights] record_call raised unexpectedly: %s", telemetry_exc)

    if rate_limit_after is not None:
        raise ClaudeRateLimited(str(api_error), retry_after=rate_limit_after) from api_error
    if api_error is not None:
        raise api_error
    if parsed is None:
        logger.error(
            "Failed to parse Claude response as JSON. Full response (first 500 chars): %s",
            raw_text[:500],
        )
        raise json.JSONDecodeError(
            "Could not extract valid JSON from Claude response", raw_text, 0
        )
    return parsed
