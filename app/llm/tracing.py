"""LangSmith call tracing -- CLAUDE.md section 15.

Every ``get_completion`` attempt posts one LangSmith run: the prompt
(both the code-authored instructions and the delimited client data), the
response, the model and provider actually used, the data-sensitivity
flag, timing, and a fingerprint of the request parameters. That is the
audit trail that makes a report number reconstructable months later --
not a separate logging system to hand-build.

No-op unless ``LANGSMITH_API_KEY`` (or ``LANGCHAIN_API_KEY``) is set:
local runs and the test suite neither need a key nor slow down without
one, and ``LANGSMITH_TRACING=false`` forces it off even when a key is
present. Any failure talking to LangSmith is swallowed with a warning --
the trail matters, but the LLM call's result is what the pipeline
actually needs, so tracing never blocks or breaks a call.

LangSmith stores client prompts and responses verbatim, so its project
falls under the same bid-outcome deletion trigger as the checkpoint
store and the DB rows (CLAUDE.md section 2) -- managed in the LangSmith
project itself, out of band from this code.

This module never imports app.llm.providers (that would be circular): it
duck-types the request/response objects and takes the sensitivity value
directly.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)

_DEFAULT_PROJECT = "apr-intelligence-agent"
_DISABLED_VALUES = {"false", "0", "no", "off"}


def _enabled() -> bool:
    if not (os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")):
        return False
    if os.getenv("LANGSMITH_TRACING", "").strip().lower() in _DISABLED_VALUES:
        return False
    return True


def _request_fingerprint(request: Any) -> str:
    """A short stable hash of the parameters that shape a call -- so two
    runs that used the same settings are visibly comparable. Not a
    security control; just a grouping key."""
    payload = {
        "temperature": getattr(request, "temperature", None),
        "max_tokens": getattr(request, "max_tokens", None),
        "response_format": getattr(request, "response_format", None),
        "tools": sorted(
            (tool.get("function", {}) or {}).get("name", "")
            for tool in (getattr(request, "tools", None) or [])
        ),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
    return digest[:12]


def _response_outputs(response: Any) -> Optional[dict]:
    if response is None:
        return None
    return {
        "content": getattr(response, "content", None),
        "tool_calls": (getattr(response, "parsed", None) or {}).get("tool_calls"),
        "model": getattr(response, "model", None),
        "provider": getattr(response, "provider_name", None),
        "finish_reason": getattr(response, "finish_reason", None),
    }


class _Span:
    """Handed to the caller inside the context so it can attach the
    response it got. Nothing else."""

    __slots__ = ("_response",)

    def __init__(self) -> None:
        self._response = None

    def set_response(self, response: Any) -> None:
        self._response = response


def _publish(record: dict) -> None:
    """Send one completed run to LangSmith. Isolated so a test can
    substitute it and assert the audit payload without the SDK or the
    network."""
    from langsmith.run_trees import RunTree

    run = RunTree(
        name=record["name"],
        run_type="llm",
        inputs=record["inputs"],
        tags=record["tags"],
        project_name=os.getenv("LANGSMITH_PROJECT") or _DEFAULT_PROJECT,
        extra={"metadata": record["metadata"]},
    )
    run.post()
    run.end(outputs=record.get("outputs"), error=record.get("error"))
    run.patch()


@contextmanager
def record_llm_call(*, request: Any, sensitivity: Any, attempt: str) -> Iterator[_Span]:
    """Wrap one provider call. Yields a span the caller uses to attach
    the response; on exit, posts the run (prompt, response or error,
    timing, sensitivity). A no-op when tracing is disabled. Never
    suppresses an exception from the wrapped call, and never raises one
    of its own."""
    span = _Span()
    if not _enabled():
        yield span
        return

    sensitivity_value = getattr(sensitivity, "value", None) or str(sensitivity)
    started = time.time()
    record: dict = {
        "name": f"llm.{attempt}",
        "inputs": {
            "instructions": getattr(request, "instructions", None),
            "data": getattr(request, "data", None),
            "tools": getattr(request, "tools", None),
            "tool_choice": getattr(request, "tool_choice", None),
            "temperature": getattr(request, "temperature", None),
            "max_tokens": getattr(request, "max_tokens", None),
            "response_format": getattr(request, "response_format", None),
        },
        "tags": [sensitivity_value, attempt],
        "metadata": {
            "data_sensitivity": sensitivity_value,
            "attempt": attempt,
            "request_fingerprint": _request_fingerprint(request),
        },
    }

    error: Optional[str] = None
    try:
        yield span
    except BaseException as exc:  # record the failed attempt, then let it propagate
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        record["outputs"] = _response_outputs(span._response)
        record["error"] = error
        record["metadata"]["latency_seconds"] = round(time.time() - started, 3)
        record["metadata"]["ended_at"] = datetime.now(timezone.utc).isoformat()
        try:
            _publish(record)
        except Exception as exc:  # noqa: BLE001 -- tracing must never break the call
            logger.warning("LangSmith trace publish failed for %s: %s", record["name"], exc)
