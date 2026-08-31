"""LangSmith call tracing -- CLAUDE.md section 15.

No network and no real LangSmith key anywhere: the disabled path is the
default (conftest clears the env vars), and the enabled path is tested
by patching app.llm.tracing._enabled / _publish.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from app.llm import providers, tracing
from app.llm.providers import DataSensitivity, LLMRequest, RateLimitError


def _request(**over):
    base = dict(instructions="score this", data="<data>client text</data>", temperature=0.0)
    base.update(over)
    return LLMRequest(**base)


def _response(content="ok", tool_calls=None):
    return SimpleNamespace(
        content=content,
        parsed={"tool_calls": tool_calls} if tool_calls else None,
        model="llama-3.3-70b-versatile",
        provider_name="groq",
        finish_reason="stop",
        raw=None,
    )


@pytest.fixture
def captured(monkeypatch):
    """Enable tracing and capture the records that would be published."""
    records = []
    monkeypatch.setattr(tracing, "_enabled", lambda: True)
    monkeypatch.setattr(tracing, "_publish", records.append)
    return records


# --- disabled path (the default) --------------------------------------


def test_disabled_tracing_is_a_pure_no_op(monkeypatch):
    calls = []
    monkeypatch.setattr(tracing, "_enabled", lambda: False)
    monkeypatch.setattr(tracing, "_publish", lambda record: calls.append(record))

    with tracing.record_llm_call(request=_request(), sensitivity=DataSensitivity.REAL, attempt="groq-primary") as span:
        span.set_response(_response())

    assert calls == []


def test_disabled_tracing_still_propagates_the_wrapped_exception(monkeypatch):
    monkeypatch.setattr(tracing, "_enabled", lambda: False)

    with pytest.raises(ValueError):
        with tracing.record_llm_call(request=_request(), sensitivity=DataSensitivity.SYNTHETIC, attempt="x"):
            raise ValueError("boom")


# --- enabled path: the audit payload ---------------------------------


def test_a_successful_call_records_prompt_response_and_metadata(captured):
    with tracing.record_llm_call(
        request=_request(temperature=0.2), sensitivity=DataSensitivity.REAL, attempt="groq-primary"
    ) as span:
        span.set_response(_response(content="done", tool_calls=[{"function": {"name": "f", "arguments": "{}"}}]))

    assert len(captured) == 1
    record = captured[0]
    assert record["name"] == "llm.groq-primary"
    assert record["inputs"]["instructions"] == "score this"
    assert record["inputs"]["data"] == "<data>client text</data>"
    assert record["inputs"]["temperature"] == 0.2
    assert record["tags"] == ["real", "groq-primary"]
    assert record["metadata"]["data_sensitivity"] == "real"
    assert record["metadata"]["attempt"] == "groq-primary"
    assert record["metadata"]["request_fingerprint"]
    assert "latency_seconds" in record["metadata"] and "ended_at" in record["metadata"]
    assert record["outputs"]["content"] == "done"
    assert record["outputs"]["tool_calls"] == [{"function": {"name": "f", "arguments": "{}"}}]
    assert record["outputs"]["model"] == "llama-3.3-70b-versatile"
    assert record["error"] is None


def test_a_failed_call_records_the_error_and_re_raises(captured):
    with pytest.raises(RateLimitError):
        with tracing.record_llm_call(
            request=_request(), sensitivity=DataSensitivity.SYNTHETIC, attempt="groq-retry"
        ):
            raise RateLimitError("429")

    assert len(captured) == 1
    assert captured[0]["error"] == "RateLimitError: 429"
    assert captured[0]["outputs"] is None
    assert captured[0]["metadata"]["data_sensitivity"] == "synthetic"


def test_a_publish_failure_is_swallowed_not_raised(monkeypatch):
    monkeypatch.setattr(tracing, "_enabled", lambda: True)
    monkeypatch.setattr(tracing, "_publish", lambda record: (_ for _ in ()).throw(RuntimeError("langsmith down")))

    got = {}
    with tracing.record_llm_call(request=_request(), sensitivity=DataSensitivity.REAL, attempt="x") as span:
        span.set_response(_response())
        got["reached_body_end"] = True

    assert got["reached_body_end"] is True  # the call completed normally despite the publish error


# --- helpers -------------------------------------------------------


def test_request_fingerprint_is_stable_and_parameter_sensitive():
    a = tracing._request_fingerprint(_request(temperature=0.0, max_tokens=500))
    assert a == tracing._request_fingerprint(_request(temperature=0.0, max_tokens=500))
    assert a != tracing._request_fingerprint(_request(temperature=0.7, max_tokens=500))
    assert a != tracing._request_fingerprint(_request(temperature=0.0, max_tokens=800))


def test_response_outputs_handles_none_and_a_bare_response():
    assert tracing._response_outputs(None) is None
    out = tracing._response_outputs(_response())
    assert out["provider"] == "groq" and out["tool_calls"] is None


@pytest.mark.parametrize(
    "env, expected",
    [
        ({}, False),
        ({"LANGSMITH_API_KEY": "k"}, True),
        ({"LANGCHAIN_API_KEY": "k"}, True),
        ({"LANGSMITH_API_KEY": "k", "LANGSMITH_TRACING": "false"}, False),
        ({"LANGSMITH_API_KEY": "k", "LANGSMITH_TRACING": "true"}, True),
    ],
)
def test_enabled_reads_the_environment(monkeypatch, env, expected):
    for var in ("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY", "LANGSMITH_TRACING"):
        monkeypatch.delenv(var, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    assert tracing._enabled() is expected


# --- integration with the provider router ---------------------------


class _FakeProvider:
    def __init__(self, script):
        self._script = list(script)
        self.calls = 0

    def complete(self, request):
        self.calls += 1
        item = self._script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


@pytest.fixture
def traced_attempts(monkeypatch):
    attempts = []

    @contextmanager
    def fake_record(*, request, sensitivity, attempt):
        entry = {"attempt": attempt, "sensitivity": sensitivity, "ok": None}
        span = SimpleNamespace(set_response=lambda r: entry.__setitem__("ok", True))
        try:
            yield span
        except BaseException:
            entry["ok"] = False
            raise
        finally:
            attempts.append(entry)

    monkeypatch.setattr(providers.tracing, "record_llm_call", fake_record)
    return attempts


def test_get_completion_traces_a_single_attempt_on_success(monkeypatch, traced_attempts):
    fake = _FakeProvider([_response()])
    monkeypatch.setattr(providers, "GroqProvider", lambda: fake)

    providers.get_completion(DataSensitivity.REAL, _request())

    assert [a["attempt"] for a in traced_attempts] == ["groq-primary"]
    assert traced_attempts[0]["ok"] is True


def test_get_completion_traces_every_attempt_through_the_retry_and_fallback(monkeypatch, traced_attempts):
    groq_fake = _FakeProvider([RateLimitError("429"), RateLimitError("429")])
    gemini_fake = _FakeProvider([_response(content="from gemini")])
    monkeypatch.setattr(providers, "GroqProvider", lambda: groq_fake)
    monkeypatch.setattr(providers, "_get_fallback_provider", lambda sensitivity: gemini_fake)

    providers.get_completion(DataSensitivity.SYNTHETIC, _request())

    assert [a["attempt"] for a in traced_attempts] == ["groq-primary", "groq-retry", "gemini-fallback"]
    assert [a["ok"] for a in traced_attempts] == [False, False, True]


def test_real_data_rate_limited_traces_both_groq_attempts_then_raises(monkeypatch, traced_attempts):
    groq_fake = _FakeProvider([RateLimitError("429"), RateLimitError("429")])
    monkeypatch.setattr(providers, "GroqProvider", lambda: groq_fake)

    with pytest.raises(providers.ProviderUnavailableError):
        providers.get_completion(DataSensitivity.REAL, _request())

    assert [a["attempt"] for a in traced_attempts] == ["groq-primary", "groq-retry"]
