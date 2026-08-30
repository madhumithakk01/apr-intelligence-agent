"""Proves the one rule branch 1 exists to enforce: real client data can
never route to Gemini, including under the one condition (persistent Groq
rate-limiting) where a fallback exists at all. No network calls, no real
API keys — the SDK client classes are monkeypatched with fakes."""

import dataclasses
from types import SimpleNamespace
from unittest.mock import MagicMock

import groq
import httpx
import pytest
from google.genai.errors import ClientError as GeminiClientError

from app.llm import providers
from app.llm.providers import (
    DataSensitivity,
    GeminiProvider,
    LLMRequest,
    ProviderUnavailableError,
    RealDataRoutingError,
    get_completion,
)


def _groq_rate_limit_error() -> groq.RateLimitError:
    response = httpx.Response(status_code=429, request=httpx.Request("POST", "http://test"))
    return groq.RateLimitError("rate limited", response=response, body=None)


def _groq_auth_error() -> groq.AuthenticationError:
    response = httpx.Response(status_code=401, request=httpx.Request("POST", "http://test"))
    return groq.AuthenticationError("invalid api key", response=response, body=None)


class _FakeGroqClient:
    """Stand-in for groq.Groq(). `responses` is consumed in call order —
    each item is either an exception instance to raise or a return value."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.call_count = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.call_count += 1
        item = self._responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def _groq_success(content="ok", tool_calls=None, finish_reason="stop"):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice])


def _patch_groq_client(monkeypatch, responses):
    fake_client = _FakeGroqClient(responses)
    monkeypatch.setattr(providers.groq, "Groq", lambda **kwargs: fake_client)
    return fake_client


def _gemini_rate_limit_error() -> GeminiClientError:
    return GeminiClientError(code=429, response_json={"error": {"message": "rate limited"}})


class _FakeGeminiClient:
    """Stand-in for genai.Client(). `responses` is consumed in call order —
    each item is either an exception instance to raise or a return value."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.call_count = 0
        self.models = SimpleNamespace(generate_content=self._generate_content)

    def _generate_content(self, **kwargs):
        self.call_count += 1
        item = self._responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def _gemini_success(text="ok"):
    return SimpleNamespace(text=text, candidates=[SimpleNamespace(finish_reason="STOP")])


def _patch_gemini_client(monkeypatch, responses):
    fake_client = _FakeGeminiClient(responses)
    monkeypatch.setattr(providers.genai, "Client", lambda **kwargs: fake_client)
    return fake_client


@pytest.fixture(autouse=True)
def _dummy_env(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")


def _request():
    return LLMRequest(instructions="classify this", data="client cell value")


def test_real_data_success_never_touches_gemini(monkeypatch):
    _patch_groq_client(monkeypatch, [_groq_success("real result")])
    gemini_mock = MagicMock()
    monkeypatch.setattr(providers, "GeminiProvider", gemini_mock)

    response = get_completion(DataSensitivity.REAL, _request())

    assert response.provider_name == "groq"
    assert response.content == "real result"
    gemini_mock.assert_not_called()


def test_real_data_persistent_rate_limit_raises_and_never_constructs_gemini(monkeypatch):
    _patch_groq_client(monkeypatch, [_groq_rate_limit_error(), _groq_rate_limit_error()])
    gemini_mock = MagicMock()
    monkeypatch.setattr(providers, "GeminiProvider", gemini_mock)

    with pytest.raises(ProviderUnavailableError):
        get_completion(DataSensitivity.REAL, _request())

    gemini_mock.assert_not_called()


def test_gemini_provider_rejects_real_data_at_construction():
    with pytest.raises(RealDataRoutingError):
        GeminiProvider(sensitivity=DataSensitivity.REAL)


def test_synthetic_persistent_rate_limit_falls_back_to_gemini(monkeypatch):
    fake_groq = _patch_groq_client(
        monkeypatch, [_groq_rate_limit_error(), _groq_rate_limit_error()]
    )
    _patch_gemini_client(monkeypatch, [_gemini_success("fallback result")])

    response = get_completion(DataSensitivity.SYNTHETIC, _request())

    assert response.provider_name == "gemini"
    assert response.content == "fallback result"
    assert fake_groq.call_count == 2


def test_real_data_rate_limited_once_then_succeeds_on_retry(monkeypatch):
    fake_groq = _patch_groq_client(
        monkeypatch, [_groq_rate_limit_error(), _groq_success("retry result")]
    )
    gemini_mock = MagicMock()
    monkeypatch.setattr(providers, "GeminiProvider", gemini_mock)

    response = get_completion(DataSensitivity.REAL, _request())

    assert response.provider_name == "groq"
    assert response.content == "retry result"
    assert fake_groq.call_count == 2
    gemini_mock.assert_not_called()


def test_non_rate_limit_error_propagates_immediately_no_retry_no_fallback(monkeypatch):
    fake_groq = _patch_groq_client(monkeypatch, [_groq_auth_error()])
    gemini_mock = MagicMock()
    monkeypatch.setattr(providers, "GeminiProvider", gemini_mock)

    with pytest.raises(groq.AuthenticationError):
        get_completion(DataSensitivity.SYNTHETIC, _request())

    assert fake_groq.call_count == 1
    gemini_mock.assert_not_called()


def test_synthetic_success_also_prefers_groq(monkeypatch):
    _patch_groq_client(monkeypatch, [_groq_success("synthetic result")])
    gemini_mock = MagicMock()
    monkeypatch.setattr(providers, "GeminiProvider", gemini_mock)

    response = get_completion(DataSensitivity.SYNTHETIC, _request())

    assert response.provider_name == "groq"
    gemini_mock.assert_not_called()


def test_llm_request_has_no_sensitivity_field():
    field_names = {f.name for f in dataclasses.fields(LLMRequest)}
    assert "sensitivity" not in field_names
