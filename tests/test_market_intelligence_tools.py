"""Market intelligence search tool -- CLAUDE.md section 8.

Never touches a real network call: every test either mocks
app.market_intelligence.tools.TavilyClient or clears TAVILY_API_KEY.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.market_intelligence import tools


class _FakeTavilyClient:
    def __init__(self, response=None, exc=None, capture=None):
        self._response = response
        self._exc = exc
        self._capture = capture

    def search(self, **kwargs):
        if self._capture is not None:
            self._capture.append(kwargs)
        if self._exc is not None:
            raise self._exc
        return self._response


def test_no_api_key_returns_none_not_empty_list(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    assert tools.search("procurement software alternatives") is None


def test_successful_search_returns_parsed_results(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "fake-key")
    response = {
        "results": [
            {"title": "Vendor A Overview", "url": "https://a.example", "content": "A does procurement."},
            {"title": "Vendor B Overview", "url": "https://b.example", "content": "B does procurement too."},
        ]
    }
    monkeypatch.setattr(tools, "TavilyClient", lambda api_key: _FakeTavilyClient(response=response))

    results = tools.search("procurement software alternatives")

    assert results is not None
    assert len(results) == 2
    assert results[0].title == "Vendor A Overview"
    assert results[0].url == "https://a.example"


def test_empty_result_set_is_a_successful_empty_list_not_none(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "fake-key")
    monkeypatch.setattr(tools, "TavilyClient", lambda api_key: _FakeTavilyClient(response={"results": []}))

    results = tools.search("an extremely niche bespoke capability")

    assert results == []  # distinct from None -- see module docstring


def test_provider_exception_returns_none(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "fake-key")
    monkeypatch.setattr(
        tools, "TavilyClient", lambda api_key: _FakeTavilyClient(exc=RuntimeError("rate limited"))
    )

    assert tools.search("procurement software alternatives") is None


def test_unexpected_response_shape_returns_none(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "fake-key")
    monkeypatch.setattr(tools, "TavilyClient", lambda api_key: _FakeTavilyClient(response="not a dict"))

    assert tools.search("procurement software alternatives") is None


def test_a_non_dict_result_shape_is_skipped_rather_than_crashing(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "fake-key")
    response = {"results": ["not a dict", {"title": "Real result", "url": "https://a.example", "content": "x"}]}
    monkeypatch.setattr(tools, "TavilyClient", lambda api_key: _FakeTavilyClient(response=response))

    results = tools.search("procurement software alternatives")

    assert len(results) == 1
    assert results[0].title == "Real result"


def test_a_result_with_no_title_and_no_content_is_dropped(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "fake-key")
    response = {"results": [{"title": "", "url": "https://a.example", "content": ""}]}
    monkeypatch.setattr(tools, "TavilyClient", lambda api_key: _FakeTavilyClient(response=response))

    assert tools.search("procurement software alternatives") == []


def test_max_results_is_forwarded_to_the_client(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "fake-key")
    captured = []
    monkeypatch.setattr(
        tools, "TavilyClient", lambda api_key: _FakeTavilyClient(response={"results": []}, capture=captured)
    )

    tools.search("procurement software alternatives", max_results=3)

    assert captured[0]["max_results"] == 3


def test_as_dict_is_json_serializable():
    import json

    result = tools.SearchResult(title="A", url="https://a.example", content="x")
    json.dumps(result.as_dict())
