"""Search tool for the market intelligence agent -- SPEC.md section 8.

A thin wrapper over Tavily, in the same spirit as app.llm.providers: one
function every caller goes through, so the branch that actually gets
retried/mocked/swapped has exactly one place to be correct. Fully
replaces app.services.market_service's ~20-vendor regex whitelist
(SPEC.md section 4 bug 8) -- that approach structurally cannot
discover a vendor it wasn't hardcoded to recognize, which is the whole
reason this branch exists.

Search results are not gated by DataSensitivity the way an LLM call is
(SPEC.md section 11): that rule is specifically about client data
reaching a model provider's training pipeline (Google's free tier for
Gemini), and Tavily is a search API the caller queries, not a provider
client data is handed to for processing. What still matters is what
goes *into* the query text -- see
app.market_intelligence.segments for why a query is built from a
capability description and technology stack, never from cost figures or
anything else that would not belong in a request to an external search
engine.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from tavily import TavilyClient

from app.scoring import governance_params as gp

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    content: str

    def as_dict(self) -> Dict[str, Any]:
        return {"title": self.title, "url": self.url, "content": self.content}


def _client() -> Optional[TavilyClient]:
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        return None
    return TavilyClient(api_key=api_key)


def search(
    query: str, *, max_results: int = gp.MARKET_SEARCH_RESULTS_PER_QUERY
) -> Optional[List[SearchResult]]:
    """None on failure -- no API key configured, a network error, a rate
    limit, an unexpected response shape; a (possibly empty) list on
    success. This distinction is load-bearing, not incidental: SPEC.md
    section 8 keeps "failure/checkpoint" (an infra problem, resume the
    branch) and "diminishing returns" (a genuinely empty or exhausted
    result set, a confident conclusion) as two different stop
    conditions, and a function that collapsed both cases to the same []
    would make that distinction impossible for the caller to draw."""
    client = _client()
    if client is None:
        logger.warning("Market search unavailable: TAVILY_API_KEY not configured.")
        return None

    try:
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=max_results,
            include_raw_content=False,
        )
    except Exception as exc:
        logger.warning("Market search failed for query %r: %s", query, exc)
        return None

    results = response.get("results") if isinstance(response, dict) else None
    if not isinstance(results, list):
        logger.warning("Market search returned an unexpected response shape for query %r.", query)
        return None

    parsed = []
    for item in results:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        url = str(item.get("url") or "").strip()
        content = str(item.get("content") or "").strip()
        if not (title or content):
            continue
        parsed.append(SearchResult(title=title, url=url, content=content))
    return parsed
