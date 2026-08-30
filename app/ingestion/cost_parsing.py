"""Safe numeric parsing for ingested cost/count cells (CLAUDE.md section 4.6).

Resolution order, each step only reached if the previous one declines:
  1. Direct numeric parse (covers the common case -- pandas already yields
     clean numeric dtypes for most real cells).
  2. Deterministic refusal-keyword match (see validators.is_refusal_text)
     -- "cannot disclose" etc. This is a business decision, not a
     data-quality defect (CLAUDE.md section 2): anything matching is
     marked withheld immediately and never reaches step 4. This is a
     keyword search, not an exhaustive phrase list, precisely so free-text
     variation on a known refusal ("client declined to disclose") is still
     caught here rather than depending on step 4's prompt compliance.
  3. Deterministic format normalization -- currency symbols, US-style
     thousands separators, k/m/b suffixes. Declines rather than guess on
     anything it isn't confident about (e.g. it will not attempt to
     disambiguate reversed decimal/thousands separator conventions).
  4. A narrow, single, structured LLM call for whatever survives steps
     1-3 -- constrained to report a number or "not a number," never free
     text, so it cannot invent a value for something that should have
     been caught at step 2. This is a second layer, not the only one:
     step 2's keyword search is the primary, code-enforced guarantee
     that withheld data is never sent here to be guessed at.
"""

import json
import logging
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

import pandas as pd

from app.ingestion.validators import is_refusal_text
from app.llm.providers import DataSensitivity, LLMRequest, get_completion

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParsedCost:
    value: Optional[float]
    status: Literal["parsed", "withheld", "unparsed"]
    raw_text: Optional[str]


_CURRENCY_SYMBOLS_RE = re.compile(r"[$€£₹¥]")
_CURRENCY_CODE_RE = re.compile(r"\b(USD|INR|EUR|GBP|JPY|CAD|AUD)\b", re.IGNORECASE)
_MULTIPLIER_SUFFIXES = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}
# Only matches unambiguous US-style grouping ("45,000", "45,000.00", "45000.5").
# Anything else (e.g. reversed EU-style "45.000,00") is left for the LLM
# fallback rather than guessed at.
_CLEAN_NUMBER_RE = re.compile(r"^\d{1,3}(,\d{3})*(\.\d+)?$")

REPORT_PARSED_COST_TOOL = {
    "type": "function",
    "function": {
        "name": "report_parsed_cost",
        "description": (
            "Report whether a spreadsheet cell is a real numeric value once "
            "formatting (currency symbols, separators, units) is normalized, "
            "and if so its literal numeric value."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "is_numeric": {
                    "type": "boolean",
                    "description": "True only if the cell represents a real number.",
                },
                "value": {
                    "type": ["number", "null"],
                    "description": "The numeric value, no currency or unit conversion. Null if is_numeric is false.",
                },
            },
            "required": ["is_numeric", "value"],
        },
    },
}


def _finite(value: float) -> Optional[float]:
    """inf/-inf/nan all parse as valid Python floats but are never a
    legitimate cost or FTE count -- and int(round(inf)) / int(nan) raise
    downstream. Reject them at every point a numeric value is produced."""
    return value if math.isfinite(value) else None


def _deterministic_normalize(text: str) -> Optional[float]:
    cleaned = _CURRENCY_CODE_RE.sub("", text)
    cleaned = _CURRENCY_SYMBOLS_RE.sub("", cleaned)
    cleaned = cleaned.strip()
    if not cleaned:
        return None

    multiplier = 1
    if cleaned[-1].lower() in _MULTIPLIER_SUFFIXES:
        multiplier = _MULTIPLIER_SUFFIXES[cleaned[-1].lower()]
        cleaned = cleaned[:-1].strip()

    if not cleaned or not _CLEAN_NUMBER_RE.match(cleaned):
        return None

    try:
        return float(cleaned.replace(",", "")) * multiplier
    except ValueError:
        return None


def normalize_ambiguous_cost(raw_text: str, *, field_name: str, application_id: str) -> Optional[float]:
    """Single structured call. Never raises for a malformed/unexpected
    response -- returns None, which the caller treats as unparsed."""
    request = LLMRequest(
        instructions=(
            "You are given the raw text of one spreadsheet cell that failed "
            "plain numeric parsing. Determine only whether it is a real "
            "numeric value in a non-standard format (currency symbols, "
            "separators, k/m/b suffixes, stray whitespace) and if so its "
            "literal numeric value -- no currency conversion, no unit "
            "conversion. Call report_parsed_cost exactly once. If the text "
            "is not a number -- including any refusal, placeholder, or "
            "non-numeric wording -- set is_numeric to false and value to null."
        ),
        data=raw_text,
        tools=[REPORT_PARSED_COST_TOOL],
        tool_choice={"type": "function", "function": {"name": "report_parsed_cost"}},
        temperature=0.0,
        max_tokens=60,
    )
    response = get_completion(DataSensitivity.REAL, request)

    tool_calls = (response.parsed or {}).get("tool_calls") or []
    if not tool_calls:
        return None
    try:
        arguments = json.loads(tool_calls[0]["function"]["arguments"])
    except (KeyError, TypeError, json.JSONDecodeError):
        return None

    if not arguments.get("is_numeric"):
        return None
    value = arguments.get("value")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    return _finite(float(value))


def _parse_numeric_cell(raw_value: Any, *, field_name: str, application_id: str) -> ParsedCost:
    if raw_value is None:
        return ParsedCost(value=None, status="unparsed", raw_text=None)
    try:
        if pd.isna(raw_value):
            return ParsedCost(value=None, status="unparsed", raw_text=None)
    except (TypeError, ValueError):
        pass

    if isinstance(raw_value, bool):
        return ParsedCost(value=None, status="unparsed", raw_text=str(raw_value))
    if isinstance(raw_value, (int, float)):
        finite_value = _finite(float(raw_value))
        if finite_value is None:
            return ParsedCost(value=None, status="unparsed", raw_text=str(raw_value))
        return ParsedCost(value=finite_value, status="parsed", raw_text=None)

    text = str(raw_value).strip()
    if not text:
        return ParsedCost(value=None, status="unparsed", raw_text=None)

    try:
        direct_value = float(text)
    except ValueError:
        direct_value = None
    if direct_value is not None:
        finite_value = _finite(direct_value)
        if finite_value is None:
            # "inf"/"nan"/"Infinity" etc. parse as valid floats but are
            # never a legitimate cost/count figure, and int(round(inf))
            # raises downstream -- treat as unparsed, not as ambiguous
            # (the LLM fallback exists for format ambiguity, not this).
            return ParsedCost(value=None, status="unparsed", raw_text=text)
        return ParsedCost(value=finite_value, status="parsed", raw_text=None)

    if is_refusal_text(text):
        return ParsedCost(value=None, status="withheld", raw_text=text)

    normalized = _deterministic_normalize(text)
    if normalized is not None:
        return ParsedCost(value=normalized, status="parsed", raw_text=None)

    try:
        llm_value = normalize_ambiguous_cost(text, field_name=field_name, application_id=application_id)
    except Exception as exc:
        # Broad on purpose: this branch's whole point is that ingestion
        # never crashes the batch, and that must hold for any provider
        # failure (rate limit, auth, network, an unexpected SDK exception),
        # not only the LLMProviderError subclasses this module defines.
        logger.warning(
            "LLM fallback unavailable for %s.%s (%r): %s -- marking unparsed",
            application_id,
            field_name,
            text,
            exc,
        )
        llm_value = None

    if llm_value is not None:
        return ParsedCost(value=llm_value, status="parsed", raw_text=None)
    return ParsedCost(value=None, status="unparsed", raw_text=text)


def parse_cost_cell(raw_value: Any, *, field_name: str, application_id: str) -> ParsedCost:
    return _parse_numeric_cell(raw_value, field_name=field_name, application_id=application_id)


def parse_fte_count(raw_value: Any, *, application_id: str) -> ParsedCost:
    return _parse_numeric_cell(raw_value, field_name="fte_count", application_id=application_id)


def build_numeric_field_notes(parsed: Dict[str, ParsedCost]) -> Optional[str]:
    notes = {
        field: {"status": result.status, "raw_text": result.raw_text}
        for field, result in parsed.items()
        if result.status != "parsed"
    }
    return json.dumps(notes) if notes else None
