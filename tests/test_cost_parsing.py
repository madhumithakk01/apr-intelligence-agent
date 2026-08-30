import json
from types import SimpleNamespace

import pytest

from app.ingestion import cost_parsing
from app.llm.providers import DataSensitivity, LLMProviderError


def _tool_call_response(is_numeric: bool, value):
    arguments = json.dumps({"is_numeric": is_numeric, "value": value})
    return SimpleNamespace(
        content="",
        parsed={"tool_calls": [{"function": {"name": "report_parsed_cost", "arguments": arguments}}]},
        model="llama-3.3-70b-versatile",
        provider_name="groq",
        finish_reason="tool_calls",
        raw=None,
    )


def _make_llm_mock(monkeypatch, return_value=None, side_effect=None):
    calls = []

    def fake_get_completion(sensitivity, request):
        calls.append((sensitivity, request))
        if side_effect is not None:
            raise side_effect
        return return_value

    monkeypatch.setattr(cost_parsing, "get_completion", fake_get_completion)
    return calls


@pytest.mark.parametrize(
    "raw_text",
    ["cannot disclose", "Confidential", "  N/A  ", "TBD", "cannot say", "withheld", "  UNKNOWN"],
)
def test_refusal_text_never_reaches_llm(monkeypatch, raw_text):
    calls = _make_llm_mock(monkeypatch)

    result = cost_parsing.parse_cost_cell(raw_text, field_name="annual_fte_cost", application_id="APP-1")

    assert result.status == "withheld"
    assert result.value is None
    assert result.raw_text == raw_text.strip()
    assert calls == []


@pytest.mark.parametrize(
    "raw_text",
    [
        "Cannot Disclose.",
        "client declined to disclose",
        "Confidential - internal only",
        "not disclosed at this time",
        "figures restricted",
    ],
)
def test_refusal_text_variations_never_reach_llm(monkeypatch, raw_text):
    """These aren't in the exact-phrase list -- they must still be caught
    deterministically by the keyword search, not left to the LLM prompt."""
    calls = _make_llm_mock(monkeypatch)

    result = cost_parsing.parse_cost_cell(raw_text, field_name="annual_fte_cost", application_id="APP-1")

    assert result.status == "withheld"
    assert result.value is None
    assert calls == []


@pytest.mark.parametrize(
    "raw_text",
    ["inf", "-inf", "Infinity", "-Infinity", "NaN", "nan", "1e400"],
)
def test_non_finite_text_never_crashes_and_never_reaches_llm(monkeypatch, raw_text):
    """float() happily parses these, but they're never a legitimate cost or
    FTE figure, and int(round(inf)) raises downstream -- must resolve to
    unparsed without calling the LLM fallback (this isn't the kind of
    ambiguous-format residual that fallback exists for)."""
    calls = _make_llm_mock(monkeypatch)

    result = cost_parsing.parse_cost_cell(raw_text, field_name="annual_fte_cost", application_id="APP-1")

    assert result.status == "unparsed"
    assert result.value is None
    assert calls == []


def test_non_finite_numeric_raw_value_never_crashes(monkeypatch):
    calls = _make_llm_mock(monkeypatch)

    result = cost_parsing.parse_cost_cell(float("inf"), field_name="annual_fte_cost", application_id="APP-1")

    assert result.status == "unparsed"
    assert result.value is None
    assert calls == []


def test_llm_fallback_non_finite_value_rejected(monkeypatch):
    _make_llm_mock(monkeypatch, return_value=_tool_call_response(True, float("inf")))

    result = cost_parsing.parse_cost_cell("45.000,00", field_name="annual_fte_cost", application_id="APP-1")

    assert result.status == "unparsed"
    assert result.value is None


@pytest.mark.parametrize(
    "raw_value,expected",
    [(45000, 45000.0), (45000.5, 45000.5), ("45000", 45000.0), ("45000.75", 45000.75)],
)
def test_clean_numeric_passthrough(monkeypatch, raw_value, expected):
    calls = _make_llm_mock(monkeypatch)

    result = cost_parsing.parse_cost_cell(raw_value, field_name="annual_fte_cost", application_id="APP-1")

    assert result.status == "parsed"
    assert result.value == expected
    assert calls == []


@pytest.mark.parametrize(
    "raw_text,expected",
    [
        ("$45,000.00 USD", 45000.0),
        ("45k INR", 45000.0),
        ("₹45,000", 45000.0),
        ("2m", 2_000_000.0),
    ],
)
def test_deterministic_normalization_never_reaches_llm(monkeypatch, raw_text, expected):
    calls = _make_llm_mock(monkeypatch)

    result = cost_parsing.parse_cost_cell(raw_text, field_name="annual_fte_cost", application_id="APP-1")

    assert result.status == "parsed"
    assert result.value == expected
    assert calls == []


def test_ambiguous_residual_resolved_via_llm_fallback(monkeypatch):
    calls = _make_llm_mock(monkeypatch, return_value=_tool_call_response(True, 45000.0))

    result = cost_parsing.parse_cost_cell("45.000,00", field_name="annual_fte_cost", application_id="APP-1")

    assert result.status == "parsed"
    assert result.value == 45000.0
    assert len(calls) == 1
    sensitivity, request = calls[0]
    assert sensitivity is DataSensitivity.REAL
    assert request.data == "45.000,00"


def test_llm_fallback_reports_not_numeric(monkeypatch):
    calls = _make_llm_mock(monkeypatch, return_value=_tool_call_response(False, None))

    result = cost_parsing.parse_cost_cell("###???", field_name="annual_fte_cost", application_id="APP-1")

    assert result.status == "unparsed"
    assert result.value is None
    assert len(calls) == 1


def test_llm_fallback_provider_error_degrades_to_unparsed(monkeypatch):
    _make_llm_mock(monkeypatch, side_effect=LLMProviderError("groq unavailable"))

    result = cost_parsing.parse_cost_cell("45.000,00", field_name="annual_fte_cost", application_id="APP-1")

    assert result.status == "unparsed"
    assert result.value is None


def test_llm_fallback_unexpected_sdk_error_degrades_to_unparsed(monkeypatch):
    # A raw SDK exception (e.g. a Groq auth/network error), not one of this
    # module's own LLMProviderError subclasses -- must still degrade
    # gracefully, since the whole point of this branch is "never crash the
    # batch" for any provider failure, not only the ones this module defines.
    _make_llm_mock(monkeypatch, side_effect=RuntimeError("connection reset"))

    result = cost_parsing.parse_cost_cell("45.000,00", field_name="annual_fte_cost", application_id="APP-1")

    assert result.status == "unparsed"
    assert result.value is None


def test_missing_and_blank_cells_never_reach_llm(monkeypatch):
    calls = _make_llm_mock(monkeypatch)

    for raw in (None, float("nan"), "", "   "):
        result = cost_parsing.parse_cost_cell(raw, field_name="annual_fte_cost", application_id="APP-1")
        assert result.status == "unparsed"
        assert result.value is None

    assert calls == []


def test_fte_count_non_numeric_does_not_crash(monkeypatch):
    calls = _make_llm_mock(monkeypatch)

    result = cost_parsing.parse_fte_count("cannot disclose", application_id="APP-1")

    assert result.status == "withheld"
    assert result.value is None
    assert calls == []


def test_build_numeric_field_notes_only_includes_non_parsed():
    parsed = {
        "annual_fte_cost": cost_parsing.ParsedCost(value=45000.0, status="parsed", raw_text=None),
        "other_costs": cost_parsing.ParsedCost(value=None, status="withheld", raw_text="cannot disclose"),
    }

    notes = cost_parsing.build_numeric_field_notes(parsed)

    assert notes is not None
    decoded = json.loads(notes)
    assert "annual_fte_cost" not in decoded
    assert decoded["other_costs"] == {"status": "withheld", "raw_text": "cannot disclose"}


def test_build_numeric_field_notes_none_when_all_parsed():
    parsed = {"annual_fte_cost": cost_parsing.ParsedCost(value=45000.0, status="parsed", raw_text=None)}
    assert cost_parsing.build_numeric_field_notes(parsed) is None
