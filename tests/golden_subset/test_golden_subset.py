"""Golden-subset regression suite -- the required check on every PR.

Runs over the committed fixture (20 invented rows -- nothing derived
from the client workbook, see README.md) and, on a machine that has one,
over the gitignored local fixture built from the real file as well.

What it gates:

  1. Neither fixture carries an identity. A row with an owner, an email,
     or a source application id fails here rather than reaching GitHub.
  2. Scoring output does not change unless the snapshots were
     regenerated on purpose -- across all four TIME bands, the
     skill-availability floor, both sides of the COTS threshold, and
     withheld-cost rows.
  3. No axis is ever scored from a value the kernel cannot interpret
     (SPEC.md section 4 bug 1), and the classification field never
     re-enters technical health (bug 4).
  4. Analyst labels, once filled, agree with the pipeline. Until then
     the suite says so out loud every run, rather than reporting green
     as though SPEC.md section 12's weights had been validated -- and
     they cannot be validated against invented rows in any case.
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import pytest

from app.scoring import governance_params as gp
from app.scoring import kernel
from tests.golden_subset import harness

ROWS = harness.load_rows()
LABELS = harness.load_labels()
LOCAL_ROWS, LOCAL_LABELS = harness.load_local_fixture()

GOLDEN_ID_RE = re.compile(r"^GOLD-\d{2}$")
TECH_TOKEN_RE = re.compile(r"^TECH-\d{2}$")

QUALITATIVE_COLUMNS = [
    column
    for column, field in harness.KERNEL_FIELD_MAP.items()
    if field
    not in {"application_id", "application_name", "business_capability_l2", "business_capability_l3"}
]

ALL_FIXTURES = [("committed", ROWS)] + ([("local", LOCAL_ROWS)] if LOCAL_ROWS else [])


def _snapshots(rows):
    return [harness.kernel_snapshot(row) for row in rows]


COMMITTED_SNAPSHOTS = _snapshots(ROWS)


# --- fixture integrity (both fixtures) --------------------------------------


@pytest.mark.parametrize("name, rows", ALL_FIXTURES)
def test_fixture_size_matches_the_documented_range(name, rows):
    assert harness.MIN_ROWS <= len(rows) <= harness.MAX_ROWS


@pytest.mark.parametrize("name, rows", ALL_FIXTURES)
def test_golden_ids_are_unique_and_carry_no_client_identifier(name, rows):
    ids = [row["Application ID"] for row in rows]
    assert len(set(ids)) == len(ids)
    assert all(GOLDEN_ID_RE.match(golden_id) for golden_id in ids)


@pytest.mark.parametrize("name, rows", ALL_FIXTURES)
def test_identity_fields_are_absent_or_null(name, rows):
    for row in rows:
        for field in harness.IDENTITY_FIELDS:
            assert row.get(field) is None, f"{row['Application ID']}: {field} carries a value"


@pytest.mark.parametrize("name, rows", ALL_FIXTURES)
def test_no_email_address_or_source_id_survives_anywhere(name, rows):
    """Checks the serialized fixture, not a field list that could drift
    when someone adds a column."""
    serialized = json.dumps(rows)
    assert "@" not in serialized
    assert not re.search(r"APP-\d{3}", serialized)


@pytest.mark.parametrize("name, rows", ALL_FIXTURES)
def test_technology_stack_is_tokenized(name, rows):
    tokens = [
        token.strip()
        for row in rows
        for token in (row.get("Technology Stack") or "").split(",")
        if token.strip()
    ]
    assert tokens, f"{name} fixture lost the technology stack axis entirely"
    assert all(TECH_TOKEN_RE.match(token) for token in tokens)


@pytest.mark.parametrize("name, rows", ALL_FIXTURES)
def test_cost_cells_are_numeric_or_preserved_refusal_text(name, rows):
    for row in rows:
        for column in harness.COST_FIELD_MAP:
            value = row.get(column)
            assert value is None or isinstance(value, (int, float, str))


@pytest.mark.parametrize("name, rows", ALL_FIXTURES)
def test_fixture_contains_a_capability_sharing_pair(name, rows):
    """Without one, the redundancy stages (branches 9-10) have nothing
    to cluster and their own golden checks would be vacuous."""
    l3_values = [row.get("Business Capability L3") for row in rows]
    assert any(l3_values.count(value) > 1 for value in set(l3_values))


def test_local_fixture_directory_is_gitignored():
    """The local fixture is the only place client-derived rows exist.
    If this entry is ever removed, a `git add -A` publishes them."""
    gitignore = (Path(__file__).resolve().parents[2] / ".gitignore").read_text(encoding="utf-8")
    assert "tests/golden_subset/local/" in gitignore.splitlines()


# --- the committed fixture exercises the paths it claims to -----------------


def test_fixture_covers_every_time_band():
    decisions = {snapshot["tim_e_decision"] for snapshot in COMMITTED_SNAPSHOTS}
    assert {"Invest", "Migrate", "Tolerate", "Eliminate", "Insufficient Data"} <= decisions


def test_fixture_covers_the_skill_availability_floor():
    """SPEC.md section 4 bug 5: low skill availability plus fragile
    stability forces a minimum Migrate over a high raw score."""
    floored = [
        snapshot
        for snapshot in COMMITTED_SNAPSHOTS
        if snapshot["floor_applied"] == "skill_availability_floor"
    ]
    assert floored, "no row exercises the skill-availability floor"
    for snapshot in floored:
        assert snapshot["tim_e_raw_decision"] == "Invest"
        assert snapshot["tim_e_decision"] == "Migrate"


def test_fixture_covers_both_sides_of_the_cots_threshold():
    scored = [s for s in COMMITTED_SNAPSHOTS if s["cots_score"] is not None]
    assert any(s["cots_meets_threshold"] for s in scored)
    assert any(not s["cots_meets_threshold"] for s in scored)


def test_fixture_covers_withheld_and_uninterpretable_values():
    statuses = {
        snapshot[axis]["status"]
        for snapshot in COMMITTED_SNAPSHOTS
        for axis in ("value_score", "health_score", "consolidation_need")
    }
    assert {"complete", "partial", "insufficient_data"} <= statuses

    withheld_cost_rows = [
        row
        for row in ROWS
        if any(isinstance(row.get(column), str) for column in harness.COST_FIELD_MAP)
    ]
    assert withheld_cost_rows, "no row exercises a withheld cost cell"


# --- scoring regression -----------------------------------------------------


def test_labels_file_covers_exactly_the_fixture_rows():
    assert set(LABELS["rows"]) == {row["Application ID"] for row in ROWS}


@pytest.mark.parametrize("row", ROWS, ids=[row["Application ID"] for row in ROWS])
def test_kernel_output_matches_the_recorded_snapshot(row):
    expected = LABELS["rows"][row["Application ID"]]["kernel_snapshot"]
    actual = harness.kernel_snapshot(row)
    assert actual == expected, (
        f"{row['Application ID']}: scoring output changed. If the change is intended, regenerate "
        "with `python tests/golden_subset/refresh_snapshots.py` and review the diff in the PR."
    )


@pytest.mark.skipif(not LOCAL_ROWS, reason="no local fixture on this machine (CI, or workbook absent)")
def test_local_fixture_matches_its_recorded_snapshots():
    mismatches = [
        row["Application ID"]
        for row in LOCAL_ROWS
        if harness.kernel_snapshot(row) != LOCAL_LABELS["rows"][row["Application ID"]]["kernel_snapshot"]
    ]
    assert not mismatches, (
        "local fixture scoring changed for: "
        + ", ".join(mismatches)
        + " -- rebuild with `python tests/golden_subset/build_local_subset.py` if intended."
    )


@pytest.mark.parametrize(
    "row",
    ROWS + LOCAL_ROWS,
    ids=[row["Application ID"] for row in ROWS] + [f"local-{row['Application ID']}" for row in LOCAL_ROWS],
)
def test_no_axis_is_scored_from_a_value_the_kernel_cannot_interpret(row):
    """SPEC.md section 4 bug 1. Every value that is not one of the five
    labels the kernel knows must land in an unscored list -- never
    contribute a number, never a silent default of 3."""
    snapshot = harness.kernel_snapshot(row)
    unscored = set(
        snapshot["value_score"]["unscored_axes"]
        + snapshot["health_score"]["unscored_axes"]
        + snapshot["consolidation_need"]["unscored_axes"]
    )
    scored = set(
        snapshot["value_score"]["scored_axes"]
        + snapshot["health_score"]["scored_axes"]
        + snapshot["consolidation_need"]["scored_axes"]
    )

    for column in QUALITATIVE_COLUMNS:
        field = harness.KERNEL_FIELD_MAP[column]
        if field not in unscored | scored:
            continue  # security classification: routed to the gate, not an axis
        value = row.get(column)
        if kernel.score_qualitative_label(value) is None:
            assert field in unscored, f"{row['Application ID']}: {field} scored from {value!r}"
        else:
            assert field in scored


@pytest.mark.parametrize("name, rows", ALL_FIXTURES)
def test_security_classification_is_never_folded_into_technical_health(name, rows):
    """SPEC.md section 4 bug 4."""
    for row in rows:
        snapshot = harness.kernel_snapshot(row)
        health = snapshot["health_score"]
        assert "application_security_level" not in health["scored_axes"] + health["unscored_axes"]
        assert snapshot["security_classification"] == row.get("Application Security Level")


@pytest.mark.parametrize("name, rows", ALL_FIXTURES)
def test_cots_threshold_has_one_source_of_truth(name, rows):
    for row in rows:
        snapshot = harness.kernel_snapshot(row)
        if snapshot["cots_score"] is None:
            assert snapshot["cots_meets_threshold"] is False
        else:
            assert snapshot["cots_meets_threshold"] == (
                snapshot["cots_score"] >= gp.COTS_REPLACE_THRESHOLD
            )


def test_time_band_boundaries_follow_the_governance_thresholds():
    """Ties the snapshots to governance_params rather than to numbers
    pasted into this file -- a weight or threshold change moves rows
    across bands and fails the snapshot test above, with this one saying
    why."""
    for snapshot in COMMITTED_SNAPSHOTS:
        score = snapshot["tim_e_score"]
        if score is None:
            assert snapshot["tim_e_raw_decision"] is None
            continue
        expected = (
            "Invest"
            if score >= gp.DECISION_THRESHOLDS["invest"]
            else "Migrate"
            if score >= gp.DECISION_THRESHOLDS["migrate"]
            else "Tolerate"
            if score >= gp.DECISION_THRESHOLDS["tolerate"]
            else "Eliminate"
        )
        assert snapshot["tim_e_raw_decision"] == expected


# --- analyst labels ---------------------------------------------------------


def test_analyst_label_slots_are_present_for_every_row():
    for golden_id in LABELS["rows"]:
        assert set(harness.analyst_labels(LABELS, golden_id)) == set(harness.ANALYST_LABEL_FIELDS)


def test_filled_analyst_labels_agree_with_the_pipeline():
    """Runs over whichever fixtures are present. Rows the kernel
    declines to score are reported as pending rather than compared, so
    labelling can start before branch 8 lands and begins gating the
    moment it does."""
    mismatches = []
    pending = []
    for rows, labels in [(ROWS, LABELS), (LOCAL_ROWS, LOCAL_LABELS)]:
        for row in rows:
            golden_id = row["Application ID"]
            judgments = harness.filled_judgments(labels, golden_id)
            if not judgments:
                continue
            snapshot = harness.kernel_snapshot(row)
            if snapshot["tim_e_decision"] == "Insufficient Data":
                pending.append(golden_id)
                continue
            expected = judgments.get("expected_time_decision")
            if expected is not None and expected != snapshot["tim_e_decision"]:
                mismatches.append(
                    f"{golden_id}: analyst {expected!r} vs pipeline {snapshot['tim_e_decision']!r}"
                )

    if pending:
        warnings.warn(
            f"golden subset: {len(pending)} labelled rows are not scoreable yet "
            f"({', '.join(pending)}) -- comparison deferred until branch 8.",
            stacklevel=2,
        )
    assert not mismatches, "analyst labels disagree with the pipeline:\n" + "\n".join(mismatches)


def test_parameter_validation_status_is_reported_rather_than_assumed():
    """The committed fixture cannot validate the section 12 weights --
    its rows were invented by the same person who would be checking
    them. That validation needs the local fixture (or the real
    portfolio), and this suite refuses to look like it has done it."""
    coverage = harness.label_coverage(LABELS)
    assert coverage["total_rows"] == len(ROWS)

    if not LOCAL_ROWS:
        warnings.warn(
            "golden subset: running against invented rows only. This gates scoring regressions; it "
            "does NOT validate the SPEC.md section 12 weights and thresholds, which remain "
            "pending against real data (build the local fixture to do that).",
            stacklevel=2,
        )
    elif harness.label_coverage(LOCAL_LABELS)["rows_with_any_judgment"] == 0:
        warnings.warn(
            "golden subset: local fixture present but unlabelled -- section 12 validation still "
            "pending. Fill analyst_labels in tests/golden_subset/local/labels.json.",
            stacklevel=2,
        )
