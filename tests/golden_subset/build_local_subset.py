"""Build the LOCAL golden subset from the client workbook.

    python tests/golden_subset/build_local_subset.py [--source Dataset.xlsx] [--sheet Sheet2]

Output goes to tests/golden_subset/local/, which is gitignored and stays
that way. Nothing derived from the client file is committed or pushed --
not the rows, not de-identified rows, not the snapshots taken from them.
CI never sees this fixture; it exists so that whoever holds the workbook
can run the same regression suite against real values, and so that the
CLAUDE.md section 12 weights and thresholds can be validated against
data that was not invented by the person validating them.

The de-identification below is therefore not what makes the output safe
to publish -- nothing makes it safe to publish. It is defense in depth
for a file sitting on a working machine.

  VERBATIM   Every field the scoring kernel, the redundancy axes, and
             cost outlier detection read.
  PSEUDONYM  Application ID, Application Name, Technology Stack. The
             stack is tokenized per component and mapped consistently
             across rows, so "these two apps share a component" survives
             without naming vendors.
  REDACT     Owner, Owner Email, Application Description -> null.

Source sheet: the workbook holds two sheets whose Application IDs
collide while describing entirely different applications (Sheet1's 25
rows and Sheet2's 100 rows share APP-001..APP-025 and agree on almost no
other column). That collision is a real ingestion finding, surfaced by
branch 2 and never silently resolved -- so this builds from one declared
sheet and refuses to merge the other. Reconciling them is a Phase 2
discovery item for the client, not a decision a fixture builder gets to
make.

Row selection: deterministic greedy stratification over the qualitative
value space, then the cost extremes, then a capability-sharing pair for
the redundancy stages. Tie-broken by Application ID, so re-sorting the
workbook does not move the subset.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from tests.golden_subset import harness  # noqa: E402

DEFAULT_SOURCE = REPO_ROOT / "Dataset.xlsx"
DEFAULT_SHEET = "Sheet2"

VERBATIM_FIELDS = [
    "Department",
    "Application Status",
    "Business Criticality",
    "Business Fitness",
    "Strategic Relevance",
    "Usage & Adoption",
    "Functional redundancy",
    "Application Security Level",
    "Maintainability",
    "Application Stability",
    "Skill availability",
    "Availability",
    "Reliability",
    "Scalability",
    "Annual FTE Cost",
    "Annual License Cost",
    "FTE Count",
    "Annual Infrastructure Cost",
    "Other Costs",
    "Business Capability L1",
    "Business Capability L2",
    "Business Capability L3",
]

STRATIFICATION_FIELDS = [
    # Kernel-scored axes first: their value space is what the section 12
    # weights and thresholds get validated against.
    "Business Criticality",
    "Strategic Relevance",
    "Business Fitness",
    "Usage & Adoption",
    "Application Stability",
    "Maintainability",
    "Availability",
    "Reliability",
    "Scalability",
    "Skill availability",
    "Functional redundancy",
    "Application Security Level",
    # Then fields the redundancy and cost stages read.
    "Application Status",
    "Business Capability L1",
]

COST_COLUMNS = [
    "Annual FTE Cost",
    "Annual License Cost",
    "Annual Infrastructure Cost",
    "Other Costs",
]

NOTE = (
    "De-identified rows derived from the client workbook. Local only -- never commit, never push "
    "(tests/golden_subset/local/ is gitignored)."
)

_STACK_SPLIT_RE_PARTS = [",", "/", ";", " and "]


def _cell(value: Any) -> Any:
    """Preserve the cell as written. Non-numeric cost text ("cannot
    disclose") is carried through as a string on purpose -- it is a
    business decision, not a defect (CLAUDE.md section 2)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (int, float)):
        return value
    text = str(value).strip()
    return text or None


class _StackPseudonymizer:
    def __init__(self) -> None:
        self._mapping: Dict[str, str] = {}

    def __call__(self, stack: Optional[str]) -> Optional[str]:
        if not stack:
            return None
        tokens = [stack]
        for separator in _STACK_SPLIT_RE_PARTS:
            tokens = [part for token in tokens for part in token.split(separator)]
        pseudonyms = []
        for token in (token.strip() for token in tokens):
            if not token:
                continue
            key = token.casefold()
            if key not in self._mapping:
                self._mapping[key] = f"TECH-{len(self._mapping) + 1:02d}"
            pseudonyms.append(self._mapping[key])
        return ", ".join(pseudonyms) or None


def _value_pairs(row: pd.Series) -> set:
    return {(field, str(row.get(field)).strip()) for field in STRATIFICATION_FIELDS}


def _total_known_cost(row: pd.Series) -> float:
    total = 0.0
    for column in COST_COLUMNS:
        value = row.get(column)
        if isinstance(value, (int, float)) and not pd.isna(value):
            total += float(value)
    return total


def select_rows(frame: pd.DataFrame, target: int = harness.MAX_ROWS) -> pd.DataFrame:
    frame = frame.sort_values("Application ID", kind="stable").reset_index(drop=True)
    remaining = {index: row for index, row in frame.iterrows()}
    uncovered = set().union(*(_value_pairs(row) for row in remaining.values()))

    chosen: List[int] = []
    while remaining and uncovered and len(chosen) < target:
        index = max(remaining, key=lambda i: (len(_value_pairs(remaining[i]) & uncovered), -i))
        if not (_value_pairs(remaining[index]) & uncovered):
            break
        uncovered -= _value_pairs(remaining[index])
        chosen.append(index)
        del remaining[index]

    def _add(index: Optional[int]) -> None:
        if index is not None and index in remaining and len(chosen) < target:
            chosen.append(index)
            del remaining[index]

    if remaining:
        by_cost = sorted(remaining, key=lambda i: (_total_known_cost(remaining[i]), i))
        _add(by_cost[0])
        _add(by_cost[-1] if by_cost[-1] in remaining else None)

    chosen_l3 = {str(frame.loc[i].get("Business Capability L3")).strip() for i in chosen}
    for index in sorted(remaining):
        if str(remaining[index].get("Business Capability L3")).strip() in chosen_l3:
            _add(index)
            break

    for index in sorted(remaining):
        if len(chosen) >= harness.MIN_ROWS:
            break
        _add(index)

    return frame.loc[sorted(chosen)].reset_index(drop=True)


def build_rows(source: Path, sheet: str = DEFAULT_SHEET) -> List[Dict[str, Any]]:
    workbook = pd.read_excel(source, sheet_name=None)
    if sheet not in workbook:
        raise SystemExit(f"sheet {sheet!r} not in {source} (found: {', '.join(workbook)})")
    frame = workbook[sheet].copy()
    frame.columns = frame.columns.str.strip()
    frame = select_rows(frame)

    pseudonymize_stack = _StackPseudonymizer()
    rows: List[Dict[str, Any]] = []
    for position, (_, source_row) in enumerate(frame.iterrows(), start=1):
        golden_id = f"GOLD-{position:02d}"
        row: Dict[str, Any] = {
            "Application ID": golden_id,
            "Application Name": f"Application {golden_id}",
            "Technology Stack": pseudonymize_stack(_cell(source_row.get("Technology Stack"))),
        }
        for field in harness.IDENTITY_FIELDS:
            row[field] = None
        for field in VERBATIM_FIELDS:
            row[field] = _cell(source_row.get(field))
        row[harness.MARKET_PRODUCT_COUNT_FIELD] = 0
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the gitignored local golden subset.")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="Path to the client Excel file")
    parser.add_argument(
        "--sheet", default=DEFAULT_SHEET, help="Sheet to build from. Sheets are never merged."
    )
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        parser.error(f"source workbook not found: {source}")

    rows = build_rows(source, args.sheet)
    existing = None
    if harness.LOCAL_LABELS_PATH.exists():
        _, existing = harness.load_local_fixture()
    labels = harness.build_labels(rows, existing, note=NOTE)

    harness.write_json(harness.LOCAL_ROWS_PATH, rows)
    harness.write_json(harness.LOCAL_LABELS_PATH, labels)

    print(f"wrote {len(rows)} de-identified rows to {harness.LOCAL_ROWS_PATH.relative_to(REPO_ROOT)}")
    print("this directory is gitignored -- do not commit or push its contents")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
