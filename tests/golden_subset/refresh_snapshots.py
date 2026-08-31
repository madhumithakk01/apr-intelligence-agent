"""Regenerate the committed fixture's kernel snapshots.

    python tests/golden_subset/refresh_snapshots.py

Reads tests/golden_subset/rows.json (20 invented rows -- nothing from
the client workbook) and rewrites labels.json's kernel_snapshot blocks
to whatever app/scoring/kernel.py produces today. Analyst labels already
filled in are carried across by golden id.

Run this only when a scoring change is intentional, and review the diff:
the snapshot diff *is* the record of what the change did to 20 rows, and
waving it through defeats the gate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tests.golden_subset import harness  # noqa: E402

NOTE = (
    "Invented rows. No value, cost, or capability name is taken or derived from the client "
    "workbook -- see README.md. Exercises the scoring paths; does not validate the SPEC.md "
    "section 12 parameters, which needs real data and stays local."
)


def main() -> int:
    rows = harness.load_rows()
    existing = harness.load_labels() if harness.LABELS_PATH.exists() else None
    labels = harness.build_labels(rows, existing, note=NOTE)
    harness.write_json(harness.LABELS_PATH, labels)

    decisions: dict[str, int] = {}
    for entry in labels["rows"].values():
        decision = entry["kernel_snapshot"]["tim_e_decision"]
        decisions[decision] = decisions.get(decision, 0) + 1
    print(f"refreshed {len(rows)} snapshots in {harness.LABELS_PATH.relative_to(REPO_ROOT)}")
    print(f"decisions: {json.dumps(decisions, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
