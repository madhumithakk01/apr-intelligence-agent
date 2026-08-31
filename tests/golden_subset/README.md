# Golden subset

The regression gate that runs on every PR (SPEC.md sections 13 and 15).

It has two halves, and the split is a confidentiality requirement rather
than a convenience: **nothing derived from the client workbook is ever
committed or pushed — de-identified or not.**

| | Committed fixture | Local fixture |
|---|---|---|
| Path | `rows.json`, `labels.json` | `local/` (gitignored) |
| Rows | 20, entirely invented | 20, de-identified from the client workbook |
| Built by | hand; snapshots by `refresh_snapshots.py` | `build_local_subset.py`, on demand |
| Runs in CI | yes | no — CI never has it |
| Can validate §12 parameters | **no** | yes |

## The committed fixture

20 invented rows. No value, cost, capability name, department, or
vendor is taken or derived from `Dataset.xlsx`. They exist to exercise
the scoring paths, and they are built to cover:

- all four TIME bands (Invest / Migrate / Tolerate / Eliminate) plus
  `Insufficient Data`
- the skill-availability floor — a row whose raw decision is `Invest`
  and whose final decision is forced to `Migrate` (SPEC.md §4.5)
- both sides of the COTS replace threshold (§12), via a fixture-only
  `Market Product Count` column standing in for branch 12's retrieval
- withheld cost cells (`"cannot disclose"`) and withheld qualitative
  cells (`"cannot say"`), so the never-default rule stays covered
- free-text qualitative values that match none of the five labels the
  deterministic kernel knows — SPEC.md §4.1, the bug that used to
  silently default to 3
- a capability-sharing trio on one L3, so the redundancy stages
  (branches 9–10) have something to cluster

Because the bands are populated, a change to the TIME weights or the
decision thresholds moves rows and fails the snapshot tests — the gate
is sensitive to the parameters, not just to structure.

**What it cannot do:** validate those parameters. Its rows were invented
by the same person who would be checking whether 0.45/0.35/0.20 is
right, so agreement proves nothing. Every run says so in a warning.

## The local fixture

`build_local_subset.py` reads the client workbook and writes
`local/rows.json` + `local/labels.json`, which are gitignored. When
present, the whole suite runs over them too — same invariants, real
values. This is where SPEC.md §12's empirical validation of the
weights and thresholds has to happen, and its verdict travels back as an
analyst label, never as data.

Its de-identification (verbatim scored fields, pseudonymized
IDs/names/stack, redacted owner/email/description) is defense in depth
for a file on a working machine — **not** a licence to publish it.

Two things that script will not do:

- **Merge the workbook's sheets.** `Sheet1` (25 rows) and `Sheet2` (100
  rows) share `APP-001`..`APP-025` while describing entirely different
  applications. That collision is a real ingestion finding — surfaced by
  branch 2, never silently resolved — so the subset is built from one
  declared sheet (`Sheet2`, the ~100-row portfolio). Reconciling the two
  is a Phase 2 discovery item for the client.
- **Take the first N rows.** Selection is deterministic greedy
  stratification over the qualitative value space, then the cost
  extremes, then a capability-sharing pair; ties broken by Application
  ID, so re-sorting the workbook does not move the subset.

## Analyst labels

`analyst_labels` are empty in both fixtures. To label a row, set any of
`expected_time_decision`, `expected_redundancy_typology`,
`expected_cots_action`, plus `reviewer` and `notes`. Filled labels are
compared against the pipeline automatically; rows the kernel cannot yet
score are reported as pending rather than compared, so labelling can
start now and begins gating the moment branch 8 lands.

Label the **local** fixture for §12 validation. Labelling the invented
one only records what the fixture was designed to produce.

## Commands

```
pytest tests/golden_subset -v                        # the gate
python tests/golden_subset/refresh_snapshots.py      # after an intended scoring change
python tests/golden_subset/build_local_subset.py     # local only, needs the workbook
```

A snapshot diff is the record of what a scoring change did to 20 rows.
Review it in the PR — waving it through defeats the gate.
