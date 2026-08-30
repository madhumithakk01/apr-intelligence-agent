"""Governance parameters -- CLAUDE.md section 12.

Single source of truth for every tunable numeric parameter in the
scoring/redundancy/market/cost pipelines. No consumer anywhere in this
codebase should hardcode one of these values locally -- import it from
here. Values already decided are transcribed verbatim from CLAUDE.md;
this module does not invent, tune, or validate any of them. Parameters
not yet consumed by any branch are included so future branches have
exactly one place to look before hardcoding locally -- see each
section's "Owned by" note.
"""

# --- Consumed by scoring/kernel.py ---

TIME_WEIGHTS = {"value": 0.45, "health": 0.35, "consolidation": 0.20}
"""Gartner TIME framework (Tolerate/Invest/Migrate/Eliminate) weighting
of the three composite TIM-E axes. Existing values carried over from
the pre-consolidation engines, cited to Gartner TIME, PENDING empirical
validation against the golden subset (test/golden-subset-harness,
branch 5). Do not treat as final until that validation runs."""

DECISION_THRESHOLDS = {"invest": 80, "migrate": 60, "tolerate": 40}
"""TIM-E score cut points. Same pending-validation status as
TIME_WEIGHTS. Non-compensatory floors (e.g. the skill-availability
floor in scoring.kernel) can still override the label these thresholds
produce."""

COTS_REPLACE_THRESHOLD = 65
"""Canonical COTS-fit score (0-100) at/above which the recommendation
is "Replace with COTS". Single source of truth -- resolves CLAUDE.md
section 4 bug 3 (the old 65-vs-70 inconsistency between the two former
scoring engines). Same pending-validation status as TIME_WEIGHTS."""

COTS_FIT_WEIGHTS = {
    "functional_redundancy": 0.5,
    "maintainability": 0.3,
    "application_stability": 0.2,
}
"""Not itemized in CLAUDE.md section 12's table; carried over unchanged
from the pre-consolidation engines and centralized here per this
module's general mandate (section 1: every numeric parameter must be
documented, not invented). No rationale beyond "matches the prior
engines" is on record yet -- pending the same validation as the other
weighted formulas above."""

MARKET_PRODUCT_BONUS_PER_PRODUCT = 1.5
MARKET_PRODUCT_BONUS_CAP = 10
"""COTS-fit score bonus: up to MARKET_PRODUCT_BONUS_CAP retrieved
market products, each worth MARKET_PRODUCT_BONUS_PER_PRODUCT points.
Same carried-over, undocumented-rationale status as COTS_FIT_WEIGHTS."""

# --- Consumed by redundancy/adjudicator.py (branch 10) ---

REDUNDANCY_ENSEMBLE_SIZE = 3
"""Fixed (CLAUDE.md section 12). Unlike qualitative scoring, redundancy
adjudication has no single-call-by-default / escalate-on-low-confidence
path -- every pairwise comparison that survives the deterministic
Indeterminate-withheld-data pre-check gets the full 3-sample ensemble
unconditionally (CLAUDE.md section 5's table entry for this stage)."""

REDUNDANCY_ENSEMBLE_TEMPERATURE = 0.7
"""Not itemized in CLAUDE.md section 12's table; same rationale as
QUALITATIVE_ENSEMBLE_TEMPERATURE (feat/qualitative-scoring, branch 8) --
3 samples at temperature 0.0 would trivially agree every time, making
the ensemble vote meaningless. Kept as its own constant rather than
reusing branch 8's, since the two ensembles are independently tunable
and there is no reason their sampling temperatures must move together.
Not yet validated against the golden subset."""

# --- Consumed by qualitative_scoring/scorer.py (branch 8) ---

QUALITATIVE_ENSEMBLE_SIZE = 3
"""Fixed (CLAUDE.md section 12). The default single call, when a field
is escalated, counts as the first of these 3 samples -- 2 more calls are
made to reach it, not 3 fresh ones."""

QUALITATIVE_ENSEMBLE_DISAGREEMENT = {
    "auto_accept_max_range": 1,
    "mandatory_review_min_range": 2,
}
"""Fixed (CLAUDE.md section 12). Range is measured in points on the
kernel's 1-5 scale across the ensemble's valid samples. These two
values are consecutive integers by design -- every possible range (0-4
points, since the scale spans 1-5) falls into exactly one bucket, with
no undefined gap between "auto-accept" and "mandatory review"."""

QUALITATIVE_ESCALATION_CONFIDENCE_THRESHOLD: float = 0.7
"""CLAUDE.md section 7: escalate to the 3-sample ensemble when the
model's own self-reported confidence (0.0-1.0) on the default single
call falls below this. 0.7 is a reasoned default, not yet empirically
tuned: informal consensus in LLM-confidence-calibration practice is that
self-reported confidence above roughly this level correlates with
actual correctness meaningfully better than below it, but that is a
general heuristic, not evidence from this system's own data. PENDING
validation against the golden subset once a labelled qualitative-scoring
column exists there (tests/golden_subset section 12 note) -- do not
treat as final, and do not lower it to reduce escalation volume without
re-deriving it from data."""

QUALITATIVE_ENSEMBLE_TEMPERATURE = 0.7
"""Not itemized in CLAUDE.md section 12's table; the ensemble's whole
point is 3 independent-ish samples to measure disagreement, which
temperature 0.0 (used for the deterministic default call and for every
other structured call in this codebase) cannot produce -- three
temperature-0.0 samples of the same prompt would trivially agree every
time, making the ensemble meaningless. 0.7 is a conventional
moderate-diversity value for a classification-style task; not yet
validated against the golden subset."""

# --- Not yet consumed by any existing branch. Transcribed now so every
# governance number has exactly one home from the start (CLAUDE.md
# section 12: "No second copy anywhere else in the codebase"). Each
# MUST be read from here, never re-hardcoded, once its owning branch
# lands. ---

MARKET_AGENT_ITERATION_CAP = 4
"""Owned by: feat/market-intelligence-agent (branch 12). Safety rail,
not the primary stop logic -- see CLAUDE.md section 8."""

MIN_PEER_CLUSTER_SIZE_FOR_COST_OUTLIER = 5
"""Owned by: feat/cost-outlier-detection (branch 11). Below this
cluster size, refuse to flag a cost outlier rather than manufacture a
signal from noise."""
