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

from typing import Optional

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

# --- Not yet consumed by any existing branch. Transcribed now so every
# governance number has exactly one home from the start (CLAUDE.md
# section 12: "No second copy anywhere else in the codebase"). Each
# MUST be read from here, never re-hardcoded, once its owning branch
# lands. ---

QUALITATIVE_ENSEMBLE_SIZE = 3
"""Owned by: feat/qualitative-scoring (branch 8)."""

QUALITATIVE_ENSEMBLE_DISAGREEMENT = {
    "auto_accept_max_range": 1,
    "mandatory_review_min_range": 2,
}
"""Owned by: feat/qualitative-scoring (branch 8)."""

REDUNDANCY_ENSEMBLE_SIZE = 3
"""Owned by: feat/redundancy-adjudicator (branch 10)."""

MARKET_AGENT_ITERATION_CAP = 4
"""Owned by: feat/market-intelligence-agent (branch 12). Safety rail,
not the primary stop logic -- see CLAUDE.md section 8."""

MIN_PEER_CLUSTER_SIZE_FOR_COST_OUTLIER = 5
"""Owned by: feat/cost-outlier-detection (branch 11). Below this
cluster size, refuse to flag a cost outlier rather than manufacture a
signal from noise."""

QUALITATIVE_ESCALATION_CONFIDENCE_THRESHOLD: Optional[float] = None
"""Not yet set. CLAUDE.md section 12: tune against the golden subset
before hardcoding -- do not guess a value without recording the
rationale. Owned by: feat/qualitative-scoring (branch 8); must stay
None until that branch records an empirically-justified value here."""
