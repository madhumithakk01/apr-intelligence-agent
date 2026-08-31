"""Top-level orchestration StateGraph -- CLAUDE.md sections 5, 10, 13.

The pipeline's control flow, and only its control flow: stage order,
four ``Send`` fan-outs, the five ``interrupt()`` gates, and a
checkpointer that makes a suspended run resumable. What each stage node
does is out of scope for this file (app/orchestration/nodes.py); most
are still pass-through stubs, and ingest/classify_disclosure have been
real since feat/disclosure-classifier (branch 6).

Building the topology before any stage exists is deliberate. This is a
portfolio-scale batch pipeline whose stages are individually cheap to
test and collectively expensive to re-wire, and a 100-row run is rate-
limited enough (CLAUDE.md section 11) that discovering a topology defect
during a real run costs an afternoon. Everything downstream plugs into
this graph, so it gets its own branch and its own tests.

Fan-out points -- the three places the portfolio stops being processed
one stage at a time:
  * per application row, for disclosure classification (section 6)
  * per application row, for qualitative scoring (section 7)
  * per candidate cluster, for redundancy adjudication (section 9)
  * per redundancy-surviving segment, for market intelligence (section 8)

Each router returns its join node when there is nothing to fan out, so
an empty portfolio still traverses every gate rather than skipping to
the end -- the empty case takes the same path as the populated one.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from app.orchestration import gates, nodes
from app.orchestration.state import GraphState

DEFAULT_RECURSION_LIMIT = 50
"""Not a governance parameter -- it caps LangGraph superstep depth, and
this graph is acyclic, so it is a runaway-execution rail rather than a
tunable. The one genuine loop in the system is the Market Intelligence
subgraph, and its iteration cap lives in governance_params (section 12)."""


def _run_context(state: GraphState) -> Dict[str, Any]:
    return {
        "run_id": state.get("run_id", ""),
        "data_sensitivity": state.get("data_sensitivity", "real"),
    }


def _fan_out_disclosure(state: GraphState) -> Union[str, List[Send]]:
    applications = state.get("applications") or []
    if not applications:
        return "calibrate_rubrics"
    context = _run_context(state)
    return [
        Send("classify_disclosure", {**context, "application": application})
        for application in applications
    ]


def _fan_out_qualitative(state: GraphState) -> Union[str, List[Send]]:
    """One branch per row that has a disclosure result -- score_row must
    only ever see the disclosure-gated application dict (CLAUDE.md
    section 2: a withheld field is never scored), never the raw one. A
    row whose disclosure branch failed has no gated_application to give
    it and is simply excluded, the same way apply_scoring_kernel's own
    'no disclosure result' handling degrades: skipped, never a fallback
    to the raw ungated value."""
    disclosure = state.get("disclosure") or {}
    context = _run_context(state)
    rubrics = state.get("rubrics") or {}

    sends: List[Send] = []
    for application in state.get("applications") or []:
        application_id = application.get("application_id")
        entry = disclosure.get(application_id) if application_id else None
        if not entry or "gated_application" not in entry:
            continue
        sends.append(
            Send("score_row", {**context, "application": entry["gated_application"], "rubrics": rubrics})
        )

    return sends if sends else "gate_qualitative_disagreement"


def _fan_out_adjudication(state: GraphState) -> Union[str, List[Send]]:
    clusters = state.get("clusters") or []
    if not clusters:
        return "gate_redundancy_verdict"
    context = _run_context(state)
    profiles = state.get("profiles") or {}
    return [
        Send(
            "adjudicate_cluster",
            {
                **context,
                "cluster": cluster,
                # Only the profiles of this cluster's own members: a
                # branch never receives the rest of the portfolio.
                "profiles": {
                    app_id: profiles[app_id]
                    for app_id in (cluster.get("application_ids") or [])
                    if app_id in profiles
                },
            },
        )
        for cluster in clusters
    ]


def _fan_out_market(state: GraphState) -> Union[str, List[Send]]:
    segments = state.get("segments") or []
    if not segments:
        return "extract_and_ground_products"
    context = _run_context(state)
    return [Send("research_segment", {**context, "segment": segment}) for segment in segments]


def build_graph(checkpointer: Any = None) -> Any:
    """Compile the pipeline. A checkpointer is required for any run that
    can hit a gate -- ``interrupt()`` has nowhere to suspend to without
    one -- so callers pass app.orchestration.checkpointer's SQLite saver
    (or the in-memory one in tests)."""
    graph: StateGraph = StateGraph(GraphState)

    graph.add_node("ingest", nodes.ingest)
    graph.add_node("classify_disclosure", nodes.classify_disclosure)
    graph.add_node("calibrate_rubrics", nodes.calibrate_rubrics)
    graph.add_node("gate_rubric_signoff", gates.gate_rubric_signoff)
    graph.add_node("score_row", nodes.score_row)
    graph.add_node("gate_qualitative_disagreement", gates.gate_qualitative_disagreement)
    graph.add_node("apply_scoring_kernel", nodes.apply_scoring_kernel)
    graph.add_node("block_capabilities", nodes.block_capabilities)
    graph.add_node("build_profiles", nodes.build_profiles)
    graph.add_node("adjudicate_cluster", nodes.adjudicate_cluster)
    graph.add_node("gate_redundancy_verdict", gates.gate_redundancy_verdict)
    graph.add_node("apply_recommendation_policy", nodes.apply_recommendation_policy)
    graph.add_node("detect_cost_outliers", nodes.detect_cost_outliers)
    graph.add_node("explain_cost_outliers", nodes.explain_cost_outliers)
    graph.add_node("gate_cost_outlier", gates.gate_cost_outlier)
    graph.add_node("build_market_segments", nodes.build_market_segments)
    graph.add_node("research_segment", nodes.research_segment)
    graph.add_node("extract_and_ground_products", nodes.extract_and_ground_products)
    graph.add_node("generate_narratives", nodes.generate_narratives)
    graph.add_node("gate_narrative_grounding", gates.gate_narrative_grounding)
    graph.add_node("render_report", nodes.render_report)

    graph.add_edge(START, "ingest")

    graph.add_conditional_edges(
        "ingest", _fan_out_disclosure, ["classify_disclosure", "calibrate_rubrics"]
    )
    graph.add_edge("classify_disclosure", "calibrate_rubrics")

    graph.add_edge("calibrate_rubrics", "gate_rubric_signoff")
    graph.add_conditional_edges(
        "gate_rubric_signoff", _fan_out_qualitative, ["score_row", "gate_qualitative_disagreement"]
    )
    graph.add_edge("score_row", "gate_qualitative_disagreement")

    graph.add_edge("gate_qualitative_disagreement", "apply_scoring_kernel")
    graph.add_edge("apply_scoring_kernel", "block_capabilities")
    graph.add_edge("block_capabilities", "build_profiles")

    graph.add_conditional_edges(
        "build_profiles", _fan_out_adjudication, ["adjudicate_cluster", "gate_redundancy_verdict"]
    )
    graph.add_edge("adjudicate_cluster", "gate_redundancy_verdict")

    graph.add_edge("gate_redundancy_verdict", "apply_recommendation_policy")
    graph.add_edge("apply_recommendation_policy", "detect_cost_outliers")
    graph.add_edge("detect_cost_outliers", "explain_cost_outliers")
    graph.add_edge("explain_cost_outliers", "gate_cost_outlier")
    graph.add_edge("gate_cost_outlier", "build_market_segments")

    graph.add_conditional_edges(
        "build_market_segments", _fan_out_market, ["research_segment", "extract_and_ground_products"]
    )
    graph.add_edge("research_segment", "extract_and_ground_products")

    graph.add_edge("extract_and_ground_products", "generate_narratives")
    graph.add_edge("generate_narratives", "gate_narrative_grounding")
    graph.add_edge("gate_narrative_grounding", "render_report")
    graph.add_edge("render_report", END)

    return graph.compile(checkpointer=checkpointer)


def run_config(
    thread_id: str, recursion_limit: int = DEFAULT_RECURSION_LIMIT
) -> Dict[str, Any]:
    """One APR run == one checkpointer thread. The thread id is what a
    later poll/resume call (branch 16's async batch endpoint, and every
    gate resume) uses to find the suspended run."""
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": recursion_limit}


def initial_state(
    run_id: str,
    applications: Optional[Sequence[Dict[str, Any]]] = None,
    data_sensitivity: str = "real",
) -> GraphState:
    """Default run input. ``data_sensitivity`` defaults to "real" so an
    omitted flag fails closed -- the strict provider routing (Groq only,
    never Gemini) applies unless a caller deliberately declares synthetic
    fixtures (CLAUDE.md section 11)."""
    return GraphState(
        run_id=run_id,
        data_sensitivity=data_sensitivity,
        applications=list(applications or []),
    )
