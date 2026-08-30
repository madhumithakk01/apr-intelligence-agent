"""Synthetic fixtures for orchestration tests.

Invented rows, not client data, and not a sample of it -- CLAUDE.md
section 2 permits real client data nowhere near a test fixture, and
section 11 permits it nowhere near Gemini. Values here are deliberately
implausible as a real portfolio (three apps, obviously fake vendors) so
nobody can mistake this file for the delivered dataset.

The fields present are only those the orchestration graph itself routes
on: an application id to key fan-out branches by, plus the capability
tags the blocking stage will later key on. Full-row fixtures belong to
the branches that actually read the other columns.
"""

from __future__ import annotations

from typing import Any, Dict, List

SYNTHETIC_APPLICATIONS: List[Dict[str, Any]] = [
    {
        "application_id": "SYN-001",
        "application_name": "Synthetic Claims Intake",
        "business_capability_l1": "Synthetic Operations",
        "business_capability_l2": "Synthetic Claims",
        "business_capability_l3": "Synthetic Claims Intake",
    },
    {
        "application_id": "SYN-002",
        "application_name": "Synthetic Claims Intake Lite",
        "business_capability_l1": "Synthetic Operations",
        "business_capability_l2": "Synthetic Claims",
        "business_capability_l3": "Synthetic Claims Intake",
    },
    {
        "application_id": "SYN-003",
        "application_name": "Synthetic Ledger",
        "business_capability_l1": "Synthetic Finance",
        "business_capability_l2": "Synthetic Ledger",
        "business_capability_l3": "Synthetic General Ledger",
    },
]

SYNTHETIC_CLUSTERS: List[Dict[str, Any]] = [
    {"cluster_id": "CL-CLAIMS", "application_ids": ["SYN-001", "SYN-002"]},
    {"cluster_id": "CL-LEDGER", "application_ids": ["SYN-003"]},
]

SYNTHETIC_SEGMENTS: List[Dict[str, Any]] = [
    # Two segments from one cluster: the Scale-Tiered Overlap case from
    # CLAUDE.md section 8, where each tier is a separately-framed
    # research target rather than one query for the pair.
    {"segment_id": "SEG-CLAIMS-HEAVY", "cluster_id": "CL-CLAIMS", "tier": "enterprise"},
    {"segment_id": "SEG-CLAIMS-LIGHT", "cluster_id": "CL-CLAIMS", "tier": "lightweight"},
    {"segment_id": "SEG-LEDGER", "cluster_id": "CL-LEDGER", "tier": "enterprise"},
]

SYNTHETIC_PROFILES: Dict[str, Dict[str, Any]] = {
    "SYN-001": {"functional": "synthetic", "scale": "enterprise"},
    "SYN-002": {"functional": "synthetic", "scale": "lightweight"},
    "SYN-003": {"functional": "synthetic", "scale": "enterprise"},
}
