"""
process_events_node — wraps event_processor.py for recurring stream identification and 90-day cash flow projection.
"""

from __future__ import annotations
from graph_state import RequestState
from event_processor import process_user_events


def process_events_node(state: RequestState) -> RequestState:
    events = state.get("resolved_events", [])
    profile = state.get("user_profile", {})
    req = state["request"]
    req_date = req["request_date"]

    processed = process_user_events(
        events=events,
        user_profile=profile,
        request_date_str=req_date
    )

    return {
        **state,
        "recurring_events": processed.get("recurring_expenses", []),
        "flexible_events": processed.get("flexible_events", [])
    }
