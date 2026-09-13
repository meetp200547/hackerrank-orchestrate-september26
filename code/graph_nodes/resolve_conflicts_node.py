"""
resolve_conflicts_node — applies 4-level precedence conflict resolution across user events.
"""

from __future__ import annotations
from graph_state import RequestState
from conflict_resolver import resolve_event_conflicts
from data_loader import get_data_loader


def resolve_conflicts_node(state: RequestState) -> RequestState:
    loader = get_data_loader()
    profile = state.get("user_profile", {})
    home_curr = profile.get("home_currency", "INR")
    events = state.get("raw_events", [])
    messages = state.get("raw_messages", [])

    resolved = resolve_event_conflicts(
        events=events,
        messages=messages,
        home_currency=home_curr,
        converter_func=loader.convert_currency
    )

    return {
        **state,
        "resolved_events": resolved
    }
