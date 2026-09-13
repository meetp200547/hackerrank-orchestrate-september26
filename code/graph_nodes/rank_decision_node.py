"""
rank_decision_node — deterministic node wrapping DecisionEngine for candidate plan ranking.
CRITICAL INVARIANT: NO LLM IMPORTS.
"""

from __future__ import annotations
from graph_state import RequestState
from financial_engine import FinancialEngine
from decision_engine import DecisionEngine
from event_processor import process_user_events


def rank_decision_node(state: RequestState) -> RequestState:
    profile = state["user_profile"]
    req = state["request"]
    events = state.get("resolved_events", [])
    opts = state.get("payment_options", [])

    curr_bal = profile.get("current_available_balance", 0.0)
    min_bal = profile.get("minimum_balance_to_keep", 0.0)
    req_date = req["request_date"]

    processed = process_user_events(events, profile, req_date)
    inflows = processed["daily_inflows"]
    outflows = processed["daily_outflows"]
    flex_events = processed["flexible_events"]

    fin_engine = FinancialEngine(
        current_balance=curr_bal,
        minimum_balance=min_bal,
        request_date_str=req_date,
        daily_inflows=inflows,
        daily_outflows=outflows
    )

    dec_engine = DecisionEngine(
        request=req,
        user_profile=profile,
        payment_options=opts,
        financial_engine=fin_engine,
        flexible_events=flex_events
    )

    decision_dict = dec_engine.select_best_decision()

    return {
        **state,
        "decision": decision_dict
    }
