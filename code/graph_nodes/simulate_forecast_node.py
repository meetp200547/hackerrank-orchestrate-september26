"""
simulate_forecast_node — deterministic node wrapping FinancialEngine.
CRITICAL INVARIANT: NO LLM IMPORTS.
"""

from __future__ import annotations
from graph_state import RequestState
from financial_engine import FinancialEngine
from event_processor import process_user_events


def simulate_forecast_node(state: RequestState) -> RequestState:
    profile = state["user_profile"]
    req = state["request"]
    events = state.get("resolved_events", [])

    curr_bal = profile.get("current_available_balance", 0.0)
    min_bal = profile.get("minimum_balance_to_keep", 0.0)
    req_date = req["request_date"]
    req_amt = float(req["requested_amount"])

    processed = process_user_events(events, profile, req_date)
    inflows = processed["daily_inflows"]
    outflows = processed["daily_outflows"]

    engine = FinancialEngine(
        current_balance=curr_bal,
        minimum_balance=min_bal,
        request_date_str=req_date,
        daily_inflows=inflows,
        daily_outflows=outflows
    )

    balances = engine.simulate_baseline_balances()
    amt_safe = engine.calculate_amount_safe_to_pay(req_amt)
    earliest_date = engine.find_earliest_date_for_full_payment(req_amt)

    return {
        **state,
        "balance_forecast": balances,
        "amount_safe_to_pay_raw": amt_safe,
        "earliest_date_for_full_payment_raw": earliest_date
    }
