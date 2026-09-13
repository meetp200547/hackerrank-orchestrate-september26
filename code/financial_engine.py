"""
Deterministic financial forecasting engine.

Simulates 90-day daily cash balances, enforces minimum balance protection,
computes maximum safe payment amounts, and calculates earliest dates for full payment.

CRITICAL INVARIANT: THIS MODULE MUST NOT IMPORT ANY LLM OR VLM CLIENT.
"""

from __future__ import annotations
from datetime import datetime, timedelta
from typing import Any


def parse_date(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%Y-%m-%d")


def format_date(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")


class FinancialEngine:
    def __init__(
        self,
        current_balance: float,
        minimum_balance: float,
        request_date_str: str,
        daily_inflows: dict[str, float],
        daily_outflows: dict[str, float],
        forecast_days: int = 90
    ):
        self.current_balance = current_balance
        self.minimum_balance = minimum_balance
        self.request_date_str = request_date_str
        self.daily_inflows = daily_inflows
        self.daily_outflows = daily_outflows
        self.forecast_days = forecast_days
        
        self.dates = [
            format_date(parse_date(request_date_str) + timedelta(days=i))
            for i in range(forecast_days + 1)
        ]

    def simulate_baseline_balances(self, custom_outflows: dict[str, float] | None = None) -> list[float]:
        """
        Simulates daily balance over 90 days starting from current_balance.
        """
        outflows = custom_outflows if custom_outflows is not None else self.daily_outflows
        balances = []
        bal = self.current_balance
        
        for d_str in self.dates:
            inflow = self.daily_inflows.get(d_str, 0.0)
            outflow = outflows.get(d_str, 0.0)
            bal += (inflow - outflow)
            balances.append(bal)
            
        return balances

    def is_plan_safe(self, payment_schedule: list[tuple[str, float]], custom_outflows: dict[str, float] | None = None) -> bool:
        """
        Checks whether a proposed payment schedule maintains balance >= minimum_balance on all 90 days.
        payment_schedule: list of (date_str, payment_amount)
        """
        outflows = dict(custom_outflows if custom_outflows is not None else self.daily_outflows)
        
        # Add schedule payments to daily outflows
        for p_date, p_amt in payment_schedule:
            outflows[p_date] = outflows.get(p_date, 0.0) + p_amt
            
        balances = self.simulate_baseline_balances(outflows)
        return all(b >= self.minimum_balance for b in balances)

    def calculate_amount_safe_to_pay(self, requested_amount: float, custom_outflows: dict[str, float] | None = None) -> float:
        """
        Calculates maximum payment amount safe on request_date pre-spending changes.
        Capped between 0.0 and requested_amount.
        """
        outflows = custom_outflows if custom_outflows is not None else self.daily_outflows
        balances = self.simulate_baseline_balances(outflows)
        
        # Minimum available margin over the forecast window
        min_margin = min(b - self.minimum_balance for b in balances)
        
        amount_safe = max(0.0, min_margin)
        return round(min(requested_amount, amount_safe), 2)

    def find_earliest_date_for_full_payment(
        self,
        requested_amount: float,
        desired_completion_date_str: str | None = None,
        custom_outflows: dict[str, float] | None = None
    ) -> str | None:
        """
        Finds the earliest date d in forecast window where paying requested_amount in full on date d is safe.
        """
        outflows = custom_outflows if custom_outflows is not None else self.daily_outflows
        
        for d_str in self.dates:
            schedule = [(d_str, requested_amount)]
            if self.is_plan_safe(schedule, outflows):
                return d_str
                
        return None

    def evaluate_spending_changes(
        self,
        flexible_events: list[dict[str, Any]],
        requested_amount: float
    ) -> tuple[dict[str, float], list[dict[str, Any]]]:
        """
        Evaluates stop/reduce_to actions on flexible events to unlock affordability.
        Enforces mutual exclusivity: an event is touched by at most one action.
        """
        modified_outflows = dict(self.daily_outflows)
        applied_changes: list[dict[str, Any]] = []
        used_event_ids: set[str] = set()

        for ev in flexible_events:
            if len(applied_changes) >= 3:
                break
                
            ev_id = ev["event_id"]
            if ev_id in used_event_ids:
                continue

            ev_amt = ev["amount"]
            ev_date_str = ev["event_date"]
            can_stop = ev.get("can_stop", False)
            can_reduce = ev.get("can_reduce", False)
            min_allowed_raw = ev.get("minimum_allowed_amount")
            min_allowed = float(min_allowed_raw) if min_allowed_raw is not None else ev_amt * 0.5

            try:
                ev_dt = parse_date(ev_date_str)
                req_dt = parse_date(self.request_date_str)
                end_dt = req_dt + timedelta(days=self.forecast_days)
                
                # Compute projected recurring dates in forecast window
                target_dates = []
                if req_dt <= ev_dt <= end_dt:
                    target_dates.append(ev_date_str)
                
                curr_yr = req_dt.year
                curr_mo = req_dt.month
                for _ in range(4):
                    try:
                        dt = datetime(curr_yr, curr_mo, min(ev_dt.day, 28))
                    except ValueError:
                        dt = datetime(curr_yr, curr_mo, 28)
                    if req_dt <= dt <= end_dt:
                        target_dates.append(format_date(dt))
                    curr_mo += 1
                    if curr_mo > 12:
                        curr_mo = 1
                        curr_yr += 1
            except Exception:
                target_dates = [ev_date_str]

            if can_stop:
                used_event_ids.add(ev_id)
                for d_str in set(target_dates):
                    if d_str in modified_outflows:
                        modified_outflows[d_str] = max(0.0, modified_outflows[d_str] - ev_amt)
                applied_changes.append({
                    "kind": "stop",
                    "event_id": ev_id,
                    "new_amount": None
                })
            elif can_reduce and min_allowed < ev_amt:
                used_event_ids.add(ev_id)
                savings = ev_amt - min_allowed
                for d_str in set(target_dates):
                    if d_str in modified_outflows:
                        modified_outflows[d_str] = max(0.0, modified_outflows[d_str] - savings)
                applied_changes.append({
                    "kind": "reduce_to",
                    "event_id": ev_id,
                    "new_amount": round(min_allowed, 2)
                })

        return modified_outflows, applied_changes
