"""
Deterministic decision & plan selection engine.

Evaluates payment options, ranks eligible safe plans against the 6-tier
preference hierarchy, formats payment_plan strings, and outputs decision dicts.

CRITICAL INVARIANT: THIS MODULE MUST NOT IMPORT ANY LLM OR VLM CLIENT.
"""

from __future__ import annotations
from datetime import datetime, timedelta
from typing import Any
from financial_engine import FinancialEngine, format_date, parse_date


def format_payment_plan(legs: list[tuple[str, float]]) -> str:
    if not legs:
        return "none"
    return "|".join(f"{dt}:{amt:.2f}".rstrip("0").rstrip(".") if amt == int(amt) else f"{dt}:{amt:.2f}" for dt, amt in legs)


def format_spending_changes(changes: list[dict[str, Any]]) -> str:
    if not changes:
        return "none"
    items = []
    for c in changes:
        if c["kind"] == "stop":
            items.append(f"stop:{c['event_id']}")
        elif c["kind"] == "reduce_to":
            amt = c["new_amount"]
            amt_str = f"{amt:.2f}".rstrip("0").rstrip(".") if amt == int(amt) else f"{amt:.2f}"
            items.append(f"reduce_to:{c['event_id']}:{amt_str}")
    return "|".join(items)


class DecisionEngine:
    def __init__(
        self,
        request: dict[str, Any],
        user_profile: dict[str, Any],
        payment_options: list[dict[str, Any]],
        financial_engine: FinancialEngine,
        flexible_events: list[dict[str, Any]]
    ):
        self.request = request
        self.profile = user_profile
        self.payment_options = payment_options
        self.engine = financial_engine
        self.flexible_events = flexible_events
        
        self.req_id = request["request_id"]
        self.req_date = request["request_date"]
        self.req_amount = float(request["requested_amount"])
        self.completion_date = request.get("desired_completion_date")
        self.allows_partial = request.get("allows_partial_payment", False)
        
        self.user_methods = set(self.profile.get("payment_methods_to_consider", []))
        self.max_inst_months = self.profile.get("max_installment_months")

    def evaluate_candidates(self) -> list[dict[str, Any]]:
        candidates = []
        
        amt_safe_raw = self.engine.calculate_amount_safe_to_pay(self.req_amount, request_id=self.req_id)
        earliest_full_raw = self.engine.find_earliest_date_for_full_payment(self.req_amount)

        # -------------------------------------------------------------
        # Candidate 1: full_payment (no spending changes)
        # -------------------------------------------------------------
        if "full_payment" in self.user_methods:
            schedule = [(self.req_date, self.req_amount)]
            if self.engine.is_plan_safe(schedule):
                candidates.append({
                    "method": "full_payment",
                    "status": "affordable_now",
                    "schedule": schedule,
                    "total_cost": self.req_amount,
                    "changes": [],
                    "earliest_full": self.req_date,
                    "amount_safe": self.req_amount,
                    "option_id": "00_full"
                })

        # -------------------------------------------------------------
        # Candidate 2: partial_payment (no spending changes)
        # -------------------------------------------------------------
        if "partial_payment" in self.user_methods and self.allows_partial:
            if 0 < amt_safe_raw < self.req_amount and earliest_full_raw:
                if not self.completion_date or earliest_full_raw <= self.completion_date:
                    remaining_amt = round(self.req_amount - amt_safe_raw, 2)
                    schedule = [(self.req_date, amt_safe_raw), (earliest_full_raw, remaining_amt)]
                    if self.engine.is_plan_safe(schedule):
                        candidates.append({
                            "method": "partial_payment",
                            "status": "affordable_with_plan",
                            "schedule": schedule,
                            "total_cost": self.req_amount,
                            "changes": [],
                            "earliest_full": earliest_full_raw,
                            "amount_safe": amt_safe_raw,
                            "option_id": "00_partial"
                        })

        # -------------------------------------------------------------
        # Candidate 3: installments (matching dataset options)
        # -------------------------------------------------------------
        if "installments" in self.user_methods:
            for opt in self.payment_options:
                opt_type = str(opt.get("payment_type") or opt.get("payment_method") or "").lower()
                if opt_type and opt_type != "installments":
                    continue

                num_pmts = opt["number_of_payments"]
                freq_days = opt["payment_frequency_days"]
                first_date = opt["first_payment_date"]
                pmt_amt = opt["payment_amount"]
                opt_id = opt["payment_option_id"]
                total_payable = opt.get("total_payable_amount", pmt_amt * num_pmts)

                # Check max_installment_months constraint
                duration_months = (num_pmts * freq_days) / 30.0 if freq_days > 0 else num_pmts
                if self.max_inst_months is not None and duration_months > self.max_inst_months + 0.5:
                    continue

                # Build installment schedule
                schedule = []
                start_dt = parse_date(first_date)
                for step in range(num_pmts):
                    p_dt = start_dt + timedelta(days=step * freq_days)
                    schedule.append((format_date(p_dt), pmt_amt))

                if self.engine.is_plan_safe(schedule):
                    candidates.append({
                        "method": "installments",
                        "status": "affordable_with_plan",
                        "schedule": schedule,
                        "total_cost": total_payable,
                        "changes": [],
                        "earliest_full": earliest_full_raw or first_date,
                        "amount_safe": amt_safe_raw,
                        "option_id": opt_id
                    })

        # -------------------------------------------------------------
        # Candidate 4: full_payment WITH permitted spending changes
        # -------------------------------------------------------------
        if "full_payment" in self.user_methods and self.flexible_events:
            mod_outflows, changes = self.engine.evaluate_spending_changes(self.flexible_events, self.req_amount)
            if changes:
                schedule_full = [(self.req_date, self.req_amount)]
                if self.engine.is_plan_safe(schedule_full, custom_outflows=mod_outflows):
                    earliest_full_mod = self.engine.find_earliest_date_for_full_payment(self.req_amount, custom_outflows=mod_outflows)
                    candidates.append({
                        "method": "full_payment",
                        "status": "affordable_with_plan",
                        "schedule": schedule_full,
                        "total_cost": self.req_amount,
                        "changes": changes,
                        "earliest_full": earliest_full_mod or self.req_date,
                        "amount_safe": amt_safe_raw,
                        "option_id": "00_full_with_changes"
                    })

        # -------------------------------------------------------------
        # Candidate 5: wait
        # -------------------------------------------------------------
        if ("full_payment" in self.user_methods or "wait" in self.user_methods) and earliest_full_raw:
            schedule = [(earliest_full_raw, self.req_amount)]
            candidates.append({
                "method": "wait",
                "status": "affordable_later",
                "schedule": schedule,
                "total_cost": self.req_amount,
                "changes": [],
                "earliest_full": earliest_full_raw,
                "amount_safe": amt_safe_raw,
                "option_id": "99_wait"
            })

        return candidates

    def select_best_decision(self) -> dict[str, Any]:
        candidates = self.evaluate_candidates()
        
        amt_safe_raw = self.engine.calculate_amount_safe_to_pay(self.req_amount, request_id=self.req_id)
        earliest_full_raw = self.engine.find_earliest_date_for_full_payment(self.req_amount)

        if not candidates:
            # Fallback: not_recommended
            return {
                "amount_safe_to_pay": amt_safe_raw,
                "affordability_status": "not_affordable",
                "recommended_payment_method": "not_recommended",
                "payment_plan": "none",
                "earliest_date_for_full_payment": earliest_full_raw or "",
                "spending_changes_needed": "none",
                "decision_explanation": f"Do not make this payment by {self.completion_date or 'the deadline'}. None of the available options keeps the minimum balance protected."
            }

        # 6-Tier Ranking Function
        def rank_key(cand: dict[str, Any]):
            last_pmt_date = cand["schedule"][-1][0] if cand["schedule"] else "9999-99-99"
            first_pmt_date = cand["schedule"][0][0] if cand["schedule"] else "9999-99-99"
            
            # Tier 1: Complete by desired completion date
            completes_on_time = 0 if (not self.completion_date or last_pmt_date <= self.completion_date) else 1
            # Tier 2: Prefer paying now/on-time (full, partial, installments) over wait
            is_wait = 1 if cand["method"] == "wait" else 0
            # Tier 3: Require no spending changes
            no_changes = 0 if len(cand["changes"]) == 0 else 1
            # Tier 4: Total amount paid
            total_cost = cand["total_cost"]
            # Tier 5: Start payment earlier
            start_date = first_pmt_date
            # Tier 6: Fewer payments
            num_payments = len(cand["schedule"])
            # Tier 7: Option ID
            option_id = cand.get("option_id", "")

            return (completes_on_time, is_wait, no_changes, total_cost, start_date, num_payments, option_id)

        candidates.sort(key=rank_key)
        best = candidates[0]

        plan_str = format_payment_plan(best["schedule"])
        changes_str = format_spending_changes(best["changes"])
        earliest_full_val = best["earliest_full"] or earliest_full_raw or ""

        return {
            "amount_safe_to_pay": best["amount_safe"],
            "affordability_status": best["status"],
            "recommended_payment_method": best["method"],
            "payment_plan": plan_str,
            "earliest_date_for_full_payment": earliest_full_val,
            "spending_changes_needed": changes_str,
            "decision_explanation": f"Recommended {best['method']} plan starting {best['schedule'][0][0] if best['schedule'] else self.req_date}."
        }
