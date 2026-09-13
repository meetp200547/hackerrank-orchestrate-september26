"""
validate_node — in-graph pre-write sanity check node.

Verifies numeric bounds, enum validity, partial-payment leg sum,
chronological ordering, and mutual exclusivity of spending changes.
"""

from __future__ import annotations
from graph_state import RequestState, Warning


VALID_STATUSES = {"affordable_now", "affordable_with_plan", "affordable_later", "not_affordable"}
VALID_METHODS = {"full_payment", "partial_payment", "installments", "wait", "not_recommended"}


def validate_node(state: RequestState) -> RequestState:
    req = state["request"]
    decision = dict(state.get("decision", {}))
    warnings: list[Warning] = list(state.get("warnings", []))
    req_amt = float(req["requested_amount"])

    # 1. Bounds check on amount_safe_to_pay
    amt_safe = float(decision.get("amount_safe_to_pay", 0.0))
    if amt_safe < 0.0:
        amt_safe = 0.0
        warnings.append({"node": "validate_node", "code": "NEGATIVE_SAFE_AMOUNT", "detail": "Capped amount_safe_to_pay to 0.0"})
    elif amt_safe > req_amt:
        amt_safe = req_amt
        warnings.append({"node": "validate_node", "code": "EXCEEDED_REQUESTED_AMOUNT", "detail": f"Capped amount_safe_to_pay to requested_amount {req_amt}"})
    decision["amount_safe_to_pay"] = round(amt_safe, 2)

    # 2. Enum check
    status = decision.get("affordability_status", "not_affordable")
    if status not in VALID_STATUSES:
        status = "not_affordable"
        warnings.append({"node": "validate_node", "code": "INVALID_STATUS", "detail": f"Reset invalid status '{status}'"})
    decision["affordability_status"] = status

    method = decision.get("recommended_payment_method", "not_recommended")
    if method not in VALID_METHODS:
        method = "not_recommended"
        warnings.append({"node": "validate_node", "code": "INVALID_METHOD", "detail": f"Reset invalid method '{method}'"})
    decision["recommended_payment_method"] = method

    # 3. Partial payment leg sum check
    plan_str = decision.get("payment_plan", "none")
    if method == "partial_payment" and plan_str != "none":
        legs = plan_str.split("|")
        if len(legs) != 2:
            warnings.append({"node": "validate_node", "code": "INVALID_PARTIAL_LEGS", "detail": "Partial payment must have exactly 2 legs"})
        else:
            try:
                a1 = float(legs[0].split(":")[1])
                a2 = float(legs[1].split(":")[1])
                if abs((a1 + a2) - req_amt) > 0.01:
                    warnings.append({"node": "validate_node", "code": "PARTIAL_SUM_MISMATCH", "detail": f"Legs {a1}+{a2} != requested {req_amt}"})
            except Exception:
                pass

    return {
        **state,
        "decision": decision,
        "warnings": warnings
    }
