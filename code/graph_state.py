"""
Shared state passed between all LangGraph nodes for a single request evaluation.

One RequestState instance == one invocation of the compiled graph == one row
in dataset/requests.csv -> one row in dataset/output.csv.
"""

from __future__ import annotations
from typing import Any, Literal, TypedDict


class TokenUsageEntry(TypedDict):
    node: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float


class Warning(TypedDict):
    node: str
    code: str  # e.g. "MISSING_RATE_FALLBACK", "LOW_CONFIDENCE_OCR"
    detail: str


class SpendingChange(TypedDict):
    kind: Literal["stop", "reduce_to"]
    event_id: str
    new_amount: float | None  # only for reduce_to


class PaymentLeg(TypedDict):
    date: str  # YYYY-MM-DD
    amount: float


class Decision(TypedDict, total=False):
    amount_safe_to_pay: float
    affordability_status: Literal[
        "affordable_now", "affordable_with_plan", "affordable_later", "not_affordable"
    ]
    recommended_payment_method: Literal[
        "full_payment", "partial_payment", "installments", "wait", "not_recommended"
    ]
    payment_plan: list[PaymentLeg]  # empty list == "none"
    earliest_date_for_full_payment: str | None
    spending_changes_needed: list[SpendingChange]  # empty list == "none"
    decision_explanation: str


class RequestState(TypedDict, total=False):
    # --- input, set by load_context_node ---
    request: dict[str, Any]  # one row of requests.csv
    user_profile: dict[str, Any]  # matching financial_profiles.csv row
    raw_events: list[dict[str, Any]]  # financial_events.csv rows for this user
    raw_messages: list[dict[str, Any]]  # messages.csv rows for this user/request
    raw_images: list[dict[str, Any]]  # images.csv rows for this user/request
    payment_options: list[dict[str, Any]]  # request_payment_options.csv rows for this request

    # --- set by resolve_conflicts_node ---
    resolved_events: list[dict[str, Any]]  # post conflict-resolution, amended amounts applied

    # --- set by extract_media_node (conditional) ---
    extracted_amounts: dict[str, float]  # event_id -> amount pulled from image/message

    # --- set by process_events_node ---
    recurring_events: list[dict[str, Any]]
    flexible_events: list[dict[str, Any]]  # subset of recurring_events, eligible for stop/reduce

    # --- set by simulate_forecast_node ---
    balance_forecast: list[float]  # 90 daily balances from request_date
    amount_safe_to_pay_raw: float  # pre-spending-change value
    earliest_date_for_full_payment_raw: str | None
    mc_breach_probability: float  # Monte Carlo risk of dropping below min_balance
    mc_p95_margin: float  # 5th percentile minimum balance margin
    mc_risk_adjusted_safe_amount: float  # 95% confidence safe payment amount

    # --- set by rank_decision_node ---
    candidates: list[dict[str, Any]]  # ranked candidate plans, best first

    # --- set by explain_node ---
    decision: Decision

    # --- cross-cutting, appended to by any node ---
    token_usage: list[TokenUsageEntry]
    warnings: list[Warning]
