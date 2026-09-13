"""
LLM Explanation Generator for Buy or Wait? AI agent.

Generates grounded, natural language decision explanations without altering numerical
outputs computed upstream by financial_engine and decision_engine.

Tracks token usage and cost.
"""

from __future__ import annotations
import os
from typing import Any
import config
from graph_state import TokenUsageEntry


def generate_explanation(
    request: dict[str, Any],
    profile: dict[str, Any],
    decision: dict[str, Any]
) -> tuple[str, TokenUsageEntry | None]:
    """
    Produces concise decision_explanation text and logs token usage.
    """
    req_amt = float(request["requested_amount"])
    home_curr = profile.get("home_currency", "INR")
    min_bal = float(profile.get("minimum_balance_to_keep", 0.0))
    
    status = decision["affordability_status"]
    method = decision["recommended_payment_method"]
    plan = decision["payment_plan"]
    amt_safe = decision["amount_safe_to_pay"]
    earliest_date = decision.get("earliest_date_for_full_payment")
    changes = decision.get("spending_changes_needed", "none")
    req_date = request["request_date"]

    gemini_key = os.environ.get("GEMINI_API_KEY")

    prompt = (
        f"You are a financial advisor assistant. Summarize the following recommendation in 1-2 concise sentences.\n"
        f"User home currency: {home_curr}\n"
        f"Requested amount: {home_curr} {req_amt:,.2f}\n"
        f"Status: {status}\n"
        f"Method: {method}\n"
        f"Payment plan: {plan}\n"
        f"Safe amount today: {home_curr} {amt_safe:,.2f}\n"
        f"Minimum balance to keep: {home_curr} {min_bal:,.2f}\n"
        f"Spending changes: {changes}\n"
        f"Earliest full date: {earliest_date}\n\n"
        f"Do not invent facts or alter numbers."
    )

    if gemini_key:
        try:
            from google import genai
            client = genai.Client(api_key=gemini_key)
            response = client.models.generate_content(
                model=config.DEFAULT_LLM_MODEL,
                contents=prompt
            )
            explanation = response.text.strip()
            in_tokens = getattr(response.usage_metadata, "prompt_token_count", 120) if hasattr(response, "usage_metadata") else 120
            out_tokens = getattr(response.usage_metadata, "candidates_token_count", 30) if hasattr(response, "usage_metadata") else 30

            # Gemini 1.5 Flash pricing via google-genai SDK
            cost = (in_tokens * 0.075 + out_tokens * 0.30) / 1_000_000

            usage_entry: TokenUsageEntry = {
                "node": "explain_node",
                "provider": "google-genai",
                "model": config.DEFAULT_LLM_MODEL,
                "input_tokens": in_tokens,
                "output_tokens": out_tokens,
                "estimated_cost_usd": cost
            }
            return explanation, usage_entry
        except Exception:
            pass  # Fall back to template generation

    # High-precision deterministic template generator matching problem statement style
    if method == "full_payment":
        if changes and changes != "none":
            explanation = f"Stop/reduce specified flexible expenses, then pay {home_curr} {req_amt:,.2f} today. This leaves at least {home_curr} {min_bal:,.2f} available."
        else:
            explanation = f"Pay {home_curr} {req_amt:,.2f} today. This keeps the {home_curr} {min_bal:,.2f} minimum available over the next 90 days."
    elif method == "installments":
        parts = plan.split("|")
        num_pmts = len(parts)
        first_amt = parts[0].split(":")[1] if parts else f"{req_amt}"
        first_date = parts[0].split(":")[0] if parts else req_date
        explanation = f"Use {num_pmts} installments of {home_curr} {float(first_amt):,.2f}, starting {first_date}. This leaves at least {home_curr} {min_bal:,.2f} available."
    elif method == "partial_payment":
        parts = plan.split("|")
        amt1 = float(parts[0].split(":")[1])
        amt2 = float(parts[1].split(":")[1])
        date2 = parts[1].split(":")[0]
        explanation = f"Pay {home_curr} {amt1:,.2f} today and the remaining {home_curr} {amt2:,.2f} on {date2}. This completes the full request and keeps the {home_curr} {min_bal:,.2f} minimum protected."
    elif method == "wait":
        wait_date = earliest_date or request.get("desired_completion_date", req_date)
        explanation = f"Pay {home_curr} {req_amt:,.2f} in full on {wait_date}. Paying earlier would take the balance below the {home_curr} {min_bal:,.2f} minimum."
    else:  # not_recommended
        comp_date = request.get("desired_completion_date", "the deadline")
        explanation = f"Do not make this payment by {comp_date}. None of the available options keeps the {home_curr} {min_bal:,.2f} minimum protected."

    usage_entry = {
        "node": "explain_node",
        "provider": "deterministic",
        "model": "template-generator",
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_cost_usd": 0.0
    }
    return explanation, usage_entry
