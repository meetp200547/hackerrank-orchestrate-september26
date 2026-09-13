"""
explain_node — wraps llm_explainer.py to populate decision_explanation and append token usage.
"""

from __future__ import annotations
from graph_state import RequestState, TokenUsageEntry
from llm_explainer import generate_explanation


def explain_node(state: RequestState) -> RequestState:
    req = state["request"]
    profile = state["user_profile"]
    decision = dict(state.get("decision", {}))
    token_usage: list[TokenUsageEntry] = list(state.get("token_usage", []))

    explanation, usage = generate_explanation(req, profile, decision)
    decision["decision_explanation"] = explanation

    if usage:
        token_usage.append(usage)

    return {
        **state,
        "decision": decision,
        "token_usage": token_usage
    }
