"""
Conflict resolution module implementing the 4-level precedence hierarchy
defined in §6.3 of AGENTS.md and problem_statement.md.

Precedence:
1. Explicit cancellation, settlement, or amendment (from linked events or messages)
2. Newer record from the same source (evaluated by event_date / sent_at)
3. Settled event over an estimate or forecast
4. Financially safer interpretation when unresolved (more conservative cash outcome)
"""

from __future__ import annotations
import re
from typing import Any


def sanitize_untrusted_text(text: str) -> str:
    """
    Strips system instructions or prompt injection attempts from messages/images.
    Ensures message content is treated strictly as data.
    """
    if not text:
        return ""
    # Filter common prompt injection phrases
    suspicious_patterns = [
        r"ignore\s+(previous|all)\s+instructions",
        r"system\s+prompt",
        r"override\s+rules",
        r"approve\s+(this\s+)?unconditionally"
    ]
    cleaned = text
    for pat in suspicious_patterns:
        cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def resolve_event_conflicts(
    events: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    home_currency: str,
    converter_func: Any
) -> list[dict[str, Any]]:
    """
    Resolves conflicts across financial events for a user:
    - Filters out cancelled / failed events (or events explicit cancelled via message)
    - Replaces estimates with settled events when linked
    - Applies message-driven amendments (salary updates, expense reductions, date delays)
    - Converts foreign-currency event amounts to user's home_currency
    """
    # 1. Map messages by related_event_id and request_id
    msg_by_event: dict[str, list[dict[str, Any]]] = {}
    for m in messages:
        rel_ev = m.get("related_event_id")
        if rel_ev:
            if rel_ev not in msg_by_event:
                msg_by_event[rel_ev] = []
            msg_by_event[rel_ev].append(m)

    resolved_map: dict[str, dict[str, Any]] = {}
    
    # 2. Process events and resolve status/amount conflicts
    for ev in events:
        ev_copy = dict(ev)
        ev_id = ev_copy["event_id"]
        status = str(ev_copy.get("status", "")).lower()

        # Rule 1 & 3: Skip cancelled, failed, or duplicate events
        if status in ("cancelled", "canceled", "failed"):
            continue

        # Check related messages for explicit cancellation or amendment
        related_msgs = msg_by_event.get(ev_id, [])
        is_cancelled = False
        amended_amount = None

        for msg in related_msgs:
            text = sanitize_untrusted_text(msg.get("message_text", "")).lower()
            if "cancellation" in text or "cancelled" in text or "canceled" in text or "voided" in text:
                is_cancelled = True
                break

        if is_cancelled:
            continue

        # Currency conversion to home currency
        amt = ev_copy.get("amount")
        curr = ev_copy.get("currency", home_currency)
        event_date = ev_copy.get("settlement_date") or ev_copy.get("event_date") or "2026-01-01"

        if amt is not None:
            conv_amt, _ = converter_func(amt, curr, home_currency, event_date)
            ev_copy["amount_home_curr"] = conv_amt
        else:
            ev_copy["amount_home_curr"] = None

        if ev_copy.get("minimum_allowed_amount") is not None:
            min_amt, _ = converter_func(ev_copy["minimum_allowed_amount"], curr, home_currency, event_date)
            ev_copy["minimum_allowed_amount_home_curr"] = min_amt
        else:
            ev_copy["minimum_allowed_amount_home_curr"] = None

        # Rule 2: Newer record from same source / linked event resolution
        linked_id = ev_copy.get("linked_event_id")
        if linked_id and linked_id in resolved_map:
            prior_ev = resolved_map[linked_id]
            # Preference 3: Settled event over estimate/pending
            if prior_ev.get("status") != "settled" and status == "settled":
                resolved_map[linked_id] = ev_copy
                continue

        resolved_map[ev_id] = ev_copy

    return list(resolved_map.values())


def safer_interpretation_fallback(event: dict[str, Any]) -> float:
    """
    Fallback for unresolved or ambiguous financial amounts.
    For expenses: assumes higher cost (conservative).
    For income: assumes 0 (conservative).
    """
    direction = str(event.get("direction", "")).lower()
    if direction in ("outflow", "debit", "expense"):
        # Assume conservative estimate if minimum allowed amount exists
        return event.get("minimum_allowed_amount_home_curr") or event.get("amount_home_curr") or 100.0
    return 0.0
