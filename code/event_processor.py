"""
Event processor module for Buy or Wait? AI financial agent.

Filters events, identifies recurring income and expense streams, classifies
flexibility against user profile preferences, and constructs 90-day cash flow projections.
"""

from __future__ import annotations
from datetime import datetime, timedelta
from typing import Any


def parse_date(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%Y-%m-%d")


def format_date(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")


def project_recurring_dates(start_dt: datetime, day_of_month: int, end_dt: datetime) -> list[datetime]:
    curr_yr = start_dt.year
    curr_mo = start_dt.month
    dates: list[datetime] = []
    for _ in range(6):
        try:
            dt = datetime(curr_yr, curr_mo, min(day_of_month, 28))
        except ValueError:
            dt = datetime(curr_yr, curr_mo, 28)
        if start_dt <= dt <= end_dt:
            dates.append(dt)
        curr_mo += 1
        if curr_mo > 12:
            curr_mo = 1
            curr_yr += 1
    return dates


def process_user_events(
    events: list[dict[str, Any]],
    user_profile: dict[str, Any],
    request_date_str: str,
    forecast_days: int = 90
) -> dict[str, Any]:
    """
    Processes resolved events for a user and projects daily cash flows over forecast_days.
    """
    req_dt = parse_date(request_date_str)
    end_dt = req_dt + timedelta(days=forecast_days)

    protected_cats = set(user_profile.get("categories_to_protect", []))
    reduce_cats = set(user_profile.get("categories_to_reduce", []))
    stop_cats = set(user_profile.get("categories_to_stop", []))

    daily_inflows: dict[str, float] = {}
    daily_outflows: dict[str, float] = {}

    for i in range(forecast_days + 1):
        dt_str = format_date(req_dt + timedelta(days=i))
        daily_inflows[dt_str] = 0.0
        daily_outflows[dt_str] = 0.0

    flexible_events_map: dict[str, dict[str, Any]] = {}
    processed_recurring_keys: set[str] = set()

    # Sort events by date descending so the most recent financial facts take precedence
    sorted_events = sorted(
        events,
        key=lambda x: parse_date(x.get("settlement_date") or x.get("event_date") or "1970-01-01"),
        reverse=True
    )

    for ev in sorted_events:
        status = str(ev.get("status", "")).lower()
        ev_type = str(ev.get("event_type", "")).lower()
        direction = str(ev.get("direction", "")).lower()
        category = str(ev.get("category", "")).lower()
        amt = ev.get("amount_home_curr")

        if amt is None or amt <= 0.0 or status in ("unrealized", "cancelled", "failed"):
            continue

        set_date_str = ev.get("settlement_date") or ev.get("event_date")
        if not set_date_str:
            continue
        ev_dt = parse_date(set_date_str)

        # Pending debits vs Pending credits
        if status == "pending" and (direction in ("inflow", "credit") or ev_type == "income"):
            continue

        is_protected = category in protected_cats
        flexibility = str(ev.get("flexibility", "fixed")).lower()
        can_stop = (category in stop_cats or flexibility in ("stoppable", "reducible_or_stoppable")) and not is_protected
        can_reduce = (category in reduce_cats or flexibility in ("reducible", "reducible_or_stoppable")) and not is_protected

        if can_stop or can_reduce:
            flexible_events_map[ev["event_id"]] = {
                "event_id": ev["event_id"],
                "category": category,
                "amount": amt,
                "minimum_allowed_amount": ev.get("minimum_allowed_amount_home_curr", amt * 0.5),
                "event_date": set_date_str,
                "can_stop": can_stop,
                "can_reduce": can_reduce,
                "description": ev.get("description", "")
            }

        day_of_mo = ev_dt.day

        # Handle Income
        if direction in ("inflow", "credit") or ev_type in ("income", "refund"):
            if status in ("settled", "scheduled"):
                if category == "salary" or ev_type == "income":
                    rec_key = f"inc_{category}"
                    if rec_key not in processed_recurring_keys:
                        processed_recurring_keys.add(rec_key)
                        for dt in project_recurring_dates(req_dt, day_of_mo, end_dt):
                            d_str = format_date(dt)
                            daily_inflows[d_str] = daily_inflows.get(d_str, 0.0) + amt
                else:
                    if req_dt <= ev_dt <= end_dt:
                        d_str = format_date(ev_dt)
                        daily_inflows[d_str] = daily_inflows.get(d_str, 0.0) + amt
        else:
            # Handle Expenses
            is_recurring = ev_type == "subscription" or category in ("rent", "housing", "utilities", "cloud_storage", "streaming", "debt_repayment")
            if is_recurring:
                rec_key = f"exp_{category}"
                if rec_key not in processed_recurring_keys:
                    processed_recurring_keys.add(rec_key)
                    for dt in project_recurring_dates(req_dt, day_of_mo, end_dt):
                        d_str = format_date(dt)
                        daily_outflows[d_str] = daily_outflows.get(d_str, 0.0) + amt
            else:
                if req_dt <= ev_dt <= end_dt:
                    d_str = format_date(ev_dt)
                    daily_outflows[d_str] = daily_outflows.get(d_str, 0.0) + amt

    # Detect periodic/recurring essential expenses (groceries, transport, healthcare, family_support, etc.)
    essential_cats = {"groceries", "transport", "utilities", "healthcare", "family_support", "education", "debt_repayment", "housing", "insurance"}.union(protected_cats)
    by_cat_history: dict[str, list[tuple[datetime, float]]] = {}
    for ev in events:
        status = str(ev.get("status", "")).lower()
        direction = str(ev.get("direction", "")).lower()
        ev_type = str(ev.get("event_type", "")).lower()
        category = str(ev.get("category", "")).lower()
        amt = ev.get("amount_home_curr")
        if amt is None or amt <= 0.0 or status in ("unrealized", "cancelled", "failed"):
            continue
        if direction in ("inflow", "credit") or ev_type in ("income", "refund"):
            continue
        if category in ("rent", "housing", "utilities", "cloud_storage", "streaming", "debt_repayment"):
            continue
        if category not in essential_cats:
            continue

        set_date_str = ev.get("settlement_date") or ev.get("event_date")
        if not set_date_str:
            continue
        ev_dt = parse_date(set_date_str)
        if ev_dt < req_dt and status == "settled":
            by_cat_history.setdefault(category, []).append((ev_dt, amt))

    for cat, items in by_cat_history.items():
        items.sort(key=lambda x: x[0])
        if len(items) >= 2:
            dates = [x[0] for x in items]
            intervals = [(dates[i] - dates[i-1]).days for i in range(1, len(dates))]
            intervals.sort()
            median_interval = intervals[len(intervals) // 2]

            if 5 <= median_interval <= 35:
                recent_amts = [x[1] for x in items[-5:]]
                avg_amt = sum(recent_amts) / len(recent_amts)
                last_dt = dates[-1]

                future_dates = [
                    parse_date(ev.get("settlement_date") or ev.get("event_date"))
                    for ev in events
                    if str(ev.get("category", "")).lower() == cat and (ev.get("settlement_date") or ev.get("event_date"))
                ]
                future_dates = [d for d in future_dates if d >= req_dt]

                curr_proj = last_dt + timedelta(days=median_interval)
                while curr_proj <= end_dt:
                    if curr_proj >= req_dt:
                        has_explicit = any(abs((d - curr_proj).days) <= 3 for d in future_dates)
                        if not has_explicit:
                            d_str = format_date(curr_proj)
                            daily_outflows[d_str] = daily_outflows.get(d_str, 0.0) + avg_amt
                    curr_proj += timedelta(days=median_interval)

    # Deduplicate flexible events keeping the most recent event per category
    flex_by_cat: dict[str, dict[str, Any]] = {}
    for item in flexible_events_map.values():
        cat = item["category"]
        if cat not in flex_by_cat or parse_date(item["event_date"]) > parse_date(flex_by_cat[cat]["event_date"]):
            flex_by_cat[cat] = item

    return {
        "daily_inflows": daily_inflows,
        "daily_outflows": daily_outflows,
        "flexible_events": list(flex_by_cat.values())
    }
