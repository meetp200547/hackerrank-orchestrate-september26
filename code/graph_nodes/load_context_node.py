"""
load_context_node — pulls pre-indexed user and request slice from DataLoader into graph state.
"""

from __future__ import annotations
from graph_state import RequestState
from data_loader import get_data_loader


def load_context_node(state: RequestState) -> RequestState:
    loader = get_data_loader()
    req = state["request"]
    u_id = req["user_id"]
    r_id = req["request_id"]

    profile = loader.profiles.get(u_id, {})
    raw_events = loader.events_by_user.get(u_id, [])
    raw_msgs = loader.messages_by_request.get(r_id, []) + loader.messages_by_user.get(u_id, [])
    raw_imgs = loader.images_by_request.get(r_id, []) + loader.images_by_user.get(u_id, [])
    opts = loader.payment_options_by_request.get(r_id, [])

    return {
        **state,
        "user_profile": profile,
        "raw_events": raw_events,
        "raw_messages": raw_msgs,
        "raw_images": raw_imgs,
        "payment_options": opts,
        "token_usage": list(state.get("token_usage", [])),
        "warnings": list(state.get("warnings", [])),
    }
