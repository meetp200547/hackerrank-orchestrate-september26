"""
Compiles the single StateGraph used to evaluate one request.

Flow:
    load_context_node
        -> resolve_conflicts_node
            -> [conditional] extract_media_node   (only if image extraction needed)
            -> process_events_node
        -> simulate_forecast_node   (deterministic, no LLM)
        -> rank_decision_node       (deterministic, no LLM)
        -> explain_node             (LLM, prose only)
        -> validate_node            (deterministic pre-write sanity check)
        -> END

Compiled once at startup in main.py; reused across all requests via .batch() / .invoke().
"""

from __future__ import annotations
from langgraph.graph import StateGraph, END

from graph_state import RequestState
from graph_nodes.load_context_node import load_context_node
from graph_nodes.resolve_conflicts_node import resolve_conflicts_node
from graph_nodes.extract_media_node import extract_media_node, needs_extraction
from graph_nodes.process_events_node import process_events_node
from graph_nodes.simulate_forecast_node import simulate_forecast_node
from graph_nodes.rank_decision_node import rank_decision_node
from graph_nodes.explain_node import explain_node
from graph_nodes.validate_node import validate_node


def _route_after_conflict_resolution(state: RequestState) -> str:
    """Conditional edge: skip extraction node when no event has a blank amount backed by an image."""
    return "extract_media_node" if needs_extraction(state) else "process_events_node"


def build_graph():
    graph = StateGraph(RequestState)

    graph.add_node("load_context_node", load_context_node)
    graph.add_node("resolve_conflicts_node", resolve_conflicts_node)
    graph.add_node("extract_media_node", extract_media_node)
    graph.add_node("process_events_node", process_events_node)
    graph.add_node("simulate_forecast_node", simulate_forecast_node)
    graph.add_node("rank_decision_node", rank_decision_node)
    graph.add_node("explain_node", explain_node)
    graph.add_node("validate_node", validate_node)

    graph.set_entry_point("load_context_node")

    graph.add_edge("load_context_node", "resolve_conflicts_node")

    graph.add_conditional_edges(
        "resolve_conflicts_node",
        _route_after_conflict_resolution,
        {
            "extract_media_node": "extract_media_node",
            "process_events_node": "process_events_node",
        },
    )
    graph.add_edge("extract_media_node", "process_events_node")

    graph.add_edge("process_events_node", "simulate_forecast_node")
    graph.add_edge("simulate_forecast_node", "rank_decision_node")
    graph.add_edge("rank_decision_node", "explain_node")
    graph.add_edge("explain_node", "validate_node")
    graph.add_edge("validate_node", END)

    return graph.compile()


_compiled_graph = None


def get_compiled_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph
