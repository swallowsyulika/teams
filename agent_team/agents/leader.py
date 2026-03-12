"""
Leader agent node — Phase 2 dispatcher.

The leader is now a **deterministic** node (no LLM).  Its only job is to
transition the graph from Phase 1 (planning) to Phase 2 (execution) and
act as a passthrough so that the conditional edge router can fan-out to
the domain subgraphs via ``Send``.
"""

from __future__ import annotations

from typing import Any

from agent_team.schemas.state import GraphState


def leader_node(state: GraphState) -> dict[str, Any]:
    """Leader node — deterministic dispatcher (no LLM).

    Simply marks the phase as "execution" and signals that dispatch
    should happen.  The actual Send fan-out is handled by the
    conditional edge ``route_after_leader`` in builder.py.

    Args:
        state: Current graph state.

    Returns:
        Partial state update with phase and current_actor.
    """
    task_list = state.get("task_list", [])

    # Check if there are any pending tasks left
    has_pending = any(t["status"] == "pending" for t in task_list)

    if not has_pending:
        return {
            "current_actor": "done",
            "phase": "execution",
        }

    return {
        "current_actor": "dispatching",
        "phase": "execution",
    }
