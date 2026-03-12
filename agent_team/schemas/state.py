"""
GraphState — the shared state passed between all LangGraph nodes.

Every agent is stateless; all routing decisions rely on this structure.
"""

from __future__ import annotations

from typing import Annotated, TypedDict


def _merge_dicts(a: dict, b: dict) -> dict:
    """Reducer that merges two dicts (b overwrites overlapping keys in a).

    Used for ``code_base`` so that parallel expert nodes' file writes
    are combined instead of one silently overwriting the other.
    """
    merged = dict(a)
    merged.update(b)
    return merged


def _merge_task_list(a: list[dict], b: list[dict]) -> list[dict]:
    """Reducer that merges two task lists by task ID (idempotent).

    When two subgraphs (frontend / backend) finish concurrently and both
    return their modified ``task_list``, LangGraph calls this reducer to
    merge them.  Each subgraph may have changed a *subset* of tasks (e.g.
    set ``fe_1`` to ``completed``), while the other tasks remain unchanged.

    Strategy:
        1. Index both lists by ``id``.
        2. Start from *a* (the accumulated state) and update with *b*
           (the new value), so the latest status wins.
        3. Return a flat list preserving insertion order.
    """
    if not b:
        return list(a)
    if not a:
        return list(b)

    merged: dict[str, dict] = {}
    for t in a:
        merged[t["id"]] = dict(t)
    for t in b:
        merged[t["id"]] = dict(t)

    return list(merged.values())


# ──────────────────────────────────────────────
# Main graph state (parent)
# ──────────────────────────────────────────────

class GraphState(TypedDict, total=False):
    """Global graph state shared by all agent nodes.

    Attributes:
        original_requirement: The user's initial software request.
        system_design: Architecture produced by the Planner.
        task_list: Flat list of sub-task dicts with keys:
            id, description, domain, status.
            Uses a task-merge reducer so parallel subgraph results
            are combined by task ID (idempotent).
        code_base: Accumulated generated files {filepath: content}.
            Uses a dict-merge reducer so parallel writes are combined.
        retry_counters: Per-task retry count {"task_id": int}.
        current_actor: Routing indicator — next node to wake.
        review_feedback: Latest reviewer feedback text.
        phase: Current execution phase — "planning" or "execution".
    """

    original_requirement: str
    system_design: str
    task_list: Annotated[list[dict], _merge_task_list]
    code_base: Annotated[dict, _merge_dicts]
    retry_counters: dict
    current_actor: str
    review_feedback: str
    phase: str


# ──────────────────────────────────────────────
# Domain subgraph state
# ──────────────────────────────────────────────

class DomainState(TypedDict, total=False):
    """State for a single domain subgraph (frontend or backend).

    Each subgraph loops independently through its assigned tasks:
        task_selector → expert → task_reviewer → task_selector → ...

    Attributes:
        domain: "frontend" or "backend".
        task_list: Tasks for this domain (filtered copy).
        code_base: Accumulated generated files.
        system_design: Architecture reference (read-only).
        current_task_id: The task currently being worked on.
        review_feedback: Feedback from the last review rejection.
        retry_count: Retry counter for the current task.
    """

    domain: str
    task_list: list[dict]
    code_base: Annotated[dict, _merge_dicts]
    system_design: str
    current_task_id: str
    review_feedback: str
    retry_count: int
