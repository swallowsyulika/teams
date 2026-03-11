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


def _reduce_current_actor(
    existing: str | list[str],
    new: str | list[str],
) -> str | list[str]:
    """Reducer for ``current_actor``.

    When parallel nodes (e.g. both experts) each write a value in the
    same super-step, LangGraph invokes this reducer to merge them.

    Rules:
        1. Flatten both sides into lists.
        2. Deduplicate (order-preserving).
        3. Collapse a single-element list back to a plain string so that
           downstream routing logic (which checks ``isinstance(..., list)``)
           works unchanged.
    """
    # Normalise to lists
    a = existing if isinstance(existing, list) else ([existing] if existing else [])
    b = new if isinstance(new, list) else ([new] if new else [])

    # The new value(s) take priority; include existing only when
    # multiple distinct targets are reported in the same step.
    seen: set[str] = set()
    merged: list[str] = []
    for item in b + a:
        if item and item not in seen:
            seen.add(item)
            merged.append(item)

    if not merged:
        return ""
    if len(merged) == 1:
        return merged[0]
    return merged


def _reduce_submissions(a: list[dict], b: list[dict]) -> list[dict]:
    """Reducer for ``expert_submissions``.

    Normally concatenates like ``operator.add``.  The critical difference:
    an **empty list** from the new value signals a *reset* — it replaces
    the accumulator entirely.  This lets the Leader clear old submissions
    between dispatch rounds so the Reviewer never sees stale entries.
    """
    if not b:
        # Empty new list ⇒ intentional clear
        return []
    return list(a) + list(b)


class GraphState(TypedDict, total=False):
    """Global graph state shared by all agent nodes.

    Attributes:
        original_requirement: The user's initial software request.
        system_design: Architecture produced by the Planner.
        task_list: Flat list of sub-task dicts with keys:
            id, description, domain ("frontend"|"backend"), status ("pending"|"in_progress"|"completed"|"failed").
        current_active_tasks: Currently executing task IDs per domain,
            e.g. {"frontend": "fe_1", "backend": "be_1"}.
        code_base: Accumulated generated files {filepath: content}.
            Uses a dict-merge reducer so parallel writes are combined.
        retry_counters: Per-task retry count {"task_id": int}.
        current_actor: Routing indicator — next node to wake.
            Uses a reducer so parallel writes (e.g. from two experts)
            are merged instead of raising InvalidUpdateError.
        review_feedback: Latest reviewer feedback text.
        phase: Current execution phase — "planning" or "execution".
        expert_submissions: Reducer-merged list of expert outputs.
            Uses _reduce_submissions: concatenates in parallel steps,
            but an empty list resets the accumulator (avoids unbounded growth).
    """

    original_requirement: str
    system_design: str
    task_list: list[dict]
    current_active_tasks: dict
    code_base: Annotated[dict, _merge_dicts]
    retry_counters: dict
    current_actor: Annotated[str | list[str], _reduce_current_actor]
    review_feedback: str
    phase: str
    expert_submissions: Annotated[list[dict], _reduce_submissions]

