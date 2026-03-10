"""
GraphState — the shared state passed between all LangGraph nodes.

Every agent is stateless; all routing decisions rely on this structure.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict


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
        retry_counters: Per-task retry count {"task_id": int}.
        current_actor: Routing indicator — next node to wake.
        review_feedback: Latest reviewer feedback text.
        phase: Current execution phase — "planning" or "execution".
        expert_submissions: Reducer-merged list of expert outputs
            (uses operator.add so parallel results are concatenated).
    """

    original_requirement: str
    system_design: dict
    task_list: list[dict]
    current_active_tasks: dict
    code_base: dict
    retry_counters: dict
    current_actor: str
    review_feedback: str
    phase: str
    expert_submissions: Annotated[list[dict], operator.add]
