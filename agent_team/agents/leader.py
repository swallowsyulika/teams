"""
Leader agent node — Phase 2 orchestrator.

Reads the global task list, identifies the next pending sub-tasks in each
domain, and uses the LangGraph ``Send`` API to dispatch them to the
corresponding expert nodes **in parallel**.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.types import Send

from agent_team.graph.config import MODEL_NAME, OPENAI_API_KEY, OPENAI_BASE_URL, TEMPERATURE
from agent_team.schemas.models import LeaderDecision
from agent_team.schemas.state import GraphState

LEADER_SYSTEM_PROMPT = """\
You are the **Leader** (dispatcher) of a multi-agent software development team.

You do NOT write code.  Your only job is:
1. Read the current task list and their statuses.
2. For each domain (frontend, backend) that still has "pending" tasks,
   pick the **next one** pending task to dispatch.
3. You may dispatch up to **one task per domain** simultaneously — this
   enables frontend and backend to work in parallel.
4. If NO pending tasks remain (all are "completed" or "failed"), set
   is_complete to true.

Output your result as structured JSON matching the LeaderDecision schema.
"""


def leader_node(state: GraphState) -> dict[str, Any]:
    """Leader node — decides which tasks to dispatch next.

    Instead of returning a simple state update, this node is used in
    conjunction with a conditional edge that returns ``Send`` objects,
    enabling true parallel fan-out to multiple expert nodes.

    Args:
        state: Current graph state.

    Returns:
        Partial state update with current_actor and current_active_tasks.
    """
    llm = ChatOpenAI(
        model=MODEL_NAME,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        temperature=TEMPERATURE,
    ).with_structured_output(LeaderDecision)

    task_list = state.get("task_list", [])

    # Build a summary for the LLM
    task_summary = "\n".join(
        f"- [{t['status']}] {t['id']}: {t['description']} (domain={t['domain']})"
        for t in task_list
    )

    system_design_summary = str(state.get("system_design", {}))

    messages = [
        SystemMessage(content=LEADER_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"## System Design\n{system_design_summary}\n\n"
                f"## Current Task List\n{task_summary}"
            )
        ),
    ]

    decision: LeaderDecision = llm.invoke(messages)

    # Update the active tasks mapping
    active = dict(state.get("current_active_tasks", {}))
    for dt in decision.dispatched_tasks:
        active[dt.domain] = dt.task_id

    # Mark dispatched tasks as in_progress
    updated_tasks = []
    dispatched_ids = {dt.task_id for dt in decision.dispatched_tasks}
    for t in task_list:
        t_copy = dict(t)
        if t_copy["id"] in dispatched_ids:
            t_copy["status"] = "in_progress"
        updated_tasks.append(t_copy)

    current_actor = "done" if decision.is_complete else "dispatching"

    return {
        "current_active_tasks": active,
        "task_list": updated_tasks,
        "current_actor": current_actor,
        "expert_submissions": [],
    }
