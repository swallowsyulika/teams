"""
Planner agent node — Phase 1.

Receives the original requirement and produces a system architecture
with fine-grained frontend/backend task lists.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from agent_team.graph.config import MODEL_NAME, OPENAI_API_KEY, OPENAI_BASE_URL, TEMPERATURE
from agent_team.schemas.models import PlannerOutput
from agent_team.schemas.state import GraphState

PLANNER_SYSTEM_PROMPT = """\
You are the **Planner** of a multi-agent software development team.

Your job:
1. Analyse the user's software requirement.
2. Design a clear, modular system architecture (technology stack, component
   breakdown, data flow, API contracts).
3. Break the implementation into an **extremely fine-grained** list of
   sub-tasks — each task must be small enough to be completed in a single
   LLM generation step.
4. Separate tasks into **frontend** and **backend** domains.
5. Each task must have a unique ID (e.g. "fe_1", "be_3"), a clear description,
   and be marked with status="pending".

Output your result as structured JSON matching the PlannerOutput schema.
If the reviewer previously rejected your plan, their feedback is included
below — revise accordingly.
"""


def planner_node(state: GraphState) -> dict[str, Any]:
    """Planner node — generates system design and task breakdown.

    Args:
        state: Current graph state.

    Returns:
        Partial state update with system_design, task_list, phase, current_actor.
    """
    llm = ChatOpenAI(
        model=MODEL_NAME,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        temperature=TEMPERATURE,
    ).with_structured_output(PlannerOutput)

    messages = [SystemMessage(content=PLANNER_SYSTEM_PROMPT)]

    # Include original requirement
    requirement = state.get("original_requirement", "")
    messages.append(HumanMessage(content=f"## User Requirement\n\n{requirement}"))

    # If there's reviewer feedback from a previous rejection, include it
    feedback = state.get("review_feedback", "")
    if feedback:
        messages.append(
            HumanMessage(
                content=f"## Reviewer Feedback (please revise your plan)\n\n{feedback}"
            )
        )

    result: PlannerOutput = llm.invoke(messages)

    # Build the flat task list
    all_tasks = [t.model_dump() for t in result.frontend_tasks] + [
        t.model_dump() for t in result.backend_tasks
    ]

    return {
        "system_design": result.system_architecture,
        "task_list": all_tasks,
        "phase": "planning",
        "current_actor": "plan_reviewer",
        "review_feedback": "",
        "retry_counters": {},
        "current_active_tasks": {},
        "code_base": state.get("code_base", {}),
        "expert_submissions": [],
    }
